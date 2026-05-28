"""TASK-070: observability metrics + tracing spans for the compute pipeline.

This module covers four acceptance criteria from ``tasks.json``:

1. Metrics include job duration, failure rate, artifact read/write time,
   review queue size, ambiguous object count, probable label error
   count, and export blocked count.
2. Tracing spans cover ingestion, manifest_builder, prediction.validate,
   model_error.analyze, tabular.profile, text_ocr.profile,
   decision_core.score, model_impact.evaluate, and export.build.
3. Metrics and traces never carry raw PII, raw text, secrets, or
   user-controlled metadata — only stable technical fields.
4. A local demo run produces structured spans/log output (the contract
   spans + metric counts are deterministic and serializable to JSON).
"""

from __future__ import annotations

import json

import pytest

from app.domain import ErrorCode
from app.telemetry import (
    METRIC_AMBIGUOUS_OBJECT_COUNT,
    METRIC_ARTIFACT_READ_MS,
    METRIC_ARTIFACT_WRITE_MS,
    METRIC_EXPORT_BLOCKED_COUNT,
    METRIC_JOB_DURATION_MS,
    METRIC_JOB_FAILURE_COUNT,
    METRIC_PROBABLE_LABEL_ERROR_COUNT,
    METRIC_REVIEW_QUEUE_SIZE,
    MetricsError,
    MetricsRegistry,
    MetricsSnapshot,
    SpanStatus,
    TracingError,
    TracingRegistry,
)
from app.telemetry.tracing import SpanRecord

# ---------------------------------------------------------------------------
# Acceptance criterion 1 — metrics cover the documented set.
# ---------------------------------------------------------------------------


def test_metrics_registry_records_documented_compute_metrics() -> None:
    metrics = MetricsRegistry()

    # Job duration histogram broken down by stage.
    metrics.observe(
        METRIC_JOB_DURATION_MS,
        value=1234.5,
        labels={"stage": "tabular.profile", "job_type": "ANALYZE_ONLY"},
    )
    metrics.observe(
        METRIC_JOB_DURATION_MS,
        value=890.0,
        labels={"stage": "tabular.profile", "job_type": "ANALYZE_ONLY"},
    )

    # Failure counter with stable error_code labels.
    metrics.record_failure(
        error_code=ErrorCode.LEAKAGE_DETECTED,
        stage="tabular.profile",
    )
    metrics.record_failure(error_code=ErrorCode.PII_RESTRICTED)

    # Artifact read/write histograms.
    metrics.observe(METRIC_ARTIFACT_READ_MS, value=12.0)
    metrics.observe(METRIC_ARTIFACT_WRITE_MS, value=24.0)

    # Review queue gauges per queue type.
    metrics.set_gauge(
        METRIC_REVIEW_QUEUE_SIZE,
        value=42,
        labels={"queue_type": "LABEL_REVIEW"},
    )
    metrics.set_gauge(
        METRIC_REVIEW_QUEUE_SIZE,
        value=8,
        labels={"queue_type": "PRIVACY_REVIEW"},
    )

    # Model-error counters.
    metrics.increment(METRIC_AMBIGUOUS_OBJECT_COUNT, amount=5)
    metrics.increment(METRIC_PROBABLE_LABEL_ERROR_COUNT, amount=3)

    # Export blocked counter labeled by stable reason_code.
    metrics.increment(
        METRIC_EXPORT_BLOCKED_COUNT,
        labels={"reason_code": "pii_unmasked"},
    )
    metrics.increment(
        METRIC_EXPORT_BLOCKED_COUNT,
        labels={"reason_code": "model_impact_rejected"},
    )

    snapshot = metrics.snapshot()

    # Step 1 — every documented metric name is present at least once.
    counter_names = {snap.metric for snap in snapshot.counters}
    histogram_names = {snap.metric for snap in snapshot.histograms}
    gauge_names = {snap.metric for snap in snapshot.gauges}
    assert {
        METRIC_JOB_FAILURE_COUNT,
        METRIC_AMBIGUOUS_OBJECT_COUNT,
        METRIC_PROBABLE_LABEL_ERROR_COUNT,
        METRIC_EXPORT_BLOCKED_COUNT,
    }.issubset(counter_names)
    assert {
        METRIC_JOB_DURATION_MS,
        METRIC_ARTIFACT_READ_MS,
        METRIC_ARTIFACT_WRITE_MS,
    }.issubset(histogram_names)
    assert METRIC_REVIEW_QUEUE_SIZE in gauge_names

    # Step 2 — counter aggregation is correct.
    by_metric_label = {
        (snap.metric, frozenset(snap.labels.items())): snap.value
        for snap in snapshot.counters
    }
    assert by_metric_label[
        (
            METRIC_JOB_FAILURE_COUNT,
            frozenset({("error_code", "LEAKAGE_DETECTED"), ("stage", "tabular.profile")}),
        )
    ] == 1.0
    assert by_metric_label[
        (METRIC_JOB_FAILURE_COUNT, frozenset({("error_code", "PII_RESTRICTED")}))
    ] == 1.0
    assert by_metric_label[(METRIC_AMBIGUOUS_OBJECT_COUNT, frozenset())] == 5.0
    assert by_metric_label[(METRIC_PROBABLE_LABEL_ERROR_COUNT, frozenset())] == 3.0

    # Step 3 — histogram aggregation captures count + sum_value + max.
    duration_snap = next(
        snap
        for snap in snapshot.histograms
        if snap.metric == METRIC_JOB_DURATION_MS
    )
    assert duration_snap.count == 2
    assert duration_snap.sum_value == pytest.approx(2124.5)
    assert duration_snap.max_value == 1234.5

    # Step 4 — gauge labels are honored.
    gauges_by_queue = {
        tuple(sorted(snap.labels.items())): snap.value
        for snap in snapshot.gauges
        if snap.metric == METRIC_REVIEW_QUEUE_SIZE
    }
    assert gauges_by_queue[(("queue_type", "LABEL_REVIEW"),)] == 42.0
    assert gauges_by_queue[(("queue_type", "PRIVACY_REVIEW"),)] == 8.0


def test_metrics_registry_rejects_unsafe_observations() -> None:
    metrics = MetricsRegistry()

    with pytest.raises(MetricsError):
        metrics.observe(METRIC_JOB_DURATION_MS, value=-1.0)
    with pytest.raises(MetricsError):
        metrics.increment(METRIC_AMBIGUOUS_OBJECT_COUNT, amount=-3)


def test_metrics_registry_drops_disallowed_labels_and_redacts_user_metadata() -> None:
    """User-controlled labels (raw text, emails) must not reach the metrics export."""
    metrics = MetricsRegistry()
    metrics.increment(
        METRIC_EXPORT_BLOCKED_COUNT,
        labels={
            "reason_code": "pii_unmasked",
            "customer_email": "alice@example.com",  # disallowed key
            "raw_text": "passport 1234 567890",  # disallowed key
        },
    )
    snapshot = metrics.snapshot()

    # Disallowed label keys are dropped silently; only allow-listed
    # labels remain, so the snapshot serializes safely.
    serialized = json.dumps(snapshot.model_dump(mode="json"))
    assert "alice@example.com" not in serialized
    assert "1234 567890" not in serialized
    assert "customer_email" not in serialized
    assert "raw_text" not in serialized

    counter = next(
        snap
        for snap in snapshot.counters
        if snap.metric == METRIC_EXPORT_BLOCKED_COUNT
    )
    assert counter.labels == {"reason_code": "pii_unmasked"}


def test_metrics_snapshot_is_deterministic_under_repeated_writes() -> None:
    """Two registries that received the same writes must produce equal snapshots."""
    a = MetricsRegistry()
    b = MetricsRegistry()
    for registry in (a, b):
        registry.increment(
            METRIC_AMBIGUOUS_OBJECT_COUNT,
            amount=3,
            labels={"stage": "model_error.analyze"},
        )
        registry.observe(METRIC_ARTIFACT_READ_MS, value=12.5)
        registry.set_gauge(
            METRIC_REVIEW_QUEUE_SIZE,
            value=4,
            labels={"queue_type": "LABEL_REVIEW"},
        )
    assert a.snapshot() == b.snapshot()
    assert isinstance(a.snapshot(), MetricsSnapshot)


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — tracing spans cover documented compute stages.
# ---------------------------------------------------------------------------


_DOCUMENTED_SPAN_NAMES: tuple[str, ...] = (
    "ingestion",
    "manifest_builder",
    "prediction.validate",
    "model_error.analyze",
    "tabular.profile",
    "text_ocr.profile",
    "decision_core.score",
    "model_impact.evaluate",
    "export.build",
)


def test_tracing_registry_records_documented_compute_spans() -> None:
    tracer = TracingRegistry()

    for name in _DOCUMENTED_SPAN_NAMES:
        with tracer.span(name, attributes={"stage": name, "job_type": "ANALYZE_ONLY"}):
            # No-op work: we are testing wiring, not duration accuracy.
            pass

    spans = tracer.snapshot()
    assert tuple(span.name for span in spans) == _DOCUMENTED_SPAN_NAMES
    for span in spans:
        assert isinstance(span, SpanRecord)
        assert span.status is SpanStatus.OK
        assert span.duration_ms >= 0.0
        # ``duration_ms`` is mirrored into the attributes dict for
        # downstream consumers that prefer attribute-level inspection.
        assert "duration_ms" in span.attributes


def test_tracing_span_records_error_status_when_block_raises() -> None:
    tracer = TracingRegistry()

    with pytest.raises(RuntimeError):
        with tracer.span("export.build", attributes={"stage": "export.build"}):
            raise RuntimeError("downstream raised")

    spans = tracer.snapshot()
    assert len(spans) == 1
    assert spans[0].status is SpanStatus.ERROR
    assert spans[0].name == "export.build"


def test_tracing_record_span_persists_pre_measured_duration() -> None:
    tracer = TracingRegistry()
    tracer.record_span(
        name="model_impact.evaluate",
        duration_ms=42.5,
        attributes={"verdict": "improved"},
    )
    spans = tracer.snapshot()
    assert len(spans) == 1
    assert spans[0].duration_ms == 42.5
    assert spans[0].attributes["verdict"] == "improved"


def test_tracing_registry_rejects_unsafe_inputs() -> None:
    tracer = TracingRegistry()

    with pytest.raises(TracingError):
        with tracer.span("", attributes=None):
            pass
    with pytest.raises(TracingError):
        tracer.record_span(name="", duration_ms=1.0)
    with pytest.raises(TracingError):
        tracer.record_span(name="export.build", duration_ms=-1.0)


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — metrics/traces never leak raw PII/secrets.
# ---------------------------------------------------------------------------


def test_tracing_attributes_drop_disallowed_keys_and_truncate_long_values() -> None:
    """Disallowed attribute keys (raw text, emails) must be dropped silently."""
    tracer = TracingRegistry()
    long_value = "x" * 5000

    with tracer.span(
        "tabular.profile",
        attributes={
            "stage": "tabular.profile",
            "row_count": 200,
            "customer_email": "alice@example.com",  # disallowed key
            "raw_text": "passport 1234 567890",  # disallowed key
            "verdict": long_value,  # bounded length
        },
    ):
        pass

    spans = tracer.snapshot()
    assert len(spans) == 1
    attributes = spans[0].attributes
    serialized = json.dumps(attributes)

    # Stable allow-listed attributes are preserved.
    assert attributes["stage"] == "tabular.profile"
    assert attributes["row_count"] == 200

    # Disallowed user-controlled keys never leak into the trace.
    assert "customer_email" not in attributes
    assert "raw_text" not in attributes
    assert "alice@example.com" not in serialized
    assert "1234 567890" not in serialized

    # Bounded string-attribute values cannot blow up cardinality.
    assert isinstance(attributes["verdict"], str)
    assert len(attributes["verdict"]) <= 64


def test_metrics_and_traces_serialize_to_json_without_user_payloads() -> None:
    """The combined snapshot must never serialize raw text or PII."""
    metrics = MetricsRegistry()
    tracer = TracingRegistry()

    metrics.increment(
        METRIC_EXPORT_BLOCKED_COUNT,
        labels={
            "reason_code": "pii_unmasked",
            "customer_email": "alice@example.com",
        },
    )
    metrics.observe(
        METRIC_JOB_DURATION_MS,
        value=12.5,
        labels={"stage": "export.build"},
    )
    with tracer.span(
        "export.build",
        attributes={
            "stage": "export.build",
            "raw_text": "passport 1234 567890",
            "blocked_count": 2,
        },
    ):
        pass

    payload = json.dumps(
        {
            "metrics": metrics.snapshot().model_dump(mode="json"),
            "traces": [span.model_dump(mode="json") for span in tracer.snapshot()],
        }
    )

    assert "alice@example.com" not in payload
    assert "1234 567890" not in payload
    assert "raw_text" not in payload
    assert "customer_email" not in payload
    # Stable technical signal must survive.
    assert "pii_unmasked" in payload
    assert "export.build" in payload
    assert "blocked_count" in payload


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — local demo produces structured spans/metrics output.
# ---------------------------------------------------------------------------


def test_local_demo_run_emits_structured_spans_and_metric_counts() -> None:
    """Simulated demo run emits a deterministic, JSON-serializable telemetry bundle."""
    metrics = MetricsRegistry()
    tracer = TracingRegistry()

    # Walk each documented compute stage, recording a span and the
    # canonical metric observation pair the platform UI consumes.
    with tracer.span("ingestion", attributes={"stage": "ingestion"}):
        metrics.observe(
            METRIC_JOB_DURATION_MS,
            value=12.0,
            labels={"stage": "ingestion"},
        )

    with tracer.span("manifest_builder", attributes={"stage": "manifest_builder"}):
        metrics.observe(METRIC_ARTIFACT_WRITE_MS, value=4.0)

    with tracer.span("prediction.validate", attributes={"stage": "prediction.validate"}):
        metrics.observe(
            METRIC_JOB_DURATION_MS,
            value=8.0,
            labels={"stage": "prediction.validate"},
        )

    with tracer.span(
        "model_error.analyze",
        attributes={
            "stage": "model_error.analyze",
            "ambiguous_object_count": 5,
            "probable_label_error_count": 3,
        },
    ):
        metrics.increment(METRIC_AMBIGUOUS_OBJECT_COUNT, amount=5)
        metrics.increment(METRIC_PROBABLE_LABEL_ERROR_COUNT, amount=3)

    with tracer.span("tabular.profile", attributes={"stage": "tabular.profile", "row_count": 200}):
        metrics.observe(METRIC_ARTIFACT_READ_MS, value=11.0)

    with tracer.span("text_ocr.profile", attributes={"stage": "text_ocr.profile"}):
        metrics.observe(METRIC_ARTIFACT_READ_MS, value=6.0)

    with tracer.span("decision_core.score", attributes={"stage": "decision_core.score"}):
        metrics.set_gauge(
            METRIC_REVIEW_QUEUE_SIZE,
            value=12,
            labels={"queue_type": "LABEL_REVIEW"},
        )

    with tracer.span(
        "model_impact.evaluate",
        attributes={"stage": "model_impact.evaluate", "verdict": "improved"},
    ):
        metrics.observe(
            METRIC_JOB_DURATION_MS,
            value=44.0,
            labels={"stage": "model_impact.evaluate"},
        )

    with tracer.span(
        "export.build",
        attributes={"stage": "export.build", "blocked_count": 0},
    ):
        # No blocked exports in the happy-path demo, so the counter is
        # not incremented; it should default to zero in the snapshot.
        pass

    spans = tracer.snapshot()
    metric_snapshot = metrics.snapshot()

    # Step 1 — every documented stage produced exactly one span.
    span_names = tuple(span.name for span in spans)
    assert span_names == _DOCUMENTED_SPAN_NAMES

    # Step 2 — every span ended OK and carries technical attributes only.
    for span in spans:
        assert span.status is SpanStatus.OK
        for key in span.attributes:
            assert key in {
                "stage",
                "row_count",
                "ambiguous_object_count",
                "probable_label_error_count",
                "verdict",
                "blocked_count",
                "duration_ms",
            }

    # Step 3 — model-error counts surface in metrics with expected totals.
    counters = {
        (snap.metric, frozenset(snap.labels.items())): snap.value
        for snap in metric_snapshot.counters
    }
    assert counters[(METRIC_AMBIGUOUS_OBJECT_COUNT, frozenset())] == 5.0
    assert counters[(METRIC_PROBABLE_LABEL_ERROR_COUNT, frozenset())] == 3.0

    # Step 4 — combined telemetry bundle is JSON-serializable.
    bundle = {
        "metrics": metric_snapshot.model_dump(mode="json"),
        "traces": [span.model_dump(mode="json") for span in spans],
    }
    text = json.dumps(bundle, sort_keys=True)
    assert "ingestion" in text
    assert "model_error.analyze" in text
    assert "model_impact.evaluate" in text
    assert "export.build" in text
