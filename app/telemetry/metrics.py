"""Privacy-safe metrics registry for the compute pipeline.

The :class:`MetricsRegistry` is a small, stdlib-only collector that
records the metrics required by PRD §15 and TASK-070:

- ``job.duration_ms`` — per-stage job durations (histogram observations);
- ``job.failure_count`` — failed compute runs broken down by stable
  ``error_code`` labels;
- ``artifact.read_ms`` / ``artifact.write_ms`` — object storage read/write
  histogram observations;
- ``review_queue.size`` — current review queue size by ``queue_type``;
- ``model_error.ambiguous_object_count`` — ambiguous objects per run;
- ``model_error.probable_label_error_count`` — probable label-error
  candidates per run;
- ``export.blocked_count`` — export packages blocked by readiness gates,
  broken down by stable ``reason_code``.

The registry never accepts raw payloads, raw text, secrets, file
contents, or freeform user metadata. Every label name and value is
sanitized through :func:`_safe_label_value` so the resulting export
artifact is safe to ship to a metrics backend or a structured log sink.
The registry also has no network egress; transport adapters live above
it.

The registry is designed to be tiny and deterministic so it can be
exercised by both the FastAPI compute boundary and tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field

from app.domain.errors import ErrorCode

# Stable metric names used across the codebase. Keep them short and
# stable so dashboards/UI can map them to copy without re-reading code.
METRIC_JOB_DURATION_MS = "job.duration_ms"
METRIC_JOB_FAILURE_COUNT = "job.failure_count"
METRIC_ARTIFACT_READ_MS = "artifact.read_ms"
METRIC_ARTIFACT_WRITE_MS = "artifact.write_ms"
METRIC_REVIEW_QUEUE_SIZE = "review_queue.size"
METRIC_AMBIGUOUS_OBJECT_COUNT = "model_error.ambiguous_object_count"
METRIC_PROBABLE_LABEL_ERROR_COUNT = "model_error.probable_label_error_count"
METRIC_EXPORT_BLOCKED_COUNT = "export.blocked_count"


# Allow-listed label keys. Anything else gets dropped so we never leak
# user-controlled metadata or PII into the metrics export.
_ALLOWED_LABEL_KEYS: frozenset[str] = frozenset(
    {
        "stage",
        "plugin",
        "job_type",
        "queue_type",
        "error_code",
        "reason_code",
        "verdict",
        "modality",
    }
)

# Bounded label-value length so a regression cannot blow up metric
# cardinality with long user-provided strings.
_MAX_LABEL_VALUE_LENGTH = 64


class MetricsError(ValueError):
    """Raised when a metrics operation is unsafe or invalid."""


class CounterSnapshot(BaseModel):
    """Single counter observation snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)
    value: float = Field(ge=0.0)


class GaugeSnapshot(BaseModel):
    """Single gauge observation snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)
    value: float


class HistogramSnapshot(BaseModel):
    """Aggregated histogram observation snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)
    count: int = Field(ge=0)
    sum_value: float = Field(ge=0.0)
    min_value: float | None = None
    max_value: float | None = None


class MetricsSnapshot(BaseModel):
    """Deterministic snapshot of a :class:`MetricsRegistry`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    counters: tuple[CounterSnapshot, ...]
    gauges: tuple[GaugeSnapshot, ...]
    histograms: tuple[HistogramSnapshot, ...]


@dataclass
class _HistogramState:
    count: int = 0
    sum_value: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum_value += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)


_LabelKey = tuple[tuple[str, str], ...]


@dataclass
class MetricsRegistry:
    """Thread-safe in-memory registry for compute metrics."""

    counters: dict[tuple[str, _LabelKey], float] = field(default_factory=dict)
    gauges: dict[tuple[str, _LabelKey], float] = field(default_factory=dict)
    histograms: dict[tuple[str, _LabelKey], _HistogramState] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def increment(
        self,
        metric: str,
        *,
        amount: float = 1.0,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        """Increment a counter by ``amount`` (default 1)."""
        if amount < 0:
            raise MetricsError("counter increments must be non-negative")
        key = (metric, _label_key(labels))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + amount

    def set_gauge(
        self,
        metric: str,
        *,
        value: float,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        """Set a gauge to ``value``."""
        key = (metric, _label_key(labels))
        with self._lock:
            self.gauges[key] = float(value)

    def observe(
        self,
        metric: str,
        *,
        value: float,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        """Append a histogram observation."""
        if value < 0:
            raise MetricsError("histogram observations must be non-negative")
        key = (metric, _label_key(labels))
        with self._lock:
            state = self.histograms.get(key)
            if state is None:
                state = _HistogramState()
                self.histograms[key] = state
            state.observe(float(value))

    def record_failure(self, *, error_code: ErrorCode, stage: str | None = None) -> None:
        """Record a failed compute run with stable error_code label."""
        labels: dict[str, str] = {"error_code": error_code.value}
        if stage:
            labels["stage"] = stage
        self.increment(METRIC_JOB_FAILURE_COUNT, labels=labels)

    def snapshot(self) -> MetricsSnapshot:
        """Return a deterministic, sorted snapshot of all metrics."""
        with self._lock:
            counter_items = sorted(self.counters.items())
            gauge_items = sorted(self.gauges.items())
            histogram_items = sorted(self.histograms.items())

            return MetricsSnapshot(
                counters=tuple(
                    CounterSnapshot(
                        metric=metric,
                        labels=dict(labels),
                        value=value,
                    )
                    for (metric, labels), value in counter_items
                ),
                gauges=tuple(
                    GaugeSnapshot(
                        metric=metric,
                        labels=dict(labels),
                        value=value,
                    )
                    for (metric, labels), value in gauge_items
                ),
                histograms=tuple(
                    HistogramSnapshot(
                        metric=metric,
                        labels=dict(labels),
                        count=state.count,
                        sum_value=state.sum_value,
                        min_value=state.min_value,
                        max_value=state.max_value,
                    )
                    for (metric, labels), state in histogram_items
                ),
            )


def _label_key(labels: Mapping[str, object] | None) -> _LabelKey:
    if labels is None:
        return ()
    sanitized: list[tuple[str, str]] = []
    for raw_key, raw_value in labels.items():
        if raw_key not in _ALLOWED_LABEL_KEYS:
            # Silently drop disallowed labels: surfacing them by name
            # would itself leak user-controlled metadata into errors.
            continue
        value = _safe_label_value(raw_value)
        if value is None:
            continue
        sanitized.append((raw_key, value))
    sanitized.sort(key=lambda item: item[0])
    return tuple(sanitized)


def _safe_label_value(value: object) -> str | None:
    """Coerce a label value into a bounded, lowercase technical string."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, ErrorCode):
        return value.value
    if hasattr(value, "value") and isinstance(value.value, str):  # StrEnum
        text = value.value
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return None
    if len(text) > _MAX_LABEL_VALUE_LENGTH:
        text = text[:_MAX_LABEL_VALUE_LENGTH]
    return text


__all__ = [
    "CounterSnapshot",
    "GaugeSnapshot",
    "HistogramSnapshot",
    "METRIC_AMBIGUOUS_OBJECT_COUNT",
    "METRIC_ARTIFACT_READ_MS",
    "METRIC_ARTIFACT_WRITE_MS",
    "METRIC_EXPORT_BLOCKED_COUNT",
    "METRIC_JOB_DURATION_MS",
    "METRIC_JOB_FAILURE_COUNT",
    "METRIC_PROBABLE_LABEL_ERROR_COUNT",
    "METRIC_REVIEW_QUEUE_SIZE",
    "MetricsError",
    "MetricsRegistry",
    "MetricsSnapshot",
]
