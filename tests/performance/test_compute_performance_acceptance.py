"""TASK-069: MVP performance acceptance checks for the compute pipeline.

This test exercises the deterministic demo archive end-to-end through the
public ANALYZE_ONLY builders and measures wall-clock time spent in each
critical stage:

- ``ingestion.manifest_builder`` — Asset Manifest build over the demo archive;
- ``predictions.validate`` — PredictionManifest ingestion + coverage report;
- ``predictions.model_error.analyze`` — model-error analysis (ambiguous /
  probable label-error candidates);
- ``tabular.profile`` — tabular profile report build;
- ``text_ocr.validate`` — text/OCR validation + PII detection report;
- ``review_queue.build`` — review queue artifact build.

Each stage's max duration is compared against a documented MVP threshold.
The suite saves a deterministic JSON ``performance_report.json`` under the
``DATAFORGE_PERFORMANCE_REPORT_DIR`` env (or pytest ``tmp_path`` by
default) and fails the test (non-zero exit) when any stage breaches its
threshold.

Thresholds are intentionally generous for local CI: they are an early
warning that a stage regressed by an order of magnitude on the demo
fixture, not a microbenchmark. They can be tightened once the platform
has historical baselines.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ErrorCode,
    ManifestRow,
    ModelErrorReportStatus,
    PredictionRow,
    ReviewQueueType,
)
from app.ingestion import (
    BuildManifestRequest,
    build_asset_manifest,
    build_validated_manifest,
    build_validated_predictions,
    manifest_rows_from_artifact,
    open_archive_path,
)
from app.kernel import (
    BuildDecisionReportRequest,
    BuildMethodRecommendationsRequest,
    build_method_recommendations,
)
from app.plugins.object_analytics import (
    BuildEvidenceBundleRequest,
    BuildObjectAnalyticsRequest,
    build_evidence_bundles,
    build_object_analytics_passports,
)
from app.plugins.predictions import analyze_model_errors
from app.plugins.predictions.analyzer import ManifestRowSummary
from app.plugins.tabular import (
    ProfileBuildRequest,
    build_tabular_profile_report,
)
from app.plugins.text_ocr.validator import (
    TextOcrBuildRequest,
    build_text_ocr_report,
    validate_ocr_records_jsonl,
    validate_support_messages_jsonl,
)
from app.reports import (
    BuildReviewQueuesRequest,
    build_decision_report_artifact,
    build_review_queues_artifact,
)
from app.telemetry.performance import (
    StageThreshold,
    StageTimer,
    write_performance_report,
)
from tests.fixtures.demo_archive import DEMO_DATASET_VERSION_ID, build_demo_archive

_ORG_ID = "org_perf"
_PROJECT_ID = "project_perf"
_DATASET_ID = "dataset_perf"
_PARENT_VERSION_ID = "dataset_version_parent"
_DATASET_VERSION_ID = DEMO_DATASET_VERSION_ID
_JOB_ID = "compute_run_perf_acceptance"
_CONFIG_HASH = "sha256:" + "a" * 64
_MODEL_ID = "fraud_baseline"
_MODEL_VERSION = "2026-05-14"
_GENERATED_AT = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)


# Stage names used by both the timer and the threshold list. Keep them
# stable: downstream platform UIs map them to copy.
_STAGE_MANIFEST = "ingestion.manifest_builder"
_STAGE_PREDICTION_VALIDATE = "predictions.validate"
_STAGE_MODEL_ERROR = "predictions.model_error.analyze"
_STAGE_TABULAR_PROFILE = "tabular.profile"
_STAGE_TEXT_OCR_VALIDATE = "text_ocr.validate"
_STAGE_REVIEW_QUEUE = "review_queue.build"


# Documented MVP thresholds for the deterministic demo archive on local
# CI. The demo archive is small (~200 transactions), so the budget is
# generous: any stage taking more than the documented ceiling on the
# demo fixture is a real regression, not a flaky sampling artifact.
#
# Thresholds may be overridden through the ``DATAFORGE_PERFORMANCE_*``
# env so local hardware variance does not flake the suite.
_DEMO_THRESHOLDS: dict[str, float] = {
    _STAGE_MANIFEST: 5_000.0,
    _STAGE_PREDICTION_VALIDATE: 5_000.0,
    _STAGE_MODEL_ERROR: 5_000.0,
    _STAGE_TABULAR_PROFILE: 10_000.0,
    _STAGE_TEXT_OCR_VALIDATE: 5_000.0,
    _STAGE_REVIEW_QUEUE: 5_000.0,
}


def test_compute_pipeline_performance_acceptance(tmp_path: Path) -> None:
    """End-to-end MVP performance acceptance run with documented thresholds.

    Step 1 — run every critical stage on the deterministic demo archive,
    measuring wall-clock duration with :class:`StageTimer`.
    Step 2 — compare each stage against its documented MVP threshold.
    Step 3 — write the report to a deterministic JSON artifact and fail
    the test if any stage breaches its threshold.
    """
    archive_path = build_demo_archive(output_dir=tmp_path / "demo_archive").archive_path
    storage, registry = _storage_and_registry()
    fake_platform = FakePlatformMetadataClient()
    timer = StageTimer(profile="demo_strict")

    # ------------------------------------------------------------------
    # 1. Manifest build (ingestion.manifest_builder).
    # ------------------------------------------------------------------
    manifest_request = BuildManifestRequest(
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
    with open_archive_path(archive_path) as reader:
        with timer.measure(_STAGE_MANIFEST):
            manifest_result = build_asset_manifest(
                reader, request=manifest_request, registry=registry
            )
    raw_manifest = manifest_result.manifest_artifact
    validated = build_validated_manifest(
        raw_manifest,
        storage=storage,
        registry=registry,
        dataset_version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        organization_id=_ORG_ID,
        project_id=_PROJECT_ID,
        audit_sink=fake_platform,
    ).validated_manifest
    manifest_rows = manifest_rows_from_artifact(storage.get(validated.uri).data)
    assert manifest_rows, "demo archive must produce manifest rows"

    # ------------------------------------------------------------------
    # 2. PredictionManifest validation (predictions.validate).
    # ------------------------------------------------------------------
    raw_predictions_artifact = _persist_demo_raw_predictions(
        registry=registry, archive_path=archive_path
    )
    manifest_object_ids = [row.object_id for row in manifest_rows]
    with timer.measure(
        _STAGE_PREDICTION_VALIDATE,
        sample_size=len(manifest_object_ids),
        sample_unit="rows",
    ):
        predictions_result = build_validated_predictions(
            raw_predictions_artifact,
            storage=storage,
            registry=registry,
            dataset_version_id=_DATASET_VERSION_ID,
            parent_version_id=_PARENT_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
            model_id=_MODEL_ID,
            model_version=_MODEL_VERSION,
            manifest_object_ids=manifest_object_ids,
            audit_sink=fake_platform,
            organization_id=_ORG_ID,
            project_id=_PROJECT_ID,
        )
    prediction_artifact = predictions_result.prediction_manifest_artifact
    prediction_rows = _parse_prediction_rows(storage.get(prediction_artifact.uri).data)

    # ------------------------------------------------------------------
    # 3. Tabular profile (tabular.profile).
    # ------------------------------------------------------------------
    profile_request = ProfileBuildRequest(
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        source_artifact_id=raw_manifest.uri,
    )
    with open_archive_path(archive_path) as reader:
        with timer.measure(
            _STAGE_TABULAR_PROFILE,
            sample_size=len(manifest_rows),
            sample_unit="rows",
        ):
            profile_result = build_tabular_profile_report(
                reader,
                request=profile_request,
                storage=storage,
                registry=registry,
                source_manifest_artifact=raw_manifest.artifact_ref,
                generated_at=_GENERATED_AT,
            )
    tabular_profile = profile_result.profile_report

    # ------------------------------------------------------------------
    # 4. Text/OCR validation + PII detection (text_ocr.validate).
    # ------------------------------------------------------------------
    text_ocr_request = TextOcrBuildRequest(
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
    support_payload, ocr_payload = _read_text_payloads(archive_path)
    with timer.measure(_STAGE_TEXT_OCR_VALIDATE, sample_unit="records"):
        support_report = validate_support_messages_jsonl(
            support_payload,
            detect_pii=True,
            registry=registry,
            build_request=text_ocr_request,
        )
        ocr_report = validate_ocr_records_jsonl(
            ocr_payload,
            detect_pii=True,
            registry=registry,
            build_request=text_ocr_request,
        )
        text_ocr_result = build_text_ocr_report(
            request=text_ocr_request,
            sources=(support_report, ocr_report),
            generated_at=_GENERATED_AT,
        )
    text_ocr_report = text_ocr_result.report

    # ------------------------------------------------------------------
    # 5. Model error analysis (predictions.model_error.analyze).
    # ------------------------------------------------------------------
    manifest_index = _manifest_index(manifest_rows)
    with timer.measure(
        _STAGE_MODEL_ERROR,
        sample_size=len(prediction_rows),
        sample_unit="rows",
    ):
        model_error_report = analyze_model_errors(
            rows=prediction_rows,
            manifest_index=manifest_index,
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            config_hash=_CONFIG_HASH,
            model_id=_MODEL_ID,
            model_version=_MODEL_VERSION,
            generated_at=_GENERATED_AT,
        )
    assert model_error_report.status is ModelErrorReportStatus.AVAILABLE

    # ------------------------------------------------------------------
    # 6. Object analytics + evidence + decision (preconditions for queue).
    # ------------------------------------------------------------------
    passports_result = build_object_analytics_passports(
        manifest_rows=manifest_rows,
        request=BuildObjectAnalyticsRequest(
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            parent_version_id=_PARENT_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
        ),
        registry=registry,
        tabular_profile=tabular_profile,
        text_ocr_report=text_ocr_report,
        model_error_report=model_error_report,
        computed_at=_GENERATED_AT,
    )
    evidence_result = build_evidence_bundles(
        passports=passports_result.passports,
        request=BuildEvidenceBundleRequest(
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            parent_version_id=_PARENT_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
        ),
        registry=registry,
        source_passports_artifact=passports_result.artifact,
    )
    decision_result = build_decision_report_artifact(
        evidence_bundles=evidence_result.evidence_bundles,
        request=BuildDecisionReportRequest(
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        registry=registry,
    )
    decision_report = decision_result.report
    # ``method_recommendations`` is part of analyze but not measured: it
    # is sub-millisecond on the demo fixture and would mostly be noise.
    _ = build_method_recommendations(
        BuildMethodRecommendationsRequest(
            tabular_profile=tabular_profile,
            decision_report=decision_report,
        )
    )

    # ------------------------------------------------------------------
    # 7. ReviewQueue build (review_queue.build).
    # ------------------------------------------------------------------
    with timer.measure(_STAGE_REVIEW_QUEUE):
        review_result = build_review_queues_artifact(
            decision_report=decision_report,
            evidence_bundles=evidence_result.evidence_bundles,
            request=BuildReviewQueuesRequest(
                dataset_id=_DATASET_ID,
                version_id=_DATASET_VERSION_ID,
                created_by_job_id=_JOB_ID,
                config_hash=_CONFIG_HASH,
                generated_at=_GENERATED_AT,
            ),
            registry=registry,
        )
    queue_types = {queue.queue_type for queue in review_result.queues}
    assert ReviewQueueType.LABEL_REVIEW in queue_types

    # ------------------------------------------------------------------
    # 8. Build report + persist + assert thresholds.
    # ------------------------------------------------------------------
    thresholds = _resolve_thresholds()
    report = timer.build_report(thresholds=thresholds)
    report_dir = _resolve_report_dir(tmp_path)
    report_path = report_dir / "performance_report.json"
    write_performance_report(report, report_path)

    # Persisted artifact must be a deterministic JSON object with at
    # least one timing per documented stage.
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["profile"] == "demo_strict"
    measured_stages = {timing["stage"] for timing in persisted["timings"]}
    expected_stages = {threshold.stage for threshold in thresholds}
    assert expected_stages.issubset(measured_stages), (
        f"missing stage timings: {sorted(expected_stages - measured_stages)}"
    )

    if not report.passed:
        violations_summary = ", ".join(
            f"{v.stage}: {v.duration_ms:.1f}ms > {v.max_duration_ms:.1f}ms"
            for v in report.violations
        )
        raise AssertionError(
            f"performance acceptance failed; violations: {violations_summary}; "
            f"see {report_path} for the full timing report"
        )


def test_stage_timer_build_report_flags_threshold_violations(tmp_path: Path) -> None:
    """The timer + report machinery must surface a failure when a stage is too slow."""
    timer = StageTimer(profile="demo_strict")
    timer.record(stage="ingestion.manifest_builder", duration_ms=10.0)
    timer.record(stage="tabular.profile", duration_ms=12_345.6)

    thresholds = (
        StageThreshold(stage="ingestion.manifest_builder", max_duration_ms=5_000.0),
        StageThreshold(stage="tabular.profile", max_duration_ms=10_000.0),
    )
    report = timer.build_report(thresholds=thresholds)

    assert report.passed is False
    violation_stages = {violation.stage for violation in report.violations}
    assert violation_stages == {"tabular.profile"}
    overshoot = next(v for v in report.violations if v.stage == "tabular.profile").overshoot_ms
    assert overshoot == pytest.approx(2_345.6, abs=0.05)

    report_path = tmp_path / "perf.json"
    write_performance_report(report, report_path)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["passed"] is False
    assert {v["stage"] for v in persisted["violations"]} == {"tabular.profile"}


def test_stage_timer_flags_missing_stage_when_threshold_documented_but_not_measured() -> None:
    """A documented stage that did not run must surface as a coverage failure."""
    timer = StageTimer(profile="demo_strict")
    timer.record(stage="ingestion.manifest_builder", duration_ms=42.0)

    thresholds = (
        StageThreshold(stage="ingestion.manifest_builder", max_duration_ms=5_000.0),
        StageThreshold(stage="review_queue.build", max_duration_ms=5_000.0),
    )
    report = timer.build_report(thresholds=thresholds)

    assert report.passed is False
    violation_stages = {violation.stage for violation in report.violations}
    assert "review_queue.build" in violation_stages


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_thresholds() -> tuple[StageThreshold, ...]:
    return tuple(
        StageThreshold(
            stage=stage,
            max_duration_ms=float(
                os.environ.get(_threshold_env_name(stage), default)
            ),
            note="demo_strict MVP threshold",
        )
        for stage, default in _DEMO_THRESHOLDS.items()
    )


def _threshold_env_name(stage: str) -> str:
    return f"DATAFORGE_PERFORMANCE_MAX_MS_{stage.upper().replace('.', '_')}"


def _resolve_report_dir(tmp_path: Path) -> Path:
    override = os.environ.get("DATAFORGE_PERFORMANCE_REPORT_DIR")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path


def _persist_demo_raw_predictions(
    *, registry: ArtifactRegistry, archive_path: Path
) -> Any:
    with open_archive_path(archive_path) as reader:
        descriptors = [d for d in reader.descriptors() if d.kind.value == "predictions"]
        assert len(descriptors) == 1
        with descriptors[0].open() as handle:
            payload = handle.read()
    return registry.save_artifact(
        artifact_kind="raw_predictions",
        data=payload,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="prediction_manifest_row.v1",
        dataset_version_id=_DATASET_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )


def _read_text_payloads(archive_path: Path) -> tuple[bytes, bytes]:
    support_payload: bytes | None = None
    ocr_payload: bytes | None = None
    with open_archive_path(archive_path) as reader:
        for descriptor in reader.descriptors():
            if descriptor.kind.value == "support_messages":
                with descriptor.open() as handle:
                    support_payload = handle.read()
            elif descriptor.kind.value == "ocr_records":
                with descriptor.open() as handle:
                    ocr_payload = handle.read()
    if support_payload is None or ocr_payload is None:
        raise AssertionError("demo archive must include support and OCR sources")
    return support_payload, ocr_payload


def _parse_prediction_rows(payload: bytes) -> tuple[PredictionRow, ...]:
    rows: list[PredictionRow] = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(PredictionRow.model_validate(json.loads(line)))
    return tuple(rows)


def _manifest_index(
    manifest_rows: tuple[ManifestRow, ...],
) -> dict[str, ManifestRowSummary]:
    index: dict[str, ManifestRowSummary] = {}
    for row in manifest_rows:
        if row.label is None:
            continue
        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        segment = metadata.get("customer_segment")
        index[row.object_id] = ManifestRowSummary(
            object_id=row.object_id,
            label=row.label,
            segment=str(segment) if segment else None,
        )
    return index


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id=_ORG_ID,
            project_id=_PROJECT_ID,
            dataset_id=_DATASET_ID,
        ),
    )
    return storage, ArtifactRegistry(storage=storage)


class _InMemoryS3Client(S3CompatibleClient):
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
            "LastModified": _GENERATED_AT,
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "Body": io.BytesIO(stored["Body"]),
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(stored["Body"])}
                for (bucket, key), stored in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc
