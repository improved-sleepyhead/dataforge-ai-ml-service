"""Tests for TASK-025A: PredictionManifest ingestion and validation.

Acceptance criteria covered:

* ANALYZE_ONLY accepts an optional predictions artifact alongside the
  validated asset manifest.
* PredictionManifest carries object_id, true_label, predicted_label,
  predicted_proba, confidence, split, model_id, model_version,
  inference_timestamp.
* Prediction rows join to ManifestRow by object_id; unmatched
  predictions and missing-prediction coverage are reported.
* predicted_proba is validated as class -> probability map; confidence =
  max(predicted_proba); predicted_label must equal argmax.
* PredictionManifest is saved as immutable artifact with hash, model_id,
  model_version, inference_timestamp and schema_version.
* Predictions are evidence, not authority — they never override labels,
  approve export or mutate the dataset (covered indirectly: ingestion
  produces only an immutable artifact + validation reports).
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    AuditEventType,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import ErrorCode
from app.ingestion import (
    PREDICTION_MANIFEST_KIND,
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    BuildManifestRequest,
    PredictionContractValidationError,
    PredictionValidationError,
    build_asset_manifest,
    build_validated_manifest,
    build_validated_predictions,
    compute_prediction_coverage,
    compute_row_uncertainty_signals,
    open_archive_path,
    validate_predictions_jsonl,
)
from app.ingestion.predictions import (
    PREDICTION_MANIFEST_FORMAT,
    PREDICTION_MANIFEST_MEDIA_TYPE,
)
from tests.fixtures.demo_archive import build_demo_archive

_DATASET_VERSION_ID = "dataset_version_demo"
_PARENT_VERSION_ID = "dataset_version_parent"
_JOB_ID = "compute_run_predictions"
_CONFIG_HASH = "sha256:" + "a" * 64
_MODEL_ID = "fraud_baseline"
_MODEL_VERSION = "2026-05-14"


# ---------------------------------------------------------------------------
# Step 1: validate demo predictions fixture
# ---------------------------------------------------------------------------


def test_validate_demo_predictions_jsonl(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        descriptors = [
            d for d in reader.descriptors() if d.kind.value == "predictions"
        ]
        assert len(descriptors) == 1
        with descriptors[0].open() as handle:
            payload = handle.read()

    report, rows = validate_predictions_jsonl(payload)

    assert report.row_count == len(rows)
    assert report.row_count >= 100
    # Every row carries the demo model identifiers.
    for row in rows:
        assert row.model_id == _MODEL_ID
        assert row.model_version == _MODEL_VERSION
    # Splits observed in demo data: train + validation + test (the demo
    # builder labels every row, but we only assert the keyset is non
    # empty and contains validation rows).
    assert "validation" in report.rows_by_split


# ---------------------------------------------------------------------------
# Step 2: object_id join coverage and unmatched/missing predictions
# ---------------------------------------------------------------------------


def test_prediction_coverage_report_against_manifest(tmp_path: Path) -> None:
    storage, registry, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest_artifact(storage, registry, archive_path)
    raw_predictions = _persist_demo_predictions(storage, registry, archive_path)

    manifest_object_ids = _manifest_object_ids_from_artifact(storage, validated)

    result = build_validated_predictions(
        raw_predictions,
        storage=storage,
        registry=registry,
        dataset_version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        manifest_object_ids=manifest_object_ids,
    )

    coverage = result.validation_report.coverage
    assert coverage is not None
    # Demo: predictions cover every transaction (200 rows), while the
    # validated manifest also includes support_messages/ocr_records/image
    # rows. Coverage is < 1.0 because non-tabular rows have no prediction;
    # those become missing_prediction_object_ids (acceptable in
    # ANALYZE_ONLY).
    assert coverage.unmatched_prediction_object_ids == ()
    assert coverage.prediction_row_count == 200
    assert coverage.manifest_row_count >= 200
    assert 0.5 < coverage.coverage_ratio <= 1.0
    # Every prediction object_id should be present in the manifest.
    assert len(coverage.matched_object_ids) == coverage.prediction_row_count
    # Manifest rows without a prediction become missing rows (text/OCR/etc.).
    if coverage.manifest_row_count > coverage.prediction_row_count:
        assert len(coverage.missing_prediction_object_ids) > 0


def test_prediction_coverage_unmatched_and_missing_rows() -> None:
    """compute_prediction_coverage reports unmatched and missing object_ids."""
    manifest_ids = ["a", "b", "c", "d"]
    rows = [
        _row_for("a", "fraud", "fraud", {"fraud": 0.9, "not_fraud": 0.1}),
        _row_for("b", "fraud", "not_fraud", {"fraud": 0.4, "not_fraud": 0.6}),
        _row_for("z", "fraud", "fraud", {"fraud": 0.7, "not_fraud": 0.3}),
    ]
    coverage = compute_prediction_coverage(
        manifest_object_ids=manifest_ids,
        prediction_rows=rows,
    )
    assert coverage.coverage_ratio == pytest.approx(0.5)
    assert coverage.unmatched_prediction_object_ids == ("z",)
    assert set(coverage.missing_prediction_object_ids) == {"c", "d"}


# ---------------------------------------------------------------------------
# Step 3: invalid probability map → CONTRACT/PREDICTION_VALIDATION_FAILED
# ---------------------------------------------------------------------------


def test_invalid_probability_map_raises_prediction_validation_failed() -> None:
    """Probabilities sum != 1 → PredictionValidationError with stable code."""
    bad_row = {
        "object_id": "obj_1",
        "true_label": "fraud",
        "predicted_label": "fraud",
        "predicted_proba": {"fraud": 0.7, "not_fraud": 0.7},
        "confidence": 0.7,
        "split": "validation",
        "model_id": _MODEL_ID,
        "model_version": _MODEL_VERSION,
        "inference_timestamp": "2026-05-20T07:15:00Z",
    }
    payload = (json.dumps(bad_row) + "\n").encode("utf-8")
    with pytest.raises(PredictionValidationError) as exc_info:
        validate_predictions_jsonl(payload)
    assert exc_info.value.code is ErrorCode.PREDICTION_VALIDATION_FAILED
    assert exc_info.value.reason_code == "probability_sum_invalid"


def test_argmax_mismatch_raises_prediction_validation_failed() -> None:
    bad_row = {
        "object_id": "obj_1",
        "true_label": "fraud",
        "predicted_label": "fraud",  # argmax is not_fraud
        "predicted_proba": {"fraud": 0.4, "not_fraud": 0.6},
        "confidence": 0.6,
        "split": "validation",
        "model_id": _MODEL_ID,
        "model_version": _MODEL_VERSION,
        "inference_timestamp": "2026-05-20T07:15:00Z",
    }
    payload = (json.dumps(bad_row) + "\n").encode("utf-8")
    with pytest.raises(PredictionValidationError) as exc_info:
        validate_predictions_jsonl(payload)
    assert exc_info.value.code is ErrorCode.PREDICTION_VALIDATION_FAILED
    assert exc_info.value.reason_code == "argmax_mismatch"


def test_missing_required_field_raises_contract_validation_failed() -> None:
    bad_row = {
        "object_id": "obj_1",
        # missing "true_label"
        "predicted_label": "fraud",
        "predicted_proba": {"fraud": 0.6, "not_fraud": 0.4},
        "confidence": 0.6,
        "split": "validation",
        "model_id": _MODEL_ID,
        "model_version": _MODEL_VERSION,
        "inference_timestamp": "2026-05-20T07:15:00Z",
    }
    payload = (json.dumps(bad_row) + "\n").encode("utf-8")
    with pytest.raises(PredictionContractValidationError) as exc_info:
        validate_predictions_jsonl(payload)
    assert exc_info.value.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    assert exc_info.value.reason_code == "missing_required_field"


def test_invalid_json_raises_contract_validation_failed() -> None:
    payload = b"{not valid json}\n"
    with pytest.raises(PredictionContractValidationError) as exc_info:
        validate_predictions_jsonl(payload)
    assert exc_info.value.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    assert exc_info.value.reason_code == "invalid_json"


def test_empty_predictions_raises_contract_validation_failed() -> None:
    payload = b""
    with pytest.raises(PredictionContractValidationError) as exc_info:
        validate_predictions_jsonl(payload)
    assert exc_info.value.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    assert exc_info.value.reason_code == "empty_predictions"


# ---------------------------------------------------------------------------
# Step 4: immutable ArtifactRef for prediction_manifest
# ---------------------------------------------------------------------------


def test_validated_predictions_artifact_metadata_and_audit(
    tmp_path: Path,
) -> None:
    storage, registry, archive_path = _setup_demo(tmp_path)
    raw_predictions = _persist_demo_predictions(storage, registry, archive_path)
    audit_sink = FakePlatformMetadataClient()

    result = build_validated_predictions(
        raw_predictions,
        storage=storage,
        registry=registry,
        dataset_version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        audit_sink=audit_sink,
        organization_id="org_test",
        project_id="project_test",
    )

    artifact = result.prediction_manifest_artifact
    assert artifact.artifact_kind == PREDICTION_MANIFEST_KIND
    assert artifact.schema_version == PREDICTION_MANIFEST_SCHEMA_VERSION
    assert artifact.uri.startswith("s3://")

    stored = storage.get(artifact.uri)
    assert stored.info.metadata["artifact-kind"] == PREDICTION_MANIFEST_KIND
    assert stored.info.metadata["schema-version"] == PREDICTION_MANIFEST_SCHEMA_VERSION
    assert stored.info.metadata["model-id"] == _MODEL_ID
    assert stored.info.metadata["model-version"] == _MODEL_VERSION
    assert "inference-timestamp" in stored.info.metadata

    # Idempotent re-run produces the same content-addressed URI/hash.
    result2 = build_validated_predictions(
        raw_predictions,
        storage=storage,
        registry=registry,
        dataset_version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
    )
    assert result2.prediction_manifest_artifact.uri == artifact.uri
    assert result2.prediction_manifest_artifact.hash == artifact.hash

    # Audit event recorded.
    audit_events = audit_sink.snapshot().audit_events
    assert any(
        event.event_type is AuditEventType.PREDICTIONS_VALIDATED
        for event in audit_events
    )
    audit = next(
        event for event in audit_events
        if event.event_type is AuditEventType.PREDICTIONS_VALIDATED
    )
    assert audit.metadata["model_id"] == _MODEL_ID
    assert audit.metadata["model_version"] == _MODEL_VERSION
    assert audit.metadata["output_artifact_kind"] == PREDICTION_MANIFEST_KIND
    assert audit.metadata["input_hash"] == raw_predictions.hash


def test_predictions_artifact_format_and_media_type() -> None:
    assert PREDICTION_MANIFEST_FORMAT == "jsonl"
    assert PREDICTION_MANIFEST_MEDIA_TYPE == "application/jsonl"


# ---------------------------------------------------------------------------
# Uncertainty signal helpers (used by TASK-025B downstream)
# ---------------------------------------------------------------------------


def test_compute_row_uncertainty_signals_basic() -> None:
    row = _row_for(
        "obj_1", "fraud", "not_fraud",
        {"fraud": 0.4, "not_fraud": 0.6},
    )
    signals = compute_row_uncertainty_signals(row)
    assert signals["confidence"] == pytest.approx(0.6)
    assert signals["margin"] == pytest.approx(0.2)
    # Two-class entropy with p=0.4/0.6.
    assert 0.0 < signals["entropy"] < 1.0
    # Normalized entropy in [0, 1].
    assert 0.0 < signals["normalized_entropy"] <= 1.0


def test_compute_row_uncertainty_signals_high_confidence() -> None:
    row = _row_for(
        "obj_1", "fraud", "fraud",
        {"fraud": 0.99, "not_fraud": 0.01},
    )
    signals = compute_row_uncertainty_signals(row)
    assert signals["confidence"] == pytest.approx(0.99)
    assert signals["margin"] == pytest.approx(0.98)
    assert signals["normalized_entropy"] < 0.2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _setup_demo(tmp_path: Path) -> tuple[
    MinioObjectStorageAdapter, ArtifactRegistry, Path,
]:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_test",
            project_id="project_test",
            dataset_id="dataset_demo",
        ),
    )
    registry = ArtifactRegistry(storage=storage)
    return storage, registry, built.archive_path


def _validated_manifest_artifact(
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    archive_path: Path,
) -> Any:
    request = BuildManifestRequest(
        dataset_id="dataset_demo",
        version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
    with open_archive_path(archive_path) as reader:
        manifest_result = build_asset_manifest(
            reader, request=request, registry=registry
        )
    return build_validated_manifest(
        manifest_result.manifest_artifact,
        storage=storage,
        registry=registry,
        dataset_version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        organization_id="org_test",
        project_id="project_test",
    ).validated_manifest


def _persist_demo_predictions(
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    archive_path: Path,
) -> Any:
    """Persist demo predictions.jsonl as a raw artifact (kind=raw_predictions)."""
    with open_archive_path(archive_path) as reader:
        descriptors = [
            d for d in reader.descriptors() if d.kind.value == "predictions"
        ]
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


def _manifest_object_ids_from_artifact(
    storage: MinioObjectStorageAdapter,
    validated: Any,
) -> list[str]:
    """Return source_object_id (raw key) values from validated manifest.

    Predictions reference the source object_id (e.g. txn_NNNNN), not the
    derived obj_<24hex>; manifest builder preserves the original key in
    metadata.source_object_id so downstream join logic can use it.
    """
    raw = storage.get(validated.uri).data.decode("utf-8")
    object_ids: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = record.get("metadata", {})
        source_id = metadata.get("source_object_id")
        if source_id is not None:
            object_ids.append(source_id)
    return object_ids


def _row_for(
    object_id: str,
    true_label: str,
    predicted_label: str,
    proba: dict[str, float],
) -> Any:
    from app.domain import DataSplit, PredictionRow

    return PredictionRow(
        object_id=object_id,
        true_label=true_label,
        predicted_label=predicted_label,
        predicted_proba=proba,
        confidence=max(proba.values()),
        split=DataSplit.VALIDATION,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        inference_timestamp=datetime(2026, 5, 20, 7, 15, tzinfo=UTC),
    )


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
            "LastModified": datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(record["Body"])}
                for (bucket, key), record in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
