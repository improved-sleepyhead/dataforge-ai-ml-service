"""Tests for TASK-054 tabular Parquet/CSV and redacted text/OCR JSONL writers."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from app.adapters import (
    ArtifactRegistry,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    DataSplit,
    ErrorCode,
    RetryPolicy,
    SplitManifest,
    TextOcrSourceKind,
)
from app.ingestion import open_archive_path
from app.plugins.export import (
    TABULAR_EXPORT_CSV_KIND,
    TABULAR_EXPORT_PARQUET_KIND,
    TEXT_OCR_EXPORT_KIND,
    TabularExportRequest,
    TabularExportWriterError,
    TextOcrExportRequest,
    TextOcrExportWriterError,
    write_tabular_export,
    write_text_ocr_export,
)
from app.plugins.export.text_ocr_writer import iter_redacted_jsonl_lines
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    ExecuteTabularImputationRequest,
    ExecuteTabularSplitRequest,
    execute_tabular_imputation_action,
    execute_tabular_split_action,
)
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "a" * 64
_GENERATED_AT = datetime(2026, 5, 31, 14, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step 1+2: tabular export build + Parquet/CSV schema readback
# ---------------------------------------------------------------------------


def test_tabular_export_writes_parquet_and_csv_with_consistent_schema(
    tmp_path: Path,
) -> None:
    """Steps 1-2: export build writes Parquet + CSV that share the same schema."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    plan_step = _imputation_step()
    imputation = execute_tabular_imputation_action(
        ExecuteTabularImputationRequest(
            action_plan_id="action_plan_export_tabular_001",
            step=plan_step,
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            target_column="is_fraud",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_export_tabular_001",
            config_hash=_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    request = TabularExportRequest(
        dataset_id="dataset_1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=imputation.candidate_artifact.artifact_ref,
        split_manifest=None,
        write_csv=True,
        write_per_split=False,
        created_by_job_id="compute_run_export_tabular_001",
        config_hash=_CONFIG_HASH,
    )
    result = write_tabular_export(request, storage=storage, registry=registry)

    # Step 1: both Parquet primary and CSV compatibility outputs exist.
    assert result.parquet_artifact.artifact_kind == TABULAR_EXPORT_PARQUET_KIND
    assert result.csv_artifact is not None
    assert result.csv_artifact.artifact_kind == TABULAR_EXPORT_CSV_KIND
    assert result.included_row_count > 0
    assert result.excluded_blocked_count == 0
    assert "object_id" in result.columns
    assert "is_fraud" in result.columns

    # Step 2: read Parquet back through pyarrow and CSV through stdlib;
    # both produce the same column set, the same row count, and Parquet
    # carries a typed schema rather than raw strings.
    parquet_bytes = storage.get(result.parquet_artifact.uri).data
    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert tuple(table.column_names) == result.columns
    assert table.num_rows == result.included_row_count
    # Numeric columns must be coerced into numeric Arrow types.
    import pyarrow as pa

    schema = table.schema
    amount_field = schema.field("amount")
    monthly_field = schema.field("monthly_income")
    is_fraud_field = schema.field("is_fraud")
    assert amount_field.type in (pa.float64(), pa.int64())
    assert monthly_field.type in (pa.float64(), pa.int64())
    assert is_fraud_field.type == pa.int64()
    object_id_field = schema.field("object_id")
    assert object_id_field.type == pa.string()

    csv_bytes = storage.get(result.csv_artifact.uri).data
    csv_reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"), newline=""))
    csv_rows = list(csv_reader)
    assert csv_reader.fieldnames is not None
    assert tuple(csv_reader.fieldnames) == result.columns
    assert len(csv_rows) == table.num_rows


# ---------------------------------------------------------------------------
# Step 1+2: per-split export and blocked-objects exclusion
# ---------------------------------------------------------------------------


def test_per_split_export_includes_train_validation_test_and_excludes_blocked(
    tmp_path: Path,
) -> None:
    """Per-split Parquet/CSV artifacts cover every split and exclude blocked ids."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split_action = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    manifest = split_action.manifest
    blocked_object_ids = tuple(
        assignment.object_id
        for assignment in manifest.assignments
        if assignment.split is DataSplit.TRAIN
    )[:1]
    assert blocked_object_ids, "demo manifest must have at least one TRAIN row"

    request = TabularExportRequest(
        dataset_id="dataset_1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=source_artifact,
        split_manifest=manifest,
        split_manifest_artifact=split_action.split_artifact.artifact_ref,
        blocked_object_ids=blocked_object_ids,
        write_csv=True,
        write_per_split=True,
        created_by_job_id="compute_run_export_tabular_002",
        config_hash=_CONFIG_HASH,
    )
    result = write_tabular_export(request, storage=storage, registry=registry)

    # The blocked id must not appear anywhere in primary, CSV, or
    # per-split artifacts.
    assert result.excluded_blocked_count == len(blocked_object_ids)
    primary_ids = _read_object_ids_from_parquet(
        storage.get(result.parquet_artifact.uri).data
    )
    assert all(blocked_id not in primary_ids for blocked_id in blocked_object_ids)

    splits_present = {entry.split for entry in result.per_split}
    assert DataSplit.TRAIN in splits_present
    assert DataSplit.VALIDATION in splits_present or DataSplit.TEST in splits_present
    assert result.split_manifest_artifact == split_action.split_artifact.artifact_ref
    assert split_action.split_artifact.artifact_ref in result.all_artifact_refs()

    for entry in result.per_split:
        ids = _read_object_ids_from_parquet(
            storage.get(entry.parquet_artifact.uri).data
        )
        for blocked_id in blocked_object_ids:
            assert blocked_id not in ids
        assert entry.row_count == len(ids)


def test_tabular_export_rejects_source_without_object_id_column(tmp_path: Path) -> None:
    """Sources without an object_id column produce an explicit reason code."""
    storage, registry = _storage_and_registry()
    payload = b"id,amount\n1,10.0\n2,20.0\n"
    artifact = registry.save_artifact(
        artifact_kind="raw_transactions",
        data=payload,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_v1",
        created_by_job_id="compute_run_export_tabular_003",
        config_hash=_CONFIG_HASH,
    )
    request = TabularExportRequest(
        dataset_id="dataset_1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=artifact.artifact_ref,
        split_manifest=None,
        write_csv=False,
        write_per_split=False,
        created_by_job_id="compute_run_export_tabular_003",
        config_hash=_CONFIG_HASH,
    )
    with pytest.raises(TabularExportWriterError) as exc:
        write_tabular_export(request, storage=storage, registry=registry)
    assert exc.value.reason_code == "object_id_column_missing_in_source"


# ---------------------------------------------------------------------------
# Step 3: redacted JSONL export carries no raw PII
# ---------------------------------------------------------------------------


def test_text_ocr_export_writes_redacted_jsonl_without_raw_pii(tmp_path: Path) -> None:
    """Step 3: the redacted JSONL export only carries safe redacted fields."""
    storage, registry = _storage_and_registry()
    raw = (
        json.dumps({"object_id": "support_001", "text": "phone 555-123-4567"})
        + "\n"
        + json.dumps(
            {
                "object_id": "support_002",
                "text": "email john.doe@example.com loves our product",
            }
        )
        + "\n"
        + json.dumps({"object_id": "support_003", "text": "everything is fine"})
        + "\n"
    ).encode("utf-8")
    source = registry.save_artifact(
        artifact_kind="text_ocr_source",
        data=raw,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="text_ocr_record.v1",
        dataset_version_id="dataset_version_v1",
        created_by_job_id="compute_run_export_text_001",
        config_hash=_CONFIG_HASH,
    )

    request = TextOcrExportRequest(
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=source.artifact_ref,
        source_kind=TextOcrSourceKind.SUPPORT_MESSAGES,
        source_name="support_messages.jsonl",
        blocked_object_ids=("support_003",),
        detect_pii_only_records=False,
        created_by_job_id="compute_run_export_text_001",
        config_hash=_CONFIG_HASH,
    )
    result = write_text_ocr_export(request, storage=storage, registry=registry)

    assert result.redacted_artifact.artifact_kind == TEXT_OCR_EXPORT_KIND
    assert result.excluded_blocked_count == 1
    # Two records remain after blocking the third object_id.
    assert result.record_count == 2

    payload = storage.get(result.redacted_artifact.uri).data
    records = list(iter_redacted_jsonl_lines(payload))
    assert len(records) == 2

    # Every record carries only safe fields.
    allowed_keys = {
        "object_id",
        "redacted_text",
        "pii_token_count",
        "redacted_text_sha256",
    }
    for record in records:
        assert set(record.keys()) <= allowed_keys
        # Raw PII must never appear in the redacted text.
        text = str(record["redacted_text"])
        assert "555-123-4567" not in text
        assert "john.doe@example.com" not in text
        assert "[REDACTED" in text or record["pii_token_count"] == 0
        assert isinstance(record["redacted_text_sha256"], str)
        assert record["redacted_text_sha256"].startswith("sha256:")

    # The blocked object_id is not present.
    blocked_ids = {record["object_id"] for record in records}
    assert "support_003" not in blocked_ids
    # PII record count tracks records with at least one finding.
    assert result.pii_record_count == 2


def test_text_ocr_export_rejects_empty_source() -> None:
    """An empty text/OCR source raises a stable reason code."""
    storage, registry = _storage_and_registry()
    artifact = registry.save_artifact(
        artifact_kind="text_ocr_source",
        data=b"",
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="text_ocr_record.v1",
        dataset_version_id="dataset_version_v1",
        created_by_job_id="compute_run_export_text_002",
        config_hash=_CONFIG_HASH,
    )
    request = TextOcrExportRequest(
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=artifact.artifact_ref,
        source_kind=TextOcrSourceKind.SUPPORT_MESSAGES,
        source_name="support_messages.jsonl",
        created_by_job_id="compute_run_export_text_002",
        config_hash=_CONFIG_HASH,
    )
    with pytest.raises(TextOcrExportWriterError) as exc:
        write_text_ocr_export(request, storage=storage, registry=registry)
    assert exc.value.reason_code == "empty_text_ocr_source"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _imputation_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="impute_income_001",
        type="IMPUTE_MISSING_VALUES",
        depends_on=(),
        idempotency_key="sha256:" + "1" * 64,
        method_id="group_median",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="method_policy_v0",
        validation_gates=("schema_validation",),
        preconditions=("source_version_is_immutable",),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind="candidate_tabular_dataset",
        config={"column": "monthly_income", "group_key": "customer_segment"},
        random_seed=None,
        retry_policy=RetryPolicy(
            max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _split_request(*, source_artifact: ArtifactRef) -> ExecuteTabularSplitRequest:
    split_step = ActionPlanStep(
        step_id="create_split_tabular",
        type="CREATE_SPLIT",
        depends_on=(),
        idempotency_key="sha256:" + "3" * 64,
        method_id="group_stratified",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "4" * 64,
        policy_version="split_policy_v0",
        validation_gates=("schema_validation",),
        preconditions=("source_version_is_immutable",),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind=SPLIT_MANIFEST_KIND,
        config={
            "strategy": "group_stratified",
            "group_key": "customer_id_hash",
            "target_column": "is_fraud",
        },
        random_seed=42,
        retry_policy=RetryPolicy(
            max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )
    return ExecuteTabularSplitRequest(
        action_plan_id="action_plan_export_tabular_002",
        step=split_step,
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=source_artifact,
        created_by_job_id="compute_run_export_tabular_002",
        config_hash=_CONFIG_HASH,
        target_column="is_fraud",
        seed=42,
        generated_at=_GENERATED_AT,
    )


def _read_object_ids_from_parquet(payload: bytes) -> set[str]:
    table = pq.read_table(io.BytesIO(payload))
    column = table.column("object_id").to_pylist()
    return {str(value) for value in column if value is not None}


def _source_transactions_artifact(
    tmp_path: Path, registry: ArtifactRegistry
) -> ArtifactRef:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        transactions = reader.find_required_transactions().read_bytes()
    return registry.save_artifact(
        artifact_kind="raw_transactions",
        data=transactions,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_v1",
        created_by_job_id="compute_run_export_tabular_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
        ),
    )
    return storage, ArtifactRegistry(storage=storage)


_ = SplitManifest  # ensure the symbol stays exported for typing tests


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
                {"Key": key, "Size": len(payload["Body"])}
                for (bucket, key), payload in self._objects.items()
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
