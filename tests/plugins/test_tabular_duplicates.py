"""Tests for TASK-042 duplicate marking/removal candidate action."""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    DuplicateActionMode,
    DuplicateActionReport,
    ErrorCode,
    RetryPolicy,
)
from app.ingestion import open_archive_path
from app.plugins.tabular import (
    DUPLICATE_GROUP_ID_COLUMN,
    DUPLICATE_REPORT_KIND,
    DUPLICATE_REPORT_SCHEMA_VERSION,
    IS_DUPLICATE_CANDIDATE_COLUMN,
    IS_DUPLICATE_KEPT_COLUMN,
    DuplicateActionError,
    ExecuteTabularDuplicatesRequest,
    execute_tabular_duplicates_action,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "e" * 64
_CREATED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
_DUP_AT = datetime(2026, 5, 24, 12, 15, tzinfo=UTC)


def test_step1_mark_mode_does_not_remove_rows_and_marks_candidates(tmp_path: Path) -> None:
    """Step 1: MARK mode adds is_duplicate_candidate columns without removing rows."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    raw_hash_before = _sha256(transactions)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)

    result = execute_tabular_duplicates_action(
        ExecuteTabularDuplicatesRequest(
            action_plan_id="action_plan_dedup_001",
            step=_mark_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            generated_at=_DUP_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert isinstance(report, DuplicateActionReport)
    assert report.mode is DuplicateActionMode.MARK
    assert report.before_row_count == report.after_row_count
    assert report.removed_count == 0
    assert report.marked_count >= 4  # demo archive has 4 duplicate pairs
    assert report.before_duplicate_pair_count >= 4
    assert report.before_duplicate_group_count >= 4
    # Mark mode keeps the same group structure in the candidate dataset.
    assert report.after_duplicate_pair_count == report.before_duplicate_pair_count
    assert report.after_duplicate_group_count == report.before_duplicate_group_count
    assert report.raw_artifact_unchanged is True
    assert report.raw_artifact_hash == raw_hash_before

    # Per-group summaries: mark mode lists every member as affected and the
    # non-canonical members as marked.
    assert len(report.groups) == report.before_duplicate_group_count
    for group in report.groups:
        assert len(group.affected_object_ids) >= 2
        assert group.kept_object_id is not None
        assert group.kept_object_id in group.affected_object_ids
        assert group.removed_object_ids == ()
        assert set(group.marked_object_ids).issubset(set(group.affected_object_ids))
        assert group.kept_object_id not in group.marked_object_ids

    # Candidate CSV: row count matches source, marker columns added.
    candidate = storage.get(result.candidate_artifact.uri)
    candidate_rows = _read_csv(candidate.data)
    assert len(candidate_rows) == report.before_row_count
    sample = candidate_rows[0]
    assert IS_DUPLICATE_CANDIDATE_COLUMN in sample
    assert DUPLICATE_GROUP_ID_COLUMN in sample
    assert IS_DUPLICATE_KEPT_COLUMN in sample
    marked_rows = [
        row for row in candidate_rows if row[IS_DUPLICATE_CANDIDATE_COLUMN] == "1"
    ]
    # Affected rows include the canonical row of every duplicate group too.
    assert len(marked_rows) >= report.marked_count + report.before_duplicate_group_count

    assert result.report_artifact.artifact_kind == DUPLICATE_REPORT_KIND
    assert result.report_artifact.schema_version == DUPLICATE_REPORT_SCHEMA_VERSION

    validate_contract_payload(
        load_contract_pack(),
        "duplicate_action_report",
        report.model_dump(mode="json"),
    )


def test_step2_remove_mode_drops_only_duplicate_followers_in_candidate(
    tmp_path: Path,
) -> None:
    """Step 2: REMOVE mode keeps one row per group and removes the rest."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    raw_hash_before = _sha256(transactions)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)

    result = execute_tabular_duplicates_action(
        ExecuteTabularDuplicatesRequest(
            action_plan_id="action_plan_dedup_001",
            step=_remove_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            generated_at=_DUP_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert report.mode is DuplicateActionMode.REMOVE_CANDIDATE
    assert report.removed_count >= 4
    assert report.removed_count == report.before_duplicate_pair_count
    assert report.after_row_count == report.before_row_count - report.removed_count
    assert report.after_duplicate_pair_count == 0
    assert report.after_duplicate_group_count == 0
    assert report.marked_count == 0
    assert report.raw_artifact_unchanged is True
    assert report.raw_artifact_hash == raw_hash_before

    for group in report.groups:
        assert group.kept_object_id is not None
        assert len(group.removed_object_ids) >= 1
        assert group.kept_object_id not in group.removed_object_ids
        # affected_object_ids is the union of kept + removed.
        assert set(group.affected_object_ids) == {
            group.kept_object_id,
            *group.removed_object_ids,
        }

    candidate = storage.get(result.candidate_artifact.uri)
    candidate_rows = _read_csv(candidate.data)
    assert len(candidate_rows) == report.after_row_count
    candidate_object_ids = {row["object_id"] for row in candidate_rows}
    for group in report.groups:
        assert group.kept_object_id in candidate_object_ids
        for removed_id in group.removed_object_ids:
            assert removed_id not in candidate_object_ids


def test_step3_raw_artifact_unchanged_after_remove(tmp_path: Path) -> None:
    """Step 3: REMOVE mode never mutates the immutable source artifact."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    raw_hash_before = _sha256(transactions)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)

    result = execute_tabular_duplicates_action(
        ExecuteTabularDuplicatesRequest(
            action_plan_id="action_plan_dedup_001",
            step=_remove_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            generated_at=_DUP_AT,
        ),
        storage=storage,
        registry=registry,
    )

    raw_after = storage.get(source_artifact.uri)
    raw_hash_after = _sha256(raw_after.data)
    assert raw_hash_after == raw_hash_before
    assert raw_after.data == transactions
    # Candidate artifact is a different content-addressed object.
    candidate = storage.get(result.candidate_artifact.uri)
    assert candidate.info.hash != raw_hash_before
    assert result.candidate_artifact.uri != source_artifact.uri


def test_invalid_step_type_is_rejected(tmp_path: Path) -> None:
    """Steps with the wrong type must not run any duplicate logic."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    bad_step = _remove_step()
    bad_step = bad_step.model_copy(update={"type": "IMPUTE_MISSING_VALUES"})

    with pytest.raises(DuplicateActionError) as excinfo:
        execute_tabular_duplicates_action(
            ExecuteTabularDuplicatesRequest(
                action_plan_id="action_plan_dedup_001",
                step=bad_step,
                dataset_id="dataset_1",
                source_dataset_version_id="dataset_version_1",
                candidate_dataset_version_id="dataset_version_2_candidate",
                source_artifact=source_artifact,
                created_by_job_id="compute_run_apply_001",
                config_hash=_CONFIG_HASH,
                generated_at=_DUP_AT,
            ),
            storage=storage,
            registry=registry,
        )
    assert excinfo.value.reason_code == "unsupported_action_step_type"
    assert excinfo.value.code is ErrorCode.ACTION_PLAN_PRECONDITION_FAILED


def test_remove_mode_is_idempotent_for_same_inputs(tmp_path: Path) -> None:
    """Same inputs produce the same candidate hash (registry idempotency)."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    request = ExecuteTabularDuplicatesRequest(
        action_plan_id="action_plan_dedup_001",
        step=_remove_step(),
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        source_artifact=source_artifact,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_DUP_AT,
        report_id="duplicate_action_report_fixed",
    )
    first = execute_tabular_duplicates_action(request, storage=storage, registry=registry)
    second = execute_tabular_duplicates_action(request, storage=storage, registry=registry)
    assert first.candidate_artifact.hash == second.candidate_artifact.hash
    assert first.report.before_row_count == second.report.before_row_count
    assert first.report.after_row_count == second.report.after_row_count
    assert first.report.removed_count == second.report.removed_count
    # Group lineage is logically identical between runs.
    first_groups = [
        (group.group_id, group.kept_object_id, tuple(group.removed_object_ids))
        for group in first.report.groups
    ]
    second_groups = [
        (group.group_id, group.kept_object_id, tuple(group.removed_object_ids))
        for group in second.report.groups
    ]
    assert first_groups == second_groups


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mark_step() -> ActionPlanStep:
    return _build_step(step_type="MARK_DUPLICATE_CANDIDATES")


def _remove_step() -> ActionPlanStep:
    return _build_step(step_type="REMOVE_DUPLICATES")


def _build_step(*, step_type: str) -> ActionPlanStep:
    return ActionPlanStep(
        step_id=f"{step_type.lower()}_tabular",
        type=step_type,
        depends_on=(),
        idempotency_key="sha256:" + "1" * 64,
        method_id="exact_signature_dedup",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="dedup_policy_v0",
        validation_gates=("schema_validation", "business_rules"),
        preconditions=("source_version_is_immutable",),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind="CANDIDATE_DATASET_VERSION",
        config={"id_column": "object_id"},
        random_seed=None,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _save_source_artifact(
    *,
    registry: ArtifactRegistry,
    data: bytes,
    version: str = "dataset_version_1",
) -> ArtifactRef:
    return registry.save_artifact(
        artifact_kind="raw_transactions",
        data=data,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id=version,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


def _read_transactions(tmp_path: Path) -> bytes:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        return reader.find_required_transactions().read_bytes()


def _read_csv(data: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    return [dict(row) for row in reader]


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


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
            "LastModified": _CREATED_AT,
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
                message="Object does not exist",
            ) from exc
