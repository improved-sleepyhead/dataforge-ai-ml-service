"""Tests for TASK-045 post-split leakage checks."""

from __future__ import annotations

import csv
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
    DataSplit,
    ErrorCode,
    LeakageCandidate,
    LeakageCheckSeverity,
    LeakageCheckStatus,
    LeakageCheckType,
    LeakageDiagnostics,
    RetryPolicy,
    SplitLeakageReport,
    TabularProfileLineage,
    TabularProfileReport,
)
from app.ingestion import open_archive_path
from app.plugins.tabular import (
    SPLIT_LEAKAGE_REPORT_KIND,
    SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION,
    ExecuteTabularSplitRequest,
    RunSplitLeakageChecksRequest,
    execute_tabular_split_action,
    run_split_leakage_checks,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "e" * 64
_LEAKAGE_CONFIG_HASH = "sha256:" + "f" * 64
_CREATED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
_LEAKAGE_AT = datetime(2026, 5, 24, 12, 5, tzinfo=UTC)


def test_step1_clean_split_passes_all_leakage_checks(tmp_path: Path) -> None:
    """Step 1: leakage checks on the deterministic demo split pass."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id="action_plan_split_001",
            step=_split_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            target_column="is_fraud",
            seed=42,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    result = run_split_leakage_checks(
        RunSplitLeakageChecksRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_LEAKAGE_CONFIG_HASH,
            action_plan_id="action_plan_split_001",
            step_id="leakage_check_001",
            generated_at=_LEAKAGE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert isinstance(report, SplitLeakageReport)
    assert report.leakage_detected is False
    assert report.block_model_evaluation is False
    assert report.block_training is False
    assert report.leakage_risk_score == 0.0
    assert report.policy_version == "split_leakage_policy_v0"
    assert report.split_manifest_id == split.manifest.split_manifest_id

    by_type = {check.check_type: check for check in report.checks}
    assert set(by_type) == {
        LeakageCheckType.EXACT_HASH,
        LeakageCheckType.GROUP_KEY,
        LeakageCheckType.TARGET_LEAKAGE_CANDIDATE,
    }
    assert by_type[LeakageCheckType.EXACT_HASH].status is LeakageCheckStatus.PASSED
    assert by_type[LeakageCheckType.GROUP_KEY].status is LeakageCheckStatus.PASSED
    assert by_type[LeakageCheckType.GROUP_KEY].affected_columns == ("customer_id_hash",)
    assert by_type[LeakageCheckType.TARGET_LEAKAGE_CANDIDATE].status is (
        LeakageCheckStatus.NOT_APPLICABLE
    )

    assert result.report_artifact.artifact_kind == SPLIT_LEAKAGE_REPORT_KIND
    assert result.report_artifact.schema_version == SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION
    stored = storage.get(result.report_artifact.uri)
    assert stored.info.metadata["leakage-detected"] == "false"
    assert stored.info.metadata["block-model-evaluation"] == "false"
    assert stored.info.metadata["block-training"] == "false"
    assert stored.info.metadata["split-manifest-id"] == split.manifest.split_manifest_id

    validate_contract_payload(
        load_contract_pack(),
        "split_leakage_report",
        report.model_dump(mode="json"),
    )


def test_step2_group_leakage_fixture_blocks_model_evaluation(tmp_path: Path) -> None:
    """Step 2: synthetic group-leakage fixture produces a BLOCK_MODEL_EVALUATION blocker."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id="action_plan_split_001",
            step=_split_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            target_column="is_fraud",
            seed=42,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    # Inject group leakage by overwriting the customer_id_hash for one
    # train row with the customer_id_hash of one test row. The split
    # manifest stays unchanged but the source CSV now contains the
    # leaked group key in two splits.
    leaked_transactions = _inject_group_leakage(transactions, split.manifest)
    leaked_source = _save_source_artifact(
        registry=registry,
        data=leaked_transactions,
        version="dataset_version_2_candidate_with_leakage",
        config_hash=_LEAKAGE_CONFIG_HASH,
    )

    result = run_split_leakage_checks(
        RunSplitLeakageChecksRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            source_artifact=leaked_source,
            created_by_job_id="compute_run_apply_001",
            config_hash=_LEAKAGE_CONFIG_HASH,
            generated_at=_LEAKAGE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert report.leakage_detected is True
    assert report.block_model_evaluation is True
    assert report.leakage_risk_score == 1.0

    by_type = {check.check_type: check for check in report.checks}
    group_check = by_type[LeakageCheckType.GROUP_KEY]
    assert group_check.status is LeakageCheckStatus.FAILED
    assert group_check.severity is LeakageCheckSeverity.BLOCKER
    assert group_check.reason_code == "split_leakage"
    assert group_check.block_action == "BLOCK_MODEL_EVALUATION"
    assert group_check.findings_count >= 1
    assert group_check.findings, "blocker check must surface concrete findings"
    finding = group_check.findings[0]
    assert len(finding.splits) >= 2
    assert finding.column == "customer_id_hash"


def test_step3_target_leakage_candidate_blocks_training(tmp_path: Path) -> None:
    """Step 3: high-target-match leakage candidate emits BLOCK_TRAINING."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id="action_plan_split_001",
            step=_split_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            target_column="is_fraud",
            seed=42,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    profile = _profile_with_target_leakage_candidate(
        source_artifact=source_artifact,
        target_column="is_fraud",
        leakage_column="manual_review_flag",
    )

    result = run_split_leakage_checks(
        RunSplitLeakageChecksRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_LEAKAGE_CONFIG_HASH,
            tabular_profile_report=profile,
            generated_at=_LEAKAGE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert report.leakage_detected is True
    assert report.block_training is True
    # The synthetic demo split has no group-key leakage, so the only
    # active blocker comes from the target leakage candidate.
    assert report.block_model_evaluation is False

    by_type = {check.check_type: check for check in report.checks}
    target_check = by_type[LeakageCheckType.TARGET_LEAKAGE_CANDIDATE]
    assert target_check.status is LeakageCheckStatus.FAILED
    assert target_check.severity is LeakageCheckSeverity.BLOCKER
    assert target_check.reason_code == "target_leakage_candidate"
    assert target_check.block_action == "BLOCK_TRAINING"
    assert target_check.affected_columns == ("manual_review_flag",)


def test_step4_exact_hash_leakage_blocks_model_evaluation(tmp_path: Path) -> None:
    """Step 4: identical row content across splits blocks model evaluation."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id="action_plan_split_001",
            step=_split_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            target_column="is_fraud",
            seed=42,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    # Inject exact row-content leakage: pick one train and one test row
    # whose customer_id_hash differs (so group_key check stays clean),
    # then copy the train row's content (excluding object_id) into the
    # test row.
    leaked_transactions = _inject_exact_hash_leakage(transactions, split.manifest)
    leaked_source = _save_source_artifact(
        registry=registry,
        data=leaked_transactions,
        version="dataset_version_2_candidate_exact_hash",
        config_hash=_LEAKAGE_CONFIG_HASH,
    )

    result = run_split_leakage_checks(
        RunSplitLeakageChecksRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            source_artifact=leaked_source,
            created_by_job_id="compute_run_apply_001",
            config_hash=_LEAKAGE_CONFIG_HASH,
            generated_at=_LEAKAGE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert report.leakage_detected is True
    assert report.block_model_evaluation is True

    by_type = {check.check_type: check for check in report.checks}
    exact_check = by_type[LeakageCheckType.EXACT_HASH]
    assert exact_check.status is LeakageCheckStatus.FAILED
    assert exact_check.severity is LeakageCheckSeverity.BLOCKER
    assert exact_check.reason_code == "split_leakage"
    assert exact_check.block_action == "BLOCK_MODEL_EVALUATION"
    assert exact_check.findings_count >= 1


def test_warning_only_target_leakage_does_not_block(tmp_path: Path) -> None:
    """Name-pattern target leakage candidate without high target match → warning, not block."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id="action_plan_split_001",
            step=_split_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            target_column="is_fraud",
            seed=42,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    profile = _profile_with_target_leakage_candidate(
        source_artifact=source_artifact,
        target_column="is_fraud",
        leakage_column="chargeback_status_after_investigation",
        reason_code="name_pattern_leakage_candidate",
    )

    result = run_split_leakage_checks(
        RunSplitLeakageChecksRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_LEAKAGE_CONFIG_HASH,
            tabular_profile_report=profile,
            generated_at=_LEAKAGE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    # Warning candidate is recorded but does not flip a hard blocker.
    assert report.block_training is False
    assert report.leakage_detected is False
    by_type = {check.check_type: check for check in report.checks}
    target_check = by_type[LeakageCheckType.TARGET_LEAKAGE_CANDIDATE]
    assert target_check.status is LeakageCheckStatus.FAILED
    assert target_check.severity is LeakageCheckSeverity.WARNING
    assert target_check.block_action is None


def test_split_manifest_object_id_missing_in_source_raises(tmp_path: Path) -> None:
    """If split manifest references an object_id missing from source, the run errors out."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id="action_plan_split_001",
            step=_split_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            target_column="is_fraud",
            seed=42,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    # Drop one row from the source CSV but keep the original split manifest.
    rows = _read_csv(transactions)
    truncated = _write_csv(rows[:-1])
    truncated_source = _save_source_artifact(
        registry=registry,
        data=truncated,
        version="dataset_version_2_candidate_truncated",
        config_hash=_LEAKAGE_CONFIG_HASH,
    )

    with pytest.raises(Exception) as excinfo:
        run_split_leakage_checks(
            RunSplitLeakageChecksRequest(
                dataset_id="dataset_1",
                source_dataset_version_id="dataset_version_1",
                candidate_dataset_version_id="dataset_version_2_candidate",
                split_manifest=split.manifest,
                split_manifest_artifact=split.split_artifact.artifact_ref,
                source_artifact=truncated_source,
                created_by_job_id="compute_run_apply_001",
                config_hash=_LEAKAGE_CONFIG_HASH,
                generated_at=_LEAKAGE_AT,
            ),
            storage=storage,
            registry=registry,
        )
    assert "split_manifest_object_id_missing_in_source" in getattr(
        excinfo.value, "reason_code", ""
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="create_split_tabular",
        type="CREATE_SPLIT",
        depends_on=(),
        idempotency_key="sha256:" + "1" * 64,
        method_id="group_stratified",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="split_policy_v0",
        validation_gates=("schema_validation", "split_policy_check"),
        preconditions=("source_version_is_immutable", "split_before_train_only_augmentation"),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind="split_manifest",
        config={
            "strategy": "group_stratified",
            "group_key": "customer_id_hash",
            "target_column": "is_fraud",
        },
        random_seed=42,
        retry_policy=RetryPolicy(max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _save_source_artifact(
    *,
    registry: ArtifactRegistry,
    data: bytes,
    version: str = "dataset_version_1",
    config_hash: str = _CONFIG_HASH,
) -> ArtifactRef:
    return registry.save_artifact(
        artifact_kind="raw_transactions",
        data=data,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id=version,
        created_by_job_id="compute_run_apply_001",
        config_hash=config_hash,
    ).artifact_ref


def _read_transactions(tmp_path: Path) -> bytes:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        return reader.find_required_transactions().read_bytes()


def _read_csv(data: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    rows = [dict(row) for row in reader]
    return rows


def _write_csv(rows: list[dict[str, str]]) -> bytes:
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _inject_group_leakage(
    transactions: bytes,
    manifest: Any,
) -> bytes:
    rows = _read_csv(transactions)
    by_object_id = {row["object_id"]: index for index, row in enumerate(rows)}

    train_target = next(
        a for a in manifest.assignments if a.split is DataSplit.TRAIN
    )
    test_donor = next(
        a
        for a in manifest.assignments
        if a.split is DataSplit.TEST and a.group_value != train_target.group_value
    )
    rows[by_object_id[train_target.object_id]]["customer_id_hash"] = (
        rows[by_object_id[test_donor.object_id]]["customer_id_hash"]
    )
    return _write_csv(rows)


def _inject_exact_hash_leakage(
    transactions: bytes,
    manifest: Any,
) -> bytes:
    rows = _read_csv(transactions)
    by_object_id = {row["object_id"]: index for index, row in enumerate(rows)}

    train_donor = next(
        a for a in manifest.assignments if a.split is DataSplit.TRAIN
    )
    test_target = next(
        a
        for a in manifest.assignments
        if a.split is DataSplit.TEST and a.group_value != train_donor.group_value
    )
    donor_row = rows[by_object_id[train_donor.object_id]]
    target_row = rows[by_object_id[test_target.object_id]]
    for column, value in donor_row.items():
        if column == "object_id":
            continue
        target_row[column] = value
    return _write_csv(rows)


def _profile_with_target_leakage_candidate(
    *,
    source_artifact: ArtifactRef,
    target_column: str,
    leakage_column: str,
    reason_code: str = "high_target_match_rate",
) -> TabularProfileReport:
    return TabularProfileReport(
        profile_id="profile_001",
        profile_schema_version="tabular_profile_report.v1",
        source_system="transactions",
        row_count=200,
        column_count=8,
        columns=(),
        target_column=target_column,
        group_key_columns=("customer_id_hash",),
        id_columns=("object_id",),
        leakage=LeakageDiagnostics(
            candidates=(
                LeakageCandidate(
                    column=leakage_column,
                    reason_code=reason_code,
                    target_match_rate=1.0 if reason_code == "high_target_match_rate" else None,
                ),
            ),
        ),
        lineage=TabularProfileLineage(
            dataset_id="dataset_1",
            version_id="dataset_version_1",
            parent_version_id="dataset_version_1",
            created_by_job_id="compute_run_analyze_001",
            config_hash=_CONFIG_HASH,
            source_manifest_artifact=source_artifact,
            source_artifact_id=source_artifact.artifact_id,
        ),
        generated_at=_CREATED_AT,
    )


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
