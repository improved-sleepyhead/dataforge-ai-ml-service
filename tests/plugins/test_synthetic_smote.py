"""Tests for TASK-046 SMOTE rare-class augmentation."""

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
    SMOTE_FORMULA,
    ActionPlanStep,
    ArtifactRef,
    DataSplit,
    ErrorCode,
    RetryPolicy,
    SyntheticDatasetReport,
    SyntheticGenerationMethod,
)
from app.ingestion import open_archive_path
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SYNTHETIC_REPORT_KIND,
    SYNTHETIC_REPORT_SCHEMA_VERSION,
    ExecuteSmoteAugmentationRequest,
    ExecuteTabularSplitRequest,
    SmoteExecutionError,
    execute_smote_augmentation_action,
    execute_tabular_split_action,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "e" * 64
_SMOTE_CONFIG_HASH = "sha256:" + "f" * 64
_CREATED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
_SMOTE_AT = datetime(2026, 5, 24, 12, 10, tzinfo=UTC)


def test_smote_generates_synthetic_train_only_rows_with_lineage(tmp_path: Path) -> None:
    """Steps 1-2-4: SMOTE produces synthetic rows in train only with full lineage."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    result = execute_smote_augmentation_action(
        ExecuteSmoteAugmentationRequest(
            action_plan_id="action_plan_smote_001",
            step=_smote_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_SMOTE_CONFIG_HASH,
            random_seed=42,
            k_neighbors=5,
            sampling_strategy=0.20,
            generated_at=_SMOTE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert isinstance(report, SyntheticDatasetReport)
    assert report.method is SyntheticGenerationMethod.SMOTE
    assert report.formula == SMOTE_FORMULA
    assert report.source_split == DataSplit.TRAIN.value
    assert report.random_seed == 42
    assert report.sampling_strategy == 0.20
    assert report.real_train_rare_count >= 1
    assert report.generated_count > 0
    # Per-class stats: only the rare class is augmented.
    assert len(report.class_stats) == 1
    rare_stats = report.class_stats[0]
    assert rare_stats.label == "1"
    assert rare_stats.generated_count == report.generated_count

    # Sample lineage carries all required fields and the SMOTE formula.
    assert len(report.sample_lineage) > 0
    for entry in report.sample_lineage:
        assert entry.formula == SMOTE_FORMULA
        assert entry.rare_class_label == "1"
        assert entry.lambda_value is not None
        assert 0.0 <= entry.lambda_value <= 1.0
        assert entry.synthetic_object_id.startswith("txn_synth_")
    assert report.full_sample_lineage_count == report.generated_count

    # Step 2: synthetic samples are only in train. Read the candidate
    # CSV and assert is_synthetic=1 rows live in train and have the rare
    # class label.
    candidate = storage.get(result.candidate_artifact.uri)
    candidate_rows = _read_csv(candidate.data)
    synthetic_rows = [row for row in candidate_rows if row.get("is_synthetic") == "1"]
    assert len(synthetic_rows) == report.generated_count
    for row in synthetic_rows:
        assert row["synthetic_source_split"] == DataSplit.TRAIN.value
        assert row["is_fraud"] == "1"
        assert row["object_id"].startswith("txn_synth_")
    # Real rows are present and unchanged in count.
    real_rows = [row for row in candidate_rows if row.get("is_synthetic") != "1"]
    assert len(real_rows) == report.real_total_count

    # Step 4: lineage refs are recorded both in the report and the
    # augmented split manifest. The augmented split manifest assigns
    # synthetic rows to TRAIN and to the rare class.
    augmented_artifact = result.augmented_split_artifact
    assert augmented_artifact.artifact_kind == SPLIT_MANIFEST_KIND
    assert augmented_artifact.schema_version == SPLIT_MANIFEST_SCHEMA_VERSION
    augmented_payload = storage.get(augmented_artifact.uri).data
    augmented_manifest = _parse_split_manifest(augmented_payload)
    synthetic_assignments = [
        a
        for a in augmented_manifest["assignments"]
        if a["object_id"].startswith("txn_synth_")
    ]
    assert len(synthetic_assignments) == report.generated_count
    for assignment in synthetic_assignments:
        assert assignment["split"] == DataSplit.TRAIN.value
        assert assignment["label"] == "1"

    # Synthetic report artifact metadata.
    assert result.report_artifact.artifact_kind == SYNTHETIC_REPORT_KIND
    assert result.report_artifact.schema_version == SYNTHETIC_REPORT_SCHEMA_VERSION
    stored_report = storage.get(result.report_artifact.uri)
    assert stored_report.info.metadata["synthetic-method"] == "smote"
    assert stored_report.info.metadata["random-seed"] == "42"
    assert stored_report.info.metadata["sampling-strategy"] == "0.2"
    assert stored_report.info.metadata["generated-count"] == str(
        report.generated_count
    )

    validate_contract_payload(
        load_contract_pack(),
        "synthetic_dataset_report",
        report.model_dump(mode="json"),
    )


def test_smote_validation_test_rows_never_used_as_synthetic_sources(tmp_path: Path) -> None:
    """Step 2: SMOTE never picks validation/test object_ids as seeds or neighbors."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    train_object_ids = {
        a.object_id for a in split.manifest.assignments if a.split is DataSplit.TRAIN
    }
    val_test_object_ids = {
        a.object_id
        for a in split.manifest.assignments
        if a.split in (DataSplit.VALIDATION, DataSplit.TEST)
    }

    result = execute_smote_augmentation_action(
        ExecuteSmoteAugmentationRequest(
            action_plan_id="action_plan_smote_001",
            step=_smote_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_SMOTE_CONFIG_HASH,
            random_seed=7,
            k_neighbors=3,
            sampling_strategy=0.25,
            generated_at=_SMOTE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    seed_ids = {entry.seed_object_id for entry in result.report.sample_lineage}
    neighbor_ids = {entry.neighbor_object_id for entry in result.report.sample_lineage}
    used_real_ids = seed_ids | neighbor_ids
    assert used_real_ids.issubset(train_object_ids)
    assert used_real_ids.isdisjoint(val_test_object_ids)


def test_smote_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    """Step 3: same random_seed produces identical synthetic rows and report."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    request = ExecuteSmoteAugmentationRequest(
        action_plan_id="action_plan_smote_001",
        step=_smote_step(),
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        source_artifact=source_artifact,
        split_manifest=split.manifest,
        split_manifest_artifact=split.split_artifact.artifact_ref,
        created_by_job_id="compute_run_apply_001",
        config_hash=_SMOTE_CONFIG_HASH,
        random_seed=42,
        k_neighbors=5,
        sampling_strategy=0.18,
        generated_at=_SMOTE_AT,
        report_id="synthetic_dataset_report_fixed",
    )
    first = execute_smote_augmentation_action(request, storage=storage, registry=registry)
    second = execute_smote_augmentation_action(request, storage=storage, registry=registry)

    # Same seed + same inputs => identical lineage and identical
    # candidate artifact hashes (idempotent registry).
    assert first.report.sample_lineage == second.report.sample_lineage
    assert first.report.generated_count == second.report.generated_count
    assert first.candidate_artifact.hash == second.candidate_artifact.hash
    assert first.augmented_split_artifact.hash == second.augmented_split_artifact.hash
    # The report payload itself embeds upstream ArtifactRef.lineage.created_at,
    # which is set from the fresh-put wall clock on the first run. Asserting
    # logical lineage equality is the deterministic check; binary hash
    # equality of the report artifact would only hold if the registry is
    # completely re-seeded with the same wall clock.
    first_lineage_records = [
        (
            entry.synthetic_object_id,
            entry.seed_object_id,
            entry.neighbor_object_id,
            entry.lambda_value,
        )
        for entry in first.report.sample_lineage
    ]
    second_lineage_records = [
        (
            entry.synthetic_object_id,
            entry.seed_object_id,
            entry.neighbor_object_id,
            entry.lambda_value,
        )
        for entry in second.report.sample_lineage
    ]
    assert first_lineage_records == second_lineage_records


def test_smote_rejects_non_train_source_split(tmp_path: Path) -> None:
    """SMOTE step config that targets validation must be rejected."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    bad_step = _smote_step(source_split="validation")
    with pytest.raises(SmoteExecutionError) as excinfo:
        execute_smote_augmentation_action(
            ExecuteSmoteAugmentationRequest(
                action_plan_id="action_plan_smote_001",
                step=bad_step,
                dataset_id="dataset_1",
                source_dataset_version_id="dataset_version_1",
                candidate_dataset_version_id="dataset_version_2_candidate",
                source_artifact=source_artifact,
                split_manifest=split.manifest,
                split_manifest_artifact=split.split_artifact.artifact_ref,
                created_by_job_id="compute_run_apply_001",
                config_hash=_SMOTE_CONFIG_HASH,
                generated_at=_SMOTE_AT,
            ),
            storage=storage,
            registry=registry,
        )
    assert excinfo.value.reason_code == "non_train_source_split"


def test_smote_rejects_unsupported_method(tmp_path: Path) -> None:
    """A non-SMOTE method id must be rejected before any execution."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    bad_step = _smote_step(method_id="ctgan")
    with pytest.raises(SmoteExecutionError) as excinfo:
        execute_smote_augmentation_action(
            ExecuteSmoteAugmentationRequest(
                action_plan_id="action_plan_smote_001",
                step=bad_step,
                dataset_id="dataset_1",
                source_dataset_version_id="dataset_version_1",
                candidate_dataset_version_id="dataset_version_2_candidate",
                source_artifact=source_artifact,
                split_manifest=split.manifest,
                split_manifest_artifact=split.split_artifact.artifact_ref,
                created_by_job_id="compute_run_apply_001",
                config_hash=_SMOTE_CONFIG_HASH,
                generated_at=_SMOTE_AT,
            ),
            storage=storage,
            registry=registry,
        )
    assert excinfo.value.reason_code == "unsupported_synthetic_method"


def test_smote_records_formula_and_neighbor_lineage(tmp_path: Path) -> None:
    """Step 4: every sample lineage entry carries seed/neighbor and the formula."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    result = execute_smote_augmentation_action(
        ExecuteSmoteAugmentationRequest(
            action_plan_id="action_plan_smote_001",
            step=_smote_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_SMOTE_CONFIG_HASH,
            random_seed=11,
            k_neighbors=2,
            sampling_strategy=0.12,
            generated_at=_SMOTE_AT,
        ),
        storage=storage,
        registry=registry,
    )

    assert result.report.full_sample_lineage_count == result.report.generated_count
    assert result.report.formula == SMOTE_FORMULA
    for entry in result.report.sample_lineage:
        assert entry.seed_object_id != entry.synthetic_object_id
        assert entry.neighbor_object_id != entry.synthetic_object_id
        assert entry.formula == SMOTE_FORMULA


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _smote_step(
    *,
    method_id: str = "smote",
    source_split: str = "train",
) -> ActionPlanStep:
    return ActionPlanStep(
        step_id="augment_rare_class_smote",
        type="AUGMENT_RARE_CLASS",
        depends_on=("create_split_tabular",),
        idempotency_key="sha256:" + "1" * 64,
        method_id=method_id,
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="synthetic_policy_v0",
        validation_gates=(
            "split_leakage_check",
            "schema_validation",
            "business_rules",
            "synthetic_dcr_check",
        ),
        preconditions=(
            "source_version_is_immutable",
            "train_split_exists",
            "leakage_checks_passed",
        ),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind="CANDIDATE_DATASET_VERSION",
        config={
            "target_column": "is_fraud",
            "rare_class_label": "1",
            "method": method_id,
            "source_split": source_split,
        },
        random_seed=42,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
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
        validation_gates=("schema_validation", "split_policy_check"),
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
            max_attempts=2,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )
    return ExecuteTabularSplitRequest(
        action_plan_id="action_plan_smote_001",
        step=split_step,
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        source_artifact=source_artifact,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        target_column="is_fraud",
        seed=42,
        generated_at=_CREATED_AT,
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


def _parse_split_manifest(data: bytes) -> dict[str, Any]:
    import json

    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("split manifest must be a JSON object")
    return payload


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
