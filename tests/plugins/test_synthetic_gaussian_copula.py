"""Tests for TASK-047 Gaussian Copula synthetic generation."""

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
    GAUSSIAN_COPULA_FORMULA,
    SMOTE_FORMULA,
    ActionPlanStep,
    ArtifactRef,
    DataSplit,
    ErrorCode,
    RetryPolicy,
    SyntheticAugmentationKind,
    SyntheticDatasetReport,
    SyntheticGenerationMethod,
    SyntheticValidationCheckStatus,
)
from app.ingestion import open_archive_path
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    SYNTHETIC_REPORT_KIND,
    SYNTHETIC_REPORT_SCHEMA_VERSION,
    ExecuteGaussianCopulaRequest,
    ExecuteSmoteAugmentationRequest,
    ExecuteTabularSplitRequest,
    GaussianCopulaPolicy,
    GaussianCopulaPolicyError,
    execute_gaussian_copula_action,
    execute_smote_augmentation_action,
    execute_tabular_split_action,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "e" * 64
_GC_CONFIG_HASH = "sha256:" + "f" * 64
_CREATED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
_GC_AT = datetime(2026, 5, 24, 12, 30, tzinfo=UTC)


def test_gaussian_copula_with_enabled_policy_passes_validation_gates(tmp_path: Path) -> None:
    """Steps 1-2: enable Gaussian Copula and run; report passes validation."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    result = execute_gaussian_copula_action(
        ExecuteGaussianCopulaRequest(
            action_plan_id="action_plan_gc_001",
            step=_gc_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_GC_CONFIG_HASH,
            policy=_enabled_policy(),
            random_seed=7,
            sampling_strategy=0.10,
            generated_at=_GC_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert isinstance(report, SyntheticDatasetReport)
    assert report.method is SyntheticGenerationMethod.GAUSSIAN_COPULA
    assert report.augmentation_kind is SyntheticAugmentationKind.DISTRIBUTION_LEVEL
    assert report.formula == GAUSSIAN_COPULA_FORMULA
    assert report.source_split == DataSplit.TRAIN.value
    assert report.k_neighbors == 0
    assert report.random_seed == 7
    assert report.sampling_strategy == 0.10
    assert report.generated_count > 0
    # Gaussian Copula does not target a single rare class label.
    assert report.rare_class_label is None

    # synthetic_validation block must be populated.
    validation = report.synthetic_validation
    assert validation is not None
    assert validation.overall_passed is True
    assert validation.blocker_present is False
    statuses = {check.check: check.status for check in validation.checks}
    assert statuses["schema_match"] is SyntheticValidationCheckStatus.PASSED
    assert statuses["privacy_exact_duplicate_to_real"] is (
        SyntheticValidationCheckStatus.PASSED
    )

    # Gaussian Copula artifacts block carries the required transformation
    # metadata. SMOTE-only fields stay None on the report.
    assert report.gaussian_copula_artifacts is not None
    gc = report.gaussian_copula_artifacts
    assert len(gc.column_distribution_transforms) == len(report.feature_columns)
    assert gc.correlation_matrix_size == len(report.feature_columns)
    assert (
        len(gc.latent_normal_cholesky_row_major)
        == gc.correlation_matrix_size * gc.correlation_matrix_size
    )
    assert "uniform_to_data" in gc.inverse_transform_metadata
    assert "normal_to_uniform" in gc.inverse_transform_metadata
    for transform in gc.column_distribution_transforms:
        assert transform.method == "empirical_cdf"
        assert transform.sample_count > 0
        assert len(transform.quantile_levels) == len(transform.quantile_values)
        assert transform.quantile_levels[0] == 0.0
        assert transform.quantile_levels[-1] == 1.0

    # Sample lineage carries latent_draw_index, no seed/neighbor refs.
    assert len(report.sample_lineage) > 0
    for entry in report.sample_lineage:
        assert entry.method is SyntheticGenerationMethod.GAUSSIAN_COPULA
        assert entry.formula == GAUSSIAN_COPULA_FORMULA
        assert entry.latent_draw_index is not None
        assert entry.seed_object_id is None
        assert entry.neighbor_object_id is None
        assert entry.lambda_value is None

    # Candidate dataset has synthetic rows marked is_synthetic=1 and
    # only in train.
    candidate = storage.get(result.candidate_artifact.uri)
    candidate_rows = _read_csv(candidate.data)
    synthetic_rows = [row for row in candidate_rows if row.get("is_synthetic") == "1"]
    assert len(synthetic_rows) == report.generated_count
    for row in synthetic_rows:
        assert row["synthetic_source_split"] == DataSplit.TRAIN.value
        assert row["object_id"].startswith("txn_synth_gc_")

    # Augmented split manifest assigns synthetic rows to TRAIN.
    augmented_payload = storage.get(result.augmented_split_artifact.uri).data
    augmented_manifest = _parse_split_manifest(augmented_payload)
    synthetic_assignments = [
        a
        for a in augmented_manifest["assignments"]
        if a["object_id"].startswith("txn_synth_gc_")
    ]
    assert len(synthetic_assignments) == report.generated_count
    for assignment in synthetic_assignments:
        assert assignment["split"] == DataSplit.TRAIN.value

    assert result.report_artifact.artifact_kind == SYNTHETIC_REPORT_KIND
    assert result.report_artifact.schema_version == SYNTHETIC_REPORT_SCHEMA_VERSION
    stored_report = storage.get(result.report_artifact.uri)
    assert stored_report.info.metadata["synthetic-method"] == "gaussian_copula"
    assert stored_report.info.metadata["augmentation-kind"] == "distribution_level"

    validate_contract_payload(
        load_contract_pack(),
        "synthetic_dataset_report",
        report.model_dump(mode="json"),
    )


def test_gaussian_copula_disabled_by_policy_returns_disabled_by_policy(tmp_path: Path) -> None:
    """Step 3: disabling the policy returns disabled_by_policy without running."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    with pytest.raises(GaussianCopulaPolicyError) as excinfo:
        execute_gaussian_copula_action(
            ExecuteGaussianCopulaRequest(
                action_plan_id="action_plan_gc_001",
                step=_gc_step(),
                dataset_id="dataset_1",
                source_dataset_version_id="dataset_version_1",
                candidate_dataset_version_id="dataset_version_2_candidate",
                source_artifact=source_artifact,
                split_manifest=split.manifest,
                split_manifest_artifact=split.split_artifact.artifact_ref,
                created_by_job_id="compute_run_apply_001",
                config_hash=_GC_CONFIG_HASH,
                policy=_disabled_policy(),
                generated_at=_GC_AT,
            ),
            storage=storage,
            registry=registry,
        )
    error = excinfo.value
    assert error.reason_code == "disabled_by_policy"
    assert error.code is ErrorCode.POLICY_BLOCKED
    assert error.details["policy_status"] == "disabled_by_policy"
    assert error.details["policy_version"] == "synthetic_policy_v0"


def test_gaussian_copula_metadata_distinguishes_from_smote(tmp_path: Path) -> None:
    """Step 4: GC report differs from SMOTE report on method/kind/formula/lineage shape."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    smote_result = execute_smote_augmentation_action(
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
            config_hash=_GC_CONFIG_HASH,
            random_seed=42,
            k_neighbors=3,
            sampling_strategy=0.20,
            generated_at=_GC_AT,
        ),
        storage=storage,
        registry=registry,
    )
    gc_result = execute_gaussian_copula_action(
        ExecuteGaussianCopulaRequest(
            action_plan_id="action_plan_gc_001",
            step=_gc_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_GC_CONFIG_HASH,
            policy=_enabled_policy(),
            random_seed=7,
            sampling_strategy=0.10,
            generated_at=_GC_AT,
        ),
        storage=storage,
        registry=registry,
    )

    assert smote_result.report.method is SyntheticGenerationMethod.SMOTE
    assert smote_result.report.augmentation_kind is (
        SyntheticAugmentationKind.TARGETED_RARE_CLASS
    )
    assert smote_result.report.formula == SMOTE_FORMULA
    assert smote_result.report.gaussian_copula_artifacts is None
    # SMOTE per-sample lineage records seed and neighbor.
    assert all(
        entry.seed_object_id is not None and entry.neighbor_object_id is not None
        for entry in smote_result.report.sample_lineage
    )

    assert gc_result.report.method is SyntheticGenerationMethod.GAUSSIAN_COPULA
    assert gc_result.report.augmentation_kind is (
        SyntheticAugmentationKind.DISTRIBUTION_LEVEL
    )
    assert gc_result.report.formula == GAUSSIAN_COPULA_FORMULA
    assert gc_result.report.gaussian_copula_artifacts is not None
    # GC per-sample lineage uses latent_draw_index instead of seed/neighbor.
    assert all(
        entry.seed_object_id is None
        and entry.neighbor_object_id is None
        and entry.latent_draw_index is not None
        for entry in gc_result.report.sample_lineage
    )


def test_gaussian_copula_validation_train_only_and_synthetic_marker(tmp_path: Path) -> None:
    """Synthetic GC rows live exclusively in train and never reuse real validation/test ids."""
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

    result = execute_gaussian_copula_action(
        ExecuteGaussianCopulaRequest(
            action_plan_id="action_plan_gc_001",
            step=_gc_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_GC_CONFIG_HASH,
            policy=_enabled_policy(),
            random_seed=11,
            sampling_strategy=0.05,
            generated_at=_GC_AT,
        ),
        storage=storage,
        registry=registry,
    )

    augmented_payload = storage.get(result.augmented_split_artifact.uri).data
    augmented_manifest = _parse_split_manifest(augmented_payload)
    real_assignments = {
        a["object_id"]: a["split"]
        for a in augmented_manifest["assignments"]
        if not a["object_id"].startswith("txn_synth_gc_")
    }
    # Real assignments preserved exactly.
    assert real_assignments == {
        a.object_id: a.split.value for a in split.manifest.assignments
    }
    # Synthetic rows carry train assignment only.
    synthetic_splits = {
        a["split"]
        for a in augmented_manifest["assignments"]
        if a["object_id"].startswith("txn_synth_gc_")
    }
    assert synthetic_splits == {DataSplit.TRAIN.value}
    # The generator never invents IDs that collide with real IDs.
    synthetic_ids = {
        a["object_id"]
        for a in augmented_manifest["assignments"]
        if a["object_id"].startswith("txn_synth_gc_")
    }
    assert synthetic_ids.isdisjoint(train_object_ids)
    assert synthetic_ids.isdisjoint(val_test_object_ids)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _enabled_policy() -> GaussianCopulaPolicy:
    return GaussianCopulaPolicy(
        enabled=True,
        readiness_status="available",
        policy_status="enabled",
        policy_version="synthetic_policy_v0",
        profile="demo_strict",
    )


def _disabled_policy() -> GaussianCopulaPolicy:
    return GaussianCopulaPolicy(
        enabled=False,
        readiness_status="available",
        policy_status="disabled_by_policy",
        policy_version="synthetic_policy_v0",
        profile="banking_strict",
    )


def _gc_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="augment_rare_class_gaussian_copula",
        type="AUGMENT_RARE_CLASS",
        depends_on=("create_split_tabular",),
        idempotency_key="sha256:" + "1" * 64,
        method_id="gaussian_copula",
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
            "method": "gaussian_copula",
            "source_split": "train",
        },
        random_seed=7,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _smote_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="augment_rare_class_smote",
        type="AUGMENT_RARE_CLASS",
        depends_on=("create_split_tabular",),
        idempotency_key="sha256:" + "3" * 64,
        method_id="smote",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "4" * 64,
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
            "method": "smote",
            "source_split": "train",
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
        idempotency_key="sha256:" + "5" * 64,
        method_id="group_stratified",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "6" * 64,
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
        action_plan_id="action_plan_gc_001",
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
