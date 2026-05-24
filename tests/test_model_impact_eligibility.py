"""Tests for TASK-050 model-impact eligibility checker."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    ClassCount,
    ClassImbalanceDiagnostics,
    ColumnMissingness,
    ErrorCode,
    LeakageCheckResult,
    LeakageCheckSeverity,
    LeakageCheckStatus,
    LeakageCheckType,
    MissingnessDiagnostics,
    ModelImpactEligibilityStatus,
    ModelImpactInputName,
    ModelImpactNotEligibleReasonCode,
    ModelImpactTaskType,
    RetryPolicy,
    SplitLeakageLineage,
    SplitLeakageReport,
    SplitManifest,
    TabularProfileReport,
)
from app.ingestion import open_archive_path
from app.kernel import (
    CheckModelImpactEligibilityRequest,
    check_model_impact_eligibility,
)
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    ExecuteTabularSplitRequest,
    execute_tabular_split_action,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "a" * 64
_GENERATED_AT = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step 1: demo dataset is eligible
# ---------------------------------------------------------------------------


def test_demo_dataset_is_eligible(tmp_path: Path) -> None:
    """Step 1: demo profile + valid split + no leakage → eligible report."""
    storage, registry = _storage_and_registry()
    profile = _demo_tabular_profile()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    leakage_report = _no_leakage_report(
        manifest=split.manifest,
        source_artifact=source_artifact,
        split_manifest_artifact=split.split_artifact.artifact_ref,
    )

    request = CheckModelImpactEligibilityRequest(
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        organization_id="org_1",
        project_id="project_1",
        task_type=ModelImpactTaskType.SUPERVISED_TABULAR_CLASSIFICATION,
        tabular_profile=profile,
        split_manifest=split.manifest,
        split_manifest_artifact=split.split_artifact.artifact_ref,
        split_leakage_report=leakage_report,
        minimum_labeled_samples=50,
        minimum_rare_class_samples=2,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        report_id="model_impact_eligibility_demo",
        generated_at=_GENERATED_AT,
    )
    result = check_model_impact_eligibility(request, registry=registry)
    report = result.report

    assert report.status is ModelImpactEligibilityStatus.ELIGIBLE
    assert report.eligible is True
    assert report.task_type is ModelImpactTaskType.SUPERVISED_TABULAR_CLASSIFICATION
    assert report.target_column == "is_fraud"
    assert report.rare_class_label == "1"
    assert report.primary_reason_code == "supervised_classification_with_target"
    expected_present = {
        ModelImpactInputName.TARGET_LABEL,
        ModelImpactInputName.ENOUGH_LABELED_SAMPLES,
        ModelImpactInputName.VALID_SPLIT_STRATEGY,
        ModelImpactInputName.NO_LEAKAGE_BLOCKERS,
        ModelImpactInputName.BASELINE_MODEL_AVAILABLE,
    }
    assert set(report.required_inputs_present) == expected_present
    assert report.required_inputs_missing == ()
    assert report.not_eligible_reason_codes == ()
    assert report.fallback_report is None
    assert report.cohort_stats  # cohort stats from class imbalance
    assert any(c.is_rare_class for c in report.cohort_stats)
    assert report.split_stats  # split stats from manifest
    assert {s.split for s in report.split_stats} == {"train", "validation", "test"}
    assert sum(s.total_count for s in report.split_stats) == sum(
        c.count for c in report.cohort_stats
    )

    # Persistence: artifact metadata signals eligibility for downstream stages.
    stored = storage.get(result.report_artifact.uri)
    assert stored.info.metadata["eligibility-status"] == "eligible"
    assert stored.info.metadata["eligible"] == "true"
    validate_contract_payload(
        load_contract_pack(),
        "model_impact_eligibility",
        report.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Step 2: missing target_label -> not eligible
# ---------------------------------------------------------------------------


def test_missing_target_label_is_not_eligible(tmp_path: Path) -> None:
    """Step 2: stripping the target column flips the candidate to not_eligible."""
    storage, registry = _storage_and_registry()
    profile = _demo_tabular_profile()
    profile_no_target = profile.model_copy(
        update={
            "target_column": None,
            "missingness": (
                profile.missingness.model_copy(
                    update={
                        "target_column": None,
                        "target_column_missing": False,
                        "missing_target_count": 0,
                    }
                )
                if profile.missingness is not None
                else None
            ),
            "class_imbalance": None,
        }
    )

    # When target is absent, the split manifest is still useful to
    # surface other issues but cannot rescue the target label. Use the
    # demo split data and verify the target_label_missing reason code
    # wins over downstream issues.
    request = CheckModelImpactEligibilityRequest(
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        tabular_profile=profile_no_target,
        split_manifest=None,
        split_manifest_artifact=None,
        split_leakage_report=None,
        minimum_labeled_samples=50,
        minimum_rare_class_samples=2,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
    )
    result = check_model_impact_eligibility(request, registry=registry)
    report = result.report

    assert report.eligible is False
    assert report.status is ModelImpactEligibilityStatus.NOT_ELIGIBLE
    assert (
        ModelImpactNotEligibleReasonCode.TARGET_LABEL_MISSING
        in report.not_eligible_reason_codes
    )
    assert ModelImpactInputName.TARGET_LABEL in report.required_inputs_missing
    assert ModelImpactInputName.TARGET_LABEL not in report.required_inputs_present
    assert "target_label_missing" in report.reasons
    assert report.primary_reason_code == "target_label_missing"
    assert (
        ErrorCode.MODEL_IMPACT_NOT_ELIGIBLE.value == "MODEL_IMPACT_NOT_ELIGIBLE"
    )


# ---------------------------------------------------------------------------
# Step 3: leakage blocker -> not eligible
# ---------------------------------------------------------------------------


def test_leakage_blocker_makes_candidate_not_eligible(tmp_path: Path) -> None:
    """Step 3: SplitLeakageReport with block_training=True → not eligible."""
    storage, registry = _storage_and_registry()
    profile = _demo_tabular_profile()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    leakage_report = _leakage_blocker_report(
        manifest=split.manifest,
        source_artifact=source_artifact,
        split_manifest_artifact=split.split_artifact.artifact_ref,
    )
    request = CheckModelImpactEligibilityRequest(
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        tabular_profile=profile,
        split_manifest=split.manifest,
        split_manifest_artifact=split.split_artifact.artifact_ref,
        split_leakage_report=leakage_report,
        minimum_labeled_samples=50,
        minimum_rare_class_samples=2,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
    )
    result = check_model_impact_eligibility(request, registry=registry)
    report = result.report

    assert report.eligible is False
    assert (
        ModelImpactNotEligibleReasonCode.LEAKAGE_BLOCKER_PRESENT
        in report.not_eligible_reason_codes
    )
    assert ModelImpactInputName.NO_LEAKAGE_BLOCKERS in report.required_inputs_missing
    assert "leakage_blocker_present" in report.reasons


def test_too_few_rare_class_samples_is_not_eligible() -> None:
    """Below the rare-class minimum the candidate is not eligible."""
    _, registry = _storage_and_registry()
    profile = _demo_tabular_profile().model_copy(
        update={
            "class_imbalance": ClassImbalanceDiagnostics(
                target_column="is_fraud",
                total_samples=200,
                class_counts=(
                    ClassCount(label="0", count=199),
                    ClassCount(label="1", count=1),
                ),
                rare_class_label="1",
                rare_class_count=1,
                rare_class_ratio=0.005,
                minority_class_label="1",
                minority_class_share=0.005,
                imbalance_ratio=199.0,
                balance_score=0.18,
                balance_score_alternative=0.01,
                effective_number_beta=0.999,
                effective_number_of_samples={"0": 200.0, "1": 1.0},
            )
        }
    )
    request = CheckModelImpactEligibilityRequest(
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        tabular_profile=profile,
        split_manifest=None,
        minimum_labeled_samples=50,
        minimum_rare_class_samples=5,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
    )
    result = check_model_impact_eligibility(request, registry=registry)
    report = result.report

    assert report.eligible is False
    assert (
        ModelImpactNotEligibleReasonCode.NOT_ENOUGH_RARE_CLASS_SAMPLES
        in report.not_eligible_reason_codes
    )
    assert (
        ModelImpactNotEligibleReasonCode.SPLIT_MANIFEST_MISSING
        in report.not_eligible_reason_codes
    )
    assert ModelImpactInputName.ENOUGH_LABELED_SAMPLES in report.required_inputs_missing
    assert ModelImpactInputName.VALID_SPLIT_STRATEGY in report.required_inputs_missing


def test_target_column_with_missing_values_is_not_eligible() -> None:
    """target_column_missing=True flags the candidate."""
    _, registry = _storage_and_registry()
    profile = _demo_tabular_profile()
    missingness = profile.missingness
    assert missingness is not None
    profile_with_missing_target = profile.model_copy(
        update={
            "missingness": missingness.model_copy(
                update={
                    "target_column": "is_fraud",
                    "target_column_missing": True,
                    "missing_target_count": 5,
                }
            ),
        }
    )
    request = CheckModelImpactEligibilityRequest(
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        tabular_profile=profile_with_missing_target,
        split_manifest=None,
        minimum_labeled_samples=50,
        minimum_rare_class_samples=2,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
    )
    result = check_model_impact_eligibility(request, registry=registry)
    report = result.report

    assert report.eligible is False
    assert (
        ModelImpactNotEligibleReasonCode.TARGET_COLUMN_HAS_MISSING_VALUES
        in report.not_eligible_reason_codes
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        e for e in pack.examples if e.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


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
            max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )
    return ExecuteTabularSplitRequest(
        action_plan_id="action_plan_eligibility_001",
        step=split_step,
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=source_artifact,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        target_column="is_fraud",
        seed=42,
        generated_at=_GENERATED_AT,
    )


def _no_leakage_report(
    *,
    manifest: SplitManifest,
    source_artifact: ArtifactRef,
    split_manifest_artifact: ArtifactRef,
) -> SplitLeakageReport:
    return SplitLeakageReport(
        report_id="split_leakage_report_clean_001",
        report_schema_version="split_leakage_report.v1",
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        split_manifest_id=manifest.split_manifest_id,
        target_column=manifest.target_column,
        policy_version="split_leakage_policy_v0",
        leakage_detected=False,
        leakage_risk_score=0.0,
        block_model_evaluation=False,
        block_training=False,
        checks=(
            LeakageCheckResult(
                check_type=LeakageCheckType.EXACT_HASH,
                status=LeakageCheckStatus.PASSED,
                severity=LeakageCheckSeverity.INFO,
                reason_code="no_exact_hash_leakage",
                findings_count=0,
            ),
        ),
        lineage=SplitLeakageLineage(
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            source_artifact=source_artifact,
            split_manifest=split_manifest_artifact,
        ),
        generated_at=_GENERATED_AT,
    )


def _leakage_blocker_report(
    *,
    manifest: SplitManifest,
    source_artifact: ArtifactRef,
    split_manifest_artifact: ArtifactRef,
) -> SplitLeakageReport:
    return SplitLeakageReport(
        report_id="split_leakage_report_blocked_001",
        report_schema_version="split_leakage_report.v1",
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        split_manifest_id=manifest.split_manifest_id,
        target_column=manifest.target_column,
        policy_version="split_leakage_policy_v0",
        leakage_detected=True,
        leakage_risk_score=1.0,
        block_model_evaluation=True,
        block_training=True,
        checks=(
            LeakageCheckResult(
                check_type=LeakageCheckType.TARGET_LEAKAGE_CANDIDATE,
                status=LeakageCheckStatus.FAILED,
                severity=LeakageCheckSeverity.BLOCKER,
                reason_code="target_leakage_candidate",
                block_action="BLOCK_TRAINING",
                findings_count=1,
                affected_columns=("manual_review_flag",),
            ),
        ),
        lineage=SplitLeakageLineage(
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            source_artifact=source_artifact,
            split_manifest=split_manifest_artifact,
        ),
        generated_at=_GENERATED_AT,
    )


def _source_transactions_artifact(tmp_path: Path, registry: ArtifactRegistry) -> ArtifactRef:
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
        created_by_job_id="compute_run_apply_001",
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


# silence unused import warning when the helper imports below shrink
_ColumnMissingness = ColumnMissingness
_MissingnessDiagnostics = MissingnessDiagnostics


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
                message="Object does not exist",
            ) from exc
