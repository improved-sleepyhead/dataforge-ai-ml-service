"""Tests for TASK-049 proposed candidate dataset version artifacts."""

from __future__ import annotations

import csv
import io
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
from app.domain import (
    ActionPlan,
    ActionPlanStep,
    ArtifactRef,
    BusinessRuleSeverity,
    CandidateDatasetVersion,
    CandidatePolicyVersions,
    CandidateVersionStatus,
    ErrorCode,
    RetryPolicy,
    SyntheticDatasetReport,
    TabularProfileReport,
    ValidationGateStatus,
    WorkflowType,
)
from app.ingestion import open_archive_path
from app.kernel import (
    CANDIDATE_DATASET_VERSION_KIND,
    BuildActionPlanPreviewRequest,
    BuildCandidateVersionRequest,
    BuildMethodRecommendationsRequest,
    CandidateVersionBuilderError,
    build_action_plan_preview,
    build_candidate_dataset_version,
    build_method_recommendations,
)
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    ExecuteSmoteAugmentationRequest,
    ExecuteTabularImputationRequest,
    ExecuteTabularSplitRequest,
    execute_smote_augmentation_action,
    execute_tabular_imputation_action,
    execute_tabular_split_action,
)
from app.plugins.tabular.rules import BusinessRule, RuleFieldCheck
from app.plugins.validation import (
    DcrThresholds,
    RunValidationGatesRequest,
    run_validation_gates,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "c" * 64
_GATES_CONFIG_HASH = "sha256:" + "d" * 64
_CANDIDATE_CONFIG_HASH = "sha256:" + "f" * 64
_GENERATED_AT = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step 1+2+3: approved imputation ActionPlan -> proposed candidate
# ---------------------------------------------------------------------------


def test_approved_action_plan_emits_proposed_candidate_metadata(tmp_path: Path) -> None:
    """Steps 1-3: imputation ActionPlan -> proposed candidate + audit event."""
    storage, registry = _storage_and_registry()
    platform_client = FakePlatformMetadataClient()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    plan = _imputation_action_plan()
    imputation = execute_tabular_imputation_action(
        ExecuteTabularImputationRequest(
            action_plan_id=plan.action_plan_id,
            step=plan.steps[0],
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            target_column="is_fraud",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    gates_request = RunValidationGatesRequest(
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        candidate_artifact=imputation.candidate_artifact.artifact_ref,
        source_artifact=source_artifact,
        candidate_artifact_kind="candidate_tabular_dataset",
        schema_columns=tuple(_demo_schema_columns(tmp_path)),
        numeric_columns=("amount", "monthly_income"),
        business_rules=(
            BusinessRule(
                rule_id="amount_is_positive",
                severity=BusinessRuleSeverity.WARNING,
                checks=(RuleFieldCheck(field="amount", op="gte", value=0.0),),
            ),
        ),
        created_by_job_id="compute_run_apply_001",
        config_hash=_GATES_CONFIG_HASH,
        action_plan_id=plan.action_plan_id,
        step_id=plan.steps[0].step_id,
        report_id="validation_gates_imputation_001",
        generated_at=_GENERATED_AT,
    )
    gates_result = run_validation_gates(gates_request, storage=storage, registry=registry)
    assert gates_result.report.overall_status is ValidationGateStatus.PASSED

    request = BuildCandidateVersionRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        proposed_version_name="dataset_version_v2_candidate",
        action_plan=plan,
        policy_versions=_policy_versions(),
        decision_report_id="decision_report_001",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CANDIDATE_CONFIG_HASH,
        source_artifacts=(source_artifact,),
        candidate_artifacts=(
            imputation.candidate_artifact.artifact_ref,
            imputation.report_artifact.artifact_ref,
            gates_result.report_artifact.artifact_ref,
        ),
        primary_dataset_artifact=imputation.candidate_artifact.artifact_ref,
        validation_gates_report=gates_result.report,
        validation_gates_report_artifact=gates_result.report_artifact.artifact_ref,
        candidate_version_id="candidate_dataset_version_test_001",
        proposed_at=_GENERATED_AT,
    )
    result = build_candidate_dataset_version(
        request, registry=registry, platform_client=platform_client
    )
    candidate = result.candidate_version

    # Step 1: imputation candidate version is proposed.
    assert isinstance(candidate, CandidateDatasetVersion)
    assert candidate.status is CandidateVersionStatus.PROPOSED
    assert candidate.block_export is False
    assert candidate.block_model_evaluation is False
    assert candidate.block_training is False
    assert candidate.blocker_reason_codes == ()

    # Step 2: candidate metadata carries parent, action_plan, policies,
    # input/output hashes, and references the validation gates report.
    lineage = candidate.lineage
    assert lineage.parent_version_id == "dataset_version_v1"
    assert lineage.proposed_version_name == "dataset_version_v2_candidate"
    assert lineage.action_plan_id == plan.action_plan_id
    assert lineage.decision_report_id == "decision_report_001"
    assert source_artifact.hash in lineage.input_artifact_hashes
    assert imputation.candidate_artifact.hash in lineage.output_artifact_hashes
    assert gates_result.report_artifact.hash in lineage.output_artifact_hashes
    assert candidate.validation_gates_report == gates_result.report_artifact.artifact_ref
    assert candidate.policy_versions.profile_policy_version == "demo_strict_v1"
    step_summary = candidate.action_plan_steps[0]
    assert step_summary.step_id == plan.steps[0].step_id
    assert step_summary.method_id == "group_median"
    assert step_summary.plugin_id == "dataforge.tabular"

    # The candidate metadata artifact is persisted as JSON in storage
    # and validates against the contract schema.
    assert result.candidate_version_artifact.artifact_kind == CANDIDATE_DATASET_VERSION_KIND
    stored = storage.get(result.candidate_version_artifact.uri)
    assert stored.info.metadata["candidate-status"] == "proposed"
    validate_contract_payload(
        load_contract_pack(),
        "candidate_dataset_version",
        candidate.model_dump(mode="json"),
    )

    # Step 3: fake platform receives a DATASET_VERSION_PROPOSED audit
    # event with safe metadata only (no raw rows / PII).
    snapshot = platform_client.snapshot()
    assert len(snapshot.audit_events) == 1
    event = snapshot.audit_events[0]
    assert event.event_type is AuditEventType.DATASET_VERSION_PROPOSED
    assert event.organization_id == "org_1"
    assert event.project_id == "project_1"
    assert event.metadata["candidate_version_id"] == candidate.candidate_version_id
    assert event.metadata["candidate_status"] == "proposed"
    assert event.metadata["candidate_version_artifact_uri"] == (
        result.candidate_version_artifact.uri
    )
    assert event.metadata["action_plan_id"] == plan.action_plan_id
    assert event.metadata["block_export"] is False


# ---------------------------------------------------------------------------
# Step 4: synthetic SMOTE candidate metadata completeness
# ---------------------------------------------------------------------------


def test_smote_candidate_carries_complete_synthetic_metadata(tmp_path: Path) -> None:
    """Step 4: SMOTE candidate exposes method/seed/source/cohort/lineage refs."""
    storage, registry = _storage_and_registry()
    platform_client = FakePlatformMetadataClient()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    smote = execute_smote_augmentation_action(
        ExecuteSmoteAugmentationRequest(
            action_plan_id="action_plan_smote_001",
            step=_smote_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_GATES_CONFIG_HASH,
            random_seed=42,
            k_neighbors=3,
            sampling_strategy=0.20,
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )
    gates_result = run_validation_gates(
        RunValidationGatesRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            candidate_artifact=smote.candidate_artifact.artifact_ref,
            source_artifact=source_artifact,
            candidate_artifact_kind="candidate_tabular_dataset",
            schema_columns=tuple(_demo_schema_columns(tmp_path)),
            numeric_columns=("amount", "monthly_income"),
            synthetic_dataset_report=smote.report,
            synthetic_dataset_report_artifact=smote.report_artifact.artifact_ref,
            split_manifest_artifact=smote.augmented_split_artifact.artifact_ref,
            dcr_thresholds=DcrThresholds(),
            created_by_job_id="compute_run_apply_001",
            config_hash=_GATES_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )
    plan = _smote_action_plan()

    request = BuildCandidateVersionRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        proposed_version_name="dataset_version_v2_candidate",
        action_plan=plan,
        policy_versions=_policy_versions(),
        decision_report_id="decision_report_smote",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CANDIDATE_CONFIG_HASH,
        source_artifacts=(source_artifact,),
        candidate_artifacts=(
            smote.candidate_artifact.artifact_ref,
            smote.augmented_split_artifact.artifact_ref,
            smote.report_artifact.artifact_ref,
            gates_result.report_artifact.artifact_ref,
        ),
        primary_dataset_artifact=smote.candidate_artifact.artifact_ref,
        validation_gates_report=gates_result.report,
        validation_gates_report_artifact=gates_result.report_artifact.artifact_ref,
        synthetic_dataset_report=smote.report,
        synthetic_dataset_report_artifact=smote.report_artifact.artifact_ref,
        synthetic_validation_report_artifact=gates_result.report_artifact.artifact_ref,
        candidate_version_id="candidate_dataset_version_smote_001",
        proposed_at=_GENERATED_AT,
    )
    result = build_candidate_dataset_version(
        request, registry=registry, platform_client=platform_client
    )
    candidate = result.candidate_version

    assert candidate.status is CandidateVersionStatus.PROPOSED
    assert candidate.synthetic_metadata is not None
    metadata = candidate.synthetic_metadata
    assert metadata.method_id == "smote"
    assert metadata.plugin_id == "dataforge.tabular"
    assert metadata.plugin_version == smote.report.method_version
    assert metadata.random_seed == 42
    assert metadata.config_hash == _CANDIDATE_CONFIG_HASH
    assert metadata.source_split == "train"
    assert metadata.source_cohort == "1"  # rare class label is "1"
    assert metadata.generated_count == smote.report.generated_count
    assert metadata.sampling_strategy == 0.20
    assert metadata.policy_version == _policy_versions().method_policy_version
    assert metadata.synthetic_dataset_report == smote.report_artifact.artifact_ref
    assert metadata.validation_report == gates_result.report_artifact.artifact_ref
    assert metadata.model_impact_report is None  # MVP: model impact lands in TASK-050
    # Source object ids must reference real train rare-class objects.
    assert metadata.full_source_object_ids_count >= 1
    assert all(
        object_id and not object_id.startswith("txn_synth_")
        for object_id in metadata.source_object_ids
    )

    # The metadata artifact is persisted and validates against the
    # contract schema even when synthetic_metadata is populated.
    validate_contract_payload(
        load_contract_pack(),
        "candidate_dataset_version",
        candidate.model_dump(mode="json"),
    )

    # The platform receives the audit event with the synthetic-method
    # markers it needs to render the proposal in the UI.
    event = platform_client.snapshot().audit_events[0]
    assert event.metadata["synthetic_method_id"] == "smote"
    assert event.metadata["synthetic_random_seed"] == 42
    assert event.metadata["synthetic_generated_count"] == smote.report.generated_count


# ---------------------------------------------------------------------------
# Failed/rejected candidates are not overwritten
# ---------------------------------------------------------------------------


def test_failed_candidate_is_blocked_and_not_overwritten(tmp_path: Path) -> None:
    """Validation failed -> candidate status BLOCKED, artifact still persisted.

    A second build attempt with a different content hash creates a new
    artifact rather than overwriting the failed one. The first
    candidate-version artifact remains byte-identical.
    """
    storage, registry = _storage_and_registry()
    platform_client = FakePlatformMetadataClient()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    plan = _imputation_action_plan()
    imputation = execute_tabular_imputation_action(
        ExecuteTabularImputationRequest(
            action_plan_id=plan.action_plan_id,
            step=plan.steps[0],
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            target_column="is_fraud",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )
    failing_rule = BusinessRule(
        rule_id="critical_amount_threshold",
        severity=BusinessRuleSeverity.CRITICAL,
        checks=(RuleFieldCheck(field="amount", op="lt", value=1.0),),
    )
    gates_result = run_validation_gates(
        RunValidationGatesRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            candidate_artifact=imputation.candidate_artifact.artifact_ref,
            source_artifact=source_artifact,
            candidate_artifact_kind="candidate_tabular_dataset",
            schema_columns=tuple(_demo_schema_columns(tmp_path)),
            numeric_columns=("amount", "monthly_income"),
            business_rules=(failing_rule,),
            created_by_job_id="compute_run_apply_001",
            config_hash=_GATES_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )
    assert gates_result.report.candidate_status.value == "validation_failed"

    request = BuildCandidateVersionRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        proposed_version_name="dataset_version_v2_candidate",
        action_plan=plan,
        policy_versions=_policy_versions(),
        decision_report_id="decision_report_001",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CANDIDATE_CONFIG_HASH,
        source_artifacts=(source_artifact,),
        candidate_artifacts=(
            imputation.candidate_artifact.artifact_ref,
            gates_result.report_artifact.artifact_ref,
        ),
        primary_dataset_artifact=imputation.candidate_artifact.artifact_ref,
        validation_gates_report=gates_result.report,
        validation_gates_report_artifact=gates_result.report_artifact.artifact_ref,
        candidate_version_id="candidate_dataset_version_blocked_001",
        proposed_at=_GENERATED_AT,
    )
    first = build_candidate_dataset_version(
        request, registry=registry, platform_client=platform_client
    )
    first_bytes = storage.get(first.candidate_version_artifact.uri).data
    assert first.candidate_version.status is CandidateVersionStatus.BLOCKED
    assert first.candidate_version.block_export is True
    assert "business_rule_failure" in first.candidate_version.blocker_reason_codes

    # Re-run with the same inputs: idempotent, returns the same
    # candidate-version artifact bytes.
    second = build_candidate_dataset_version(
        request, registry=registry, platform_client=platform_client
    )
    assert second.candidate_version_artifact.uri == first.candidate_version_artifact.uri
    assert (
        storage.get(second.candidate_version_artifact.uri).data == first_bytes
    )

    # A different config_hash creates a *new* artifact instead of
    # overwriting the blocked one. The original blocked candidate
    # artifact must remain byte-identical.
    new_request = request.model_copy(
        update={
            "config_hash": "sha256:" + "1" * 64,
            "candidate_version_id": "candidate_dataset_version_blocked_002",
        }
    )
    third = build_candidate_dataset_version(
        new_request, registry=registry, platform_client=platform_client
    )
    assert third.candidate_version_artifact.uri != first.candidate_version_artifact.uri
    assert storage.get(first.candidate_version_artifact.uri).data == first_bytes


def test_synthetic_metadata_requires_synthetic_dataset_report_artifact() -> None:
    """Synthetic candidate without artifact ref raises explicit reason code."""
    storage, registry = _storage_and_registry()
    pack = load_contract_pack()
    payload = next(
        e.payload for e in pack.examples if e.name == "synthetic_dataset_report.smote"
    )
    synthetic_report = SyntheticDatasetReport.model_validate(payload)
    source_csv = b"object_id,is_fraud\n1,0\n2,1\n"
    source_artifact = registry.save_artifact(
        artifact_kind="raw_transactions",
        data=source_csv,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_v1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref
    plan = _smote_action_plan()

    with pytest.raises(CandidateVersionBuilderError) as exc:
        build_candidate_dataset_version(
            BuildCandidateVersionRequest(
                organization_id="org_1",
                project_id="project_1",
                dataset_id="dataset_1",
                parent_version_id="dataset_version_v1",
                proposed_version_name="dataset_version_v2_candidate",
                action_plan=plan,
                policy_versions=_policy_versions(),
                created_by_job_id="compute_run_apply_001",
                config_hash=_CANDIDATE_CONFIG_HASH,
                source_artifacts=(source_artifact,),
                candidate_artifacts=(source_artifact,),
                primary_dataset_artifact=source_artifact,
                synthetic_dataset_report=synthetic_report,
                synthetic_dataset_report_artifact=None,
                proposed_at=_GENERATED_AT,
            ),
            registry=registry,
        )
    assert exc.value.reason_code == "synthetic_dataset_report_artifact_required"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _policy_versions() -> CandidatePolicyVersions:
    return CandidatePolicyVersions(
        profile_policy_version="demo_strict_v1",
        decision_policy_version="decision_policy_v0",
        score_policy_version="dataforge_score_v0",
        method_policy_version="method_policy_v0",
        validation_gates_policy_version="validation_gates_policy_v0",
    )


def _imputation_action_plan() -> ActionPlan:
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )
    imputation = next(
        recommendation
        for recommendation in recommendations
        if recommendation.action_type == "IMPUTE_MISSING_VALUES"
    )
    return build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_001",
            source_dataset_version_id="dataset_version_v1",
            selected_decision_ids=(imputation.recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(imputation,),
            created_by_user_id="platform_user_123",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
            ),
            target_version_name="dataset_version_v2_candidate",
            created_at=_GENERATED_AT,
        )
    )


def _smote_action_plan() -> ActionPlan:
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
    smote = _smote_step()
    return ActionPlan(
        action_plan_id="action_plan_smote_001",
        plan_schema_version="action_plan.v1",
        source_dataset_version_id="dataset_version_v1",
        target_version_name="dataset_version_v2_candidate",
        created_from_decision_report="decision_report_smote",
        selected_decision_ids=("rec_smote_001",),
        created_by_user_id="platform_user_123",
        policy_version="synthetic_policy_v0",
        requires_approval=False,
        approval_request_id=None,
        execution_mode=WorkflowType.APPLY_SELECTED_ACTIONS,
        steps=(split_step, smote),
        validation_gates=(
            "split_leakage_check",
            "schema_validation",
            "business_rules",
            "synthetic_dcr_check",
        ),
        expected_outputs=(
            "split_manifest",
            "candidate_dataset_version",
            "synthetic_dataset_report",
        ),
        created_at=_GENERATED_AT,
    )


def _smote_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="augment_rare_class_smote",
        type="AUGMENT_RARE_CLASS",
        depends_on=("create_split_tabular",),
        idempotency_key="sha256:" + "1" * 64,
        method_id="smote",
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
            "method": "smote",
            "source_split": "train",
        },
        random_seed=42,
        retry_policy=RetryPolicy(
            max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _split_request(*, source_artifact: ArtifactRef) -> ExecuteTabularSplitRequest:
    split_step = _smote_action_plan().steps[0]
    return ExecuteTabularSplitRequest(
        action_plan_id="action_plan_smote_001",
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


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example for example in pack.examples if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


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


def _demo_schema_columns(tmp_path: Path) -> list[str]:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive_schema_probe")
    with open_archive_path(built.archive_path) as reader:
        header = reader.find_required_transactions().read_bytes().splitlines()[0]
    return header.decode("utf-8").split(",")


def _read_csv(data: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    return [dict(row) for row in reader]


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
