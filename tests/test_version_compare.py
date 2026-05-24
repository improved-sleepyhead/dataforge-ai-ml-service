"""Tests for TASK-052 Version Compare artifact builder."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from app.adapters import (
    ArtifactRegistry,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlan,
    ActionPlanStep,
    ArtifactRef,
    BaselineModelConfig,
    CandidateActionStepSummary,
    CandidateArtifactStatus,
    CandidateDatasetVersion,
    CandidateDatasetVersionLineage,
    CandidatePolicyVersions,
    CandidateValidationGate,
    CandidateVersionStatus,
    ClassCount,
    ClassificationMetrics,
    ClassImbalanceDiagnostics,
    CompareSignalStatus,
    ConfusionMatrixCell,
    DuplicateActionLineage,
    DuplicateActionMode,
    DuplicateActionReport,
    DuplicateDiagnostics,
    DuplicateGroupSummary,
    ImputationColumnReport,
    ImputationMethod,
    MetricStatus,
    ModelImpactReport,
    ModelImpactReportLineage,
    ModelImpactVerdict,
    RetryPolicy,
    SyntheticUtilityStatus,
    TabularImputationLineage,
    TabularImputationReport,
    TextOcrReport,
    TstrTrtsMetrics,
    ValidationGateSeverity,
    ValidationGatesLineage,
    ValidationGatesReport,
    ValidationGateStatus,
    ValidationGateType,
    VersionCompareReport,
    WorkflowType,
)
from app.kernel import (
    BuildDataForgeScoreRequest,
    BuildVersionCompareRequest,
    build_dataforge_score,
    build_version_compare_report,
)
from app.kernel.version_compare import VERSION_COMPARE_REPORT_KIND
from app.validation.contracts import load_contract_pack, validate_contract_payload

_GENERATED_AT = datetime(2026, 5, 28, 13, 0, tzinfo=UTC)
_CONFIG_HASH = "sha256:" + "a" * 64
_BASE_SOURCE_HASH = "sha256:" + "b" * 64
_CANDIDATE_HASH = "sha256:" + "c" * 64
_GATES_HASH = "sha256:" + "d" * 64
_MODEL_IMPACT_HASH = "sha256:" + "e" * 64


# ---------------------------------------------------------------------------
# Step 1+2+3: imputation candidate -> compare with imputed_fields and synthetic_added=0
# ---------------------------------------------------------------------------


def test_imputation_candidate_compare_includes_imputed_fields_and_action_plan() -> None:
    """Steps 1-3: imputation candidate compare records imputed_fields and the action plan id."""
    storage, registry = _storage_and_registry()
    plan = _imputation_action_plan()
    imputation_report = _imputation_report(plan=plan)
    gates_report = _passing_gates_report()
    candidate = _candidate_dataset_version(
        plan=plan,
        gates_report=gates_report,
    )
    baseline_score = build_dataforge_score(
        BuildDataForgeScoreRequest(
            tabular_profile=_baseline_tabular_profile(),
            text_ocr_report=_baseline_text_ocr_report(),
        )
    )
    candidate_score = build_dataforge_score(
        BuildDataForgeScoreRequest(
            tabular_profile=_candidate_tabular_profile_imputed(),
            text_ocr_report=_baseline_text_ocr_report(),
        )
    )

    request = BuildVersionCompareRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        base_version_id="dataset_version_v1",
        candidate_version_id="dataset_version_v2_candidate",
        candidate_dataset_version=candidate,
        baseline_tabular_profile=_baseline_tabular_profile(),
        candidate_tabular_profile=_candidate_tabular_profile_imputed(),
        baseline_text_ocr_report=_baseline_text_ocr_report(),
        candidate_text_ocr_report=_baseline_text_ocr_report(),
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        imputation_report=imputation_report,
        validation_gates_report=gates_report,
        validation_gates_report_artifact=_validation_gates_artifact_ref(),
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        report_id="version_compare_imputation_001",
        generated_at=_GENERATED_AT,
    )
    result = build_version_compare_report(request, registry=registry)
    report = result.report

    # Step 1: candidate creation -> compare artifact reflects parent/candidate ids.
    assert isinstance(report, VersionCompareReport)
    assert report.base_version_id == "dataset_version_v1"
    assert report.candidate_version_id == "dataset_version_v2_candidate"

    # Step 2: imputed_fields are populated and action_plan_id matches the
    # plan that produced the diff.
    imputed_fields = {entry.column: entry for entry in report.imputed_fields}
    assert "monthly_income" in imputed_fields
    monthly = imputed_fields["monthly_income"]
    assert monthly.method == "group_median"
    assert monthly.imputed_count == 4
    assert monthly.before_missing_count == 4
    assert monthly.after_missing_count == 0
    assert report.action_plan_id == plan.action_plan_id
    assert report.lineage.action_plan_id == plan.action_plan_id

    # changed_objects records imputed_object_count and synthetic_added=0 for
    # an imputation-only ActionPlan.
    assert report.changed_objects.imputed_object_count == 4
    assert report.changed_objects.synthetic_added_count == 0
    assert report.changed_objects.duplicate_marked_count == 0
    assert report.changed_objects.duplicate_removed_count == 0

    # PII risk is available with no delta when both reports are identical.
    assert report.pii_risk.status is CompareSignalStatus.AVAILABLE
    assert report.pii_risk.pii_record_count_delta == 0

    # Score diff exposes per-component decomposition and reproduces the
    # baseline policy version.
    assert report.score.status is CompareSignalStatus.AVAILABLE
    assert report.score.policy_version == baseline_score.policy_version
    components = {entry.component: entry for entry in report.score.components}
    assert "completeness" in components
    completeness = components["completeness"]
    # Imputation removes missingness, so completeness must not regress.
    assert completeness.value_after >= completeness.value_before

    # Step 3: report validates against the contract schema.
    validate_contract_payload(
        load_contract_pack(),
        "version_compare_report",
        report.model_dump(mode="json"),
    )

    # Persistence metadata + idempotency.
    assert result.report_artifact.artifact_kind == VERSION_COMPARE_REPORT_KIND
    stored = storage.get(result.report_artifact.uri)
    assert stored.info.metadata["report-id"] == "version_compare_imputation_001"
    assert stored.info.metadata["base-version-id"] == "dataset_version_v1"


# ---------------------------------------------------------------------------
# Step 2+3: synthetic SMOTE candidate -> synthetic_added + class balance shift
# ---------------------------------------------------------------------------


def test_smote_candidate_compare_records_synthetic_added_and_class_balance() -> None:
    """Steps 2-3: synthetic candidate exposes synthetic_added/class balance/model metrics."""
    storage, registry = _storage_and_registry()
    plan = _smote_action_plan()
    duplicate_report = _duplicate_action_report(plan=plan)
    gates_report = _passing_gates_report()
    candidate = _candidate_dataset_version(
        plan=plan,
        gates_report=gates_report,
    )
    baseline_score = build_dataforge_score(
        BuildDataForgeScoreRequest(tabular_profile=_baseline_tabular_profile())
    )
    candidate_score = build_dataforge_score(
        BuildDataForgeScoreRequest(
            tabular_profile=_candidate_tabular_profile_smote(),
        )
    )
    model_impact = _model_impact_report(verdict=ModelImpactVerdict.IMPROVED)

    request = BuildVersionCompareRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        base_version_id="dataset_version_v1",
        candidate_version_id="dataset_version_v2_candidate",
        candidate_dataset_version=candidate,
        baseline_tabular_profile=_baseline_tabular_profile(),
        candidate_tabular_profile=_candidate_tabular_profile_smote(),
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        model_impact_report=model_impact,
        model_impact_report_artifact=_model_impact_artifact_ref(),
        duplicate_action_report=duplicate_report,
        validation_gates_report=gates_report,
        validation_gates_report_artifact=_validation_gates_artifact_ref(),
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        report_id="version_compare_smote_001",
        generated_at=_GENERATED_AT,
    )
    result = build_version_compare_report(request, registry=registry)
    report = result.report

    # synthetic_added is reported separately from imputed_object_count.
    # We did not ship a synthetic_dataset_report on the request because
    # a SMOTE candidate is represented by its candidate_dataset_version
    # synthetic metadata, but we DO ship a duplicate-action report so
    # the duplicate_removed_count surfaces here.
    assert report.changed_objects.duplicate_removed_count == 4
    assert report.changed_objects.synthetic_added_count == 0
    # When no synthetic dataset report is supplied the synthetic_added
    # block is zero. Our second compare path tests synthetic_added > 0:
    request_with_synthetic = request.model_copy(
        update={
            "synthetic_dataset_report": _synthetic_smote_report(),
            "report_id": "version_compare_smote_with_synthetic",
        }
    )
    with_synthetic = build_version_compare_report(
        request_with_synthetic, registry=registry
    )
    assert with_synthetic.report.changed_objects.synthetic_added_count == 18

    # Class balance diff exposes per-class counts and rare-class shift.
    cb = report.class_balance
    assert cb.status is CompareSignalStatus.AVAILABLE
    assert cb.target_column == "is_fraud"
    assert cb.rare_class_label == "1"
    assert cb.rare_class_count_before == 5
    assert cb.rare_class_count_after == 23
    assert cb.rare_class_ratio_after is not None
    assert cb.rare_class_ratio_before is not None
    assert cb.rare_class_ratio_after > cb.rare_class_ratio_before

    # Model metrics are present because model impact eligibility is met.
    assert report.model_metrics.status is CompareSignalStatus.AVAILABLE
    assert report.model_metrics.verdict == ModelImpactVerdict.IMPROVED.value
    assert report.model_metrics.rare_class_recall_before == 0.30
    assert report.model_metrics.rare_class_recall_after == 0.55
    assert report.model_metrics.macro_f1_delta == 0.10
    assert report.model_metrics.model_impact_report is not None
    assert (
        report.model_metrics.model_impact_report.hash == _model_impact_artifact_ref().hash
    )

    # action_plan_id propagates from the candidate dataset version.
    assert report.action_plan_id == plan.action_plan_id

    # Validation gates summary mirrors the candidate version blockers.
    assert report.validation_gates.block_export is False
    assert report.validation_gates.overall_status == "passed"
    assert report.validation_gates.candidate_status == "ok"

    # Score diff includes the balance component and exposes a positive
    # delta when the candidate boosts rare-class balance.
    components = {entry.component: entry for entry in report.score.components}
    assert components["balance"].delta >= 0.0

    # Compare report validates against the contract schema.
    validate_contract_payload(
        load_contract_pack(),
        "version_compare_report",
        report.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Model metrics not_applicable when model impact eligibility is missing
# ---------------------------------------------------------------------------


def test_compare_marks_model_metrics_not_applicable_when_impact_missing() -> None:
    """A candidate without a model-impact report exposes status=not_applicable."""
    storage, registry = _storage_and_registry()
    plan = _imputation_action_plan()
    candidate = _candidate_dataset_version(
        plan=plan,
        gates_report=_passing_gates_report(),
    )
    request = BuildVersionCompareRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        base_version_id="dataset_version_v1",
        candidate_version_id="dataset_version_v2_candidate",
        candidate_dataset_version=candidate,
        baseline_tabular_profile=_baseline_tabular_profile(),
        candidate_tabular_profile=_candidate_tabular_profile_imputed(),
        validation_gates_report=_passing_gates_report(),
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
    )
    result = build_version_compare_report(request, registry=registry)
    report = result.report

    assert report.model_metrics.status is CompareSignalStatus.NOT_APPLICABLE
    assert report.model_metrics.not_applicable_reason == "model_impact_report_not_provided"
    assert report.score.status is CompareSignalStatus.NOT_APPLICABLE
    assert report.score.not_applicable_reason == "dataforge_score_not_provided"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _imputation_action_plan() -> ActionPlan:
    step = ActionPlanStep(
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
    return ActionPlan(
        action_plan_id="action_plan_imputation_001",
        plan_schema_version="action_plan.v1",
        source_dataset_version_id="dataset_version_v1",
        target_version_name="dataset_version_v2_candidate",
        created_from_decision_report="decision_report_001",
        selected_decision_ids=("rec_imputation_001",),
        created_by_user_id="platform_user_123",
        policy_version="method_policy_v0",
        requires_approval=False,
        approval_request_id=None,
        execution_mode=WorkflowType.APPLY_SELECTED_ACTIONS,
        steps=(step,),
        validation_gates=("schema_validation", "business_rules"),
        expected_outputs=("imputation_report", "candidate_tabular_dataset"),
        created_at=_GENERATED_AT,
    )


def _smote_action_plan() -> ActionPlan:
    step = ActionPlanStep(
        step_id="augment_rare_class_smote",
        type="AUGMENT_RARE_CLASS",
        depends_on=(),
        idempotency_key="sha256:" + "3" * 64,
        method_id="smote",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "4" * 64,
        policy_version="synthetic_policy_v0",
        validation_gates=("schema_validation", "business_rules", "synthetic_dcr_check"),
        preconditions=(
            "source_version_is_immutable",
            "train_split_exists",
            "leakage_checks_passed",
        ),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind="candidate_tabular_dataset",
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
        steps=(step,),
        validation_gates=("schema_validation", "synthetic_dcr_check"),
        expected_outputs=(
            "candidate_tabular_dataset",
            "synthetic_dataset_report",
        ),
        created_at=_GENERATED_AT,
    )


def _imputation_report(*, plan: ActionPlan) -> TabularImputationReport:
    source_artifact = _baseline_artifact_ref()
    candidate_artifact = _candidate_artifact_ref()
    return TabularImputationReport(
        report_id="tabular_imputation_report_test",
        action_plan_id=plan.action_plan_id,
        step_id=plan.steps[0].step_id,
        target_column="is_fraud",
        target_unchanged=True,
        before_row_count=100,
        after_row_count=100,
        before_missing_total=4,
        after_missing_total=0,
        columns=(
            ImputationColumnReport(
                column="monthly_income",
                method=ImputationMethod.GROUP_MEDIAN,
                indicator_column="monthly_income_was_missing",
                group_key="customer_segment",
                before_missing_count=4,
                after_missing_count=0,
                imputed_count=4,
                total_count=100,
                fill_value=None,
                group_imputed_counts={"young_customers": 3, "regular_customers": 1},
            ),
        ),
        candidate_artifact=candidate_artifact,
        lineage=TabularImputationLineage(
            action_plan_id=plan.action_plan_id,
            step_id=plan.steps[0].step_id,
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            source_artifact=source_artifact,
            candidate_artifact=candidate_artifact,
        ),
        generated_at=_GENERATED_AT,
    )


def _duplicate_action_report(*, plan: ActionPlan) -> DuplicateActionReport:
    source_artifact = _baseline_artifact_ref()
    candidate_artifact = _candidate_artifact_ref()
    return DuplicateActionReport(
        report_id="duplicate_action_report_test",
        mode=DuplicateActionMode.REMOVE_CANDIDATE,
        id_column="object_id",
        signature_columns=("amount", "is_fraud"),
        before_row_count=100,
        after_row_count=96,
        before_duplicate_pair_count=4,
        before_duplicate_group_count=4,
        after_duplicate_pair_count=0,
        after_duplicate_group_count=0,
        marked_count=0,
        removed_count=4,
        groups=(
            DuplicateGroupSummary(
                group_id="duplicate_group_0001",
                signature_hash="sha256:" + "0" * 64,
                affected_object_ids=("txn_001", "txn_002"),
                kept_object_id="txn_001",
                removed_object_ids=("txn_002",),
                marked_object_ids=(),
            ),
        ),
        raw_artifact_hash=_BASE_SOURCE_HASH,
        raw_artifact_unchanged=True,
        lineage=DuplicateActionLineage(
            action_plan_id=plan.action_plan_id,
            step_id=plan.steps[0].step_id,
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            source_artifact=source_artifact,
            candidate_artifact=candidate_artifact,
        ),
        generated_at=_GENERATED_AT,
    )


def _synthetic_smote_report() -> Any:
    """Load the SMOTE example from the contract pack and parse it.

    The SMOTE example carries 18 generated rows, which the compare
    report must surface as ``synthetic_added_count``.
    """
    from app.domain import SyntheticDatasetReport

    pack = load_contract_pack()
    payload = next(
        e.payload for e in pack.examples if e.name == "synthetic_dataset_report.smote"
    )
    return SyntheticDatasetReport.model_validate(payload)


def _passing_gates_report() -> ValidationGatesReport:
    return ValidationGatesReport(
        report_id="validation_gates_report_test",
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        candidate_artifact_kind="candidate_tabular_dataset",
        policy_version="validation_gates_policy_v0",
        overall_status=ValidationGateStatus.PASSED,
        candidate_status=CandidateArtifactStatus.OK,
        raw_artifact_unchanged=True,
        block_export=False,
        block_model_evaluation=False,
        block_training=False,
        blocker_present=False,
        blocker_gate_types=(),
        gates=(
            CandidateValidationGate(
                gate_type=ValidationGateType.SCHEMA_VALIDATION,
                status=ValidationGateStatus.PASSED,
                severity=ValidationGateSeverity.BLOCKER,
                reason_code="schema_columns_match",
                block_action=None,
                metrics=(),
                findings=(),
                notes=None,
                not_applicable_reason=None,
            ),
        ),
        lineage=ValidationGatesLineage(
            action_plan_id="action_plan_imputation_001",
            step_id="impute_income_001",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            source_artifact=_baseline_artifact_ref(),
            candidate_artifact=_candidate_artifact_ref(),
        ),
        generated_at=_GENERATED_AT,
    )


def _candidate_dataset_version(
    *, plan: ActionPlan, gates_report: ValidationGatesReport
) -> CandidateDatasetVersion:
    summary = tuple(
        CandidateActionStepSummary(
            step_id=step.step_id,
            step_type=step.type,
            method_id=step.method_id,
            plugin_id=step.plugin_id,
            plugin_version=step.plugin_version,
            config_hash=step.config_hash,
            output_artifact_kind=step.output_artifact_kind,
            random_seed=step.random_seed,
        )
        for step in plan.steps
    )
    return CandidateDatasetVersion(
        candidate_version_id="candidate_dataset_version_test_001",
        status=CandidateVersionStatus.PROPOSED,
        policy_versions=CandidatePolicyVersions(
            profile_policy_version="demo_strict_v1",
            decision_policy_version="decision_policy_v0",
            score_policy_version="dataforge_score_v0",
            method_policy_version="method_policy_v0",
            validation_gates_policy_version="validation_gates_policy_v0",
        ),
        lineage=CandidateDatasetVersionLineage(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
            parent_version_id="dataset_version_v1",
            proposed_version_name="dataset_version_v2_candidate",
            action_plan_id=plan.action_plan_id,
            decision_report_id="decision_report_001",
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            input_artifact_hashes=(_BASE_SOURCE_HASH,),
            output_artifact_hashes=(_CANDIDATE_HASH, _GATES_HASH),
        ),
        candidate_artifacts=(_candidate_artifact_ref(),),
        primary_dataset_artifact=_candidate_artifact_ref(),
        validation_gates_report=_validation_gates_artifact_ref(),
        validation_gates_summary={
            "overall_status": gates_report.overall_status.value,
            "candidate_status": gates_report.candidate_status.value,
            "raw_artifact_unchanged": gates_report.raw_artifact_unchanged,
            "blocker_present": gates_report.blocker_present,
        },
        block_export=gates_report.block_export,
        block_model_evaluation=gates_report.block_model_evaluation,
        block_training=gates_report.block_training,
        blocker_reason_codes=(),
        action_plan_steps=summary,
        synthetic_metadata=None,
        proposed_at=_GENERATED_AT,
    )


def _baseline_tabular_profile() -> Any:
    from app.domain import TabularProfileReport

    pack = load_contract_pack()
    payload = next(
        e.payload for e in pack.examples if e.name == "tabular_profile_report.fraud"
    )
    profile = TabularProfileReport.model_validate(payload)
    # Pin counts to a stable baseline used across the compare tests.
    return profile.model_copy(
        update={
            "row_count": 100,
            "duplicates": DuplicateDiagnostics(
                duplicate_pair_count=4,
                duplicate_group_count=4,
                affected_object_ids=("txn_002", "txn_010"),
                signature_columns=("amount", "is_fraud"),
                id_column="object_id",
            ),
            "class_imbalance": ClassImbalanceDiagnostics(
                target_column="is_fraud",
                total_samples=100,
                class_counts=(
                    ClassCount(label="0", count=95),
                    ClassCount(label="1", count=5),
                ),
                rare_class_label="1",
                rare_class_count=5,
                rare_class_ratio=0.05,
                minority_class_label="1",
                minority_class_share=0.05,
                imbalance_ratio=19.0,
                balance_score=0.42,
                balance_score_alternative=0.1,
                effective_number_beta=0.999,
                effective_number_of_samples={"0": 91.0, "1": 5.0},
            ),
        }
    )


def _candidate_tabular_profile_imputed() -> Any:
    profile = _baseline_tabular_profile()
    if profile.missingness is None:
        return profile
    cleared_columns = tuple(
        column.model_copy(update={"missing_count": 0, "missing_ratio": 0.0})
        for column in profile.missingness.columns
    )
    cleared_missingness = profile.missingness.model_copy(
        update={
            "target_column_missing": False,
            "columns": cleared_columns,
        }
    )
    return profile.model_copy(update={"missingness": cleared_missingness})


def _candidate_tabular_profile_smote() -> Any:
    profile = _baseline_tabular_profile()
    rare_after = 23
    majority_after = 95
    total = rare_after + majority_after
    return profile.model_copy(
        update={
            "row_count": total,
            "duplicates": DuplicateDiagnostics(
                duplicate_pair_count=0,
                duplicate_group_count=0,
                affected_object_ids=(),
                signature_columns=("amount", "is_fraud"),
                id_column="object_id",
            ),
            "class_imbalance": ClassImbalanceDiagnostics(
                target_column="is_fraud",
                total_samples=total,
                class_counts=(
                    ClassCount(label="0", count=majority_after),
                    ClassCount(label="1", count=rare_after),
                ),
                rare_class_label="1",
                rare_class_count=rare_after,
                rare_class_ratio=rare_after / total,
                minority_class_label="1",
                minority_class_share=rare_after / total,
                imbalance_ratio=majority_after / rare_after,
                balance_score=0.65,
                balance_score_alternative=0.45,
                effective_number_beta=0.999,
                effective_number_of_samples={
                    "0": float(majority_after),
                    "1": float(rare_after),
                },
            ),
        }
    )


def _baseline_text_ocr_report() -> TextOcrReport:
    pack = load_contract_pack()
    payload = next(
        e.payload for e in pack.examples if e.name == "text_ocr_report.privacy"
    )
    return TextOcrReport.model_validate(payload)


def _model_impact_report(*, verdict: ModelImpactVerdict) -> ModelImpactReport:
    metrics_before = ClassificationMetrics(
        rare_class_label="1",
        rare_class_recall=0.30,
        rare_class_precision=0.40,
        macro_f1=0.60,
        weighted_f1=0.85,
        pr_auc=0.45,
        pr_auc_status=MetricStatus.AVAILABLE,
        pr_auc_reason=None,
        confusion_matrix=(
            ConfusionMatrixCell(true_label="0", predicted_label="0", count=28),
            ConfusionMatrixCell(true_label="0", predicted_label="1", count=1),
            ConfusionMatrixCell(true_label="1", predicted_label="0", count=1),
            ConfusionMatrixCell(true_label="1", predicted_label="1", count=0),
        ),
        sample_count=30,
    )
    metrics_after = metrics_before.model_copy(
        update={
            "rare_class_recall": 0.55,
            "rare_class_precision": 0.50,
            "macro_f1": 0.70,
            "weighted_f1": 0.88,
            "pr_auc": 0.55,
            "confusion_matrix": (
                ConfusionMatrixCell(true_label="0", predicted_label="0", count=27),
                ConfusionMatrixCell(true_label="0", predicted_label="1", count=2),
                ConfusionMatrixCell(true_label="1", predicted_label="0", count=0),
                ConfusionMatrixCell(true_label="1", predicted_label="1", count=1),
            ),
        }
    )
    return ModelImpactReport(
        report_id="model_impact_report_test",
        metric_library_version="1.5.0",
        verdict=verdict,
        verdict_reason_codes=("rare_class_recall_improved",),
        notes=None,
        baseline_metrics=metrics_before,
        candidate_metrics=metrics_after,
        rare_class_recall_before=0.30,
        rare_class_recall_after=0.55,
        rare_class_recall_delta=0.25,
        macro_f1_before=0.60,
        macro_f1_after=0.70,
        macro_f1_delta=0.10,
        weighted_f1_before=0.85,
        weighted_f1_after=0.88,
        weighted_f1_delta=0.03,
        pr_auc_before=0.45,
        pr_auc_after=0.55,
        pr_auc_delta=0.10,
        pr_auc_status=MetricStatus.AVAILABLE,
        pr_auc_reason=None,
        tstr_trts=TstrTrtsMetrics(
            status=MetricStatus.NOT_APPLICABLE,
            reason="no_synthetic_rows_in_candidate",
        ),
        synthetic_utility_status=SyntheticUtilityStatus.NOT_APPLICABLE,
        synthetic_utility_reason_codes=(),
        baseline_model_config=BaselineModelConfig(
            algorithm="LogisticRegression",
            library="scikit-learn",
            library_version="1.5.0",
            hyperparameters={"max_iter": 500, "solver": "lbfgs"},
            feature_columns=("amount", "monthly_income"),
            target_column="is_fraud",
            excluded_columns=("object_id",),
            random_seed=42,
        ),
        candidate_model_config=BaselineModelConfig(
            algorithm="LogisticRegression",
            library="scikit-learn",
            library_version="1.5.0",
            hyperparameters={"max_iter": 500, "solver": "lbfgs"},
            feature_columns=("amount", "monthly_income"),
            target_column="is_fraud",
            excluded_columns=("object_id",),
            random_seed=42,
        ),
        lineage=ModelImpactReportLineage(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
            parent_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            candidate_version_artifact=None,
            eligibility_report_artifact=None,
            source_split_manifest=None,
            candidate_split_manifest=None,
            validation_gates_report_artifact=None,
            synthetic_dataset_report_artifact=None,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
        ),
        generated_at=_GENERATED_AT,
    )


def _baseline_artifact_ref() -> ArtifactRef:
    from app.domain import ArtifactLineage

    return ArtifactRef(
        artifact_id="raw_transactions:" + "b" * 16,
        kind="raw_transactions",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v1/transactions.csv",
        hash=_BASE_SOURCE_HASH,
        media_type="text/csv",
        size_bytes=2048,
        schema_version="tabular_dataset.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v1",
            job_id="compute_run_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _candidate_artifact_ref() -> ArtifactRef:
    from app.domain import ArtifactLineage

    return ArtifactRef(
        artifact_id="candidate_tabular_dataset:" + "c" * 16,
        kind="candidate_tabular_dataset",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/transactions.csv",
        hash=_CANDIDATE_HASH,
        media_type="text/csv",
        size_bytes=2048,
        schema_version="tabular_dataset.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _validation_gates_artifact_ref() -> ArtifactRef:
    from app.domain import ArtifactLineage

    return ArtifactRef(
        artifact_id="validation_gates_report:" + "d" * 16,
        kind="validation_gates_report",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/validation_gates_report.json",
        hash=_GATES_HASH,
        media_type="application/json",
        size_bytes=1024,
        schema_version="validation_gates_report.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _model_impact_artifact_ref() -> ArtifactRef:
    from app.domain import ArtifactLineage

    return ArtifactRef(
        artifact_id="model_impact_report:" + "e" * 16,
        kind="model_impact_report",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/model_impact_report.json",
        hash=_MODEL_IMPACT_HASH,
        media_type="application/json",
        size_bytes=4096,
        schema_version="model_impact_report.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
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
            from app.domain import ErrorCode

            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc
