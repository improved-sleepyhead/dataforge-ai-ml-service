"""Tests for TASK-048 validation gates against candidate artifacts."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters import (
    ArtifactRegistry,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlan,
    ArtifactRef,
    BusinessRuleSeverity,
    CandidateArtifactStatus,
    ErrorCode,
    RetryPolicy,
    SyntheticDatasetReport,
    TabularProfileReport,
    ValidationGateSeverity,
    ValidationGatesReport,
    ValidationGateStatus,
    ValidationGateType,
)
from app.ingestion import open_archive_path
from app.kernel import (
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    build_action_plan_preview,
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
    VALIDATION_GATES_REPORT_KIND,
    VALIDATION_GATES_REPORT_SCHEMA_VERSION,
    DcrThresholds,
    RunValidationGatesRequest,
    run_validation_gates,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "c" * 64
_GATES_CONFIG_HASH = "sha256:" + "d" * 64
_GENERATED_AT = datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step 1: valid candidate action -> all gates pass
# ---------------------------------------------------------------------------


def test_valid_candidate_action_passes_all_gates(tmp_path: Path) -> None:
    """Step 1: a valid imputation candidate passes all applicable gates."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    candidate_artifact = _imputation_candidate(
        storage=storage, registry=registry, source_artifact=source_artifact
    )

    request = _build_request(
        source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
        candidate_artifact_kind="candidate_tabular_dataset",
        business_rules=_passing_rules(),
    )
    result = run_validation_gates(request, storage=storage, registry=registry)
    report = result.report

    assert isinstance(report, ValidationGatesReport)
    assert report.overall_status is ValidationGateStatus.PASSED
    assert report.candidate_status is CandidateArtifactStatus.OK
    assert report.raw_artifact_unchanged is True
    assert report.block_export is False
    assert report.block_model_evaluation is False
    assert report.block_training is False
    assert report.blocker_present is False
    assert report.blocker_gate_types == ()

    gate_status = {gate.gate_type: gate.status for gate in report.gates}
    assert gate_status[ValidationGateType.SCHEMA_VALIDATION] is ValidationGateStatus.PASSED
    assert gate_status[ValidationGateType.BUSINESS_RULES] is ValidationGateStatus.PASSED
    assert gate_status[ValidationGateType.PRIVACY_CHECK] is ValidationGateStatus.PASSED
    assert (
        gate_status[ValidationGateType.RAW_ARTIFACT_IMMUTABILITY]
        is ValidationGateStatus.PASSED
    )

    # Step 5: TSTR/TRTS/SHAP/TabSynDex must surface explicit
    # not_applicable reasons rather than silently disappearing.
    not_applicable = {
        gate.gate_type: gate
        for gate in report.gates
        if gate.status is ValidationGateStatus.NOT_APPLICABLE
    }
    assert ValidationGateType.SYNTHETIC_TSTR in not_applicable
    assert ValidationGateType.SYNTHETIC_TRTS in not_applicable
    assert ValidationGateType.SYNTHETIC_SHAP_CONSISTENCY in not_applicable
    assert ValidationGateType.SYNTHETIC_TABSYNDEX in not_applicable
    for gate in not_applicable.values():
        assert gate.not_applicable_reason is not None

    assert result.report_artifact.artifact_kind == VALIDATION_GATES_REPORT_KIND
    assert result.report_artifact.schema_version == VALIDATION_GATES_REPORT_SCHEMA_VERSION
    stored_metadata = storage.get(result.report_artifact.uri).info.metadata
    assert stored_metadata["overall-status"] == "passed"
    assert stored_metadata["candidate-status"] == "ok"
    assert stored_metadata["raw-artifact-unchanged"] == "true"
    assert stored_metadata["block-export"] == "false"

    validate_contract_payload(
        load_contract_pack(),
        "validation_gates_report",
        report.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Step 2: business rule violation -> failed gate, candidate marked
# ---------------------------------------------------------------------------


def test_business_rule_violation_marks_candidate_validation_failed(tmp_path: Path) -> None:
    """Step 2: a critical rule violation produces a failed gate and blocks export."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    candidate_artifact = _imputation_candidate(
        storage=storage, registry=registry, source_artifact=source_artifact
    )

    # Critical rule: amount must be < 1; demo data has many higher
    # amounts, so this rule will fail with severity=critical.
    impossible_rule = BusinessRule(
        rule_id="critical_amount_threshold",
        description="amount must stay below 1.0",
        severity=BusinessRuleSeverity.CRITICAL,
        checks=(RuleFieldCheck(field="amount", op="lt", value=1.0),),
    )
    request = _build_request(
        source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
        candidate_artifact_kind="candidate_tabular_dataset",
        business_rules=(impossible_rule,),
    )
    result = run_validation_gates(request, storage=storage, registry=registry)
    report = result.report

    assert report.overall_status is ValidationGateStatus.FAILED
    assert report.candidate_status is CandidateArtifactStatus.VALIDATION_FAILED
    assert report.block_export is True
    assert ValidationGateType.BUSINESS_RULES in report.blocker_gate_types

    business_rules_gate = next(
        gate for gate in report.gates if gate.gate_type is ValidationGateType.BUSINESS_RULES
    )
    assert business_rules_gate.status is ValidationGateStatus.FAILED
    assert business_rules_gate.severity is ValidationGateSeverity.BLOCKER
    assert business_rules_gate.reason_code == "business_rule_failure"
    assert business_rules_gate.block_action == "BLOCK_EXPORT"
    assert business_rules_gate.findings_count > 0
    assert business_rules_gate.findings  # bounded sample present

    stored_metadata = storage.get(result.report_artifact.uri).info.metadata
    assert stored_metadata["candidate-status"] == "validation_failed"
    assert stored_metadata["block-export"] == "true"


# ---------------------------------------------------------------------------
# Step 3: raw / source artifact unchanged after gates run
# ---------------------------------------------------------------------------


def test_raw_artifact_unchanged_after_gates_failure(tmp_path: Path) -> None:
    """Step 3: source artifact is byte-identical after a failed gates run."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    source_bytes_before = storage.get(source_artifact.uri).data

    candidate_artifact = _imputation_candidate(
        storage=storage, registry=registry, source_artifact=source_artifact
    )
    candidate_bytes_before = storage.get(candidate_artifact.uri).data

    # Use a critical rule guaranteed to fail so the gates report blocks
    # export. The source / candidate artifacts must remain unchanged.
    request = _build_request(
        source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
        candidate_artifact_kind="candidate_tabular_dataset",
        business_rules=(
            BusinessRule(
                rule_id="critical_amount_threshold",
                severity=BusinessRuleSeverity.CRITICAL,
                checks=(RuleFieldCheck(field="amount", op="lt", value=1.0),),
            ),
        ),
    )
    result = run_validation_gates(request, storage=storage, registry=registry)
    assert result.report.candidate_status is CandidateArtifactStatus.VALIDATION_FAILED

    source_bytes_after = storage.get(source_artifact.uri).data
    candidate_bytes_after = storage.get(candidate_artifact.uri).data
    assert source_bytes_after == source_bytes_before
    assert candidate_bytes_after == candidate_bytes_before
    immutability_gate = next(
        gate
        for gate in result.report.gates
        if gate.gate_type is ValidationGateType.RAW_ARTIFACT_IMMUTABILITY
    )
    assert immutability_gate.status is ValidationGateStatus.PASSED
    assert result.report.raw_artifact_unchanged is True


# ---------------------------------------------------------------------------
# Step 4: synthetic row that duplicates a real row is blocked
# ---------------------------------------------------------------------------


def test_synthetic_exact_duplicate_to_real_blocks_export() -> None:
    """Step 4: a synthetic candidate with an exact-duplicate row is blocked."""
    storage, registry = _storage_and_registry()
    source_csv = (
        "object_id,is_fraud,amount,monthly_income\n"
        "txn_1,0,10,1000\n"
        "txn_2,1,200,1500\n"
        "txn_3,0,30,1200\n"
    )
    source_artifact = registry.save_artifact(
        artifact_kind="raw_transactions",
        data=source_csv.encode("utf-8"),
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref

    # Candidate carries a synthetic row whose feature vector exactly
    # matches txn_2. The gate must fail closed and block export.
    candidate_csv = (
        "object_id,is_fraud,amount,monthly_income,is_synthetic,synthetic_source_split\n"
        "txn_1,0,10,1000,0,\n"
        "txn_2,1,200,1500,0,\n"
        "txn_3,0,30,1200,0,\n"
        "txn_synth_0001,1,200,1500,1,train\n"
    )
    candidate_artifact = registry.save_artifact(
        artifact_kind="candidate_tabular_dataset",
        data=candidate_csv.encode("utf-8"),
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_2_candidate",
        created_by_job_id="compute_run_apply_001",
        config_hash=_GATES_CONFIG_HASH,
    ).artifact_ref

    synthetic_report = _make_synthetic_report(
        source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
    )
    request = RunValidationGatesRequest(
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        candidate_artifact=candidate_artifact,
        source_artifact=source_artifact,
        candidate_artifact_kind="candidate_tabular_dataset",
        schema_columns=("object_id", "is_fraud", "amount", "monthly_income"),
        numeric_columns=("amount", "monthly_income"),
        synthetic_dataset_report=synthetic_report,
        dcr_thresholds=DcrThresholds(),
        created_by_job_id="compute_run_apply_001",
        config_hash=_GATES_CONFIG_HASH,
        action_plan_id="action_plan_smote_001",
        step_id="augment_rare_class_smote",
        report_id="validation_gates_smote_dup",
        generated_at=_GENERATED_AT,
    )
    result = run_validation_gates(request, storage=storage, registry=registry)
    report = result.report

    assert report.candidate_status is CandidateArtifactStatus.VALIDATION_FAILED
    assert report.block_export is True
    assert ValidationGateType.SYNTHETIC_EXACT_DUPLICATE_TO_REAL in report.blocker_gate_types

    exact_duplicate_gate = next(
        gate
        for gate in report.gates
        if gate.gate_type is ValidationGateType.SYNTHETIC_EXACT_DUPLICATE_TO_REAL
    )
    assert exact_duplicate_gate.status is ValidationGateStatus.FAILED
    assert exact_duplicate_gate.severity is ValidationGateSeverity.BLOCKER
    assert exact_duplicate_gate.block_action == "BLOCK_EXPORT"
    assert exact_duplicate_gate.findings_count >= 1

    # DCR gate must also fire because the nearest neighbor distance is
    # zero for the duplicated synthetic row.
    dcr_gate = next(
        gate
        for gate in report.gates
        if gate.gate_type is ValidationGateType.SYNTHETIC_DCR_CHECK
    )
    assert dcr_gate.status is ValidationGateStatus.FAILED
    dcr_min = next(metric for metric in dcr_gate.metrics if metric.name == "dcr_min")
    assert dcr_min.formula == "DCR(x_synth) = min_{x_real in D_real} distance(x_synth, x_real)"
    assert dcr_min.value == 0.0


# ---------------------------------------------------------------------------
# Step 5: explicit not_applicable reasons for unsupported metrics
# ---------------------------------------------------------------------------


def test_dcr_and_tstr_emit_explicit_not_applicable_when_no_synthetic_report(
    tmp_path: Path,
) -> None:
    """Step 5: missing synthetic context produces explicit not_applicable reasons."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    candidate_artifact = _imputation_candidate(
        storage=storage, registry=registry, source_artifact=source_artifact
    )

    request = _build_request(
        source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
        candidate_artifact_kind="candidate_tabular_dataset",
        business_rules=(),
    )
    result = run_validation_gates(request, storage=storage, registry=registry)
    report = result.report

    not_applicable_gates = {
        gate.gate_type: gate
        for gate in report.gates
        if gate.status is ValidationGateStatus.NOT_APPLICABLE
    }
    assert (
        not_applicable_gates[
            ValidationGateType.SYNTHETIC_DCR_CHECK
        ].not_applicable_reason
        == "synthetic_artifact_not_provided"
    )
    assert (
        not_applicable_gates[ValidationGateType.SYNTHETIC_TSTR].not_applicable_reason
        == "model_impact_pipeline_not_available"
    )
    assert (
        not_applicable_gates[ValidationGateType.SYNTHETIC_TRTS].not_applicable_reason
        == "model_impact_pipeline_not_available"
    )
    # Business rules also resolve to not_applicable when no rules are
    # configured for the candidate.
    business_gate = next(
        gate
        for gate in report.gates
        if gate.gate_type is ValidationGateType.BUSINESS_RULES
    )
    assert business_gate.status is ValidationGateStatus.NOT_APPLICABLE
    assert business_gate.reason_code == "no_business_rules_configured"


def test_smote_candidate_passes_dcr_gate_when_no_collisions(tmp_path: Path) -> None:
    """Synthetic candidate with no real-row duplicates passes synthetic gates."""
    storage, registry = _storage_and_registry()
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
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
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

    candidate_artifact = smote.candidate_artifact.artifact_ref
    request = RunValidationGatesRequest(
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        candidate_artifact=candidate_artifact,
        source_artifact=source_artifact,
        candidate_artifact_kind="candidate_tabular_dataset",
        schema_columns=tuple(_demo_schema_columns()),
        numeric_columns=("amount", "monthly_income"),
        synthetic_dataset_report=smote.report,
        synthetic_dataset_report_artifact=smote.report_artifact.artifact_ref,
        split_manifest_artifact=smote.augmented_split_artifact.artifact_ref,
        dcr_thresholds=DcrThresholds(),
        created_by_job_id="compute_run_apply_001",
        config_hash=_GATES_CONFIG_HASH,
        action_plan_id="action_plan_smote_001",
        step_id="augment_rare_class_smote",
        report_id="validation_gates_smote_pass",
        generated_at=_GENERATED_AT,
    )
    result = run_validation_gates(request, storage=storage, registry=registry)
    report = result.report

    # SMOTE-generated rows are unique by construction, so the synthetic
    # privacy gates must pass and the candidate is OK.
    dcr_gate = next(
        gate for gate in report.gates if gate.gate_type is ValidationGateType.SYNTHETIC_DCR_CHECK
    )
    exact_gate = next(
        gate
        for gate in report.gates
        if gate.gate_type is ValidationGateType.SYNTHETIC_EXACT_DUPLICATE_TO_REAL
    )
    assert dcr_gate.status is ValidationGateStatus.PASSED
    assert exact_gate.status is ValidationGateStatus.PASSED
    assert report.block_export is False
    assert report.candidate_status is CandidateArtifactStatus.OK


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_request(
    *,
    source_artifact: ArtifactRef,
    candidate_artifact: ArtifactRef,
    candidate_artifact_kind: str,
    business_rules: tuple[BusinessRule, ...],
) -> RunValidationGatesRequest:
    return RunValidationGatesRequest(
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        candidate_artifact=candidate_artifact,
        source_artifact=source_artifact,
        candidate_artifact_kind=candidate_artifact_kind,
        schema_columns=tuple(_demo_schema_columns()),
        numeric_columns=("amount", "monthly_income"),
        business_rules=business_rules,
        rules_version="tabular_business_rules.v1",
        rules_config_hash=None,
        created_by_job_id="compute_run_apply_001",
        config_hash=_GATES_CONFIG_HASH,
        action_plan_id="action_plan_001",
        step_id="impute_income_001",
        report_id="validation_gates_imputation_001",
        generated_at=_GENERATED_AT,
    )


def _passing_rules() -> tuple[BusinessRule, ...]:
    return (
        BusinessRule(
            rule_id="amount_is_positive",
            severity=BusinessRuleSeverity.WARNING,
            checks=(RuleFieldCheck(field="amount", op="gte", value=0.0),),
        ),
    )


def _imputation_candidate(
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    source_artifact: ArtifactRef,
) -> ArtifactRef:
    plan = _imputation_action_plan()
    result = execute_tabular_imputation_action(
        ExecuteTabularImputationRequest(
            action_plan_id=plan.action_plan_id,
            step=plan.steps[0],
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            target_column="is_fraud",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            report_id="tabular_imputation_report_test",
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )
    return result.candidate_artifact.artifact_ref


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
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(imputation.recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(imputation,),
            created_by_user_id="platform_user_123",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
            ),
            target_version_name="dataset_version_2_candidate",
            created_at=_GENERATED_AT,
        )
    )


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example for example in pack.examples if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def _smote_step() -> Any:
    from app.domain import ActionPlanStep

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
            max_attempts=2,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _split_request(*, source_artifact: ArtifactRef) -> ExecuteTabularSplitRequest:
    from app.domain import ActionPlanStep

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
        generated_at=_GENERATED_AT,
    )


def _make_synthetic_report(
    *,
    source_artifact: ArtifactRef,
    candidate_artifact: ArtifactRef,
) -> SyntheticDatasetReport:
    """Build a minimal synthetic report fixture for the duplicate-to-real test."""
    from app.domain import (
        SMOTE_FORMULA,
        SyntheticAugmentationKind,
        SyntheticClassStats,
        SyntheticDatasetLineage,
        SyntheticDatasetReport,
        SyntheticGenerationMethod,
        SyntheticSampleLineage,
    )

    return SyntheticDatasetReport(
        report_id="synthetic_dataset_report_dup",
        method=SyntheticGenerationMethod.SMOTE,
        method_version="0.1.0",
        augmentation_kind=SyntheticAugmentationKind.TARGETED_RARE_CLASS,
        formula=SMOTE_FORMULA,
        target_column="is_fraud",
        rare_class_label="1",
        source_split="train",
        random_seed=42,
        k_neighbors=3,
        sampling_strategy=0.5,
        feature_columns=("amount", "monthly_income"),
        excluded_columns=(),
        real_total_count=3,
        real_train_count=2,
        real_train_rare_count=1,
        generated_count=1,
        class_stats=(
            SyntheticClassStats(
                label="1",
                real_count_in_source_split=1,
                real_count_in_majority_split=2,
                target_count_after_augmentation=2,
                generated_count=1,
                achieved_ratio=1.0,
            ),
        ),
        sample_lineage=(
            SyntheticSampleLineage(
                synthetic_object_id="txn_synth_0001",
                method=SyntheticGenerationMethod.SMOTE,
                formula=SMOTE_FORMULA,
                seed_object_id="txn_2",
                neighbor_object_id="txn_2",
                lambda_value=0.0,
                rare_class_label="1",
            ),
        ),
        sample_lineage_truncated=False,
        full_sample_lineage_count=1,
        synthetic_validation=None,
        gaussian_copula_artifacts=None,
        lineage=SyntheticDatasetLineage(
            action_plan_id="action_plan_smote_001",
            step_id="augment_rare_class_smote",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            created_by_job_id="compute_run_apply_001",
            config_hash=_GATES_CONFIG_HASH,
            source_artifact=source_artifact,
            split_manifest=source_artifact,
            candidate_artifact=candidate_artifact,
            augmented_split_manifest=source_artifact,
        ),
        generated_at=_GENERATED_AT,
    )


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
        dataset_version_id="dataset_version_1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


def _demo_schema_columns() -> list[str]:
    """Return the column header of the deterministic demo CSV."""
    from tests.fixtures.demo_archive import build_demo_archive

    built = build_demo_archive(output_dir=Path("/tmp/_dataforge_schema_probe"))
    with open_archive_path(built.archive_path) as reader:
        header = reader.find_required_transactions().read_bytes().splitlines()[0]
    return header.decode("utf-8").split(",")


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
