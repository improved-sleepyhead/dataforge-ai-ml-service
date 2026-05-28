"""Tests for TASK-053 export readiness gates and ExportPackage builder."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ArtifactLineage,
    ArtifactRef,
    BaselineModelConfig,
    CandidateActionStepSummary,
    CandidateArtifactStatus,
    CandidateDatasetVersion,
    CandidateDatasetVersionLineage,
    CandidatePolicyVersions,
    CandidateValidationGate,
    CandidateVersionStatus,
    ClassificationMetrics,
    ConfusionMatrixCell,
    ErrorCode,
    ExportPackage,
    ExportPackageStatus,
    GateStatus,
    MetricStatus,
    ModelImpactReport,
    ModelImpactReportLineage,
    ModelImpactVerdict,
    PiiCategory,
    PiiFinding,
    RedactionStatus,
    SyntheticUtilityStatus,
    TextDuplicateGroup,
    TextOcrReport,
    TextOcrSourceKind,
    TextOcrSourceReport,
    TextPiiFindingsForRecord,
    TextValidationIssue,
    TstrTrtsMetrics,
    ValidationGateSeverity,
    ValidationGatesLineage,
    ValidationGatesReport,
    ValidationGateStatus,
    ValidationGateType,
)
from app.kernel import (
    EXPORT_PACKAGE_KIND,
    BuildExportPackageRequest,
    build_export_package,
)
from app.kernel.export_package import (
    GATE_BLOCKED_OBJECTS_EXCLUDED,
    GATE_CANDIDATE_VALIDATION,
    GATE_MODEL_IMPACT,
    GATE_PRIVACY_CHECK,
    GATE_RAW_IMMUTABILITY,
    GATE_REQUIRED_ARTIFACTS,
    GATE_VALIDATION_GATES,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload

_CONFIG_HASH = "sha256:" + "a" * 64
_GENERATED_AT = datetime(2026, 5, 30, 12, 0, tzinfo=UTC)
_BASE_HASH = "sha256:" + "b" * 64
_CANDIDATE_HASH = "sha256:" + "c" * 64
_GATES_HASH = "sha256:" + "d" * 64
_MODEL_IMPACT_HASH = "sha256:" + "e" * 64
_DATASET_CARD_HASH = "sha256:" + "1" * 64
_LINEAGE_HASH = "sha256:" + "2" * 64
_DATAFORGE_REPORT_HASH = "sha256:" + "3" * 64
_REVIEW_QUEUE_HASH = "sha256:" + "4" * 64
_TABULAR_HASH = "sha256:" + "5" * 64
_TEXT_HASH = "sha256:" + "6" * 64
_MANIFEST_HASH = "sha256:" + "7" * 64


# ---------------------------------------------------------------------------
# Step 1: PII blocker -> EXPORT_BLOCKED
# ---------------------------------------------------------------------------


def test_export_blocked_when_pii_unredacted_and_block_export_flag() -> None:
    """Step 1: candidate with unredacted PII produces a BLOCKED export package."""
    storage, registry = _storage_and_registry()
    candidate = _candidate(
        block_export=True,
        blocker_reason_codes=("pii_unmasked",),
    )
    request = BuildExportPackageRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        validation_gates_report=_passing_gates_report(),
        text_ocr_report=_text_ocr_report_with_unredacted_pii(),
        decision_report_id="decision_report_001",
        export_manifest_artifact=_manifest_artifact(),
        tabular_artifacts=(_tabular_artifact(),),
        text_artifacts=(_text_artifact(),),
        dataset_card_artifact=_dataset_card_artifact(),
        lineage_artifact=_lineage_artifact(),
        dataforge_report_artifact=_dataforge_report_artifact(),
        review_queue_artifact=_review_queue_artifact(),
        included_object_count=100,
        blocked_object_count=2,
        excluded_object_count=0,
        created_by_job_id="compute_run_export_001",
        config_hash=_CONFIG_HASH,
        export_package_id="export_package_blocked_001",
        created_at=_GENERATED_AT,
    )
    result = build_export_package(request, registry=registry)
    package = result.export_package

    assert isinstance(package, ExportPackage)
    assert package.status is ExportPackageStatus.BLOCKED
    assert result.blocked is True
    # The candidate-validation gate failed with the candidate's
    # blocker_reason_codes, the privacy gate failed with pii_unmasked,
    # and blocked-objects-excluded gate failed because blocked_count > 0.
    gates_by_name = {gate.name: gate for gate in package.validation_gates}
    assert gates_by_name[GATE_CANDIDATE_VALIDATION].status is GateStatus.FAILED
    assert gates_by_name[GATE_PRIVACY_CHECK].status is GateStatus.FAILED
    assert "pii_unmasked" in gates_by_name[GATE_PRIVACY_CHECK].reason_codes
    assert (
        gates_by_name[GATE_BLOCKED_OBJECTS_EXCLUDED].status is GateStatus.FAILED
    )
    # Blocker reason codes include the candidate's reasons + privacy + blocked.
    blocker = set(package.blocked_reason_codes)
    assert "pii_unmasked" in blocker
    assert "blocked_objects_present" in blocker

    # Object counts: included must be zero on blocked exports so
    # downstream consumers can never accidentally import blocked rows.
    assert package.object_counts.included == 0
    assert package.object_counts.blocked == 2

    # Blocked package only carries the export-manifest artifact ref;
    # tabular/text outputs/dataset_card are NOT included.
    assert tuple(a.kind for a in package.artifacts) == ("EXPORT_MANIFEST",)

    # Persistence + contract validation
    assert result.package_artifact.artifact_kind == EXPORT_PACKAGE_KIND
    stored = storage.get(result.package_artifact.uri)
    assert stored.info.metadata["export-status"] == "BLOCKED"
    validate_contract_payload(
        load_contract_pack(),
        "export_package",
        package.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Step 2: redaction -> READY export
# ---------------------------------------------------------------------------


def test_redaction_unblocks_export_and_returns_ready_package() -> None:
    """Step 2: after redaction the export is READY and contains every artifact."""
    storage, registry = _storage_and_registry()
    candidate = _candidate(block_export=False, blocker_reason_codes=())
    request = BuildExportPackageRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        validation_gates_report=_passing_gates_report(),
        model_impact_report=_model_impact_report(verdict=ModelImpactVerdict.IMPROVED),
        text_ocr_report=_text_ocr_report_fully_redacted(),
        decision_report_id="decision_report_001",
        export_manifest_artifact=_manifest_artifact(),
        tabular_artifacts=(_tabular_artifact(),),
        text_artifacts=(_text_artifact(),),
        dataset_card_artifact=_dataset_card_artifact(),
        lineage_artifact=_lineage_artifact(),
        dataforge_report_artifact=_dataforge_report_artifact(),
        review_queue_artifact=_review_queue_artifact(),
        included_object_count=100,
        blocked_object_count=0,
        excluded_object_count=2,
        created_by_job_id="compute_run_export_002",
        config_hash=_CONFIG_HASH,
        export_package_id="export_package_ready_001",
        created_at=_GENERATED_AT,
    )
    result = build_export_package(request, registry=registry)
    package = result.export_package

    assert package.status is ExportPackageStatus.READY
    assert result.blocked is False
    assert result.blocker_reason_codes == ()
    assert package.blocked_reason_codes == ()

    # Privacy + blocked-objects gates pass; model impact gate passes.
    gates_by_name = {gate.name: gate for gate in package.validation_gates}
    assert gates_by_name[GATE_CANDIDATE_VALIDATION].status is GateStatus.PASSED
    assert gates_by_name[GATE_PRIVACY_CHECK].status is GateStatus.PASSED
    assert gates_by_name[GATE_BLOCKED_OBJECTS_EXCLUDED].status is GateStatus.PASSED
    assert gates_by_name[GATE_MODEL_IMPACT].status is GateStatus.PASSED
    assert gates_by_name[GATE_RAW_IMMUTABILITY].status is GateStatus.PASSED
    assert gates_by_name[GATE_VALIDATION_GATES].status is GateStatus.PASSED
    assert gates_by_name[GATE_REQUIRED_ARTIFACTS].status is GateStatus.PASSED

    # Object counts mirror the request when READY.
    assert package.object_counts.included == 100
    assert package.object_counts.blocked == 0
    assert package.object_counts.excluded == 2

    # Idempotency: re-running with the same content produces the same
    # bytes through the content-addressed registry.
    rerun = build_export_package(request, registry=registry)
    assert rerun.package_artifact.uri == result.package_artifact.uri
    assert (
        storage.get(rerun.package_artifact.uri).data
        == storage.get(result.package_artifact.uri).data
    )


# ---------------------------------------------------------------------------
# Step 3: ready package contains the expected artifact list
# ---------------------------------------------------------------------------


def test_ready_export_package_includes_required_artifact_list() -> None:
    """Step 3: READY export carries manifest + tabular + dataset_card + lineage + report + queue."""
    storage, registry = _storage_and_registry()
    candidate = _candidate(block_export=False, blocker_reason_codes=())
    request = BuildExportPackageRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        validation_gates_report=_passing_gates_report(),
        model_impact_report=_model_impact_report(verdict=ModelImpactVerdict.IMPROVED),
        text_ocr_report=_text_ocr_report_fully_redacted(),
        decision_report_id="decision_report_001",
        export_manifest_artifact=_manifest_artifact(),
        tabular_artifacts=(_tabular_artifact(),),
        text_artifacts=(_text_artifact(),),
        dataset_card_artifact=_dataset_card_artifact(),
        lineage_artifact=_lineage_artifact(),
        dataforge_report_artifact=_dataforge_report_artifact(),
        review_queue_artifact=_review_queue_artifact(),
        included_object_count=100,
        blocked_object_count=0,
        excluded_object_count=0,
        created_by_job_id="compute_run_export_003",
        config_hash=_CONFIG_HASH,
        export_package_id="export_package_ready_full",
        created_at=_GENERATED_AT,
    )
    result = build_export_package(request, registry=registry)
    package = result.export_package

    # Persisted package URIs must reflect every artifact ref the caller
    # registered. This proves the package can actually be promoted —
    # blocked objects are not in the artifact list.
    artifact_kinds = tuple(a.kind for a in package.artifacts)
    assert "EXPORT_MANIFEST" in artifact_kinds
    assert "candidate_tabular_dataset" in artifact_kinds
    assert "redacted_text_ocr" in artifact_kinds
    assert "DATASET_CARD" in artifact_kinds
    assert "LINEAGE_REPORT" in artifact_kinds
    assert "dataforge_report" in artifact_kinds
    assert "review_queue" in artifact_kinds

    # Lineage refs the parent (immutable raw) and candidate version,
    # plus the action_plan that produced the candidate.
    assert package.source_version_id == "dataset_version_v1"
    assert package.version_id == "dataset_version_v2_candidate"
    assert package.lineage.parent_version_id == "dataset_version_v1"
    assert package.lineage.action_plan_id == "action_plan_export_001"
    assert package.lineage.decision_report_id == "decision_report_001"
    assert package.lineage.config_hash == _CONFIG_HASH

    # Persistence metadata
    stored = storage.get(result.package_artifact.uri)
    assert stored.info.metadata["export-status"] == "READY"
    assert stored.info.metadata["version-id"] == "dataset_version_v2_candidate"

    validate_contract_payload(
        load_contract_pack(),
        "export_package",
        package.model_dump(mode="json"),
    )


def test_validation_gates_blocker_blocks_export() -> None:
    """A failing validation gate blocker_present is enough to BLOCK the export."""
    storage, registry = _storage_and_registry()
    candidate = _candidate(block_export=False, blocker_reason_codes=())
    failing_gates = _failing_gates_report(
        gate_type=ValidationGateType.BUSINESS_RULES,
        reason_code="business_rule_failure",
    )
    request = BuildExportPackageRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        validation_gates_report=failing_gates,
        text_ocr_report=_text_ocr_report_fully_redacted(),
        decision_report_id="decision_report_001",
        export_manifest_artifact=_manifest_artifact(),
        tabular_artifacts=(_tabular_artifact(),),
        dataset_card_artifact=_dataset_card_artifact(),
        lineage_artifact=_lineage_artifact(),
        included_object_count=100,
        created_by_job_id="compute_run_export_004",
        config_hash=_CONFIG_HASH,
        export_package_id="export_package_blocked_gates",
        created_at=_GENERATED_AT,
    )
    result = build_export_package(request, registry=registry)
    package = result.export_package
    assert package.status is ExportPackageStatus.BLOCKED
    assert "business_rules" in package.blocked_reason_codes
    gates_by_name = {gate.name: gate for gate in package.validation_gates}
    assert gates_by_name[GATE_VALIDATION_GATES].status is GateStatus.FAILED


def test_missing_required_artifact_blocks_export() -> None:
    """Missing dataset_card / lineage artifacts trigger required_artifacts gate failure."""
    storage, registry = _storage_and_registry()
    candidate = _candidate(block_export=False, blocker_reason_codes=())
    request = BuildExportPackageRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        validation_gates_report=_passing_gates_report(),
        decision_report_id="decision_report_001",
        export_manifest_artifact=_manifest_artifact(),
        tabular_artifacts=(_tabular_artifact(),),
        dataset_card_artifact=None,
        lineage_artifact=None,
        included_object_count=100,
        created_by_job_id="compute_run_export_005",
        config_hash=_CONFIG_HASH,
        created_at=_GENERATED_AT,
    )
    result = build_export_package(request, registry=registry)
    package = result.export_package
    assert package.status is ExportPackageStatus.BLOCKED
    assert "dataset_card_artifact_missing" in package.blocked_reason_codes
    assert "lineage_artifact_missing" in package.blocked_reason_codes


def test_model_impact_required_blocks_when_report_missing() -> None:
    """When require_model_impact_eligibility is set, missing report blocks export."""
    _, registry = _storage_and_registry()
    candidate = _candidate(block_export=False, blocker_reason_codes=())
    request = BuildExportPackageRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        validation_gates_report=_passing_gates_report(),
        decision_report_id="decision_report_001",
        export_manifest_artifact=_manifest_artifact(),
        tabular_artifacts=(_tabular_artifact(),),
        dataset_card_artifact=_dataset_card_artifact(),
        lineage_artifact=_lineage_artifact(),
        included_object_count=100,
        require_model_impact_eligibility=True,
        created_by_job_id="compute_run_export_006",
        config_hash=_CONFIG_HASH,
        created_at=_GENERATED_AT,
    )
    result = build_export_package(request, registry=registry)
    assert result.export_package.status is ExportPackageStatus.BLOCKED
    assert "model_impact_report_required" in result.blocker_reason_codes


def test_model_impact_rejected_blocks_export() -> None:
    """A REJECTED model-impact verdict must block the export."""
    _, registry = _storage_and_registry()
    candidate = _candidate(block_export=False, blocker_reason_codes=())
    request = BuildExportPackageRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        validation_gates_report=_passing_gates_report(),
        model_impact_report=_model_impact_report(verdict=ModelImpactVerdict.REJECTED),
        text_ocr_report=_text_ocr_report_fully_redacted(),
        decision_report_id="decision_report_001",
        export_manifest_artifact=_manifest_artifact(),
        tabular_artifacts=(_tabular_artifact(),),
        dataset_card_artifact=_dataset_card_artifact(),
        lineage_artifact=_lineage_artifact(),
        included_object_count=100,
        created_by_job_id="compute_run_export_007",
        config_hash=_CONFIG_HASH,
        created_at=_GENERATED_AT,
    )
    result = build_export_package(request, registry=registry)
    assert result.export_package.status is ExportPackageStatus.BLOCKED
    assert "model_impact_rejected" in result.blocker_reason_codes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _candidate(
    *,
    block_export: bool,
    blocker_reason_codes: tuple[str, ...],
) -> CandidateDatasetVersion:
    return CandidateDatasetVersion(
        candidate_version_id="candidate_dataset_version_test_export",
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
            action_plan_id="action_plan_export_001",
            decision_report_id="decision_report_001",
            created_by_job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            input_artifact_hashes=(_BASE_HASH,),
            output_artifact_hashes=(_CANDIDATE_HASH,),
        ),
        candidate_artifacts=(_candidate_artifact(),),
        primary_dataset_artifact=_candidate_artifact(),
        validation_gates_report=_validation_gates_artifact(),
        validation_gates_summary={
            "overall_status": "passed",
            "candidate_status": "ok",
            "raw_artifact_unchanged": True,
            "blocker_present": False,
        },
        block_export=block_export,
        block_model_evaluation=block_export,
        block_training=block_export,
        blocker_reason_codes=blocker_reason_codes,
        action_plan_steps=(
            CandidateActionStepSummary(
                step_id="impute_income_001",
                step_type="IMPUTE_MISSING_VALUES",
                method_id="group_median",
                plugin_id="dataforge.tabular",
                plugin_version="0.1.0",
                config_hash=_CONFIG_HASH,
                output_artifact_kind="candidate_tabular_dataset",
                random_seed=None,
            ),
        ),
        synthetic_metadata=None,
        proposed_at=_GENERATED_AT,
    )


def _passing_gates_report() -> ValidationGatesReport:
    return ValidationGatesReport(
        report_id="validation_gates_report_passed",
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
            ),
        ),
        lineage=ValidationGatesLineage(
            action_plan_id="action_plan_export_001",
            step_id="impute_income_001",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            source_artifact=_baseline_artifact(),
            candidate_artifact=_candidate_artifact(),
        ),
        generated_at=_GENERATED_AT,
    )


def _failing_gates_report(
    *, gate_type: ValidationGateType, reason_code: str
) -> ValidationGatesReport:
    return ValidationGatesReport(
        report_id="validation_gates_report_failed",
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        candidate_artifact_kind="candidate_tabular_dataset",
        policy_version="validation_gates_policy_v0",
        overall_status=ValidationGateStatus.FAILED,
        candidate_status=CandidateArtifactStatus.VALIDATION_FAILED,
        raw_artifact_unchanged=True,
        block_export=True,
        block_model_evaluation=True,
        block_training=True,
        blocker_present=True,
        blocker_gate_types=(gate_type,),
        gates=(
            CandidateValidationGate(
                gate_type=gate_type,
                status=ValidationGateStatus.FAILED,
                severity=ValidationGateSeverity.BLOCKER,
                reason_code=reason_code,
                block_action="BLOCK_EXPORT",
            ),
        ),
        lineage=ValidationGatesLineage(
            action_plan_id="action_plan_export_001",
            step_id="impute_income_001",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            source_artifact=_baseline_artifact(),
            candidate_artifact=_candidate_artifact(),
        ),
        generated_at=_GENERATED_AT,
    )


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
        update={"rare_class_recall": 0.55, "macro_f1": 0.70}
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
            created_by_job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
        ),
        generated_at=_GENERATED_AT,
    )


def _text_ocr_report_with_unredacted_pii() -> TextOcrReport:
    pack = load_contract_pack()
    payload = next(
        e.payload for e in pack.examples if e.name == "text_ocr_report.privacy"
    )
    report = TextOcrReport.model_validate(payload)
    return report.model_copy(update={"total_redacted_record_count": 0})


def _text_ocr_report_fully_redacted() -> TextOcrReport:
    pack = load_contract_pack()
    payload = next(
        e.payload for e in pack.examples if e.name == "text_ocr_report.privacy"
    )
    report = TextOcrReport.model_validate(payload)
    return report.model_copy(
        update={"total_redacted_record_count": report.total_pii_record_count}
    )


def _baseline_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="raw_transactions:" + "b" * 16,
        kind="raw_transactions",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v1/transactions.csv",
        hash=_BASE_HASH,
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


def _candidate_artifact() -> ArtifactRef:
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
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _validation_gates_artifact() -> ArtifactRef:
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
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _manifest_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="export_manifest:" + "7" * 16,
        kind="EXPORT_MANIFEST",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/manifest.jsonl",
        hash=_MANIFEST_HASH,
        media_type="application/jsonl",
        size_bytes=8192,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _tabular_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="candidate_tabular_dataset:" + "5" * 16,
        kind="candidate_tabular_dataset",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/train.parquet",
        hash=_TABULAR_HASH,
        media_type="application/x-parquet",
        size_bytes=4096,
        schema_version="tabular_dataset.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _text_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="redacted_text_ocr:" + "6" * 16,
        kind="redacted_text_ocr",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/text_redacted.jsonl",
        hash=_TEXT_HASH,
        media_type="application/jsonl",
        size_bytes=2048,
        schema_version="text_ocr_record.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _dataset_card_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="dataset_card:" + "1" * 16,
        kind="DATASET_CARD",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/dataset_card.md",
        hash=_DATASET_CARD_HASH,
        media_type="text/markdown",
        size_bytes=2048,
        schema_version="dataset_card.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _lineage_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="lineage_report:" + "2" * 16,
        kind="LINEAGE_REPORT",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/lineage.json",
        hash=_LINEAGE_HASH,
        media_type="application/json",
        size_bytes=2048,
        schema_version="lineage_report.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _dataforge_report_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="dataforge_report:" + "3" * 16,
        kind="dataforge_report",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/dataforge_report.json",
        hash=_DATAFORGE_REPORT_HASH,
        media_type="application/json",
        size_bytes=4096,
        schema_version="dataforge_report.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_export_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _review_queue_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="review_queue:" + "4" * 16,
        kind="review_queue",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/review_queue.jsonl",
        hash=_REVIEW_QUEUE_HASH,
        media_type="application/jsonl",
        size_bytes=1024,
        schema_version="review_queue.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_export_001",
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


# Silence unused-import warnings for typing helpers that the contract
# uses but the test does not reference directly.
_ = (
    PiiCategory,
    PiiFinding,
    RedactionStatus,
    TextDuplicateGroup,
    TextOcrSourceKind,
    TextOcrSourceReport,
    TextPiiFindingsForRecord,
    TextValidationIssue,
    pytest,
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
