"""Tests for TASK-056 dataset_card.md builder."""

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
    CriticalBlocker,
    DataForgeScore,
    DataForgeScorePenalty,
    DataModality,
    DatasetDecision,
    DatasetReadiness,
    DataSplit,
    DecisionAction,
    DecisionReport,
    ErrorCode,
    ExportObjectCounts,
    ExportPackage,
    ExportPackageLineage,
    ExportPackageStatus,
    GateStatus,
    MetricStatus,
    ModelImpactReport,
    ModelImpactReportLineage,
    ModelImpactVerdict,
    ObjectLevelDecision,
    PiiCategory,
    PiiFinding,
    PolicyVersions,
    ReadinessAssessment,
    SignalStatus,
    SplitAssignment,
    SplitClassDistribution,
    SplitManifest,
    SplitManifestLineage,
    SplitStrategy,
    SyntheticCandidateMetadata,
    SyntheticUtilityStatus,
    TextOcrReport,
    TextOcrSourceKind,
    TextOcrSourceReport,
    TextPiiFindingsForRecord,
    TstrTrtsMetrics,
    ValidationGateResult,
    ValidationGateSeverity,
    ValidationGatesLineage,
    ValidationGatesReport,
    ValidationGateStatus,
    ValidationGateType,
    WorkflowType,
)
from app.reports import (
    DATASET_CARD_ARTIFACT_FORMAT,
    DATASET_CARD_ARTIFACT_KIND,
    DATASET_CARD_MEDIA_TYPE,
    DATASET_CARD_SCHEMA_VERSION,
    BuildDatasetCardRequest,
    DatasetCardBuilderError,
    build_dataset_card_artifact,
    render_dataset_card,
    serialize_dataset_card,
)

_CONFIG_HASH = "sha256:" + "a" * 64
_GENERATED_AT = datetime(2026, 5, 31, 10, 0, tzinfo=UTC)
_BASE_HASH = "sha256:" + "b" * 64
_CANDIDATE_HASH = "sha256:" + "c" * 64
_GATES_HASH = "sha256:" + "d" * 64
_VALIDATION_REPORT_HASH = "sha256:" + "e" * 64
_SYNTHETIC_REPORT_HASH = "sha256:" + "1" * 64
_REVIEW_QUEUE_HASH = "sha256:" + "2" * 64


# ---------------------------------------------------------------------------
# Step 1: build a dataset card for a synthetic-augmented candidate
# ---------------------------------------------------------------------------


def test_dataset_card_includes_required_sections_and_synthetic_provenance() -> None:
    """Build a dataset_card with summary, splits, actions, lineage and synthetic provenance."""
    storage, registry = _storage_and_registry()
    candidate = _candidate(synthetic=True)
    request = BuildDatasetCardRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        decision_report=_decision_report(),
        validation_gates_report=_passing_gates_report(),
        model_impact_report=_model_impact_report(verdict=ModelImpactVerdict.IMPROVED),
        text_ocr_report=_text_ocr_report_redacted(),
        export_package=_ready_export_package(candidate=candidate),
        split_manifest=_split_manifest(),
        dataforge_score=_dataforge_score(),
        declared_modalities=(DataModality.TABULAR,),
        review_queue_artifacts=(_review_queue_artifact(),),
        privacy_policy_version="privacy_v0",
        export_policy_version="export_policy_v0",
        created_by_job_id="compute_run_card_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
        dataset_card_id="dataset_card_test_001",
        object_count=120,
    )

    result = build_dataset_card_artifact(request, registry=registry)

    # Persistence + artifact metadata
    assert result.artifact.artifact_kind == DATASET_CARD_ARTIFACT_KIND
    assert result.artifact.format == DATASET_CARD_ARTIFACT_FORMAT
    assert result.artifact.schema_version == DATASET_CARD_SCHEMA_VERSION
    stored = storage.get(result.artifact.uri)
    assert stored.info.content_type == DATASET_CARD_MEDIA_TYPE
    assert stored.info.metadata["dataset-card-id"] == "dataset_card_test_001"
    assert stored.info.metadata["version-id"] == "dataset_version_v2_candidate"
    assert stored.data == serialize_dataset_card(result.markdown)

    md = result.markdown

    # Step 1: dataset summary, task type, modalities, split info, actions
    assert "# Dataset Card" in md
    assert "## Dataset Summary" in md
    assert "dataset_1" in md
    assert "dataset_version_v2_candidate" in md
    assert "dataset_version_v1" in md
    # Decision report fields surface
    assert "READY_FOR_TRAINING_WITH_REVIEW_NOTES" in md
    assert "READY_WITH_WARNINGS" in md
    # DataForge score surfaces with raw + value
    assert "DataForge score" in md
    # Task & target
    assert "## Task Type and Target" in md
    assert "is_fraud" in md
    assert "LogisticRegression" in md
    # Modalities listed (tabular declared, text from TextOcrReport)
    assert "## Modalities" in md
    assert "- tabular" in md
    assert "- text" in md
    # Split distribution
    assert "## Object Counts and Split Distribution" in md
    assert "### Split Distribution" in md
    assert "train" in md
    # Action plan steps
    assert "## Applied Actions" in md
    assert "AUGMENT_RARE_CLASS" in md
    assert "smote" in md
    assert "dataforge.tabular" in md

    # Step 2: synthetic provenance is explicit
    assert "## Synthetic and Augmented Provenance" in md
    assert "Synthetic method" in md
    assert "smote" in md  # appears in action + synthetic table
    assert "Random seed" in md
    assert "42" in md
    assert "Generated row count" in md
    assert "Source split" in md
    assert "train" in md
    assert "Validation report" in md
    assert "Synthetic dataset report" in md

    # Privacy / export restrictions are explicit, even when no PII is unredacted
    assert "## Privacy and Export Restrictions" in md
    assert "Records with detected PII" in md
    assert "Records redacted" in md
    assert "Records with unredacted PII" in md
    assert "privacy_v0" in md
    assert "export_policy_v0" in md
    assert "Raw text and raw PII must not be present" in md
    assert "Blocked objects are excluded from the export" in md

    # Step 3: blockers/review notes section, validation gates, model impact, lineage refs
    assert "## Blockers and Review Notes" in md
    assert "No critical blockers reported." in md
    assert "## Validation Gates" in md
    assert "schema_validation" in md
    assert "candidate_validation_passed" in md  # export readiness gate row
    assert "## Model Impact Summary" in md
    assert "improved" in md
    assert "## Lineage References" in md
    assert "action_plan_card_001" in md
    assert "decision_report_card_001" in md
    assert _CONFIG_HASH in md
    assert "method_policy_v0" in md


# ---------------------------------------------------------------------------
# Step 2: when no synthetic metadata is attached the card states it explicitly
# ---------------------------------------------------------------------------


def test_dataset_card_states_no_synthetic_when_metadata_missing() -> None:
    """Card prints a stable 'No synthetic data' line when no synthetic rows were created."""
    _, registry = _storage_and_registry()
    candidate = _candidate(synthetic=False)
    request = BuildDatasetCardRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        decision_report=_decision_report(),
        validation_gates_report=_passing_gates_report(),
        text_ocr_report=_text_ocr_report_redacted(),
        privacy_policy_version="privacy_v0",
        export_policy_version="export_policy_v0",
        created_by_job_id="compute_run_card_002",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
        dataset_card_id="dataset_card_test_002",
        object_count=80,
    )

    result = build_dataset_card_artifact(request, registry=registry)
    md = result.markdown

    assert "## Synthetic and Augmented Provenance" in md
    assert "No synthetic data was generated for this candidate version." in md
    # The card must not contain the synthetic metadata table headers when
    # there is no synthetic provenance.
    assert "Generated row count" not in md
    assert "Synthetic policy" not in md


# ---------------------------------------------------------------------------
# Step 3: card never echoes raw PII tokens
# ---------------------------------------------------------------------------


def test_dataset_card_does_not_include_raw_pii() -> None:
    """Card never embeds raw PII tokens like emails or phone digits."""
    _, registry = _storage_and_registry()
    candidate = _candidate(synthetic=False)
    request = BuildDatasetCardRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        text_ocr_report=_text_ocr_report_redacted(),
        privacy_policy_version="privacy_v0",
        export_policy_version="export_policy_v0",
        created_by_job_id="compute_run_card_003",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
        dataset_card_id="dataset_card_test_003",
        object_count=80,
    )
    result = build_dataset_card_artifact(request, registry=registry)
    md = result.markdown

    # Belt-and-braces guard: serialize_dataset_card raises if any
    # forbidden raw PII token is present.
    serialize_dataset_card(md)

    forbidden_substrings = (
        "@example.com",
        "@gmail.com",
        "555-123",
        "555-",
    )
    for token in forbidden_substrings:
        assert token not in md, f"dataset_card.md must not embed raw PII token {token}"

    # The card carries privacy posture (counts) but never inline values.
    assert "Records with detected PII" in md


def test_serialize_dataset_card_rejects_raw_pii_tokens() -> None:
    """serialize_dataset_card must guard against accidental PII echo."""
    poisoned_markdown = (
        "# Dataset Card\n\n"
        "- fake email john.doe@bank.test\n"
        "- fake phone +1 415-555-0199\n"
        "- fake ssn 123-45-6789\n"
        "- fake token api_key=sk_test_placeholder\n"
    )
    with pytest.raises(DatasetCardBuilderError) as exc:
        serialize_dataset_card(poisoned_markdown)
    assert exc.value.reason_code == "dataset_card_contains_raw_pii_tokens"


def test_dataset_card_builder_rejects_raw_pii_before_persisting() -> None:
    """The persistence path must run the same PII guard as the serializer."""
    _, registry = _storage_and_registry()
    request = BuildDatasetCardRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=_candidate(synthetic=False),
        additional_review_notes=("Call reviewer at +1 415-555-0199",),
        privacy_policy_version="privacy_v0",
        export_policy_version="export_policy_v0",
        created_by_job_id="compute_run_card_pii_guard",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
        dataset_card_id="dataset_card_test_pii_guard",
        object_count=80,
    )

    with pytest.raises(DatasetCardBuilderError) as exc:
        build_dataset_card_artifact(request, registry=registry)
    assert exc.value.reason_code == "dataset_card_contains_raw_pii_tokens"


# ---------------------------------------------------------------------------
# Step 4: idempotent rendering and storage
# ---------------------------------------------------------------------------


def test_dataset_card_is_idempotent_for_same_inputs() -> None:
    """Re-running the builder with identical inputs is byte-identical."""
    storage, registry = _storage_and_registry()
    candidate = _candidate(synthetic=False)
    base_request = BuildDatasetCardRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        decision_report=_decision_report(),
        validation_gates_report=_passing_gates_report(),
        text_ocr_report=_text_ocr_report_redacted(),
        privacy_policy_version="privacy_v0",
        export_policy_version="export_policy_v0",
        created_by_job_id="compute_run_card_004",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
        dataset_card_id="dataset_card_test_004",
        object_count=80,
    )

    first = build_dataset_card_artifact(base_request, registry=registry)
    # Same render through the pure function returns the same Markdown.
    assert render_dataset_card(base_request) == first.markdown
    second = build_dataset_card_artifact(base_request, registry=registry)
    assert first.artifact.uri == second.artifact.uri
    assert first.artifact.hash == second.artifact.hash
    assert (
        storage.get(first.artifact.uri).data == storage.get(second.artifact.uri).data
    )


# ---------------------------------------------------------------------------
# Step 5: blockers section lists candidate + decision + export blockers
# ---------------------------------------------------------------------------


def test_dataset_card_blockers_section_lists_critical_and_export_blockers() -> None:
    """When PII/blocker issues exist, dataset_card surfaces them with reason codes."""
    _, registry = _storage_and_registry()
    candidate = _candidate(synthetic=False, block_export=True, blocker_codes=("pii_unmasked",))
    decision_report = _decision_report().model_copy(
        update={
            "critical_blockers": (
                CriticalBlocker(
                    code="pii_unmasked",
                    severity="critical",
                    message="Detected unredacted PII tokens; redaction required.",
                ),
            ),
            "dataset_decision": DatasetDecision.BLOCKED,
            "readiness": ReadinessAssessment(
                status=DatasetReadiness.BLOCKED,
                score=0.20,
                reason_codes=("pii_unmasked",),
            ),
        }
    )
    blocked_export = _ready_export_package(candidate=candidate).model_copy(
        update={
            "status": ExportPackageStatus.BLOCKED,
            "blocked_reason_codes": ("pii_unmasked", "blocked_objects_present"),
            "object_counts": ExportObjectCounts(included=0, blocked=2, excluded=0),
        }
    )
    request = BuildDatasetCardRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        decision_report=decision_report,
        export_package=blocked_export,
        text_ocr_report=_text_ocr_report_with_unredacted_pii(),
        privacy_policy_version="privacy_v0",
        export_policy_version="export_policy_v0",
        created_by_job_id="compute_run_card_005",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
        dataset_card_id="dataset_card_test_005",
        object_count=80,
    )
    result = build_dataset_card_artifact(request, registry=registry)
    md = result.markdown

    assert "## Blockers and Review Notes" in md
    assert "`pii_unmasked`" in md
    assert "`blocked_objects_present`" in md
    assert "Export package status at card generation: `BLOCKED`" in md
    assert "export blocker: `pii_unmasked`" in md
    # The privacy section reports the unredacted PII count explicitly.
    assert "Records with unredacted PII" in md


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _candidate(
    *,
    synthetic: bool,
    block_export: bool = False,
    blocker_codes: tuple[str, ...] = (),
) -> CandidateDatasetVersion:
    steps: list[CandidateActionStepSummary] = [
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
    ]
    synthetic_metadata: SyntheticCandidateMetadata | None = None
    if synthetic:
        steps.append(
            CandidateActionStepSummary(
                step_id="augment_rare_001",
                step_type="AUGMENT_RARE_CLASS",
                method_id="smote",
                plugin_id="dataforge.tabular",
                plugin_version="0.1.0",
                config_hash=_CONFIG_HASH,
                output_artifact_kind="candidate_tabular_dataset",
                random_seed=42,
            )
        )
        synthetic_metadata = SyntheticCandidateMetadata(
            method_id="smote",
            plugin_id="dataforge.tabular",
            plugin_version="0.1.0",
            random_seed=42,
            config_hash=_CONFIG_HASH,
            source_split="train",
            source_cohort=None,
            source_object_ids=("txn_001", "txn_002"),
            source_object_ids_truncated=False,
            full_source_object_ids_count=2,
            generated_count=20,
            sampling_strategy=0.5,
            policy_version="synthetic_policy_v0",
            validation_report=_synthetic_validation_artifact(),
            model_impact_report=None,
            synthetic_dataset_report=_synthetic_dataset_artifact(),
        )

    return CandidateDatasetVersion(
        candidate_version_id="candidate_dataset_version_card_001",
        status=(
            CandidateVersionStatus.BLOCKED
            if block_export
            else CandidateVersionStatus.PROPOSED
        ),
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
            action_plan_id="action_plan_card_001",
            decision_report_id="decision_report_card_001",
            created_by_job_id="compute_run_card_001",
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
        blocker_reason_codes=blocker_codes,
        action_plan_steps=tuple(steps),
        synthetic_metadata=synthetic_metadata,
        proposed_at=_GENERATED_AT,
    )


def _decision_report() -> DecisionReport:
    return DecisionReport(
        decision_report_id="decision_report_card_001",
        dataset_id="dataset_1",
        version_id="dataset_version_v2_candidate",
        decision_schema_version="decision_report.v1",
        dataset_decision=DatasetDecision.READY_FOR_TRAINING_WITH_REVIEW_NOTES,
        readiness=ReadinessAssessment(
            status=DatasetReadiness.READY_WITH_WARNINGS,
            score=0.74,
            reason_codes=("review_pending",),
        ),
        critical_blockers=(),
        safe_actions_available=True,
        recommended_next_job=None,
        object_decisions=(
            ObjectLevelDecision(
                object_id="txn_001",
                modality=DataModality.TABULAR,
                action=DecisionAction.KEEP,
                reasons=("rare_class_candidate",),
                blocked_actions=(),
                object_value_score=0.75,
            ),
        ),
        recommended_actions=(),
        policy_versions=PolicyVersions(
            decision_policy="decision_policy_v0",
            score_policy="dataforge_score_v0",
            privacy_policy="privacy_v0",
            method_policy="method_policy_v0",
        ),
        generated_at=_GENERATED_AT,
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
            action_plan_id="action_plan_card_001",
            step_id="impute_income_001",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_card_001",
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
    config = BaselineModelConfig(
        algorithm="LogisticRegression",
        library="scikit-learn",
        library_version="1.5.0",
        hyperparameters={"max_iter": 500, "solver": "lbfgs"},
        feature_columns=("amount", "monthly_income"),
        target_column="is_fraud",
        excluded_columns=("object_id",),
        random_seed=42,
    )
    return ModelImpactReport(
        report_id="model_impact_report_card",
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
        baseline_model_config=config,
        candidate_model_config=config,
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
            created_by_job_id="compute_run_card_001",
            config_hash=_CONFIG_HASH,
        ),
        generated_at=_GENERATED_AT,
    )


def _split_manifest() -> SplitManifest:
    return SplitManifest(
        split_manifest_id="split_manifest_card_001",
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        target_column="is_fraud",
        strategy=SplitStrategy.GROUP_STRATIFIED,
        seed=42,
        group_key="customer_id_hash",
        policy_version="split_policy_v0",
        split_ratios={
            DataSplit.TRAIN: 0.70,
            DataSplit.VALIDATION: 0.15,
            DataSplit.TEST: 0.15,
        },
        assignments=(
            SplitAssignment(object_id="txn_001", split=DataSplit.TRAIN, label="0"),
            SplitAssignment(object_id="txn_002", split=DataSplit.TRAIN, label="1"),
            SplitAssignment(
                object_id="txn_010", split=DataSplit.VALIDATION, label="0"
            ),
            SplitAssignment(object_id="txn_020", split=DataSplit.TEST, label="1"),
        ),
        class_distribution=(
            SplitClassDistribution(
                split=DataSplit.TRAIN,
                total_count=70,
                class_counts={"0": 65, "1": 5},
                class_ratios={"0": 0.928571, "1": 0.071429},
            ),
            SplitClassDistribution(
                split=DataSplit.VALIDATION,
                total_count=15,
                class_counts={"0": 14, "1": 1},
                class_ratios={"0": 0.933333, "1": 0.066667},
            ),
            SplitClassDistribution(
                split=DataSplit.TEST,
                total_count=15,
                class_counts={"0": 14, "1": 1},
                class_ratios={"0": 0.933333, "1": 0.066667},
            ),
        ),
        lineage=SplitManifestLineage(
            action_plan_id="action_plan_card_001",
            step_id="split_001",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            created_by_job_id="compute_run_card_001",
            config_hash=_CONFIG_HASH,
            source_artifact=_baseline_artifact(),
        ),
        generated_at=_GENERATED_AT,
    )


def _dataforge_score() -> DataForgeScore:
    return DataForgeScore(
        value=0.74,
        raw_score=78.0,
        policy_version="dataforge_score_v0",
        formula="DataForgeScore = 100 * weighted_components - penalties",
        weights={"completeness": 0.15, "validity": 0.15, "uniqueness": 0.12},
        components={"completeness": 0.92, "validity": 0.85, "uniqueness": 0.95},
        weighted_components={
            "completeness": 13.8,
            "validity": 12.75,
            "uniqueness": 11.4,
        },
        penalties=(
            DataForgeScorePenalty(
                reason_code="severe_class_imbalance", value=-10.0, applied=True
            ),
        ),
        hard_blocked=False,
        readiness_status=DatasetReadiness.READY_WITH_WARNINGS,
        reason_codes=("rare_fraud_class",),
    )


def _ready_export_package(candidate: CandidateDatasetVersion) -> ExportPackage:
    return ExportPackage(
        export_package_id="export_package_card_001",
        export_schema_version="export_package.v1",
        dataset_id="dataset_1",
        version_id=candidate.lineage.proposed_version_name,
        source_version_id=candidate.lineage.parent_version_id,
        created_by_job_id="compute_run_card_001",
        status=ExportPackageStatus.READY,
        artifacts=(_export_manifest_artifact(),),
        validation_gates=(
            ValidationGateResult(
                name="candidate_validation_passed",
                status=GateStatus.PASSED,
                reason_codes=(),
            ),
            ValidationGateResult(
                name="privacy_check",
                status=GateStatus.PASSED,
                reason_codes=(),
            ),
        ),
        object_counts=ExportObjectCounts(included=100, blocked=0, excluded=0),
        blocked_reason_codes=(),
        lineage=ExportPackageLineage(
            parent_version_id=candidate.lineage.parent_version_id,
            action_plan_id=candidate.lineage.action_plan_id,
            decision_report_id="decision_report_card_001",
            config_hash=_CONFIG_HASH,
        ),
        created_at=_GENERATED_AT,
    )


def _text_ocr_report_redacted() -> TextOcrReport:
    return TextOcrReport(
        report_id="text_ocr_report_card_redacted",
        dataset_id="dataset_1",
        version_id="dataset_version_v2_candidate",
        parent_version_id="dataset_version_v1",
        created_by_job_id="compute_run_card_001",
        config_hash=_CONFIG_HASH,
        sources=(
            TextOcrSourceReport(
                source_kind=TextOcrSourceKind.SUPPORT_MESSAGES,
                source_name="support_messages.jsonl",
                record_count=10,
                valid_record_count=10,
                issue_count=0,
                issues=(),
                duplicate_groups=(),
                duplicate_record_count=0,
                duplicate_object_ids=(),
                average_text_length=120.0,
                min_text_length=15,
                max_text_length=400,
                average_ocr_confidence=None,
                pii_findings=(
                    TextPiiFindingsForRecord(
                        object_id="msg_001",
                        findings=(
                            PiiFinding(category=PiiCategory.EMAIL, occurrence_count=1),
                        ),
                        pii_token_count=1,
                        pii_risk_score=0.6,
                        redacted_text_sha256="sha256:" + "a" * 64,
                    ),
                ),
                pii_token_count=1,
                pii_record_count=1,
                redacted_record_count=1,
                redacted_artifact_uri=None,
                redacted_artifact_hash=None,
                review_queue_object_ids=("msg_001",),
            ),
        ),
        total_record_count=10,
        total_valid_record_count=10,
        total_issue_count=0,
        total_duplicate_group_count=0,
        total_duplicate_record_count=0,
        total_pii_record_count=1,
        total_pii_token_count=1,
        total_redacted_record_count=1,
        review_queue_object_ids=("msg_001",),
        generated_at=_GENERATED_AT,
    )


def _text_ocr_report_with_unredacted_pii() -> TextOcrReport:
    base = _text_ocr_report_redacted()
    return base.model_copy(update={"total_redacted_record_count": 0})


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
            job_id="compute_run_card_001",
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
            job_id="compute_run_card_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _synthetic_validation_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="synthetic_validation_report:" + "e" * 16,
        kind="synthetic_validation_report",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/synthetic_validation.json",
        hash=_VALIDATION_REPORT_HASH,
        media_type="application/json",
        size_bytes=1024,
        schema_version="synthetic_validation_report.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_card_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _synthetic_dataset_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="synthetic_dataset_report:" + "1" * 16,
        kind="synthetic_dataset_report",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/synthetic_dataset.json",
        hash=_SYNTHETIC_REPORT_HASH,
        media_type="application/json",
        size_bytes=1024,
        schema_version="synthetic_dataset_report.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_card_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _export_manifest_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="export_manifest:" + "7" * 16,
        kind="EXPORT_MANIFEST",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/manifest.jsonl",
        hash="sha256:" + "7" * 64,
        media_type="application/jsonl",
        size_bytes=1024,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_card_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _review_queue_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="review_queue:" + "2" * 16,
        kind="review_queue",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/review_queue.jsonl",
        hash=_REVIEW_QUEUE_HASH,
        media_type="application/jsonl",
        size_bytes=1024,
        schema_version="review_queue.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_card_001",
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


# Silence unused-import warnings for typing helpers reused by future tests.
_ = (SignalStatus, WorkflowType, pytest)


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
