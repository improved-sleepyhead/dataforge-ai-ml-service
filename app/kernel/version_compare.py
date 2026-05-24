"""Builder for the candidate Version Compare artifact (TASK-052).

The builder consumes only normalized contract reports — never raw row
content or PII — and assembles the before/after diff envelope that the
platform exposes through ``GET /api/v1/dataset-versions/{base}/compare/
{candidate}`` (PRD §27.2.3).

Inputs:

* ``CandidateDatasetVersion`` — supplies action_plan_id, validation
  gates summary, lineage, optional synthetic metadata.
* baseline / candidate ``TabularProfileReport`` (optional) — drives
  object counts, duplicates, class balance.
* baseline / candidate ``TextOcrReport`` (optional) — drives PII risk.
* baseline / candidate ``DataForgeScore`` (optional) — drives score
  decomposition diff.
* ``ModelImpactReport`` (optional) — drives the model-metrics block,
  produced when model impact is eligible.
* ``TabularImputationReport`` (optional) — supplies imputed_fields and
  the imputed-object count.
* ``DuplicateActionReport`` (optional) — supplies duplicate marked /
  removed counts.
* ``SyntheticDatasetReport`` (optional) — supplies synthetic_added.

The builder persists the artifact through ``ArtifactRegistry`` as
``version_compare_report.v1`` JSON. It is contract-first, idempotent
on identical inputs (content-addressed registry path), and never
mutates source artifacts.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    VERSION_COMPARE_REPORT_SCHEMA_VERSION,
    CandidateDatasetVersion,
    ChangedObjectsBlock,
    ClassBalanceClassEntry,
    ClassBalanceDiff,
    CompareSignalStatus,
    DataForgeScore,
    DuplicateActionMode,
    DuplicateActionReport,
    DuplicateCountDiff,
    ImputedFieldEntry,
    ModelImpactReport,
    ModelMetricsCompare,
    ObjectCountDiff,
    PiiRiskDiff,
    ScoreComponentDiff,
    ScoreDiff,
    ScorePenaltyDiff,
    SyntheticDatasetReport,
    TabularImputationReport,
    TabularProfileReport,
    TextOcrReport,
    ValidationGatesReport,
    ValidationGatesSummary,
    VersionCompareLineage,
    VersionCompareReport,
)
from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Sha256Digest

VERSION_COMPARE_REPORT_KIND = "version_compare_report"
VERSION_COMPARE_REPORT_FORMAT = "json"
VERSION_COMPARE_REPORT_MEDIA_TYPE = "application/json"


class VersionCompareBuilderError(ValueError):
    """Raised when the version-compare builder cannot assemble safely."""

    def __init__(self, *, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class BuildVersionCompareRequest(BaseModel):
    """Inputs for :func:`build_version_compare_report`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    base_version_id: NonEmptyStr
    candidate_version_id: NonEmptyStr
    candidate_dataset_version: CandidateDatasetVersion
    candidate_version_artifact: ArtifactRef | None = None
    baseline_tabular_profile: TabularProfileReport | None = None
    candidate_tabular_profile: TabularProfileReport | None = None
    baseline_text_ocr_report: TextOcrReport | None = None
    candidate_text_ocr_report: TextOcrReport | None = None
    baseline_score: DataForgeScore | None = None
    candidate_score: DataForgeScore | None = None
    model_impact_report: ModelImpactReport | None = None
    model_impact_report_artifact: ArtifactRef | None = None
    imputation_report: TabularImputationReport | None = None
    duplicate_action_report: DuplicateActionReport | None = None
    synthetic_dataset_report: SyntheticDatasetReport | None = None
    validation_gates_report: ValidationGatesReport | None = None
    validation_gates_report_artifact: ArtifactRef | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class BuildVersionCompareResult:
    """Persisted Version Compare artifact + parsed report."""

    report: VersionCompareReport
    report_artifact: RegisteredArtifact


def build_version_compare_report(
    request: BuildVersionCompareRequest,
    *,
    registry: ArtifactRegistry,
) -> BuildVersionCompareResult:
    """Assemble + persist a Version Compare artifact.

    The builder consumes only normalized contract reports. It returns
    a ``VersionCompareReport`` and the ``RegisteredArtifact`` record
    for the persisted JSON artifact.
    """
    object_counts = _build_object_counts(
        request.baseline_tabular_profile,
        request.candidate_tabular_profile,
    )
    changed_objects = _build_changed_objects(
        imputation=request.imputation_report,
        duplicates=request.duplicate_action_report,
        synthetic=request.synthetic_dataset_report,
        text_ocr_before=request.baseline_text_ocr_report,
        text_ocr_after=request.candidate_text_ocr_report,
        object_counts=object_counts,
    )
    imputed_fields = _build_imputed_fields(request.imputation_report)
    pii_risk = _build_pii_risk(
        request.baseline_text_ocr_report,
        request.candidate_text_ocr_report,
    )
    duplicates = _build_duplicate_diff(
        request.baseline_tabular_profile,
        request.candidate_tabular_profile,
    )
    class_balance = _build_class_balance(
        request.baseline_tabular_profile,
        request.candidate_tabular_profile,
    )
    model_metrics = _build_model_metrics(
        request.model_impact_report,
        request.model_impact_report_artifact,
    )
    score_diff = _build_score_diff(request.baseline_score, request.candidate_score)
    gates_summary = _build_validation_gates_summary(
        candidate=request.candidate_dataset_version,
        gates_report=request.validation_gates_report,
        gates_artifact=request.validation_gates_report_artifact,
    )

    lineage = VersionCompareLineage(
        organization_id=request.organization_id,
        project_id=request.project_id,
        dataset_id=request.dataset_id,
        parent_version_id=request.base_version_id,
        candidate_dataset_version_id=request.candidate_version_id,
        candidate_version_artifact=request.candidate_version_artifact,
        action_plan_id=request.candidate_dataset_version.lineage.action_plan_id,
        decision_report_id=request.candidate_dataset_version.lineage.decision_report_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
    )

    report_id = request.report_id or f"version_compare_{uuid.uuid4().hex[:16]}"
    report = VersionCompareReport(
        report_id=report_id,
        report_schema_version=VERSION_COMPARE_REPORT_SCHEMA_VERSION,
        base_version_id=request.base_version_id,
        candidate_version_id=request.candidate_version_id,
        object_counts=object_counts,
        changed_objects=changed_objects,
        imputed_fields=imputed_fields,
        pii_risk=pii_risk,
        duplicate_counts=duplicates,
        class_balance=class_balance,
        model_metrics=model_metrics,
        score=score_diff,
        validation_gates=gates_summary,
        action_plan_id=request.candidate_dataset_version.lineage.action_plan_id,
        lineage=lineage,
        generated_at=request.generated_at or datetime.now(UTC),
    )
    payload = json.dumps(
        report.model_dump(mode="json"), sort_keys=True, indent=2
    ).encode("utf-8")

    metadata = {
        "report-id": report_id,
        "base-version-id": request.base_version_id,
        "candidate-version-id": request.candidate_version_id,
        "action-plan-id": report.action_plan_id,
        "model-metrics-status": model_metrics.status.value,
        "score-status": score_diff.status.value,
    }
    artifact = registry.save_artifact(
        artifact_kind=VERSION_COMPARE_REPORT_KIND,
        data=payload,
        artifact_format=VERSION_COMPARE_REPORT_FORMAT,
        media_type=VERSION_COMPARE_REPORT_MEDIA_TYPE,
        schema_version=VERSION_COMPARE_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata=metadata,
    )
    return BuildVersionCompareResult(report=report, report_artifact=artifact)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_object_counts(
    baseline: TabularProfileReport | None,
    candidate: TabularProfileReport | None,
) -> ObjectCountDiff:
    before = baseline.row_count if baseline is not None else 0
    after = candidate.row_count if candidate is not None else before
    return ObjectCountDiff(before=before, after=after, delta=after - before)


def _build_changed_objects(
    *,
    imputation: TabularImputationReport | None,
    duplicates: DuplicateActionReport | None,
    synthetic: SyntheticDatasetReport | None,
    text_ocr_before: TextOcrReport | None,
    text_ocr_after: TextOcrReport | None,
    object_counts: ObjectCountDiff,
) -> ChangedObjectsBlock:
    imputed_object_count = (
        sum(column.imputed_count for column in imputation.columns)
        if imputation is not None
        else 0
    )
    duplicate_marked = 0
    duplicate_removed = 0
    if duplicates is not None:
        if duplicates.mode is DuplicateActionMode.MARK:
            duplicate_marked = duplicates.marked_count
        if duplicates.mode is DuplicateActionMode.REMOVE_CANDIDATE:
            duplicate_removed = duplicates.removed_count
    synthetic_added = synthetic.generated_count if synthetic is not None else 0
    redacted_object_count = _redacted_record_delta(text_ocr_before, text_ocr_after)
    # If candidate has fewer rows than baseline beyond explicit removed
    # duplicates, surface the gap as ``removed_or_blocked``.
    removed_or_blocked = max(
        0,
        (-object_counts.delta if synthetic_added == 0 else 0)
        - duplicate_removed,
    )
    if duplicate_removed > 0 and object_counts.delta < 0:
        removed_or_blocked = max(removed_or_blocked, -object_counts.delta - duplicate_removed)
    changed_total = (
        imputed_object_count
        + duplicate_marked
        + duplicate_removed
        + redacted_object_count
        + synthetic_added
        + removed_or_blocked
    )
    return ChangedObjectsBlock(
        changed_objects_total=changed_total,
        imputed_object_count=imputed_object_count,
        duplicate_marked_count=duplicate_marked,
        duplicate_removed_count=duplicate_removed,
        redacted_object_count=redacted_object_count,
        synthetic_added_count=synthetic_added,
        removed_or_blocked_count=removed_or_blocked,
    )


def _redacted_record_delta(
    before: TextOcrReport | None,
    after: TextOcrReport | None,
) -> int:
    if before is None or after is None:
        return 0
    return max(0, after.total_redacted_record_count - before.total_redacted_record_count)


def _build_imputed_fields(
    imputation: TabularImputationReport | None,
) -> tuple[ImputedFieldEntry, ...]:
    if imputation is None:
        return ()
    return tuple(
        ImputedFieldEntry(
            column=column.column,
            method=column.method.value,
            before_missing_count=column.before_missing_count,
            after_missing_count=column.after_missing_count,
            imputed_count=column.imputed_count,
        )
        for column in imputation.columns
        if column.imputed_count > 0 or column.before_missing_count > 0
    )


def _build_pii_risk(
    before: TextOcrReport | None,
    after: TextOcrReport | None,
) -> PiiRiskDiff:
    if before is None and after is None:
        return PiiRiskDiff(
            status=CompareSignalStatus.NOT_APPLICABLE,
            not_applicable_reason="text_ocr_report_not_provided",
        )
    pii_before = before.total_pii_record_count if before is not None else 0
    pii_after = after.total_pii_record_count if after is not None else 0
    redacted_before = before.total_redacted_record_count if before is not None else 0
    redacted_after = after.total_redacted_record_count if after is not None else 0
    unredacted_before = max(0, pii_before - redacted_before)
    unredacted_after = max(0, pii_after - redacted_after)
    return PiiRiskDiff(
        status=CompareSignalStatus.AVAILABLE,
        not_applicable_reason=None,
        pii_record_count_before=pii_before,
        pii_record_count_after=pii_after,
        pii_record_count_delta=pii_after - pii_before,
        redacted_record_count_before=redacted_before,
        redacted_record_count_after=redacted_after,
        unredacted_pii_record_count_before=unredacted_before,
        unredacted_pii_record_count_after=unredacted_after,
    )


def _build_duplicate_diff(
    baseline: TabularProfileReport | None,
    candidate: TabularProfileReport | None,
) -> DuplicateCountDiff:
    if baseline is None and candidate is None:
        return DuplicateCountDiff(
            status=CompareSignalStatus.NOT_APPLICABLE,
            not_applicable_reason="tabular_profile_not_provided",
        )
    before = (
        baseline.duplicates.duplicate_pair_count
        if baseline is not None and baseline.duplicates is not None
        else 0
    )
    after = (
        candidate.duplicates.duplicate_pair_count
        if candidate is not None and candidate.duplicates is not None
        else 0
    )
    return DuplicateCountDiff(
        status=CompareSignalStatus.AVAILABLE,
        duplicate_pair_count_before=before,
        duplicate_pair_count_after=after,
        duplicate_pair_count_delta=after - before,
    )


def _build_class_balance(
    baseline: TabularProfileReport | None,
    candidate: TabularProfileReport | None,
) -> ClassBalanceDiff:
    baseline_imbalance = (
        baseline.class_imbalance if baseline is not None else None
    )
    candidate_imbalance = (
        candidate.class_imbalance if candidate is not None else None
    )
    if baseline_imbalance is None and candidate_imbalance is None:
        return ClassBalanceDiff(
            status=CompareSignalStatus.NOT_APPLICABLE,
            not_applicable_reason="class_imbalance_not_provided",
        )
    target_column = (
        baseline_imbalance.target_column
        if baseline_imbalance is not None
        else (candidate_imbalance.target_column if candidate_imbalance is not None else None)
    )
    rare_label = (
        baseline_imbalance.rare_class_label
        if baseline_imbalance is not None
        else (
            candidate_imbalance.rare_class_label
            if candidate_imbalance is not None
            else None
        )
    )
    counts_before: dict[str, int] = (
        {entry.label: entry.count for entry in baseline_imbalance.class_counts}
        if baseline_imbalance is not None
        else {}
    )
    counts_after: dict[str, int] = (
        {entry.label: entry.count for entry in candidate_imbalance.class_counts}
        if candidate_imbalance is not None
        else {}
    )
    labels = sorted(set(counts_before) | set(counts_after))
    classes = tuple(
        ClassBalanceClassEntry(
            label=label,
            count_before=counts_before.get(label, 0),
            count_after=counts_after.get(label, 0),
        )
        for label in labels
    )

    rare_count_before = (
        baseline_imbalance.rare_class_count if baseline_imbalance is not None else None
    )
    rare_count_after = (
        candidate_imbalance.rare_class_count if candidate_imbalance is not None else None
    )
    rare_ratio_before = (
        baseline_imbalance.rare_class_ratio if baseline_imbalance is not None else None
    )
    rare_ratio_after = (
        candidate_imbalance.rare_class_ratio if candidate_imbalance is not None else None
    )
    rare_ratio_delta = (
        rare_ratio_after - rare_ratio_before
        if rare_ratio_before is not None and rare_ratio_after is not None
        else None
    )
    imbalance_before = (
        baseline_imbalance.imbalance_ratio if baseline_imbalance is not None else None
    )
    imbalance_after = (
        candidate_imbalance.imbalance_ratio if candidate_imbalance is not None else None
    )
    imbalance_delta = (
        imbalance_after - imbalance_before
        if imbalance_before is not None and imbalance_after is not None
        else None
    )
    minority_before = (
        baseline_imbalance.minority_class_share if baseline_imbalance is not None else None
    )
    minority_after = (
        candidate_imbalance.minority_class_share if candidate_imbalance is not None else None
    )
    return ClassBalanceDiff(
        status=CompareSignalStatus.AVAILABLE,
        target_column=target_column,
        rare_class_label=rare_label,
        rare_class_count_before=rare_count_before,
        rare_class_count_after=rare_count_after,
        rare_class_ratio_before=rare_ratio_before,
        rare_class_ratio_after=rare_ratio_after,
        rare_class_ratio_delta=rare_ratio_delta,
        imbalance_ratio_before=imbalance_before,
        imbalance_ratio_after=imbalance_after,
        imbalance_ratio_delta=imbalance_delta,
        minority_share_before=minority_before,
        minority_share_after=minority_after,
        classes=classes,
    )


def _build_model_metrics(
    report: ModelImpactReport | None,
    artifact: ArtifactRef | None,
) -> ModelMetricsCompare:
    if report is None:
        return ModelMetricsCompare(
            status=CompareSignalStatus.NOT_APPLICABLE,
            not_applicable_reason="model_impact_report_not_provided",
        )
    pr_status = (
        CompareSignalStatus.AVAILABLE
        if report.pr_auc_status.value == "available"
        else CompareSignalStatus.NOT_APPLICABLE
    )
    return ModelMetricsCompare(
        status=CompareSignalStatus.AVAILABLE,
        verdict=report.verdict.value,
        rare_class_recall_before=report.rare_class_recall_before,
        rare_class_recall_after=report.rare_class_recall_after,
        rare_class_recall_delta=report.rare_class_recall_delta,
        macro_f1_before=report.macro_f1_before,
        macro_f1_after=report.macro_f1_after,
        macro_f1_delta=report.macro_f1_delta,
        weighted_f1_before=report.weighted_f1_before,
        weighted_f1_after=report.weighted_f1_after,
        weighted_f1_delta=report.weighted_f1_delta,
        pr_auc_before=report.pr_auc_before,
        pr_auc_after=report.pr_auc_after,
        pr_auc_delta=report.pr_auc_delta,
        pr_auc_status=pr_status,
        pr_auc_reason=report.pr_auc_reason,
        model_impact_report=artifact,
    )


def _build_score_diff(
    baseline: DataForgeScore | None,
    candidate: DataForgeScore | None,
) -> ScoreDiff:
    if baseline is None and candidate is None:
        return ScoreDiff(
            status=CompareSignalStatus.NOT_APPLICABLE,
            not_applicable_reason="dataforge_score_not_provided",
        )
    if baseline is None or candidate is None:
        return ScoreDiff(
            status=CompareSignalStatus.NOT_APPLICABLE,
            not_applicable_reason="dataforge_score_only_one_side_provided",
        )
    components_before = baseline.components
    components_after = candidate.components
    weighted_before = baseline.weighted_components
    weighted_after = candidate.weighted_components
    weights = baseline.weights
    component_names = sorted(set(components_before) | set(components_after))
    components: list[ScoreComponentDiff] = []
    for name in component_names:
        value_before = components_before.get(name, 0.0)
        value_after = components_after.get(name, value_before)
        weight = weights.get(name, candidate.weights.get(name, 0.0))
        weighted_b = weighted_before.get(name, value_before * weight)
        weighted_a = weighted_after.get(name, value_after * weight)
        components.append(
            ScoreComponentDiff(
                component=name,
                weight=weight,
                value_before=value_before,
                value_after=value_after,
                weighted_before=weighted_b,
                weighted_after=weighted_a,
                delta=value_after - value_before,
            )
        )
    penalties_before = {p.reason_code: p for p in baseline.penalties}
    penalties_after = {p.reason_code: p for p in candidate.penalties}
    penalty_codes = sorted(set(penalties_before) | set(penalties_after))
    penalties: list[ScorePenaltyDiff] = []
    for code in penalty_codes:
        before = penalties_before.get(code)
        after = penalties_after.get(code)
        penalties.append(
            ScorePenaltyDiff(
                reason_code=code,
                value_before=before.value if before is not None else 0.0,
                value_after=after.value if after is not None else 0.0,
                applied_before=before.applied if before is not None else False,
                applied_after=after.applied if after is not None else False,
            )
        )
    return ScoreDiff(
        status=CompareSignalStatus.AVAILABLE,
        policy_version=baseline.policy_version,
        formula=baseline.formula,
        raw_score_before=baseline.raw_score,
        raw_score_after=candidate.raw_score,
        raw_score_delta=candidate.raw_score - baseline.raw_score,
        value_before=baseline.value,
        value_after=candidate.value,
        value_delta=candidate.value - baseline.value,
        components=tuple(components),
        penalties=tuple(penalties),
        readiness_status_before=baseline.readiness_status.value,
        readiness_status_after=candidate.readiness_status.value,
    )


def _build_validation_gates_summary(
    *,
    candidate: CandidateDatasetVersion,
    gates_report: ValidationGatesReport | None,
    gates_artifact: ArtifactRef | None,
) -> ValidationGatesSummary:
    overall_status = (
        gates_report.overall_status.value
        if gates_report is not None
        else "not_applicable"
    )
    candidate_status = (
        gates_report.candidate_status.value
        if gates_report is not None
        else "not_applicable"
    )
    raw_unchanged = (
        gates_report.raw_artifact_unchanged if gates_report is not None else True
    )
    blocker_present = (
        gates_report.blocker_present if gates_report is not None else False
    )
    blocker_gate_types: tuple[NonEmptyStr, ...] = (
        tuple(gate.value for gate in gates_report.blocker_gate_types)
        if gates_report is not None
        else ()
    )
    artifact = gates_artifact or candidate.validation_gates_report
    return ValidationGatesSummary(
        overall_status=overall_status,
        candidate_status=candidate_status,
        raw_artifact_unchanged=raw_unchanged,
        blocker_present=blocker_present,
        blocker_gate_types=blocker_gate_types,
        block_export=candidate.block_export,
        block_model_evaluation=candidate.block_model_evaluation,
        block_training=candidate.block_training,
        blocker_reason_codes=candidate.blocker_reason_codes,
        validation_gates_report=artifact,
    )


__all__ = [
    "BuildVersionCompareRequest",
    "BuildVersionCompareResult",
    "VERSION_COMPARE_REPORT_FORMAT",
    "VERSION_COMPARE_REPORT_KIND",
    "VERSION_COMPARE_REPORT_MEDIA_TYPE",
    "VersionCompareBuilderError",
    "build_version_compare_report",
]
