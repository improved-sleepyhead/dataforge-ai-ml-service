"""Contracts for the candidate Version Compare artifact (TASK-052).

The Version Compare artifact is the first-class before/after report
produced for any applied ``ActionPlan``. PRD §27.2.3 specifies the
contract shape:

* object counts before/after;
* changed objects;
* added synthetic objects;
* removed or blocked objects;
* imputed fields;
* PII risk before/after;
* duplicate count before/after;
* class balance before/after;
* model metrics before/after if eligible;
* score decomposition diff;
* ActionPlan that produced the diff;
* validation gates result.

The artifact consumes only normalized contract reports
(``TabularProfileReport``, ``TextOcrReport``, ``DataForgeScore``,
``CandidateDatasetVersion``, ``ModelImpactReport``,
``TabularImputationReport``, ``DuplicateActionReport``,
``SyntheticDatasetReport``). It never reads raw rows, raw text or
raw PII payloads. Missing/unavailable metrics are represented
explicitly as ``not_applicable`` with a reason.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest

VERSION_COMPARE_REPORT_SCHEMA_VERSION = "version_compare_report.v1"


class CompareSignalStatus(StrEnum):
    """Status used when a compare metric cannot be computed."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"


class ObjectCountDiff(BaseModel):
    """Object counts before/after the ActionPlan was applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    before: int = Field(ge=0)
    after: int = Field(ge=0)
    delta: int


class ChangedObjectsBlock(BaseModel):
    """Aggregate counts of changed/added/removed objects.

    All numbers are aggregate counts derived from contract reports
    (imputation, duplicate action, synthetic generation). Object ids
    are intentionally never included to keep the artifact PII-safe.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    changed_objects_total: int = Field(ge=0)
    imputed_object_count: int = Field(ge=0)
    duplicate_marked_count: int = Field(ge=0)
    duplicate_removed_count: int = Field(ge=0)
    redacted_object_count: int = Field(ge=0)
    synthetic_added_count: int = Field(ge=0)
    removed_or_blocked_count: int = Field(ge=0)


class ImputedFieldEntry(BaseModel):
    """One imputed field entry (column, method, before/after counts)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: NonEmptyStr
    method: NonEmptyStr
    before_missing_count: int = Field(ge=0)
    after_missing_count: int = Field(ge=0)
    imputed_count: int = Field(ge=0)


class PiiRiskDiff(BaseModel):
    """PII risk before/after.

    PRD §11.7: text/OCR PII findings drive privacy risk. The compare
    block aggregates record-level PII counts and the redaction-coverage
    rate so the UI can render a single "risk reduced" indicator.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CompareSignalStatus = CompareSignalStatus.AVAILABLE
    not_applicable_reason: str | None = None
    pii_record_count_before: int = Field(ge=0, default=0)
    pii_record_count_after: int = Field(ge=0, default=0)
    pii_record_count_delta: int = 0
    redacted_record_count_before: int = Field(ge=0, default=0)
    redacted_record_count_after: int = Field(ge=0, default=0)
    unredacted_pii_record_count_before: int = Field(ge=0, default=0)
    unredacted_pii_record_count_after: int = Field(ge=0, default=0)


class DuplicateCountDiff(BaseModel):
    """Duplicate-pair counts before/after."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CompareSignalStatus = CompareSignalStatus.AVAILABLE
    not_applicable_reason: str | None = None
    duplicate_pair_count_before: int = Field(ge=0, default=0)
    duplicate_pair_count_after: int = Field(ge=0, default=0)
    duplicate_pair_count_delta: int = 0


class ClassBalanceClassEntry(BaseModel):
    """Per-class count before/after."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonEmptyStr
    count_before: int = Field(ge=0)
    count_after: int = Field(ge=0)


class ClassBalanceDiff(BaseModel):
    """Class-balance diff for a supervised target column.

    For non-supervised candidates the block is ``not_applicable`` with
    a reason. Otherwise the block carries per-class counts before and
    after so the UI can show synthetic rare-class augmentation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CompareSignalStatus = CompareSignalStatus.NOT_APPLICABLE
    not_applicable_reason: str | None = None
    target_column: str | None = None
    rare_class_label: str | None = None
    rare_class_count_before: int | None = Field(ge=0, default=None)
    rare_class_count_after: int | None = Field(ge=0, default=None)
    rare_class_ratio_before: Score | None = None
    rare_class_ratio_after: Score | None = None
    rare_class_ratio_delta: float | None = None
    imbalance_ratio_before: float | None = None
    imbalance_ratio_after: float | None = None
    imbalance_ratio_delta: float | None = None
    minority_share_before: Score | None = None
    minority_share_after: Score | None = None
    classes: tuple[ClassBalanceClassEntry, ...] = ()


class ModelMetricsCompare(BaseModel):
    """Model metrics before/after (when model impact is eligible)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CompareSignalStatus = CompareSignalStatus.NOT_APPLICABLE
    not_applicable_reason: str | None = None
    verdict: str | None = None
    rare_class_recall_before: Score | None = None
    rare_class_recall_after: Score | None = None
    rare_class_recall_delta: float | None = None
    macro_f1_before: Score | None = None
    macro_f1_after: Score | None = None
    macro_f1_delta: float | None = None
    weighted_f1_before: Score | None = None
    weighted_f1_after: Score | None = None
    weighted_f1_delta: float | None = None
    pr_auc_before: Score | None = None
    pr_auc_after: Score | None = None
    pr_auc_delta: float | None = None
    pr_auc_status: CompareSignalStatus = CompareSignalStatus.NOT_APPLICABLE
    pr_auc_reason: str | None = None
    model_impact_report: ArtifactRef | None = None


class ScoreComponentDiff(BaseModel):
    """One DataForge Score component before/after."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: NonEmptyStr
    weight: float
    value_before: Score
    value_after: Score
    weighted_before: float
    weighted_after: float
    delta: float


class ScorePenaltyDiff(BaseModel):
    """One score-penalty entry before/after."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: NonEmptyStr
    value_before: float
    value_after: float
    applied_before: bool
    applied_after: bool


class ScoreDiff(BaseModel):
    """DataForge Score before/after with full decomposition diff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CompareSignalStatus = CompareSignalStatus.NOT_APPLICABLE
    not_applicable_reason: str | None = None
    policy_version: str | None = None
    formula: str | None = None
    raw_score_before: float | None = None
    raw_score_after: float | None = None
    raw_score_delta: float | None = None
    value_before: Score | None = None
    value_after: Score | None = None
    value_delta: float | None = None
    components: tuple[ScoreComponentDiff, ...] = ()
    penalties: tuple[ScorePenaltyDiff, ...] = ()
    readiness_status_before: str | None = None
    readiness_status_after: str | None = None


class ValidationGatesSummary(BaseModel):
    """Compact summary of the candidate validation gates report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_status: NonEmptyStr
    candidate_status: NonEmptyStr
    raw_artifact_unchanged: bool
    blocker_present: bool
    blocker_gate_types: tuple[NonEmptyStr, ...] = ()
    block_export: bool
    block_model_evaluation: bool
    block_training: bool
    blocker_reason_codes: tuple[NonEmptyStr, ...] = ()
    validation_gates_report: ArtifactRef | None = None


class VersionCompareLineage(BaseModel):
    """Lineage envelope for the Version Compare artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    candidate_version_artifact: ArtifactRef | None = None
    action_plan_id: NonEmptyStr
    decision_report_id: NonEmptyStr | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest


class VersionCompareReport(BaseModel):
    """Before/after Version Compare artifact for an applied ActionPlan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = VERSION_COMPARE_REPORT_SCHEMA_VERSION
    base_version_id: NonEmptyStr
    candidate_version_id: NonEmptyStr
    object_counts: ObjectCountDiff
    changed_objects: ChangedObjectsBlock
    imputed_fields: tuple[ImputedFieldEntry, ...] = ()
    pii_risk: PiiRiskDiff
    duplicate_counts: DuplicateCountDiff
    class_balance: ClassBalanceDiff
    model_metrics: ModelMetricsCompare
    score: ScoreDiff
    validation_gates: ValidationGatesSummary
    action_plan_id: NonEmptyStr
    lineage: VersionCompareLineage
    generated_at: datetime


__all__ = [
    "CompareSignalStatus",
    "ChangedObjectsBlock",
    "ClassBalanceClassEntry",
    "ClassBalanceDiff",
    "DuplicateCountDiff",
    "ImputedFieldEntry",
    "ModelMetricsCompare",
    "ObjectCountDiff",
    "PiiRiskDiff",
    "ScoreComponentDiff",
    "ScoreDiff",
    "ScorePenaltyDiff",
    "VERSION_COMPARE_REPORT_SCHEMA_VERSION",
    "ValidationGatesSummary",
    "VersionCompareLineage",
    "VersionCompareReport",
]
