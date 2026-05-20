"""Decision, action planning, review, report, and export contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, S3Uri, Score, Sha256Digest
from app.domain.compute import WorkflowType
from app.domain.evidence import EvidenceRef, SignalStatus
from app.domain.manifest import DataModality


class DatasetDecision(StrEnum):
    """Dataset-level decision outcomes produced by Decision Core."""

    READY_FOR_TRAINING = "READY_FOR_TRAINING"
    READY_FOR_TRAINING_WITH_REVIEW_NOTES = "READY_FOR_TRAINING_WITH_REVIEW_NOTES"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class DatasetReadiness(StrEnum):
    """Export/training readiness state."""

    READY_FOR_EXPORT = "READY_FOR_EXPORT"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NOT_READY_FOR_EXPORT = "NOT_READY_FOR_EXPORT"
    BLOCKED = "BLOCKED"


class MethodCandidateStatus(StrEnum):
    """Method availability status in MethodRecommendation."""

    RECOMMENDED = "recommended"
    AVAILABLE = "available"
    DISABLED_BY_POLICY = "disabled_by_policy"
    DISABLED_BY_READINESS = "disabled_by_readiness"
    BLOCKED = "blocked"


class PolicyStatus(StrEnum):
    """Policy state for a candidate method."""

    ENABLED = "enabled"
    DISABLED_BY_POLICY = "disabled_by_policy"
    DISABLED_BY_READINESS = "disabled_by_readiness"
    BLOCKED = "blocked"


class ReviewQueueType(StrEnum):
    """Supported review queue categories."""

    LABEL_REVIEW = "LABEL_REVIEW"
    PRIVACY_REVIEW = "PRIVACY_REVIEW"
    DUPLICATE_REVIEW = "DUPLICATE_REVIEW"
    ANNOTATION_REVIEW = "ANNOTATION_REVIEW"
    ACTIVE_LEARNING = "ACTIVE_LEARNING"


class GateStatus(StrEnum):
    """Validation gate result status."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class ExportPackageStatus(StrEnum):
    """Export package readiness."""

    READY = "READY"
    BLOCKED = "BLOCKED"


class PolicyVersions(BaseModel):
    """Versioned policies used for a decision artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_policy: NonEmptyStr
    score_policy: NonEmptyStr
    privacy_policy: NonEmptyStr
    method_policy: NonEmptyStr


class ReadinessAssessment(BaseModel):
    """Machine-readable dataset readiness assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DatasetReadiness
    score: Score
    reason_codes: tuple[NonEmptyStr, ...]


class CriticalBlocker(BaseModel):
    """Hard blocker produced by policy gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: NonEmptyStr
    severity: NonEmptyStr
    message: NonEmptyStr
    evidence_refs: tuple[EvidenceRef, ...] = ()


class RecommendedAction(BaseModel):
    """User-facing action recommendation, not a dataset mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommendation_id: NonEmptyStr
    action: NonEmptyStr
    modality: DataModality
    count: int = Field(ge=0)
    reason_codes: tuple[NonEmptyStr, ...]
    reason: NonEmptyStr
    segment: str | None = None
    method_recommendation_id: str | None = None
    requires_approval: bool


class DecisionReport(BaseModel):
    """Decision Core output with dataset readiness and next recommendations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_report_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    decision_schema_version: NonEmptyStr
    dataset_decision: DatasetDecision
    readiness: ReadinessAssessment
    critical_blockers: tuple[CriticalBlocker, ...]
    safe_actions_available: bool
    recommended_next_job: WorkflowType | None = None
    recommended_actions: tuple[RecommendedAction, ...]
    policy_versions: PolicyVersions
    generated_at: datetime


class RecommendedMethod(BaseModel):
    """Selected best method for a recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: NonEmptyStr
    plugin_id: NonEmptyStr
    readiness_level: int = Field(ge=0)
    requires_approval: bool
    reason_codes: tuple[NonEmptyStr, ...]


class MethodCandidate(BaseModel):
    """Candidate method with score, availability, and policy state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: NonEmptyStr
    status: MethodCandidateStatus
    quality_score: Score | None
    risk_score: Score | None
    policy_status: PolicyStatus
    reason: str | None = None


class BlockedMethod(BaseModel):
    """Hard-blocked method that cannot be selected for an ActionPlan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: NonEmptyStr
    reason_code: NonEmptyStr
    reason: NonEmptyStr


class MethodRecommendation(BaseModel):
    """Method selection contract for action planning and UI explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommendation_id: NonEmptyStr
    issue_id: NonEmptyStr
    action_type: NonEmptyStr
    target: dict[str, Any] = Field(default_factory=dict)
    recommended_method: RecommendedMethod
    candidate_methods: tuple[MethodCandidate, ...]
    blocked_methods: tuple[BlockedMethod, ...]
    expected_outputs: tuple[NonEmptyStr, ...]
    model_impact_required: bool

    @model_validator(mode="after")
    def validate_recommended_method_is_candidate(self) -> Self:
        if self.recommended_method.method_id not in {
            candidate.method_id for candidate in self.candidate_methods
        }:
            raise ValueError("recommended_method must appear in candidate_methods")
        return self


class RetryPolicy(BaseModel):
    """Retry behavior for one ActionPlan step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(ge=1)
    retryable_errors: tuple[NonEmptyStr, ...]


class ActionPlanStep(BaseModel):
    """Deterministic, idempotent unit of an ActionPlan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: NonEmptyStr
    type: NonEmptyStr
    depends_on: tuple[NonEmptyStr, ...] = ()
    idempotency_key: Sha256Digest
    method_id: NonEmptyStr
    plugin_id: NonEmptyStr
    plugin_version: NonEmptyStr
    config_hash: Sha256Digest
    policy_version: NonEmptyStr
    validation_gates: tuple[NonEmptyStr, ...]
    preconditions: tuple[NonEmptyStr, ...] = ()
    input_artifacts: tuple[S3Uri, ...]
    output_artifact_kind: NonEmptyStr
    config: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    retry_policy: RetryPolicy
    on_failure: NonEmptyStr
    promotion_scope: NonEmptyStr


class ActionPlan(BaseModel):
    """Versioned, auditable plan created from selected recommendations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    plan_schema_version: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    target_version_name: NonEmptyStr
    created_from_decision_report: NonEmptyStr
    selected_decision_ids: tuple[NonEmptyStr, ...]
    created_by_user_id: NonEmptyStr
    policy_version: NonEmptyStr
    requires_approval: bool
    approval_request_id: str | None = None
    execution_mode: WorkflowType
    steps: tuple[ActionPlanStep, ...]
    validation_gates: tuple[NonEmptyStr, ...]
    expected_outputs: tuple[NonEmptyStr, ...]
    created_at: datetime

    @model_validator(mode="after")
    def validate_step_dependencies(self) -> Self:
        step_ids = {step.step_id for step in self.steps}
        if len(step_ids) != len(self.steps):
            raise ValueError("ActionPlan step_id values must be unique")
        missing_dependencies = sorted(
            dependency
            for step in self.steps
            for dependency in step.depends_on
            if dependency not in step_ids
        )
        if missing_dependencies:
            raise ValueError("ActionPlan step dependencies must reference existing steps")
        return self


class ReviewExportPolicy(BaseModel):
    """Export constraints for review queue integrations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_pii_allowed: bool
    redacted_only: bool
    allow_external_tool: bool


class ReviewQueueItem(BaseModel):
    """One prioritized object requiring human review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    modality: DataModality
    object_type: NonEmptyStr
    reason_codes: tuple[NonEmptyStr, ...]
    priority: Score
    safe_preview_uri: S3Uri
    evidence_refs: tuple[EvidenceRef, ...]


class ReviewQueue(BaseModel):
    """Review queue artifact with safe previews and evidence references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_queue_id: NonEmptyStr
    queue_schema_version: NonEmptyStr
    queue_type: ReviewQueueType
    target_tool: str | None = None
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    objects: tuple[ReviewQueueItem, ...]
    export_policy: ReviewExportPolicy
    created_at: datetime


class DataForgeScore(BaseModel):
    """Dataset-level score with decomposition and reason codes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Score
    policy_version: NonEmptyStr
    components: dict[NonEmptyStr, Score]
    reason_codes: tuple[NonEmptyStr, ...]


class PredictionReportSection(BaseModel):
    """Prediction-derived report metadata with explicit missingness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SignalStatus
    reason: str | None = None
    prediction_manifest_ref: EvidenceRef | None = None
    ambiguous_object_count: int = Field(ge=0)
    probable_label_error_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_prediction_section_state(self) -> Self:
        if self.status is SignalStatus.AVAILABLE and self.prediction_manifest_ref is None:
            raise ValueError("available prediction section must include prediction_manifest_ref")
        if self.status is not SignalStatus.AVAILABLE and not self.reason:
            raise ValueError("missing prediction section must include a reason")
        return self


class DataForgeReport(BaseModel):
    """High-level ANALYZE_ONLY report with detail artifact references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    workflow_type: WorkflowType
    mutates_dataset: bool
    overview: dict[str, Any] = Field(default_factory=dict)
    score: DataForgeScore
    modality_scores: dict[DataModality, Score]
    blockers: tuple[CriticalBlocker, ...]
    recommendations: tuple[RecommendedAction, ...]
    review_queues: tuple[EvidenceRef, ...]
    detail_artifacts: tuple[ArtifactRef, ...]
    prediction_section: PredictionReportSection
    generated_at: datetime


class ValidationGateResult(BaseModel):
    """Executed validation gate result for export readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyStr
    status: GateStatus
    reason_codes: tuple[NonEmptyStr, ...] = ()


class ExportObjectCounts(BaseModel):
    """Counts proving blocked objects are excluded from export outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    included: int = Field(ge=0)
    blocked: int = Field(ge=0)
    excluded: int = Field(ge=0)


class ExportPackageLineage(BaseModel):
    """Lineage metadata for an ExportPackage artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_version_id: NonEmptyStr
    action_plan_id: NonEmptyStr | None = None
    decision_report_id: NonEmptyStr
    config_hash: Sha256Digest


class ExportPackage(BaseModel):
    """Exportable package manifest after export readiness gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    export_package_id: NonEmptyStr
    export_schema_version: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    source_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    status: ExportPackageStatus
    artifacts: tuple[ArtifactRef, ...]
    validation_gates: tuple[ValidationGateResult, ...]
    object_counts: ExportObjectCounts
    blocked_reason_codes: tuple[NonEmptyStr, ...]
    lineage: ExportPackageLineage
    created_at: datetime
