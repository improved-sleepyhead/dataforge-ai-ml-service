"""API request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain import ActionPlan, ArtifactRef, ComputeRunStatus, MethodRecommendation, WorkflowType
from app.domain.common import NonEmptyStr, S3Uri, Sha256Digest
from app.kernel import ActionPlanApprovalMetadata


class HealthResponse(BaseModel):
    """Health response for service and contract compatibility checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: NonEmptyStr
    service_version: NonEmptyStr
    contract_pack_version: NonEmptyStr


class CancelJobRequest(BaseModel):
    """Signed platform request to cancel a running compute job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    reason_code: NonEmptyStr = "platform_user_cancelled"


class CancelJobResponse(BaseModel):
    """Cancellation acknowledgement; the launcher emits CANCELLED through the bridge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[ComputeRunStatus.CANCELLED, ComputeRunStatus.SKIPPED]
    job_id: NonEmptyStr
    reason_code: NonEmptyStr
    cancellation_accepted: bool


class AnalyzeDatasetRequest(BaseModel):
    """Signed platform request to start an ANALYZE_ONLY dataset workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    dataset_version_id: NonEmptyStr
    dataset_object_refs: tuple[ArtifactRef, ...] = Field(min_length=1)
    prediction_artifact_refs: tuple[ArtifactRef, ...] = ()


class AnalyzeDatasetAcceptedResponse(BaseModel):
    """Response returned once the ANALYZE_ONLY workflow is accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[ComputeRunStatus.ACCEPTED]
    job_id: NonEmptyStr
    status_url: NonEmptyStr
    expected_outputs: tuple[NonEmptyStr, ...]
    materialized_assets: tuple[NonEmptyStr, ...]
    idempotency_key: Sha256Digest
    mutates_dataset: Literal[False] = False


class ActionPlanPreviewRequest(BaseModel):
    """Signed platform request to preview selected recommended actions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    decision_report_id: NonEmptyStr
    selected_decision_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    selected_method_overrides: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    method_recommendations: tuple[MethodRecommendation, ...] = Field(min_length=1)
    created_by_user_id: NonEmptyStr
    input_artifacts: tuple[S3Uri, ...]
    target_version_name: NonEmptyStr | None = None


class ActionPlanPreviewResponse(BaseModel):
    """ActionPlan preview response; it never mutates dataset artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["PREVIEW_READY"]
    job_id: NonEmptyStr
    action_plan: ActionPlan
    mutates_dataset: Literal[False] = False


class ActionPlanExecuteApprovedRequest(BaseModel):
    """Signed platform request to accept an approved ActionPlan for execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    action_plan: ActionPlan
    approval_metadata: ActionPlanApprovalMetadata | None = None
    source_artifacts: tuple[ArtifactRef, ...] = ()


class ActionPlanExecuteApprovedResponse(BaseModel):
    """Accepted execution state after signature and integrity validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[ComputeRunStatus.ACCEPTED]
    job_id: NonEmptyStr
    workflow_type: Literal[WorkflowType.APPLY_SELECTED_ACTIONS]
    action_plan_id: NonEmptyStr
    action_plan_hash: Sha256Digest
    accepted_step_ids: tuple[NonEmptyStr, ...]
    status_url: NonEmptyStr
    expected_outputs: tuple[NonEmptyStr, ...]
    materialized_assets: tuple[NonEmptyStr, ...]
    idempotency_key: Sha256Digest
    candidate_artifact_uri: S3Uri | None = None
    candidate_artifact_hash: Sha256Digest | None = None
    synthetic_artifact_uri: S3Uri | None = None
    synthetic_status: NonEmptyStr
    model_impact_artifact_uri: S3Uri | None = None
    export_package_artifact_uri: S3Uri | None = None
    mutates_dataset: Literal[True] = True


class JobStatusResponse(BaseModel):
    """Compute run status response for ``GET /jobs/{job_id}``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: NonEmptyStr
    workflow_type: WorkflowType
    status: ComputeRunStatus
    status_url: NonEmptyStr
    expected_outputs: tuple[NonEmptyStr, ...]
    materialized_assets: tuple[NonEmptyStr, ...]
    mutates_dataset: bool
    idempotency_key: Sha256Digest
    action_plan_id: NonEmptyStr | None = None
    action_plan_hash: Sha256Digest | None = None
    candidate_artifact_uri: S3Uri | None = None
    candidate_artifact_hash: Sha256Digest | None = None
    synthetic_artifact_uri: S3Uri | None = None
    synthetic_status: NonEmptyStr | None = None
    model_impact_artifact_uri: S3Uri | None = None
    export_package_artifact_uri: S3Uri | None = None


class ActionPlanGetResponse(BaseModel):
    """ActionPlan record returned by ``GET /action-plans/{id}``.

    Carries the preview itself plus optional execution metadata when the
    plan has been accepted for an approved APPLY run. ``mutates_dataset``
    stays ``False`` for plans that have not been executed yet.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan: ActionPlan
    job_id: NonEmptyStr | None = None
    action_plan_hash: Sha256Digest | None = None
    accepted_step_ids: tuple[NonEmptyStr, ...] = ()
    workflow_type: WorkflowType | None = None
    status_url: NonEmptyStr | None = None
    mutates_dataset: bool


class ReviewQueueSummaryEntry(BaseModel):
    """Privacy-safe per-queue summary used by ``GET /reports/{id}/issues``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    queue_type: NonEmptyStr
    item_count: int
    raw_pii_allowed: bool
    redacted_only: bool


class ReportIssuesResponse(BaseModel):
    """Aggregated issues view returned by ``GET /reports/{id}/issues``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    review_queue_summary: tuple[ReviewQueueSummaryEntry, ...]
    total_item_count: int
