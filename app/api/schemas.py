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
