"""API request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain import ArtifactRef, ComputeRunStatus
from app.domain.common import NonEmptyStr


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
    mutates_dataset: Literal[False] = False
