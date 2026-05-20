"""Compute-plane context models supplied by the platform backend."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.domain.common import NonEmptyStr


class RiskProfile(StrEnum):
    """Supported runtime/security profiles in dataset context."""

    DEMO_STRICT = "demo_strict"
    BANKING_STRICT = "banking_strict"


class PlatformJobType(StrEnum):
    """Platform-visible job types accepted by the compute plane."""

    ANALYZE_DATASET = "ANALYZE_DATASET"
    PREVIEW_ACTION_PLAN = "PREVIEW_ACTION_PLAN"
    APPLY_SELECTED_ACTIONS = "APPLY_SELECTED_ACTIONS"
    EXPORT = "EXPORT"


class DatasetVersionContext(BaseModel):
    """Dataset version context read from the platform control plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr | None = None
    version_name: NonEmptyStr
    created_by_job_id: NonEmptyStr
    created_at: datetime
    risk_profile: RiskProfile
    policy_version: NonEmptyStr
    artifact_refs: tuple[NonEmptyStr, ...]


class PlatformJobContext(BaseModel):
    """Signed platform job context for internal service-to-service requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    dataset_version_id: NonEmptyStr
    job_type: PlatformJobType
    requested_by: NonEmptyStr
    requested_at: datetime
    callback_url: NonEmptyStr
