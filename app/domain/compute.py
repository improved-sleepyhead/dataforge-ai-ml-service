"""Compute run contracts emitted by Python/Dagster execution."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.domain.common import NonEmptyStr, Sha256Digest


class WorkflowType(StrEnum):
    """Supported compute workflow modes."""

    ANALYZE_ONLY = "ANALYZE_ONLY"
    PREVIEW_ACTION_PLAN = "PREVIEW_ACTION_PLAN"
    APPLY_SELECTED_ACTIONS = "APPLY_SELECTED_ACTIONS"
    EXPORT = "EXPORT"


class ComputeRunStatus(StrEnum):
    """Lifecycle states for compute-plane runs."""

    QUEUED = "QUEUED"
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"
    WAITING_APPROVAL = "WAITING_APPROVAL"


class ComputeRun(BaseModel):
    """Compute-plane execution record for reports and platform callbacks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compute_run_id: NonEmptyStr
    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    dataset_version_id: NonEmptyStr
    workflow_type: WorkflowType
    status: ComputeRunStatus
    contract_pack_version: NonEmptyStr
    config_hash: Sha256Digest
    started_at: datetime
    completed_at: datetime | None = None
    artifacts: tuple[NonEmptyStr, ...]
