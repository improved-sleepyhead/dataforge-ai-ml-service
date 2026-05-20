"""Compute-plane status bridge for platform job events.

Dagster compute runs publish stage events back to the platform control plane
through the ``FakePlatformMetadataClient`` in dev/tests, and through a real
platform adapter in production. The bridge is intentionally a small helper
so that Python never becomes the source of truth for approvals or final
dataset promotion.

Privacy: stage events carry only stable technical fields (compute_run_id,
platform_job_id, dataset_version_id, asset/stage name, status). Raw PII,
raw text, or secrets must never reach status events.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.adapters import FakePlatformMetadataClient, PlatformJobEvent
from app.domain import ComputeRunStatus
from app.domain.common import NonEmptyStr


@dataclass(frozen=True)
class RunContext:
    """Per-run identifiers carried alongside Dagster compute materializations."""

    compute_run_id: NonEmptyStr
    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    dataset_version_id: NonEmptyStr


def emit_stage_event(
    *,
    fake_platform: FakePlatformMetadataClient,
    run_context: RunContext,
    stage: str,
    status: ComputeRunStatus,
) -> PlatformJobEvent:
    """Record one safe stage event for a Dagster asset materialization.

    The fake platform client redacts obvious PII patterns from event details
    before storing the event, but callers must still avoid putting raw text
    or secrets into the ``stage`` name or details.
    """
    event = PlatformJobEvent(
        platform_job_id=run_context.platform_job_id,
        organization_id=run_context.organization_id,
        project_id=run_context.project_id,
        status=status,
        stage=stage,
        details={
            "compute_run_id": run_context.compute_run_id,
            "dataset_id": run_context.dataset_id,
            "dataset_version_id": run_context.dataset_version_id,
        },
    )
    return fake_platform.record_job_event(event)


__all__ = ["RunContext", "emit_stage_event"]
