"""Compute-plane status bridge for platform job events.

Dagster compute runs publish stage events back to the platform control plane
through the ``FakePlatformMetadataClient`` in dev/tests, and through a real
platform adapter in production. The bridge is intentionally a small helper
so that Python never becomes the source of truth for approvals or final
dataset promotion.

Privacy: stage events carry only stable technical fields (compute_run_id,
platform_job_id, dataset_version_id, asset/stage name, status). Raw PII,
raw text, or secrets must never reach status events. The bridge forwards
events to ``FakePlatformMetadataClient`` which performs defense-in-depth
redaction; in addition this module exposes :func:`scan_event_for_raw_pii`
so tests can prove events do not leak PII even before redaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.adapters import FakePlatformMetadataClient, PlatformJobEvent
from app.domain import ArtifactRef, ComputeRunStatus
from app.domain.common import NonEmptyStr
from app.orchestration.job_event import JobEvent, JobStage

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?:\+?\d[\s().-]*){10,}\d")
_PASSPORT_PATTERN = re.compile(r"\b(?:passport\s*)?\d{4}[\s-]?\d{6}\b", re.IGNORECASE)
_SECRET_PATTERN = re.compile(
    r"\b(?:token|secret|password|api[_-]?key)\s*[:=]\s*['\"]?[^'\"\s,}]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RunContext:
    """Per-run identifiers carried alongside Dagster compute materializations."""

    compute_run_id: NonEmptyStr
    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    dataset_version_id: NonEmptyStr


@dataclass(frozen=True)
class RunStatusBridge:
    """Helper that translates compute lifecycle into safe ``JobEvent`` records.

    The bridge keeps a reference to the fake platform client used in
    dev/tests; in production it would wrap a real platform metadata adapter
    with the same shape (``record_job_event``).
    """

    fake_platform: FakePlatformMetadataClient

    def emit_started(
        self,
        *,
        run_context: RunContext,
        stage: JobStage = JobStage.QUEUED,
        progress: float = 0.0,
    ) -> JobEvent:
        """Emit the initial QUEUED-or-INGESTING event for a compute run."""
        return self._emit(
            run_context=run_context,
            stage=stage,
            status=ComputeRunStatus.RUNNING
            if stage is not JobStage.QUEUED
            else ComputeRunStatus.ACCEPTED,
            progress=progress,
            artifact_refs=(),
        )

    def emit_stage(
        self,
        *,
        run_context: RunContext,
        stage: JobStage,
        progress: float,
        artifact_refs: tuple[ArtifactRef, ...] = (),
    ) -> JobEvent:
        """Emit a RUNNING event for an in-progress stage."""
        return self._emit(
            run_context=run_context,
            stage=stage,
            status=ComputeRunStatus.RUNNING,
            progress=progress,
            artifact_refs=artifact_refs,
        )

    def emit_completed(
        self,
        *,
        run_context: RunContext,
        artifact_refs: tuple[ArtifactRef, ...] = (),
    ) -> JobEvent:
        """Emit the terminal COMPLETED event for a successful run."""
        return self._emit(
            run_context=run_context,
            stage=JobStage.COMPLETED,
            status=ComputeRunStatus.COMPLETED,
            progress=1.0,
            artifact_refs=artifact_refs,
        )

    def emit_failed(
        self,
        *,
        run_context: RunContext,
        progress: float = 0.0,
        error_code: str | None = None,
    ) -> JobEvent:
        """Emit the terminal FAILED event with a stable error code."""
        return self._emit(
            run_context=run_context,
            stage=JobStage.FAILED,
            status=ComputeRunStatus.FAILED,
            progress=progress,
            artifact_refs=(),
            extra_details={"error_code": error_code} if error_code else None,
        )

    def emit_cancelled(
        self,
        *,
        run_context: RunContext,
        progress: float = 0.0,
    ) -> JobEvent:
        """Emit the terminal CANCELLED event."""
        return self._emit(
            run_context=run_context,
            stage=JobStage.CANCELLED,
            status=ComputeRunStatus.FAILED,
            progress=progress,
            artifact_refs=(),
        )

    def _emit(
        self,
        *,
        run_context: RunContext,
        stage: JobStage,
        status: ComputeRunStatus,
        progress: float,
        artifact_refs: tuple[ArtifactRef, ...],
        extra_details: dict[str, Any] | None = None,
    ) -> JobEvent:
        event = JobEvent(
            job_id=run_context.platform_job_id,
            stage=stage,
            status=status,
            progress=progress,
            artifact_refs=artifact_refs,
        )
        details: dict[str, Any] = {
            "compute_run_id": run_context.compute_run_id,
            "dataset_id": run_context.dataset_id,
            "dataset_version_id": run_context.dataset_version_id,
            "progress": event.progress,
            "artifact_uris": [ref.uri for ref in event.artifact_refs],
        }
        if extra_details:
            details.update(extra_details)
        platform_event = PlatformJobEvent(
            platform_job_id=run_context.platform_job_id,
            organization_id=run_context.organization_id,
            project_id=run_context.project_id,
            status=status,
            stage=stage.value,
            details=details,
        )
        self.fake_platform.record_job_event(platform_event)
        return event


def emit_stage_event(
    *,
    fake_platform: FakePlatformMetadataClient,
    run_context: RunContext,
    stage: JobStage,
    status: ComputeRunStatus = ComputeRunStatus.RUNNING,
    progress: float = 0.0,
    artifact_refs: tuple[ArtifactRef, ...] = (),
) -> JobEvent:
    """Compatibility helper for assets that emit a single stage event.

    Kept as a thin wrapper around :class:`RunStatusBridge` so that asset
    code does not need to instantiate the bridge for every stage.
    """
    bridge = RunStatusBridge(fake_platform=fake_platform)
    if status is ComputeRunStatus.RUNNING:
        return bridge.emit_stage(
            run_context=run_context,
            stage=stage,
            progress=progress,
            artifact_refs=artifact_refs,
        )
    if status is ComputeRunStatus.COMPLETED:
        return bridge.emit_completed(
            run_context=run_context,
            artifact_refs=artifact_refs,
        )
    if status is ComputeRunStatus.FAILED:
        return bridge.emit_failed(run_context=run_context, progress=progress)
    return bridge.emit_started(
        run_context=run_context,
        stage=stage,
        progress=progress,
    )


def scan_event_for_raw_pii(event: JobEvent) -> tuple[str, ...]:
    """Return PII categories detected in a serialized ``JobEvent``.

    Used by tests to prove the status bridge does not leak raw PII even
    before defense-in-depth redaction in the platform adapter.
    """
    payload = event.model_dump_json()
    categories: list[str] = []
    if _EMAIL_PATTERN.search(payload):
        categories.append("email")
    if _PHONE_PATTERN.search(payload):
        categories.append("phone")
    if _PASSPORT_PATTERN.search(payload):
        categories.append("passport")
    if _SECRET_PATTERN.search(payload):
        categories.append("secret")
    return tuple(categories)


__all__ = [
    "RunContext",
    "RunStatusBridge",
    "emit_stage_event",
    "scan_event_for_raw_pii",
]
