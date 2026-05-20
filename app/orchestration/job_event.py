"""Compute-plane job lifecycle event contract.

The status bridge converts compute progress into safe ``JobEvent`` records
that the platform control plane can consume. ``JobEvent`` is intentionally
narrow: only stable identifiers, the well-known compute lifecycle stage,
the technical run status, a normalized progress value, optional artifact
references emitted by the stage, and the event timestamp.

Constraints honored by this contract:

* ``JobEvent`` must not carry raw PII, raw text, raw payloads, or secrets.
  Callers must build it from technical metadata only; the redaction layer
  in the fake platform adapter is a defense-in-depth, not the primary
  guarantee.
* ``stage`` uses the well-known :class:`JobStage` enum so the platform UI
  can render a stable progress timeline.
* ``progress`` is constrained to ``[0, 1]`` and represents the stage-wise
  progress of one compute run.
* ``artifact_refs`` is an immutable tuple of ``ArtifactRef`` records that
  the stage just produced. It is allowed to be empty for stages that do
  not yet emit artifacts (skeleton mode).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain import ArtifactRef, ComputeRunStatus
from app.domain.common import NonEmptyStr, Score


class JobStage(StrEnum):
    """Stable compute-plane lifecycle stages emitted to the platform."""

    QUEUED = "QUEUED"
    INGESTING = "INGESTING"
    BUILDING_MANIFEST = "BUILDING_MANIFEST"
    VALIDATING = "VALIDATING"
    PROFILING_TABULAR = "PROFILING_TABULAR"
    BUILDING_EVIDENCE = "BUILDING_EVIDENCE"
    RUNNING_DECISION_CORE = "RUNNING_DECISION_CORE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobEvent(BaseModel):
    """Safe compute lifecycle event delivered to the platform via status bridge.

    ``job_id`` corresponds to the platform job identifier. ``progress`` is a
    normalized value, typically the share of completed analyze stages.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: NonEmptyStr
    stage: JobStage
    status: ComputeRunStatus
    progress: Score
    artifact_refs: tuple[ArtifactRef, ...] = ()
    created_at: datetime = Field(default_factory=_utcnow)


__all__ = ["JobEvent", "JobStage"]
