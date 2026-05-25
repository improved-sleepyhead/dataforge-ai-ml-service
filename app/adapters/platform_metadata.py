"""Platform metadata callback ports and dev/test fakes."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from app.domain import ArtifactRef, ComputeRunStatus
from app.domain.common import NonEmptyStr

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SENSITIVE_KEY_PATTERN = re.compile(r"(email|phone|token|secret|password|ssn|pii)", re.IGNORECASE)
_REDACTED = "[REDACTED]"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AuditEventType(StrEnum):
    """Compute-plane audit events accepted by the fake platform adapter."""

    JOB_STATUS_CHANGED = "JOB_STATUS_CHANGED"
    ARTIFACT_REGISTERED = "ARTIFACT_REGISTERED"
    CALLBACK_DELIVERED = "CALLBACK_DELIVERED"
    MANIFEST_VALIDATED = "MANIFEST_VALIDATED"
    PREDICTIONS_VALIDATED = "PREDICTIONS_VALIDATED"
    DATASET_VERSION_PROPOSED = "DATASET_VERSION_PROPOSED"
    LINEAGE_REPORT_BUILT = "LINEAGE_REPORT_BUILT"
    EXPORT_PACKAGE_BUILT = "EXPORT_PACKAGE_BUILT"


class PlatformJobEvent(BaseModel):
    """Safe job lifecycle callback emitted by ML service code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform_job_id: NonEmptyStr
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    status: ComputeRunStatus
    stage: NonEmptyStr
    occurred_at: datetime = Field(default_factory=_utcnow)
    details: dict[str, Any] = Field(default_factory=dict)

    def redacted(self) -> PlatformJobEvent:
        return self.model_copy(update={"details": redact_metadata(self.details)})


class PlatformAuditEvent(BaseModel):
    """Safe compute audit callback for fake platform tests/dev."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_event_id: NonEmptyStr
    event_type: AuditEventType
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    actor_service: NonEmptyStr = "dataforgeai-ml-service"
    occurred_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def redacted(self) -> PlatformAuditEvent:
        return self.model_copy(update={"metadata": redact_metadata(self.metadata)})


class FakePlatformState(BaseModel):
    """Serializable in-memory fake platform callback store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_events: tuple[PlatformJobEvent, ...]
    audit_events: tuple[PlatformAuditEvent, ...]
    artifact_refs: tuple[ArtifactRef, ...]


class FakePlatformMetadataClient:
    """In-memory fake used by tests/dev instead of a production platform adapter."""

    def __init__(self) -> None:
        self._job_events: list[PlatformJobEvent] = []
        self._audit_events: list[PlatformAuditEvent] = []
        self._artifact_refs: list[ArtifactRef] = []

    def record_job_event(self, event: PlatformJobEvent) -> PlatformJobEvent:
        stored = event.redacted()
        self._job_events.append(stored)
        return stored

    def record_audit_event(self, event: PlatformAuditEvent) -> PlatformAuditEvent:
        stored = event.redacted()
        self._audit_events.append(stored)
        return stored

    def record_artifact_ref(self, artifact_ref: ArtifactRef) -> ArtifactRef:
        self._artifact_refs.append(artifact_ref)
        return artifact_ref

    def snapshot(self) -> FakePlatformState:
        return FakePlatformState(
            job_events=tuple(self._job_events),
            audit_events=tuple(self._audit_events),
            artifact_refs=tuple(self._artifact_refs),
        )


def get_fake_platform_client(request: Request) -> FakePlatformMetadataClient:
    client = getattr(request.app.state, "fake_platform_client", None)
    if not isinstance(client, FakePlatformMetadataClient):
        raise RuntimeError("Fake platform client is not configured")
    return client


FakePlatformClientDep = Annotated[FakePlatformMetadataClient, Depends(get_fake_platform_client)]


def create_fake_platform_app(
    client: FakePlatformMetadataClient | None = None,
) -> FastAPI:
    """Create an in-memory fake platform callback server for tests/dev only."""
    application = FastAPI(
        title="DataForge AI Fake Platform Callback Server",
        version="0.1.0-dev",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.fake_platform_client = client or FakePlatformMetadataClient()

    @application.post("/platform/jobs/events", response_model=PlatformJobEvent)
    async def record_job_event(
        event: PlatformJobEvent,
        fake_client: FakePlatformClientDep,
    ) -> PlatformJobEvent:
        return fake_client.record_job_event(event)

    @application.post("/platform/audit-events", response_model=PlatformAuditEvent)
    async def record_audit_event(
        event: PlatformAuditEvent,
        fake_client: FakePlatformClientDep,
    ) -> PlatformAuditEvent:
        return fake_client.record_audit_event(event)

    @application.post("/platform/artifacts", response_model=ArtifactRef)
    async def record_artifact_ref(
        artifact_ref: ArtifactRef,
        fake_client: FakePlatformClientDep,
    ) -> ArtifactRef:
        return fake_client.record_artifact_ref(artifact_ref)

    @application.get("/platform/state", response_model=FakePlatformState)
    async def get_state(fake_client: FakePlatformClientDep) -> FakePlatformState:
        return fake_client.snapshot()

    return application


def redact_metadata(value: Any) -> Any:
    """Redact obvious PII/secrets from fake platform metadata recursively."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _SENSITIVE_KEY_PATTERN.search(str(key)):
                redacted[str(key)] = _REDACTED
            else:
                redacted[str(key)] = redact_metadata(item)
        return redacted
    if isinstance(value, list):
        return [redact_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_metadata(item) for item in value)
    if isinstance(value, str):
        return _EMAIL_PATTERN.sub(_REDACTED, value)
    return value

