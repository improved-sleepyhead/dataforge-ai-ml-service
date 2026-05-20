"""Tests for fake platform metadata client and callback server."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.adapters import (
    AuditEventType,
    FakePlatformMetadataClient,
    PlatformAuditEvent,
    PlatformJobEvent,
    create_fake_platform_app,
)
from app.domain import ArtifactLineage, ArtifactRef, ComputeRunStatus


def test_fake_platform_client_records_events_and_artifact_refs_without_raw_pii() -> None:
    fake_client = FakePlatformMetadataClient()

    job_event = fake_client.record_job_event(
        PlatformJobEvent(
            platform_job_id="platform_job_001",
            organization_id="org_1",
            project_id="project_1",
            status=ComputeRunStatus.RUNNING,
            stage="dagster.submit",
            details={
                "duration_ms": 12,
                "operator_email": "steward@example.com",
                "message": "contact steward@example.com",
            },
        )
    )
    audit_event = fake_client.record_audit_event(
        PlatformAuditEvent(
            audit_event_id="audit_001",
            event_type=AuditEventType.JOB_STATUS_CHANGED,
            organization_id="org_1",
            project_id="project_1",
            metadata={"service_token": "secret-token-value", "status": "RUNNING"},
        )
    )
    artifact_ref = fake_client.record_artifact_ref(_artifact_ref())

    snapshot = fake_client.snapshot()
    serialized = snapshot.model_dump_json()
    assert snapshot.job_events == (job_event,)
    assert snapshot.audit_events == (audit_event,)
    assert snapshot.artifact_refs == (artifact_ref,)
    assert "steward@example.com" not in serialized
    assert "secret-token-value" not in serialized
    assert job_event.details["operator_email"] == "[REDACTED]"
    assert audit_event.metadata["service_token"] == "[REDACTED]"


def test_fake_platform_callback_server_records_job_event_from_ml_client() -> None:
    fake_client = FakePlatformMetadataClient()
    client = TestClient(create_fake_platform_app(fake_client))

    response = client.post(
        "/platform/jobs/events",
        json={
            "platform_job_id": "platform_job_001",
            "organization_id": "org_1",
            "project_id": "project_1",
            "status": "COMPLETED",
            "stage": "status_bridge.callback",
            "details": {
                "artifact_count": 1,
                "raw_email": "analyst@example.com",
            },
        },
    )

    assert response.status_code == 200
    assert "analyst@example.com" not in response.text
    assert response.json()["details"]["raw_email"] == "[REDACTED]"

    state_response = client.get("/platform/state")

    assert state_response.status_code == 200
    state = state_response.json()
    assert len(state["job_events"]) == 1
    assert state["job_events"][0]["platform_job_id"] == "platform_job_001"
    assert state["job_events"][0]["details"]["raw_email"] == "[REDACTED]"
    assert "analyst@example.com" not in state_response.text


def test_fake_platform_callback_server_records_audit_events_and_artifacts() -> None:
    client = TestClient(create_fake_platform_app())

    audit_response = client.post(
        "/platform/audit-events",
        json={
            "audit_event_id": "audit_001",
            "event_type": "ARTIFACT_REGISTERED",
            "organization_id": "org_1",
            "project_id": "project_1",
            "metadata": {"token": "do-not-store", "artifact_id": "artifact_manifest_001"},
        },
    )
    artifact_response = client.post(
        "/platform/artifacts",
        json=_artifact_ref().model_dump(mode="json"),
    )

    assert audit_response.status_code == 200
    assert artifact_response.status_code == 200
    assert "do-not-store" not in audit_response.text

    state_response = client.get("/platform/state")

    assert state_response.status_code == 200
    state = state_response.json()
    assert len(state["audit_events"]) == 1
    assert len(state["artifact_refs"]) == 1
    assert state["audit_events"][0]["metadata"]["token"] == "[REDACTED]"
    assert state["artifact_refs"][0]["artifact_id"] == "artifact_manifest_001"


def _artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="artifact_manifest_001",
        kind="asset_manifest",
        uri="s3://dataforge/org_1/project_1/dataset_1/v1/manifest.jsonl",
        hash="sha256:" + "a" * 64,
        media_type="application/jsonl",
        size_bytes=128,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_1",
            job_id="compute_run_001",
            config_hash="sha256:" + "b" * 64,
            created_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        ),
    )
