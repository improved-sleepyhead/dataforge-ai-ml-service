"""FastAPI app boundary tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.security import (
    ORGANIZATION_ID_HEADER,
    PROJECT_ID_HEADER,
    SERVICE_IDENTITY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    build_service_signature,
)
from app.domain import ArtifactLineage, ArtifactRef, ComputeRunStatus, ErrorCode
from app.kernel.config import ServiceConfig, load_config
from app.orchestration.analyze_workflow import expected_analyze_outputs
from app.validation.contracts import load_contract_pack


def test_health_returns_service_and_contract_versions() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "service_version": "0.1.0",
        "contract_pack_version": load_contract_pack().version,
    }


def test_unhandled_exception_returns_safe_error_response_without_raw_details() -> None:
    client = TestClient(create_app(include_test_error_route=True), raise_server_exceptions=False)

    response = client.get("/__test__/unhandled-error")

    assert response.status_code == 500
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.PLUGIN_EXECUTION_FAILED
    assert payload["error"]["message"] == "Internal compute service error."
    assert payload["error"]["stage"] == "api"
    assert "demo@example.com" not in response.text
    assert "raw secret token" not in response.text


def test_validation_error_returns_error_response() -> None:
    client = TestClient(create_app(include_test_error_route=True), raise_server_exceptions=False)

    response = client.get("/__test__/validation-error", params={"limit": 0})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.INVALID_JOB_PAYLOAD
    assert payload["error"]["recoverable"] is True
    assert payload["error"]["details"] == {"error_count": 1}


def test_openapi_generates_health_and_error_schemas() -> None:
    client = TestClient(create_app())

    response = client.get("/api/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/jobs/analyze-dataset" in schema["paths"]
    assert "HealthResponse" in schema["components"]["schemas"]
    assert "AnalyzeDatasetAcceptedResponse" in schema["components"]["schemas"]
    assert "ErrorResponse" in schema["components"]["schemas"]


def test_analyze_dataset_endpoint_accepts_signed_request_and_materializes_base_assets() -> None:
    config = _test_config()
    app = create_app(config=config)
    payload = _analyze_payload()
    body = _body_bytes(payload)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/jobs/analyze-dataset",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == ComputeRunStatus.ACCEPTED
    assert data["job_id"] == payload["platform_job_id"]
    assert data["status_url"] == f"/api/v1/jobs/{payload['platform_job_id']}/status"
    assert data["expected_outputs"] == list(expected_analyze_outputs(include_predictions=False))
    assert set(data["materialized_assets"]) == set(data["expected_outputs"])
    assert data["mutates_dataset"] is False

    snapshot = app.state.fake_platform_client.snapshot()
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED
    assert all(event.details.get("dataset_id") == "dataset_1" for event in snapshot.job_events)
    assert "candidate_dataset" not in response.text


def test_analyze_dataset_endpoint_materializes_prediction_outputs_when_refs_provided() -> None:
    config = _test_config()
    app = create_app(config=config)
    payload = _analyze_payload(
        prediction_artifact_refs=[_artifact_ref("prediction_manifest_1", "prediction_manifest")]
    )
    body = _body_bytes(payload)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/jobs/analyze-dataset",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 202
    data = response.json()
    expected = expected_analyze_outputs(include_predictions=True)
    assert data["expected_outputs"] == list(expected)
    assert set(data["materialized_assets"]) == set(expected)
    assert {
        "prediction_manifest",
        "prediction_validation_report",
        "model_error_analysis_report",
        "ambiguous_object_candidates",
        "probable_label_error_candidates",
    }.issubset(data["materialized_assets"])
    assert data["mutates_dataset"] is False
    snapshot = app.state.fake_platform_client.snapshot()
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED


def _test_config() -> ServiceConfig:
    return load_config(
        {
            "DATAFORGE_PROFILE": "demo_strict",
            "DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL": "http://localhost:9000",
            "DATAFORGE_OBJECT_STORAGE_BUCKET": "dataforge-local",
            "DATAFORGE_PLATFORM_CALLBACK_URL": "http://platform.local/api/ml/jobs/callback",
            "DATAFORGE_SERVICE_SIGNING_SECRET": "local-dev-signing-secret",
            "DATAFORGE_DAGSTER_HOME": "/tmp/dataforge-dagster",
            "DATAFORGE_POLICY_CONFIG_PATH": "configs/policies/demo_strict.yaml",
            "DATAFORGE_DECISION_POLICY_PATH": "configs/policies/decision_v0.yaml",
            "DATAFORGE_SCORE_POLICY_PATH": "configs/policies/score_v0.yaml",
        }
    )


def _analyze_payload(
    *,
    prediction_artifact_refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "platform_job_id": "platform_job_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "dataset_version_id": "dataset_version_1",
        "dataset_object_refs": [_artifact_ref("raw_archive_1", "raw_dataset_archive")],
        "prediction_artifact_refs": []
        if prediction_artifact_refs is None
        else prediction_artifact_refs,
    }


def _artifact_ref(artifact_id: str, kind: str) -> dict[str, object]:
    artifact = ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=f"s3://dataforge-local/dataforge/org_1/project_1/dataset_1/{artifact_id}.json",
        hash="sha256:" + "a" * 64,
        media_type="application/json",
        size_bytes=128,
        schema_version="v0.1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_1",
            job_id="platform_job_001",
            config_hash="sha256:" + "b" * 64,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    return artifact.model_dump(mode="json")


def _body_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _signed_headers(
    *,
    config: ServiceConfig,
    body: bytes,
    payload: dict[str, object],
) -> dict[str, str]:
    signed_at = datetime.now(UTC).isoformat()
    service_identity = config.platform.service_identity
    organization_id = str(payload["organization_id"])
    project_id = str(payload["project_id"])
    return {
        SERVICE_IDENTITY_HEADER: service_identity,
        TIMESTAMP_HEADER: signed_at,
        ORGANIZATION_ID_HEADER: organization_id,
        PROJECT_ID_HEADER: project_id,
        SIGNATURE_HEADER: build_service_signature(
            secret=config.platform.service_signing_secret.get_secret_value(),
            service_identity=service_identity,
            timestamp=signed_at,
            organization_id=organization_id,
            project_id=project_id,
            body=body,
        ),
        "content-type": "application/json",
    }
