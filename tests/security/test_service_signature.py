"""Service-to-service signature validation tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

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
from app.domain import ErrorCode
from app.kernel.config import ServiceConfig, load_config


def test_unsigned_protected_request_is_rejected() -> None:
    client = TestClient(
        create_app(
            include_test_protected_route=True,
            config=_test_config(),
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/__test__/protected-compute-request",
        content=_body_bytes(_payload()),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 401
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert payload["error"]["details"] == {"reason_code": "missing_service_identity"}


def test_fake_valid_platform_signature_allows_protected_request() -> None:
    config = _test_config()
    payload = _payload()
    body = _body_bytes(payload)
    client = TestClient(
        create_app(
            include_test_protected_route=True,
            config=config,
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/__test__/protected-compute-request",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "service_identity": "dataforge-platform",
        "organization_id": "org_1",
        "project_id": "project_1",
    }


def test_changing_project_id_in_signed_payload_is_rejected() -> None:
    config = _test_config()
    original_payload = _payload()
    tampered_payload = {**original_payload, "project_id": "project_2"}
    original_body = _body_bytes(original_payload)
    tampered_body = _body_bytes(tampered_payload)
    client = TestClient(
        create_app(
            include_test_protected_route=True,
            config=config,
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/__test__/protected-compute-request",
        content=tampered_body,
        headers=_signed_headers(config=config, body=original_body, payload=original_payload),
    )

    assert response.status_code == 403
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert payload["error"]["details"] == {"reason_code": "scope_mismatch"}


def test_user_authorization_header_is_not_treated_as_service_auth() -> None:
    client = TestClient(
        create_app(
            include_test_protected_route=True,
            config=_test_config(),
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/__test__/protected-compute-request",
        content=_body_bytes(_payload()),
        headers={
            "authorization": "Bearer user-jwt-that-python-must-ignore",
            "content-type": "application/json",
        },
    )

    assert response.status_code == 401
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert "user-jwt-that-python-must-ignore" not in response.text


def test_stale_platform_signature_is_rejected() -> None:
    config = _test_config()
    payload = _payload()
    body = _body_bytes(payload)
    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    client = TestClient(
        create_app(
            include_test_protected_route=True,
            config=config,
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/__test__/protected-compute-request",
        content=body,
        headers=_signed_headers(
            config=config,
            body=body,
            payload=payload,
            timestamp=stale_timestamp,
        ),
    )

    assert response.status_code == 401
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert payload["error"]["details"] == {"reason_code": "stale_timestamp"}


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


def _payload() -> dict[str, str]:
    return {
        "platform_job_id": "platform_job_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "dataset_version_id": "dataset_version_1",
    }


def _body_bytes(payload: dict[str, str]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _signed_headers(
    *,
    config: ServiceConfig,
    body: bytes,
    payload: dict[str, str],
    timestamp: str | None = None,
) -> dict[str, str]:
    signed_at = timestamp or datetime.now(UTC).isoformat()
    service_identity = config.platform.service_identity
    organization_id = payload["organization_id"]
    project_id = payload["project_id"]
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
