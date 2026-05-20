"""FastAPI app boundary tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.domain import ErrorCode
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
    assert "HealthResponse" in schema["components"]["schemas"]
    assert "ErrorResponse" in schema["components"]["schemas"]
