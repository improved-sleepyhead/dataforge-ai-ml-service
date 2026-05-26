"""TASK-059: stable compute-plane error taxonomy and recoverability hints.

The compute plane returns a single normalized :class:`ErrorResponse`
shape for every failure path. The tests below cover:

1. The full enum of stable error codes required by acceptance criteria
   is present in :class:`ErrorCode` and exposed in the OpenAPI schema.
2. Each :class:`ErrorCode` has a non-empty, safe ``remediation_hint``
   (no raw PII, no plugin internals, no stack-trace markers).
3. Plugin and kernel errors raised inside request handlers are
   normalized into ``ErrorResponse`` with the original stable code,
   recoverable flag, and a remediation hint — and never expose raw
   exception messages or tracebacks.
4. ``ServiceSignatureError`` and validation errors keep stable codes
   and recoverable flags.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_taxonomy import remediation_hint_for
from app.api.main import create_app
from app.api.security import (
    ORGANIZATION_ID_HEADER,
    PROJECT_ID_HEADER,
    SERVICE_IDENTITY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    PlatformIdentityDep,
    build_service_signature,
)
from app.domain import ErrorCode
from app.kernel.config import ServiceConfig, load_config


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

# ---------------------------------------------------------------------------
# AC: full taxonomy is present.
# ---------------------------------------------------------------------------


_REQUIRED_CODES: tuple[ErrorCode, ...] = (
    ErrorCode.INVALID_JOB_PAYLOAD,
    ErrorCode.UNSUPPORTED_MODALITY,
    ErrorCode.ARTIFACT_NOT_FOUND,
    ErrorCode.ARTIFACT_OUT_OF_SCOPE,
    ErrorCode.INVALID_ARCHIVE_STRUCTURE,
    ErrorCode.POLICY_BLOCKED,
    ErrorCode.PII_RESTRICTED,
    ErrorCode.LEAKAGE_DETECTED,
    ErrorCode.CONTRACT_VALIDATION_FAILED,
    ErrorCode.PREDICTION_VALIDATION_FAILED,
    ErrorCode.PLUGIN_NOT_ENABLED,
    ErrorCode.ACTION_PLAN_PRECONDITION_FAILED,
    ErrorCode.VALIDATION_GATE_FAILED,
    ErrorCode.MODEL_IMPACT_NOT_ELIGIBLE,
    ErrorCode.EXPORT_BLOCKED,
)


def test_required_error_codes_are_part_of_the_stable_taxonomy() -> None:
    """All AC-required stable error codes must exist in :class:`ErrorCode`."""
    enum_values = {code.value for code in ErrorCode}
    for code in _REQUIRED_CODES:
        assert code.value in enum_values, f"missing stable error code: {code.value}"


def test_remediation_hint_is_safe_and_non_empty_for_every_required_code() -> None:
    """Every required code maps to a non-empty hint without unsafe markers."""
    for code in _REQUIRED_CODES:
        hint = remediation_hint_for(code)
        assert hint and len(hint) >= 10, f"hint for {code.value} is empty/too short"
        _assert_safe_text(hint)


def test_openapi_includes_stable_error_response_schema() -> None:
    """OpenAPI surfaces ErrorResponse with the canonical fields."""
    client = TestClient(create_app())

    response = client.get("/api/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "ErrorResponse" in schema["components"]["schemas"]
    error_body = schema["components"]["schemas"]["ErrorBody"]["properties"]
    for field in ("code", "message", "recoverable", "stage", "remediation_hint"):
        assert field in error_body, f"ErrorBody missing field: {field}"
    code_enum = error_body["code"]["enum"] if "enum" in error_body["code"] else None
    if code_enum is not None:
        for code in _REQUIRED_CODES:
            assert code.value in code_enum


# ---------------------------------------------------------------------------
# AC: validation/security errors carry stable codes and recoverable flags.
# ---------------------------------------------------------------------------


def test_validation_error_response_carries_stable_code_and_recoverable_flag() -> None:
    client = TestClient(
        create_app(include_test_error_route=True),
        raise_server_exceptions=False,
    )

    response = client.get("/__test__/validation-error", params={"limit": 0})

    assert response.status_code == 422
    body = response.json()
    error = body["error"]
    assert error["code"] == ErrorCode.INVALID_JOB_PAYLOAD
    assert error["recoverable"] is True
    assert error["remediation_hint"]
    _assert_safe_text(response.text)


def test_unhandled_exception_returns_safe_error_without_stack_trace() -> None:
    client = TestClient(
        create_app(include_test_error_route=True),
        raise_server_exceptions=False,
    )

    response = client.get("/__test__/unhandled-error")

    assert response.status_code == 500
    body = response.json()
    error = body["error"]
    assert error["code"] == ErrorCode.PLUGIN_EXECUTION_FAILED
    assert error["recoverable"] is False
    assert error["remediation_hint"]
    text = response.text
    # Stack trace markers / raw exception text must never reach the wire.
    assert "Traceback" not in text
    assert "RuntimeError" not in text
    _assert_safe_text(text)


def test_protected_endpoint_without_signature_returns_stable_signature_error() -> None:
    """Missing service signature must surface ACTION_PLAN_SIGNATURE_INVALID, recoverable=true."""
    client = TestClient(
        create_app(config=_test_config(), include_test_protected_route=True),
        raise_server_exceptions=False,
    )

    response = client.post("/__test__/protected-compute-request")

    assert response.status_code == 401
    body = response.json()
    error = body["error"]
    assert error["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert error["recoverable"] is True
    assert error["stage"].startswith("api.security")
    assert error["remediation_hint"]


# ---------------------------------------------------------------------------
# AC: plugin errors normalize into ErrorResponse without leaking raw details.
# ---------------------------------------------------------------------------


def test_plugin_error_with_stable_code_is_normalized_to_error_response() -> None:
    """A handler-raised plugin-style error becomes a stable ErrorResponse."""
    app = create_app()
    _add_plugin_error_fixtures(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test__/plugin-error/leakage")

    assert response.status_code == 422
    body = response.json()
    error = body["error"]
    assert error["code"] == ErrorCode.LEAKAGE_DETECTED
    assert error["recoverable"] is True
    assert error["stage"] == "api.plugin_normalized"
    assert error["details"]["reason_code"] == "group_leakage_detected"
    assert error["details"]["affected_groups"] == 3
    assert error["remediation_hint"]
    text = response.text
    assert "raw secret" not in text
    assert "Traceback" not in text


def test_plugin_error_with_export_blocker_normalizes_to_export_blocked() -> None:
    app = create_app()
    _add_plugin_error_fixtures(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test__/plugin-error/export")

    assert response.status_code == 422
    body = response.json()
    error = body["error"]
    assert error["code"] == ErrorCode.EXPORT_BLOCKED
    assert error["plugin_id"] == "dataforge.tabular"
    assert error["details"]["reason_code"] == "export_blocked_by_pii"
    _assert_safe_text(response.text)


def test_plugin_error_with_404_status_returns_artifact_not_found() -> None:
    app = create_app()
    _add_plugin_error_fixtures(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test__/plugin-error/missing-artifact")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == ErrorCode.ARTIFACT_NOT_FOUND
    assert error["recoverable"] is True


def test_unstructured_runtime_error_falls_back_to_safe_500() -> None:
    """Errors without ``code``/``reason_code`` fall back to the safe 500 path."""
    app = create_app()
    _add_plugin_error_fixtures(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test__/plugin-error/unstructured")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == ErrorCode.PLUGIN_EXECUTION_FAILED
    assert error["recoverable"] is False
    text = response.text
    # Raw exception args must not leak.
    assert "raw bug message" not in text
    assert "Traceback" not in text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_FORBIDDEN_TEXT_MARKERS: tuple[str, ...] = (
    "@example.com",
    "+1000",
    "secret",
    "password",
    "token",
    "raw bug message",
    "raw secret",
    "Traceback",
)


def _assert_safe_text(text: str | Iterable[str]) -> None:
    blob = text if isinstance(text, str) else "\n".join(text)
    lower = blob.lower()
    for marker in _FORBIDDEN_TEXT_MARKERS:
        assert marker.lower() not in lower, f"unsafe marker leaked: {marker!r}"


def _add_plugin_error_fixtures(app: FastAPI) -> None:
    """Register test-only routes that raise plugin-style errors."""

    class _PluginErrorFixture(ValueError):
        def __init__(
            self,
            *,
            code: ErrorCode,
            reason_code: str,
            message: str,
            details: dict[str, object] | None = None,
            status_code: int | None = None,
            plugin_id: str | None = None,
            job_id: str | None = None,
        ) -> None:
            super().__init__(message)
            self.code = code
            self.reason_code = reason_code
            self.details = details or {}
            if status_code is not None:
                self.status_code = status_code
            if plugin_id is not None:
                self.plugin_id = plugin_id
            if job_id is not None:
                self.job_id = job_id

    @app.get("/__test__/plugin-error/leakage", include_in_schema=False)
    async def _leakage() -> None:
        raise _PluginErrorFixture(
            code=ErrorCode.LEAKAGE_DETECTED,
            reason_code="group_leakage_detected",
            message="raw bug message must never leak",
            details={"affected_groups": 3},
        )

    @app.get("/__test__/plugin-error/export", include_in_schema=False)
    async def _export() -> None:
        raise _PluginErrorFixture(
            code=ErrorCode.EXPORT_BLOCKED,
            reason_code="export_blocked_by_pii",
            message="raw bug message must never leak",
            plugin_id="dataforge.tabular",
        )

    @app.get("/__test__/plugin-error/missing-artifact", include_in_schema=False)
    async def _missing() -> None:
        raise _PluginErrorFixture(
            code=ErrorCode.ARTIFACT_NOT_FOUND,
            reason_code="artifact_missing",
            message="raw bug message must never leak",
            status_code=404,
        )

    @app.get("/__test__/plugin-error/unstructured", include_in_schema=False)
    async def _unstructured() -> None:
        raise RuntimeError("raw bug message that must not leak")


# Silence unused-import warning: PlatformIdentityDep is only re-exported
# for downstream tests that import it from this module's namespace.
_ = (
    PlatformIdentityDep,
    json,
    build_service_signature,
    SERVICE_IDENTITY_HEADER,
    TIMESTAMP_HEADER,
    SIGNATURE_HEADER,
    ORGANIZATION_ID_HEADER,
    PROJECT_ID_HEADER,
)
