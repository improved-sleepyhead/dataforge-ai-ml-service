"""TASK-066: connected plugin compatibility test suite.

This suite proves the five acceptance criteria from ``tasks.json`` against
the real plugin SDK and capability registry (no mocks):

1. ``PluginManifest`` validates: shape, readiness levels, capability ids,
   safety policy invariants (no source mutation, no raw-payload logs).
2. Readiness level is enforced: ``contract-ready`` plugins cannot execute
   compute actions even when allowlisted.
3. Disabled plugins cannot execute, with stable ``PLUGIN_NOT_ENABLED``
   error code and a structured ``reason_code``.
4. Plugin output schema is validated before it reaches the
   ``EvidenceBundle`` builder: malformed plugin reports are rejected by
   the contract pack with stable ``CONTRACT_VALIDATION_FAILED``, and
   ``ObjectAnalyticalPassport`` Pydantic validation also rejects them.
5. Plugin failure surfaces a structured ``ErrorResponse`` with stable
   error code, ``stage="api.plugin_normalized"``, and no raw exception
   leak in the response body.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.main import create_app
from app.domain import DataModality, ErrorCode
from app.plugin_sdk import (
    CapabilityDescriptor,
    CapabilityRegistry,
    PluginExecutionBlocked,
    PluginManifest,
    PluginReadiness,
    PluginRuntimePolicy,
    ResourceRequirements,
)
from app.plugins import build_static_plugin_manager, build_static_plugin_registry
from app.validation.contracts import (
    ContractValidationError,
    load_contract_pack,
    validate_contract_payload,
)

# ---------------------------------------------------------------------------
# Acceptance criterion 1 — Plugin manifest validates
# ---------------------------------------------------------------------------


def test_static_plugin_manifests_pass_compatibility_validation() -> None:
    """All allowlisted plugins must produce a valid PluginManifest."""
    registry = build_static_plugin_registry()
    plugin_ids = tuple(manifest.plugin_id for manifest in registry.manifests)

    # Every shipped plugin manifest is parsed and frozen as a Pydantic model.
    assert plugin_ids == (
        "dataforge.tabular",
        "dataforge.text_ocr",
        "image_stub",
        "audio_stub",
        "video_stub",
    )
    for manifest in registry.manifests:
        # Capability ids must be unique inside a single plugin.
        capability_ids = [c.capability_id for c in manifest.capabilities]
        assert len(capability_ids) == len(set(capability_ids))
        # Safety invariants required by PRD §3 / runtime_policy.
        assert manifest.runtime_policy.can_mutate_source_data is False
        assert manifest.runtime_policy.logs_may_include_raw_payloads is False


def test_manifest_with_duplicate_capability_ids_fails_validation() -> None:
    """Two capabilities with the same id must fail Pydantic validation."""
    with pytest.raises(ValidationError) as exc_info:
        _build_test_manifest(
            capability_ids=("tabular_profile", "tabular_profile"),
        )
    assert "capability_id" in str(exc_info.value)


def test_manifest_with_invalid_plugin_id_fails_validation() -> None:
    """plugin_id must match the documented regex (lowercase, ascii)."""
    with pytest.raises(ValidationError):
        _build_test_manifest(plugin_id="DataForge.Tabular")


def test_manifest_that_logs_raw_payloads_is_rejected() -> None:
    """Safety invariant: plugins must not declare raw-payload logging."""
    with pytest.raises(ValidationError) as exc_info:
        _build_test_manifest(logs_raw_payloads=True)
    assert "raw payloads" in str(exc_info.value)


def test_manifest_that_mutates_source_data_is_rejected() -> None:
    """Safety invariant: plugins must not declare source-data mutation."""
    with pytest.raises(ValidationError) as exc_info:
        _build_test_manifest(can_mutate_source=True)
    assert "must not mutate source dataset artifacts" in str(exc_info.value)


def test_manifest_with_egress_capability_without_runtime_policy_is_rejected() -> None:
    """external egress capability requires explicit runtime_policy enablement."""
    with pytest.raises(ValidationError):
        _build_test_manifest(requires_external_egress=True)


def test_capability_registry_rejects_duplicate_plugin_ids() -> None:
    """CapabilityRegistry does not accept two manifests with the same plugin_id."""
    manifest_a = _build_test_manifest(plugin_id="dataforge.test")
    manifest_b = _build_test_manifest(plugin_id="dataforge.test")
    with pytest.raises(ValueError, match="plugin_id"):
        CapabilityRegistry((manifest_a, manifest_b))


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — Readiness level enforced
# ---------------------------------------------------------------------------


def test_contract_ready_plugins_cannot_execute_even_when_allowlisted() -> None:
    """contract-ready plugins must report executable=False at the registry boundary."""
    registry = build_static_plugin_registry()
    summaries = {plugin.plugin_id: plugin for plugin in registry.capabilities().plugins}

    assert summaries["dataforge.tabular"].executable is True
    assert summaries["dataforge.tabular"].readiness == PluginReadiness.IMPLEMENTED
    assert summaries["dataforge.text_ocr"].executable is True
    assert summaries["dataforge.text_ocr"].readiness == PluginReadiness.PROOF

    for stub in ("image_stub", "audio_stub", "video_stub"):
        assert summaries[stub].executable is False
        assert summaries[stub].readiness == PluginReadiness.CONTRACT_READY
        assert summaries[stub].blocked_reason == "contract_ready_only"


def test_capabilities_endpoint_exposes_readiness_to_callers() -> None:
    """The /api/v1/capabilities endpoint must mirror the registry readiness view."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    by_plugin = {item["plugin_id"]: item for item in response.json()["plugins"]}
    assert by_plugin["image_stub"]["executable"] is False
    assert by_plugin["image_stub"]["blocked_reason"] == "contract_ready_only"
    assert by_plugin["dataforge.tabular"]["executable"] is True
    assert by_plugin["dataforge.tabular"]["readiness"] == "implemented"


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — Disabled plugin cannot execute
# ---------------------------------------------------------------------------


def test_disabled_plugin_cannot_execute_and_returns_stable_error_code() -> None:
    """Disabled plugins surface PLUGIN_NOT_ENABLED with reason_code=plugin_disabled."""
    manager = build_static_plugin_manager()
    disabled = _build_test_manifest(
        plugin_id="dataforge.disabled_test",
        enabled=False,
    )
    custom_registry = CapabilityRegistry((*manager.capability_registry.manifests, disabled))

    with pytest.raises(PluginExecutionBlocked) as exc_info:
        custom_registry.assert_can_execute(
            plugin_id="dataforge.disabled_test",
            capability_id="capability_a",
        )

    assert exc_info.value.code is ErrorCode.PLUGIN_NOT_ENABLED
    assert exc_info.value.reason_code == "plugin_disabled"
    assert custom_registry.can_execute(
        plugin_id="dataforge.disabled_test",
        capability_id="capability_a",
    ) is False


def test_contract_ready_plugin_cannot_execute_compute_action() -> None:
    """contract-ready plugins must raise PLUGIN_NOT_ENABLED with reason=contract_ready_only."""
    registry = build_static_plugin_registry()
    with pytest.raises(PluginExecutionBlocked) as exc_info:
        registry.assert_can_execute(
            plugin_id="image_stub",
            capability_id="image_technical_validation",
        )
    assert exc_info.value.code is ErrorCode.PLUGIN_NOT_ENABLED
    assert exc_info.value.reason_code == "contract_ready_only"


def test_unknown_plugin_id_is_rejected_with_stable_reason_code() -> None:
    registry = build_static_plugin_registry()
    with pytest.raises(PluginExecutionBlocked) as exc_info:
        registry.assert_can_execute(
            plugin_id="dataforge.does_not_exist",
            capability_id="anything",
        )
    assert exc_info.value.code is ErrorCode.PLUGIN_NOT_ENABLED
    assert exc_info.value.reason_code == "plugin_not_allowlisted"


def test_unknown_capability_id_for_known_plugin_is_rejected() -> None:
    registry = build_static_plugin_registry()
    with pytest.raises(PluginExecutionBlocked) as exc_info:
        registry.assert_can_execute(
            plugin_id="dataforge.tabular",
            capability_id="unknown_capability",
        )
    assert exc_info.value.code is ErrorCode.PLUGIN_CONTRACT_FAILED
    assert exc_info.value.reason_code == "capability_not_declared"


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — Plugin output schema validated before EvidenceBundle
# ---------------------------------------------------------------------------


def test_valid_plugin_output_passes_contract_validation_for_passport() -> None:
    """A real ``ObjectAnalyticalPassport`` example validates against the contract pack."""
    pack = load_contract_pack()
    payload = next(
        example.payload
        for example in pack.examples
        if example.name.startswith("object_analytical_passport.")
    )
    # Does not raise.
    validate_contract_payload(pack, "object_analytical_passport", payload)


def test_malformed_plugin_output_is_rejected_before_evidence_bundle() -> None:
    """A passport-shaped payload missing required identity is rejected by the contract."""
    pack = load_contract_pack()
    base_payload = next(
        example.payload
        for example in pack.examples
        if example.name.startswith("object_analytical_passport.")
    )
    broken = dict(base_payload)
    # Drop a required identity field that downstream EvidenceBundle relies on.
    broken_identity = dict(broken["identity"])
    broken_identity.pop("object_id")
    broken["identity"] = broken_identity

    with pytest.raises(ContractValidationError) as exc_info:
        validate_contract_payload(pack, "object_analytical_passport", broken)
    assert "object_analytical_passport validation failed" in str(exc_info.value)


def test_pydantic_passport_model_also_rejects_malformed_plugin_output() -> None:
    """The Pydantic ObjectAnalyticalPassport model is the second line of defense."""
    from app.domain import ObjectAnalyticalPassport

    pack = load_contract_pack()
    base_payload = next(
        example.payload
        for example in pack.examples
        if example.name.startswith("object_analytical_passport.")
    )
    broken = dict(base_payload)
    broken_identity = dict(broken["identity"])
    broken_identity.pop("object_id")
    broken["identity"] = broken_identity

    with pytest.raises(ValidationError):
        ObjectAnalyticalPassport.model_validate(broken)


def test_plugin_specific_report_schemas_validate_real_examples() -> None:
    """Plugin-specific output schemas (tabular, text/OCR) validate their examples."""
    pack = load_contract_pack()

    tabular_payload = next(
        example.payload
        for example in pack.examples
        if example.name.startswith("tabular_profile_report.")
    )
    validate_contract_payload(pack, "tabular_profile_report", tabular_payload)

    text_ocr_payload = next(
        example.payload
        for example in pack.examples
        if example.name.startswith("text_ocr_report.")
    )
    validate_contract_payload(pack, "text_ocr_report", text_ocr_payload)


# ---------------------------------------------------------------------------
# Acceptance criterion 5 — Plugin failure returns structured ErrorResponse
# ---------------------------------------------------------------------------


def test_plugin_execution_blocked_normalizes_into_safe_error_response() -> None:
    """A FastAPI route raising PluginExecutionBlocked yields a stable ErrorResponse."""
    app = create_app()
    _attach_plugin_failure_route(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test__/plugin-execution-blocked")

    assert response.status_code == 422
    body = response.json()
    error = body["error"]
    assert error["code"] == ErrorCode.PLUGIN_NOT_ENABLED
    assert error["stage"] == "api.plugin_normalized"
    assert error["details"]["reason_code"] == "plugin_disabled"
    # No raw exception text or tracebacks may leak through the response.
    assert "raw bug message must not leak" not in response.text
    assert "Traceback" not in response.text


def test_plugin_contract_failure_normalizes_into_contract_validation_failed() -> None:
    """A plugin contract violation surfaces CONTRACT_VALIDATION_FAILED."""
    app = create_app()
    _attach_plugin_contract_failure_route(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test__/plugin-contract-failure")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == ErrorCode.CONTRACT_VALIDATION_FAILED
    assert error["stage"] == "api.plugin_normalized"
    assert error["details"]["reason_code"] == "plugin_output_schema_invalid"
    assert error["details"]["schema_name"] == "object_analytical_passport"
    assert "raw bug message must not leak" not in response.text


def test_unstructured_plugin_runtime_error_falls_back_to_safe_500() -> None:
    """Plugin errors without ``code`` fall back to a generic, safe 500."""
    app = create_app()
    _attach_unstructured_plugin_failure_route(app)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/__test__/plugin-unstructured-failure")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == ErrorCode.PLUGIN_EXECUTION_FAILED
    assert error["recoverable"] is False
    # Raw runtime error text must not leak through the API boundary.
    assert "raw bug message must not leak" not in response.text
    assert "Traceback" not in response.text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_test_manifest(
    *,
    plugin_id: str = "dataforge.test",
    capability_ids: tuple[str, ...] = ("capability_a",),
    enabled: bool = True,
    requires_external_egress: bool = False,
    can_mutate_source: bool = False,
    logs_raw_payloads: bool = False,
) -> PluginManifest:
    return PluginManifest(
        plugin_id=plugin_id,
        name=f"Test plugin {plugin_id}",
        version="0.1.0",
        owner="dataforge",
        readiness=PluginReadiness.IMPLEMENTED,
        enabled=enabled,
        capabilities=tuple(
            CapabilityDescriptor(
                capability_id=capability_id,
                name=f"{capability_id} capability",
                task_types=("ANALYZE_DATASET",),
                modalities=(DataModality.TABULAR,),
                input_schema_refs=("manifest_row.v1",),
                output_schema_refs=("evidence_bundle.v1",),
                feeds=("EvidenceBundle",),
                requires_external_egress=requires_external_egress,
            )
            for capability_id in capability_ids
        ),
        required_permissions=("read_scoped_artifacts",),
        resource_requirements=ResourceRequirements(
            cpu_cores=1,
            memory_mb=256,
            timeout_seconds=60,
            requires_gpu=False,
        ),
        runtime_policy=PluginRuntimePolicy(
            network_egress_allowed=False,
            filesystem_read_scope="scoped_artifact_refs",
            filesystem_write_scope="scoped_compute_outputs",
            can_mutate_source_data=can_mutate_source,
            logs_may_include_raw_payloads=logs_raw_payloads,
        ),
        deterministic=True,
    )


def _attach_plugin_failure_route(app: FastAPI) -> None:
    """Register a test-only route that raises PluginExecutionBlocked."""

    @app.get("/__test__/plugin-execution-blocked", include_in_schema=False)
    async def _blocked() -> None:
        raise PluginExecutionBlocked(
            code=ErrorCode.PLUGIN_NOT_ENABLED,
            message="raw bug message must not leak",
            reason_code="plugin_disabled",
        )


def _attach_plugin_contract_failure_route(app: FastAPI) -> None:
    """Register a test-only route that raises a plugin contract failure."""

    class _PluginContractError(ValueError):
        def __init__(self) -> None:
            super().__init__("raw bug message must not leak")
            self.code = ErrorCode.CONTRACT_VALIDATION_FAILED
            self.reason_code = "plugin_output_schema_invalid"
            self.details: dict[str, object] = {
                "schema_name": "object_analytical_passport",
                "missing_field": "identity.object_id",
            }
            self.plugin_id = "dataforge.tabular"

    @app.get("/__test__/plugin-contract-failure", include_in_schema=False)
    async def _contract_failure() -> None:
        raise _PluginContractError()


def _attach_unstructured_plugin_failure_route(app: FastAPI) -> None:
    """Register a test-only route that raises an unstructured runtime error."""

    @app.get("/__test__/plugin-unstructured-failure", include_in_schema=False)
    async def _unstructured() -> None:
        raise RuntimeError("raw bug message must not leak")
