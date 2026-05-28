"""Plugin manifest compatibility and capability registry tests."""

from __future__ import annotations

import pytest
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


def test_capabilities_endpoint_reports_static_plugin_readiness() -> None:
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    payload = response.json()
    readiness_by_plugin = {
        item["plugin_id"]: item["readiness"] for item in payload["plugins"]
    }
    assert readiness_by_plugin == {
        "dataforge.tabular": "implemented",
        "dataforge.text_ocr": "proof",
        "image_stub": "contract-ready",
        "audio_stub": "contract-ready",
        "video_stub": "contract-ready",
    }


def test_static_manifest_compatibility_and_execution_policy() -> None:
    registry = build_static_plugin_registry()
    payload = registry.capabilities()
    by_plugin = {plugin.plugin_id: plugin for plugin in payload.plugins}

    assert by_plugin["dataforge.tabular"].executable is True
    assert by_plugin["dataforge.text_ocr"].executable is True
    assert by_plugin["image_stub"].executable is False
    assert by_plugin["image_stub"].blocked_reason == "contract_ready_only"
    assert registry.can_execute(
        plugin_id="dataforge.tabular",
        capability_id="tabular_profile",
    )
    assert not registry.can_execute(
        plugin_id="image_stub",
        capability_id="image_technical_validation",
    )


def test_static_plugin_manager_delegates_execution_policy() -> None:
    manager = build_static_plugin_manager()

    assert manager.can_execute(
        plugin_id="dataforge.tabular",
        capability_id="tabular_profile",
    )
    with pytest.raises(PluginExecutionBlocked):
        manager.assert_can_execute(
            plugin_id="video_stub",
            capability_id="video_metadata_validation",
        )


def test_disabled_or_contract_only_plugins_cannot_execute_compute_actions() -> None:
    disabled_registry = CapabilityRegistry((_valid_manifest(enabled=False),))

    with pytest.raises(PluginExecutionBlocked) as disabled_exc:
        disabled_registry.assert_can_execute(
            plugin_id="dataforge.disabled",
            capability_id="disabled_capability",
        )

    with pytest.raises(PluginExecutionBlocked) as contract_ready_exc:
        build_static_plugin_registry().assert_can_execute(
            plugin_id="audio_stub",
            capability_id="audio_metadata_validation",
        )

    assert disabled_exc.value.code is ErrorCode.PLUGIN_NOT_ENABLED
    assert disabled_exc.value.reason_code == "plugin_disabled"
    assert contract_ready_exc.value.code is ErrorCode.PLUGIN_NOT_ENABLED
    assert contract_ready_exc.value.reason_code == "contract_ready_only"


def test_broken_manifest_fails_compatibility_validation() -> None:
    with pytest.raises(ValidationError):
        _valid_manifest(capability_ids=("duplicate", "duplicate"))


def _valid_manifest(
    *,
    enabled: bool = True,
    capability_ids: tuple[str, ...] = ("disabled_capability",),
) -> PluginManifest:
    return PluginManifest(
        plugin_id="dataforge.disabled",
        name="Disabled test plugin",
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
            can_mutate_source_data=False,
            logs_may_include_raw_payloads=False,
        ),
        deterministic=True,
    )
