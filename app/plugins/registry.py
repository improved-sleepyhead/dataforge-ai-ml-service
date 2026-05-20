"""Static allowlisted plugin manifests for the MVP compute plane."""

from __future__ import annotations

from app.domain import DataModality
from app.plugin_sdk import (
    CapabilityDescriptor,
    CapabilityRegistry,
    PluginManager,
    PluginManifest,
    PluginReadiness,
    PluginRuntimePolicy,
    ResourceRequirements,
)


def build_static_plugin_manager() -> PluginManager:
    """Build the static MVP plugin manager."""
    return PluginManager(capability_registry=build_static_plugin_registry())


def build_static_plugin_registry() -> CapabilityRegistry:
    """Build the static MVP plugin capability registry."""
    return CapabilityRegistry(_static_manifests())


def _static_manifests() -> tuple[PluginManifest, ...]:
    safe_runtime = PluginRuntimePolicy(
        network_egress_allowed=False,
        filesystem_read_scope="scoped_artifact_refs",
        filesystem_write_scope="scoped_compute_outputs",
        can_mutate_source_data=False,
        logs_may_include_raw_payloads=False,
    )
    default_resources = ResourceRequirements(
        cpu_cores=1,
        memory_mb=512,
        timeout_seconds=300,
        requires_gpu=False,
    )
    stub_resources = ResourceRequirements(
        cpu_cores=1,
        memory_mb=256,
        timeout_seconds=60,
        requires_gpu=False,
    )
    return (
        PluginManifest(
            plugin_id="dataforge.tabular",
            name="DataForge Tabular Diagnostics",
            version="0.1.0",
            owner="dataforge",
            readiness=PluginReadiness.IMPLEMENTED,
            enabled=True,
            capabilities=(
                CapabilityDescriptor(
                    capability_id="tabular_profile",
                    name="Tabular profile diagnostics",
                    task_types=("ANALYZE_DATASET",),
                    modalities=(DataModality.TABULAR,),
                    input_schema_refs=("manifest_row.v1",),
                    output_schema_refs=("evidence_bundle.v1", "dataforge_report.v1"),
                    feeds=("EvidenceBundle", "DataForgeReport"),
                ),
            ),
            required_permissions=("read_scoped_artifacts", "write_scoped_artifacts"),
            resource_requirements=default_resources,
            runtime_policy=safe_runtime,
            deterministic=True,
        ),
        PluginManifest(
            plugin_id="dataforge.text_ocr",
            name="DataForge Text/OCR Proof Plugin",
            version="0.1.0",
            owner="dataforge",
            readiness=PluginReadiness.PROOF,
            enabled=True,
            capabilities=(
                CapabilityDescriptor(
                    capability_id="text_ocr_validation",
                    name="Text/OCR JSONL validation and duplicate checks",
                    task_types=("ANALYZE_DATASET",),
                    modalities=(DataModality.TEXT, DataModality.DOCUMENT_OCR),
                    input_schema_refs=("manifest_row.v1",),
                    output_schema_refs=("evidence_bundle.v1", "review_queue.v1"),
                    feeds=("EvidenceBundle", "ReviewQueue"),
                ),
            ),
            required_permissions=("read_scoped_artifacts", "write_scoped_artifacts"),
            resource_requirements=default_resources,
            runtime_policy=safe_runtime,
            deterministic=True,
        ),
        _contract_ready_stub(
            plugin_id="image_stub",
            name="Image Contract Stub",
            modality=DataModality.IMAGE,
            capability_id="image_technical_validation",
            resources=stub_resources,
            runtime_policy=safe_runtime,
        ),
        _contract_ready_stub(
            plugin_id="audio_stub",
            name="Audio Contract Stub",
            modality=DataModality.AUDIO,
            capability_id="audio_metadata_validation",
            resources=stub_resources,
            runtime_policy=safe_runtime,
        ),
        _contract_ready_stub(
            plugin_id="video_stub",
            name="Video Contract Stub",
            modality=DataModality.VIDEO,
            capability_id="video_metadata_validation",
            resources=stub_resources,
            runtime_policy=safe_runtime,
        ),
    )


def _contract_ready_stub(
    *,
    plugin_id: str,
    name: str,
    modality: DataModality,
    capability_id: str,
    resources: ResourceRequirements,
    runtime_policy: PluginRuntimePolicy,
) -> PluginManifest:
    return PluginManifest(
        plugin_id=plugin_id,
        name=name,
        version="0.1.0",
        owner="dataforge",
        readiness=PluginReadiness.CONTRACT_READY,
        enabled=True,
        capabilities=(
            CapabilityDescriptor(
                capability_id=capability_id,
                name=name,
                task_types=("ANALYZE_DATASET",),
                modalities=(modality,),
                input_schema_refs=("manifest_row.v1",),
                output_schema_refs=("evidence_bundle.v1",),
                feeds=("EvidenceBundle",),
            ),
        ),
        required_permissions=("read_scoped_artifacts", "write_scoped_artifacts"),
        resource_requirements=resources,
        runtime_policy=runtime_policy,
        deterministic=True,
    )
