"""Stable plugin SDK interfaces and result contracts."""

from app.plugin_sdk.manifest import (
    CapabilitiesResponse,
    CapabilityDescriptor,
    CapabilityRegistry,
    CapabilitySummary,
    PluginExecutionBlocked,
    PluginManager,
    PluginManifest,
    PluginReadiness,
    PluginRuntimePolicy,
    ResourceRequirements,
)

__all__ = [
    "CapabilitiesResponse",
    "CapabilityDescriptor",
    "CapabilityRegistry",
    "CapabilitySummary",
    "PluginExecutionBlocked",
    "PluginManager",
    "PluginManifest",
    "PluginReadiness",
    "PluginRuntimePolicy",
    "ResourceRequirements",
]
