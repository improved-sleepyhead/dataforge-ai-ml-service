"""Plugin manifest and capability registry contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.common import NonEmptyStr
from app.domain.errors import ErrorCode
from app.domain.manifest import DataModality


class PluginReadiness(StrEnum):
    """Readiness level exposed through the capability registry."""

    IMPLEMENTED = "implemented"
    PROOF = "proof"
    CONTRACT_READY = "contract-ready"


class CapabilityDescriptor(BaseModel):
    """Stable capability advertised by one plugin manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: NonEmptyStr
    name: NonEmptyStr
    task_types: tuple[NonEmptyStr, ...] = Field(min_length=1)
    modalities: tuple[DataModality, ...] = Field(min_length=1)
    input_schema_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    output_schema_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    feeds: tuple[NonEmptyStr, ...] = Field(min_length=1)
    enabled: bool = True
    compute_action: bool = True
    requires_gpu: bool = False
    requires_external_egress: bool = False


class ResourceRequirements(BaseModel):
    """Bounded runtime requirements declared before plugin execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_cores: int = Field(ge=1)
    memory_mb: int = Field(ge=128)
    timeout_seconds: int = Field(ge=1)
    requires_gpu: bool = False


class PluginRuntimePolicy(BaseModel):
    """Safety policy attached to a plugin manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    network_egress_allowed: bool = False
    filesystem_read_scope: NonEmptyStr
    filesystem_write_scope: NonEmptyStr
    can_mutate_source_data: bool = False
    logs_may_include_raw_payloads: bool = False


class PluginManifest(BaseModel):
    """Validated allowlisted plugin manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    name: NonEmptyStr
    version: NonEmptyStr
    owner: NonEmptyStr
    readiness: PluginReadiness
    enabled: bool = True
    capabilities: tuple[CapabilityDescriptor, ...] = Field(min_length=1)
    required_permissions: tuple[NonEmptyStr, ...] = ()
    resource_requirements: ResourceRequirements
    runtime_policy: PluginRuntimePolicy
    deterministic: bool

    @model_validator(mode="after")
    def validate_manifest_safety(self) -> Self:
        capability_ids = [capability.capability_id for capability in self.capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability_id values must be unique within one plugin")
        if self.runtime_policy.can_mutate_source_data:
            raise ValueError("plugins must not mutate source dataset artifacts")
        if self.runtime_policy.logs_may_include_raw_payloads:
            raise ValueError("plugins must not log raw payloads")
        if any(capability.requires_external_egress for capability in self.capabilities):
            if not self.runtime_policy.network_egress_allowed:
                raise ValueError("external egress capability requires explicit runtime policy")
        return self


class CapabilitySummary(BaseModel):
    """API-safe plugin capability summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin_id: NonEmptyStr
    name: NonEmptyStr
    version: NonEmptyStr
    readiness: PluginReadiness
    enabled: bool
    executable: bool
    blocked_reason: str | None = None
    capabilities: tuple[CapabilityDescriptor, ...]


class CapabilitiesResponse(BaseModel):
    """Response body for the public capability registry endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugins: tuple[CapabilitySummary, ...]


class PluginExecutionBlocked(ValueError):
    """Raised when policy/readiness prevents plugin compute execution."""

    def __init__(self, *, code: ErrorCode, message: str, reason_code: str) -> None:
        super().__init__(message)
        self.code = code
        self.reason_code = reason_code


class CapabilityRegistry:
    """Read-only capability registry built from validated plugin manifests."""

    def __init__(self, manifests: tuple[PluginManifest, ...]) -> None:
        plugin_ids = [manifest.plugin_id for manifest in manifests]
        if len(plugin_ids) != len(set(plugin_ids)):
            raise ValueError("plugin_id values must be unique")
        self._manifests = manifests
        self._by_plugin_id = {manifest.plugin_id: manifest for manifest in manifests}

    @property
    def manifests(self) -> tuple[PluginManifest, ...]:
        return self._manifests

    def capabilities(self) -> CapabilitiesResponse:
        """Return readiness and execution status for all allowlisted plugins."""
        return CapabilitiesResponse(
            plugins=tuple(_summary_for_manifest(manifest) for manifest in self._manifests)
        )

    def manifest_for(self, plugin_id: str) -> PluginManifest:
        """Return one manifest or raise a stable plugin error."""
        try:
            return self._by_plugin_id[plugin_id]
        except KeyError as exc:
            raise PluginExecutionBlocked(
                code=ErrorCode.PLUGIN_NOT_ENABLED,
                message="Plugin is not allowlisted",
                reason_code="plugin_not_allowlisted",
            ) from exc

    def can_execute(self, *, plugin_id: str, capability_id: str) -> bool:
        """Return whether a plugin capability can execute compute actions."""
        try:
            self.assert_can_execute(plugin_id=plugin_id, capability_id=capability_id)
        except PluginExecutionBlocked:
            return False
        return True

    def assert_can_execute(self, *, plugin_id: str, capability_id: str) -> None:
        """Reject disabled, contract-only, or non-compute plugin capabilities."""
        manifest = self.manifest_for(plugin_id)
        if not manifest.enabled:
            raise PluginExecutionBlocked(
                code=ErrorCode.PLUGIN_NOT_ENABLED,
                message="Plugin is disabled by policy",
                reason_code="plugin_disabled",
            )
        if manifest.readiness is PluginReadiness.CONTRACT_READY:
            raise PluginExecutionBlocked(
                code=ErrorCode.PLUGIN_NOT_ENABLED,
                message="Plugin is contract-ready only and cannot execute compute actions",
                reason_code="contract_ready_only",
            )
        capability = _capability_for(manifest, capability_id)
        if not capability.enabled:
            raise PluginExecutionBlocked(
                code=ErrorCode.PLUGIN_NOT_ENABLED,
                message="Plugin capability is disabled by policy",
                reason_code="capability_disabled",
            )
        if not capability.compute_action:
            raise PluginExecutionBlocked(
                code=ErrorCode.PLUGIN_CONTRACT_FAILED,
                message="Plugin capability is not a compute action",
                reason_code="not_a_compute_action",
        )


class PluginManager:
    """Static plugin manager facade used before real plugin execution exists."""

    def __init__(self, *, capability_registry: CapabilityRegistry) -> None:
        self._capability_registry = capability_registry

    @property
    def capability_registry(self) -> CapabilityRegistry:
        return self._capability_registry

    def capabilities(self) -> CapabilitiesResponse:
        """Return the safe capability view for all static plugins."""
        return self._capability_registry.capabilities()

    def can_execute(self, *, plugin_id: str, capability_id: str) -> bool:
        """Return whether a plugin capability can execute compute actions."""
        return self._capability_registry.can_execute(
            plugin_id=plugin_id,
            capability_id=capability_id,
        )

    def assert_can_execute(self, *, plugin_id: str, capability_id: str) -> None:
        """Raise if a plugin capability is disabled or contract-only."""
        self._capability_registry.assert_can_execute(
            plugin_id=plugin_id,
            capability_id=capability_id,
        )


def _summary_for_manifest(manifest: PluginManifest) -> CapabilitySummary:
    blocked_reason: str | None = None
    executable = manifest.enabled and manifest.readiness is not PluginReadiness.CONTRACT_READY
    if not manifest.enabled:
        blocked_reason = "plugin_disabled"
    elif manifest.readiness is PluginReadiness.CONTRACT_READY:
        blocked_reason = "contract_ready_only"
    return CapabilitySummary(
        plugin_id=manifest.plugin_id,
        name=manifest.name,
        version=manifest.version,
        readiness=manifest.readiness,
        enabled=manifest.enabled,
        executable=executable,
        blocked_reason=blocked_reason,
        capabilities=manifest.capabilities,
    )


def _capability_for(manifest: PluginManifest, capability_id: str) -> CapabilityDescriptor:
    for capability in manifest.capabilities:
        if capability.capability_id == capability_id:
            return capability
    raise PluginExecutionBlocked(
        code=ErrorCode.PLUGIN_CONTRACT_FAILED,
        message="Plugin capability is not declared",
        reason_code="capability_not_declared",
    )
