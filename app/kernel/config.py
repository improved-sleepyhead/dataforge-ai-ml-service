"""Typed environment configuration for the ML compute service."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError


class ConfigError(ValueError):
    """Raised when environment configuration is missing or invalid."""


class RuntimeProfile(StrEnum):
    """Supported runtime and security profiles."""

    DEMO_STRICT = "demo_strict"
    BANKING_STRICT = "banking_strict"


class ObjectStorageSettings(BaseModel):
    """Object storage connection settings supplied by deployment config."""

    model_config = ConfigDict(frozen=True)

    endpoint_url: str = Field(min_length=1)
    bucket_name: str = Field(min_length=1)
    region: str = Field(default="local", min_length=1)
    prefix_root: str = Field(default="dataforge", min_length=1)


class PlatformSettings(BaseModel):
    """Platform callback and service identity settings."""

    model_config = ConfigDict(frozen=True)

    callback_url: str = Field(min_length=1)
    service_signing_secret: SecretStr = Field(min_length=1)


class DagsterSettings(BaseModel):
    """Dagster runtime settings for compute-plane materialization."""

    model_config = ConfigDict(frozen=True)

    home: str = Field(min_length=1)
    job_name: str = Field(default="dataforge_analyze_dataset", min_length=1)
    run_queue: str = Field(default="default", min_length=1)


class PolicySettings(BaseModel):
    """Versioned policy/config paths used by reports and lineage."""

    model_config = ConfigDict(frozen=True)

    policy_config_path: str = Field(min_length=1)
    decision_policy_path: str = Field(min_length=1)
    score_policy_path: str = Field(min_length=1)


class ExternalAISettings(BaseModel):
    """External AI connector policy."""

    model_config = ConfigDict(frozen=True)

    allow_external_api: bool = False
    provider_allowlist_path: str | None = None


class ProfileDefaults(BaseModel):
    """Documented security posture derived from the selected runtime profile."""

    model_config = ConfigDict(frozen=True)

    object_storage_profile: str
    pii_detection: str
    audit_mode: str
    egress_policy: str
    secrets_provider: str
    tenant_isolation: str
    policy_management: str


class ServiceConfig(BaseModel):
    """Top-level typed configuration for the Python ML service."""

    model_config = ConfigDict(frozen=True)

    profile: RuntimeProfile = RuntimeProfile.DEMO_STRICT
    object_storage: ObjectStorageSettings
    platform: PlatformSettings
    dagster: DagsterSettings
    policies: PolicySettings
    contract_pack_version: str = Field(default="local-fallback-v0.1.0-demo", min_length=1)
    external_ai: ExternalAISettings = Field(default_factory=ExternalAISettings)
    profile_defaults: ProfileDefaults

    @property
    def config_hash(self) -> str:
        """Stable redacted hash for report metadata and lineage references."""
        redacted = self.model_dump(mode="json")
        redacted["platform"]["service_signing_secret"] = "***"
        encoded = json.dumps(redacted, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


REQUIRED_ENV_VARS = (
    "DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL",
    "DATAFORGE_OBJECT_STORAGE_BUCKET",
    "DATAFORGE_PLATFORM_CALLBACK_URL",
    "DATAFORGE_SERVICE_SIGNING_SECRET",
    "DATAFORGE_DAGSTER_HOME",
    "DATAFORGE_POLICY_CONFIG_PATH",
    "DATAFORGE_DECISION_POLICY_PATH",
    "DATAFORGE_SCORE_POLICY_PATH",
)


def load_config(env: Mapping[str, str] | None = None) -> ServiceConfig:
    """Load and validate service configuration from environment variables."""
    source = os.environ if env is None else env
    missing = [name for name in REQUIRED_ENV_VARS if not source.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required environment variables: {joined}")

    profile = _runtime_profile(source.get("DATAFORGE_PROFILE", RuntimeProfile.DEMO_STRICT.value))
    try:
        return ServiceConfig(
            profile=profile,
            object_storage=ObjectStorageSettings(
                endpoint_url=_required(source, "DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL"),
                bucket_name=_required(source, "DATAFORGE_OBJECT_STORAGE_BUCKET"),
                region=source.get("DATAFORGE_OBJECT_STORAGE_REGION", "local"),
                prefix_root=source.get("DATAFORGE_OBJECT_STORAGE_PREFIX_ROOT", "dataforge"),
            ),
            platform=PlatformSettings(
                callback_url=_required(source, "DATAFORGE_PLATFORM_CALLBACK_URL"),
                service_signing_secret=SecretStr(
                    _required(source, "DATAFORGE_SERVICE_SIGNING_SECRET")
                ),
            ),
            dagster=DagsterSettings(
                home=_required(source, "DATAFORGE_DAGSTER_HOME"),
                job_name=source.get("DATAFORGE_DAGSTER_JOB_NAME", "dataforge_analyze_dataset"),
                run_queue=source.get("DATAFORGE_DAGSTER_RUN_QUEUE", "default"),
            ),
            policies=PolicySettings(
                policy_config_path=_required(source, "DATAFORGE_POLICY_CONFIG_PATH"),
                decision_policy_path=_required(source, "DATAFORGE_DECISION_POLICY_PATH"),
                score_policy_path=_required(source, "DATAFORGE_SCORE_POLICY_PATH"),
            ),
            contract_pack_version=source.get(
                "DATAFORGE_CONTRACT_PACK_VERSION", "local-fallback-v0.1.0-demo"
            ),
            external_ai=ExternalAISettings(
                allow_external_api=_env_bool(source, "DATAFORGE_ALLOW_EXTERNAL_API", False),
                provider_allowlist_path=source.get("DATAFORGE_EXTERNAL_AI_ALLOWLIST_PATH"),
            ),
            profile_defaults=profile_defaults(profile),
        )
    except ValidationError as exc:
        raise ConfigError(f"Invalid service configuration: {exc}") from exc


def profile_defaults(profile: RuntimeProfile) -> ProfileDefaults:
    """Return documented profile behavior from the PRD runtime profile matrix."""
    if profile is RuntimeProfile.DEMO_STRICT:
        return ProfileDefaults(
            object_storage_profile="local_minio",
            pii_detection="regex_or_presidio_lite",
            audit_mode="json_artifacts_plus_platform_events",
            egress_policy="deny_unless_configured",
            secrets_provider="local_env",
            tenant_isolation="project_prefix_scoped",
            policy_management="static_versioned_files",
        )
    return ProfileDefaults(
        object_storage_profile="enterprise_s3_or_ceph",
        pii_detection="stronger_rules_and_models",
        audit_mode="append_only",
        egress_policy="deny_by_default",
        secrets_provider="vault_or_kms",
        tenant_isolation="enforced",
        policy_management="versioned_admin_workflow",
    )


def _runtime_profile(value: str) -> RuntimeProfile:
    try:
        return RuntimeProfile(value)
    except ValueError as exc:
        supported = ", ".join(profile.value for profile in RuntimeProfile)
        message = f"Invalid DATAFORGE_PROFILE={value!r}; expected one of: {supported}"
        raise ConfigError(message) from exc


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Invalid boolean for {name}: expected true/false")


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value
