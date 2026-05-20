"""Tests for typed environment configuration."""

import pytest

from app.kernel.config import ConfigError, RuntimeProfile, load_config


def demo_env() -> dict[str, str]:
    return {
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


def test_load_config_with_demo_env() -> None:
    config = load_config(demo_env())

    assert config.profile is RuntimeProfile.DEMO_STRICT
    assert config.object_storage.endpoint_url == "http://localhost:9000"
    assert config.object_storage.bucket_name == "dataforge-local"
    assert config.platform.callback_url == "http://platform.local/api/ml/jobs/callback"
    assert config.platform.service_signing_secret.get_secret_value() == "local-dev-signing-secret"
    assert config.platform.service_identity == "dataforge-platform"
    assert config.platform.signature_max_age_seconds == 300
    assert config.dagster.home == "/tmp/dataforge-dagster"
    assert config.policies.policy_config_path == "configs/policies/demo_strict.yaml"
    assert config.contract_pack_version == "local-fallback-v0.1.0-demo"
    assert config.profile_defaults.object_storage_profile == "local_minio"
    assert config.config_hash.startswith("sha256:")


def test_missing_required_env_raises_clear_error() -> None:
    env = demo_env()
    del env["DATAFORGE_SERVICE_SIGNING_SECRET"]

    with pytest.raises(ConfigError, match="Missing required environment variables"):
        load_config(env)


def test_allow_external_api_is_false_by_default() -> None:
    config = load_config(demo_env())

    assert config.external_ai.allow_external_api is False


def test_banking_strict_profile_defaults() -> None:
    env = demo_env()
    env["DATAFORGE_PROFILE"] = "banking_strict"

    config = load_config(env)

    assert config.profile is RuntimeProfile.BANKING_STRICT
    assert config.profile_defaults.object_storage_profile == "enterprise_s3_or_ceph"
    assert config.profile_defaults.egress_policy == "deny_by_default"


def test_invalid_boolean_env_raises_clear_error() -> None:
    env = demo_env()
    env["DATAFORGE_ALLOW_EXTERNAL_API"] = "maybe"

    with pytest.raises(ConfigError, match="Invalid boolean for DATAFORGE_ALLOW_EXTERNAL_API"):
        load_config(env)


def test_invalid_signature_max_age_env_raises_clear_error() -> None:
    env = demo_env()
    env["DATAFORGE_PLATFORM_SIGNATURE_MAX_AGE_SECONDS"] = "soon"

    with pytest.raises(
        ConfigError,
        match="Invalid integer for DATAFORGE_PLATFORM_SIGNATURE_MAX_AGE_SECONDS",
    ):
        load_config(env)
