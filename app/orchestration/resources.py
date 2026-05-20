"""Dagster resources for the DataForge AI compute plane.

Dagster runs in the compute plane only. These resources expose typed adapters
to assets/jobs without leaking provider clients (boto3, real platform clients,
real DBs) into kernel/domain layers.

Resources here intentionally use ``ResourceDefinition.hardcoded_resource`` so
that ``Definitions`` can be assembled from already-instantiated objects. This
keeps Dagster startup deterministic and free of secret/env coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

from dagster import ResourceDefinition

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
)
from app.kernel.config import ServiceConfig

OBJECT_STORAGE_RESOURCE_KEY = "object_storage"
ARTIFACT_REGISTRY_RESOURCE_KEY = "artifact_registry"
FAKE_PLATFORM_RESOURCE_KEY = "fake_platform"
SERVICE_CONFIG_RESOURCE_KEY = "service_config"


@dataclass(frozen=True)
class ComputeResources:
    """Container of typed compute-plane adapters used by Dagster runs."""

    service_config: ServiceConfig
    object_storage: MinioObjectStorageAdapter
    artifact_registry: ArtifactRegistry
    fake_platform: FakePlatformMetadataClient

    def to_dagster_mapping(self) -> dict[str, ResourceDefinition]:
        """Return a Dagster resource mapping wrapping already-built adapters.

        Adapters are wrapped via ``ResourceDefinition.hardcoded_resource`` so
        they can be passed directly into ``Definitions(resources=...)`` and
        reused by ``materialize(...)`` calls in tests/local runs without
        any env-dependent initialization.
        """
        return {
            SERVICE_CONFIG_RESOURCE_KEY: ResourceDefinition.hardcoded_resource(
                self.service_config
            ),
            OBJECT_STORAGE_RESOURCE_KEY: ResourceDefinition.hardcoded_resource(
                self.object_storage
            ),
            ARTIFACT_REGISTRY_RESOURCE_KEY: ResourceDefinition.hardcoded_resource(
                self.artifact_registry
            ),
            FAKE_PLATFORM_RESOURCE_KEY: ResourceDefinition.hardcoded_resource(
                self.fake_platform
            ),
        }


__all__ = [
    "ARTIFACT_REGISTRY_RESOURCE_KEY",
    "ComputeResources",
    "FAKE_PLATFORM_RESOURCE_KEY",
    "OBJECT_STORAGE_RESOURCE_KEY",
    "SERVICE_CONFIG_RESOURCE_KEY",
]
