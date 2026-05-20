"""Top-level Dagster definitions for the DataForge AI compute plane.

This module exposes:

* :func:`build_definitions` — build a :class:`dagster.Definitions` from
  already-instantiated adapters and a per-run context resource;
* :func:`build_local_demo_definitions` — build a deterministic local
  Definitions instance using in-memory adapters, suitable for ``dagster dev``
  smoke runs without a real backend, real MinIO, or real signing keys;
* ``defs`` — a module-level Definitions object loaded by the Dagster CLI
  (``dagster definitions list -m app.orchestration.definitions``).

The compute plane intentionally builds Definitions from already-built typed
adapters via ``ResourceDefinition.hardcoded_resource``. This keeps Dagster
startup deterministic and free of secret/env coupling, so test runs and
local smoke runs do not need real credentials.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from dagster import Definitions, ResourceDefinition
from dagster._core.definitions.unresolved_asset_job_definition import (
    UnresolvedAssetJobDefinition,
)

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.domain import WorkflowType
from app.kernel.config import (
    DagsterSettings,
    ExternalAISettings,
    ObjectStorageSettings,
    PlatformSettings,
    PolicySettings,
    RuntimeProfile,
    ServiceConfig,
    profile_defaults,
)
from app.orchestration.apply_assets import APPLY_ASSETS
from app.orchestration.assets import ANALYZE_ASSETS
from app.orchestration.jobs import build_analyze_job, build_apply_job
from app.orchestration.resources import (
    ComputeResources,
)
from app.orchestration.run_context import RunContextResource
from app.orchestration.status_bridge import RunContext

RUN_CONTEXT_RESOURCE_KEY = "run_context"


def build_definitions(
    *,
    compute_resources: ComputeResources,
    run_context_resource: RunContextResource,
    extra_jobs: Iterable[UnresolvedAssetJobDefinition] | None = None,
) -> Definitions:
    """Build a Definitions object from already-instantiated adapters."""
    resources = compute_resources.to_dagster_mapping()
    resources[RUN_CONTEXT_RESOURCE_KEY] = ResourceDefinition.hardcoded_resource(
        run_context_resource
    )

    jobs: list[UnresolvedAssetJobDefinition] = [build_analyze_job(), build_apply_job()]
    if extra_jobs is not None:
        jobs.extend(extra_jobs)

    return Definitions(
        assets=[*ANALYZE_ASSETS, *APPLY_ASSETS],
        jobs=jobs,
        resources=resources,
    )


def build_local_demo_definitions() -> Definitions:
    """Build deterministic local Definitions backed by in-memory adapters.

    The local demo Definitions are used by:

    * the Dagster definitions load check in CI/local;
    * ``dagster dev`` smoke runs without real platform credentials;
    * exploratory development of the compute graph.

    They never write to a real MinIO bucket or call a real backend. The
    storage adapter is wired through an in-memory S3 fake; the platform
    client is the in-memory fake used by the rest of the test/dev suite.
    """
    config = _local_demo_config()
    scope = ObjectStorageScope(
        organization_id="org_demo",
        project_id="project_demo",
        dataset_id="dataset_demo",
    )
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=scope,
    )
    registry = ArtifactRegistry(storage=storage)
    fake_platform = FakePlatformMetadataClient()

    compute_resources = ComputeResources(
        service_config=config,
        object_storage=storage,
        artifact_registry=registry,
        fake_platform=fake_platform,
    )
    run_context = RunContext(
        compute_run_id="compute_run_demo",
        platform_job_id="platform_job_demo",
        organization_id=scope.organization_id,
        project_id=scope.project_id,
        dataset_id=scope.dataset_id,
        dataset_version_id="dataset_version_demo",
    )
    run_context_resource = RunContextResource(
        run_context=run_context,
        workflow_type=WorkflowType.ANALYZE_ONLY,
    )
    return build_definitions(
        compute_resources=compute_resources,
        run_context_resource=run_context_resource,
    )


def _local_demo_config() -> ServiceConfig:
    profile = RuntimeProfile.DEMO_STRICT
    return ServiceConfig(
        profile=profile,
        object_storage=ObjectStorageSettings(
            endpoint_url="http://localhost:9000",
            bucket_name="dataforge-local",
            region="local",
            prefix_root="dataforge",
        ),
        platform=PlatformSettings(
            callback_url="http://platform.local/api/ml/jobs/callback",
            service_signing_secret="local-dev-signing-secret",  # type: ignore[arg-type]
            service_identity="dataforge-platform",
            signature_max_age_seconds=300,
        ),
        dagster=DagsterSettings(
            home="/tmp/dataforge-dagster",
            job_name="dataforge_analyze_dataset",
            run_queue="default",
        ),
        policies=PolicySettings(
            policy_config_path="configs/policies/demo_strict.yaml",
            decision_policy_path="configs/policies/decision_v0.yaml",
            score_policy_path="configs/policies/score_v0.yaml",
        ),
        contract_pack_version="local-fallback-v0.1.0-demo",
        external_ai=ExternalAISettings(allow_external_api=False),
        profile_defaults=profile_defaults(profile),
    )


class _InMemoryS3Client:
    """Tiny S3-compatible fake used only by the local demo Definitions."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, object]] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._objects[(Bucket, Key)]
        from io import BytesIO

        body = record["Body"]
        if not isinstance(body, bytes):  # pragma: no cover - defensive
            raise TypeError("InMemoryS3Client body must be bytes")
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._objects[(Bucket, Key)]
        body = record["Body"]
        if not isinstance(body, bytes):  # pragma: no cover - defensive
            raise TypeError("InMemoryS3Client body must be bytes")
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        contents: list[dict[str, object]] = []
        for (bucket, key), record in sorted(self._objects.items()):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            body = record["Body"]
            if not isinstance(body, bytes):  # pragma: no cover - defensive
                continue
            contents.append({"Key": key, "Size": len(body)})
        return {"Contents": contents}


defs = build_local_demo_definitions()


__all__ = [
    "RUN_CONTEXT_RESOURCE_KEY",
    "build_definitions",
    "build_local_demo_definitions",
    "defs",
]
