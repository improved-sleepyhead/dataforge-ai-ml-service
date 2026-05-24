"""FastAPI-facing launcher for ANALYZE_ONLY Dagster materializations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from dagster import AssetSelection, materialize

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.api.schemas import AnalyzeDatasetRequest
from app.domain import ComputeRunStatus, WorkflowType
from app.kernel.config import ServiceConfig
from app.orchestration.assets import (
    ANALYZE_ASSETS,
    BASE_ANALYZE_ASSET_KEYS,
    PREDICTION_ANALYZE_ASSET_KEYS,
)
from app.orchestration.definitions import build_definitions
from app.orchestration.resources import ComputeResources
from app.orchestration.run_context import RunContextResource
from app.orchestration.status_bridge import RunContext, RunStatusBridge


@dataclass(frozen=True)
class AnalyzeWorkflowResult:
    """Safe summary of an accepted local ANALYZE_ONLY materialization."""

    job_id: str
    status: ComputeRunStatus
    status_url: str
    expected_outputs: tuple[str, ...]
    materialized_assets: tuple[str, ...]
    mutates_dataset: bool = False


def launch_analyze_dataset_workflow(
    *,
    request: AnalyzeDatasetRequest,
    config: ServiceConfig,
    fake_platform: FakePlatformMetadataClient,
) -> AnalyzeWorkflowResult:
    """Materialize the requested ANALYZE_ONLY asset selection in process.

    This is the test/dev launcher for the API boundary. It wires Dagster with
    in-memory adapters and a fake platform status sink, so the endpoint can
    prove request signing, status callbacks, asset selection, and no-mutation
    behavior without requiring a real Dagster daemon, backend, or MinIO.
    """
    expected_outputs = expected_analyze_outputs(
        include_predictions=bool(request.prediction_artifact_refs)
    )
    run_context = RunContext(
        compute_run_id=f"compute_{request.platform_job_id}",
        platform_job_id=request.platform_job_id,
        organization_id=request.organization_id,
        project_id=request.project_id,
        dataset_id=request.dataset_id,
        dataset_version_id=request.dataset_version_id,
    )
    definitions = build_definitions(
        compute_resources=_build_in_memory_resources(
            config=config,
            request=request,
            fake_platform=fake_platform,
        ),
        run_context_resource=RunContextResource(
            run_context=run_context,
            workflow_type=WorkflowType.ANALYZE_ONLY,
        ),
    )

    result = materialize(
        ANALYZE_ASSETS,
        selection=AssetSelection.assets(*expected_outputs),
        resources=definitions.resources,
        raise_on_error=False,
    )
    if not result.success:
        RunStatusBridge(fake_platform=fake_platform).emit_failed(
            run_context=run_context,
            progress=0.0,
            error_code="ANALYZE_WORKFLOW_FAILED",
        )
        raise RuntimeError("ANALYZE_ONLY workflow materialization failed")

    RunStatusBridge(fake_platform=fake_platform).emit_completed(run_context=run_context)
    materialized_assets = tuple(
        _asset_name(event.asset_key) for event in result.get_asset_materialization_events()
    )
    return AnalyzeWorkflowResult(
        job_id=request.platform_job_id,
        status=ComputeRunStatus.ACCEPTED,
        status_url=f"/api/v1/jobs/{request.platform_job_id}/status",
        expected_outputs=expected_outputs,
        materialized_assets=materialized_assets,
        mutates_dataset=False,
    )


def expected_analyze_outputs(*, include_predictions: bool) -> tuple[str, ...]:
    """Return the asset names expected for a base or prediction-aware analyze run."""
    base = tuple(key.path[-1] for key in BASE_ANALYZE_ASSET_KEYS)
    if not include_predictions:
        return base
    prediction = tuple(key.path[-1] for key in PREDICTION_ANALYZE_ASSET_KEYS)
    return (*base, *prediction)


def _asset_name(asset_key: object) -> str:
    if not hasattr(asset_key, "path"):
        raise RuntimeError("Dagster materialization event did not include an asset key")
    path = asset_key.path
    if not isinstance(path, list | tuple) or not path:
        raise RuntimeError("Dagster materialization event included an invalid asset key")
    return str(path[-1])


def _build_in_memory_resources(
    *,
    config: ServiceConfig,
    request: AnalyzeDatasetRequest,
    fake_platform: FakePlatformMetadataClient,
) -> ComputeResources:
    scope = ObjectStorageScope(
        organization_id=request.organization_id,
        project_id=request.project_id,
        dataset_id=request.dataset_id,
    )
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=scope,
    )
    return ComputeResources(
        service_config=config,
        object_storage=storage,
        artifact_registry=ArtifactRegistry(storage=storage),
        fake_platform=fake_platform,
    )


class _InMemoryS3Client:
    """Tiny S3-compatible fake used only by the API analyze launcher."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

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


__all__ = [
    "AnalyzeWorkflowResult",
    "expected_analyze_outputs",
    "launch_analyze_dataset_workflow",
]
