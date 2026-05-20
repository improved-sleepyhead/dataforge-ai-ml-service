"""Dagster runtime tests for the ML compute plane.

Covers TASK-015 acceptance criteria:

* Dagster definitions load without errors;
* skeleton analyze job materializes against in-memory adapters;
* the fake platform client receives stage events through the status bridge;
* APPLY assets refuse to run without an approved ApplyRunContext;
* ANALYZE assets refuse to run under APPLY workflow type.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from typing import Any

import pytest
from dagster import (
    AssetSelection,
    Definitions,
    JobDefinition,
    materialize,
)

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.domain import ComputeRunStatus, WorkflowType
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
from app.orchestration import (
    ANALYZE_ASSET_KEYS,
    ANALYZE_ASSETS,
    ANALYZE_JOB_NAME,
    APPLY_ASSET_KEYS,
    APPLY_ASSETS,
    APPLY_JOB_NAME,
    ApplyRunContext,
    ComputeResources,
    RunContext,
    RunContextResource,
    build_definitions,
    build_local_demo_definitions,
)


def test_local_demo_definitions_load_without_errors() -> None:
    definitions = build_local_demo_definitions()

    assert isinstance(definitions, Definitions)
    repository = definitions.get_repository_def()

    assets_by_key = {key for key in repository.assets_defs_by_key}
    for expected_key in ANALYZE_ASSET_KEYS:
        assert expected_key in assets_by_key
    for expected_key in APPLY_ASSET_KEYS:
        assert expected_key in assets_by_key

    job_names = {job.name for job in repository.get_all_jobs()}
    assert ANALYZE_JOB_NAME in job_names
    assert APPLY_JOB_NAME in job_names


def test_skeleton_analyze_job_materializes_and_records_stage_events() -> None:
    definitions = build_local_demo_definitions()

    analyze_job = _job_by_name(definitions, ANALYZE_JOB_NAME)
    result = analyze_job.execute_in_process()

    assert result.success
    materialized_keys = {
        materialization.asset_key
        for materialization in result.get_asset_materialization_events()
    }
    assert materialized_keys == set(ANALYZE_ASSET_KEYS)

    fake_platform = _resolve_fake_platform(definitions)
    snapshot = fake_platform.snapshot()

    expected_stages = {
        "INGESTING",
        "BUILDING_MANIFEST",
        "PROFILING_TABULAR",
        "BUILDING_EVIDENCE",
        "RUNNING_DECISION_CORE",
    }
    actual_stages = {event.stage for event in snapshot.job_events}
    assert expected_stages.issubset(actual_stages)
    assert all(event.status is ComputeRunStatus.RUNNING for event in snapshot.job_events)
    assert all(
        event.platform_job_id == "platform_job_demo" for event in snapshot.job_events
    )


def test_skeleton_analyze_materialization_metadata_has_no_raw_payloads() -> None:
    definitions = build_local_demo_definitions()
    analyze_job = _job_by_name(definitions, ANALYZE_JOB_NAME)

    result = analyze_job.execute_in_process()

    forbidden_keys = {"raw_text", "raw_payload", "secret", "token", "email"}
    for event in result.get_asset_materialization_events():
        materialization = event.event_specific_data.materialization  # type: ignore[union-attr]
        metadata = materialization.metadata
        assert metadata["mutates_dataset"].value is False
        assert metadata["workflow_type"].value == WorkflowType.ANALYZE_ONLY.value
        assert metadata["skeleton"].value is True
        assert forbidden_keys.isdisjoint(metadata.keys())


def test_analyze_asset_refuses_to_run_under_apply_workflow_type() -> None:
    fake_platform = FakePlatformMetadataClient()
    compute_resources = _build_in_memory_resources(fake_platform=fake_platform)
    run_context_resource = RunContextResource(
        run_context=_run_context(),
        workflow_type=WorkflowType.APPLY_SELECTED_ACTIONS,
    )
    definitions = build_definitions(
        compute_resources=compute_resources,
        run_context_resource=run_context_resource,
    )

    result = materialize(
        ANALYZE_ASSETS,
        selection=AssetSelection.assets("raw_manifest"),
        resources=definitions.resources,
        raise_on_error=False,
    )

    assert not result.success


def test_apply_assets_require_apply_context() -> None:
    fake_platform = FakePlatformMetadataClient()
    compute_resources = _build_in_memory_resources(fake_platform=fake_platform)
    run_context_resource = RunContextResource(
        run_context=_run_context(),
        workflow_type=WorkflowType.APPLY_SELECTED_ACTIONS,
        apply_context=None,
    )
    definitions = build_definitions(
        compute_resources=compute_resources,
        run_context_resource=run_context_resource,
    )

    result = materialize(
        APPLY_ASSETS,
        selection=AssetSelection.assets("action_plan"),
        resources=definitions.resources,
        raise_on_error=False,
    )

    assert not result.success


def test_apply_assets_succeed_with_apply_context_and_emit_stage_events() -> None:
    fake_platform = FakePlatformMetadataClient()
    compute_resources = _build_in_memory_resources(fake_platform=fake_platform)
    run_context_resource = RunContextResource(
        run_context=_run_context(),
        workflow_type=WorkflowType.APPLY_SELECTED_ACTIONS,
        apply_context=ApplyRunContext(
            action_plan_id="action_plan_001",
            decision_report_id="decision_report_001",
        ),
    )
    definitions = build_definitions(
        compute_resources=compute_resources,
        run_context_resource=run_context_resource,
    )

    apply_job = _job_by_name(definitions, APPLY_JOB_NAME)
    result = apply_job.execute_in_process()

    assert result.success
    materialized = {
        event.asset_key for event in result.get_asset_materialization_events()
    }
    assert materialized == set(APPLY_ASSET_KEYS)

    snapshot = fake_platform.snapshot()
    apply_stages = {event.stage for event in snapshot.job_events}
    assert "RUNNING_DECISION_CORE" in apply_stages
    assert "COMPLETED" in apply_stages


def _job_by_name(definitions: Definitions, name: str) -> JobDefinition:
    for job in definitions.get_repository_def().get_all_jobs():
        if job.name == name:
            return job
    raise AssertionError(f"job {name!r} not found")


def _resolve_fake_platform(definitions: Definitions) -> FakePlatformMetadataClient:
    resources = dict(definitions.resources or {})
    resource_definition = resources["fake_platform"]
    # ResourceDefinition.hardcoded_resource exposes the wrapped value via the
    # resource_fn returned by the public API; build the resource without any
    # init context to obtain the underlying client.
    from dagster._core.execution.build_resources import build_resources

    with build_resources({"fake_platform": resource_definition}) as built:
        client = built.fake_platform
    assert isinstance(client, FakePlatformMetadataClient)
    return client


def _run_context() -> RunContext:
    return RunContext(
        compute_run_id="compute_run_test",
        platform_job_id="platform_job_test",
        organization_id="org_test",
        project_id="project_test",
        dataset_id="dataset_test",
        dataset_version_id="dataset_version_test",
    )


def _build_in_memory_resources(
    *, fake_platform: FakePlatformMetadataClient
) -> ComputeResources:
    config = _config_for_test()
    scope = ObjectStorageScope(
        organization_id="org_test",
        project_id="project_test",
        dataset_id="dataset_test",
    )
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=scope,
    )
    registry = ArtifactRegistry(storage=storage)
    return ComputeResources(
        service_config=config,
        object_storage=storage,
        artifact_registry=registry,
        fake_platform=fake_platform,
    )


def _config_for_test() -> ServiceConfig:
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
    """In-memory S3-compatible fake used only by orchestration tests."""

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

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        record = self._objects[(Bucket, Key)]
        body = record["Body"]
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        record = self._objects[(Bucket, Key)]
        body = record["Body"]
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> dict[str, Any]:
        contents = [
            {"Key": key, "Size": len(record["Body"])}
            for (bucket, key), record in sorted(self._objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents}


# Some pytest collection environments warn on unused imports; pin pytest as used.
_ = pytest
