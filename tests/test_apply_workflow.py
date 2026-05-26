"""Tests for TASK-057 APPLY_SELECTED_ACTIONS workflow launcher."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError
from app.api.schemas import ActionPlanExecuteApprovedRequest
from app.domain import (
    ArtifactRef,
    ComputeRunStatus,
    DecisionAction,
    ErrorCode,
    TabularProfileReport,
)
from app.ingestion import open_archive_path
from app.kernel import (
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    action_plan_integrity_hash,
    build_action_plan_preview,
    build_method_recommendations,
)
from app.kernel.action_plan import ActionPlanApprovalMetadata
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
from app.orchestration.apply_assets import APPLY_ASSET_KEYS
from app.orchestration.apply_workflow import (
    ApplyWorkflowResult,
    launch_apply_actions_workflow,
)
from app.orchestration.resources import ComputeResources
from app.validation.contracts import load_contract_pack
from tests.fixtures.demo_archive import build_demo_archive

_GENERATED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step 1: launch APPLY workflow with valid signed context
# ---------------------------------------------------------------------------


def test_launch_apply_workflow_materializes_full_asset_graph_and_records_progress(
    tmp_path: Path,
) -> None:
    """Approved APPLY produces real candidate/export refs and platform progress."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash, resources = _execute_request(tmp_path, config, fake_platform)

    result = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=request.source_artifacts,
        compute_resources=resources,
    )

    # Step 2: completed status with full asset graph materialized
    assert isinstance(result, ApplyWorkflowResult)
    assert result.status is ComputeRunStatus.ACCEPTED
    assert result.job_id == request.platform_job_id
    assert result.action_plan_id == request.action_plan.action_plan_id
    assert result.action_plan_hash == plan_hash
    assert result.mutates_dataset is True
    assert result.expected_outputs == tuple(
        key.path[-1] for key in APPLY_ASSET_KEYS
    )
    # All apply assets must be reported as materialized
    assert sorted(result.materialized_assets) == sorted(result.expected_outputs)

    # Step 3: real gated outputs are exposed only after validation/export builders run.
    assert result.candidate_artifact_uri is not None
    assert result.candidate_artifact_hash is not None
    assert result.synthetic_artifact_uri is None
    # imputation-only plan -> synthetic stage is emitted but flagged not_applicable
    assert result.synthetic_status == "not_applicable"
    assert result.export_package_artifact_uri is not None
    assert ".placeholder." not in result.candidate_artifact_uri
    assert ".placeholder." not in result.export_package_artifact_uri

    # Fake platform receives execution progress + final completed state
    snapshot = fake_platform.snapshot()
    job_events = list(snapshot.job_events)
    stages = [event.stage for event in job_events]
    assert "RUNNING_DECISION_CORE" in stages
    assert stages[-1] == "COMPLETED"
    last_event = job_events[-1]
    assert last_event.status is ComputeRunStatus.COMPLETED
    assert last_event.platform_job_id == request.platform_job_id


def test_synthetic_dataset_materialized_when_synthetic_step_is_present(
    tmp_path: Path,
) -> None:
    """Synthetic APPLY materializes a real synthetic report behind gates."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash, resources = _execute_request(tmp_path, config, fake_platform)

    # Patch the action plan to include a synthetic step (AUGMENT_RARE_CLASS).
    plan = request.action_plan
    new_steps = (
        plan.steps[0],
        plan.steps[0].model_copy(
            update={
                "step_id": "synthetic_step_001",
                "type": DecisionAction.AUGMENT_RARE_CLASS.value,
                "method_id": "smote",
                "depends_on": (plan.steps[0].step_id,),
                "config": {
                    "target_column": "is_fraud",
                    "rare_class_label": "1",
                    "method": "smote",
                    "source_split": "train",
                    "sampling_strategy": 0.20,
                    "k_neighbors": 3,
                },
                "random_seed": 42,
            }
        ),
    )
    patched_plan = plan.model_copy(update={"steps": new_steps})
    patched_request = request.model_copy(update={"action_plan": patched_plan})

    result = launch_apply_actions_workflow(
        request=patched_request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=patched_request.source_artifacts,
        compute_resources=resources,
    )

    assert result.synthetic_status in {"materialized", "blocked"}
    assert result.synthetic_artifact_uri is not None
    assert "synthetic_dataset" in result.materialized_assets


def test_launch_apply_workflow_refuses_request_without_approval_metadata(
    tmp_path: Path,
) -> None:
    """Defensive guard: launcher rejects requests that lack approval metadata."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash, resources = _execute_request(tmp_path, config, fake_platform)
    unsigned_request = request.model_copy(update={"approval_metadata": None})

    with pytest.raises(ValueError, match="approval_metadata"):
        launch_apply_actions_workflow(
            request=unsigned_request,
            action_plan_hash=plan_hash,
            config=config,
            fake_platform=fake_platform,
            input_artifacts=request.source_artifacts,
            compute_resources=resources,
        )

    # No platform job events should have been emitted on the rejected
    # path, because materialize() must never run without approval.
    snapshot = fake_platform.snapshot()
    assert snapshot.job_events == ()


def test_apply_real_gated_run_is_idempotent_for_same_inputs(tmp_path: Path) -> None:
    """Re-running real APPLY shares an idempotency key and stable final refs."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash, resources = _execute_request(tmp_path, config, fake_platform)

    first = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=request.source_artifacts,
        compute_resources=resources,
    )

    fake_platform_two = FakePlatformMetadataClient()
    second = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform_two,
        input_artifacts=request.source_artifacts,
        compute_resources=resources,
    )

    assert first.idempotency_key == second.idempotency_key
    assert first.candidate_artifact_uri == second.candidate_artifact_uri
    assert first.export_package_artifact_uri == second.export_package_artifact_uri


def test_apply_failed_validation_gate_does_not_expose_final_refs(tmp_path: Path) -> None:
    """Failed validation gates keep candidate/export refs out of the API result."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, _plan_hash, resources = _execute_request(
        tmp_path,
        config,
        fake_platform,
        source_bytes=_demo_transactions_with_email(tmp_path),
    )
    step = request.action_plan.steps[0]
    patched_step = step.model_copy(
        update={"config": {**step.config, "pii_restricted": True}}
    )
    patched_plan = request.action_plan.model_copy(update={"steps": (patched_step,)})
    patched_hash = action_plan_integrity_hash(patched_plan)
    patched_request = request.model_copy(
        update={
            "action_plan": patched_plan,
            "approval_metadata": request.approval_metadata.model_copy(
                update={"action_plan_hash": patched_hash}
            )
            if request.approval_metadata is not None
            else None,
        }
    )

    result = launch_apply_actions_workflow(
        request=patched_request,
        action_plan_hash=patched_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=patched_request.source_artifacts,
        compute_resources=resources,
    )

    assert result.candidate_artifact_uri is None
    assert result.candidate_artifact_hash is None
    assert result.export_package_artifact_uri is None
    assert sorted(result.materialized_assets) == sorted(result.expected_outputs)

    snapshot = fake_platform.snapshot()
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED
    validation_refs = [
        uri
        for event in snapshot.job_events
        for uri in event.details.get("artifact_uris", []) or []
        if "validation_gates_report" in uri
    ]
    assert validation_refs


def _test_config() -> ServiceConfig:
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
            service_signing_secret="test-signing-secret",  # type: ignore[arg-type]
            service_identity="dataforge-platform",
            signature_max_age_seconds=300,
        ),
        dagster=DagsterSettings(
            home="/tmp/dataforge-dagster-test",
            job_name="dataforge_apply_selected_actions",
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


def _execute_request(
    tmp_path: Path,
    config: ServiceConfig,
    fake_platform: FakePlatformMetadataClient,
    *,
    source_bytes: bytes | None = None,
) -> tuple[ActionPlanExecuteApprovedRequest, str, ComputeResources]:
    resources, source_artifact = _compute_resources_with_source_artifact(
        tmp_path=tmp_path,
        config=config,
        fake_platform=fake_platform,
        source_bytes=source_bytes,
    )
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_apply_001",
            source_dataset_version_id="dataset_version_v1",
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_apply",
            input_artifacts=(source_artifact.uri,),
            target_version_name="dataset_version_v2_candidate",
            created_at=_GENERATED_AT,
        )
    ).model_copy(
        update={
            "requires_approval": True,
            "approval_request_id": "approval_request_apply_001",
        }
    )
    plan_hash = action_plan_integrity_hash(plan)
    approval = ActionPlanApprovalMetadata(
        approval_id="approval_apply_001",
        approval_request_id="approval_request_apply_001",
        approved_by_user_id="platform_owner_apply",
        approved_at=_GENERATED_AT,
        action_plan_id=plan.action_plan_id,
        action_plan_hash=plan_hash,
        decision_report_id=plan.created_from_decision_report,
        source_dataset_version_id=plan.source_dataset_version_id,
    )
    request = ActionPlanExecuteApprovedRequest(
        platform_job_id="platform_job_apply_001",
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        source_dataset_version_id=plan.source_dataset_version_id,
        action_plan=plan,
        approval_metadata=approval,
        source_artifacts=(source_artifact,),
    )
    return request, plan_hash, resources


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        e for e in pack.examples if e.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def _compute_resources_with_source_artifact(
    *,
    tmp_path: Path,
    config: ServiceConfig,
    fake_platform: FakePlatformMetadataClient,
    source_bytes: bytes | None = None,
) -> tuple[ComputeResources, ArtifactRef]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=ObjectStorageScope(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
        ),
    )
    registry = ArtifactRegistry(storage=storage)
    if source_bytes is None:
        built = build_demo_archive(output_dir=tmp_path / "demo_archive")
        with open_archive_path(built.archive_path) as reader:
            transactions = reader.find_required_transactions().read_bytes()
    else:
        transactions = source_bytes
    source_artifact = registry.save_artifact(
        artifact_kind="raw_transactions",
        data=transactions,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_v1",
        created_by_job_id="compute_run_source_fixture",
        config_hash="sha256:" + "9" * 64,
    ).artifact_ref
    return (
        ComputeResources(
            service_config=config,
            object_storage=storage,
            artifact_registry=registry,
            fake_platform=fake_platform,
        ),
        source_artifact,
    )


def _demo_transactions_with_email(tmp_path: Path) -> bytes:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive_with_email")
    with open_archive_path(built.archive_path) as reader:
        transactions = reader.find_required_transactions().read_bytes().decode("utf-8")
    input_rows = list(csv.DictReader(transactions.splitlines()))
    fieldnames = list(input_rows[0])
    fieldnames.insert(-1, "customer_email")
    output = []
    for index, row in enumerate(input_rows):
        row["customer_email"] = f"customer{index:03d}@demo.invalid"
        output.append(row)
    text_sink = StringIO(newline="")
    writer = csv.DictWriter(text_sink, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output)
    return text_sink.getvalue().encode("utf-8")


class _InMemoryS3Client:
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
        record = self._object(Bucket, Key)
        body = record["Body"]
        if not isinstance(body, bytes):
            raise TypeError("InMemoryS3Client body must be bytes")
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        if not isinstance(body, bytes):
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
            if isinstance(body, bytes):
                contents.append({"Key": key, "Size": len(body)})
        return {"Contents": contents}

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc
