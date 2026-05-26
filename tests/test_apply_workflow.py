"""Tests for TASK-057 APPLY_SELECTED_ACTIONS workflow launcher."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.adapters import FakePlatformMetadataClient
from app.api.schemas import ActionPlanExecuteApprovedRequest
from app.domain import ComputeRunStatus, DecisionAction, TabularProfileReport
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
from app.validation.contracts import load_contract_pack

_GENERATED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step 1: launch APPLY workflow with valid signed context
# ---------------------------------------------------------------------------


def test_launch_apply_workflow_materializes_full_asset_graph_and_records_progress() -> None:
    """All TASK-057 asset names must materialize; status events reach the fake platform."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash = _execute_request()

    result = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
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

    # Step 3: placeholder artifacts are materialized for observability,
    # but they are not exposed as final candidate/model-impact/export refs.
    assert result.candidate_artifact_uri is None
    assert result.candidate_artifact_hash is None
    assert result.synthetic_artifact_uri is None
    # imputation-only plan -> synthetic stage is emitted but flagged not_applicable
    assert result.synthetic_status == "not_applicable"
    assert result.model_impact_artifact_uri is None
    assert result.export_package_artifact_uri is None

    # Fake platform receives execution progress + final completed state
    snapshot = fake_platform.snapshot()
    job_events = list(snapshot.job_events)
    stages = [event.stage for event in job_events]
    assert "RUNNING_DECISION_CORE" in stages
    assert stages[-1] == "COMPLETED"
    last_event = job_events[-1]
    assert last_event.status is ComputeRunStatus.COMPLETED
    assert last_event.platform_job_id == request.platform_job_id


def test_synthetic_dataset_marked_provisional_when_synthetic_step_is_present() -> None:
    """Synthetic placeholder is observable but not exposed as a final artifact."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash = _execute_request()

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
    )

    assert result.synthetic_status == "provisional_placeholder"
    assert result.synthetic_artifact_uri is None
    assert "synthetic_dataset" in result.materialized_assets


def test_launch_apply_workflow_refuses_request_without_approval_metadata() -> None:
    """Defensive guard: launcher rejects requests that lack approval metadata."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash = _execute_request()
    unsigned_request = request.model_copy(update={"approval_metadata": None})

    with pytest.raises(ValueError, match="approval_metadata"):
        launch_apply_actions_workflow(
            request=unsigned_request,
            action_plan_hash=plan_hash,
            config=config,
            fake_platform=fake_platform,
        )

    # No platform job events should have been emitted on the rejected
    # path, because materialize() must never run without approval.
    snapshot = fake_platform.snapshot()
    assert snapshot.job_events == ()


def test_apply_placeholder_run_is_idempotent_without_final_artifact_refs() -> None:
    """Re-running placeholder APPLY shares a key but exposes no final refs."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash = _execute_request()

    first = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
    )

    fake_platform_two = FakePlatformMetadataClient()
    second = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform_two,
    )

    assert first.idempotency_key == second.idempotency_key
    assert first.candidate_artifact_uri is None
    assert second.candidate_artifact_uri is None
    assert first.export_package_artifact_uri is None
    assert second.export_package_artifact_uri is None


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


def _execute_request() -> tuple[ActionPlanExecuteApprovedRequest, str]:
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
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
            ),
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
    )
    return request, plan_hash


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        e for e in pack.examples if e.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)
