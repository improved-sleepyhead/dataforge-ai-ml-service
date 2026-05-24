"""Tests for TASK-039 ActionPlan preview generation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import ActionPlan, ErrorCode, MethodRecommendation, TabularProfileReport
from app.kernel import (
    ActionPlanApprovalMetadata,
    ActionPlanExecutionError,
    ActionPlanPreviewError,
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    ValidateActionPlanExecutionRequest,
    action_plan_integrity_hash,
    build_action_plan_preview,
    build_method_recommendations,
    validate_action_plan_execution,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload


def test_action_plan_preview_from_imputation_and_augmentation_recommendations() -> None:
    """Steps 1-3: selected recommendations become planned idempotent steps."""
    recommendations = _method_recommendations()
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_001",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=tuple(
                recommendation.recommendation_id for recommendation in recommendations
            ),
            selected_method_overrides={},
            method_recommendations=recommendations,
            created_by_user_id="platform_user_123",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
            ),
            target_version_name="dataset_version_2_preview",
            created_at=datetime(2026, 5, 24, tzinfo=UTC),
        )
    )

    validate_contract_payload(load_contract_pack(), "action_plan", plan.model_dump(mode="json"))
    assert plan.execution_mode == "PREVIEW_ACTION_PLAN"
    assert plan.created_from_decision_report == "decision_report_001"
    assert plan.requires_approval is False
    assert [step.method_id for step in plan.steps] == ["group_median", "class_weights"]
    assert plan.steps[1].depends_on == (plan.steps[0].step_id,)
    assert all(step.idempotency_key.startswith("sha256:") for step in plan.steps)
    assert all(step.config_hash.startswith("sha256:") for step in plan.steps)
    assert "model_impact_check" in plan.validation_gates
    assert "candidate_dataset_version" in plan.expected_outputs


def test_action_plan_preview_rejects_ctgan_disabled_by_policy() -> None:
    """Step 4: disabled CTGAN selection is rejected before ActionPlan creation."""
    rare_class = next(
        recommendation
        for recommendation in _method_recommendations()
        if recommendation.action_type == "AUGMENT_RARE_CLASS"
    )

    with pytest.raises(ActionPlanPreviewError) as error:
        build_action_plan_preview(
            BuildActionPlanPreviewRequest(
                decision_report_id="decision_report_001",
                source_dataset_version_id="dataset_version_1",
                selected_decision_ids=(rare_class.recommendation_id,),
                selected_method_overrides={rare_class.recommendation_id: "ctgan"},
                method_recommendations=(rare_class,),
                created_by_user_id="platform_user_123",
                input_artifacts=(
                    "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
                ),
            )
        )

    assert error.value.code is ErrorCode.POLICY_BLOCKED
    assert error.value.reason_code == "method_not_selectable"
    assert error.value.details["method_id"] == "ctgan"


def _method_recommendations() -> tuple[MethodRecommendation, ...]:
    return build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example
        for example in pack.examples
        if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def test_validate_action_plan_execution_accepts_fake_approval_metadata() -> None:
    """TASK-040 step 2-3: valid platform approval metadata accepts execution state."""
    plan = _approved_action_plan()
    approval_metadata = _approval_metadata(plan)

    action_plan_hash = validate_action_plan_execution(
        ValidateActionPlanExecutionRequest(
            action_plan=plan,
            source_dataset_version_id=plan.source_dataset_version_id,
            approval_metadata=approval_metadata,
        )
    )

    assert action_plan_hash == approval_metadata.action_plan_hash
    assert action_plan_hash.startswith("sha256:")


def test_validate_action_plan_execution_requires_approval_metadata() -> None:
    plan = _approved_action_plan()

    with pytest.raises(ActionPlanExecutionError) as error:
        validate_action_plan_execution(
            ValidateActionPlanExecutionRequest(
                action_plan=plan,
                source_dataset_version_id=plan.source_dataset_version_id,
            )
        )

    assert error.value.code is ErrorCode.ACTION_PLAN_REQUIRES_APPROVAL
    assert error.value.reason_code == "approval_metadata_required"


def test_validate_action_plan_execution_rejects_tampered_step_integrity() -> None:
    plan = _approved_action_plan()
    tampered_step = plan.steps[0].model_copy(
        update={"idempotency_key": "sha256:" + "d" * 64}
    )
    tampered_plan = plan.model_copy(update={"steps": (tampered_step, *plan.steps[1:])})

    with pytest.raises(ActionPlanExecutionError) as error:
        validate_action_plan_execution(
            ValidateActionPlanExecutionRequest(
                action_plan=tampered_plan,
                source_dataset_version_id=tampered_plan.source_dataset_version_id,
                approval_metadata=_approval_metadata(tampered_plan),
            )
        )

    assert error.value.code is ErrorCode.ACTION_PLAN_PRECONDITION_FAILED
    assert error.value.reason_code == "step_idempotency_key_mismatch"


def _approved_action_plan() -> ActionPlan:
    recommendations = _method_recommendations()
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_001",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_123",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
            ),
            target_version_name="dataset_version_2_preview",
            created_at=datetime(2026, 5, 24, tzinfo=UTC),
        )
    )
    return plan.model_copy(
        update={"requires_approval": True, "approval_request_id": "approval_request_001"}
    )


def _approval_metadata(plan: ActionPlan) -> ActionPlanApprovalMetadata:
    return ActionPlanApprovalMetadata(
        approval_id="approval_001",
        approval_request_id="approval_request_001",
        approved_by_user_id="platform_owner_001",
        approved_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
        action_plan_id=plan.action_plan_id,
        action_plan_hash=action_plan_integrity_hash(plan),
        decision_report_id=plan.created_from_decision_report,
        source_dataset_version_id=plan.source_dataset_version_id,
    )
