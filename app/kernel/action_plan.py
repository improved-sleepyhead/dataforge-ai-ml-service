"""ActionPlan preview builder for selected method recommendations."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain import (
    ActionPlan,
    ActionPlanStep,
    DecisionAction,
    ErrorCode,
    MethodCandidate,
    MethodCandidateStatus,
    MethodRecommendation,
    PolicyStatus,
    RetryPolicy,
    WorkflowType,
)
from app.domain.common import NonEmptyStr, S3Uri

ACTION_PLAN_SCHEMA_VERSION = "action_plan.v1"
DEFAULT_PLUGIN_VERSION = "0.1.0"
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    retryable_errors=("TRANSIENT_STORAGE_ERROR", "DAGSTER_WORKER_RESTART"),
)
_BASE_VALIDATION_GATES = ("schema_validation", "business_rules", "privacy_check")
_SYNTHETIC_VALIDATION_GATES = (
    "split_leakage_check",
    "schema_validation",
    "business_rules_check",
    "synthetic_quality_check",
    "privacy_check",
    "model_impact_check",
)
_MODEL_IMPACT_GATE = "model_impact_check"


class ActionPlanPreviewError(ValueError):
    """Raised when selected recommendations cannot form an ActionPlan preview."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.ACTION_PLAN_PRECONDITION_FAILED,
        status_code: int = 422,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.status_code = status_code
        self.details = {} if details is None else details


class BuildActionPlanPreviewRequest(BaseModel):
    """Inputs for deterministic ActionPlan preview generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_report_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    selected_decision_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    selected_method_overrides: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    method_recommendations: tuple[MethodRecommendation, ...] = Field(min_length=1)
    created_by_user_id: NonEmptyStr
    input_artifacts: tuple[S3Uri, ...]
    target_version_name: NonEmptyStr | None = None
    created_at: datetime | None = None


def build_action_plan_preview(request: BuildActionPlanPreviewRequest) -> ActionPlan:
    """Build an idempotent ActionPlan preview without mutating dataset artifacts."""
    recommendations = _recommendations_by_id(request.method_recommendations)
    selected = [
        _selected_recommendation(recommendations, selected_id)
        for selected_id in request.selected_decision_ids
    ]

    steps: list[ActionPlanStep] = []
    for recommendation in selected:
        method_id = request.selected_method_overrides.get(
            recommendation.recommendation_id,
            recommendation.recommended_method.method_id,
        )
        candidate = _select_candidate(recommendation=recommendation, method_id=method_id)
        depends_on = (steps[-1].step_id,) if steps else ()
        steps.append(
            _build_step(
                recommendation=recommendation,
                candidate=candidate,
                depends_on=depends_on,
                input_artifacts=request.input_artifacts,
            )
        )

    validation_gates = _ordered_unique(gate for step in steps for gate in step.validation_gates)
    expected_outputs = _ordered_unique(
        output for recommendation in selected for output in recommendation.expected_outputs
    )
    action_plan_id = _stable_id(
        "action_plan",
        {
            "decision_report_id": request.decision_report_id,
            "selected_decision_ids": request.selected_decision_ids,
            "selected_method_overrides": request.selected_method_overrides,
            "source_dataset_version_id": request.source_dataset_version_id,
        },
    )
    return ActionPlan(
        action_plan_id=action_plan_id,
        plan_schema_version=ACTION_PLAN_SCHEMA_VERSION,
        source_dataset_version_id=request.source_dataset_version_id,
        target_version_name=request.target_version_name
        or f"{request.source_dataset_version_id}_candidate_preview",
        created_from_decision_report=request.decision_report_id,
        selected_decision_ids=request.selected_decision_ids,
        created_by_user_id=request.created_by_user_id,
        policy_version=_policy_version(selected),
        requires_approval=any(
            recommendation.recommended_method.requires_approval for recommendation in selected
        ),
        approval_request_id=None,
        execution_mode=WorkflowType.PREVIEW_ACTION_PLAN,
        steps=tuple(steps),
        validation_gates=validation_gates,
        expected_outputs=expected_outputs,
        created_at=request.created_at or datetime.now(UTC),
    )


def _recommendations_by_id(
    recommendations: tuple[MethodRecommendation, ...],
) -> dict[str, MethodRecommendation]:
    indexed = {
        recommendation.recommendation_id: recommendation
        for recommendation in recommendations
    }
    if len(indexed) != len(recommendations):
        raise ActionPlanPreviewError(
            reason_code="duplicate_recommendation_id",
            message="Method recommendation ids must be unique.",
            details={"recommendation_count": len(recommendations)},
        )
    return indexed


def _selected_recommendation(
    recommendations: dict[str, MethodRecommendation],
    selected_id: str,
) -> MethodRecommendation:
    try:
        return recommendations[selected_id]
    except KeyError as exc:
        raise ActionPlanPreviewError(
            reason_code="selected_decision_not_found",
            message="Selected decision id was not found in method recommendations.",
            details={"selected_decision_id": selected_id},
        ) from exc


def _select_candidate(
    *,
    recommendation: MethodRecommendation,
    method_id: str,
) -> MethodCandidate:
    blocked = {blocked.method_id: blocked for blocked in recommendation.blocked_methods}
    if method_id in blocked:
        raise ActionPlanPreviewError(
            reason_code=blocked[method_id].reason_code,
            message="Selected method is blocked by policy.",
            code=ErrorCode.POLICY_BLOCKED,
            status_code=422,
            details={
                "recommendation_id": recommendation.recommendation_id,
                "method_id": method_id,
            },
        )
    candidates = {candidate.method_id: candidate for candidate in recommendation.candidate_methods}
    candidate = candidates.get(method_id)
    if candidate is None:
        raise ActionPlanPreviewError(
            reason_code="method_candidate_not_found",
            message="Selected method is not a candidate for this recommendation.",
            details={
                "recommendation_id": recommendation.recommendation_id,
                "method_id": method_id,
            },
        )
    if candidate.status not in {
        MethodCandidateStatus.RECOMMENDED,
        MethodCandidateStatus.AVAILABLE,
    } or candidate.policy_status is not PolicyStatus.ENABLED:
        raise ActionPlanPreviewError(
            reason_code="method_not_selectable",
            message="Selected method is disabled or blocked and cannot be added to ActionPlan.",
            code=ErrorCode.POLICY_BLOCKED,
            status_code=422,
            details={
                "recommendation_id": recommendation.recommendation_id,
                "method_id": method_id,
                "candidate_status": candidate.status.value,
                "policy_status": candidate.policy_status.value,
            },
        )
    return candidate


def _build_step(
    *,
    recommendation: MethodRecommendation,
    candidate: MethodCandidate,
    depends_on: tuple[str, ...],
    input_artifacts: tuple[str, ...],
) -> ActionPlanStep:
    method_id = candidate.method_id
    config = _step_config(recommendation=recommendation, method_id=method_id)
    step_type = _step_type(recommendation.action_type, method_id)
    config_hash = _stable_hash(
        {
            "action_type": recommendation.action_type,
            "config": config,
            "method_id": method_id,
            "recommendation_id": recommendation.recommendation_id,
        }
    )
    step_id = _step_id(recommendation=recommendation, method_id=method_id)
    return ActionPlanStep(
        step_id=step_id,
        type=step_type,
        depends_on=depends_on,
        idempotency_key=_stable_hash(
            {
                "config_hash": config_hash,
                "depends_on": depends_on,
                "step_id": step_id,
            }
        ),
        method_id=method_id,
        plugin_id=recommendation.recommended_method.plugin_id,
        plugin_version=DEFAULT_PLUGIN_VERSION,
        config_hash=config_hash,
        policy_version=recommendation.policy_version,
        validation_gates=_validation_gates(recommendation=recommendation, method_id=method_id),
        preconditions=_preconditions(recommendation=recommendation, method_id=method_id),
        input_artifacts=input_artifacts,
        output_artifact_kind=_output_artifact_kind(recommendation.action_type, method_id),
        config=config,
        random_seed=_random_seed(method_id),
        retry_policy=DEFAULT_RETRY_POLICY,
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _step_config(
    *,
    recommendation: MethodRecommendation,
    method_id: str,
) -> dict[str, Any]:
    target = dict(recommendation.target)
    if recommendation.action_type == DecisionAction.IMPUTE_MISSING_VALUES.value:
        config: dict[str, Any] = {
            "column": target.get("column"),
            "method": method_id,
            "add_missingness_indicator": True,
        }
        if method_id == "group_median":
            config["group_key"] = target.get("group_key", "customer_segment")
        return config
    if recommendation.action_type == DecisionAction.AUGMENT_RARE_CLASS.value:
        return {
            "target_column": target.get("target_column"),
            "rare_class_label": target.get("rare_class_label"),
            "method": method_id,
            "source_split": "train" if method_id in _synthetic_methods() else None,
        }
    return {"method": method_id, "target": target}


def _validation_gates(
    *,
    recommendation: MethodRecommendation,
    method_id: str,
) -> tuple[str, ...]:
    if recommendation.action_type == DecisionAction.AUGMENT_RARE_CLASS.value:
        if method_id in _synthetic_methods():
            return _SYNTHETIC_VALIDATION_GATES
        return ("schema_validation", "business_rules", _MODEL_IMPACT_GATE)
    gates = list(_BASE_VALIDATION_GATES)
    if recommendation.model_impact_required:
        gates.append(_MODEL_IMPACT_GATE)
    return tuple(gates)


def _preconditions(
    *,
    recommendation: MethodRecommendation,
    method_id: str,
) -> tuple[str, ...]:
    preconditions = ["source_version_is_immutable"]
    if recommendation.action_type == DecisionAction.IMPUTE_MISSING_VALUES.value:
        preconditions.append("target_column_not_used_for_imputation")
    if method_id in _synthetic_methods():
        preconditions.extend(("train_split_exists", "leakage_checks_passed"))
    return tuple(preconditions)


def _step_type(action_type: str, method_id: str) -> str:
    if action_type == DecisionAction.AUGMENT_RARE_CLASS.value and method_id == "class_weights":
        return "CONFIGURE_CLASS_WEIGHTS"
    return action_type


def _output_artifact_kind(action_type: str, method_id: str) -> str:
    if action_type == DecisionAction.AUGMENT_RARE_CLASS.value and method_id == "class_weights":
        return "TRAINING_STRATEGY_CONFIG"
    return "CANDIDATE_DATASET_VERSION"


def _random_seed(method_id: str) -> int | None:
    return 42 if method_id in _synthetic_methods() else None


def _synthetic_methods() -> set[str]:
    return {"smote", "gaussian_copula", "borderline_smote", "adasyn", "ctgan"}


def _step_id(*, recommendation: MethodRecommendation, method_id: str) -> str:
    return _slug(f"{recommendation.action_type}_{method_id}_{recommendation.issue_id}")


def _stable_id(prefix: str, payload: object) -> str:
    digest = _stable_hash(payload).removeprefix("sha256:")
    return f"{prefix}_{digest[:16]}"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _ordered_unique(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _policy_version(recommendations: list[MethodRecommendation]) -> str:
    versions = _ordered_unique(recommendation.policy_version for recommendation in recommendations)
    return versions[0] if len(versions) == 1 else "mixed_method_policies"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "step"


__all__ = [
    "ACTION_PLAN_SCHEMA_VERSION",
    "ActionPlanPreviewError",
    "BuildActionPlanPreviewRequest",
    "build_action_plan_preview",
]
