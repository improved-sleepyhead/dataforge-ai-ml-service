"""Unit tests for decision, action planning, review, report, and export contracts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.domain import (
    ActionPlan,
    DataForgeReport,
    DecisionReport,
    ExportPackage,
    MethodCandidateStatus,
    MethodRecommendation,
    ReviewQueue,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload


def example_payload(name: str) -> dict[str, Any]:
    pack = load_contract_pack()
    for example in pack.examples:
        if example.name == name:
            return example.payload
    raise AssertionError(f"missing example {name}")


def test_decision_action_review_report_and_export_examples_validate() -> None:
    pack = load_contract_pack()
    cases: list[tuple[str, str, type[BaseModel]]] = [
        ("decision_report.needs_review", "decision_report", DecisionReport),
        ("method_recommendation.imputation", "method_recommendation", MethodRecommendation),
        ("action_plan.imputation_preview", "action_plan", ActionPlan),
        ("review_queue.label_review", "review_queue", ReviewQueue),
        ("dataforge_report.analyze_only", "dataforge_report", DataForgeReport),
        ("export_package.ready", "export_package", ExportPackage),
    ]

    for example_name, schema_name, model in cases:
        payload = example_payload(example_name)
        parsed = model.model_validate(payload)
        validate_contract_payload(pack, schema_name, parsed.model_dump(mode="json"))


def test_decision_report_contains_dataset_decision_readiness_and_policy_versions() -> None:
    report = DecisionReport.model_validate(example_payload("decision_report.needs_review"))

    assert report.dataset_decision == "NEEDS_REVIEW"
    assert report.readiness.status == "NOT_READY_FOR_EXPORT"
    assert report.critical_blockers[0].code == "PII_UNMASKED"
    assert report.recommended_actions[0].recommendation_id == "rec_label_review_001"
    assert report.policy_versions.decision_policy == "decision_v0"


def test_method_recommendation_includes_disabled_by_policy_candidate() -> None:
    recommendation = MethodRecommendation.model_validate(
        example_payload("method_recommendation.imputation")
    )
    candidates_by_method = {
        candidate.method_id: candidate for candidate in recommendation.candidate_methods
    }

    assert candidates_by_method["group_median"].status is MethodCandidateStatus.RECOMMENDED
    assert candidates_by_method["global_median"].status is MethodCandidateStatus.AVAILABLE
    assert candidates_by_method["pmm"].status is MethodCandidateStatus.DISABLED_BY_POLICY
    assert recommendation.blocked_methods[0].method_id == "target_imputation"


def test_method_recommendation_requires_recommended_method_to_be_candidate() -> None:
    payload = example_payload("method_recommendation.imputation")
    payload["recommended_method"]["method_id"] = "knn_imputer"

    with pytest.raises(ValidationError, match="recommended_method"):
        MethodRecommendation.model_validate(payload)


def test_action_plan_step_without_idempotency_key_is_invalid() -> None:
    payload = example_payload("action_plan.imputation_preview")
    del payload["steps"][0]["idempotency_key"]

    with pytest.raises(ValidationError):
        ActionPlan.model_validate(payload)


def test_action_plan_rejects_unknown_step_dependency() -> None:
    payload = example_payload("action_plan.imputation_preview")
    payload["steps"][0]["depends_on"] = ["missing_step"]

    with pytest.raises(ValidationError, match="dependencies"):
        ActionPlan.model_validate(payload)
