"""Tests for TASK-034: MethodRecommendation builder and MethodScore."""

from __future__ import annotations

from app.domain import (
    MethodCandidate,
    MethodCandidateStatus,
    MethodRecommendation,
    PolicyStatus,
    TabularProfileReport,
)
from app.kernel import (
    IMPUTATION_POLICY_COMPONENT_WEIGHTS,
    METHOD_SCORE_FORMULA,
    METHOD_SCORE_NOTATION_FORMULA,
    BuildMethodRecommendationsRequest,
    build_method_recommendations,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload


def test_method_recommendations_from_demo_profile_validate_contract() -> None:
    """Step 1: run MethodRecommendation builder and validate outputs."""
    recommendations = _build_recommendations()

    assert {recommendation.action_type for recommendation in recommendations} == {
        "IMPUTE_MISSING_VALUES",
        "AUGMENT_RARE_CLASS",
    }
    pack = load_contract_pack()
    for recommendation in recommendations:
        validate_contract_payload(
            pack,
            "method_recommendation",
            recommendation.model_dump(mode="json"),
        )


def test_monthly_income_recommends_group_median_and_blocks_target_imputation() -> None:
    """Step 2: monthly_income gets group_median; target imputation is blocked."""
    recommendation = _recommendation_by_action("IMPUTE_MISSING_VALUES")
    candidates = _candidates_by_id(recommendation)

    assert recommendation.target["column"] == "monthly_income"
    assert recommendation.recommended_method.method_id == "group_median"
    assert candidates["group_median"].status is MethodCandidateStatus.RECOMMENDED
    assert candidates["median"].status is MethodCandidateStatus.AVAILABLE
    assert "segment_dependent_missingness" in recommendation.recommended_method.reason_codes
    assert recommendation.blocked_methods[0].method_id == "target_imputation"
    assert (
        recommendation.blocked_methods[0].reason_code
        == "target_column_auto_imputation_forbidden"
    )
    assert recommendation.explanation.recommended_method == "group_median"
    assert "median" in recommendation.explanation.alternatives
    assert "target_imputation" in recommendation.explanation.blocked_methods


def test_method_score_decomposition_and_imputation_weights_match_policy() -> None:
    """Step 5: MethodScore formula, notation and DATASETS.md imputation weights."""
    recommendation = _recommendation_by_action("IMPUTE_MISSING_VALUES")
    method_score = _candidates_by_id(recommendation)["group_median"].method_score

    assert method_score.formula == METHOD_SCORE_FORMULA
    assert method_score.notation_formula == METHOD_SCORE_NOTATION_FORMULA
    assert method_score.policy_component_weights == IMPUTATION_POLICY_COMPONENT_WEIGHTS
    assert method_score.components["quality_score"] > method_score.components["risk_score"]
    assert method_score.weighted_components["risk_score"] < 0
    assert method_score.weighted_components["cost_score"] < 0


def test_rare_fraud_class_candidates_and_mvp_policy_statuses() -> None:
    """Steps 3-5: rare-class candidates include MVP-safe and disabled methods."""
    recommendation = _recommendation_by_action("AUGMENT_RARE_CLASS")
    candidates = _candidates_by_id(recommendation)

    assert recommendation.target["target_column"] == "is_fraud"
    assert recommendation.target["rare_class_label"] == "1"
    assert recommendation.recommended_method.method_id == "class_weights"
    assert {
        "class_weights",
        "undersampling_majority",
        "rare_case_review",
        "collect_more_real_data",
        "smote",
        "gaussian_copula",
        "borderline_smote",
        "adasyn",
        "ctgan",
    } <= set(candidates)

    assert candidates["class_weights"].status is MethodCandidateStatus.RECOMMENDED
    assert candidates["smote"].policy_status is PolicyStatus.DISABLED_BY_READINESS
    assert candidates["gaussian_copula"].policy_status is PolicyStatus.ENABLED
    assert candidates["borderline_smote"].status is MethodCandidateStatus.DISABLED_BY_READINESS
    assert candidates["adasyn"].status is MethodCandidateStatus.DISABLED_BY_READINESS
    assert candidates["ctgan"].status is MethodCandidateStatus.DISABLED_BY_POLICY

    for method_id in (
        "class_weights",
        "undersampling_majority",
        "rare_case_review",
        "collect_more_real_data",
    ):
        assert candidates[method_id].method_score.value > 0


def _recommendation_by_action(action_type: str) -> MethodRecommendation:
    for recommendation in _build_recommendations():
        if recommendation.action_type == action_type:
            return recommendation
    raise AssertionError(f"missing recommendation action {action_type}")


def _candidates_by_id(
    recommendation: MethodRecommendation,
) -> dict[str, MethodCandidate]:
    return {candidate.method_id: candidate for candidate in recommendation.candidate_methods}


def _build_recommendations() -> tuple[MethodRecommendation, ...]:
    return build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_profile())
    )


def _demo_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example
        for example in pack.examples
        if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)
