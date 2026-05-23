"""Policy-driven MethodRecommendation builder for ANALYZE_ONLY outputs."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from app.domain import (
    BlockedMethod,
    ClassImbalanceDiagnostics,
    ColumnMissingness,
    ColumnProfile,
    ColumnType,
    DecisionAction,
    DecisionReport,
    MethodCandidate,
    MethodCandidateStatus,
    MethodRecommendation,
    MethodRecommendationExplanation,
    MethodScore,
    PolicyStatus,
    RecommendedMethod,
    TabularProfileReport,
)
from app.domain.common import NonEmptyStr, Score

METHOD_SELECTION_POLICY_VERSION = "method_selection_v0"
METHOD_SCORE_FORMULA = (
    "MethodScore(m) = wq * QualityScore(m) - wr * RiskScore(m) - wc * CostScore(m) "
    "+ we * ExplainabilityScore(m) + wi * ExpectedModelImpact(m) "
    "+ wp * PolicyCompatibility(m)"
)
METHOD_SCORE_NOTATION_FORMULA = (
    "MethodScore(m) = w_q * QualityScore(m) - w_r * RiskScore(m) "
    "- w_c * CostScore(m) + w_e * ExplainabilityScore(m) "
    "+ w_i * ExpectedModelImpact(m) + w_p * PolicyCompatibility(m)"
)
METHOD_SCORE_FORMULA_WEIGHTS: dict[str, float] = {
    "wq": 0.30,
    "wr": 0.20,
    "wc": 0.10,
    "we": 0.10,
    "wi": 0.15,
    "wp": 0.15,
}
IMPUTATION_POLICY_COMPONENT_WEIGHTS: dict[str, float] = {
    "distribution_preservation": 0.20,
    "variance_preservation": 0.12,
    "business_rules_pass_rate": 0.15,
    "model_impact_delta": 0.18,
    "leakage_safety": 0.15,
    "privacy_safety": 0.10,
    "explainability": 0.05,
    "computational_cost": -0.05,
}

_SEGMENT_DEPENDENT_MISSINGNESS_GAP = 0.20
_RARE_CLASS_SHARE_THRESHOLD = 0.10
_IMBALANCE_RATIO_THRESHOLD = 3.0


class BuildMethodRecommendationsRequest(BaseModel):
    """Inputs needed to derive method recommendations from profile evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tabular_profile: TabularProfileReport
    decision_report: DecisionReport | None = None
    policy_version: NonEmptyStr = METHOD_SELECTION_POLICY_VERSION


def build_method_recommendations(
    request: BuildMethodRecommendationsRequest,
) -> tuple[MethodRecommendation, ...]:
    """Build method recommendations for tabular imputation and rare-class handling."""
    profile = request.tabular_profile
    columns_by_name = {column.name: column for column in profile.columns}
    recommendations: list[MethodRecommendation] = []

    if profile.missingness is not None:
        for missingness in profile.missingness.columns:
            if missingness.missing_count == 0:
                continue
            if missingness.column == profile.missingness.target_column:
                continue
            column = columns_by_name.get(missingness.column)
            if column is None:
                continue
            if _is_numeric(column):
                recommendations.append(
                    _build_numeric_imputation_recommendation(
                        missingness=missingness,
                        column=column,
                        target_column=profile.missingness.target_column,
                        policy_version=request.policy_version,
                    )
                )

    if _rare_class_needs_policy(profile.class_imbalance):
        recommendations.append(
            _build_rare_class_recommendation(
                class_imbalance=profile.class_imbalance,
                policy_version=request.policy_version,
            )
        )

    return tuple(recommendations)


def _build_numeric_imputation_recommendation(
    *,
    missingness: ColumnMissingness,
    column: ColumnProfile,
    target_column: str | None,
    policy_version: str,
) -> MethodRecommendation:
    segment_dependent = _is_segment_dependent(missingness)
    recommended_id = "group_median" if segment_dependent else "median"
    reason_codes = _imputation_reason_codes(
        segment_dependent=segment_dependent,
        target_column=target_column,
    )
    candidates = (
        _candidate(
            method_id="group_median",
            status=(
                MethodCandidateStatus.RECOMMENDED
                if recommended_id == "group_median"
                else MethodCandidateStatus.AVAILABLE
            ),
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.84 if segment_dependent else 0.72,
            risk_score=0.12,
            cost_score=0.18,
            explainability_score=0.94,
            expected_model_impact=0.72,
            policy_compatibility=0.95,
            policy_component_weights=IMPUTATION_POLICY_COMPONENT_WEIGHTS,
            reason_codes=reason_codes if recommended_id == "group_median" else ("target_safe",),
            reason=None,
            policy_version=policy_version,
        ),
        _candidate(
            method_id="median",
            status=(
                MethodCandidateStatus.RECOMMENDED
                if recommended_id == "median"
                else MethodCandidateStatus.AVAILABLE
            ),
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.70,
            risk_score=0.08,
            cost_score=0.08,
            explainability_score=0.98,
            expected_model_impact=0.55,
            policy_compatibility=0.95,
            policy_component_weights=IMPUTATION_POLICY_COMPONENT_WEIGHTS,
            reason_codes=reason_codes if recommended_id == "median" else ("target_safe",),
            reason=None,
            policy_version=policy_version,
        ),
        _candidate(
            method_id="missingness_indicator",
            status=MethodCandidateStatus.AVAILABLE,
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.66,
            risk_score=0.06,
            cost_score=0.10,
            explainability_score=0.92,
            expected_model_impact=0.58,
            policy_compatibility=0.90,
            policy_component_weights=IMPUTATION_POLICY_COMPONENT_WEIGHTS,
            reason_codes=("predictive_missingness_indicator_candidate",),
            reason="Missingness may carry signal and can be preserved as an indicator.",
            policy_version=policy_version,
        ),
        _candidate(
            method_id="pmm",
            status=MethodCandidateStatus.DISABLED_BY_POLICY,
            policy_status=PolicyStatus.DISABLED_BY_POLICY,
            quality_score=0.86,
            risk_score=0.18,
            cost_score=0.45,
            explainability_score=0.56,
            expected_model_impact=0.70,
            policy_compatibility=0.20,
            policy_component_weights=IMPUTATION_POLICY_COMPONENT_WEIGHTS,
            reason_codes=("method_disabled_by_policy", "mvp_readiness_not_enabled"),
            reason="PMM is a pilot capability and is disabled in the MVP profile.",
            policy_version=policy_version,
        ),
    )
    blocked_methods = (
        BlockedMethod(
            method_id="target_imputation",
            reason_code="target_column_auto_imputation_forbidden",
            reason="Target-aware imputation is blocked by policy for this column.",
        ),
    )
    alternatives = tuple(
        candidate.method_id
        for candidate in candidates
        if candidate.method_id != recommended_id
        and candidate.status is not MethodCandidateStatus.BLOCKED
    )
    return MethodRecommendation(
        recommendation_id=f"method_rec_impute_{_slug(column.name)}",
        issue_id=f"issue_missing_{_slug(column.name)}",
        action_type=DecisionAction.IMPUTE_MISSING_VALUES.value,
        policy_version=policy_version,
        target={
            "scope": "column",
            "column": column.name,
            "target_column": target_column,
            "missing_rate": missingness.missing_rate,
            "missing_count": missingness.missing_count,
        },
        recommended_method=RecommendedMethod(
            method_id=recommended_id,
            plugin_id="dataforge.tabular",
            readiness_level=4,
            requires_approval=False,
            reason_codes=reason_codes,
        ),
        candidate_methods=candidates,
        blocked_methods=blocked_methods,
        explanation=MethodRecommendationExplanation(
            recommended_method=recommended_id,
            why_this_method=_imputation_explanation(recommended_id),
            alternatives=alternatives,
            blocked_methods=("target_imputation",),
            reason_codes=reason_codes,
        ),
        expected_outputs=(
            "imputation_report",
            "candidate_dataset_version",
            "version_compare_report",
        ),
        model_impact_required=True,
    )


def _build_rare_class_recommendation(
    *,
    class_imbalance: ClassImbalanceDiagnostics | None,
    policy_version: str,
) -> MethodRecommendation:
    if class_imbalance is None:
        raise ValueError("class_imbalance is required for rare-class recommendation")
    reason_codes = (
        "rare_class_underrepresented",
        "train_time_safe_no_dataset_mutation",
        "split_required_for_synthetic_augmentation",
    )
    candidates = (
        _candidate(
            method_id="class_weights",
            status=MethodCandidateStatus.RECOMMENDED,
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.76,
            risk_score=0.05,
            cost_score=0.06,
            explainability_score=0.90,
            expected_model_impact=0.68,
            policy_compatibility=0.95,
            policy_component_weights={},
            reason_codes=reason_codes,
            reason=None,
            policy_version=policy_version,
        ),
        _candidate(
            method_id="undersampling_majority",
            status=MethodCandidateStatus.AVAILABLE,
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.62,
            risk_score=0.20,
            cost_score=0.08,
            explainability_score=0.84,
            expected_model_impact=0.52,
            policy_compatibility=0.82,
            policy_component_weights={},
            reason_codes=("rare_class_underrepresented", "majority_reduction_candidate"),
            reason="May reduce majority dominance, but can discard useful real examples.",
            policy_version=policy_version,
        ),
        _candidate(
            method_id="rare_case_review",
            status=MethodCandidateStatus.AVAILABLE,
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.70,
            risk_score=0.04,
            cost_score=0.35,
            explainability_score=0.96,
            expected_model_impact=0.50,
            policy_compatibility=0.90,
            policy_component_weights={},
            reason_codes=("rare_class_underrepresented", "human_review_candidate"),
            reason="Prioritizes rare examples for review before synthetic augmentation.",
            policy_version=policy_version,
        ),
        _candidate(
            method_id="collect_more_real_data",
            status=MethodCandidateStatus.AVAILABLE,
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.88,
            risk_score=0.02,
            cost_score=0.82,
            explainability_score=0.98,
            expected_model_impact=0.82,
            policy_compatibility=0.88,
            policy_component_weights={},
            reason_codes=("rare_class_underrepresented", "real_data_preferred"),
            reason="Real rare-class data is preferred when collection is feasible.",
            policy_version=policy_version,
        ),
        _candidate(
            method_id="smote",
            status=MethodCandidateStatus.DISABLED_BY_READINESS,
            policy_status=PolicyStatus.DISABLED_BY_READINESS,
            quality_score=0.78,
            risk_score=0.22,
            cost_score=0.22,
            explainability_score=0.66,
            expected_model_impact=0.74,
            policy_compatibility=0.72,
            policy_component_weights={},
            reason_codes=("split_required_for_synthetic_augmentation",),
            reason="SMOTE is policy-supported, but requires train-only split and leakage gates.",
            policy_version=policy_version,
        ),
        _candidate(
            method_id="gaussian_copula",
            status=MethodCandidateStatus.AVAILABLE,
            policy_status=PolicyStatus.ENABLED,
            quality_score=0.66,
            risk_score=0.28,
            cost_score=0.34,
            explainability_score=0.58,
            expected_model_impact=0.48,
            policy_compatibility=0.64,
            policy_component_weights={},
            reason_codes=("synthetic_distribution_candidate", "model_impact_required"),
            reason=(
                "Distribution-level synthetic generation is available but not a "
                "SMOTE replacement."
            ),
            policy_version=policy_version,
        ),
        _candidate(
            method_id="borderline_smote",
            status=MethodCandidateStatus.DISABLED_BY_READINESS,
            policy_status=PolicyStatus.DISABLED_BY_READINESS,
            quality_score=0.80,
            risk_score=0.30,
            cost_score=0.30,
            explainability_score=0.58,
            expected_model_impact=0.70,
            policy_compatibility=0.46,
            policy_component_weights={},
            reason_codes=("mvp_readiness_not_enabled", "split_required_for_synthetic_augmentation"),
            reason="Borderline-SMOTE is policy-controlled and not MVP-selectable yet.",
            policy_version=policy_version,
        ),
        _candidate(
            method_id="adasyn",
            status=MethodCandidateStatus.DISABLED_BY_READINESS,
            policy_status=PolicyStatus.DISABLED_BY_READINESS,
            quality_score=0.74,
            risk_score=0.36,
            cost_score=0.32,
            explainability_score=0.52,
            expected_model_impact=0.66,
            policy_compatibility=0.42,
            policy_component_weights={},
            reason_codes=("mvp_readiness_not_enabled", "label_noise_amplification_risk"),
            reason="ADASYN is disabled until readiness and label-noise gates exist.",
            policy_version=policy_version,
        ),
        _candidate(
            method_id="ctgan",
            status=MethodCandidateStatus.DISABLED_BY_POLICY,
            policy_status=PolicyStatus.DISABLED_BY_POLICY,
            quality_score=0.82,
            risk_score=0.42,
            cost_score=0.70,
            explainability_score=0.34,
            expected_model_impact=0.64,
            policy_compatibility=0.18,
            policy_component_weights={},
            reason_codes=("method_disabled_by_policy", "mvp_readiness_not_enabled"),
            reason="CTGAN is disabled by policy in the MVP profile.",
            policy_version=policy_version,
        ),
    )
    return MethodRecommendation(
        recommendation_id=f"method_rec_rare_class_{_slug(class_imbalance.rare_class_label)}",
        issue_id=f"issue_rare_class_{_slug(class_imbalance.rare_class_label)}",
        action_type=DecisionAction.AUGMENT_RARE_CLASS.value,
        policy_version=policy_version,
        target={
            "scope": "class",
            "target_column": class_imbalance.target_column,
            "rare_class_label": class_imbalance.rare_class_label,
            "rare_class_count": class_imbalance.rare_class_count,
            "rare_class_ratio": class_imbalance.rare_class_ratio,
            "imbalance_ratio": class_imbalance.imbalance_ratio,
        },
        recommended_method=RecommendedMethod(
            method_id="class_weights",
            plugin_id="dataforge.tabular",
            readiness_level=4,
            requires_approval=False,
            reason_codes=reason_codes,
        ),
        candidate_methods=candidates,
        blocked_methods=(),
        explanation=MethodRecommendationExplanation(
            recommended_method="class_weights",
            why_this_method=(
                "Class weights can address rare-class underrepresentation without mutating "
                "dataset artifacts before split and leakage gates exist."
            ),
            alternatives=tuple(
                candidate.method_id
                for candidate in candidates
                if candidate.method_id != "class_weights"
            ),
            blocked_methods=(),
            reason_codes=reason_codes,
        ),
        expected_outputs=(
            "rare_class_method_report",
            "model_impact_report",
            "candidate_dataset_version",
        ),
        model_impact_required=True,
    )


def _candidate(
    *,
    method_id: str,
    status: MethodCandidateStatus,
    policy_status: PolicyStatus,
    quality_score: Score,
    risk_score: Score,
    cost_score: Score,
    explainability_score: Score,
    expected_model_impact: Score,
    policy_compatibility: Score,
    policy_component_weights: dict[str, float],
    reason_codes: tuple[str, ...],
    reason: str | None,
    policy_version: str,
) -> MethodCandidate:
    method_score = _method_score(
        quality_score=quality_score,
        risk_score=risk_score,
        cost_score=cost_score,
        explainability_score=explainability_score,
        expected_model_impact=expected_model_impact,
        policy_compatibility=policy_compatibility,
        policy_component_weights=policy_component_weights,
        reason_codes=reason_codes,
        policy_version=policy_version,
    )
    return MethodCandidate(
        method_id=method_id,
        status=status,
        quality_score=quality_score,
        risk_score=risk_score,
        method_score=method_score,
        policy_status=policy_status,
        reason=reason,
    )


def _method_score(
    *,
    quality_score: Score,
    risk_score: Score,
    cost_score: Score,
    explainability_score: Score,
    expected_model_impact: Score,
    policy_compatibility: Score,
    policy_component_weights: dict[str, float],
    reason_codes: tuple[str, ...],
    policy_version: str,
) -> MethodScore:
    components = {
        "quality_score": quality_score,
        "risk_score": risk_score,
        "cost_score": cost_score,
        "explainability_score": explainability_score,
        "expected_model_impact": expected_model_impact,
        "policy_compatibility": policy_compatibility,
    }
    weighted = {
        "quality_score": METHOD_SCORE_FORMULA_WEIGHTS["wq"] * quality_score,
        "risk_score": -METHOD_SCORE_FORMULA_WEIGHTS["wr"] * risk_score,
        "cost_score": -METHOD_SCORE_FORMULA_WEIGHTS["wc"] * cost_score,
        "explainability_score": METHOD_SCORE_FORMULA_WEIGHTS["we"] * explainability_score,
        "expected_model_impact": METHOD_SCORE_FORMULA_WEIGHTS["wi"] * expected_model_impact,
        "policy_compatibility": METHOD_SCORE_FORMULA_WEIGHTS["wp"] * policy_compatibility,
    }
    raw_value = sum(weighted.values())
    return MethodScore(
        value=_clamp(raw_value),
        policy_version=policy_version,
        formula=METHOD_SCORE_FORMULA,
        notation_formula=METHOD_SCORE_NOTATION_FORMULA,
        formula_weights=METHOD_SCORE_FORMULA_WEIGHTS,
        components=components,
        weighted_components=weighted,
        policy_component_weights=policy_component_weights,
        reason_codes=reason_codes,
    )


def _imputation_reason_codes(
    *,
    segment_dependent: bool,
    target_column: str | None,
) -> tuple[str, ...]:
    codes = ["skewed_numeric_distribution"]
    if segment_dependent:
        codes.append("segment_dependent_missingness")
    codes.append("target_safe" if target_column else "target_not_declared")
    return tuple(codes)


def _imputation_explanation(recommended_id: str) -> str:
    if recommended_id == "group_median":
        return (
            "Group median preserves segment-dependent missingness better than a global "
            "median while staying explainable and target-safe."
        )
    return "Median imputation is explainable, robust for numeric skew, and policy-safe."


def _is_numeric(column: ColumnProfile) -> bool:
    return column.type in {ColumnType.NUMERIC_FLOAT, ColumnType.NUMERIC_INTEGER}


def _is_segment_dependent(missingness: ColumnMissingness) -> bool:
    if missingness.by_segment is None or not missingness.by_segment.groups:
        return False
    ratios = [group.missing_ratio for group in missingness.by_segment.groups.values()]
    return max(ratios) - min(ratios) >= _SEGMENT_DEPENDENT_MISSINGNESS_GAP


def _rare_class_needs_policy(
    class_imbalance: ClassImbalanceDiagnostics | None,
) -> bool:
    if class_imbalance is None:
        return False
    return (
        class_imbalance.minority_class_share <= _RARE_CLASS_SHARE_THRESHOLD
        or class_imbalance.imbalance_ratio >= _IMBALANCE_RATIO_THRESHOLD
    )


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, round(value, 6)))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"


__all__ = [
    "IMPUTATION_POLICY_COMPONENT_WEIGHTS",
    "METHOD_SCORE_FORMULA",
    "METHOD_SCORE_FORMULA_WEIGHTS",
    "METHOD_SCORE_NOTATION_FORMULA",
    "METHOD_SELECTION_POLICY_VERSION",
    "BuildMethodRecommendationsRequest",
    "build_method_recommendations",
]
