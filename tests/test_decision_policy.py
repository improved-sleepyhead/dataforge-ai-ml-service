"""Tests for TASK-031: DecisionPolicy v0 and reason-code registry."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain import (
    DataModality,
    EvidenceBundle,
    EvidenceSignals,
    MethodCandidateStatus,
    NormalizedSignal,
    PolicyStatus,
    SignalStatus,
)
from app.kernel import (
    DECISION_POLICY_VERSION,
    OBJECT_VALUE_POLICY_VERSION,
    HardGateId,
    MethodAvailability,
    PolicyInputEnvelope,
    RecommendedActionType,
    compute_object_value_score,
    disabled_outlier_capping_reason,
    evaluate_hard_gates,
    evaluate_object_decision,
    load_decision_policy_v0,
    load_reason_code_registry,
    recommend_from_evidence,
)


def test_policy_loading_exposes_required_gates_inputs_registry_and_hash() -> None:
    """Step 1: policy loading test."""
    policy = load_decision_policy_v0()
    registry = load_reason_code_registry()

    assert policy.policy_version == DECISION_POLICY_VERSION
    assert policy.object_value_policy.policy_version == OBJECT_VALUE_POLICY_VERSION
    assert policy.config_hash.startswith("sha256:")
    assert len(policy.config_hash) == len("sha256:" + "a" * 64)
    assert {gate.gate_id for gate in policy.hard_gates} == set(HardGateId)
    assert {
        "normalized_dataset_profile",
        "plugin_readiness",
        "method_availability",
        "user_role_project_policy",
        "risk_profile",
        "split_leakage_report",
        "synthetic_validation_report",
    }.issubset(set(policy.policy_input_names))
    for code in (
        "pii_unmasked",
        "target_column_missing",
        "split_leakage",
        "external_api_raw_pii",
        "target_leakage_candidate",
        "business_rule_failure",
        "plugin_disabled",
        "method_disabled_by_policy",
        "method_disabled_by_readiness",
        "missing_numeric",
        "exact_duplicate",
        "rare_class_underrepresented",
        "text_pii_detected",
        "severe_outlier",
        "ambiguous_object",
        "probable_label_error",
    ):
        assert registry.require(code).code == code


def test_policy_input_envelope_rejects_raw_plugin_outputs() -> None:
    with pytest.raises(ValidationError, match="raw plugin outputs"):
        PolicyInputEnvelope(
            normalized_dataset_profile={"raw_plugin_outputs": {"rows": []}},
        )


def test_target_column_missing_hard_gate() -> None:
    """Step 2: target_column_missing hard gate."""
    policy = load_decision_policy_v0()
    inputs = PolicyInputEnvelope(
        normalized_dataset_profile={"target_column_missing": True},
    )
    gates = {result.gate_id: result for result in evaluate_hard_gates(policy=policy, inputs=inputs)}

    gate = gates[HardGateId.TARGET_COLUMN_MISSING]
    assert gate.triggered is True
    assert gate.reason_code == "target_column_missing"
    assert gate.action == "BLOCK_TRAINING"


def test_rare_class_underrepresented_reason_code() -> None:
    """Step 3: rare class reason code."""
    policy = load_decision_policy_v0()
    recommendations = recommend_from_evidence(
        policy=policy,
        evidence=_evidence_bundle(
            object_id="obj_rare",
            modality=DataModality.TABULAR,
            rare_segment_score=0.98,
        ),
    )

    assert any(
        recommendation.action is RecommendedActionType.RECOMMEND_RARE_CLASS_AUGMENTATION
        and "rare_class_underrepresented" in recommendation.reason_codes
        for recommendation in recommendations
    )


def test_leakage_plugin_readiness_method_gates_and_outlier_policy() -> None:
    """Step 4: leakage/plugin/method gates plus capping disabled_by_policy."""
    policy = load_decision_policy_v0()
    inputs = PolicyInputEnvelope(
        normalized_dataset_profile={"target_leakage_candidate": True},
        plugin_readiness={"dataforge.text_ocr": "disabled"},
        method_availability=(
            MethodAvailability(
                method_id="outlier_capping",
                status=MethodCandidateStatus.DISABLED_BY_POLICY,
                policy_status=PolicyStatus.DISABLED_BY_POLICY,
                reason_code="method_disabled_by_policy",
            ),
            MethodAvailability(
                method_id="smote",
                status=MethodCandidateStatus.DISABLED_BY_READINESS,
                policy_status=PolicyStatus.DISABLED_BY_READINESS,
                reason_code="method_disabled_by_readiness",
            ),
        ),
    )
    gates = {result.gate_id: result for result in evaluate_hard_gates(policy=policy, inputs=inputs)}

    assert gates[HardGateId.TARGET_LEAKAGE_CANDIDATE].triggered is True
    assert gates[HardGateId.TARGET_LEAKAGE_CANDIDATE].reason_code == "target_leakage_candidate"
    assert gates[HardGateId.PLUGIN_DISABLED].triggered is True
    assert gates[HardGateId.PLUGIN_DISABLED].reason_code == "plugin_disabled"
    assert gates[HardGateId.METHOD_DISABLED_BY_POLICY].triggered is True
    assert gates[HardGateId.METHOD_DISABLED_BY_POLICY].reason_code == "method_disabled_by_policy"
    assert gates[HardGateId.METHOD_DISABLED_BY_READINESS].triggered is True
    assert (
        gates[HardGateId.METHOD_DISABLED_BY_READINESS].reason_code
        == "method_disabled_by_readiness"
    )
    assert disabled_outlier_capping_reason(policy) == "outlier_capping_disabled_by_policy"


def test_ambiguous_and_probable_label_error_reason_codes() -> None:
    """Step 5: distinct reason codes for ambiguous and probable label error."""
    policy = load_decision_policy_v0()
    ambiguous = recommend_from_evidence(
        policy=policy,
        evidence=_evidence_bundle(
            object_id="obj_ambiguous",
            ambiguous_object_score=0.8,
            model_uncertainty=0.48,
            prediction_margin=0.04,
        ),
    )
    probable = recommend_from_evidence(
        policy=policy,
        evidence=_evidence_bundle(
            object_id="obj_probable_error",
            probable_label_error_score=0.9,
            prediction_confidence=0.95,
            prediction_margin=0.9,
        ),
    )

    ambiguous_codes = {code for item in ambiguous for code in item.reason_codes}
    probable_codes = {code for item in probable for code in item.reason_codes}

    assert {"ambiguous_object", "high_model_uncertainty", "low_prediction_margin"}.issubset(
        ambiguous_codes
    )
    assert "probable_label_error" not in ambiguous_codes
    assert {"probable_label_error", "high_confidence_label_conflict"}.issubset(
        probable_codes
    )
    assert "ambiguous_object" not in probable_codes


def test_high_privacy_high_learning_is_blocked_before_soft_score() -> None:
    """TASK-032 step 1+2+5: high score cannot override BLOCK_PRIVACY_REVIEW."""
    policy = load_decision_policy_v0()
    evidence = _evidence_bundle(
        object_id="obj_sensitive_high_value",
        privacy_risk=0.92,
        rare_segment_score=1.0,
        model_uncertainty=1.0,
        ambiguous_object_score=1.0,
        technical_quality=1.0,
    )

    evaluation = evaluate_object_decision(policy=policy, evidence=evidence)

    assert evaluation.hard_gates_evaluated_before_score is True
    assert evaluation.hard_blocked is True
    assert evaluation.action == "BLOCK_PRIVACY_REVIEW"
    assert "pii_unmasked" in evaluation.reason_codes
    assert evaluation.object_value_score.hard_blocked is True
    assert "BLOCK_PRIVACY_REVIEW" in evaluation.object_value_score.blocker_actions
    # The score is still computed for explanation, but it cannot override the gate.
    assert evaluation.object_value_score.value > 0.0


def test_object_value_score_decomposition_formula_and_weights() -> None:
    """TASK-032 step 3+5: decomposition uses DATASETS.md weights and formula."""
    policy = load_decision_policy_v0()
    evidence = _evidence_bundle(
        object_id="obj_score",
        technical_quality=0.8,
        duplicate_score=0.4,
        privacy_risk=0.3,
        label_issue_score=0.2,
        rare_segment_score=0.7,
        model_uncertainty=0.6,
        ambiguous_object_score=0.5,
        probable_label_error_score=0.2,
    )

    score = compute_object_value_score(policy=policy, evidence=evidence)

    assert score.policy_version == OBJECT_VALUE_POLICY_VERSION
    assert score.weights == {
        "learning_value": 0.25,
        "rarity": 0.20,
        "uncertainty": 0.20,
        "diversity": 0.15,
        "business_importance": 0.10,
        "duplicate_penalty": -0.15,
        "quality_penalty": -0.10,
        "privacy_risk_penalty": -0.25,
        "label_risk_penalty": -0.10,
    }
    assert score.components["learning_value"] == 0.7
    assert score.components["rarity"] == 0.7
    assert score.components["uncertainty"] == 0.6
    assert score.components["duplicate_penalty"] == 0.4
    assert score.components["quality_penalty"] == pytest.approx(0.2)
    assert score.components["privacy_risk_penalty"] == 0.3
    assert score.components["label_risk_penalty"] == 0.2
    expected_raw = (
        0.25 * 0.7
        + 0.20 * 0.7
        + 0.20 * 0.6
        + 0.15 * 0.0
        + 0.10 * 0.0
        - 0.15 * 0.4
        - 0.10 * 0.2
        - 0.25 * 0.3
        - 0.10 * 0.2
    )
    assert score.raw_value == pytest.approx(expected_raw)
    assert score.value == pytest.approx(expected_raw)
    assert score.weighted_components["uncertainty"] == pytest.approx(0.12)


def test_uncertainty_contribution_records_probability_notation_and_formulas() -> None:
    """TASK-032 step 4: uncertainty contribution and probability notation."""
    policy = load_decision_policy_v0()
    evidence = _evidence_bundle(
        object_id="obj_uncertain",
        model_uncertainty=None,
        prediction_confidence=0.55,
        prediction_entropy=0.88,
        ambiguous_object_score=0.7,
    )

    score = compute_object_value_score(policy=policy, evidence=evidence)

    assert score.components["uncertainty"] == 0.88
    assert score.weighted_components["uncertainty"] == pytest.approx(0.176)
    assert score.uncertainty_probability_notation == "p_i,k = P(Y = k | x_i)"
    assert "U(x_i) = 1 - max_k p_i,k" in score.uncertainty_formulas
    assert "H(x_i) = - sum_k p_i,k * log(p_i,k)" in score.uncertainty_formulas
    assert "H_norm(x_i) = H(x_i) / log(K)" in score.uncertainty_formulas


def _evidence_bundle(
    *,
    object_id: str,
    modality: DataModality = DataModality.TABULAR,
    technical_quality: float = 1.0,
    duplicate_score: float = 0.0,
    privacy_risk: float = 0.0,
    label_issue_score: float | None = None,
    rare_segment_score: float = 0.0,
    model_uncertainty: float | None = None,
    prediction_confidence: float | None = None,
    prediction_margin: float | None = None,
    prediction_entropy: float | None = None,
    ambiguous_object_score: float | None = None,
    probable_label_error_score: float | None = None,
) -> EvidenceBundle:
    return EvidenceBundle(
        evidence_bundle_id=f"evidence_{object_id}",
        object_id=object_id,
        dataset_id="dataset_demo",
        version_id="version_demo",
        modality=modality,
        object_type="table_row" if modality is DataModality.TABULAR else "text_record",
        signals=EvidenceSignals(
            technical_quality=_available(technical_quality),
            duplicate_score=_available(duplicate_score),
            privacy_risk=_available(privacy_risk),
            label_issue_score=_prediction_signal(label_issue_score),
            rare_segment_score=_available(rare_segment_score),
            business_importance=_not_applicable("business_importance_not_available"),
            model_uncertainty=_prediction_signal(model_uncertainty),
            prediction_confidence=_prediction_signal(prediction_confidence),
            prediction_margin=_prediction_signal(prediction_margin),
            prediction_entropy=_prediction_signal(prediction_entropy),
            ambiguous_object_score=_prediction_signal(ambiguous_object_score),
            probable_label_error_score=_prediction_signal(probable_label_error_score),
        ),
        confidence={"technical_quality": 0.95},
        evidence_refs=(),
        computed_by_job_id="compute_run_policy",
        evidence_schema_version="evidence_bundle.v1",
    )


def _available(value: float) -> NormalizedSignal:
    return NormalizedSignal(value=value, status=SignalStatus.AVAILABLE, reason=None)


def _prediction_signal(value: float | None) -> NormalizedSignal:
    if value is None:
        return _not_applicable("prediction_manifest_not_provided")
    return _available(value)


def _not_applicable(reason: str) -> NormalizedSignal:
    return NormalizedSignal(value=None, status=SignalStatus.NOT_APPLICABLE, reason=reason)
