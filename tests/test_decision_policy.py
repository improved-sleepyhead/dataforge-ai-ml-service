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
    HardGateId,
    MethodAvailability,
    PolicyInputEnvelope,
    RecommendedActionType,
    disabled_outlier_capping_reason,
    evaluate_hard_gates,
    load_decision_policy_v0,
    load_reason_code_registry,
    recommend_from_evidence,
)


def test_policy_loading_exposes_required_gates_inputs_registry_and_hash() -> None:
    """Step 1: policy loading test."""
    policy = load_decision_policy_v0()
    registry = load_reason_code_registry()

    assert policy.policy_version == DECISION_POLICY_VERSION
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
