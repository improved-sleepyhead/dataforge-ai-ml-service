"""Tests for TASK-035: DataForge Score v0 and decomposition."""

from __future__ import annotations

from typing import Any

from app.domain import (
    ClassCount,
    ClassImbalanceDiagnostics,
    CriticalBlocker,
    DatasetReadiness,
    DecisionReport,
    DuplicateDiagnostics,
    ReadinessAssessment,
    TabularProfileReport,
    TextOcrReport,
)
from app.kernel import (
    DATAFORGE_SCORE_FORMULA,
    DATAFORGE_SCORE_PENALTIES,
    DATAFORGE_SCORE_POLICY_VERSION,
    DATAFORGE_SCORE_WEIGHTS,
    BuildDataForgeScoreRequest,
    build_dataforge_score,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload


def test_dataforge_score_demo_data_validates_contract() -> None:
    """Step 1: run scoring on demo data and validate DataForgeReport score shape."""
    score = build_dataforge_score(
        BuildDataForgeScoreRequest(
            tabular_profile=_tabular_profile(),
            text_ocr_report=_text_ocr_report(),
            decision_report=_decision_report(),
        )
    )

    assert score.policy_version == DATAFORGE_SCORE_POLICY_VERSION
    assert score.formula == DATAFORGE_SCORE_FORMULA
    assert score.weights == DATAFORGE_SCORE_WEIGHTS
    assert set(score.components) == set(DATAFORGE_SCORE_WEIGHTS)
    assert score.hard_blocked is True
    assert score.readiness_status == "BLOCKED"

    report_payload = _example_payload("dataforge_report.analyze_only")
    report_payload["score"] = score.model_dump(mode="json")
    report_payload["modality_scores"] = {"tabular": score.value}
    validate_contract_payload(load_contract_pack(), "dataforge_report", report_payload)


def test_penalties_for_unmasked_pii_duplicates_and_class_imbalance() -> None:
    """Step 2: penalties match DATASETS.md defaults."""
    text = _text_ocr_report().model_copy(update={"total_redacted_record_count": 0})
    score = build_dataforge_score(
        BuildDataForgeScoreRequest(
            tabular_profile=_tabular_profile(),
            text_ocr_report=text,
            decision_report=_decision_report(),
        )
    )
    penalties = {penalty.reason_code: penalty.value for penalty in score.penalties}

    assert penalties["unmasked_pii"] == DATAFORGE_SCORE_PENALTIES["unmasked_pii"]
    assert penalties["severe_duplicates"] == DATAFORGE_SCORE_PENALTIES["severe_duplicates"]
    assert (
        penalties["severe_class_imbalance"]
        == DATAFORGE_SCORE_PENALTIES["severe_class_imbalance"]
    )
    assert {"unmasked_pii", "severe_duplicates", "severe_class_imbalance"} <= set(
        score.reason_codes
    )


def test_score_formula_weights_components_and_target_leakage_penalty() -> None:
    """Steps 3+4: formula, weights, components and leakage penalty are decomposed."""
    score = build_dataforge_score(
        BuildDataForgeScoreRequest(
            tabular_profile=_tabular_profile(),
            text_ocr_report=_text_ocr_report(),
            decision_report=_decision_report(),
            split_leakage_detected=True,
        )
    )

    assert score.policy_version == "dataforge_score_v0"
    assert score.weights["completeness"] == 0.15
    assert score.weights["privacy_safety"] == 0.18
    assert score.weights["lineage_completeness"] == 0.10
    assert score.weighted_components["completeness"] == (
        score.weights["completeness"] * score.components["completeness"]
    )
    assert {penalty.reason_code for penalty in score.penalties} >= {
        "split_leakage",
        "severe_class_imbalance",
    }
    assert score.components["model_readiness"] == 0.0


def test_high_numeric_score_cannot_override_hard_blockers() -> None:
    """Step 4: hard blockers preserve BLOCKED readiness independently of score."""
    profile = _mostly_clean_profile()
    score = build_dataforge_score(
        BuildDataForgeScoreRequest(
            tabular_profile=profile,
            decision_report=_non_penalized_hard_blocker_report(),
        )
    )

    assert score.raw_score > 60.0
    assert score.value > 0.60
    assert score.hard_blocked is True
    assert score.readiness_status == "BLOCKED"
    assert "business_rule_failure" in score.reason_codes


def _tabular_profile() -> TabularProfileReport:
    return TabularProfileReport.model_validate(
        _example_payload("tabular_profile_report.fraud")
    )


def _text_ocr_report() -> TextOcrReport:
    return TextOcrReport.model_validate(_example_payload("text_ocr_report.privacy"))


def _decision_report() -> DecisionReport:
    return DecisionReport.model_validate(_example_payload("decision_report.needs_review"))


def _mostly_clean_profile() -> TabularProfileReport:
    profile = _tabular_profile()
    return profile.model_copy(
        update={
            "column_count": 0,
            "columns": (),
            "missingness": None,
            "duplicates": DuplicateDiagnostics(
                duplicate_pair_count=0,
                duplicate_group_count=0,
                affected_object_ids=(),
                signature_columns=("amount",),
                id_column="object_id",
            ),
            "class_imbalance": ClassImbalanceDiagnostics(
                target_column="is_fraud",
                total_samples=200,
                class_counts=(
                    ClassCount(label="0", count=100),
                    ClassCount(label="1", count=100),
                ),
                rare_class_label="0",
                rare_class_count=100,
                rare_class_ratio=0.5,
                minority_class_label="0",
                minority_class_share=0.5,
                imbalance_ratio=1.0,
                balance_score=1.0,
                balance_score_alternative=1.0,
                effective_number_beta=0.999,
                effective_number_of_samples={"0": 95.2, "1": 95.2},
            ),
        }
    )


def _non_penalized_hard_blocker_report() -> DecisionReport:
    report = _decision_report()
    return report.model_copy(
        update={
            "critical_blockers": (
                CriticalBlocker(
                    code="BUSINESS_RULE_FAILURE",
                    severity="critical",
                    message="A critical business rule failed.",
                    evidence_refs=(),
                ),
            ),
            "readiness": ReadinessAssessment(
                status=DatasetReadiness.BLOCKED,
                score=0.0,
                reason_codes=("business_rule_failure",),
            ),
            "recommended_actions": (),
        }
    )


def _example_payload(name: str) -> dict[str, Any]:
    pack = load_contract_pack()
    for example in pack.examples:
        if example.name == name:
            return example.payload
    raise AssertionError(f"missing example {name}")
