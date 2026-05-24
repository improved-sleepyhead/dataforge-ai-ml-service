"""Tests for TASK-033: DecisionReport dataset and object decisions."""

from __future__ import annotations

import json

from app.domain import (
    DataModality,
    DecisionAction,
    DecisionReport,
    EvidenceBundle,
    EvidenceRef,
    EvidenceSignals,
    NormalizedSignal,
    SignalStatus,
)
from app.kernel import BuildDecisionReportRequest, build_decision_report
from app.plugins.object_analytics import build_evidence_bundles, build_object_analytics_passports
from app.reports import (
    DECISION_REPORT_ARTIFACT_KIND,
    build_decision_report_artifact,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.plugins.test_evidence_bundle_builder import _evidence_request
from tests.plugins.test_object_analytics_passports import (
    _COMPUTED_AT,
    _CONFIG_HASH,
    _DATASET_ID,
    _JOB_ID,
    _VERSION_ID,
    _manifest_rows,
    _model_error_report,
    _request,
    _storage_and_registry,
    _tabular_profile,
    _text_ocr_report,
)


def test_decision_report_artifact_from_demo_evidence_validates_contract() -> None:
    """Step 1+2+3: run Decision Core on demo evidence and persist valid report."""
    storage, registry = _storage_and_registry()
    passports = build_object_analytics_passports(
        manifest_rows=_manifest_rows(),
        request=_request(),
        registry=registry,
        tabular_profile=_tabular_profile(),
        text_ocr_report=_text_ocr_report(),
        model_error_report=_model_error_report(),
        computed_at=_COMPUTED_AT,
    )
    evidence = build_evidence_bundles(
        passports=passports.passports,
        request=_evidence_request(),
        registry=registry,
        source_passports_artifact=passports.artifact,
    )

    result = build_decision_report_artifact(
        evidence_bundles=evidence.evidence_bundles,
        request=_report_request(),
        registry=registry,
    )

    assert result.artifact.artifact_kind == DECISION_REPORT_ARTIFACT_KIND
    assert result.report.dataset_decision == "NEEDS_REVIEW"
    assert result.report.readiness.status == "NOT_READY_FOR_EXPORT"
    assert result.report.safe_actions_available is True
    assert len(result.report.object_decisions) == len(evidence.evidence_bundles)
    assert any(
        decision.action is DecisionAction.SEND_TO_LABEL_REVIEW
        and "ambiguous_object" in decision.reasons
        for decision in result.report.object_decisions
    )

    stored = storage.get(result.artifact.uri)
    assert stored.info.metadata["object-decision-count"] == str(
        len(result.report.object_decisions)
    )
    parsed = DecisionReport.model_validate(json.loads(stored.data.decode("utf-8")))
    validate_contract_payload(
        load_contract_pack(),
        "decision_report",
        parsed.model_dump(mode="json"),
    )


def test_report_blocks_export_for_hard_gate_before_object_score() -> None:
    """Step 2: blockers and object blocked_actions are included."""
    report = build_decision_report(
        evidence_bundles=(
            _evidence_bundle(
                object_id="obj_sensitive",
                modality=DataModality.TEXT,
                privacy_risk=0.92,
                rare_segment_score=1.0,
                model_uncertainty=0.9,
            ),
        ),
        request=_report_request(),
    )

    assert report.dataset_decision == "BLOCKED"
    assert report.readiness.status == "BLOCKED"
    assert report.critical_blockers[0].code == "PII_UNMASKED"
    decision = report.object_decisions[0]
    assert decision.action is DecisionAction.BLOCK_EXPORT
    assert decision.object_value_score > 0.0
    assert decision.blocked_actions[0].action is DecisionAction.EXPORT_READY
    assert "pii_unmasked" in decision.blocked_actions[0].reason_codes
    assert any(
        action.action is DecisionAction.SEND_TO_PRIVACY_REVIEW
        for action in report.recommended_actions
    )


def test_decision_action_taxonomy_contains_allowed_report_actions() -> None:
    """Step 4: allowed decision action taxonomy."""
    assert {
        "KEEP",
        "REMOVE_DUPLICATE",
        "SEND_TO_LABEL_REVIEW",
        "SEND_TO_PRIVACY_REVIEW",
        "IMPUTE_MISSING_VALUES",
        "AUGMENT_RARE_CLASS",
        "GENERATE_SYNTHETIC_CANDIDATE",
        "BLOCK_EXPORT",
        "EXPORT_READY",
    }.issubset({action.value for action in DecisionAction})


def test_ambiguous_and_probable_label_error_recommendations_are_separate() -> None:
    """Step 5: ambiguous/probable label error recommendations stay separate."""
    report = build_decision_report(
        evidence_bundles=(
            _evidence_bundle(
                object_id="obj_ambiguous",
                ambiguous_object_score=0.8,
                model_uncertainty=0.7,
                prediction_margin=0.04,
            ),
            _evidence_bundle(
                object_id="obj_probable_label_error",
                probable_label_error_score=0.9,
                prediction_confidence=0.96,
                prediction_margin=0.92,
            ),
        ),
        request=_report_request(),
    )

    label_review_actions = [
        action
        for action in report.recommended_actions
        if action.action is DecisionAction.SEND_TO_LABEL_REVIEW
    ]
    assert len(label_review_actions) == 2
    segments = {action.segment for action in label_review_actions}
    assert "reason=ambiguous_object" in segments
    assert "reason=probable_label_error" in segments
    assert all(action.count == 1 for action in label_review_actions)


def _report_request() -> BuildDecisionReportRequest:
    return BuildDecisionReportRequest(
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        decision_report_id="decision_report_demo",
        generated_at=_COMPUTED_AT,
    )


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
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
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
        evidence_refs=(
            EvidenceRef(
                kind="EVIDENCE_BUNDLE",
                uri="s3://dataforge/org_test/project_test/dataset_demo/evidence.json",
            ),
        ),
        computed_by_job_id=_JOB_ID,
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
