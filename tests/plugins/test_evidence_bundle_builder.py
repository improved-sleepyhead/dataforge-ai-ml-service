"""Tests for TASK-030: EvidenceBundle builder."""

from __future__ import annotations

import json

from app.domain import EvidenceBundle, SignalStatus
from app.plugins.object_analytics import (
    EVIDENCE_BUNDLE_ARTIFACT_KIND,
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    BuildEvidenceBundleRequest,
    build_evidence_bundles,
    build_object_analytics_passports,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.plugins.test_object_analytics_passports import (
    _COMPUTED_AT,
    _CONFIG_HASH,
    _DATASET_ID,
    _JOB_ID,
    _PARENT_VERSION_ID,
    _VERSION_ID,
    _manifest_rows,
    _model_error_report,
    _request,
    _storage_and_registry,
    _tabular_profile,
    _text_ocr_report,
)


def test_evidence_bundle_stage_persists_contract_valid_artifact() -> None:
    """Step 1+3: run evidence stage and validate persisted EvidenceBundle JSONL."""
    storage, registry = _storage_and_registry()
    passports_result = build_object_analytics_passports(
        manifest_rows=_manifest_rows(),
        request=_request(),
        registry=registry,
        tabular_profile=_tabular_profile(),
        text_ocr_report=_text_ocr_report(),
        model_error_report=_model_error_report(),
        computed_at=_COMPUTED_AT,
    )

    result = build_evidence_bundles(
        passports=passports_result.passports,
        request=_evidence_request(),
        registry=registry,
        source_passports_artifact=passports_result.artifact,
    )

    assert result.artifact.artifact_kind == EVIDENCE_BUNDLE_ARTIFACT_KIND
    assert result.artifact.schema_version == EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert len(result.evidence_bundles) == len(passports_result.passports)

    stored = storage.get(result.artifact.uri)
    assert stored.info.metadata["evidence_bundle_count"] == str(
        len(passports_result.passports)
    )
    parsed = [
        EvidenceBundle.model_validate(json.loads(line))
        for line in stored.data.decode("utf-8").splitlines()
    ]

    pack = load_contract_pack()
    for bundle in parsed:
        validate_contract_payload(pack, "evidence_bundle", bundle.model_dump(mode="json"))


def test_evidence_bundle_for_pii_duplicate_object_has_normalized_signals() -> None:
    """Step 2: PII+duplicate object carries normalized privacy and duplicate evidence."""
    _, registry = _storage_and_registry()
    passports_result = build_object_analytics_passports(
        manifest_rows=_manifest_rows(),
        request=_request(),
        registry=registry,
        tabular_profile=_tabular_profile(),
        text_ocr_report=_text_ocr_report(),
        model_error_report=_model_error_report(),
        computed_at=_COMPUTED_AT,
    )
    result = build_evidence_bundles(
        passports=passports_result.passports,
        request=_evidence_request(),
        registry=registry,
        source_passports_artifact=passports_result.artifact,
    )
    bundles = {bundle.object_id: bundle for bundle in result.evidence_bundles}
    text_bundle = bundles["obj_text_1"]

    assert text_bundle.object_type == "text_record"
    assert text_bundle.signals.privacy_risk.status is SignalStatus.AVAILABLE
    assert text_bundle.signals.privacy_risk.value == 0.5
    assert text_bundle.signals.duplicate_score.status is SignalStatus.AVAILABLE
    assert text_bundle.signals.duplicate_score.value == 1.0
    assert text_bundle.signals.model_uncertainty.status is SignalStatus.NOT_APPLICABLE
    assert (
        text_bundle.signals.model_uncertainty.reason
        == "prediction_manifest_not_provided"
    )
    assert text_bundle.signals.business_importance.status is SignalStatus.NOT_APPLICABLE
    assert text_bundle.confidence["privacy_risk"] == 0.9
    assert text_bundle.confidence["duplicate_score"] == 0.9
    evidence_ref_kinds = {ref.kind for ref in text_bundle.evidence_refs}
    assert EVIDENCE_BUNDLE_ARTIFACT_KIND not in evidence_ref_kinds
    assert any(
        ref.kind == "object_analytics_passports"
        for ref in text_bundle.evidence_refs
    )


def test_prediction_signals_are_available_for_model_error_passport() -> None:
    _, registry = _storage_and_registry()
    passports_result = build_object_analytics_passports(
        manifest_rows=_manifest_rows(),
        request=_request(),
        registry=registry,
        tabular_profile=_tabular_profile(),
        text_ocr_report=_text_ocr_report(),
        model_error_report=_model_error_report(),
        computed_at=_COMPUTED_AT,
    )
    result = build_evidence_bundles(
        passports=passports_result.passports,
        request=_evidence_request(),
        registry=registry,
        source_passports_artifact=passports_result.artifact,
    )
    tabular = {bundle.object_id: bundle for bundle in result.evidence_bundles}[
        "obj_tabular_1"
    ]

    assert tabular.signals.prediction_confidence.value == 0.52
    assert tabular.signals.prediction_margin.value == 0.04
    assert tabular.signals.prediction_entropy.value == 0.99
    assert tabular.signals.ambiguous_object_score.value == 0.76
    assert tabular.signals.probable_label_error_score.value == 0.0
    assert tabular.signals.rare_segment_score.value == 0.98
    assert tabular.confidence["ambiguous_object_score"] == 0.82


def _evidence_request() -> BuildEvidenceBundleRequest:
    return BuildEvidenceBundleRequest(
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
