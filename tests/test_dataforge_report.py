"""Tests for TASK-037: ANALYZE_ONLY DataForgeReport summary artifact."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from app.adapters import ArtifactRegistry
from app.domain import DataForgeReport, DataModality, EvidenceBundle, EvidenceRef, SignalStatus
from app.kernel import BuildDataForgeScoreRequest, build_dataforge_score, build_decision_report
from app.reports import (
    DATAFORGE_REPORT_ARTIFACT_KIND,
    BuildDataForgeReportRequest,
    BuildReviewQueuesRequest,
    build_dataforge_report_artifact,
    build_review_queues_artifact,
    detail_artifact_refs,
    review_queue_ref,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.test_artifact_registry import InMemoryS3Client, _storage
from tests.test_decision_report import _evidence_bundle, _report_request

_CONFIG_HASH = "sha256:" + "a" * 64
_COMPUTED_AT = datetime(2026, 5, 24, 13, 0, tzinfo=UTC)


def test_dataforge_report_artifact_validates_contract_with_prediction_refs() -> None:
    """Steps 1-3: build, open stored JSON, and validate the report contract."""
    storage = _storage(InMemoryS3Client())
    registry = ArtifactRegistry(storage=storage)
    evidence = _prediction_evidence()
    decision_report = build_decision_report(
        evidence_bundles=evidence,
        request=_report_request(),
    )
    score = build_dataforge_score(
        BuildDataForgeScoreRequest(decision_report=decision_report)
    )
    review_queues = build_review_queues_artifact(
        decision_report=decision_report,
        evidence_bundles=evidence,
        request=BuildReviewQueuesRequest(
            dataset_id="dataset_demo",
            version_id="version_demo",
            created_by_job_id="compute_run_dataforge_report",
            config_hash=_CONFIG_HASH,
            generated_at=_COMPUTED_AT,
        ),
        registry=registry,
    )
    decision_artifact = registry.save_artifact(
        artifact_kind="decision_report",
        data=(decision_report.model_dump_json() + "\n").encode("utf-8"),
        artifact_format="json",
        media_type="application/json",
        schema_version="decision_report.v1",
        dataset_version_id="version_demo",
        created_by_job_id="compute_run_dataforge_report",
        config_hash=_CONFIG_HASH,
    )

    result = build_dataforge_report_artifact(
        request=BuildDataForgeReportRequest(
            dataset_id="dataset_demo",
            version_id="version_demo",
            created_by_job_id="compute_run_dataforge_report",
            config_hash=_CONFIG_HASH,
            score=score,
            decision_report=decision_report,
            review_queue_refs=(review_queue_ref(review_queues.artifact),),
            detail_artifacts=detail_artifact_refs((decision_artifact, review_queues.artifact)),
            prediction_manifest_ref=_prediction_ref("PREDICTION_MANIFEST", "predictions.jsonl"),
            prediction_validation_report_ref=_prediction_ref(
                "PREDICTION_VALIDATION_REPORT",
                "prediction_validation_report.json",
            ),
            model_error_analysis_report_ref=_prediction_ref(
                "MODEL_ERROR_ANALYSIS_REPORT",
                "model_error_analysis_report.json",
            ),
            risk_profile="demo_strict",
            generated_at=_COMPUTED_AT,
        ),
        registry=registry,
    )

    assert result.artifact.artifact_kind == DATAFORGE_REPORT_ARTIFACT_KIND
    stored = storage.get(result.artifact.uri)
    parsed = DataForgeReport.model_validate(json.loads(stored.data.decode("utf-8")))
    assert parsed.workflow_type == "ANALYZE_ONLY"
    assert parsed.mutates_dataset is False
    assert parsed.review_queues[0].kind == "REVIEW_QUEUE"
    assert len(parsed.detail_artifacts) == 2
    assert parsed.prediction_section.status is SignalStatus.AVAILABLE
    assert parsed.prediction_section.ambiguous_object_count == 1
    assert parsed.prediction_section.probable_label_error_count == 1
    assert parsed.overview["ambiguous_object_candidates"] == 1
    assert parsed.overview["probable_label_error_candidates"] == 1
    assert DataModality.TABULAR in parsed.modality_scores

    validate_contract_payload(
        load_contract_pack(),
        "dataforge_report",
        parsed.model_dump(mode="json"),
    )


def test_dataforge_report_prediction_section_not_applicable_when_absent() -> None:
    """Step 4: absent predictions produce explicit not_applicable section."""
    storage = _storage(InMemoryS3Client())
    registry = ArtifactRegistry(storage=storage)
    evidence = (_evidence_bundle(object_id="obj_no_prediction"),)
    decision_report = build_decision_report(
        evidence_bundles=evidence,
        request=_report_request(),
    )
    score = build_dataforge_score(
        BuildDataForgeScoreRequest(decision_report=decision_report)
    )

    result = build_dataforge_report_artifact(
        request=BuildDataForgeReportRequest(
            dataset_id="dataset_demo",
            version_id="version_demo",
            created_by_job_id="compute_run_dataforge_report",
            config_hash=_CONFIG_HASH,
            score=score,
            decision_report=decision_report,
            generated_at=_COMPUTED_AT,
        ),
        registry=registry,
    )

    assert result.report.prediction_section.status is SignalStatus.NOT_APPLICABLE
    assert result.report.prediction_section.reason == "prediction_manifest_not_provided"
    assert result.report.prediction_section.prediction_manifest_ref is None
    assert result.report.prediction_section.prediction_validation_report_ref is None
    assert result.report.prediction_section.model_error_analysis_report_ref is None
    assert result.report.prediction_section.ambiguous_object_count == 0
    assert result.report.prediction_section.probable_label_error_count == 0
    assert result.report.workflow_type == "ANALYZE_ONLY"
    assert result.report.mutates_dataset is False


def _prediction_evidence() -> tuple[EvidenceBundle, ...]:
    return (
        _evidence_bundle(
            object_id="obj_ambiguous",
            ambiguous_object_score=0.8,
            model_uncertainty=0.7,
            prediction_margin=0.04,
        ),
        _evidence_bundle(
            object_id="obj_probable",
            probable_label_error_score=0.9,
            prediction_confidence=0.96,
            prediction_margin=0.92,
        ),
    )


def _prediction_ref(kind: str, name: str) -> EvidenceRef:
    return EvidenceRef(
        kind=kind,
        uri=f"s3://dataforge/org_1/project_1/dataset_demo/version_demo/{name}",
    )
