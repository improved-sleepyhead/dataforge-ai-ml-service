"""Tests for TASK-036: ReviewQueue JSONL artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from app.adapters import ArtifactRegistry
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    DataModality,
    DecisionAction,
    DecisionReport,
    EvidenceBundle,
    ReviewQueue,
    ReviewQueueType,
)
from app.kernel import build_decision_report
from app.reports import (
    REVIEW_QUEUE_ARTIFACT_KIND,
    SAFE_PREVIEW_ARTIFACT_KIND,
    BuildReviewQueuesRequest,
    build_review_queues_artifact,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.test_artifact_registry import InMemoryS3Client, _storage
from tests.test_decision_report import _evidence_bundle, _report_request

_CONFIG_HASH = "sha256:" + "a" * 64
_COMPUTED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)


def test_review_queue_stage_persists_contract_valid_jsonl_artifact() -> None:
    """Step 1: run review queue stage and validate each queued contract row."""
    storage, registry = _registry()
    evidence = _review_evidence()
    report = _decision_report(evidence)

    result = build_review_queues_artifact(
        decision_report=report,
        evidence_bundles=evidence,
        request=_request(),
        registry=registry,
    )

    assert result.artifact.artifact_kind == REVIEW_QUEUE_ARTIFACT_KIND
    assert result.artifact.format == "jsonl"
    assert len(result.queues) == 3
    assert {queue.queue_type for queue in result.queues} == {
        ReviewQueueType.LABEL_REVIEW,
        ReviewQueueType.PRIVACY_REVIEW,
        ReviewQueueType.DUPLICATE_REVIEW,
    }

    stored = storage.get(result.artifact.uri)
    parsed = [
        ReviewQueue.model_validate(json.loads(line))
        for line in stored.data.decode("utf-8").splitlines()
    ]
    assert len(parsed) == len(result.queues)
    pack = load_contract_pack()
    for queue in parsed:
        validate_contract_payload(pack, "review_queue", queue.model_dump(mode="json"))


def test_privacy_review_uses_redacted_safe_preview_without_raw_pii() -> None:
    """Steps 2+3: privacy entries point to safe previews and do not expose PII."""
    storage, registry = _registry()
    result = build_review_queues_artifact(
        decision_report=_decision_report(_review_evidence()),
        evidence_bundles=_review_evidence(),
        request=_request(),
        registry=registry,
    )

    privacy_queue = _queue(result.queues, ReviewQueueType.PRIVACY_REVIEW)
    assert privacy_queue.export_policy.raw_pii_allowed is False
    assert privacy_queue.export_policy.redacted_only is True
    assert privacy_queue.export_policy.allow_external_tool is False
    assert privacy_queue.objects[0].reason_codes == (
        "text_pii_detected",
        "pii_unmasked",
        "privacy_review_required",
    )
    assert privacy_queue.objects[0].safe_preview_uri.startswith("s3://")

    preview = storage.get(privacy_queue.objects[0].safe_preview_uri)
    preview_text = preview.data.decode("utf-8")
    assert "email@example.com" not in preview_text
    assert "support@example.com" not in preview_text
    assert '"raw_content_included":false' in preview_text
    assert '"pii_values_included":false' in preview_text
    assert all(
        artifact.artifact_kind == SAFE_PREVIEW_ARTIFACT_KIND
        for artifact in result.safe_preview_artifacts
    )


def test_label_review_separates_ambiguous_and_probable_label_error_entries() -> None:
    """Step 4: ambiguous and probable label-error entries use separate reason codes."""
    result = build_review_queues_artifact(
        decision_report=_decision_report(_review_evidence()),
        evidence_bundles=_review_evidence(),
        request=_request(),
        registry=_registry()[1],
    )
    label_queue = _queue(result.queues, ReviewQueueType.LABEL_REVIEW)
    entries = {item.object_id: item for item in label_queue.objects}

    assert entries["obj_ambiguous"].reason_codes == (
        "high_model_uncertainty",
        "low_prediction_margin",
        "ambiguous_object",
    )
    assert entries["obj_probable"].reason_codes == (
        "probable_label_error",
        "high_confidence_label_conflict",
        "neighbor_label_disagreement",
    )
    assert entries["obj_ambiguous"].priority > 0.7
    assert entries["obj_probable"].priority > 0.8


def test_duplicate_review_entries_include_duplicate_reason_and_evidence_refs() -> None:
    result = build_review_queues_artifact(
        decision_report=_decision_report(_review_evidence()),
        evidence_bundles=_review_evidence(),
        request=_request(),
        registry=_registry()[1],
    )
    duplicate_queue = _queue(result.queues, ReviewQueueType.DUPLICATE_REVIEW)

    assert duplicate_queue.objects[0].object_id == "obj_duplicate"
    assert duplicate_queue.objects[0].reason_codes == (
        "exact_duplicate",
        "duplicate_review_required",
    )
    assert duplicate_queue.objects[0].evidence_refs


def _review_evidence() -> tuple[EvidenceBundle, ...]:
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
        _evidence_bundle(
            object_id="obj_privacy",
            modality=DataModality.TEXT,
            privacy_risk=0.92,
        ),
        _evidence_bundle(
            object_id="obj_duplicate",
            duplicate_score=1.0,
        ),
    )


def _decision_report(evidence: tuple[EvidenceBundle, ...]) -> DecisionReport:
    report = build_decision_report(evidence_bundles=evidence, request=_report_request())
    decisions = []
    for decision in report.object_decisions:
        if decision.object_id == "obj_probable":
            decision = decision.model_copy(
                update={
                    "reasons": tuple(
                        dict.fromkeys(
                            [
                                *decision.reasons,
                                "high_confidence_label_conflict",
                                "neighbor_label_disagreement",
                            ]
                        )
                    )
                }
            )
        if decision.object_id == "obj_duplicate":
            decision = decision.model_copy(
                update={
                    "action": DecisionAction.REMOVE_DUPLICATE,
                    "reasons": ("exact_duplicate",),
                }
            )
        decisions.append(decision)
    return report.model_copy(update={"object_decisions": tuple(decisions)})


def _request() -> BuildReviewQueuesRequest:
    return BuildReviewQueuesRequest(
        dataset_id="dataset_demo",
        version_id="version_demo",
        created_by_job_id="compute_run_review_queue",
        config_hash=_CONFIG_HASH,
        generated_at=_COMPUTED_AT,
    )


def _registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = _storage(InMemoryS3Client())
    return storage, ArtifactRegistry(storage=storage)


def _queue(queues: tuple[ReviewQueue, ...], queue_type: ReviewQueueType) -> ReviewQueue:
    for queue in queues:
        if queue.queue_type is queue_type:
            return queue
    raise AssertionError(f"missing queue {queue_type}")
