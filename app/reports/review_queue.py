"""Build and persist ReviewQueue artifacts from DecisionReport outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    DataModality,
    DecisionAction,
    DecisionReport,
    EvidenceBundle,
    ReviewExportPolicy,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
)
from app.domain.common import NonEmptyStr, Sha256Digest

REVIEW_QUEUE_ARTIFACT_KIND = "review_queue"
REVIEW_QUEUE_ARTIFACT_FORMAT = "jsonl"
REVIEW_QUEUE_MEDIA_TYPE = "application/jsonl"
REVIEW_QUEUE_SCHEMA_VERSION = "review_queue.v1"

SAFE_PREVIEW_ARTIFACT_KIND = "review_queue_safe_preview"
SAFE_PREVIEW_ARTIFACT_FORMAT = "json"
SAFE_PREVIEW_MEDIA_TYPE = "application/json"
SAFE_PREVIEW_SCHEMA_VERSION = "review_safe_preview.v1"


class BuildReviewQueuesRequest(BaseModel):
    """Inputs required to build review queues from Decision Core output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    generated_at: datetime | None = None
    label_target_tool: str | None = "label_studio"


@dataclass(frozen=True)
class BuildReviewQueuesArtifactResult:
    """ReviewQueue stage output plus the immutable JSONL artifact."""

    queues: tuple[ReviewQueue, ...]
    artifact: RegisteredArtifact
    safe_preview_artifacts: tuple[RegisteredArtifact, ...]


def build_review_queues_artifact(
    *,
    decision_report: DecisionReport,
    evidence_bundles: Iterable[EvidenceBundle],
    request: BuildReviewQueuesRequest,
    registry: ArtifactRegistry,
) -> BuildReviewQueuesArtifactResult:
    """Create review queues and persist safe previews plus queue JSONL.

    The stage consumes normalized DecisionReport/EvidenceBundle contracts only.
    Safe preview artifacts intentionally contain object ids, reason codes and
    coarse signal summaries, never raw row values, raw text, or PII snippets.
    """
    evidence_by_object = {bundle.object_id: bundle for bundle in evidence_bundles}
    safe_preview_artifacts: list[RegisteredArtifact] = []
    queue_items: dict[ReviewQueueType, list[ReviewQueueItem]] = {
        ReviewQueueType.LABEL_REVIEW: [],
        ReviewQueueType.PRIVACY_REVIEW: [],
        ReviewQueueType.DUPLICATE_REVIEW: [],
        ReviewQueueType.ANNOTATION_REVIEW: [],
    }

    for decision in decision_report.object_decisions:
        bundle = evidence_by_object.get(decision.object_id)
        object_type = bundle.object_type if bundle is not None else _object_type(decision.modality)
        for queue_type, reason_codes in _queue_memberships(decision.action, decision.reasons):
            preview = _save_safe_preview(
                queue_type=queue_type,
                object_id=decision.object_id,
                modality=decision.modality,
                object_type=object_type,
                reason_codes=reason_codes,
                evidence=bundle,
                request=request,
                registry=registry,
            )
            safe_preview_artifacts.append(preview)
            queue_items[queue_type].append(
                ReviewQueueItem(
                    object_id=decision.object_id,
                    modality=decision.modality,
                    object_type=bundle.object_type
                    if bundle is not None
                    else _object_type(decision.modality),
                    reason_codes=reason_codes,
                    priority=_priority(
                        queue_type=queue_type,
                        reason_codes=reason_codes,
                        evidence=bundle,
                    ),
                    safe_preview_uri=preview.uri,
                    evidence_refs=bundle.evidence_refs if bundle is not None else (),
                )
            )

    queues = tuple(
        _queue(
            queue_type=queue_type,
            objects=tuple(items),
            request=request,
        )
        for queue_type, items in queue_items.items()
        if items
    )
    artifact = registry.save_artifact(
        artifact_kind=REVIEW_QUEUE_ARTIFACT_KIND,
        data=_serialize_queues(queues),
        artifact_format=REVIEW_QUEUE_ARTIFACT_FORMAT,
        media_type=REVIEW_QUEUE_MEDIA_TYPE,
        schema_version=REVIEW_QUEUE_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "review-queue-count": str(len(queues)),
            "review-object-count": str(sum(len(queue.objects) for queue in queues)),
        },
    )
    return BuildReviewQueuesArtifactResult(
        queues=queues,
        artifact=artifact,
        safe_preview_artifacts=tuple(safe_preview_artifacts),
    )


def _queue_memberships(
    action: DecisionAction,
    reasons: tuple[str, ...],
) -> tuple[tuple[ReviewQueueType, tuple[str, ...]], ...]:
    memberships: list[tuple[ReviewQueueType, tuple[str, ...]]] = []
    reason_set = set(reasons)
    if action is DecisionAction.SEND_TO_LABEL_REVIEW or reason_set & {
        "ambiguous_object",
        "probable_label_error",
    }:
        if "ambiguous_object" in reason_set:
            memberships.append(
                (
                    ReviewQueueType.LABEL_REVIEW,
                    ("high_model_uncertainty", "low_prediction_margin", "ambiguous_object"),
                )
            )
        if "probable_label_error" in reason_set:
            probable_reasons = [
                "probable_label_error",
                "high_confidence_label_conflict",
            ]
            if "neighbor_label_disagreement" in reason_set:
                probable_reasons.append("neighbor_label_disagreement")
            memberships.append((ReviewQueueType.LABEL_REVIEW, tuple(probable_reasons)))
    if action is DecisionAction.SEND_TO_PRIVACY_REVIEW or reason_set & {
        "text_pii_detected",
        "pii_unmasked",
    }:
        memberships.append(
            (
                ReviewQueueType.PRIVACY_REVIEW,
                tuple(
                    dict.fromkeys(
                        [
                            *(
                                code
                                for code in reasons
                                if code in {"text_pii_detected", "pii_unmasked"}
                            ),
                            "privacy_review_required",
                        ]
                    )
                ),
            )
        )
    if action is DecisionAction.REMOVE_DUPLICATE or reason_set & {
        "exact_duplicate",
        "duplicate_candidate",
    }:
        memberships.append(
            (
                ReviewQueueType.DUPLICATE_REVIEW,
                tuple(
                    dict.fromkeys(
                        [
                            *(
                                code
                                for code in reasons
                                if code in {"exact_duplicate", "duplicate_candidate"}
                            ),
                            "duplicate_review_required",
                        ]
                    )
                ),
            )
        )
    if reason_set & {"missing_numeric", "target_column_missing", "business_rule_failure"}:
        memberships.append(
            (
                ReviewQueueType.ANNOTATION_REVIEW,
                tuple(
                    dict.fromkeys(
                        [
                            *(
                                code
                                for code in reasons
                                if code
                                in {
                                    "missing_numeric",
                                    "target_column_missing",
                                    "business_rule_failure",
                                }
                            ),
                            "schema_review_required",
                        ]
                    )
                ),
            )
        )
    return tuple(memberships)


def _save_safe_preview(
    *,
    queue_type: ReviewQueueType,
    object_id: str,
    modality: DataModality,
    object_type: str,
    reason_codes: tuple[str, ...],
    evidence: EvidenceBundle | None,
    request: BuildReviewQueuesRequest,
    registry: ArtifactRegistry,
) -> RegisteredArtifact:
    payload = {
        "safe_preview_schema_version": SAFE_PREVIEW_SCHEMA_VERSION,
        "object_id": object_id,
        "dataset_id": request.dataset_id,
        "version_id": request.version_id,
        "queue_type": queue_type.value,
        "modality": modality.value,
        "object_type": object_type,
        "reason_codes": list(reason_codes),
        "signals": _safe_signal_summary(evidence),
        "raw_content_included": False,
        "pii_values_included": False,
    }
    return registry.save_artifact(
        artifact_kind=SAFE_PREVIEW_ARTIFACT_KIND,
        data=(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
        artifact_format=SAFE_PREVIEW_ARTIFACT_FORMAT,
        media_type=SAFE_PREVIEW_MEDIA_TYPE,
        schema_version=SAFE_PREVIEW_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "review-queue-type": queue_type.value,
            "review-object-id": object_id,
        },
    )


def _safe_signal_summary(evidence: EvidenceBundle | None) -> dict[str, float | None]:
    if evidence is None:
        return {}
    signals = evidence.signals
    return {
        "privacy_risk": signals.privacy_risk.value,
        "duplicate_score": signals.duplicate_score.value,
        "model_uncertainty": signals.model_uncertainty.value,
        "prediction_margin": signals.prediction_margin.value,
        "prediction_confidence": signals.prediction_confidence.value,
        "ambiguous_object_score": signals.ambiguous_object_score.value,
        "probable_label_error_score": signals.probable_label_error_score.value,
    }


def _queue(
    *,
    queue_type: ReviewQueueType,
    objects: tuple[ReviewQueueItem, ...],
    request: BuildReviewQueuesRequest,
) -> ReviewQueue:
    return ReviewQueue(
        review_queue_id=f"review_queue_{queue_type.value.lower()}",
        queue_schema_version=REVIEW_QUEUE_SCHEMA_VERSION,
        queue_type=queue_type,
        target_tool=_target_tool(queue_type, request),
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        objects=tuple(sorted(objects, key=lambda item: (-item.priority, item.object_id))),
        export_policy=_export_policy(queue_type),
        created_at=request.generated_at or datetime.now(UTC),
    )


def _target_tool(queue_type: ReviewQueueType, request: BuildReviewQueuesRequest) -> str | None:
    if queue_type is ReviewQueueType.LABEL_REVIEW:
        return request.label_target_tool
    return None


def _export_policy(queue_type: ReviewQueueType) -> ReviewExportPolicy:
    return ReviewExportPolicy(
        raw_pii_allowed=False,
        redacted_only=True,
        allow_external_tool=queue_type is ReviewQueueType.LABEL_REVIEW,
    )


def _priority(
    *,
    queue_type: ReviewQueueType,
    reason_codes: tuple[str, ...],
    evidence: EvidenceBundle | None,
) -> float:
    if evidence is None:
        return 0.5
    signals = evidence.signals
    values: list[float] = []
    if queue_type is ReviewQueueType.PRIVACY_REVIEW:
        values.append(signals.privacy_risk.value or 0.0)
    if queue_type is ReviewQueueType.DUPLICATE_REVIEW:
        values.append(signals.duplicate_score.value or 0.0)
    if queue_type is ReviewQueueType.LABEL_REVIEW:
        if "ambiguous_object" in reason_codes:
            values.extend(
                (
                    signals.ambiguous_object_score.value or 0.0,
                    signals.model_uncertainty.value or 0.0,
                    1.0 - (signals.prediction_margin.value or 1.0),
                )
            )
        if "probable_label_error" in reason_codes:
            values.extend(
                (
                    signals.probable_label_error_score.value or 0.0,
                    signals.prediction_confidence.value or 0.0,
                )
            )
    if queue_type is ReviewQueueType.ANNOTATION_REVIEW:
        values.append(1.0 - (signals.technical_quality.value or 1.0))
    return min(1.0, max(0.0, round(max(values, default=0.5), 6)))


def _object_type(modality: DataModality) -> str:
    if modality is DataModality.TABULAR:
        return "table_row"
    if modality is DataModality.TEXT:
        return "text_record"
    if modality is DataModality.DOCUMENT_OCR:
        return "ocr_record"
    return f"{modality.value}_asset"


def _serialize_queues(queues: tuple[ReviewQueue, ...]) -> bytes:
    lines = [
        json.dumps(queue.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for queue in queues
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


__all__ = [
    "REVIEW_QUEUE_ARTIFACT_FORMAT",
    "REVIEW_QUEUE_ARTIFACT_KIND",
    "REVIEW_QUEUE_MEDIA_TYPE",
    "REVIEW_QUEUE_SCHEMA_VERSION",
    "SAFE_PREVIEW_ARTIFACT_KIND",
    "SAFE_PREVIEW_SCHEMA_VERSION",
    "BuildReviewQueuesArtifactResult",
    "BuildReviewQueuesRequest",
    "build_review_queues_artifact",
]
