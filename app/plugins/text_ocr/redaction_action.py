"""Approved REDACT_PII action executor for text/OCR artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    ErrorCode,
    EvidenceRef,
    TextOcrReport,
    TextOcrSourceKind,
    TextPiiFindingsForRecord,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.plugins.text_ocr.pii import RedactedRecord, aggregate_pii
from app.plugins.text_ocr.validator import (
    TEXT_OCR_REDACTED_ARTIFACT_KIND,
    TEXT_OCR_REDACTED_FORMAT,
    TEXT_OCR_REDACTED_MEDIA_TYPE,
    TEXT_OCR_REDACTED_SCHEMA_VERSION,
    produce_redacted_jsonl,
)

REDACT_PII_STEP_TYPE = "REDACT_PII"
TEXT_OCR_PRIVACY_POLICY_VERSION = "privacy_v0"
_RAW_RESTRICTED_REASON = "raw_restricted_text_ocr_requires_redaction"
_COMPLETE_REDACTION_REASON = "redacted_export_requires_complete_records"
_SUPPORTED_SOURCE_KINDS = {
    TextOcrSourceKind.SUPPORT_MESSAGES,
    TextOcrSourceKind.OCR_RECORDS,
}


class TextOcrRedactionExecutionError(ValueError):
    """Raised when an approved REDACT_PII step is unsafe or invalid."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.ACTION_PLAN_PRECONDITION_FAILED,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class TextOcrExportPolicyStatus(StrEnum):
    """Export policy decision for raw vs redacted text/OCR artifacts."""

    READY = "ready"
    BLOCKED = "blocked"


class TextOcrExportPolicyDecision(BaseModel):
    """Machine-readable export policy for text/OCR artifacts after redaction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TextOcrExportPolicyStatus
    raw_artifact_allowed: bool
    redacted_artifact_required: bool
    export_artifact: ArtifactRef | None
    reason_codes: tuple[NonEmptyStr, ...]


class ExecuteTextOcrRedactionRequest(BaseModel):
    """Inputs for executing one approved REDACT_PII text/OCR step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step: ActionPlanStep
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    source_kind: TextOcrSourceKind
    source_name: NonEmptyStr
    source_artifact: ArtifactRef
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    evidence_refs: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class ExecuteTextOcrRedactionResult:
    """Artifact and metadata produced by REDACT_PII execution."""

    redacted_artifact: RegisteredArtifact
    redacted_records: tuple[RedactedRecord, ...]
    pii_findings: tuple[TextPiiFindingsForRecord, ...]
    pii_record_count: int
    pii_token_count: int
    export_policy: TextOcrExportPolicyDecision


def execute_text_ocr_redaction_action(
    request: ExecuteTextOcrRedactionRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> ExecuteTextOcrRedactionResult:
    """Execute one safe REDACT_PII step and persist a redacted JSONL artifact.

    The source text/OCR artifact is read once and never overwritten. The output
    artifact includes every valid input record with only ``redacted_text`` and
    redaction metadata, so it can be used as the exportable replacement for raw
    restricted text/OCR payloads.
    """
    _validate_step(request)
    source = storage.get(request.source_artifact.uri)
    redacted_bytes, records = produce_redacted_jsonl(
        source.data,
        detect_pii_only_records=False,
    )
    if not redacted_bytes or not records:
        raise TextOcrRedactionExecutionError(
            reason_code="empty_redacted_text_ocr_output",
            message="REDACT_PII produced no valid redacted JSONL records.",
            details={"step_id": request.step.step_id, "source_name": request.source_name},
        )

    redacted_records = tuple(records)
    pii_findings, pii_record_count, pii_token_count = aggregate_pii(redacted_records)
    redacted_record_count = sum(1 for record in redacted_records if record.pii_token_count > 0)
    redacted_artifact = registry.save_artifact(
        artifact_kind=TEXT_OCR_REDACTED_ARTIFACT_KIND,
        data=redacted_bytes,
        artifact_format=TEXT_OCR_REDACTED_FORMAT,
        media_type=TEXT_OCR_REDACTED_MEDIA_TYPE,
        schema_version=TEXT_OCR_REDACTED_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": request.step.step_id,
            "source-dataset-version-id": request.source_dataset_version_id,
            "source-artifact-hash": request.source_artifact.hash,
            "source-artifact-uri": request.source_artifact.uri,
            "source-kind": request.source_kind.value,
            "source-name": request.source_name,
            "privacy-policy": TEXT_OCR_PRIVACY_POLICY_VERSION,
            "redaction-status": "redacted" if pii_record_count else "not_needed",
            "pii-record-count": str(pii_record_count),
            "pii-token-count": str(pii_token_count),
            "redacted-record-count": str(redacted_record_count),
            "output-record-count": str(len(redacted_records)),
            "raw-export-policy": "redacted_only" if pii_record_count else "raw_allowed",
            "evidence-refs": _evidence_refs_metadata(request.evidence_refs),
        },
    )
    export_policy = evaluate_text_ocr_export_policy(
        text_ocr_report=None,
        source_artifact=request.source_artifact,
        redacted_artifact=redacted_artifact.artifact_ref,
        pii_record_count=pii_record_count,
    )
    return ExecuteTextOcrRedactionResult(
        redacted_artifact=redacted_artifact,
        redacted_records=redacted_records,
        pii_findings=pii_findings,
        pii_record_count=pii_record_count,
        pii_token_count=pii_token_count,
        export_policy=export_policy,
    )


def evaluate_text_ocr_export_policy(
    *,
    source_artifact: ArtifactRef,
    text_ocr_report: TextOcrReport | None = None,
    redacted_artifact: ArtifactRef | None = None,
    pii_record_count: int | None = None,
) -> TextOcrExportPolicyDecision:
    """Block raw restricted text/OCR export unless a redacted artifact is used."""
    detected_pii_records = (
        text_ocr_report.total_pii_record_count
        if text_ocr_report is not None
        else (0 if pii_record_count is None else pii_record_count)
    )
    if detected_pii_records <= 0:
        return TextOcrExportPolicyDecision(
            status=TextOcrExportPolicyStatus.READY,
            raw_artifact_allowed=True,
            redacted_artifact_required=False,
            export_artifact=source_artifact,
            reason_codes=(),
        )
    if redacted_artifact is None:
        return TextOcrExportPolicyDecision(
            status=TextOcrExportPolicyStatus.BLOCKED,
            raw_artifact_allowed=False,
            redacted_artifact_required=True,
            export_artifact=None,
            reason_codes=(_RAW_RESTRICTED_REASON, "PII_UNMASKED"),
        )
    if redacted_artifact.kind != TEXT_OCR_REDACTED_ARTIFACT_KIND:
        return TextOcrExportPolicyDecision(
            status=TextOcrExportPolicyStatus.BLOCKED,
            raw_artifact_allowed=False,
            redacted_artifact_required=True,
            export_artifact=None,
            reason_codes=("redacted_text_ocr_artifact_required",),
        )
    return TextOcrExportPolicyDecision(
        status=TextOcrExportPolicyStatus.READY,
        raw_artifact_allowed=False,
        redacted_artifact_required=True,
        export_artifact=redacted_artifact,
        reason_codes=("raw_text_ocr_export_replaced_by_redacted_artifact",),
    )


def _validate_step(request: ExecuteTextOcrRedactionRequest) -> None:
    step = request.step
    if step.type != REDACT_PII_STEP_TYPE:
        raise TextOcrRedactionExecutionError(
            reason_code="unsupported_action_step_type",
            message="Only REDACT_PII steps can be executed by the text/OCR redactor.",
            details={"step_id": step.step_id, "step_type": step.type},
        )
    if request.source_kind not in _SUPPORTED_SOURCE_KINDS:
        raise TextOcrRedactionExecutionError(
            reason_code="unsupported_text_ocr_source_kind",
            message="REDACT_PII supports only support_messages and ocr_records sources.",
            details={"source_kind": request.source_kind.value},
        )
    configured_source_kind = step.config.get("source_kind")
    if configured_source_kind is not None and configured_source_kind != request.source_kind.value:
        raise TextOcrRedactionExecutionError(
            reason_code="source_kind_mismatch",
            message="ActionPlan step source_kind does not match the execution request.",
            details={"step_id": step.step_id, "source_kind": request.source_kind.value},
        )
    configured_source_name = step.config.get("source_name")
    if configured_source_name is not None and configured_source_name != request.source_name:
        raise TextOcrRedactionExecutionError(
            reason_code="source_name_mismatch",
            message="ActionPlan step source_name does not match the execution request.",
            details={"step_id": step.step_id, "source_name": request.source_name},
        )
    include_clean_records = step.config.get("include_clean_records", True)
    if include_clean_records is not True:
        raise TextOcrRedactionExecutionError(
            reason_code=_COMPLETE_REDACTION_REASON,
            message="REDACT_PII export artifacts must include clean and redacted records.",
            code=ErrorCode.POLICY_BLOCKED,
            details={"step_id": step.step_id},
        )
    if request.source_artifact.media_type != TEXT_OCR_REDACTED_MEDIA_TYPE:
        raise TextOcrRedactionExecutionError(
            reason_code="unsupported_text_ocr_artifact_media_type",
            message="REDACT_PII source artifact must be application/jsonl.",
            details={
                "step_id": step.step_id,
                "media_type": request.source_artifact.media_type,
            },
        )


def _evidence_refs_metadata(evidence_refs: tuple[EvidenceRef, ...]) -> str:
    return json.dumps(
        [ref.model_dump(mode="json") for ref in evidence_refs],
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "ExecuteTextOcrRedactionRequest",
    "ExecuteTextOcrRedactionResult",
    "REDACT_PII_STEP_TYPE",
    "TEXT_OCR_PRIVACY_POLICY_VERSION",
    "TextOcrExportPolicyDecision",
    "TextOcrExportPolicyStatus",
    "TextOcrRedactionExecutionError",
    "evaluate_text_ocr_export_policy",
    "execute_text_ocr_redaction_action",
]
