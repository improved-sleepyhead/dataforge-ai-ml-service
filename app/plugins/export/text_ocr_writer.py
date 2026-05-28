"""Text/OCR redacted JSONL export writer (TASK-054).

The writer reads the immutable text/OCR JSONL source from object
storage and produces an export-grade redacted JSONL artifact: each
output line carries ``object_id``, ``redacted_text``,
``pii_token_count`` and ``redacted_text_sha256`` only — never raw
text, never raw PII matches.

When the privacy profile requires raw PII to be excluded (the default
for restricted text/OCR sources), the writer is the only sanctioned
way to materialize text/OCR data into an export package.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import ArtifactRef, ErrorCode, TextOcrSourceKind
from app.domain.common import NonEmptyStr, Sha256Digest
from app.plugins.text_ocr.validator import produce_redacted_jsonl

TEXT_OCR_EXPORT_KIND = "text_ocr_export_redacted_jsonl"
TEXT_OCR_EXPORT_FORMAT = "jsonl"
TEXT_OCR_EXPORT_MEDIA_TYPE = "application/jsonl"
TEXT_OCR_EXPORT_SCHEMA_VERSION = "text_ocr_export.v1"


class TextOcrExportWriterError(ValueError):
    """Raised when the text/OCR export writer cannot run safely."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.INVALID_JOB_PAYLOAD,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class TextOcrExportRequest(BaseModel):
    """Inputs for :func:`write_text_ocr_export`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_dataset_version_id: NonEmptyStr
    source_artifact: ArtifactRef
    source_kind: TextOcrSourceKind
    source_name: NonEmptyStr
    blocked_object_ids: tuple[NonEmptyStr, ...] = ()
    detect_pii_only_records: bool = False
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    export_name_prefix: NonEmptyStr = Field(default="text_ocr_export")


@dataclass(frozen=True)
class TextOcrExportArtifacts:
    """Result of :func:`write_text_ocr_export`."""

    redacted_artifact: RegisteredArtifact
    record_count: int
    pii_record_count: int
    excluded_blocked_count: int

    def all_artifact_refs(self) -> tuple[ArtifactRef, ...]:
        return (self.redacted_artifact.artifact_ref,)


def write_text_ocr_export(
    request: TextOcrExportRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> TextOcrExportArtifacts:
    """Persist a redacted JSONL export artifact for a text/OCR source."""
    source = storage.get(request.source_artifact.uri)
    redacted_bytes, records = produce_redacted_jsonl(
        source.data,
        detect_pii_only_records=request.detect_pii_only_records,
    )
    if not records:
        raise TextOcrExportWriterError(
            reason_code="empty_text_ocr_source",
            message="Text/OCR source did not produce any redacted records.",
            details={
                "source_kind": request.source_kind.value,
                "source_name": request.source_name,
            },
        )

    blocked_ids = set(request.blocked_object_ids)
    if blocked_ids:
        filtered_lines: list[bytes] = []
        kept_records = 0
        for line in redacted_bytes.splitlines():
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            object_id = payload.get("object_id")
            if object_id in blocked_ids:
                continue
            filtered_lines.append(line)
            kept_records += 1
        excluded = len(records) - kept_records
        redacted_bytes = b"\n".join(filtered_lines) + (b"\n" if filtered_lines else b"")
    else:
        excluded = 0

    pii_record_count = sum(1 for record in records if record.pii_token_count > 0)
    artifact = registry.save_artifact(
        artifact_kind=TEXT_OCR_EXPORT_KIND,
        data=redacted_bytes,
        artifact_format=TEXT_OCR_EXPORT_FORMAT,
        media_type=TEXT_OCR_EXPORT_MEDIA_TYPE,
        schema_version=TEXT_OCR_EXPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "export-name": request.export_name_prefix,
            "source-kind": request.source_kind.value,
            "source-name": request.source_name,
            "source-artifact-hash": request.source_artifact.hash,
            "candidate-version-id": request.candidate_dataset_version_id,
            "record-count": str(len(records) - excluded),
            "pii-record-count": str(pii_record_count),
        },
    )
    return TextOcrExportArtifacts(
        redacted_artifact=artifact,
        record_count=len(records) - excluded,
        pii_record_count=pii_record_count,
        excluded_blocked_count=excluded,
    )


def iter_redacted_jsonl_lines(payload: bytes) -> Iterable[dict[str, object]]:
    """Yield parsed redacted JSONL records from ``payload``.

    Helper used by tests and callers that need to assert on the
    redacted content without re-implementing the JSONL parsing.
    """
    text = payload.decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


__all__ = [
    "TEXT_OCR_EXPORT_FORMAT",
    "TEXT_OCR_EXPORT_KIND",
    "TEXT_OCR_EXPORT_MEDIA_TYPE",
    "TEXT_OCR_EXPORT_SCHEMA_VERSION",
    "TextOcrExportArtifacts",
    "TextOcrExportRequest",
    "TextOcrExportWriterError",
    "iter_redacted_jsonl_lines",
    "write_text_ocr_export",
]
