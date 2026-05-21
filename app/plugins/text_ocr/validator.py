"""Validator and duplicate detector for text/OCR JSONL records.

The validator walks ``support_messages.jsonl`` / ``ocr_records.jsonl``
line by line:

* parses each line as JSON and checks required fields;
* enforces minimum text length (``min_text_length``, default 1) and
  rejects empty/whitespace-only text;
* records validation issues as :class:`TextValidationIssue` with stable
  reason codes (``invalid_json``, ``missing_required_field``,
  ``empty_text``, ``text_too_short``, ``invalid_ocr_confidence``,
  ``duplicate_object_id`` for in-source object_id collisions);
* computes a normalized text sha256 (lowercase + collapsed whitespace)
  and groups records by that hash to detect exact duplicates;
* aggregates per-source counts and lengths into a
  :class:`TextOcrSourceReport`.

The ``build_text_ocr_report`` entry point combines one or more sources
into a top-level :class:`TextOcrReport` ready for an EvidenceBundle.
The plugin never logs or echoes raw text — issues carry only stable
identifiers and reason codes.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain import (
    TextDuplicateGroup,
    TextOcrReport,
    TextOcrSourceKind,
    TextOcrSourceReport,
    TextValidationIssue,
)
from app.domain.common import NonEmptyStr, Sha256Digest

SUPPORT_MESSAGE_SCHEMA_NAME = "support_message"
OCR_RECORD_SCHEMA_NAME = "ocr_record"
TEXT_OCR_REPORT_SCHEMA_VERSION = "text_ocr_report.v1"

_DEFAULT_MIN_TEXT_LENGTH = 1
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextRecordParser:
    """Configurable parser for one source kind.

    The parser carries the contract it expects: required fields,
    minimum text length, and whether ``ocr_confidence`` is required.
    """

    source_kind: TextOcrSourceKind
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    min_text_length: int = _DEFAULT_MIN_TEXT_LENGTH
    require_ocr_confidence: bool = False


_SUPPORT_PARSER = TextRecordParser(
    source_kind=TextOcrSourceKind.SUPPORT_MESSAGES,
    required_fields=("object_id", "text"),
    optional_fields=("case_id", "language"),
)
_OCR_PARSER = TextRecordParser(
    source_kind=TextOcrSourceKind.OCR_RECORDS,
    required_fields=("object_id", "text"),
    optional_fields=("document_id", "page", "language", "ocr_confidence"),
    require_ocr_confidence=False,
)


class TextOcrBuildRequest(BaseModel):
    """Inputs the text/OCR plugin requires from the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    min_text_length: int = Field(default=_DEFAULT_MIN_TEXT_LENGTH, ge=1)


@dataclass(frozen=True)
class BuildTextOcrReportResult:
    """Return value of :func:`build_text_ocr_report`."""

    report: TextOcrReport


def normalize_text(text: str) -> str:
    """Lowercase + whitespace-collapse normalization for duplicate hashing.

    The function is intentionally simple: lowercase, strip leading/
    trailing whitespace, collapse internal runs of whitespace into a
    single space. PII detection / token normalization (Unicode NFKC,
    accent stripping, etc.) is out of scope and lives in TASK-028.
    """
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def validate_support_messages_jsonl(
    data: bytes,
    *,
    source_name: str = "support_messages.jsonl",
    min_text_length: int = _DEFAULT_MIN_TEXT_LENGTH,
) -> TextOcrSourceReport:
    parser = TextRecordParser(
        source_kind=_SUPPORT_PARSER.source_kind,
        required_fields=_SUPPORT_PARSER.required_fields,
        optional_fields=_SUPPORT_PARSER.optional_fields,
        min_text_length=min_text_length,
        require_ocr_confidence=False,
    )
    return _validate_jsonl(data, parser=parser, source_name=source_name)


def validate_ocr_records_jsonl(
    data: bytes,
    *,
    source_name: str = "ocr_records.jsonl",
    min_text_length: int = _DEFAULT_MIN_TEXT_LENGTH,
) -> TextOcrSourceReport:
    parser = TextRecordParser(
        source_kind=_OCR_PARSER.source_kind,
        required_fields=_OCR_PARSER.required_fields,
        optional_fields=_OCR_PARSER.optional_fields,
        min_text_length=min_text_length,
        require_ocr_confidence=_OCR_PARSER.require_ocr_confidence,
    )
    return _validate_jsonl(data, parser=parser, source_name=source_name)


def build_text_ocr_report(
    *,
    request: TextOcrBuildRequest,
    sources: Iterable[TextOcrSourceReport],
    report_id: str | None = None,
    generated_at: datetime | None = None,
) -> BuildTextOcrReportResult:
    """Combine per-source reports into a top-level :class:`TextOcrReport`."""
    sources_tuple = tuple(sources)
    total_records = sum(s.record_count for s in sources_tuple)
    total_valid = sum(s.valid_record_count for s in sources_tuple)
    total_issues = sum(s.issue_count for s in sources_tuple)
    total_dup_groups = sum(len(s.duplicate_groups) for s in sources_tuple)
    total_dup_records = sum(s.duplicate_record_count for s in sources_tuple)
    report = TextOcrReport(
        report_id=report_id or f"text_ocr_report_{uuid.uuid4().hex[:16]}",
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        parent_version_id=request.parent_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        sources=sources_tuple,
        total_record_count=total_records,
        total_valid_record_count=total_valid,
        total_issue_count=total_issues,
        total_duplicate_group_count=total_dup_groups,
        total_duplicate_record_count=total_dup_records,
        generated_at=generated_at or datetime.now(UTC),
    )
    return BuildTextOcrReportResult(report=report)


# ---------------------------------------------------------------------------
# core JSONL walker
# ---------------------------------------------------------------------------


def _validate_jsonl(
    data: bytes,
    *,
    parser: TextRecordParser,
    source_name: str,
) -> TextOcrSourceReport:
    issues: list[TextValidationIssue] = []
    records: list[_ParsedRecord] = []
    seen_object_ids: dict[str, int] = {}

    text_lines = _decode_or_issue(data, issues)
    for line_number, line in enumerate(text_lines, start=1):
        if not line.strip():
            continue
        record_or_issue = _parse_line(
            line=line, line_number=line_number, parser=parser
        )
        if isinstance(record_or_issue, TextValidationIssue):
            issues.append(record_or_issue)
            continue
        record = record_or_issue
        # Detect in-source object_id collisions before counting the
        # record as valid; the second occurrence becomes an issue.
        if record.object_id in seen_object_ids:
            issues.append(
                TextValidationIssue(
                    line_number=line_number,
                    object_id=record.object_id,
                    reason_code="duplicate_object_id",
                    message=(
                        f"object_id repeats line "
                        f"{seen_object_ids[record.object_id]}"
                    ),
                )
            )
            continue
        seen_object_ids[record.object_id] = line_number
        records.append(record)

    duplicate_groups = _build_duplicate_groups(records)
    duplicate_object_ids: list[str] = []
    duplicate_record_count = 0
    for group in duplicate_groups:
        duplicate_record_count += group.count
        duplicate_object_ids.extend(group.object_ids)

    text_lengths = [len(record.raw_text) for record in records]
    average_text_length = (
        sum(text_lengths) / len(text_lengths) if text_lengths else 0.0
    )
    min_length = min(text_lengths) if text_lengths else 0
    max_length = max(text_lengths) if text_lengths else 0
    ocr_confidences = [
        record.ocr_confidence
        for record in records
        if record.ocr_confidence is not None
    ]
    average_ocr_confidence = (
        sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
    )

    return TextOcrSourceReport(
        source_kind=parser.source_kind,
        source_name=source_name,
        record_count=len(records) + len(issues),
        valid_record_count=len(records),
        issue_count=len(issues),
        issues=tuple(issues),
        duplicate_groups=duplicate_groups,
        duplicate_record_count=duplicate_record_count,
        duplicate_object_ids=tuple(sorted(set(duplicate_object_ids))),
        average_text_length=average_text_length,
        min_text_length=min_length,
        max_text_length=max_length,
        average_ocr_confidence=average_ocr_confidence,
    )


@dataclass(frozen=True)
class _ParsedRecord:
    object_id: str
    raw_text: str
    normalized_text_hash: str
    ocr_confidence: float | None


def _decode_or_issue(data: bytes, issues: list[TextValidationIssue]) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        issues.append(
            TextValidationIssue(
                line_number=1,
                reason_code="not_utf8_jsonl",
                message="payload must be UTF-8 encoded JSONL",
            )
        )
        return []
    return text.splitlines()


def _parse_line(
    *,
    line: str,
    line_number: int,
    parser: TextRecordParser,
) -> _ParsedRecord | TextValidationIssue:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return TextValidationIssue(
            line_number=line_number,
            reason_code="invalid_json",
            message=f"line {line_number} is not valid JSON: {exc.msg}",
        )
    if not isinstance(payload, dict):
        return TextValidationIssue(
            line_number=line_number,
            reason_code="row_not_object",
            message=f"line {line_number} is not a JSON object",
        )
    for field in parser.required_fields:
        if field not in payload or payload[field] in (None, ""):
            return TextValidationIssue(
                line_number=line_number,
                object_id=_safe_object_id(payload),
                reason_code="missing_required_field",
                message=f"line {line_number} missing required field '{field}'",
            )
    raw_text = payload["text"]
    if not isinstance(raw_text, str):
        return TextValidationIssue(
            line_number=line_number,
            object_id=_safe_object_id(payload),
            reason_code="invalid_text_type",
            message=f"line {line_number} 'text' must be a string",
        )
    if not raw_text.strip():
        return TextValidationIssue(
            line_number=line_number,
            object_id=_safe_object_id(payload),
            reason_code="empty_text",
            message=f"line {line_number} 'text' is empty",
        )
    if len(raw_text.strip()) < parser.min_text_length:
        return TextValidationIssue(
            line_number=line_number,
            object_id=_safe_object_id(payload),
            reason_code="text_too_short",
            message=(
                f"line {line_number} 'text' length below {parser.min_text_length}"
            ),
        )
    object_id_value = payload["object_id"]
    if not isinstance(object_id_value, str):
        return TextValidationIssue(
            line_number=line_number,
            reason_code="invalid_object_id_type",
            message=f"line {line_number} 'object_id' must be a string",
        )
    ocr_confidence: float | None = None
    if "ocr_confidence" in payload and payload["ocr_confidence"] is not None:
        candidate = payload["ocr_confidence"]
        if not isinstance(candidate, (int, float)) or not (0.0 <= candidate <= 1.0):
            return TextValidationIssue(
                line_number=line_number,
                object_id=object_id_value,
                reason_code="invalid_ocr_confidence",
                message=(
                    f"line {line_number} 'ocr_confidence' must be a number in [0, 1]"
                ),
            )
        ocr_confidence = float(candidate)
    if parser.require_ocr_confidence and ocr_confidence is None:
        return TextValidationIssue(
            line_number=line_number,
            object_id=object_id_value,
            reason_code="missing_required_field",
            message=f"line {line_number} missing required field 'ocr_confidence'",
        )
    normalized = normalize_text(raw_text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return _ParsedRecord(
        object_id=object_id_value,
        raw_text=raw_text,
        normalized_text_hash=f"sha256:{digest}",
        ocr_confidence=ocr_confidence,
    )


def _safe_object_id(payload: dict[str, object]) -> str | None:
    candidate = payload.get("object_id")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _build_duplicate_groups(
    records: list[_ParsedRecord],
) -> tuple[TextDuplicateGroup, ...]:
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(record.normalized_text_hash, []).append(record.object_id)
    duplicates: list[TextDuplicateGroup] = []
    for digest, object_ids in sorted(groups.items()):
        if len(object_ids) < 2:
            continue
        duplicates.append(
            TextDuplicateGroup(
                text_sha256=digest,
                object_ids=tuple(sorted(object_ids)),
                count=len(object_ids),
            )
        )
    return tuple(duplicates)


__all__ = [
    "BuildTextOcrReportResult",
    "OCR_RECORD_SCHEMA_NAME",
    "SUPPORT_MESSAGE_SCHEMA_NAME",
    "TEXT_OCR_REPORT_SCHEMA_VERSION",
    "TextOcrBuildRequest",
    "TextRecordParser",
    "build_text_ocr_report",
    "normalize_text",
    "validate_ocr_records_jsonl",
    "validate_support_messages_jsonl",
]
