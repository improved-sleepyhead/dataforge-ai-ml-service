"""Text/OCR plugin contracts.

These contracts describe the output of the text/OCR mini plugin that
validates ``support_messages.jsonl`` and ``ocr_records.jsonl`` from the
ingestion archive and detects exact duplicates over normalized text.

TASK-027 scope is intentionally narrow:

* per-record validation issues with stable reason codes (invalid JSONL,
  missing fields, empty/short text, OCR confidence out of range);
* exact-duplicate detection over normalized text with affected
  ``object_id`` set;
* aggregate counts that can flow into ``EvidenceBundle``;
* explicit lineage to the source artifact.

PII detection and redaction are intentionally out of scope here and
land in TASK-028.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.common import NonEmptyStr, Score, Sha256Digest


class TextOcrSourceKind(StrEnum):
    """Allowed source kinds the plugin can validate."""

    SUPPORT_MESSAGES = "support_messages"
    OCR_RECORDS = "ocr_records"


class TextValidationIssue(BaseModel):
    """One validation issue for a single text/OCR record.

    ``reason_code`` is a short machine-readable token so the API/report
    layer can render an explanation without re-parsing the issue
    message. ``object_id`` is the record's source key when present;
    issues that fail before parsing (e.g. invalid JSON) carry a line
    number instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    line_number: int = Field(ge=1)
    object_id: str | None = None
    reason_code: NonEmptyStr
    message: NonEmptyStr


class TextDuplicateGroup(BaseModel):
    """Exact-duplicate group keyed by normalized text hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text_sha256: Sha256Digest
    object_ids: tuple[NonEmptyStr, ...]
    count: int = Field(ge=2)


class PiiCategory(StrEnum):
    """Detected PII categories in text/OCR records."""

    EMAIL = "email"
    PHONE = "phone"
    PASSPORT = "passport"
    PAYMENT_CARD = "payment_card"
    BANK_ACCOUNT = "bank_account"
    SECRET = "secret"


class PiiFinding(BaseModel):
    """One PII match within a text/OCR record.

    The finding never echoes the raw matched value — only a stable
    category, the position range and the count of equivalent matches in
    the record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: PiiCategory
    occurrence_count: int = Field(ge=1)


class TextPiiFindingsForRecord(BaseModel):
    """Aggregate per-record PII findings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    findings: tuple[PiiFinding, ...]
    pii_token_count: int = Field(ge=0)
    pii_risk_score: Score
    redacted_text_sha256: Sha256Digest


class RedactionStatus(StrEnum):
    """Outcome of running the redactor against a record."""

    NOT_NEEDED = "not_needed"
    REDACTED = "redacted"
    BLOCKED = "blocked"
    REQUIRES_REVIEW = "requires_review"


class TextOcrSourceReport(BaseModel):
    """Per-source validation/duplicate report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: TextOcrSourceKind
    source_name: NonEmptyStr
    record_count: int = Field(ge=0)
    valid_record_count: int = Field(ge=0)
    issue_count: int = Field(ge=0)
    issues: tuple[TextValidationIssue, ...] = ()
    duplicate_groups: tuple[TextDuplicateGroup, ...] = ()
    duplicate_record_count: int = Field(ge=0)
    duplicate_object_ids: tuple[NonEmptyStr, ...] = ()
    average_text_length: float = Field(ge=0.0)
    min_text_length: int = Field(ge=0)
    max_text_length: int = Field(ge=0)
    average_ocr_confidence: Score | None = None
    pii_findings: tuple[TextPiiFindingsForRecord, ...] = ()
    pii_token_count: int = Field(default=0, ge=0)
    pii_record_count: int = Field(default=0, ge=0)
    redacted_record_count: int = Field(default=0, ge=0)
    redacted_artifact_uri: str | None = None
    redacted_artifact_hash: str | None = None
    review_queue_object_ids: tuple[NonEmptyStr, ...] = ()


class TextOcrReport(BaseModel):
    """Top-level text/OCR plugin report.

    The report aggregates one or more ``TextOcrSourceReport`` blocks
    (one per validated archive entry) plus a top-level summary so the
    Decision Core / EvidenceBundle layer can consume both per-source and
    dataset-wide signals without re-walking source records.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "text_ocr_report.v1"
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    sources: tuple[TextOcrSourceReport, ...]
    total_record_count: int = Field(ge=0)
    total_valid_record_count: int = Field(ge=0)
    total_issue_count: int = Field(ge=0)
    total_duplicate_group_count: int = Field(ge=0)
    total_duplicate_record_count: int = Field(ge=0)
    total_pii_record_count: int = Field(default=0, ge=0)
    total_pii_token_count: int = Field(default=0, ge=0)
    total_redacted_record_count: int = Field(default=0, ge=0)
    review_queue_object_ids: tuple[NonEmptyStr, ...] = ()
    generated_at: datetime


__all__ = [
    "PiiCategory",
    "PiiFinding",
    "RedactionStatus",
    "TextDuplicateGroup",
    "TextOcrReport",
    "TextOcrSourceKind",
    "TextOcrSourceReport",
    "TextPiiFindingsForRecord",
    "TextValidationIssue",
]
