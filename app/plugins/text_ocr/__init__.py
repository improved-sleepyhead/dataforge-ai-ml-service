"""Text/OCR mini plugin: JSONL validation and exact-duplicate detection.

The plugin validates ``support_messages.jsonl`` and ``ocr_records.jsonl``
records against a small contract (``object_id``, ``text`` non-empty,
optional ``language``/``case_id``/``document_id``/``page``/``ocr_confidence``)
and detects exact duplicates over a normalized text hash.

PII detection and redaction are intentionally separated and live in
TASK-028 so this plugin can stay deterministic, stdlib-only and policy
free.
"""

from app.plugins.text_ocr.pii import (
    RedactedRecord,
    aggregate_pii,
    detect_pii,
    redact_record,
)
from app.plugins.text_ocr.validator import (
    OCR_RECORD_SCHEMA_NAME,
    SUPPORT_MESSAGE_SCHEMA_NAME,
    TEXT_OCR_REDACTED_ARTIFACT_KIND,
    TEXT_OCR_REDACTED_FORMAT,
    TEXT_OCR_REDACTED_MEDIA_TYPE,
    TEXT_OCR_REDACTED_SCHEMA_VERSION,
    TEXT_OCR_REPORT_SCHEMA_VERSION,
    BuildTextOcrReportResult,
    PersistedRedactedJsonl,
    TextOcrBuildRequest,
    TextRecordParser,
    build_text_ocr_report,
    normalize_text,
    persist_redacted_jsonl,
    produce_redacted_jsonl,
    validate_ocr_records_jsonl,
    validate_support_messages_jsonl,
)

__all__ = [
    "BuildTextOcrReportResult",
    "OCR_RECORD_SCHEMA_NAME",
    "PersistedRedactedJsonl",
    "SUPPORT_MESSAGE_SCHEMA_NAME",
    "TEXT_OCR_REPORT_SCHEMA_VERSION",
    "TEXT_OCR_REDACTED_ARTIFACT_KIND",
    "TEXT_OCR_REDACTED_FORMAT",
    "TEXT_OCR_REDACTED_MEDIA_TYPE",
    "TEXT_OCR_REDACTED_SCHEMA_VERSION",
    "RedactedRecord",
    "TextOcrBuildRequest",
    "TextRecordParser",
    "aggregate_pii",
    "build_text_ocr_report",
    "detect_pii",
    "normalize_text",
    "persist_redacted_jsonl",
    "produce_redacted_jsonl",
    "redact_record",
    "validate_ocr_records_jsonl",
    "validate_support_messages_jsonl",
]
