"""Tests for TASK-027: text/OCR mini plugin (JSONL validation + duplicates).

Acceptance criteria covered:

* plugin validates support_messages.jsonl and ocr_records.jsonl;
* invalid JSONL, empty and too-short records become validation issues;
* exact duplicates detected by normalized text hash;
* plugin output is contract-shaped and ready to feed an EvidenceBundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain import (
    TextOcrSourceKind,
)
from app.ingestion import open_archive_path
from app.plugins.text_ocr import (
    TextOcrBuildRequest,
    build_text_ocr_report,
    normalize_text,
    validate_ocr_records_jsonl,
    validate_support_messages_jsonl,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_DATASET_ID = "dataset_demo"
_VERSION_ID = "version_demo"
_PARENT_VERSION_ID = "version_demo_parent"
_JOB_ID = "compute_run_text_ocr"
_CONFIG_HASH = "sha256:" + "a" * 64


# ---------------------------------------------------------------------------
# Step 1: run plugin on demo fixtures
# ---------------------------------------------------------------------------


def test_validates_demo_support_messages(tmp_path: Path) -> None:
    payload = _read_archive_entry(tmp_path, "support_messages.jsonl")
    report = validate_support_messages_jsonl(payload)
    assert report.source_kind is TextOcrSourceKind.SUPPORT_MESSAGES
    assert report.source_name == "support_messages.jsonl"
    assert report.record_count == 15
    assert report.valid_record_count == 15
    assert report.issue_count == 0
    # Demo archive injects an exact duplicate of message #0; duplicate
    # detection may surface additional groups when multiple messages
    # share the same generic content (the demo also reuses the no-PII
    # template). Assert that at least one group exists and that the
    # injected duplicate is captured.
    assert len(report.duplicate_groups) >= 1
    all_object_ids = {oid for group in report.duplicate_groups for oid in group.object_ids}
    assert "support_0000" in all_object_ids
    assert "support_0014" in all_object_ids
    assert report.duplicate_record_count >= 2


def test_validates_demo_ocr_records(tmp_path: Path) -> None:
    payload = _read_archive_entry(tmp_path, "ocr_records.jsonl")
    report = validate_ocr_records_jsonl(payload)
    assert report.source_kind is TextOcrSourceKind.OCR_RECORDS
    assert report.record_count == 10
    assert report.valid_record_count == 10
    assert report.issue_count == 0
    # Demo injects one exact duplicate of OCR record #0; duplicate
    # detection may surface additional groups when multiple records
    # share the same generic content.
    assert len(report.duplicate_groups) >= 1
    all_object_ids = {oid for group in report.duplicate_groups for oid in group.object_ids}
    assert "ocr_0000" in all_object_ids
    assert "ocr_0009" in all_object_ids
    assert report.average_ocr_confidence is not None
    assert 0.0 < report.average_ocr_confidence <= 1.0


def test_build_text_ocr_report_aggregates_sources(tmp_path: Path) -> None:
    support_payload = _read_archive_entry(tmp_path, "support_messages.jsonl")
    ocr_payload = _read_archive_entry(tmp_path, "ocr_records.jsonl")
    sources = [
        validate_support_messages_jsonl(support_payload),
        validate_ocr_records_jsonl(ocr_payload),
    ]
    request = TextOcrBuildRequest(
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
    result = build_text_ocr_report(request=request, sources=sources)
    report = result.report
    assert report.report_schema_version == "text_ocr_report.v1"
    assert len(report.sources) == 2
    assert report.total_record_count == 15 + 10
    assert report.total_valid_record_count == 15 + 10
    assert report.total_duplicate_group_count >= 2
    assert report.total_duplicate_record_count >= 4


def test_text_ocr_report_validates_against_contract_pack(tmp_path: Path) -> None:
    support_payload = _read_archive_entry(tmp_path, "support_messages.jsonl")
    ocr_payload = _read_archive_entry(tmp_path, "ocr_records.jsonl")
    request = TextOcrBuildRequest(
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
    result = build_text_ocr_report(
        request=request,
        sources=[
            validate_support_messages_jsonl(support_payload, detect_pii=True),
            validate_ocr_records_jsonl(ocr_payload, detect_pii=True),
        ],
    )

    validate_contract_payload(
        load_contract_pack(),
        "text_ocr_report",
        result.report.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Step 2: duplicate report
# ---------------------------------------------------------------------------


def test_duplicate_detection_normalizes_whitespace_and_case() -> None:
    """Records that differ only in whitespace/case still hash to the same digest."""
    payload = _make_jsonl(
        [
            {"object_id": "a", "text": "Hello WORLD"},
            {"object_id": "b", "text": "  hello   world "},
            {"object_id": "c", "text": "different message"},
        ]
    )
    report = validate_support_messages_jsonl(payload)
    assert report.valid_record_count == 3
    assert report.issue_count == 0
    assert len(report.duplicate_groups) == 1
    group = report.duplicate_groups[0]
    assert set(group.object_ids) == {"a", "b"}
    assert group.count == 2


def test_normalize_text_is_deterministic() -> None:
    assert normalize_text("Hello   WORLD") == "hello world"
    assert normalize_text("  Hello\tWorld\n") == "hello world"
    assert normalize_text("a") == "a"


# ---------------------------------------------------------------------------
# Step 3: invalid JSONL row -> validation issue (no exception)
# ---------------------------------------------------------------------------


def test_invalid_json_line_is_reported_as_issue() -> None:
    payload = b'{"object_id": "ok", "text": "valid"}\n{not valid json}\n'
    report = validate_support_messages_jsonl(payload)
    assert report.valid_record_count == 1
    assert report.issue_count == 1
    issue = report.issues[0]
    assert issue.line_number == 2
    assert issue.reason_code == "invalid_json"


def test_missing_required_field_becomes_issue() -> None:
    payload = _make_jsonl(
        [
            {"object_id": "ok", "text": "valid"},
            {"object_id": "missing_text"},  # no text
            {"text": "missing_id"},  # no object_id
        ]
    )
    report = validate_support_messages_jsonl(payload)
    assert report.valid_record_count == 1
    assert report.issue_count == 2
    reasons = {issue.reason_code for issue in report.issues}
    assert reasons == {"missing_required_field"}


def test_empty_text_becomes_issue() -> None:
    payload = _make_jsonl(
        [
            {"object_id": "a", "text": ""},
            {"object_id": "b", "text": "   "},
            {"object_id": "c", "text": "valid"},
        ]
    )
    report = validate_support_messages_jsonl(payload)
    assert report.valid_record_count == 1
    assert report.issue_count == 2
    # The first record fails on missing_required_field (empty string is
    # treated as missing), the whitespace-only one fails on empty_text.
    reasons = {issue.reason_code for issue in report.issues}
    assert reasons == {"missing_required_field", "empty_text"}


def test_text_too_short_becomes_issue() -> None:
    payload = _make_jsonl(
        [
            {"object_id": "a", "text": "hi"},
            {"object_id": "b", "text": "this is long enough"},
        ]
    )
    report = validate_support_messages_jsonl(payload, min_text_length=5)
    assert report.valid_record_count == 1
    assert report.issue_count == 1
    issue = report.issues[0]
    assert issue.reason_code == "text_too_short"
    assert issue.object_id == "a"


def test_invalid_ocr_confidence_becomes_issue() -> None:
    payload = _make_jsonl(
        [
            {
                "object_id": "ok",
                "document_id": "doc1",
                "page": 1,
                "text": "valid",
                "ocr_confidence": 0.95,
            },
            {
                "object_id": "bad",
                "document_id": "doc2",
                "page": 1,
                "text": "valid",
                "ocr_confidence": 1.5,
            },
        ]
    )
    report = validate_ocr_records_jsonl(payload)
    assert report.valid_record_count == 1
    assert report.issue_count == 1
    issue = report.issues[0]
    assert issue.reason_code == "invalid_ocr_confidence"
    assert issue.object_id == "bad"


def test_duplicate_object_id_becomes_issue() -> None:
    payload = _make_jsonl(
        [
            {"object_id": "a", "text": "first"},
            {"object_id": "a", "text": "second"},  # duplicate object_id
        ]
    )
    report = validate_support_messages_jsonl(payload)
    assert report.valid_record_count == 1
    assert report.issue_count == 1
    issue = report.issues[0]
    assert issue.reason_code == "duplicate_object_id"
    assert issue.object_id == "a"


def test_record_not_object_becomes_issue() -> None:
    payload = b'{"object_id": "ok", "text": "valid"}\n["not", "an", "object"]\n'
    report = validate_support_messages_jsonl(payload)
    assert report.valid_record_count == 1
    assert report.issue_count == 1
    assert report.issues[0].reason_code == "row_not_object"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_archive_entry(tmp_path: Path, name: str) -> bytes:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        for descriptor in reader.descriptors():
            if descriptor.name == name:
                with descriptor.open() as handle:
                    return handle.read()
    raise AssertionError(f"archive entry {name!r} not found")


def _make_jsonl(records: list[dict[str, object]]) -> bytes:
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")
