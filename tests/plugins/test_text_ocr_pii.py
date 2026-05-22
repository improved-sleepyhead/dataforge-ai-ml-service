"""Tests for TASK-028: PII detection and redaction for text/OCR mini plugin.

Acceptance criteria covered:

* PII detector finds email, phone, passport-like ids, payment-card and
  account-like tokens, generic credentials/secrets;
* raw PII never reaches logs (verified via app.telemetry.scan_log_text);
* redacted artifact is produced for risky records;
* PII findings drive privacy_risk score and review queue candidates.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import PiiCategory
from app.domain.errors import ErrorCode
from app.ingestion import open_archive_path
from app.plugins.text_ocr import (
    TEXT_OCR_REDACTED_ARTIFACT_KIND,
    TEXT_OCR_REDACTED_SCHEMA_VERSION,
    TextOcrBuildRequest,
    aggregate_pii,
    detect_pii,
    produce_redacted_jsonl,
    redact_record,
    validate_ocr_records_jsonl,
    validate_support_messages_jsonl,
)
from app.telemetry import scan_log_text
from tests.fixtures.demo_archive import build_demo_archive

# Synthetic test PII tokens (no real customer data).
_FAKE_EMAIL = "alice.doe@example.test"
_FAKE_PHONE = "+1 (415) 555-0123"
_FAKE_PASSPORT = "1234 567890"
_FAKE_CARD = "4111 1111 1111 1111"  # valid Luhn
_FAKE_NON_CARD = "1234 5678 9012 3457"  # invalid Luhn -> not flagged as card
_FAKE_SECRET = "api_key=AKIAIOSFODNN7EXAMPLE"
_CONFIG_HASH = "sha256:" + "a" * 64


def test_detect_pii_email_phone_passport_card_secret() -> None:
    text = (
        "Email me at "
        + _FAKE_EMAIL
        + " or call "
        + _FAKE_PHONE
        + ". Passport "
        + _FAKE_PASSPORT
        + ". Card "
        + _FAKE_CARD
        + ". token=secret_value"
    )
    counts, redacted = detect_pii(text)
    assert PiiCategory.EMAIL in counts
    assert PiiCategory.PHONE in counts
    assert PiiCategory.PASSPORT in counts
    assert PiiCategory.PAYMENT_CARD in counts
    assert PiiCategory.SECRET in counts
    # Redacted text must not contain raw tokens.
    assert _FAKE_EMAIL not in redacted
    assert _FAKE_PHONE not in redacted
    assert _FAKE_CARD not in redacted
    assert "AKIA" not in redacted
    # Standard placeholder tokens are present.
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PAYMENT_CARD]" in redacted


def test_payment_card_requires_luhn() -> None:
    """Long digit runs that fail Luhn must not be flagged as payment cards.

    They may still be redacted by the phone-style long-digit detector
    (DataForge prefers false positives over leaks), but the
    payment-card category specifically is gated by Luhn.
    """
    text = "Some non-card number " + _FAKE_NON_CARD
    counts, _redacted = detect_pii(text)
    assert PiiCategory.PAYMENT_CARD not in counts


def test_redact_record_produces_no_raw_pii() -> None:
    record = redact_record(
        object_id="obj_1",
        text="contact " + _FAKE_EMAIL + " or " + _FAKE_PHONE,
    )
    assert _FAKE_EMAIL not in record.redacted_text
    assert _FAKE_PHONE not in record.redacted_text
    assert record.pii_token_count >= 2
    assert 0.0 < record.pii_risk_score <= 1.0
    assert record.redacted_text_sha256.startswith("sha256:")
    # Findings carry only category + count, no raw value.
    for finding in record.findings:
        assert finding.occurrence_count >= 1


def test_clean_text_has_zero_findings_and_score_zero() -> None:
    record = redact_record(object_id="obj_1", text="No PII here, all clean.")
    assert record.pii_token_count == 0
    assert record.pii_risk_score == 0.0
    assert record.findings == ()


def test_aggregate_pii_filters_clean_records() -> None:
    records = [
        redact_record(object_id="dirty", text="contact " + _FAKE_EMAIL),
        redact_record(object_id="clean", text="all good"),
    ]
    findings, pii_record_count, total_tokens = aggregate_pii(records)
    assert pii_record_count == 1
    assert total_tokens >= 1
    assert {f.object_id for f in findings} == {"dirty"}


def test_validator_with_pii_detection_flags_demo_records(tmp_path: Path) -> None:
    payload = _read_archive_entry(tmp_path, "support_messages.jsonl")
    report = validate_support_messages_jsonl(payload, detect_pii=True)
    # Demo archive injects PII-like emails and phones in 6 messages.
    assert report.pii_record_count >= 1
    assert report.redacted_record_count >= 1
    assert len(report.review_queue_object_ids) >= 1
    # Per-record findings carry digest + counts only, no raw text.
    for finding in report.pii_findings:
        for sub in finding.findings:
            assert sub.occurrence_count >= 1


def test_validator_persists_redacted_artifact_for_risky_records(
    tmp_path: Path,
) -> None:
    payload = _read_archive_entry(tmp_path, "support_messages.jsonl")
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_test",
            project_id="project_test",
            dataset_id="dataset_demo",
        ),
    )
    registry = ArtifactRegistry(storage=storage)
    request = TextOcrBuildRequest(
        dataset_id="dataset_demo",
        version_id="dataset_version_demo",
        parent_version_id="dataset_version_parent",
        created_by_job_id="compute_run_text_ocr",
        config_hash=_CONFIG_HASH,
    )

    report = validate_support_messages_jsonl(
        payload,
        detect_pii=True,
        registry=registry,
        build_request=request,
    )

    assert report.redacted_artifact_uri is not None
    assert report.redacted_artifact_hash is not None
    stored = storage.get(report.redacted_artifact_uri)
    redacted_text = stored.data.decode("utf-8")
    assert stored.info.metadata["artifact-kind"] == TEXT_OCR_REDACTED_ARTIFACT_KIND
    assert stored.info.metadata["schema-version"] == TEXT_OCR_REDACTED_SCHEMA_VERSION
    assert stored.info.metadata["source-kind"] == "support_messages"
    assert "alex@example.test" not in redacted_text
    assert "+10000001234" not in redacted_text
    assert "[REDACTED_EMAIL]" in redacted_text
    assert scan_log_text(redacted_text).passed
    output_lines = [json.loads(line) for line in redacted_text.splitlines() if line]
    assert len(output_lines) == report.redacted_record_count
    assert all("redacted_text" in line for line in output_lines)


def test_ocr_validator_pii_detection_passport_tokens(tmp_path: Path) -> None:
    payload = _read_archive_entry(tmp_path, "ocr_records.jsonl")
    report = validate_ocr_records_jsonl(payload, detect_pii=True)
    assert report.pii_record_count >= 1
    assert report.redacted_record_count >= 1


def test_produce_redacted_jsonl_only_pii_records() -> None:
    payload = _make_jsonl(
        [
            {"object_id": "a", "text": "contact " + _FAKE_EMAIL},
            {"object_id": "b", "text": "no contact info here"},
        ]
    )
    redacted_bytes, redacted_records = produce_redacted_jsonl(
        payload, detect_pii_only_records=True
    )
    output_text = redacted_bytes.decode("utf-8")
    assert _FAKE_EMAIL not in output_text
    assert "[REDACTED_EMAIL]" in output_text
    # Only the dirty record makes it into the redacted output.
    output_lines = [json.loads(line) for line in output_text.splitlines() if line.strip()]
    assert len(output_lines) == 1
    assert output_lines[0]["object_id"] == "a"
    # We still get findings for both records so callers can score privacy.
    assert {r.object_id for r in redacted_records} == {"a", "b"}


def test_redacted_jsonl_for_all_records_carries_no_raw_pii() -> None:
    payload = _make_jsonl(
        [
            {"object_id": "a", "text": "contact " + _FAKE_EMAIL},
            {"object_id": "b", "text": "no contact info here"},
        ]
    )
    redacted_bytes, _ = produce_redacted_jsonl(payload, detect_pii_only_records=False)
    text = redacted_bytes.decode("utf-8")
    assert _FAKE_EMAIL not in text
    assert "[REDACTED_EMAIL]" in text
    output_lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(output_lines) == 2


def test_redacted_text_is_safe_for_logs() -> None:
    """Verify redacted output passes the privacy log scanner from TASK-011."""
    record = redact_record(
        object_id="o1",
        text=(
            "email "
            + _FAKE_EMAIL
            + " phone "
            + _FAKE_PHONE
            + " passport "
            + _FAKE_PASSPORT
        ),
    )
    scan = scan_log_text(record.redacted_text)
    assert scan.passed, scan.violations


def test_findings_never_contain_raw_text() -> None:
    """Per-record findings must not echo raw tokens; only category/count/digest."""
    record = redact_record(
        object_id="o1",
        text="email " + _FAKE_EMAIL + " phone " + _FAKE_PHONE,
    )
    findings = record.to_record_findings()
    serialized = findings.model_dump_json()
    assert _FAKE_EMAIL not in serialized
    assert _FAKE_PHONE not in serialized


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
    raise AssertionError("archive entry " + repr(name) + " not found")


def _make_jsonl(records: list[dict[str, object]]) -> bytes:
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")


class _InMemoryS3Client(S3CompatibleClient):
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
            "LastModified": datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(record["Body"])}
                for (bucket, key), record in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
