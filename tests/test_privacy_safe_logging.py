"""Privacy-safe structured logging tests."""

from __future__ import annotations

import io
import json
import logging

from app.domain import ErrorCode
from app.telemetry import LogStatus, StructuredLogEvent, emit_structured_log, scan_log_text


def test_pipeline_log_with_pii_like_text_is_redacted_and_scanner_passes() -> None:
    logger, stream = _memory_logger()

    emit_structured_log(
        logger,
        StructuredLogEvent(
            job_id="job_001",
            project_id="project_1",
            dataset_id="dataset_1",
            version_id="dataset_version_1",
            plugin="tabular_profiler",
            job_type="ANALYZE_ONLY",
            stage="profile.input_preview",
            duration_ms=42,
            status=LogStatus.FAILED,
            error_code=ErrorCode.INVALID_JOB_PAYLOAD,
            metadata={
                "row_count": 10,
                "raw_text": "Alice email alice@example.com phone +7 999 123-45-67",
                "diagnostic": "passport 1234 567890 token=super-secret-token",
                "nested": {"customer_email": "bob@example.com"},
            },
        ),
    )

    log_text = stream.getvalue()
    parsed = json.loads(log_text)

    assert scan_log_text(log_text).passed is True
    assert "alice@example.com" not in log_text
    assert "+7 999 123-45-67" not in log_text
    assert "1234 567890" not in log_text
    assert "super-secret-token" not in log_text
    assert parsed["job_id"] == "job_001"
    assert parsed["project_id"] == "project_1"
    assert parsed["dataset_id"] == "dataset_1"
    assert parsed["version_id"] == "dataset_version_1"
    assert parsed["plugin"] == "tabular_profiler"
    assert parsed["job_type"] == "ANALYZE_ONLY"
    assert parsed["stage"] == "profile.input_preview"
    assert parsed["duration_ms"] == 42
    assert parsed["status"] == "FAILED"
    assert parsed["error_code"] == "INVALID_JOB_PAYLOAD"
    assert parsed["metadata"]["raw_text"] == "[REDACTED]"
    assert parsed["metadata"]["nested"]["customer_email"] == "[REDACTED]"


def test_completed_log_contains_required_fields_and_null_error_code() -> None:
    logger, stream = _memory_logger()

    emit_structured_log(
        logger,
        StructuredLogEvent(
            job_id="job_002",
            project_id="project_1",
            dataset_id="dataset_1",
            version_id="dataset_version_1",
            plugin="decision_core",
            job_type="ANALYZE_ONLY",
            stage="decision.report",
            duration_ms=7,
            status=LogStatus.COMPLETED,
        ),
    )

    parsed = json.loads(stream.getvalue())

    assert parsed["job_id"] == "job_002"
    assert parsed["stage"] == "decision.report"
    assert "error_code" in parsed
    assert parsed["error_code"] is None
    assert scan_log_text(stream.getvalue()).violations == ()


def test_log_scanner_flags_raw_pii_tokens() -> None:
    result = scan_log_text(
        "job failed for analyst@example.com phone +1 415 555 0199 passport 1234 567890"
    )

    assert result.passed is False
    assert result.violations == ("email", "phone", "passport")


def _memory_logger() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.getLogger("tests.privacy_safe_logging")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, stream
