"""Privacy-safe structured logging helpers for compute-plane events."""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain import ErrorCode
from app.domain.common import NonEmptyStr

_REDACTED = "[REDACTED]"
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?:\+?\d[\s().-]*){10,}\d")
_PASSPORT_PATTERN = re.compile(r"\b(?:passport\s*)?\d{4}[\s-]?\d{6}\b", re.IGNORECASE)
_SECRET_PATTERN = re.compile(
    r"\b(?:token|secret|password|api[_-]?key)\s*[:=]\s*['\"]?[^'\"\s,}]+",
    re.IGNORECASE,
)
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(raw|text|content|email|phone|passport|token|secret|password|api[_-]?key|pii)",
    re.IGNORECASE,
)


class LogStatus(StrEnum):
    """Stable compute log status values."""

    STARTED = "STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StructuredLogEvent(BaseModel):
    """Required structured fields for privacy-safe compute logs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    plugin: NonEmptyStr
    job_type: NonEmptyStr
    stage: NonEmptyStr
    duration_ms: int = Field(ge=0)
    status: LogStatus
    error_code: ErrorCode | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def redacted_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["metadata"] = redact_for_logging(payload["metadata"])
        return payload


class LogScannerResult(BaseModel):
    """Result of scanning structured logs for obvious raw PII/secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    violations: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations


def emit_structured_log(
    logger: logging.Logger,
    event: StructuredLogEvent,
    *,
    level: int = logging.INFO,
) -> None:
    """Emit one JSON structured log line after deterministic privacy redaction."""
    logger.log(level, json.dumps(event.redacted_payload(), sort_keys=True, separators=(",", ":")))


def redact_for_logging(value: Any) -> Any:
    """Recursively redact obvious PII, raw text/content, and secrets before logging."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if _SENSITIVE_KEY_PATTERN.search(normalized_key):
                redacted[normalized_key] = _REDACTED
            else:
                redacted[normalized_key] = redact_for_logging(item)
        return redacted
    if isinstance(value, list):
        return [redact_for_logging(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_for_logging(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(value: str) -> str:
    """Mask PII-like tokens in otherwise safe diagnostic strings."""
    redacted = _SECRET_PATTERN.sub(_REDACTED, value)
    redacted = _EMAIL_PATTERN.sub(_REDACTED, redacted)
    redacted = _PHONE_PATTERN.sub(_REDACTED, redacted)
    return _PASSPORT_PATTERN.sub(_REDACTED, redacted)


def scan_log_text(log_text: str) -> LogScannerResult:
    """Scan log output for PII-like tokens that should never reach logs."""
    violations: list[str] = []
    if _EMAIL_PATTERN.search(log_text):
        violations.append("email")
    if _PHONE_PATTERN.search(log_text):
        violations.append("phone")
    if _PASSPORT_PATTERN.search(log_text):
        violations.append("passport")
    if _SECRET_PATTERN.search(log_text):
        violations.append("secret")
    return LogScannerResult(violations=tuple(violations))
