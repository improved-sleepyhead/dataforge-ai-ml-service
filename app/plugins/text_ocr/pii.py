"""PII detection and redaction for text/OCR records.

The detector is deterministic and stdlib-only. It scans for:

* email addresses;
* phone-like number runs;
* passport-like alphanumeric tokens;
* payment-card numbers (with Luhn validation to reduce false positives);
* bank-account-like long digit runs;
* common credential/secret tokens (``token=``, ``secret=`` etc.).

Any match is redacted in-place with a stable token (``[REDACTED_<CAT>]``)
and the redacted text is what gets persisted/exported. Findings record
only the category and occurrence count — never the raw matched text.

Detector patterns intentionally overlap with
``app.telemetry.logging`` so log scanning and content scanning share
the same threat model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from app.domain import (
    PiiCategory,
    PiiFinding,
    TextPiiFindingsForRecord,
)
from app.domain.common import Score

_REDACTED_TEMPLATE: Final[str] = "[REDACTED_{category}]"

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
# Phone numbers: at least 10 digits with optional separators / leading +.
# Anchored by digit boundaries so it does not eat all numbers.
_PHONE_PATTERN = re.compile(
    r"(?<![\d.])(?:\+?\d[\s().\-]*){9,}\d(?!\d)"
)
_PASSPORT_PATTERN = re.compile(
    r"\b(?:passport\s*[:#-]?\s*)?[A-Z]{0,2}\d{4}[\s\-]?\d{6}\b",
    re.IGNORECASE,
)
_PAYMENT_CARD_PATTERN = re.compile(
    r"\b(?:\d[ \-]?){12,18}\d\b"
)
_BANK_ACCOUNT_PATTERN = re.compile(
    r"\b(?:account|acct|iban)[:#\s\-]*([A-Z0-9]{8,34})\b",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    r"\b(?:token|secret|password|api[_\-]?key)\s*[:=]\s*['\"]?[^'\"\s,}]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _DetectorPattern:
    category: PiiCategory
    pattern: re.Pattern[str]
    require_luhn: bool = False


_DETECTORS: Sequence[_DetectorPattern] = (
    # Order matters: secrets first (so credentials never get re-classified
    # as phones), then deterministic identifiers, then heuristics.
    _DetectorPattern(category=PiiCategory.SECRET, pattern=_SECRET_PATTERN),
    _DetectorPattern(category=PiiCategory.EMAIL, pattern=_EMAIL_PATTERN),
    _DetectorPattern(category=PiiCategory.PASSPORT, pattern=_PASSPORT_PATTERN),
    _DetectorPattern(
        category=PiiCategory.PAYMENT_CARD,
        pattern=_PAYMENT_CARD_PATTERN,
        require_luhn=True,
    ),
    _DetectorPattern(category=PiiCategory.BANK_ACCOUNT, pattern=_BANK_ACCOUNT_PATTERN),
    _DetectorPattern(category=PiiCategory.PHONE, pattern=_PHONE_PATTERN),
)


@dataclass(frozen=True)
class RedactedRecord:
    """Result of redacting one text/OCR record.

    ``redacted_text`` carries no raw matched values — every match is
    replaced by ``[REDACTED_<CATEGORY>]``. ``findings`` records only
    category + occurrence count + the digest of the redacted text so
    Decision Core can link review-queue items by hash.
    """

    object_id: str
    redacted_text: str
    findings: tuple[PiiFinding, ...]
    pii_token_count: int
    pii_risk_score: Score
    redacted_text_sha256: str

    def to_record_findings(self) -> TextPiiFindingsForRecord:
        return TextPiiFindingsForRecord(
            object_id=self.object_id,
            findings=self.findings,
            pii_token_count=self.pii_token_count,
            pii_risk_score=self.pii_risk_score,
            redacted_text_sha256=self.redacted_text_sha256,
        )


def detect_pii(text: str) -> tuple[dict[PiiCategory, int], str]:
    """Detect PII categories in ``text`` and return the redacted version.

    Returns a tuple ``(category_counts, redacted_text)`` where
    ``category_counts`` maps category -> occurrence count and
    ``redacted_text`` has every matched span replaced with
    ``[REDACTED_<CATEGORY>]``.

    The function never echoes raw matches — callers should never log
    ``text``; only ``redacted_text`` is safe for logs/reports.
    """
    counts: dict[PiiCategory, int] = {}
    redacted = text
    for detector in _DETECTORS:
        replacement = _REDACTED_TEMPLATE.format(category=detector.category.value.upper())
        if detector.require_luhn:
            redacted, hits = _replace_with_luhn(redacted, detector.pattern, replacement)
        else:
            hits = len(detector.pattern.findall(redacted))
            if hits:
                redacted = detector.pattern.sub(replacement, redacted)
        if hits:
            counts[detector.category] = counts.get(detector.category, 0) + hits
    return counts, redacted


def redact_record(
    *,
    object_id: str,
    text: str,
) -> RedactedRecord:
    """Detect + redact PII in one record."""
    counts, redacted = detect_pii(text)
    findings = tuple(
        PiiFinding(category=category, occurrence_count=count)
        for category, count in sorted(counts.items())
    )
    pii_token_count = sum(counts.values())
    pii_risk_score = _compute_risk_score(counts, text_length=len(text))
    digest = hashlib.sha256(redacted.encode("utf-8")).hexdigest()
    return RedactedRecord(
        object_id=object_id,
        redacted_text=redacted,
        findings=findings,
        pii_token_count=pii_token_count,
        pii_risk_score=pii_risk_score,
        redacted_text_sha256=f"sha256:{digest}",
    )


def aggregate_pii(
    redacted_records: Iterable[RedactedRecord],
) -> tuple[tuple[TextPiiFindingsForRecord, ...], int, int]:
    """Aggregate redacted records into per-record findings + totals."""
    findings: list[TextPiiFindingsForRecord] = []
    pii_record_count = 0
    total_tokens = 0
    for record in redacted_records:
        if record.pii_token_count > 0:
            pii_record_count += 1
            total_tokens += record.pii_token_count
            findings.append(record.to_record_findings())
    return tuple(findings), pii_record_count, total_tokens


def _replace_with_luhn(
    text: str,
    pattern: re.Pattern[str],
    replacement: str,
) -> tuple[str, int]:
    """Replace pattern matches that pass the Luhn check.

    Reduces false positives for long digit runs that are not real
    payment cards.
    """
    hits = 0
    out_parts: list[str] = []
    last_end = 0
    for match in pattern.finditer(text):
        digits = re.sub(r"[\s\-]", "", match.group(0))
        if _luhn_valid(digits):
            out_parts.append(text[last_end : match.start()])
            out_parts.append(replacement)
            last_end = match.end()
            hits += 1
    out_parts.append(text[last_end:])
    return "".join(out_parts), hits


def _luhn_valid(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _compute_risk_score(
    counts: dict[PiiCategory, int],
    *,
    text_length: int,
) -> float:
    if not counts:
        return 0.0
    weights = {
        PiiCategory.PASSPORT: 1.0,
        PiiCategory.PAYMENT_CARD: 1.0,
        PiiCategory.BANK_ACCOUNT: 0.9,
        PiiCategory.SECRET: 1.0,
        PiiCategory.EMAIL: 0.5,
        PiiCategory.PHONE: 0.5,
    }
    weighted = sum(weights[category] * count for category, count in counts.items())
    # Normalize by text length so a single PII match in a 30-char
    # message scores higher than the same match in a long article.
    denominator = max(1.0, text_length / 100.0)
    score = weighted / denominator
    return min(1.0, score)


__all__ = [
    "RedactedRecord",
    "aggregate_pii",
    "detect_pii",
    "redact_record",
]
