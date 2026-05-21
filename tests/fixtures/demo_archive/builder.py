"""Deterministic builder for ``demo_archive.zip`` and ``expected_counts.json``.

The builder generates a small fake-fraud tabular dataset, a matching
prediction manifest, and text/OCR proof records. It is fully deterministic:
given the same ``seed`` it produces byte-identical files. This allows tests
to assert on:

* total/rare/duplicate/leakage counts in transactions.csv;
* missing ``monthly_income`` segment behaviour;
* ambiguous object and probable-label-error counts in predictions.jsonl;
* PII-like tokens and exact duplicates in support_messages.jsonl and
  ocr_records.jsonl;
* file hashes, used to detect accidental drift.

Synthetic values only. No real customer data, real PII, or real secrets.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SEED = 20260520

# Tunables. Changing these requires regenerating expected_counts.json.
_TOTAL_TRANSACTIONS = 200
_RARE_FRAUD_RATIO = 0.02  # 2% rare class — within the 1-3% requirement.
_DUPLICATE_PAIRS = 4  # 4 duplicated transaction rows (8 affected rows).
_OUTLIER_COUNT = 3
_LEAKAGE_TRUE_FOR_FRAUD = 1.0  # leakage candidate matches fraud label completely.
_MISSING_INCOME_SEGMENT = "young_customers"
_MISSING_INCOME_RATIO_IN_SEGMENT = 0.55
_PROBABLE_LABEL_ERROR_COUNT = 3
_AMBIGUOUS_COUNT = 5
_DEMO_ARCHIVE_NAME = "demo_archive.zip"
_EXPECTED_COUNTS_NAME = "expected_counts.json"

_FAKE_PII_EMAILS = ("alex@example.test", "lena@example.test", "ivan@example.test")
_FAKE_PII_PHONES = ("+10000001234", "+10000005678", "+10000009999")
_FAKE_PII_PASSPORTS = ("0001 100200", "0002 300400")

_SUPPORT_DUPLICATE_BODY = (
    "Please reset my account password, contact me at alex@example.test"
)
_OCR_DUPLICATE_BODY = "Statement issued for client account 0001 100200, see attached"


@dataclass(frozen=True)
class DemoArchive:
    """Built demo archive paths and stable metadata."""

    archive_path: Path
    expected_counts_path: Path
    expected_counts: dict[str, Any]
    archive_sha256: str


def build_demo_archive(
    *,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
) -> DemoArchive:
    """Build ``demo_archive.zip`` and ``expected_counts.json`` deterministically.

    The zip file contains:

    * ``transactions.csv``
    * ``predictions.jsonl``
    * ``support_messages.jsonl``
    * ``ocr_records.jsonl``
    * ``README.md``

    Returns the on-disk paths plus the parsed ``expected_counts.json`` and
    the sha256 digest of the archive bytes for drift detection in tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    transactions, transactions_meta = _build_transactions(rng)
    predictions, predictions_meta = _build_predictions(rng, transactions)
    support_messages, support_meta = _build_support_messages(rng)
    ocr_records, ocr_meta = _build_ocr_records(rng)
    readme = _build_readme(seed=seed)

    files: dict[str, bytes] = {
        "transactions.csv": _to_csv_bytes(transactions),
        "predictions.jsonl": _to_jsonl_bytes(predictions),
        "support_messages.jsonl": _to_jsonl_bytes(support_messages),
        "ocr_records.jsonl": _to_jsonl_bytes(ocr_records),
        "README.md": readme.encode("utf-8"),
    }

    archive_bytes = _zip_bytes_deterministic(files)
    archive_path = output_dir / _DEMO_ARCHIVE_NAME
    archive_path.write_bytes(archive_bytes)

    expected_counts: dict[str, Any] = {
        "seed": seed,
        "archive_sha256": _sha256(archive_bytes),
        "files": {name: {"sha256": _sha256(data)} for name, data in files.items()},
        "transactions": transactions_meta,
        "predictions": predictions_meta,
        "support_messages": support_meta,
        "ocr_records": ocr_meta,
    }
    expected_counts_path = output_dir / _EXPECTED_COUNTS_NAME
    expected_counts_path.write_text(
        json.dumps(expected_counts, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return DemoArchive(
        archive_path=archive_path,
        expected_counts_path=expected_counts_path,
        expected_counts=expected_counts,
        archive_sha256=expected_counts["archive_sha256"],
    )


def load_expected_counts(path: Path) -> dict[str, Any]:
    """Load ``expected_counts.json`` from disk."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("expected_counts.json must contain a JSON object")
    return parsed


# ---------------------------------------------------------------------------
# transactions.csv
# ---------------------------------------------------------------------------


_TRANSACTION_COLUMNS = (
    "object_id",
    "customer_id_hash",
    "case_id",
    "amount",
    "monthly_income",
    "customer_segment",
    "manual_review_flag",
    "is_fraud",
)


def _build_transactions(rng: random.Random) -> tuple[list[dict[str, str]], dict[str, Any]]:
    fraud_count = max(round(_TOTAL_TRANSACTIONS * _RARE_FRAUD_RATIO), 1)
    fraud_indices = set(rng.sample(range(_TOTAL_TRANSACTIONS), fraud_count))
    outlier_indices = set(rng.sample(range(_TOTAL_TRANSACTIONS), _OUTLIER_COUNT))

    rows: list[dict[str, str]] = []
    missing_income_in_segment = 0
    missing_income_outside_segment = 0

    for idx in range(_TOTAL_TRANSACTIONS):
        is_fraud = idx in fraud_indices
        segment = "young_customers" if idx % 3 == 0 else "regular_customers"
        amount = round(rng.uniform(10.0, 500.0), 2)
        if idx in outlier_indices:
            amount = round(rng.uniform(50_000.0, 100_000.0), 2)

        income_value = round(rng.uniform(1500.0, 9000.0), 2)
        income_str = f"{income_value:.2f}"
        if segment == _MISSING_INCOME_SEGMENT and rng.random() < _MISSING_INCOME_RATIO_IN_SEGMENT:
            income_str = ""
            missing_income_in_segment += 1
        elif segment != _MISSING_INCOME_SEGMENT and rng.random() < 0.05:
            income_str = ""
            missing_income_outside_segment += 1

        manual_review_flag = (
            1 if (is_fraud and rng.random() < _LEAKAGE_TRUE_FOR_FRAUD) else 0
        )
        if not is_fraud and rng.random() < 0.005:
            manual_review_flag = 1

        rows.append(
            {
                "object_id": f"txn_{idx:05d}",
                "customer_id_hash": _stable_hash(f"customer:{idx % 47}"),
                "case_id": f"case_{idx % 31:04d}",
                "amount": f"{amount:.2f}",
                "monthly_income": income_str,
                "customer_segment": segment,
                "manual_review_flag": str(manual_review_flag),
                "is_fraud": "1" if is_fraud else "0",
            }
        )

    # Inject deterministic exact duplicates: copy the FIRST _DUPLICATE_PAIRS
    # rows to overwrite the LAST _DUPLICATE_PAIRS rows (excluding outliers
    # and fraud samples to keep counts stable).
    safe_source_indices = [
        i for i in range(20) if i not in fraud_indices and i not in outlier_indices
    ]
    safe_target_indices = [
        i
        for i in range(_TOTAL_TRANSACTIONS - 1, _TOTAL_TRANSACTIONS - 25, -1)
        if i not in fraud_indices and i not in outlier_indices
    ]
    duplicate_pairs: list[tuple[int, int]] = []
    for source_idx, target_idx in zip(safe_source_indices, safe_target_indices, strict=False):
        if len(duplicate_pairs) >= _DUPLICATE_PAIRS:
            break
        duplicated_row = dict(rows[source_idx])
        duplicated_row["object_id"] = rows[target_idx]["object_id"]
        rows[target_idx] = duplicated_row
        duplicate_pairs.append((source_idx, target_idx))

    fraud_total = sum(1 for row in rows if row["is_fraud"] == "1")
    leakage_match_count = sum(
        1
        for row in rows
        if row["manual_review_flag"] == "1" and row["is_fraud"] == "1"
    )
    leakage_false_match_count = sum(
        1
        for row in rows
        if row["manual_review_flag"] == "1" and row["is_fraud"] == "0"
    )

    meta: dict[str, Any] = {
        "total_rows": len(rows),
        "columns": list(_TRANSACTION_COLUMNS),
        "rare_class_label": "is_fraud=1",
        "rare_class_count": fraud_total,
        "rare_class_ratio": fraud_total / len(rows),
        "duplicate_row_pairs": len(duplicate_pairs),
        "outlier_rows": _OUTLIER_COUNT,
        "missing_monthly_income": {
            "segment": _MISSING_INCOME_SEGMENT,
            "in_segment_count": missing_income_in_segment,
            "outside_segment_count": missing_income_outside_segment,
        },
        "leakage_candidate_column": "manual_review_flag",
        "leakage_match_with_fraud": leakage_match_count,
        "leakage_match_without_fraud": leakage_false_match_count,
    }
    return rows, meta


# ---------------------------------------------------------------------------
# predictions.jsonl
# ---------------------------------------------------------------------------


def _build_predictions(
    rng: random.Random,
    transactions: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fraud_indices = [i for i, row in enumerate(transactions) if row["is_fraud"] == "1"]
    not_fraud_indices = [
        i for i, row in enumerate(transactions) if row["is_fraud"] == "0"
    ]

    rng.shuffle(fraud_indices)
    rng.shuffle(not_fraud_indices)

    probable_label_error_indices = set(
        not_fraud_indices[:_PROBABLE_LABEL_ERROR_COUNT]
    )
    ambiguous_indices_pool = [
        i
        for i in not_fraud_indices[_PROBABLE_LABEL_ERROR_COUNT:]
        if i not in probable_label_error_indices
    ]
    ambiguous_indices = set(ambiguous_indices_pool[:_AMBIGUOUS_COUNT])

    rows: list[dict[str, Any]] = []
    ambiguous_count = 0
    probable_error_count = 0
    high_confidence_correct = 0

    for idx, transaction in enumerate(transactions):
        true_label = "fraud" if transaction["is_fraud"] == "1" else "not_fraud"
        if idx in probable_label_error_indices:
            # High-confidence disagreement: model says fraud, label says not_fraud.
            fraud_proba = 0.95 + rng.uniform(0.0, 0.04)
            predicted_label = "fraud"
            probable_error_count += 1
        elif idx in ambiguous_indices:
            # Low margin around 0.5 — ambiguous object, low-confidence case.
            fraud_proba = 0.50 + rng.uniform(-0.03, 0.03)
            predicted_label = "fraud" if fraud_proba >= 0.5 else "not_fraud"
            ambiguous_count += 1
        elif idx in fraud_indices:
            # Confident fraud predictions for the rare class.
            fraud_proba = 0.85 + rng.uniform(0.0, 0.10)
            predicted_label = "fraud"
            high_confidence_correct += 1
        else:
            # Confident not-fraud predictions for majority class.
            fraud_proba = rng.uniform(0.02, 0.10)
            predicted_label = "not_fraud"
            high_confidence_correct += 1

        not_fraud_proba = 1.0 - fraud_proba
        if predicted_label == "fraud":
            confidence = fraud_proba
        else:
            confidence = not_fraud_proba

        split = "train" if idx % 5 != 0 else "validation"

        rows.append(
            {
                "object_id": transaction["object_id"],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_proba": {
                    "fraud": round(fraud_proba, 6),
                    "not_fraud": round(not_fraud_proba, 6),
                },
                "confidence": round(confidence, 6),
                "split": split,
                "model_id": "fraud_baseline",
                "model_version": "2026-05-14",
                "inference_timestamp": (
                    datetime(2026, 5, 14, 12, 0, tzinfo=UTC).isoformat()
                ),
            }
        )

    meta: dict[str, Any] = {
        "total_rows": len(rows),
        "ambiguous_object_count": ambiguous_count,
        "probable_label_error_count": probable_error_count,
        "high_confidence_count": high_confidence_correct,
        "splits": sorted({row["split"] for row in rows}),
        "model_id": "fraud_baseline",
        "model_version": "2026-05-14",
    }
    return rows, meta


# ---------------------------------------------------------------------------
# support_messages.jsonl and ocr_records.jsonl
# ---------------------------------------------------------------------------


def _build_support_messages(
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_messages: list[dict[str, Any]] = []
    for idx in range(15):
        if idx < 6:
            email = _FAKE_PII_EMAILS[idx % len(_FAKE_PII_EMAILS)]
            phone = _FAKE_PII_PHONES[idx % len(_FAKE_PII_PHONES)]
            text = (
                f"Hello team, please contact me at {email} or {phone}, "
                "I cannot log in to my account."
            )
        else:
            text = "Routine status update with no contact information."
        base_messages.append(
            {
                "object_id": f"support_{idx:04d}",
                "case_id": f"case_{idx % 9:04d}",
                "language": "en",
                "text": text,
            }
        )

    # Inject one exact duplicate of message #0 to demonstrate text duplicates.
    duplicate_target_index = 14
    base_messages[duplicate_target_index] = {
        **base_messages[0],
        "object_id": f"support_{duplicate_target_index:04d}",
    }

    pii_tokens = sum(
        1
        for record in base_messages
        if any(token in record["text"] for token in _FAKE_PII_EMAILS + _FAKE_PII_PHONES)
    )
    duplicate_count = sum(
        1
        for record in base_messages
        if record["text"] == base_messages[0]["text"]
    )

    support_text_hash = _sha256(base_messages[0]["text"].encode("utf-8"))
    meta: dict[str, Any] = {
        "total_rows": len(base_messages),
        "pii_like_records": pii_tokens,
        "exact_duplicate_groups": [
            {"text_sha256": support_text_hash, "count": duplicate_count}
        ],
    }
    _ = rng  # rng reserved for future stochastic variations
    return base_messages, meta


def _build_ocr_records(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_records: list[dict[str, Any]] = []
    for idx in range(10):
        if idx < 4:
            passport = _FAKE_PII_PASSPORTS[idx % len(_FAKE_PII_PASSPORTS)]
            text = (
                f"Statement issued for client account {passport}, see attached"
            )
        else:
            text = f"Account statement reference number {idx:06d}, no PII present"
        base_records.append(
            {
                "object_id": f"ocr_{idx:04d}",
                "document_id": f"doc_{idx % 5:04d}",
                "page": idx % 3 + 1,
                "language": "en",
                "text": text,
                "ocr_confidence": round(rng.uniform(0.80, 0.99), 4),
            }
        )

    duplicate_target_index = 9
    base_records[duplicate_target_index] = {
        **base_records[0],
        "object_id": f"ocr_{duplicate_target_index:04d}",
    }

    pii_tokens = sum(
        1
        for record in base_records
        if any(token in record["text"] for token in _FAKE_PII_PASSPORTS)
    )
    duplicate_count = sum(
        1
        for record in base_records
        if record["text"] == base_records[0]["text"]
    )

    ocr_text_hash = _sha256(base_records[0]["text"].encode("utf-8"))
    meta: dict[str, Any] = {
        "total_rows": len(base_records),
        "pii_like_records": pii_tokens,
        "exact_duplicate_groups": [
            {"text_sha256": ocr_text_hash, "count": duplicate_count}
        ],
    }
    return base_records, meta


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_readme(*, seed: int) -> str:
    return (
        "# DataForge AI demo archive\n\n"
        "Synthetic fixture for ML-service tests and local Dagster demo runs.\n"
        "All values are fake; no real customer data, real PII, or real secrets.\n\n"
        f"Generation seed: {seed}\n"
    )


def _to_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(_TRANSACTION_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _to_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode("utf-8")


def _zip_bytes_deterministic(files: dict[str, bytes]) -> bytes:
    """Build a zip with sorted entries and a fixed timestamp for byte-stable output."""
    buffer = io.BytesIO()
    fixed_dt = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(filename=name, date_time=fixed_dt)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "DEFAULT_SEED",
    "DemoArchive",
    "build_demo_archive",
    "load_expected_counts",
]
