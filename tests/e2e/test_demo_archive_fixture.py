"""Tests for the deterministic demo archive fixture (TASK-017).

Acceptance:

* tests/fixtures/demo_archive contains a generator that builds demo_archive.zip.
* transactions.csv contains is_fraud, customer_id_hash, rare class 1-3%,
  missing monthly_income, duplicates, outliers, leakage candidate.
* predictions.jsonl contains object_id, true_label, predicted_label,
  predicted_proba, confidence, split, model_id, model_version,
  inference_timestamp.
* Predictions fixture contains at least one ambiguous object and one probable
  label-error candidate.
* support_messages.jsonl and ocr_records.jsonl contain PII-like tokens and
  duplicates.
* Generation uses a fixed seed and emits expected_counts.json.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any, TypedDict

import pytest

from tests.fixtures.demo_archive import (
    DEFAULT_SEED,
    build_demo_archive,
    load_expected_counts,
)


class _DemoArchiveFixture(TypedDict):
    archive_path: Path
    expected_counts_path: Path
    expected_counts: dict[str, Any]
    archive_sha256: str
    files: dict[str, bytes]

_REQUIRED_TRANSACTION_COLUMNS = {
    "object_id",
    "customer_id_hash",
    "case_id",
    "amount",
    "monthly_income",
    "customer_segment",
    "manual_review_flag",
    "is_fraud",
}

_REQUIRED_PREDICTION_FIELDS = {
    "object_id",
    "true_label",
    "predicted_label",
    "predicted_proba",
    "confidence",
    "split",
    "model_id",
    "model_version",
    "inference_timestamp",
}

_FAKE_EMAIL_TOKENS = ("alex@example.test", "lena@example.test", "ivan@example.test")
_FAKE_PHONE_TOKENS = ("+10000001234", "+10000005678", "+10000009999")
_FAKE_PASSPORT_TOKENS = ("0001 100200", "0002 300400")


@pytest.fixture(scope="module")
def demo_archive(tmp_path_factory: pytest.TempPathFactory) -> _DemoArchiveFixture:
    output_dir = tmp_path_factory.mktemp("demo_archive_fixture")
    built = build_demo_archive(output_dir=output_dir)
    files = _read_archive_files(built.archive_path)
    return {
        "archive_path": built.archive_path,
        "expected_counts_path": built.expected_counts_path,
        "expected_counts": built.expected_counts,
        "archive_sha256": built.archive_sha256,
        "files": files,
    }


def test_demo_archive_is_deterministic_across_runs(tmp_path: Path) -> None:
    first = build_demo_archive(output_dir=tmp_path / "first")
    second = build_demo_archive(output_dir=tmp_path / "second")

    assert first.archive_sha256 == second.archive_sha256
    assert first.expected_counts == second.expected_counts


def test_demo_archive_contains_all_required_files(
    demo_archive: _DemoArchiveFixture,
) -> None:
    files = demo_archive["files"]
    assert isinstance(files, dict)
    expected = {
        "transactions.csv",
        "predictions.jsonl",
        "support_messages.jsonl",
        "ocr_records.jsonl",
        "README.md",
    }
    assert expected.issubset(files.keys())


def test_expected_counts_file_records_seed_and_archive_hash(
    demo_archive: _DemoArchiveFixture,
) -> None:
    counts_path = demo_archive["expected_counts_path"]
    assert isinstance(counts_path, Path)
    counts = load_expected_counts(counts_path)
    assert counts["seed"] == DEFAULT_SEED
    assert counts["archive_sha256"] == demo_archive["archive_sha256"]
    for filename in (
        "transactions.csv",
        "predictions.jsonl",
        "support_messages.jsonl",
        "ocr_records.jsonl",
        "README.md",
    ):
        assert counts["files"][filename]["sha256"].startswith("sha256:")


def test_transactions_csv_required_columns_and_rare_class_ratio(
    demo_archive: _DemoArchiveFixture,
) -> None:
    rows = _parse_csv_rows(demo_archive, "transactions.csv")
    assert len(rows) > 0

    columns = set(rows[0].keys())
    assert _REQUIRED_TRANSACTION_COLUMNS.issubset(columns)

    fraud_rows = [row for row in rows if row["is_fraud"] == "1"]
    rare_ratio = len(fraud_rows) / len(rows)
    assert 0.01 <= rare_ratio <= 0.03

    counts = demo_archive["expected_counts"]["transactions"]
    assert counts["rare_class_count"] == len(fraud_rows)
    assert counts["rare_class_ratio"] == pytest.approx(rare_ratio)


def test_transactions_csv_has_missing_income_in_segment_and_duplicates_and_outliers(
    demo_archive: _DemoArchiveFixture,
) -> None:
    rows = _parse_csv_rows(demo_archive, "transactions.csv")

    missing_in_segment = sum(
        1
        for row in rows
        if row["customer_segment"] == "young_customers" and row["monthly_income"] == ""
    )
    assert missing_in_segment > 0

    counts = demo_archive["expected_counts"]["transactions"]
    assert counts["missing_monthly_income"]["in_segment_count"] == missing_in_segment

    # Duplicate rows: at least one (object_id, customer_id_hash, amount, ...) tuple appears twice.
    keys = [
        (row["customer_id_hash"], row["amount"], row["customer_segment"], row["is_fraud"])
        for row in rows
    ]
    duplicates = len(keys) - len(set(keys))
    assert duplicates >= counts["duplicate_row_pairs"]

    high_amounts = [float(row["amount"]) for row in rows if float(row["amount"]) > 10_000.0]
    assert len(high_amounts) >= counts["outlier_rows"]


def test_transactions_csv_has_leakage_candidate_strongly_correlated_with_fraud(
    demo_archive: _DemoArchiveFixture,
) -> None:
    rows = _parse_csv_rows(demo_archive, "transactions.csv")
    counts = demo_archive["expected_counts"]["transactions"]
    assert counts["leakage_candidate_column"] == "manual_review_flag"

    leakage_match = sum(
        1
        for row in rows
        if row["manual_review_flag"] == "1" and row["is_fraud"] == "1"
    )
    leakage_match_without = sum(
        1
        for row in rows
        if row["manual_review_flag"] == "1" and row["is_fraud"] == "0"
    )
    assert leakage_match == counts["leakage_match_with_fraud"]
    # Leakage candidate must align with fraud strongly enough that a profiling
    # plugin can flag it; the exact ratio is checked from expected counts.
    assert leakage_match >= leakage_match_without


def test_predictions_jsonl_carries_required_fields_and_valid_probabilities(
    demo_archive: _DemoArchiveFixture,
) -> None:
    rows = _parse_jsonl(demo_archive, "predictions.jsonl")
    assert len(rows) > 0
    for row in rows:
        assert _REQUIRED_PREDICTION_FIELDS.issubset(row.keys())
        proba = row["predicted_proba"]
        assert isinstance(proba, dict) and proba
        total = sum(proba.values())
        assert abs(total - 1.0) < 1e-3

        top_label, top_value = max(proba.items(), key=lambda item: item[1])
        assert row["predicted_label"] == top_label
        assert abs(row["confidence"] - top_value) < 1e-3
        assert row["model_id"] == "fraud_baseline"
        assert row["model_version"] == "2026-05-14"
        assert row["split"] in {"train", "validation", "test", "holdout", "unknown"}


def test_predictions_jsonl_contains_ambiguous_and_probable_label_error(
    demo_archive: _DemoArchiveFixture,
) -> None:
    rows = _parse_jsonl(demo_archive, "predictions.jsonl")
    counts = demo_archive["expected_counts"]["predictions"]

    ambiguous = [
        row
        for row in rows
        if isinstance(row["predicted_proba"], dict)
        and abs(
            float(row["predicted_proba"]["fraud"])
            - float(row["predicted_proba"]["not_fraud"])
        )
        <= 0.10
    ]
    probable_label_error = [
        row
        for row in rows
        if row["true_label"] == "not_fraud"
        and row["predicted_label"] == "fraud"
        and isinstance(row["confidence"], (int, float))
        and float(row["confidence"]) >= 0.90
    ]

    assert len(ambiguous) >= 1
    assert len(probable_label_error) >= 1
    assert len(ambiguous) == counts["ambiguous_object_count"]
    assert len(probable_label_error) == counts["probable_label_error_count"]


def test_support_messages_contain_pii_tokens_and_exact_duplicate(
    demo_archive: _DemoArchiveFixture,
) -> None:
    rows = _parse_jsonl(demo_archive, "support_messages.jsonl")
    counts = demo_archive["expected_counts"]["support_messages"]

    pii_records = [
        row
        for row in rows
        if any(token in str(row["text"]) for token in _FAKE_EMAIL_TOKENS + _FAKE_PHONE_TOKENS)
    ]
    assert len(pii_records) >= 1
    assert len(pii_records) == counts["pii_like_records"]

    duplicate_groups = counts["exact_duplicate_groups"]
    assert any(group["count"] >= 2 for group in duplicate_groups)


def test_ocr_records_contain_pii_tokens_and_exact_duplicate(
    demo_archive: _DemoArchiveFixture,
) -> None:
    rows = _parse_jsonl(demo_archive, "ocr_records.jsonl")
    counts = demo_archive["expected_counts"]["ocr_records"]

    pii_records = [
        row
        for row in rows
        if any(token in str(row["text"]) for token in _FAKE_PASSPORT_TOKENS)
    ]
    assert len(pii_records) >= 1
    assert len(pii_records) == counts["pii_like_records"]

    duplicate_groups = counts["exact_duplicate_groups"]
    assert any(group["count"] >= 2 for group in duplicate_groups)


def _parse_csv_rows(demo_archive: _DemoArchiveFixture, filename: str) -> list[dict[str, str]]:
    payload = demo_archive["files"][filename]
    assert isinstance(payload, bytes)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    return list(reader)


def _parse_jsonl(demo_archive: _DemoArchiveFixture, filename: str) -> list[dict[str, object]]:
    payload = demo_archive["files"][filename]
    assert isinstance(payload, bytes)
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def _read_archive_files(archive_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive_path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}
