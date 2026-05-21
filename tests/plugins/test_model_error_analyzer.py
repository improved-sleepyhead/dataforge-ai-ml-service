"""Tests for TASK-025B: model-error analysis (ambiguous vs probable label error).

Acceptance criteria covered:

* analyzer consumes validated PredictionManifest + ManifestRow labels
  without treating predictions as label authority;
* per-object signals: confidence, margin, entropy, normalized_entropy;
* ambiguous_object_score grows with low confidence/low margin/high
  normalized_entropy;
* probable_label_error_score grows with predicted_label != true_label,
  high confidence, large margin, low normalized_entropy;
* separate reason codes for ambiguous vs probable label error;
* report carries confusion matrix, high-confidence errors, uncertain
  objects, segment-wise + class-wise error concentration;
* absent PredictionManifest → NOT_APPLICABLE with explicit reason
  ``prediction_manifest_not_provided``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.domain import (
    DataSplit,
    ModelErrorReportStatus,
    ModelErrorThresholds,
    PredictionRow,
)
from app.ingestion import (
    open_archive_path,
    validate_predictions_jsonl,
)
from app.plugins.predictions import (
    analyze_model_errors,
    build_not_applicable_report,
    compute_object_signals,
)
from app.plugins.predictions.analyzer import ManifestRowSummary
from tests.fixtures.demo_archive import build_demo_archive

_DATASET_ID = "dataset_demo"
_VERSION_ID = "version_demo"
_CONFIG_HASH = "sha256:" + "a" * 64
_MODEL_ID = "fraud_baseline"
_MODEL_VERSION = "2026-05-14"


# ---------------------------------------------------------------------------
# Step 1: run analyzer on demo predictions fixture
# ---------------------------------------------------------------------------


def test_analyzer_runs_on_demo_archive(tmp_path: Path) -> None:
    rows, manifest_index = _load_demo_inputs(tmp_path)
    report = analyze_model_errors(
        rows=rows,
        manifest_index=manifest_index,
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        config_hash=_CONFIG_HASH,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
    )

    assert report.status is ModelErrorReportStatus.AVAILABLE
    assert report.reason is None
    assert report.report_schema_version == "model_error_report.v1"
    assert report.aggregate_metrics is not None
    assert report.aggregate_metrics.label_conflict_count >= 0
    assert 0.0 <= report.aggregate_metrics.accuracy <= 1.0
    # Two known classes in the demo archive.
    assert set(report.classes) >= {"fraud", "not_fraud"}
    # Confusion matrix has at least one entry per observed (true, pred) pair.
    assert len(report.confusion_matrix) >= 1
    # At least one segment bucket (young_customers / regular_customers).
    assert len(report.segment_error_concentration) >= 1
    assert len(report.class_error_concentration) >= 1


# ---------------------------------------------------------------------------
# Step 2: ambiguous_object_score for low-margin fixture
# ---------------------------------------------------------------------------


def test_ambiguous_object_score_high_for_low_margin_row() -> None:
    """Two-class row with p=0.51/0.49 must rank as ambiguous."""
    row = _row("obj_amb", "fraud", "fraud", {"fraud": 0.51, "not_fraud": 0.49})
    signals = compute_object_signals(row, label_override="fraud")
    assert signals.label_conflict is False
    assert signals.confidence == pytest.approx(0.51)
    assert signals.margin == pytest.approx(0.02)
    assert signals.normalized_entropy > 0.95
    assert signals.ambiguous_object_score > 0.7
    assert signals.probable_label_error_score == 0.0
    assert "ambiguous_object" in signals.reason_codes
    assert "high_model_uncertainty" in signals.reason_codes
    assert "low_prediction_margin" in signals.reason_codes
    assert "probable_label_error" not in signals.reason_codes


# ---------------------------------------------------------------------------
# Step 3: probable_label_error_score for high-confidence label conflict
# ---------------------------------------------------------------------------


def test_probable_label_error_score_high_for_high_confidence_conflict() -> None:
    """Predicted label disagrees with manifest label at high confidence."""
    row = _row("obj_err", "dog", "cat", {"cat": 0.92, "dog": 0.05, "wolf": 0.03})
    signals = compute_object_signals(row, label_override="dog")
    assert signals.label_conflict is True
    assert signals.confidence == pytest.approx(0.92)
    assert signals.predicted_label == "cat"
    assert signals.true_label == "dog"
    assert signals.probable_label_error_score > 0.7
    assert signals.ambiguous_object_score < 0.4
    # Reason codes split clearly between ambiguous and probable label error.
    assert "probable_label_error" in signals.reason_codes
    assert "high_confidence_label_conflict" in signals.reason_codes
    assert "label_conflict" in signals.reason_codes


def test_low_confidence_label_conflict_does_not_raise_probable_label_error() -> None:
    """Conflict at low confidence is ambiguity, not a probable label error."""
    row = _row("obj_amb_err", "fraud", "not_fraud", {"fraud": 0.49, "not_fraud": 0.51})
    signals = compute_object_signals(row, label_override="fraud")
    assert signals.label_conflict is True
    assert signals.confidence == pytest.approx(0.51)
    # Below default label_error_confidence (0.85) and label_error_margin
    # (0.6) thresholds -> should not be flagged as probable label error.
    assert "high_confidence_label_conflict" not in signals.reason_codes
    assert "large_margin_label_conflict" not in signals.reason_codes
    assert "ambiguous_object" in signals.reason_codes


# ---------------------------------------------------------------------------
# Step 4: confusion matrix, segment-wise concentration and reason codes
# ---------------------------------------------------------------------------


def test_confusion_matrix_and_segment_concentration() -> None:
    rows = [
        _row("o1", "fraud", "fraud", {"fraud": 0.9, "not_fraud": 0.1}),
        _row("o2", "fraud", "not_fraud", {"fraud": 0.4, "not_fraud": 0.6}),
        _row("o3", "not_fraud", "fraud", {"fraud": 0.95, "not_fraud": 0.05}),
        _row("o4", "not_fraud", "not_fraud", {"fraud": 0.1, "not_fraud": 0.9}),
    ]
    manifest_index = {
        "o1": ManifestRowSummary(object_id="o1", label="fraud", segment="young"),
        "o2": ManifestRowSummary(object_id="o2", label="fraud", segment="young"),
        "o3": ManifestRowSummary(object_id="o3", label="not_fraud", segment="regular"),
        "o4": ManifestRowSummary(object_id="o4", label="not_fraud", segment="regular"),
    }
    report = analyze_model_errors(
        rows=rows,
        manifest_index=manifest_index,
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        config_hash=_CONFIG_HASH,
    )

    assert report.aggregate_metrics is not None
    assert report.aggregate_metrics.accuracy == pytest.approx(0.5)
    assert report.aggregate_metrics.label_conflict_count == 2
    # o3 has confidence=0.95 ≥ 0.85 → high-confidence error.
    assert report.aggregate_metrics.high_confidence_error_count == 1
    assert "o3" in report.high_confidence_errors

    # Confusion matrix sums to 4 entries (one per row).
    matrix_total = sum(entry.count for entry in report.confusion_matrix)
    assert matrix_total == 4

    segments = {entry.bucket: entry for entry in report.segment_error_concentration}
    assert segments["young"].errors == 1  # o2 was wrong
    assert segments["regular"].errors == 1  # o3 was wrong
    assert segments["young"].error_rate == pytest.approx(0.5)


def test_predictions_never_override_manifest_label() -> None:
    """Even if PredictionRow.true_label disagrees with manifest, manifest wins."""
    row = _row(
        "obj_clash", "wrong_label", "fraud",
        {"fraud": 0.9, "not_fraud": 0.1},
    )
    signals = compute_object_signals(row, label_override="not_fraud")
    # The analyzer must use the manifest label, not the prediction's true_label.
    assert signals.true_label == "not_fraud"
    assert signals.label_conflict is True
    assert signals.predicted_label == "fraud"


# ---------------------------------------------------------------------------
# Step 5: analyze without predictions → NOT_APPLICABLE
# ---------------------------------------------------------------------------


def test_no_predictions_produces_not_applicable_report() -> None:
    report = build_not_applicable_report(
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        config_hash=_CONFIG_HASH,
    )
    assert report.status is ModelErrorReportStatus.NOT_APPLICABLE
    assert report.reason == "prediction_manifest_not_provided"
    assert report.aggregate_metrics is None
    assert report.confusion_matrix == ()
    assert report.object_signals == ()


def test_explicit_reason_can_be_overridden() -> None:
    report = build_not_applicable_report(
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        config_hash=_CONFIG_HASH,
        reason="external_ai_disabled",
    )
    assert report.reason == "external_ai_disabled"


# ---------------------------------------------------------------------------
# Uncertainty math sanity checks
# ---------------------------------------------------------------------------


def test_uncertainty_math_matches_documented_formulas() -> None:
    """confidence/margin/entropy/normalized_entropy follow the docs."""
    proba = {"a": 0.5, "b": 0.3, "c": 0.2}
    row = _row("obj_math", "a", "a", proba)
    signals = compute_object_signals(row, label_override="a")
    expected_entropy = -sum(p * math.log(p) for p in proba.values())
    expected_normalized = expected_entropy / math.log(3)
    assert signals.confidence == pytest.approx(0.5)
    assert signals.margin == pytest.approx(0.2)
    assert signals.entropy == pytest.approx(expected_entropy)
    assert signals.normalized_entropy == pytest.approx(expected_normalized)


def test_thresholds_are_serialized_with_report() -> None:
    rows = [
        _row("o1", "fraud", "fraud", {"fraud": 0.9, "not_fraud": 0.1}),
    ]
    manifest_index = {
        "o1": ManifestRowSummary(object_id="o1", label="fraud", segment="x"),
    }
    custom = ModelErrorThresholds(
        label_error_confidence=0.95,
        label_error_margin=0.8,
    )
    report = analyze_model_errors(
        rows=rows,
        manifest_index=manifest_index,
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        config_hash=_CONFIG_HASH,
        thresholds=custom,
    )
    assert report.thresholds.label_error_confidence == 0.95
    assert report.thresholds.label_error_margin == 0.8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(
    object_id: str,
    true_label: str,
    predicted_label: str,
    proba: dict[str, float],
) -> PredictionRow:
    return PredictionRow(
        object_id=object_id,
        true_label=true_label,
        predicted_label=predicted_label,
        predicted_proba=proba,
        confidence=max(proba.values()),
        split=DataSplit.VALIDATION,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        inference_timestamp=datetime(2026, 5, 20, 7, 15, tzinfo=UTC),
    )


def _load_demo_inputs(
    tmp_path: Path,
) -> tuple[list[PredictionRow], dict[str, ManifestRowSummary]]:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        descriptors = [
            d for d in reader.descriptors() if d.kind.value == "predictions"
        ]
        with descriptors[0].open() as handle:
            payload = handle.read()
    _report, rows = validate_predictions_jsonl(payload)

    # Build manifest_index from transactions.csv: object_id (raw) → label,
    # segment.
    import csv
    import io
    import zipfile

    with zipfile.ZipFile(built.archive_path) as zf:
        with zf.open("transactions.csv") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as wrapper:
                tabular = list(csv.DictReader(wrapper))

    manifest_index = {
        record["object_id"]: ManifestRowSummary(
            object_id=record["object_id"],
            label="fraud" if record["is_fraud"] == "1" else "not_fraud",
            segment=record.get("customer_segment") or None,
        )
        for record in tabular
    }
    return rows, manifest_index
