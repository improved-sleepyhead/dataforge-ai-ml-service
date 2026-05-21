"""Model-error analyzer producing a contract-shaped report.

The analyzer is deterministic and stdlib-only. It takes:

* a list of validated :class:`PredictionRow` records (already passed
  through ``app.ingestion.predictions.validate_predictions_jsonl``);
* a mapping ``object_id -> ManifestRowSummary`` with the label and the
  optional segment value used for segment-wise error concentration;
* :class:`ModelErrorThresholds` controlling reason codes.

It returns a :class:`ModelErrorReport` carrying:

* per-object uncertainty signals (confidence, margin, entropy,
  normalized_entropy);
* per-object ``ambiguous_object_score`` and ``probable_label_error_score``
  with explicit reason codes;
* dataset-level confusion matrix, accuracy, label_conflict_count,
  high-confidence-error count, ambiguous_object_count,
  probable_label_error_count;
* segment-wise and class-wise error concentration for explainable
  decisions.

Privacy/security notes:

* the analyzer never echoes raw row payload — only labels, technical
  IDs and aggregate counts;
* predictions remain evidence, not authority: outputs feed Decision
  Core but do not override labels or approve mutations/export.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from app.domain import (
    ConfusionMatrixEntry,
    ErrorConcentrationEntry,
    ModelErrorAggregateMetrics,
    ModelErrorReport,
    ModelErrorReportStatus,
    ModelErrorThresholds,
    ObjectModelErrorSignals,
    PredictionRow,
)

_REASON_AMBIGUOUS = "ambiguous_object"
_REASON_HIGH_MODEL_UNCERTAINTY = "high_model_uncertainty"
_REASON_LOW_PREDICTION_MARGIN = "low_prediction_margin"
_REASON_PROBABLE_LABEL_ERROR = "probable_label_error"
_REASON_HIGH_CONFIDENCE_LABEL_CONFLICT = "high_confidence_label_conflict"
_REASON_LARGE_MARGIN_LABEL_CONFLICT = "large_margin_label_conflict"
_REASON_LOW_ENTROPY_LABEL_CONFLICT = "low_entropy_label_conflict"
_REASON_LABEL_CONFLICT = "label_conflict"


@dataclass(frozen=True)
class ManifestRowSummary:
    """Manifest projection used by the analyzer.

    The analyzer needs only the bare minimum from a manifest row — the
    object id (the source key that predictions reference), an optional
    label override (``ManifestRow.label`` may differ from the prediction
    row ``true_label``; the analyzer always trusts the manifest label
    when it is set), and an optional segment string used for
    segment-wise concentration.
    """

    object_id: str
    label: str | None = None
    segment: str | None = None


_DEFAULT_THRESHOLDS = ModelErrorThresholds()


def compute_object_signals(
    row: PredictionRow,
    *,
    label_override: str | None = None,
    thresholds: ModelErrorThresholds | None = None,
) -> ObjectModelErrorSignals:
    """Compute uncertainty and label-error signals for one prediction row.

    The label used for conflict detection is ``label_override`` (the
    manifest label, treated as authoritative) when supplied, falling back
    to ``row.true_label``. Predictions never override the manifest label.
    """
    thresholds = thresholds if thresholds is not None else _DEFAULT_THRESHOLDS
    probabilities = sorted(row.predicted_proba.values(), reverse=True)
    confidence = probabilities[0]
    margin = (
        probabilities[0] - probabilities[1] if len(probabilities) >= 2 else 1.0
    )
    entropy = 0.0
    for probability in probabilities:
        if probability > 0:
            entropy -= probability * math.log(probability)
    log_k = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    normalized_entropy = entropy / log_k if log_k > 0 else 0.0

    true_label = label_override if label_override is not None else row.true_label
    label_conflict = row.predicted_label != true_label

    # Ambiguous object: low confidence, low margin, high normalized
    # entropy. Build a smooth score in [0, 1] from these three soft
    # indicators so Decision Core can rank objects.
    ambiguous_components = [
        max(0.0, 1.0 - confidence),
        max(0.0, 1.0 - margin),
        normalized_entropy,
    ]
    ambiguous_object_score = sum(ambiguous_components) / len(ambiguous_components)
    ambiguous_object_score = min(1.0, max(0.0, ambiguous_object_score))

    # Probable label error: increases when the predicted label disagrees
    # with the manifest label AND the model is confident (high
    # confidence, large margin, low normalized_entropy).
    if label_conflict:
        confidence_component = confidence
        margin_component = margin
        entropy_component = max(0.0, 1.0 - normalized_entropy)
        probable_label_error_score = (
            confidence_component + margin_component + entropy_component
        ) / 3.0
    else:
        probable_label_error_score = 0.0
    probable_label_error_score = min(
        1.0, max(0.0, probable_label_error_score)
    )

    reasons: list[str] = []
    if (
        confidence <= thresholds.ambiguous_max_confidence
        or margin <= thresholds.ambiguous_max_margin
        or normalized_entropy >= thresholds.ambiguous_min_normalized_entropy
    ):
        reasons.append(_REASON_AMBIGUOUS)
    if normalized_entropy >= thresholds.ambiguous_min_normalized_entropy:
        reasons.append(_REASON_HIGH_MODEL_UNCERTAINTY)
    if margin <= thresholds.ambiguous_max_margin:
        reasons.append(_REASON_LOW_PREDICTION_MARGIN)
    if label_conflict:
        reasons.append(_REASON_LABEL_CONFLICT)
        if confidence >= thresholds.label_error_confidence:
            reasons.append(_REASON_PROBABLE_LABEL_ERROR)
            reasons.append(_REASON_HIGH_CONFIDENCE_LABEL_CONFLICT)
        if margin >= thresholds.label_error_margin:
            reasons.append(_REASON_LARGE_MARGIN_LABEL_CONFLICT)
        if normalized_entropy <= 1.0 - thresholds.label_error_margin:
            reasons.append(_REASON_LOW_ENTROPY_LABEL_CONFLICT)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique_reasons: list[str] = []
    for code in reasons:
        if code not in seen:
            seen.add(code)
            unique_reasons.append(code)

    return ObjectModelErrorSignals(
        object_id=row.object_id,
        true_label=true_label,
        predicted_label=row.predicted_label,
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        label_conflict=label_conflict,
        ambiguous_object_score=ambiguous_object_score,
        probable_label_error_score=probable_label_error_score,
        reason_codes=tuple(unique_reasons),
    )


def analyze_model_errors(
    *,
    rows: Iterable[PredictionRow],
    manifest_index: Mapping[str, ManifestRowSummary] | None = None,
    dataset_id: str,
    version_id: str,
    config_hash: str,
    model_id: str | None = None,
    model_version: str | None = None,
    thresholds: ModelErrorThresholds | None = None,
    report_id: str | None = None,
    generated_at: datetime | None = None,
) -> ModelErrorReport:
    """Analyze validated prediction rows into a :class:`ModelErrorReport`.

    Predictions for object_ids not present in ``manifest_index`` are
    skipped (the join coverage is reported separately by
    ``app.ingestion.predictions.compute_prediction_coverage``).

    A manifest row whose ``label`` is ``None`` is also skipped — the
    analyzer cannot decide a label conflict without an authoritative
    label.
    """
    thresholds = thresholds if thresholds is not None else _DEFAULT_THRESHOLDS
    rows_list = list(rows)
    manifest = manifest_index or {}

    object_signals: list[ObjectModelErrorSignals] = []
    confusion: dict[tuple[str, str], int] = {}
    classes_seen: set[str] = set()
    segment_buckets: dict[str, _BucketStats] = {}
    class_buckets: dict[str, _BucketStats] = {}
    high_confidence_errors: list[str] = []
    uncertain_objects: list[str] = []
    ambiguous_count = 0
    probable_label_error_count = 0
    high_confidence_error_count = 0
    label_conflict_count = 0
    matched_count = 0
    correct_count = 0

    for row in rows_list:
        summary = manifest.get(row.object_id)
        if summary is None or summary.label is None:
            continue
        signals = compute_object_signals(
            row, label_override=summary.label, thresholds=thresholds
        )
        object_signals.append(signals)
        classes_seen.add(signals.true_label)
        classes_seen.add(signals.predicted_label)
        confusion_key = (signals.true_label, signals.predicted_label)
        confusion[confusion_key] = confusion.get(confusion_key, 0) + 1
        matched_count += 1
        if not signals.label_conflict:
            correct_count += 1
        else:
            label_conflict_count += 1
            if signals.confidence >= thresholds.label_error_confidence:
                high_confidence_error_count += 1
                high_confidence_errors.append(signals.object_id)
        if (
            signals.confidence <= thresholds.ambiguous_max_confidence
            or signals.normalized_entropy
            >= thresholds.ambiguous_min_normalized_entropy
            or signals.margin <= thresholds.ambiguous_max_margin
        ):
            ambiguous_count += 1
            uncertain_objects.append(signals.object_id)
        if signals.label_conflict and (
            signals.confidence >= thresholds.label_error_confidence
            or signals.margin >= thresholds.label_error_margin
        ):
            probable_label_error_count += 1

        # Segment/class concentration.
        if summary.segment:
            bucket = segment_buckets.setdefault(summary.segment, _BucketStats())
            bucket.total += 1
            if signals.label_conflict:
                bucket.errors += 1
        class_bucket = class_buckets.setdefault(signals.true_label, _BucketStats())
        class_bucket.total += 1
        if signals.label_conflict:
            class_bucket.errors += 1

    accuracy = correct_count / matched_count if matched_count else 0.0
    aggregate = ModelErrorAggregateMetrics(
        accuracy=accuracy,
        label_conflict_count=label_conflict_count,
        high_confidence_error_count=high_confidence_error_count,
        ambiguous_object_count=ambiguous_count,
        probable_label_error_count=probable_label_error_count,
    )

    confusion_entries = tuple(
        ConfusionMatrixEntry(true_label=t, predicted_label=p, count=count)
        for (t, p), count in sorted(confusion.items())
    )
    segment_entries = tuple(
        ErrorConcentrationEntry(
            bucket=bucket,
            total=stats.total,
            errors=stats.errors,
            error_rate=(stats.errors / stats.total) if stats.total else 0.0,
        )
        for bucket, stats in sorted(segment_buckets.items())
    )
    class_entries = tuple(
        ErrorConcentrationEntry(
            bucket=bucket,
            total=stats.total,
            errors=stats.errors,
            error_rate=(stats.errors / stats.total) if stats.total else 0.0,
        )
        for bucket, stats in sorted(class_buckets.items())
    )

    return ModelErrorReport(
        report_id=report_id or f"model_error_report_{uuid.uuid4().hex[:16]}",
        status=ModelErrorReportStatus.AVAILABLE,
        reason=None,
        dataset_id=dataset_id,
        version_id=version_id,
        model_id=model_id,
        model_version=model_version,
        classes=tuple(sorted(classes_seen)),
        confusion_matrix=confusion_entries,
        aggregate_metrics=aggregate,
        segment_error_concentration=segment_entries,
        class_error_concentration=class_entries,
        high_confidence_errors=tuple(sorted(set(high_confidence_errors))),
        uncertain_objects=tuple(sorted(set(uncertain_objects))),
        object_signals=tuple(object_signals),
        thresholds=thresholds,
        config_hash=config_hash,
        generated_at=generated_at or datetime.now(UTC),
    )


def build_not_applicable_report(
    *,
    dataset_id: str,
    version_id: str,
    config_hash: str,
    reason: str = "prediction_manifest_not_provided",
    report_id: str | None = None,
    generated_at: datetime | None = None,
) -> ModelErrorReport:
    """Return a NOT_APPLICABLE report carrying the explicit reason.

    Decision Core treats the absent prediction manifest as evidence
    missing, not as zero-error.
    """
    return ModelErrorReport(
        report_id=report_id or f"model_error_report_{uuid.uuid4().hex[:16]}",
        status=ModelErrorReportStatus.NOT_APPLICABLE,
        reason=reason,
        dataset_id=dataset_id,
        version_id=version_id,
        config_hash=config_hash,
        generated_at=generated_at or datetime.now(UTC),
    )


@dataclass
class _BucketStats:
    total: int = 0
    errors: int = 0


__all__ = [
    "ManifestRowSummary",
    "analyze_model_errors",
    "build_not_applicable_report",
    "compute_object_signals",
]
