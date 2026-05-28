"""Model-error analysis contracts.

These contracts describe the output of the model-error analyzer that
turns a validated :class:`PredictionManifest` + :class:`ManifestRow`
labels into:

* per-object uncertainty signals (confidence, margin, entropy,
  normalized_entropy);
* per-object ambiguous_object_score and probable_label_error_score;
* dataset-level confusion matrix + segment/class-wise error
  concentration;
* explicit ``not_applicable`` representation when no
  :class:`PredictionManifest` is supplied (so Decision Core never
  silently treats absent predictions as zero-score evidence).

Keeping these contracts in ``app/domain`` (not ``app/plugins``) means
both the predictions plugin and Decision Core can consume them without
crossing layer boundaries.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.common import NonEmptyStr, Score, Sha256Digest


class ModelErrorReportStatus(StrEnum):
    """Status of the model-error analyzer output.

    ``available``: a :class:`PredictionManifest` was supplied and the
    analyzer produced per-object scores plus dataset-level aggregates.
    ``not_applicable``: predictions were not supplied; downstream
    consumers see explicit ``reason`` field.
    """

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"


class ObjectModelErrorSignals(BaseModel):
    """Per-object prediction-derived signals.

    Keep ``ambiguous_object_score`` and ``probable_label_error_score``
    separate; they answer different questions and feed different
    review queues. Reason codes describe why the score is high (or low)
    for that object so Decision Core can render explainable decisions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    true_label: NonEmptyStr
    predicted_label: NonEmptyStr
    confidence: Score
    margin: Score
    entropy: float = Field(ge=0.0)
    normalized_entropy: Score
    label_conflict: bool
    ambiguous_object_score: Score
    probable_label_error_score: Score
    neighbor_label_support: Score | None = None
    cluster_label_support: Score | None = None
    label_support_status: NonEmptyStr = "not_applicable"
    reason_codes: tuple[NonEmptyStr, ...] = ()


class ConfusionMatrixEntry(BaseModel):
    """One cell of the confusion matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_label: NonEmptyStr
    predicted_label: NonEmptyStr
    count: int = Field(ge=0)


class ErrorConcentrationEntry(BaseModel):
    """Per-bucket error counts (for segment/class concentration)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket: NonEmptyStr
    total: int = Field(ge=0)
    errors: int = Field(ge=0)
    error_rate: Score


class ModelErrorAggregateMetrics(BaseModel):
    """Dataset-level aggregate metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accuracy: Score
    label_conflict_count: int = Field(ge=0)
    high_confidence_error_count: int = Field(ge=0)
    ambiguous_object_count: int = Field(ge=0)
    probable_label_error_count: int = Field(ge=0)


class ModelErrorThresholds(BaseModel):
    """Policy thresholds used to classify reasons.

    These values are part of the report so the report is self-describing
    and reproducible (different policy versions -> different reports).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_error_confidence: Score = 0.85
    label_error_margin: Score = 0.6
    ambiguous_max_confidence: Score = 0.6
    ambiguous_min_normalized_entropy: Score = 0.7
    ambiguous_max_margin: Score = 0.2
    label_support_threshold: Score = 0.6


class ModelErrorReport(BaseModel):
    """Top-level model-error analysis report.

    The report is contract-shaped so it can be referenced as a
    ``DataForgeReport.detail_artifacts`` ``ArtifactRef`` once persisted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "model_error_report.v1"
    status: ModelErrorReportStatus
    reason: str | None = None
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    model_id: str | None = None
    model_version: str | None = None
    classes: tuple[NonEmptyStr, ...] = ()
    confusion_matrix: tuple[ConfusionMatrixEntry, ...] = ()
    aggregate_metrics: ModelErrorAggregateMetrics | None = None
    segment_error_concentration: tuple[ErrorConcentrationEntry, ...] = ()
    class_error_concentration: tuple[ErrorConcentrationEntry, ...] = ()
    high_confidence_errors: tuple[NonEmptyStr, ...] = ()
    uncertain_objects: tuple[NonEmptyStr, ...] = ()
    object_signals: tuple[ObjectModelErrorSignals, ...] = ()
    thresholds: ModelErrorThresholds = ModelErrorThresholds()
    config_hash: Sha256Digest
    generated_at: datetime


__all__ = [
    "ConfusionMatrixEntry",
    "ErrorConcentrationEntry",
    "ModelErrorAggregateMetrics",
    "ModelErrorReport",
    "ModelErrorReportStatus",
    "ModelErrorThresholds",
    "ObjectModelErrorSignals",
]
