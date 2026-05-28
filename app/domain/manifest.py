"""Manifest and prediction contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.common import NonEmptyStr, S3Uri, Score, Sha256Digest


class DataModality(StrEnum):
    """Supported manifest modalities."""

    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT_OCR = "document_ocr"
    MULTIMODAL = "multimodal"


class DataSplit(StrEnum):
    """Dataset split identifiers used by manifests and predictions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HOLDOUT = "holdout"
    UNKNOWN = "unknown"


class ManifestLineage(BaseModel):
    """Lineage for one manifest object row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_artifact_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest


class ManifestRow(BaseModel):
    """One object in a versioned Asset Manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    modality: DataModality
    asset_uri: S3Uri
    hash: Sha256Digest
    metadata: dict[str, Any] = Field(default_factory=dict)
    lineage: ManifestLineage
    annotation_uri: S3Uri | None = None
    label: str | None = None
    split: DataSplit | None = None
    source_system: str | None = None
    embedding_refs: dict[str, str] = Field(default_factory=dict)


class PredictionArtifactRef(BaseModel):
    """Immutable prediction artifact pointer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: S3Uri
    hash: Sha256Digest


class PredictionRow(BaseModel):
    """Model prediction row joined to ManifestRow by object_id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    true_label: NonEmptyStr
    predicted_label: NonEmptyStr
    predicted_proba: dict[NonEmptyStr, Score]
    confidence: Score
    split: DataSplit
    model_id: NonEmptyStr
    model_version: NonEmptyStr
    inference_timestamp: datetime

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        if not self.predicted_proba:
            raise ValueError("predicted_proba must contain at least one class probability")

        probability_sum = sum(self.predicted_proba.values())
        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError("predicted_proba values must sum to 1.0")

        top_label, top_probability = max(self.predicted_proba.items(), key=lambda item: item[1])
        if self.predicted_label != top_label:
            raise ValueError("predicted_label must match argmax(predicted_proba)")
        if abs(self.confidence - top_probability) > 1e-6:
            raise ValueError("confidence must equal max(predicted_proba)")
        return self


class PredictionManifest(BaseModel):
    """Immutable prediction/probability manifest for model error analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_manifest_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    model_id: NonEmptyStr
    model_version: NonEmptyStr
    task_type: NonEmptyStr
    schema_version: NonEmptyStr
    artifact: PredictionArtifactRef
    rows: tuple[PredictionRow, ...]
