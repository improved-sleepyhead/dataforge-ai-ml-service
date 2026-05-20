"""Object analytics and normalized evidence contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.common import NonEmptyStr, S3Uri, Score, Sha256Digest
from app.domain.manifest import DataModality


class SignalStatus(StrEnum):
    """Availability state for a normalized evidence signal."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


class EvidenceRef(BaseModel):
    """Reference to immutable evidence artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: NonEmptyStr
    uri: S3Uri


class NormalizedSignal(BaseModel):
    """Single normalized evidence signal with explicit missingness semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Score | None
    status: SignalStatus
    reason: str | None = None

    @model_validator(mode="after")
    def validate_signal_state(self) -> Self:
        if self.status is SignalStatus.AVAILABLE and self.value is None:
            raise ValueError("available signal must include a numeric value")
        if self.status is not SignalStatus.AVAILABLE and self.value is not None:
            raise ValueError("missing signal must use value=null")
        if self.status is not SignalStatus.AVAILABLE and not self.reason:
            raise ValueError("missing signal must include a reason")
        return self


class EvidenceSignals(BaseModel):
    """Normalized object-level evidence consumed by Decision Core."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    technical_quality: NormalizedSignal
    duplicate_score: NormalizedSignal
    privacy_risk: NormalizedSignal
    label_issue_score: NormalizedSignal
    rare_segment_score: NormalizedSignal
    business_importance: NormalizedSignal
    model_uncertainty: NormalizedSignal
    prediction_confidence: NormalizedSignal
    prediction_margin: NormalizedSignal
    prediction_entropy: NormalizedSignal
    ambiguous_object_score: NormalizedSignal
    probable_label_error_score: NormalizedSignal


class EvidenceBundle(BaseModel):
    """Normalized evidence bundle passed to deterministic Decision Core."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_bundle_id: NonEmptyStr
    object_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    modality: DataModality
    object_type: NonEmptyStr
    signals: EvidenceSignals
    confidence: dict[str, Score]
    evidence_refs: tuple[EvidenceRef, ...]
    computed_by_job_id: NonEmptyStr
    evidence_schema_version: NonEmptyStr


class ObjectIdentity(BaseModel):
    """Identity and lineage group inside an analytical passport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    modality: DataModality
    hash: Sha256Digest
    parent_object_id: str | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    metadata: dict[str, Any] = Field(default_factory=dict)


class TechnicalQualityBlock(BaseModel):
    """Technical quality metrics normalized for object analytics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: Score | None
    status: SignalStatus = SignalStatus.AVAILABLE
    metrics: dict[str, Any] = Field(default_factory=dict)


class PrivacyBlock(BaseModel):
    """Privacy risk and export eligibility signals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_score: Score | None
    status: SignalStatus = SignalStatus.AVAILABLE
    pii_detected: bool
    pii_types: tuple[str, ...] = ()
    export_eligibility: NonEmptyStr


class DuplicateSignals(BaseModel):
    """Duplicate and near-duplicate signals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duplicate_score: Score | None
    status: SignalStatus = SignalStatus.AVAILABLE
    duplicate_cluster_id: str | None = None


class LearningValueSignals(BaseModel):
    """Object usefulness and prediction-derived learning signals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rare_segment_score: Score | None
    diversity_score: Score | None
    model_uncertainty: Score | None
    label_issue_score: Score | None
    ambiguous_object_score: Score | None
    probable_label_error_score: Score | None
    decision_value_score: Score | None
    missing_reason: str | None = None


class PredictionBlock(BaseModel):
    """Prediction-derived analytics copied from validated prediction evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: NonEmptyStr
    model_version: NonEmptyStr
    true_label: NonEmptyStr
    predicted_label: NonEmptyStr
    confidence: Score
    margin: Score | None = None
    normalized_entropy: Score | None = None


class ObjectDecisionBlock(BaseModel):
    """Object-level recommendation produced after Decision Core execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: NonEmptyStr
    reason_codes: tuple[NonEmptyStr, ...]


class ObjectAnalyticalPassport(BaseModel):
    """Normalized per-object analytical passport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    analytics_schema_version: NonEmptyStr
    identity: ObjectIdentity
    technical_quality: TechnicalQualityBlock
    privacy: PrivacyBlock
    duplicate_signals: DuplicateSignals
    learning_value: LearningValueSignals
    prediction: PredictionBlock | None = None
    decision: ObjectDecisionBlock
    evidence_refs: tuple[EvidenceRef, ...]
    computed_at: datetime
