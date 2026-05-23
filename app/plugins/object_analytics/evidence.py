"""Build normalized EvidenceBundle artifacts from validated passports."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    DataModality,
    EvidenceBundle,
    EvidenceRef,
    EvidenceSignals,
    NormalizedSignal,
    ObjectAnalyticalPassport,
    SignalStatus,
)
from app.domain.common import NonEmptyStr, Sha256Digest

EVIDENCE_BUNDLE_ARTIFACT_KIND = "evidence_bundle"
EVIDENCE_BUNDLE_ARTIFACT_FORMAT = "jsonl"
EVIDENCE_BUNDLE_MEDIA_TYPE = "application/jsonl"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "evidence_bundle.v1"

_PREDICTION_NOT_PROVIDED_REASON = "prediction_manifest_not_provided"


class BuildEvidenceBundleRequest(BaseModel):
    """Inputs required to build normalized EvidenceBundle rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest


@dataclass(frozen=True)
class BuildEvidenceBundleResult:
    """Evidence bundle stage output."""

    evidence_bundles: tuple[EvidenceBundle, ...]
    artifact: RegisteredArtifact


def build_evidence_bundles(
    *,
    passports: Iterable[ObjectAnalyticalPassport],
    request: BuildEvidenceBundleRequest,
    registry: ArtifactRegistry,
    source_passports_artifact: RegisteredArtifact | None = None,
) -> BuildEvidenceBundleResult:
    """Normalize validated analytical passports into Decision Core evidence.

    The builder consumes contract-shaped passports only. It does not read raw
    plugin reports, raw text, raw files, or prediction manifests directly.
    """
    passport_tuple = tuple(passports)
    artifact_ref = (
        EvidenceRef(
            kind=source_passports_artifact.artifact_kind,
            uri=source_passports_artifact.uri,
        )
        if source_passports_artifact is not None
        else None
    )
    bundles = tuple(
        _bundle_from_passport(
            passport=passport,
            request=request,
            source_passport_ref=artifact_ref,
        )
        for passport in passport_tuple
    )
    payload = _serialize_bundles(bundles)
    artifact = registry.save_artifact(
        artifact_kind=EVIDENCE_BUNDLE_ARTIFACT_KIND,
        data=payload,
        artifact_format=EVIDENCE_BUNDLE_ARTIFACT_FORMAT,
        media_type=EVIDENCE_BUNDLE_MEDIA_TYPE,
        schema_version=EVIDENCE_BUNDLE_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={"evidence_bundle_count": str(len(bundles))},
    )
    return BuildEvidenceBundleResult(evidence_bundles=bundles, artifact=artifact)


def _bundle_from_passport(
    *,
    passport: ObjectAnalyticalPassport,
    request: BuildEvidenceBundleRequest,
    source_passport_ref: EvidenceRef | None,
) -> EvidenceBundle:
    refs = _bundle_refs(passport=passport, source_passport_ref=source_passport_ref)
    signals = EvidenceSignals(
        technical_quality=_signal_from_value(
            passport.technical_quality.score,
            status=passport.technical_quality.status,
            missing_reason=_missing_reason(passport, "technical_quality"),
        ),
        duplicate_score=_signal_from_value(
            passport.duplicate_signals.duplicate_score,
            status=passport.duplicate_signals.status,
            missing_reason=_missing_reason(passport, "duplicate_score"),
        ),
        privacy_risk=_signal_from_value(
            passport.privacy.risk_score,
            status=passport.privacy.status,
            missing_reason=_missing_reason(passport, "privacy_risk"),
        ),
        label_issue_score=_prediction_signal_from_value(
            passport.learning_value.label_issue_score,
            passport=passport,
        ),
        rare_segment_score=_signal_from_value(
            passport.learning_value.rare_segment_score,
            status=_learning_status(passport),
            missing_reason=passport.learning_value.missing_reason
            or "rare_segment_not_available",
        ),
        business_importance=_signal_from_value(
            None,
            status=SignalStatus.NOT_APPLICABLE,
            missing_reason="business_importance_not_available",
        ),
        model_uncertainty=_prediction_signal_from_value(
            passport.learning_value.model_uncertainty,
            passport=passport,
        ),
        prediction_confidence=_prediction_signal_from_value(
            passport.prediction.confidence if passport.prediction else None,
            passport=passport,
        ),
        prediction_margin=_prediction_signal_from_value(
            passport.prediction.margin if passport.prediction else None,
            passport=passport,
        ),
        prediction_entropy=_prediction_signal_from_value(
            passport.prediction.normalized_entropy if passport.prediction else None,
            passport=passport,
        ),
        ambiguous_object_score=_prediction_signal_from_value(
            passport.learning_value.ambiguous_object_score,
            passport=passport,
        ),
        probable_label_error_score=_prediction_signal_from_value(
            passport.learning_value.probable_label_error_score,
            passport=passport,
        ),
    )
    return EvidenceBundle(
        evidence_bundle_id=f"evidence_{passport.object_id}",
        object_id=passport.object_id,
        dataset_id=passport.identity.dataset_id,
        version_id=passport.identity.version_id,
        modality=passport.identity.modality,
        object_type=_object_type(passport.identity.modality),
        signals=signals,
        confidence=_confidence(signals),
        evidence_refs=refs,
        computed_by_job_id=request.created_by_job_id,
        evidence_schema_version=EVIDENCE_BUNDLE_SCHEMA_VERSION,
    )


def _signal_from_value(
    value: float | None,
    *,
    status: SignalStatus,
    missing_reason: str,
) -> NormalizedSignal:
    if value is None:
        return NormalizedSignal(value=None, status=status, reason=missing_reason)
    return NormalizedSignal(value=value, status=SignalStatus.AVAILABLE, reason=None)


def _prediction_signal_from_value(
    value: float | None,
    *,
    passport: ObjectAnalyticalPassport,
) -> NormalizedSignal:
    if value is None:
        reason = (
            _PREDICTION_NOT_PROVIDED_REASON
            if passport.prediction is None
            else "prediction_signal_not_available"
        )
        return NormalizedSignal(
            value=None,
            status=SignalStatus.NOT_APPLICABLE,
            reason=reason,
        )
    return NormalizedSignal(value=value, status=SignalStatus.AVAILABLE, reason=None)


def _learning_status(passport: ObjectAnalyticalPassport) -> SignalStatus:
    if passport.learning_value.rare_segment_score is None:
        return SignalStatus.NOT_APPLICABLE
    return SignalStatus.AVAILABLE


def _missing_reason(passport: ObjectAnalyticalPassport, signal_name: str) -> str:
    if passport.learning_value.missing_reason == "skeleton_modality_not_profiled":
        return "skeleton_modality_not_profiled"
    return f"{signal_name}_not_available"


def _bundle_refs(
    *,
    passport: ObjectAnalyticalPassport,
    source_passport_ref: EvidenceRef | None,
) -> tuple[EvidenceRef, ...]:
    refs = list(passport.evidence_refs)
    if source_passport_ref is not None:
        refs.append(source_passport_ref)
    unique: dict[tuple[str, str], EvidenceRef] = {}
    for ref in refs:
        unique[(ref.kind, ref.uri)] = ref
    return tuple(unique[key] for key in sorted(unique))


def _object_type(modality: DataModality) -> str:
    if modality is DataModality.TABULAR:
        return "table_row"
    if modality is DataModality.TEXT:
        return "text_record"
    if modality is DataModality.DOCUMENT_OCR:
        return "ocr_record"
    if modality is DataModality.IMAGE:
        return "image_asset"
    if modality is DataModality.VIDEO:
        return "video_asset"
    if modality is DataModality.AUDIO:
        return "audio_asset"
    return "multimodal_case"


def _confidence(signals: EvidenceSignals) -> dict[str, float]:
    confidence: dict[str, float] = {}
    if signals.technical_quality.status is SignalStatus.AVAILABLE:
        confidence["technical_quality"] = 0.95
    if signals.duplicate_score.status is SignalStatus.AVAILABLE:
        confidence["duplicate_score"] = 0.9
    if signals.privacy_risk.status is SignalStatus.AVAILABLE:
        confidence["privacy_risk"] = 0.9
    prediction_signal_names = (
        "label_issue_score",
        "model_uncertainty",
        "prediction_confidence",
        "prediction_margin",
        "prediction_entropy",
        "ambiguous_object_score",
        "probable_label_error_score",
    )
    for name in prediction_signal_names:
        signal = getattr(signals, name)
        if signal.status is SignalStatus.AVAILABLE:
            confidence[name] = 0.82
    return confidence


def _serialize_bundles(bundles: tuple[EvidenceBundle, ...]) -> bytes:
    if not bundles:
        return b""
    lines = [
        json.dumps(
            bundle.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for bundle in bundles
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = [
    "BuildEvidenceBundleRequest",
    "BuildEvidenceBundleResult",
    "EVIDENCE_BUNDLE_ARTIFACT_FORMAT",
    "EVIDENCE_BUNDLE_ARTIFACT_KIND",
    "EVIDENCE_BUNDLE_MEDIA_TYPE",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "build_evidence_bundles",
]
