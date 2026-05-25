"""Build Object Analytical Passport artifacts from validated plugin outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    DataModality,
    DuplicateSignals,
    EvidenceRef,
    LearningValueSignals,
    ManifestRow,
    ModelErrorReport,
    ModelErrorReportStatus,
    ObjectAnalyticalPassport,
    ObjectDecisionBlock,
    ObjectIdentity,
    ObjectModelErrorSignals,
    PredictionBlock,
    PrivacyBlock,
    SignalStatus,
    TabularProfileReport,
    TechnicalQualityBlock,
    TextOcrReport,
    TextOcrSourceReport,
)
from app.domain.common import NonEmptyStr, Sha256Digest

OBJECT_ANALYTICS_ARTIFACT_KIND = "object_analytics_passports"
OBJECT_ANALYTICS_ARTIFACT_FORMAT = "jsonl"
OBJECT_ANALYTICS_MEDIA_TYPE = "application/jsonl"
OBJECT_ANALYTICS_SCHEMA_VERSION = "object_analytical_passport.v1"

_SKELETON_MODALITIES = frozenset(
    {
        DataModality.IMAGE,
        DataModality.VIDEO,
        DataModality.AUDIO,
        DataModality.MULTIMODAL,
    }
)


class BuildObjectAnalyticsRequest(BaseModel):
    """Inputs required to build per-object analytical passports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest


@dataclass(frozen=True)
class BuildObjectAnalyticsResult:
    """Object analytics stage output."""

    passports: tuple[ObjectAnalyticalPassport, ...]
    artifact: RegisteredArtifact


@dataclass(frozen=True)
class _TextSignals:
    duplicate_score: float | None = None
    duplicate_cluster_id: str | None = None
    pii_risk_score: float | None = None
    pii_types: tuple[str, ...] = ()
    pii_detected: bool = False
    has_validation_issue: bool = False


def build_object_analytics_passports(
    *,
    manifest_rows: Iterable[ManifestRow],
    request: BuildObjectAnalyticsRequest,
    registry: ArtifactRegistry,
    tabular_profile: TabularProfileReport | None = None,
    text_ocr_report: TextOcrReport | None = None,
    model_error_report: ModelErrorReport | None = None,
    evidence_refs: tuple[EvidenceRef, ...] = (),
    computed_at: datetime | None = None,
) -> BuildObjectAnalyticsResult:
    """Create one contract-compatible passport per manifest row and persist JSONL.

    Inputs are already-normalized contracts from ingestion/plugins. The builder
    never re-reads raw archive payloads and never stores raw PII in the passport.
    """
    rows = tuple(manifest_rows)
    timestamp = computed_at or datetime.now(UTC)
    model_signals = _model_signals_by_object(model_error_report)
    text_signals = _text_signals_by_manifest_object(rows, text_ocr_report)
    source_to_manifest = _source_object_id_to_manifest_id(rows)
    tabular_duplicate_ids = _tabular_duplicate_object_ids(
        tabular_profile,
        source_to_manifest=source_to_manifest,
    )
    tabular_outlier_columns = _tabular_outlier_columns_by_object(
        tabular_profile,
        source_to_manifest=source_to_manifest,
    )
    tabular_business_rules = _tabular_rule_violations_by_object(
        tabular_profile,
        source_to_manifest=source_to_manifest,
    )
    rare_label, rare_score = _tabular_rare_class_signal(tabular_profile)

    passports = tuple(
        _build_passport(
            row=row,
            request=request,
            evidence_refs=evidence_refs,
            computed_at=timestamp,
            tabular_profile=tabular_profile,
            text_signal=text_signals.get(row.object_id),
            model_signal=model_signals.get(row.object_id),
            model_report=model_error_report,
            tabular_duplicate_ids=tabular_duplicate_ids,
            tabular_outlier_columns=tabular_outlier_columns,
            tabular_business_rules=tabular_business_rules,
            rare_label=rare_label,
            rare_score=rare_score,
        )
        for row in rows
    )
    payload = _serialize_passports(passports)
    artifact = registry.save_artifact(
        artifact_kind=OBJECT_ANALYTICS_ARTIFACT_KIND,
        data=payload,
        artifact_format=OBJECT_ANALYTICS_ARTIFACT_FORMAT,
        media_type=OBJECT_ANALYTICS_MEDIA_TYPE,
        schema_version=OBJECT_ANALYTICS_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={"passport_count": str(len(passports))},
    )
    return BuildObjectAnalyticsResult(passports=passports, artifact=artifact)


def _build_passport(
    *,
    row: ManifestRow,
    request: BuildObjectAnalyticsRequest,
    evidence_refs: tuple[EvidenceRef, ...],
    computed_at: datetime,
    tabular_profile: TabularProfileReport | None,
    text_signal: _TextSignals | None,
    model_signal: ObjectModelErrorSignals | None,
    model_report: ModelErrorReport | None,
    tabular_duplicate_ids: set[str],
    tabular_outlier_columns: dict[str, tuple[str, ...]],
    tabular_business_rules: dict[str, tuple[str, ...]],
    rare_label: str | None,
    rare_score: float | None,
) -> ObjectAnalyticalPassport:
    skeleton = row.modality in _SKELETON_MODALITIES
    technical_quality = _technical_quality_block(
        row=row,
        skeleton=skeleton,
        tabular_profile=tabular_profile,
        text_signal=text_signal,
        outlier_columns=tabular_outlier_columns.get(row.object_id, ()),
        business_rule_ids=tabular_business_rules.get(row.object_id, ()),
    )
    privacy = _privacy_block(row=row, skeleton=skeleton, text_signal=text_signal)
    duplicate = _duplicate_block(
        row=row,
        skeleton=skeleton,
        text_signal=text_signal,
        tabular_duplicate_ids=tabular_duplicate_ids,
    )
    learning = _learning_value_block(
        row=row,
        skeleton=skeleton,
        model_signal=model_signal,
        rare_label=rare_label,
        rare_score=rare_score,
    )
    prediction = _prediction_block(model_signal=model_signal, model_report=model_report)

    action, reason_codes = _decision_stub(
        skeleton=skeleton,
        privacy=privacy,
        duplicate=duplicate,
        learning=learning,
        technical_quality=technical_quality,
    )
    return ObjectAnalyticalPassport(
        object_id=row.object_id,
        analytics_schema_version=OBJECT_ANALYTICS_SCHEMA_VERSION,
        identity=ObjectIdentity(
            object_id=row.object_id,
            dataset_id=row.dataset_id,
            version_id=row.version_id,
            modality=row.modality,
            hash=row.hash,
            parent_object_id=_optional_string(row.metadata.get("parent_object_id")),
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            metadata={
                "source_system": row.source_system,
                "source_object_id": row.metadata.get("source_object_id"),
                "validation_status": (
                    "validated_skeleton" if skeleton else "profiled"
                ),
            },
        ),
        technical_quality=technical_quality,
        privacy=privacy,
        duplicate_signals=duplicate,
        learning_value=learning,
        prediction=prediction,
        decision=ObjectDecisionBlock(action=action, reason_codes=reason_codes),
        evidence_refs=evidence_refs,
        computed_at=computed_at,
    )


def _technical_quality_block(
    *,
    row: ManifestRow,
    skeleton: bool,
    tabular_profile: TabularProfileReport | None,
    text_signal: _TextSignals | None,
    outlier_columns: tuple[str, ...],
    business_rule_ids: tuple[str, ...],
) -> TechnicalQualityBlock:
    if skeleton:
        return TechnicalQualityBlock(
            score=None,
            status=SignalStatus.NOT_APPLICABLE,
            metrics={"validation_status": "validated_skeleton"},
        )

    penalties = 0.0
    metrics: dict[str, object] = {"validation_status": "profiled"}
    if row.modality is DataModality.TABULAR:
        metrics["profile_schema_version"] = (
            tabular_profile.profile_schema_version if tabular_profile else None
        )
        metrics["outlier_columns"] = outlier_columns
        metrics["business_rule_violation_rule_ids"] = business_rule_ids
        if outlier_columns:
            penalties += 0.2
        if business_rule_ids:
            penalties += 0.3
    if row.modality in {DataModality.TEXT, DataModality.DOCUMENT_OCR}:
        has_issue = text_signal.has_validation_issue if text_signal else False
        metrics["text_validation_issue"] = has_issue
        if has_issue:
            penalties += 0.4
    return TechnicalQualityBlock(
        score=max(0.0, 1.0 - penalties),
        status=SignalStatus.AVAILABLE,
        metrics=metrics,
    )


def _privacy_block(
    *,
    row: ManifestRow,
    skeleton: bool,
    text_signal: _TextSignals | None,
) -> PrivacyBlock:
    if skeleton:
        return PrivacyBlock(
            risk_score=None,
            status=SignalStatus.NOT_APPLICABLE,
            pii_detected=False,
            pii_types=(),
            export_eligibility="policy_review_required",
        )
    if row.modality in {DataModality.TEXT, DataModality.DOCUMENT_OCR}:
        score = (
            text_signal.pii_risk_score
            if text_signal is not None and text_signal.pii_risk_score is not None
            else 0.0
        )
        pii_detected = text_signal.pii_detected if text_signal else False
        pii_types = text_signal.pii_types if text_signal else ()
        return PrivacyBlock(
            risk_score=score,
            status=SignalStatus.AVAILABLE,
            pii_detected=pii_detected,
            pii_types=pii_types,
            export_eligibility="redacted_only" if pii_detected else "allowed",
        )
    return PrivacyBlock(
        risk_score=0.0,
        status=SignalStatus.AVAILABLE,
        pii_detected=False,
        pii_types=(),
        export_eligibility="allowed",
    )


def _duplicate_block(
    *,
    row: ManifestRow,
    skeleton: bool,
    text_signal: _TextSignals | None,
    tabular_duplicate_ids: set[str],
) -> DuplicateSignals:
    if skeleton:
        return DuplicateSignals(
            duplicate_score=None,
            status=SignalStatus.NOT_APPLICABLE,
            duplicate_cluster_id=None,
        )
    if row.modality is DataModality.TABULAR:
        is_duplicate = row.object_id in tabular_duplicate_ids
        return DuplicateSignals(
            duplicate_score=1.0 if is_duplicate else 0.0,
            status=SignalStatus.AVAILABLE,
            duplicate_cluster_id="tabular_exact_duplicate" if is_duplicate else None,
        )
    duplicate_score = (
        text_signal.duplicate_score
        if text_signal is not None and text_signal.duplicate_score is not None
        else 0.0
    )
    return DuplicateSignals(
        duplicate_score=duplicate_score,
        status=SignalStatus.AVAILABLE,
        duplicate_cluster_id=text_signal.duplicate_cluster_id if text_signal else None,
    )


def _learning_value_block(
    *,
    row: ManifestRow,
    skeleton: bool,
    model_signal: ObjectModelErrorSignals | None,
    rare_label: str | None,
    rare_score: float | None,
) -> LearningValueSignals:
    if skeleton:
        return LearningValueSignals(
            rare_segment_score=None,
            diversity_score=None,
            model_uncertainty=None,
            label_issue_score=None,
            ambiguous_object_score=None,
            probable_label_error_score=None,
            decision_value_score=None,
            missing_reason="skeleton_modality_not_profiled",
        )
    rare_segment_score = (
        rare_score
        if row.label is not None and row.label == _canonical_profile_label(rare_label)
        else 0.0
    )
    model_uncertainty = (
        1.0 - model_signal.confidence if model_signal is not None else None
    )
    label_issue = (
        model_signal.probable_label_error_score if model_signal is not None else None
    )
    ambiguous = (
        model_signal.ambiguous_object_score if model_signal is not None else None
    )
    probable = (
        model_signal.probable_label_error_score if model_signal is not None else None
    )
    decision_value = _decision_value_score(
        rare_segment_score=rare_segment_score,
        ambiguous_object_score=ambiguous,
        probable_label_error_score=probable,
    )
    return LearningValueSignals(
        rare_segment_score=rare_segment_score,
        diversity_score=None,
        model_uncertainty=model_uncertainty,
        label_issue_score=label_issue,
        ambiguous_object_score=ambiguous,
        probable_label_error_score=probable,
        decision_value_score=decision_value,
        missing_reason="learning_value_v0_stub",
    )


def _prediction_block(
    *,
    model_signal: ObjectModelErrorSignals | None,
    model_report: ModelErrorReport | None,
) -> PredictionBlock | None:
    if model_signal is None:
        return None
    model_id = "unknown_model"
    model_version = "unknown"
    if model_report is not None:
        model_id = model_report.model_id or model_id
        model_version = model_report.model_version or model_version
    return PredictionBlock(
        model_id=model_id,
        model_version=model_version,
        true_label=model_signal.true_label,
        predicted_label=model_signal.predicted_label,
        confidence=model_signal.confidence,
        margin=model_signal.margin,
        normalized_entropy=model_signal.normalized_entropy,
    )


def _decision_stub(
    *,
    skeleton: bool,
    privacy: PrivacyBlock,
    duplicate: DuplicateSignals,
    learning: LearningValueSignals,
    technical_quality: TechnicalQualityBlock,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if skeleton:
        return "PENDING_DECISION_CORE", ("validated_skeleton_only",)
    if privacy.pii_detected:
        reasons.append("pii_detected")
    if duplicate.duplicate_score and duplicate.duplicate_score > 0:
        reasons.append("duplicate_candidate")
    if learning.probable_label_error_score and learning.probable_label_error_score > 0:
        reasons.append("probable_label_error")
    if learning.ambiguous_object_score and learning.ambiguous_object_score > 0:
        reasons.append("ambiguous_object")
    if technical_quality.score is not None and technical_quality.score < 1.0:
        reasons.append("technical_quality_issue")
    if not reasons:
        reasons.append("profiled_no_blocker_candidate")
    return "PENDING_DECISION_CORE", tuple(sorted(set(reasons)))


def _model_signals_by_object(
    report: ModelErrorReport | None,
) -> dict[str, ObjectModelErrorSignals]:
    if report is None or report.status is not ModelErrorReportStatus.AVAILABLE:
        return {}
    return {signals.object_id: signals for signals in report.object_signals}


def _text_signals_by_manifest_object(
    manifest_rows: tuple[ManifestRow, ...],
    report: TextOcrReport | None,
) -> dict[str, _TextSignals]:
    if report is None:
        return {}
    source_to_manifest = _source_object_id_to_manifest_id(manifest_rows)
    by_manifest: dict[str, _TextSignals] = {}
    for source in report.sources:
        _merge_text_source_signals(
            by_manifest=by_manifest,
            source_to_manifest=source_to_manifest,
            source=source,
        )
    return by_manifest


def _merge_text_source_signals(
    *,
    by_manifest: dict[str, _TextSignals],
    source_to_manifest: dict[str, str],
    source: TextOcrSourceReport,
) -> None:
    issue_ids = {
        issue.object_id
        for issue in source.issues
        if issue.object_id is not None
    }
    for group in source.duplicate_groups:
        cluster_id = group.text_sha256
        for source_object_id in group.object_ids:
            manifest_id = source_to_manifest.get(source_object_id)
            if manifest_id is None:
                continue
            existing = by_manifest.get(manifest_id, _TextSignals())
            by_manifest[manifest_id] = _TextSignals(
                duplicate_score=1.0,
                duplicate_cluster_id=cluster_id,
                pii_risk_score=existing.pii_risk_score,
                pii_types=existing.pii_types,
                pii_detected=existing.pii_detected,
                has_validation_issue=existing.has_validation_issue,
            )
    for finding in source.pii_findings:
        manifest_id = source_to_manifest.get(finding.object_id)
        if manifest_id is None:
            continue
        existing = by_manifest.get(manifest_id, _TextSignals())
        by_manifest[manifest_id] = _TextSignals(
            duplicate_score=existing.duplicate_score,
            duplicate_cluster_id=existing.duplicate_cluster_id,
            pii_risk_score=max(
                existing.pii_risk_score or 0.0,
                finding.pii_risk_score,
            ),
            pii_types=tuple(
                sorted(
                    set(existing.pii_types)
                    | {item.category.value for item in finding.findings}
                )
            ),
            pii_detected=True,
            has_validation_issue=existing.has_validation_issue,
        )
    for source_object_id in issue_ids:
        manifest_id = source_to_manifest.get(source_object_id)
        if manifest_id is None:
            continue
        existing = by_manifest.get(manifest_id, _TextSignals())
        by_manifest[manifest_id] = _TextSignals(
            duplicate_score=existing.duplicate_score,
            duplicate_cluster_id=existing.duplicate_cluster_id,
            pii_risk_score=existing.pii_risk_score,
            pii_types=existing.pii_types,
            pii_detected=existing.pii_detected,
            has_validation_issue=True,
        )


def _source_object_id_to_manifest_id(
    manifest_rows: tuple[ManifestRow, ...],
) -> dict[str, str]:
    return {
        str(row.metadata["source_object_id"]): row.object_id
        for row in manifest_rows
        if row.metadata.get("source_object_id") is not None
    }


def _tabular_duplicate_object_ids(
    report: TabularProfileReport | None,
    *,
    source_to_manifest: dict[str, str],
) -> set[str]:
    if report is None or report.duplicates is None:
        return set()
    return {
        source_to_manifest[source_object_id]
        for source_object_id in report.duplicates.affected_object_ids
        if source_object_id in source_to_manifest
    }


def _tabular_outlier_columns_by_object(
    report: TabularProfileReport | None,
    *,
    source_to_manifest: dict[str, str],
) -> dict[str, tuple[str, ...]]:
    if report is None or report.outliers is None:
        return {}
    by_object: dict[str, list[str]] = {}
    for column in report.outliers.columns:
        for source_object_id in column.affected_object_ids:
            object_id = source_to_manifest.get(source_object_id)
            if object_id is None:
                continue
            by_object.setdefault(object_id, []).append(column.column)
    return {
        object_id: tuple(sorted(columns))
        for object_id, columns in by_object.items()
    }


def _tabular_rule_violations_by_object(
    report: TabularProfileReport | None,
    *,
    source_to_manifest: dict[str, str],
) -> dict[str, tuple[str, ...]]:
    if report is None or report.business_rules is None:
        return {}
    by_object: dict[str, list[str]] = {}
    for violation in report.business_rules.sample_violations:
        if violation.object_id is None:
            continue
        object_id = source_to_manifest.get(violation.object_id)
        if object_id is None:
            continue
        by_object.setdefault(object_id, []).append(violation.rule_id)
    return {
        object_id: tuple(sorted(rule_ids))
        for object_id, rule_ids in by_object.items()
    }


def _tabular_rare_class_signal(
    report: TabularProfileReport | None,
) -> tuple[str | None, float | None]:
    if report is None or report.class_imbalance is None:
        return None, None
    rare_share = report.class_imbalance.rare_class_ratio
    return report.class_imbalance.rare_class_label, max(0.0, min(1.0, 1.0 - rare_share))


def _canonical_profile_label(label: str | None) -> str | None:
    if label == "1":
        return "fraud"
    if label == "0":
        return "not_fraud"
    return label


def _decision_value_score(
    *,
    rare_segment_score: float | None,
    ambiguous_object_score: float | None,
    probable_label_error_score: float | None,
) -> float | None:
    values = [
        value
        for value in (
            rare_segment_score,
            ambiguous_object_score,
            probable_label_error_score,
        )
        if value is not None
    ]
    if not values:
        return None
    return max(values)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _serialize_passports(passports: tuple[ObjectAnalyticalPassport, ...]) -> bytes:
    if not passports:
        return b""
    lines = [
        json.dumps(
            passport.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for passport in passports
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = [
    "BuildObjectAnalyticsRequest",
    "BuildObjectAnalyticsResult",
    "OBJECT_ANALYTICS_ARTIFACT_FORMAT",
    "OBJECT_ANALYTICS_ARTIFACT_KIND",
    "OBJECT_ANALYTICS_MEDIA_TYPE",
    "OBJECT_ANALYTICS_SCHEMA_VERSION",
    "build_object_analytics_passports",
]
