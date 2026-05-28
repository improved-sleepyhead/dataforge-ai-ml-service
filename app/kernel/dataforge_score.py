"""Dataset-level DataForge Score v0 with decomposed policy inputs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.domain import (
    DataForgeScore,
    DataForgeScorePenalty,
    DatasetReadiness,
    DecisionReport,
    TabularProfileReport,
    TextOcrReport,
)

DATAFORGE_SCORE_POLICY_VERSION = "dataforge_score_v0"
DATAFORGE_SCORE_FORMULA = (
    "DataForgeScore = 100 * (w_c * Completeness + w_v * Validity "
    "+ w_u * Uniqueness + w_b * Balance + w_p * PrivacySafety "
    "+ w_m * ModelReadiness + w_l * LineageCompleteness) - Penalties"
)
DATAFORGE_SCORE_WEIGHTS: dict[str, float] = {
    "completeness": 0.15,
    "validity": 0.15,
    "uniqueness": 0.12,
    "balance": 0.15,
    "privacy_safety": 0.18,
    "model_readiness": 0.15,
    "lineage_completeness": 0.10,
}
DATAFORGE_SCORE_PENALTIES: dict[str, float] = {
    "unmasked_pii": -30.0,
    "split_leakage": -40.0,
    "target_leakage_candidate": -25.0,
    "severe_duplicates": -10.0,
    "severe_class_imbalance": -10.0,
}

_SEVERE_DUPLICATE_RATIO = 0.05
_SEVERE_CLASS_IMBALANCE_RATIO = 10.0
_SEVERE_MINORITY_SHARE = 0.05


class BuildDataForgeScoreRequest(BaseModel):
    """Inputs required to compute dataset-level DataForge Score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tabular_profile: TabularProfileReport | None = None
    text_ocr_report: TextOcrReport | None = None
    decision_report: DecisionReport | None = None
    split_leakage_detected: bool = False


def build_dataforge_score(request: BuildDataForgeScoreRequest) -> DataForgeScore:
    """Compute DataForgeScore from aggregate contract reports only."""
    components = {
        "completeness": _completeness(request.tabular_profile),
        "validity": _validity(request.tabular_profile, request.text_ocr_report),
        "uniqueness": _uniqueness(request.tabular_profile, request.text_ocr_report),
        "balance": _balance(request.tabular_profile),
        "privacy_safety": _privacy_safety(request.text_ocr_report, request.decision_report),
        "model_readiness": _model_readiness(
            request.tabular_profile,
            request.decision_report,
            split_leakage_detected=request.split_leakage_detected,
        ),
        "lineage_completeness": _lineage_completeness(request.tabular_profile),
    }
    weighted = {
        component: DATAFORGE_SCORE_WEIGHTS[component] * value
        for component, value in components.items()
    }
    penalties = _penalties(request)
    raw_score = _clamp_100(100.0 * sum(weighted.values()) + sum(p.value for p in penalties))
    hard_blocked = _hard_blocked(request.decision_report)
    readiness = (
        DatasetReadiness.BLOCKED
        if hard_blocked
        else DatasetReadiness.READY_WITH_WARNINGS
        if penalties
        else DatasetReadiness.READY_FOR_EXPORT
    )
    reason_codes = tuple(
        dict.fromkeys(
            [
                *(penalty.reason_code for penalty in penalties if penalty.applied),
                *_decision_reason_codes(request.decision_report),
            ]
        )
    )
    return DataForgeScore(
        value=round(raw_score / 100.0, 6),
        raw_score=round(raw_score, 6),
        policy_version=DATAFORGE_SCORE_POLICY_VERSION,
        formula=DATAFORGE_SCORE_FORMULA,
        weights=DATAFORGE_SCORE_WEIGHTS,
        components=components,
        weighted_components=weighted,
        penalties=penalties,
        hard_blocked=hard_blocked,
        readiness_status=readiness,
        reason_codes=reason_codes,
    )


def _completeness(profile: TabularProfileReport | None) -> float:
    if profile is None or profile.row_count == 0 or profile.column_count == 0:
        return 1.0
    if profile.missingness is not None and profile.missingness.columns:
        total_cells = sum(column.total_count for column in profile.missingness.columns)
        missing_cells = sum(column.missing_count for column in profile.missingness.columns)
        if total_cells > 0:
            return _clamp01(1.0 - (missing_cells / total_cells))
    total_cells = profile.row_count * profile.column_count
    missing_cells = sum(column.null_count for column in profile.columns)
    return _clamp01(1.0 - (missing_cells / total_cells)) if total_cells else 1.0


def _validity(
    profile: TabularProfileReport | None,
    text_ocr_report: TextOcrReport | None,
) -> float:
    scores: list[float] = []
    if profile is not None and profile.business_rules is not None and profile.business_rules.rules:
        scores.append(
            sum(rule.pass_rate for rule in profile.business_rules.rules)
            / len(profile.business_rules.rules)
        )
    if profile is not None and profile.missingness is not None:
        scores.append(0.0 if profile.missingness.target_column_missing else 1.0)
    if text_ocr_report is not None and text_ocr_report.total_record_count > 0:
        scores.append(text_ocr_report.total_valid_record_count / text_ocr_report.total_record_count)
    return _clamp01(sum(scores) / len(scores)) if scores else 1.0


def _uniqueness(
    profile: TabularProfileReport | None,
    text_ocr_report: TextOcrReport | None,
) -> float:
    duplicate_ratios: list[float] = []
    if profile is not None and profile.row_count > 0 and profile.duplicates is not None:
        duplicate_ratios.append(
            len(profile.duplicates.affected_object_ids) / profile.row_count
        )
    if text_ocr_report is not None and text_ocr_report.total_valid_record_count > 0:
        duplicate_ratios.append(
            text_ocr_report.total_duplicate_record_count
            / text_ocr_report.total_valid_record_count
        )
    duplicate_ratio = max(duplicate_ratios, default=0.0)
    return _clamp01(1.0 - duplicate_ratio)


def _balance(profile: TabularProfileReport | None) -> float:
    if profile is None or profile.class_imbalance is None:
        return 1.0
    return _clamp01(profile.class_imbalance.balance_score)


def _privacy_safety(
    text_ocr_report: TextOcrReport | None,
    decision_report: DecisionReport | None,
) -> float:
    if _has_blocker(decision_report, "PII_UNMASKED"):
        return 0.0
    if text_ocr_report is None or text_ocr_report.total_valid_record_count == 0:
        return 1.0
    unredacted_pii = max(
        0,
        text_ocr_report.total_pii_record_count - text_ocr_report.total_redacted_record_count,
    )
    return _clamp01(1.0 - (unredacted_pii / text_ocr_report.total_valid_record_count))


def _model_readiness(
    profile: TabularProfileReport | None,
    decision_report: DecisionReport | None,
    *,
    split_leakage_detected: bool,
) -> float:
    if split_leakage_detected or _has_blocker(decision_report, "SPLIT_LEAKAGE"):
        return 0.0
    if _has_blocker(decision_report, "TARGET_LEAKAGE_CANDIDATE"):
        return 0.0
    if profile is not None and profile.leakage is not None and profile.leakage.candidates:
        return 0.3
    if profile is not None and profile.missingness is not None:
        if profile.missingness.target_column_missing:
            return 0.0
    return 1.0


def _lineage_completeness(profile: TabularProfileReport | None) -> float:
    if profile is None:
        return 1.0
    lineage = profile.lineage
    checks = (
        bool(lineage.dataset_id),
        bool(lineage.version_id),
        bool(lineage.parent_version_id),
        bool(lineage.created_by_job_id),
        bool(lineage.config_hash),
        bool(lineage.source_artifact_id),
        bool(lineage.source_manifest_artifact.hash),
    )
    return sum(1 for check in checks if check) / len(checks)


def _penalties(request: BuildDataForgeScoreRequest) -> tuple[DataForgeScorePenalty, ...]:
    applied: list[DataForgeScorePenalty] = []
    for reason_code, triggered in {
        "unmasked_pii": _unmasked_pii(request),
        "split_leakage": request.split_leakage_detected
        or _has_blocker(request.decision_report, "SPLIT_LEAKAGE"),
        "target_leakage_candidate": _target_leakage(request),
        "severe_duplicates": _severe_duplicates(request),
        "severe_class_imbalance": _severe_class_imbalance(request.tabular_profile),
    }.items():
        if not triggered:
            continue
        applied.append(
            DataForgeScorePenalty(
                reason_code=reason_code,
                value=DATAFORGE_SCORE_PENALTIES[reason_code],
                applied=True,
            )
        )
    return tuple(applied)


def _unmasked_pii(request: BuildDataForgeScoreRequest) -> bool:
    if _has_blocker(request.decision_report, "PII_UNMASKED"):
        return True
    report = request.text_ocr_report
    if report is None:
        return False
    return report.total_pii_record_count > report.total_redacted_record_count


def _target_leakage(request: BuildDataForgeScoreRequest) -> bool:
    if _has_blocker(request.decision_report, "TARGET_LEAKAGE_CANDIDATE"):
        return True
    profile = request.tabular_profile
    return bool(profile is not None and profile.leakage is not None and profile.leakage.candidates)


def _severe_duplicates(request: BuildDataForgeScoreRequest) -> bool:
    profile = request.tabular_profile
    if profile is not None and profile.row_count > 0 and profile.duplicates is not None:
        duplicate_ratio = len(profile.duplicates.affected_object_ids) / profile.row_count
        if duplicate_ratio >= _SEVERE_DUPLICATE_RATIO:
            return True
    report = request.text_ocr_report
    if report is not None and report.total_valid_record_count > 0:
        return (
            report.total_duplicate_record_count / report.total_valid_record_count
            >= _SEVERE_DUPLICATE_RATIO
        )
    return False


def _severe_class_imbalance(profile: TabularProfileReport | None) -> bool:
    if profile is None or profile.class_imbalance is None:
        return False
    return (
        profile.class_imbalance.imbalance_ratio >= _SEVERE_CLASS_IMBALANCE_RATIO
        or profile.class_imbalance.minority_class_share <= _SEVERE_MINORITY_SHARE
    )


def _hard_blocked(decision_report: DecisionReport | None) -> bool:
    if decision_report is None:
        return False
    return (
        decision_report.readiness.status is DatasetReadiness.BLOCKED
        or len(decision_report.critical_blockers) > 0
    )


def _has_blocker(decision_report: DecisionReport | None, code: str) -> bool:
    if decision_report is None:
        return False
    return any(blocker.code == code for blocker in decision_report.critical_blockers)


def _decision_reason_codes(decision_report: DecisionReport | None) -> tuple[str, ...]:
    if decision_report is None:
        return ()
    return tuple(
        dict.fromkeys(
            [
                *(blocker.code.lower() for blocker in decision_report.critical_blockers),
                *(
                    code
                    for action in decision_report.recommended_actions
                    for code in action.reason_codes
                ),
            ]
        )
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, round(value, 6)))


def _clamp_100(value: float) -> float:
    return min(100.0, max(0.0, value))


__all__ = [
    "DATAFORGE_SCORE_FORMULA",
    "DATAFORGE_SCORE_PENALTIES",
    "DATAFORGE_SCORE_POLICY_VERSION",
    "DATAFORGE_SCORE_WEIGHTS",
    "BuildDataForgeScoreRequest",
    "build_dataforge_score",
]
