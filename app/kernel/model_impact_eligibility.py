"""Eligibility checker for model-impact evaluation on tabular candidates.

The eligibility check decides whether a candidate dataset version is
ready for a baseline model-impact run. PRD §21.1.1 requires:

- a target label is present;
- enough labeled samples are available for a deterministic baseline;
- a valid split manifest exists;
- no leakage blockers are active;
- a supported baseline evaluator exists for the task type.

When the candidate is not eligible, the builder still produces a
contract-valid eligibility artifact carrying explicit reason codes and
a reference to the fallback readiness report (Dataset Readiness /
Quality Improvement / Privacy Risk Reduction / Duplicate Reduction).

Privacy rules:

- the report carries only column names, counts and ratios;
- raw rows, raw values, raw PII never enter the artifact;
- when minorities are too rare we record the count, never the
  individual object ids.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    DEFAULT_MIN_LABELED_SAMPLES,
    DEFAULT_MIN_RARE_CLASS_SAMPLES,
    DEFAULT_MIN_SPLITS,
    MODEL_IMPACT_ELIGIBILITY_SCHEMA_VERSION,
    ArtifactRef,
    ClassImbalanceDiagnostics,
    ErrorCode,
    FallbackReadinessReportRef,
    ModelImpactCohortStats,
    ModelImpactEligibilityLineage,
    ModelImpactEligibilityReport,
    ModelImpactEligibilityStatus,
    ModelImpactInputName,
    ModelImpactNotEligibleReasonCode,
    ModelImpactSplitStats,
    ModelImpactTaskType,
    SplitLeakageReport,
    SplitManifest,
    TabularProfileReport,
)
from app.domain.common import NonEmptyStr, Sha256Digest

MODEL_IMPACT_ELIGIBILITY_KIND = "model_impact_eligibility_report"
MODEL_IMPACT_ELIGIBILITY_FORMAT = "json"
MODEL_IMPACT_ELIGIBILITY_MEDIA_TYPE = "application/json"

_SUPPORTED_TASK_TYPES: frozenset[ModelImpactTaskType] = frozenset(
    {ModelImpactTaskType.SUPERVISED_TABULAR_CLASSIFICATION}
)
"""MVP only ships a baseline classifier; regression eligibility lives behind a future task."""

_PRIMARY_REASON_ELIGIBLE = "supervised_classification_with_target"


class ModelImpactEligibilityError(ValueError):
    """Raised when the eligibility checker cannot run safely."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.MODEL_IMPACT_NOT_ELIGIBLE,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class CheckModelImpactEligibilityRequest(BaseModel):
    """Inputs for :func:`check_model_impact_eligibility`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    organization_id: NonEmptyStr | None = None
    project_id: NonEmptyStr | None = None
    task_type: ModelImpactTaskType = ModelImpactTaskType.SUPERVISED_TABULAR_CLASSIFICATION
    tabular_profile: TabularProfileReport | None = None
    tabular_profile_artifact: ArtifactRef | None = None
    split_manifest: SplitManifest | None = None
    split_manifest_artifact: ArtifactRef | None = None
    split_leakage_report: SplitLeakageReport | None = None
    split_leakage_report_artifact: ArtifactRef | None = None
    candidate_version_artifact: ArtifactRef | None = None
    fallback_report: FallbackReadinessReportRef | None = None
    minimum_labeled_samples: int = DEFAULT_MIN_LABELED_SAMPLES
    minimum_rare_class_samples: int = DEFAULT_MIN_RARE_CLASS_SAMPLES
    minimum_splits: int = DEFAULT_MIN_SPLITS
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class CheckModelImpactEligibilityResult:
    """Persisted eligibility report and its registry record."""

    report: ModelImpactEligibilityReport
    report_artifact: RegisteredArtifact


def check_model_impact_eligibility(
    request: CheckModelImpactEligibilityRequest,
    *,
    registry: ArtifactRegistry,
) -> CheckModelImpactEligibilityResult:
    """Evaluate eligibility, persist the report, and return the result.

    The function never mutates raw or candidate dataset artifacts. It
    reads aggregate signals from the tabular profile, split manifest
    and leakage report and writes a deterministic immutable JSON
    artifact through ``ArtifactRegistry``.
    """
    target_column = _resolve_target_column(request)
    target_present = target_column is not None
    target_missing_values = _has_target_missing_values(request.tabular_profile)
    cohort_stats = _resolve_cohort_stats(request.tabular_profile)
    rare_class_label = _resolve_rare_class_label(request.tabular_profile, cohort_stats)
    total_labeled = _resolve_total_labeled(
        cohort_stats=cohort_stats, profile=request.tabular_profile
    )
    rare_class_count = _resolve_rare_class_count(
        cohort_stats=cohort_stats, rare_class_label=rare_class_label
    )
    split_stats = _resolve_split_stats(request.split_manifest, rare_class_label)
    not_eligible_reason_codes: list[ModelImpactNotEligibleReasonCode] = []
    reasons: list[str] = []
    required_inputs_present: list[ModelImpactInputName] = []
    required_inputs_missing: list[ModelImpactInputName] = []

    if request.task_type not in _SUPPORTED_TASK_TYPES:
        not_eligible_reason_codes.append(
            ModelImpactNotEligibleReasonCode.UNSUPPORTED_TASK_TYPE
        )
    else:
        required_inputs_present.append(ModelImpactInputName.BASELINE_MODEL_AVAILABLE)

    if target_present and not target_missing_values:
        required_inputs_present.append(ModelImpactInputName.TARGET_LABEL)
        reasons.append("target_label_present")
    else:
        required_inputs_missing.append(ModelImpactInputName.TARGET_LABEL)
        if not target_present:
            not_eligible_reason_codes.append(
                ModelImpactNotEligibleReasonCode.TARGET_LABEL_MISSING
            )
            reasons.append("target_label_missing")
        if target_missing_values:
            not_eligible_reason_codes.append(
                ModelImpactNotEligibleReasonCode.TARGET_COLUMN_HAS_MISSING_VALUES
            )
            reasons.append("target_column_has_missing_values")

    enough_labeled = total_labeled >= request.minimum_labeled_samples
    enough_rare = rare_class_count >= request.minimum_rare_class_samples
    if enough_labeled and enough_rare:
        required_inputs_present.append(ModelImpactInputName.ENOUGH_LABELED_SAMPLES)
        reasons.append("enough_labeled_samples")
    else:
        required_inputs_missing.append(ModelImpactInputName.ENOUGH_LABELED_SAMPLES)
        if not enough_labeled:
            not_eligible_reason_codes.append(
                ModelImpactNotEligibleReasonCode.NOT_ENOUGH_LABELED_SAMPLES
            )
            reasons.append("not_enough_labeled_samples")
        if not enough_rare:
            not_eligible_reason_codes.append(
                ModelImpactNotEligibleReasonCode.NOT_ENOUGH_RARE_CLASS_SAMPLES
            )
            reasons.append("not_enough_rare_class_samples")

    valid_split = _has_valid_split(
        manifest=request.split_manifest, minimum_splits=request.minimum_splits
    )
    if valid_split:
        required_inputs_present.append(ModelImpactInputName.VALID_SPLIT_STRATEGY)
        reasons.append("split_manifest_valid")
    else:
        required_inputs_missing.append(ModelImpactInputName.VALID_SPLIT_STRATEGY)
        if request.split_manifest is None:
            not_eligible_reason_codes.append(
                ModelImpactNotEligibleReasonCode.SPLIT_MANIFEST_MISSING
            )
            reasons.append("split_manifest_missing")
        else:
            not_eligible_reason_codes.append(
                ModelImpactNotEligibleReasonCode.SPLIT_MANIFEST_INSUFFICIENT_SPLITS
            )
            reasons.append("split_manifest_insufficient_splits")

    leakage_blocker = _has_leakage_blocker(request.split_leakage_report)
    if not leakage_blocker:
        required_inputs_present.append(ModelImpactInputName.NO_LEAKAGE_BLOCKERS)
        reasons.append("no_leakage_blockers")
    else:
        required_inputs_missing.append(ModelImpactInputName.NO_LEAKAGE_BLOCKERS)
        not_eligible_reason_codes.append(
            ModelImpactNotEligibleReasonCode.LEAKAGE_BLOCKER_PRESENT
        )
        reasons.append("leakage_blocker_present")

    eligible = (
        not not_eligible_reason_codes
        and ModelImpactInputName.TARGET_LABEL in required_inputs_present
        and ModelImpactInputName.ENOUGH_LABELED_SAMPLES in required_inputs_present
        and ModelImpactInputName.VALID_SPLIT_STRATEGY in required_inputs_present
        and ModelImpactInputName.NO_LEAKAGE_BLOCKERS in required_inputs_present
        and ModelImpactInputName.BASELINE_MODEL_AVAILABLE in required_inputs_present
    )
    primary_reason_code = (
        _PRIMARY_REASON_ELIGIBLE
        if eligible
        else (
            not_eligible_reason_codes[0].value
            if not_eligible_reason_codes
            else "unknown_eligibility_failure"
        )
    )
    fallback = None if eligible else request.fallback_report
    fallback_reasons = (
        tuple(code.value for code in not_eligible_reason_codes) if fallback is None else ()
    )
    if fallback is None and not eligible:
        # No fallback artifact was provided: still record the reasons
        # so the platform can decide which fallback report kind to
        # produce next.
        _ = fallback_reasons  # documented but unused when fallback is None

    report = ModelImpactEligibilityReport(
        report_id=request.report_id
        or f"model_impact_eligibility_{uuid.uuid4().hex[:16]}",
        report_schema_version=MODEL_IMPACT_ELIGIBILITY_SCHEMA_VERSION,
        status=(
            ModelImpactEligibilityStatus.ELIGIBLE
            if eligible
            else ModelImpactEligibilityStatus.NOT_ELIGIBLE
        ),
        eligible=eligible,
        task_type=request.task_type,
        target_column=target_column,
        rare_class_label=rare_class_label,
        primary_reason_code=primary_reason_code,
        reasons=tuple(_dedupe(reasons)),
        required_inputs_present=tuple(_dedupe(required_inputs_present)),
        required_inputs_missing=tuple(_dedupe(required_inputs_missing)),
        not_eligible_reason_codes=tuple(_dedupe(not_eligible_reason_codes)),
        cohort_stats=cohort_stats,
        split_stats=split_stats,
        minimum_labeled_samples=request.minimum_labeled_samples,
        minimum_rare_class_samples=request.minimum_rare_class_samples,
        fallback_report=fallback,
        lineage=ModelImpactEligibilityLineage(
            organization_id=request.organization_id,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            parent_version_id=request.parent_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            candidate_version_artifact=request.candidate_version_artifact,
            tabular_profile_artifact=request.tabular_profile_artifact,
            split_manifest_artifact=request.split_manifest_artifact,
            split_leakage_report_artifact=request.split_leakage_report_artifact,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )

    artifact = registry.save_artifact(
        artifact_kind=MODEL_IMPACT_ELIGIBILITY_KIND,
        data=_serialize(report),
        artifact_format=MODEL_IMPACT_ELIGIBILITY_FORMAT,
        media_type=MODEL_IMPACT_ELIGIBILITY_MEDIA_TYPE,
        schema_version=MODEL_IMPACT_ELIGIBILITY_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "eligibility-status": report.status.value,
            "eligible": "true" if report.eligible else "false",
            "primary-reason-code": report.primary_reason_code,
            "task-type": request.task_type.value,
            "candidate-dataset-version-id": request.candidate_dataset_version_id,
            "parent-version-id": request.parent_version_id,
        },
    )
    return CheckModelImpactEligibilityResult(report=report, report_artifact=artifact)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_target_column(
    request: CheckModelImpactEligibilityRequest,
) -> str | None:
    profile = request.tabular_profile
    if profile is not None and profile.target_column:
        return profile.target_column
    if request.split_manifest is not None and request.split_manifest.target_column:
        return request.split_manifest.target_column
    return None


def _has_target_missing_values(profile: TabularProfileReport | None) -> bool:
    if profile is None or profile.missingness is None:
        return False
    return bool(profile.missingness.target_column_missing)


def _resolve_cohort_stats(
    profile: TabularProfileReport | None,
) -> tuple[ModelImpactCohortStats, ...]:
    diagnostics = _class_imbalance(profile)
    if diagnostics is None:
        return ()
    total = diagnostics.total_samples
    rare_label = diagnostics.rare_class_label
    cohorts: list[ModelImpactCohortStats] = []
    for entry in diagnostics.class_counts:
        ratio = entry.count / total if total else 0.0
        cohorts.append(
            ModelImpactCohortStats(
                label=entry.label,
                count=entry.count,
                ratio=min(max(ratio, 0.0), 1.0),
                is_rare_class=entry.label == rare_label,
            )
        )
    return tuple(cohorts)


def _resolve_rare_class_label(
    profile: TabularProfileReport | None,
    cohorts: Sequence[ModelImpactCohortStats],
) -> str | None:
    diagnostics = _class_imbalance(profile)
    if diagnostics is not None:
        return diagnostics.rare_class_label
    if not cohorts:
        return None
    return min(cohorts, key=lambda c: c.count).label


def _resolve_total_labeled(
    *,
    cohort_stats: Sequence[ModelImpactCohortStats],
    profile: TabularProfileReport | None,
) -> int:
    if cohort_stats:
        return sum(c.count for c in cohort_stats)
    if profile is None:
        return 0
    diagnostics = _class_imbalance(profile)
    if diagnostics is None:
        return 0
    return diagnostics.total_samples


def _resolve_rare_class_count(
    *,
    cohort_stats: Sequence[ModelImpactCohortStats],
    rare_class_label: str | None,
) -> int:
    if rare_class_label is None or not cohort_stats:
        return 0
    for cohort in cohort_stats:
        if cohort.label == rare_class_label:
            return cohort.count
    return 0


def _resolve_split_stats(
    manifest: SplitManifest | None,
    rare_class_label: str | None,
) -> tuple[ModelImpactSplitStats, ...]:
    if manifest is None:
        return ()
    counts: dict[str, dict[str, int]] = {}
    for assignment in manifest.assignments:
        bucket = counts.setdefault(assignment.split.value, {"total": 0, "rare": 0})
        bucket["total"] += 1
        if rare_class_label is not None and assignment.label == rare_class_label:
            bucket["rare"] += 1
    return tuple(
        ModelImpactSplitStats(
            split=split,
            total_count=bucket["total"],
            rare_class_count=bucket["rare"],
        )
        for split, bucket in sorted(counts.items())
    )


def _has_valid_split(
    *,
    manifest: SplitManifest | None,
    minimum_splits: int,
) -> bool:
    if manifest is None or not manifest.assignments:
        return False
    distinct_splits = {assignment.split for assignment in manifest.assignments}
    return len(distinct_splits) >= minimum_splits


def _has_leakage_blocker(report: SplitLeakageReport | None) -> bool:
    if report is None:
        return False
    if report.block_training or report.block_model_evaluation:
        return True
    return report.leakage_detected


def _class_imbalance(
    profile: TabularProfileReport | None,
) -> ClassImbalanceDiagnostics | None:
    if profile is None:
        return None
    return profile.class_imbalance


def _dedupe(values: Iterable[_T]) -> list[_T]:
    seen: list[_T] = []
    for value in values:
        if value in seen:
            continue
        seen.append(value)
    return seen


_T = TypeVar("_T")


def _serialize(report: ModelImpactEligibilityReport) -> bytes:
    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
    ).encode("utf-8")


__all__ = [
    "CheckModelImpactEligibilityRequest",
    "CheckModelImpactEligibilityResult",
    "MODEL_IMPACT_ELIGIBILITY_FORMAT",
    "MODEL_IMPACT_ELIGIBILITY_KIND",
    "MODEL_IMPACT_ELIGIBILITY_MEDIA_TYPE",
    "ModelImpactEligibilityError",
    "check_model_impact_eligibility",
]
