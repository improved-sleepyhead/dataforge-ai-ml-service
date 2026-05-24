"""Contracts for model-impact eligibility and fallback readiness reports.

PRD §21.1.1 makes model-impact evaluation conditional: the platform may
only promise it when the dataset, task and split make a baseline
classifier comparison meaningful. The eligibility check encodes those
preconditions as a contract artifact so:

- the platform UI can surface the exact reason a candidate is or is
  not eligible;
- downstream stages (model-impact builder, export readiness) can
  consume the eligibility decision without re-deriving it;
- candidates that are not eligible can still receive a deterministic
  *fallback readiness report* covering completeness, validity, privacy
  safety, duplicate reduction and review-queue precision proxies.

The eligibility report is privacy-safe: it carries reason codes,
counts, the target column name, the supported task type and lineage
references only — no raw row payloads, no PII.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest

MODEL_IMPACT_ELIGIBILITY_SCHEMA_VERSION = "model_impact_eligibility.v1"
"""Schema version stamped on every eligibility artifact."""

DEFAULT_MIN_LABELED_SAMPLES = 50
"""Minimum number of labeled samples required to fit a deterministic baseline.

The default is intentionally conservative — for fewer than 50 labeled
rows a sklearn baseline cannot produce a stable rare-class recall
estimate.  Tests/dev runners may override the threshold via the
eligibility request when the demo dataset is smaller.
"""

DEFAULT_MIN_RARE_CLASS_SAMPLES = 5
"""Minimum number of minority-class rows required for rare-class recall."""

DEFAULT_MIN_SPLITS = 2
"""Minimum number of distinct splits the manifest must define (train+test)."""


class ModelImpactTaskType(StrEnum):
    """Task types for which a baseline model-impact evaluator exists."""

    SUPERVISED_TABULAR_CLASSIFICATION = "supervised_tabular_classification"
    SUPERVISED_TABULAR_REGRESSION = "supervised_tabular_regression"


class ModelImpactInputName(StrEnum):
    """Required-input markers used in the eligibility report."""

    TARGET_LABEL = "target_label"
    ENOUGH_LABELED_SAMPLES = "enough_labeled_samples"
    VALID_SPLIT_STRATEGY = "valid_split_strategy"
    NO_LEAKAGE_BLOCKERS = "no_leakage_blockers"
    BASELINE_MODEL_AVAILABLE = "baseline_model_available"


class ModelImpactNotEligibleReasonCode(StrEnum):
    """Stable reason codes recorded when a candidate is not eligible."""

    TARGET_LABEL_MISSING = "target_label_missing"
    TARGET_COLUMN_HAS_MISSING_VALUES = "target_column_has_missing_values"
    NOT_ENOUGH_LABELED_SAMPLES = "not_enough_labeled_samples"
    NOT_ENOUGH_RARE_CLASS_SAMPLES = "not_enough_rare_class_samples"
    SPLIT_MANIFEST_MISSING = "split_manifest_missing"
    SPLIT_MANIFEST_INSUFFICIENT_SPLITS = "split_manifest_insufficient_splits"
    LEAKAGE_BLOCKER_PRESENT = "leakage_blocker_present"
    BASELINE_MODEL_UNAVAILABLE = "baseline_model_unavailable"
    UNSUPPORTED_TASK_TYPE = "unsupported_task_type"


class ModelImpactEligibilityStatus(StrEnum):
    """Aggregate status used by Decision Core / report builders."""

    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"


class FallbackReadinessReportRef(BaseModel):
    """Pointer to the deterministic fallback readiness report.

    The fallback report is itself an immutable artifact (typically a
    DataForgeReport with workflow_type=ANALYZE_ONLY); the eligibility
    contract carries a reference and the explicit reason set so the
    UI can render the proper non-model-impact narrative.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fallback_kind: NonEmptyStr
    fallback_artifact: ArtifactRef
    reasons: tuple[NonEmptyStr, ...]


class ModelImpactCohortStats(BaseModel):
    """Per-class size summary used by the eligibility check.

    Captures only counts and ratios. The class label and counts are
    audit-safe because they are aggregate statistics, not raw rows.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonEmptyStr
    count: int = Field(ge=0)
    ratio: Score
    is_rare_class: bool = False


class ModelImpactSplitStats(BaseModel):
    """Per-split size summary used by the eligibility check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: NonEmptyStr
    total_count: int = Field(ge=0)
    rare_class_count: int = Field(ge=0)


class ModelImpactEligibilityLineage(BaseModel):
    """Lineage envelope for the eligibility artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr | None = None
    project_id: NonEmptyStr | None = None
    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    candidate_version_artifact: ArtifactRef | None = None
    tabular_profile_artifact: ArtifactRef | None = None
    split_manifest_artifact: ArtifactRef | None = None
    split_leakage_report_artifact: ArtifactRef | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest


class ModelImpactEligibilityReport(BaseModel):
    """Decision artifact for whether model-impact evaluation may run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = MODEL_IMPACT_ELIGIBILITY_SCHEMA_VERSION
    status: ModelImpactEligibilityStatus
    eligible: bool
    task_type: ModelImpactTaskType | None = None
    target_column: str | None = None
    rare_class_label: str | None = None
    primary_reason_code: NonEmptyStr
    reasons: tuple[NonEmptyStr, ...]
    required_inputs_present: tuple[ModelImpactInputName, ...]
    required_inputs_missing: tuple[ModelImpactInputName, ...]
    not_eligible_reason_codes: tuple[ModelImpactNotEligibleReasonCode, ...] = ()
    cohort_stats: tuple[ModelImpactCohortStats, ...] = ()
    split_stats: tuple[ModelImpactSplitStats, ...] = ()
    minimum_labeled_samples: int = Field(ge=0)
    minimum_rare_class_samples: int = Field(ge=0)
    fallback_report: FallbackReadinessReportRef | None = None
    lineage: ModelImpactEligibilityLineage
    generated_at: datetime


__all__ = [
    "DEFAULT_MIN_LABELED_SAMPLES",
    "DEFAULT_MIN_RARE_CLASS_SAMPLES",
    "DEFAULT_MIN_SPLITS",
    "FallbackReadinessReportRef",
    "MODEL_IMPACT_ELIGIBILITY_SCHEMA_VERSION",
    "ModelImpactCohortStats",
    "ModelImpactEligibilityLineage",
    "ModelImpactEligibilityReport",
    "ModelImpactEligibilityStatus",
    "ModelImpactInputName",
    "ModelImpactNotEligibleReasonCode",
    "ModelImpactSplitStats",
    "ModelImpactTaskType",
]
