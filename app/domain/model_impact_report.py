"""Contracts for the sklearn baseline model-impact report.

PRD §21.1.1 / §21.2 / DATASETS.md §11 require the report to compare
the source dataset version against the candidate dataset version using
a deterministic baseline classifier and to classify the candidate as
``improved``, ``degraded``, ``requires_review`` or ``rejected``.

For synthetic candidates the report must additionally include TSTR
(Train-Synthetic-Test-Real) and TRTS (Train-Real-Test-Synthetic)
results, or explicit ``not_applicable`` reasons. Synthetic utility
rules consume rare-class recall / macro F1 / weighted F1 / PR-AUC and
the validation gates summary to mark the synthetic method.

The report carries:

- baseline metrics computed on the candidate split structure;
- candidate metrics computed on the candidate dataset version;
- TSTR / TRTS metrics when the candidate is a synthetic candidate;
- a stable utility verdict;
- model configuration (algorithm, hyperparameters), random seed, and
  metric library/schema versions for reproducibility;
- lineage references to the candidate version, source / candidate
  split manifests, validation gates report and synthetic dataset
  report when applicable.

All values are aggregate metrics. Raw rows, raw values and PII never
enter the artifact.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest

MODEL_IMPACT_REPORT_SCHEMA_VERSION = "model_impact_report.v1"


class ModelImpactVerdict(StrEnum):
    """Top-level outcome for the candidate vs baseline comparison."""

    IMPROVED = "improved"
    REQUIRES_REVIEW = "requires_review"
    DEGRADED = "degraded"
    REJECTED = "rejected"


class SyntheticUtilityStatus(StrEnum):
    """Synthetic-method utility verdict per PRD §11.3 / DATASETS.md §11."""

    RECOMMENDED = "recommended"
    REQUIRES_REVIEW = "requires_review"
    REJECTED = "rejected"
    NOT_APPLICABLE = "not_applicable"


class MetricStatus(StrEnum):
    """Status used when a metric cannot be computed."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"


class ConfusionMatrixCell(BaseModel):
    """One cell of a confusion matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_label: NonEmptyStr
    predicted_label: NonEmptyStr
    count: int = Field(ge=0)


class ClassificationMetrics(BaseModel):
    """Classification metrics for one dataset / split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rare_class_label: NonEmptyStr
    rare_class_recall: Score
    rare_class_precision: Score
    macro_f1: Score
    weighted_f1: Score
    pr_auc: Score | None = None
    pr_auc_status: MetricStatus = MetricStatus.AVAILABLE
    pr_auc_reason: str | None = None
    confusion_matrix: tuple[ConfusionMatrixCell, ...]
    sample_count: int = Field(ge=0)


class TstrTrtsMetrics(BaseModel):
    """TSTR / TRTS metric block.

    When the candidate is not synthetic, ``status=not_applicable`` and
    an explicit ``reason`` is provided. TSTR/TRTS thresholds are also
    stored so reviewers can audit the rule that produced the verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: MetricStatus
    reason: str | None = None
    tstr_metrics: ClassificationMetrics | None = None
    trts_metrics: ClassificationMetrics | None = None
    tstr_macro_f1_drop: float | None = None
    tstr_macro_f1_threshold: float | None = None
    trts_macro_f1_delta: float | None = None
    trts_unstable_threshold: float | None = None


class BaselineModelConfig(BaseModel):
    """Baseline model configuration recorded for reproducibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: NonEmptyStr
    library: NonEmptyStr
    library_version: NonEmptyStr
    hyperparameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    feature_columns: tuple[NonEmptyStr, ...]
    target_column: NonEmptyStr
    excluded_columns: tuple[NonEmptyStr, ...] = ()
    random_seed: int


class ModelImpactReportLineage(BaseModel):
    """Lineage envelope for the model-impact report artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr | None = None
    project_id: NonEmptyStr | None = None
    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    candidate_version_artifact: ArtifactRef | None = None
    eligibility_report_artifact: ArtifactRef | None = None
    source_split_manifest: ArtifactRef | None = None
    candidate_split_manifest: ArtifactRef | None = None
    validation_gates_report_artifact: ArtifactRef | None = None
    synthetic_dataset_report_artifact: ArtifactRef | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest


class ModelImpactReport(BaseModel):
    """Sklearn baseline model-impact report.

    The report records before/after metrics on a configurable baseline
    classifier (LogisticRegression by default) along with the
    metric-library version and the random seed so that audit can
    reproduce the result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = MODEL_IMPACT_REPORT_SCHEMA_VERSION
    metric_library: NonEmptyStr = "scikit-learn"
    metric_library_version: NonEmptyStr
    verdict: ModelImpactVerdict
    verdict_reason_codes: tuple[NonEmptyStr, ...]
    notes: str | None = None
    baseline_metrics: ClassificationMetrics
    candidate_metrics: ClassificationMetrics
    rare_class_recall_before: Score
    rare_class_recall_after: Score
    rare_class_recall_delta: float
    macro_f1_before: Score
    macro_f1_after: Score
    macro_f1_delta: float
    weighted_f1_before: Score
    weighted_f1_after: Score
    weighted_f1_delta: float
    pr_auc_before: Score | None = None
    pr_auc_after: Score | None = None
    pr_auc_delta: float | None = None
    pr_auc_status: MetricStatus = MetricStatus.AVAILABLE
    pr_auc_reason: str | None = None
    tstr_trts: TstrTrtsMetrics
    synthetic_utility_status: SyntheticUtilityStatus = SyntheticUtilityStatus.NOT_APPLICABLE
    synthetic_utility_reason_codes: tuple[NonEmptyStr, ...] = ()
    baseline_model_config: BaselineModelConfig
    candidate_model_config: BaselineModelConfig
    lineage: ModelImpactReportLineage
    generated_at: datetime


__all__ = [
    "BaselineModelConfig",
    "ClassificationMetrics",
    "ConfusionMatrixCell",
    "MODEL_IMPACT_REPORT_SCHEMA_VERSION",
    "MetricStatus",
    "ModelImpactReport",
    "ModelImpactReportLineage",
    "ModelImpactVerdict",
    "SyntheticUtilityStatus",
    "TstrTrtsMetrics",
]
