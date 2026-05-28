"""Contracts for tabular synthetic generation artifacts.

The MVP synthetic plugin implements two methods:

- ``SMOTE`` for targeted rare-class augmentation on the training split;
- ``GAUSSIAN_COPULA`` for distribution-level synthetic generation when
  enabled by policy.

Both methods produce the same top-level ``SyntheticDatasetReport``
shape so the kernel and Decision Core can consume them uniformly.
Method-specific metadata lives in dedicated optional blocks
(``gaussian_copula_artifacts``).

Reports never carry raw row payloads — only ``object_id`` values,
column names, formula identifiers, validation pass rates, and method
parameters. This keeps synthetic evidence safe for logs/exports while
preserving full reproducibility (seed + lineage + method parameters
are recorded).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest

SMOTE_FORMULA = "x_new = x_i + lambda * (x_nn - x_i)"
GAUSSIAN_COPULA_FORMULA = (
    "x_new = inverse_ecdf(Phi(L @ w + mu)), w ~ N(0, I)"
)


class SyntheticGenerationMethod(StrEnum):
    """Supported synthetic generation methods."""

    SMOTE = "smote"
    GAUSSIAN_COPULA = "gaussian_copula"


class SyntheticAugmentationKind(StrEnum):
    """High-level distinction between targeted vs distribution-level augmentation.

    SMOTE produces synthetic samples for a specific minority class on the
    training split. Gaussian Copula generates synthetic rows from the
    fitted joint distribution of the training split and may produce rows
    for any class observed in training. The kernel uses this enum to
    render method behaviour to the user without exposing algorithm
    internals.
    """

    TARGETED_RARE_CLASS = "targeted_rare_class"
    DISTRIBUTION_LEVEL = "distribution_level"


class SyntheticSampleLineage(BaseModel):
    """Lineage block for one generated synthetic sample.

    SMOTE entries reference a seed real object plus a neighbor and a
    sampled ``lambda``. Gaussian Copula entries reference the latent
    draw index instead. The model accepts both shapes so the same
    contract can describe either method.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    synthetic_object_id: NonEmptyStr
    method: SyntheticGenerationMethod
    formula: NonEmptyStr
    seed_object_id: str | None = None
    neighbor_object_id: str | None = None
    lambda_value: float | None = Field(default=None, ge=0.0, le=1.0)
    latent_draw_index: int | None = Field(default=None, ge=0)
    rare_class_label: str | None = None


class SyntheticClassStats(BaseModel):
    """Per-class generation counts for the synthetic report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonEmptyStr
    real_count_in_source_split: int = Field(ge=0)
    real_count_in_majority_split: int = Field(ge=0)
    target_count_after_augmentation: int = Field(ge=0)
    generated_count: int = Field(ge=0)
    achieved_ratio: Score


class SyntheticValidationCheckStatus(StrEnum):
    """Outcome of a single synthetic validation check."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class SyntheticValidationCheck(BaseModel):
    """One validation gate run against synthetic output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check: NonEmptyStr
    status: SyntheticValidationCheckStatus
    pass_rate: Score | None = None
    findings_count: int = Field(default=0, ge=0)
    blocker: bool = False
    notes: str | None = None


class SyntheticValidationReport(BaseModel):
    """Aggregate validation report for synthetic output.

    Captures schema, type, business-rule, and privacy checks performed
    against the generated synthetic rows. Each check carries a stable
    ``check`` identifier and may set ``blocker=True`` to indicate that
    Decision Core / export gates must reject the candidate dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_passed: bool
    blocker_present: bool
    checks: tuple[SyntheticValidationCheck, ...]


class ColumnDistributionTransform(BaseModel):
    """Empirical marginal distribution recorded for a feature column.

    Records the column name, the marginal-distribution method, and a
    bounded set of quantile snapshots used to invert the marginal back
    to data space after sampling from the latent normal.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: NonEmptyStr
    method: NonEmptyStr = "empirical_cdf"
    sample_count: int = Field(ge=0)
    quantile_levels: tuple[float, ...]
    quantile_values: tuple[float, ...]


class GaussianCopulaArtifacts(BaseModel):
    """Method-specific metadata for Gaussian Copula generation.

    ``column_distribution_transforms`` lists the marginal-distribution
    snapshots used to map data → uniform → latent normal. The latent
    space mean and Cholesky factor of the covariance let auditors
    re-fit the same generator from the report; ``correlation_matrix``
    is recorded for explainability. ``inverse_transform_metadata``
    documents how latent samples were mapped back to data space.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    column_distribution_transforms: tuple[ColumnDistributionTransform, ...]
    latent_normal_mean: tuple[float, ...]
    latent_normal_covariance_row_major: tuple[float, ...]
    latent_normal_cholesky_row_major: tuple[float, ...]
    correlation_matrix_row_major: tuple[float, ...]
    correlation_matrix_size: int = Field(ge=0)
    inverse_transform_metadata: dict[str, str]
    ridge_epsilon: float = Field(ge=0.0)
    cdf_clamp_epsilon: float = Field(ge=0.0)


class SyntheticDatasetLineage(BaseModel):
    """Lineage linking the synthetic report to its inputs/outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact: ArtifactRef
    split_manifest: ArtifactRef
    candidate_artifact: ArtifactRef
    augmented_split_manifest: ArtifactRef


class SyntheticDatasetReport(BaseModel):
    """Top-level synthetic dataset generation report.

    The report carries reproducibility parameters, per-class generation
    stats, an explicit ``augmentation_kind`` (so Decision Core/UI can
    distinguish targeted rare-class augmentation from distribution-
    level generation), an optional ``synthetic_validation`` block
    (schema/type/business-rules/privacy checks against generated rows),
    and an optional ``gaussian_copula_artifacts`` block carrying
    method-specific transformation metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "synthetic_dataset_report.v1"
    method: SyntheticGenerationMethod
    method_version: NonEmptyStr
    augmentation_kind: SyntheticAugmentationKind
    formula: NonEmptyStr
    target_column: NonEmptyStr
    rare_class_label: str | None = None
    source_split: NonEmptyStr = "train"
    random_seed: int
    k_neighbors: int = Field(ge=0)
    sampling_strategy: Score
    feature_columns: tuple[NonEmptyStr, ...]
    excluded_columns: tuple[NonEmptyStr, ...] = ()
    real_total_count: int = Field(ge=0)
    real_train_count: int = Field(ge=0)
    real_train_rare_count: int = Field(ge=0)
    generated_count: int = Field(ge=0)
    class_stats: tuple[SyntheticClassStats, ...]
    sample_lineage: tuple[SyntheticSampleLineage, ...]
    sample_lineage_truncated: bool = False
    full_sample_lineage_count: int = Field(ge=0)
    synthetic_validation: SyntheticValidationReport | None = None
    gaussian_copula_artifacts: GaussianCopulaArtifacts | None = None
    lineage: SyntheticDatasetLineage
    generated_at: datetime


__all__ = [
    "ColumnDistributionTransform",
    "GAUSSIAN_COPULA_FORMULA",
    "GaussianCopulaArtifacts",
    "SMOTE_FORMULA",
    "SyntheticAugmentationKind",
    "SyntheticClassStats",
    "SyntheticDatasetLineage",
    "SyntheticDatasetReport",
    "SyntheticGenerationMethod",
    "SyntheticSampleLineage",
    "SyntheticValidationCheck",
    "SyntheticValidationCheckStatus",
    "SyntheticValidationReport",
]
