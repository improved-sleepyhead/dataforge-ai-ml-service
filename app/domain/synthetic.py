"""Contracts for tabular synthetic generation artifacts.

The MVP synthetic generator implements SMOTE for rare-class augmentation
on the training split only. The contracts described here cover three
artifacts produced by an approved ``AUGMENT_RARE_CLASS`` ActionPlan
step:

- ``synthetic_dataset_report``: top-level report carrying parameters,
  per-class generation stats, and a bounded sample of per-sample
  lineage entries for human review;
- ``candidate_tabular_dataset`` (existing kind): the augmented CSV with
  synthetic rows appended and marked through an ``is_synthetic`` flag;
- ``split_manifest`` v1 (existing): the executor extends the original
  manifest with synthetic-row assignments in the training split.

Reports never carry raw row payloads — only ``object_id``, neighbor
``object_id``, the sampled ``lambda``, and method parameters. This keeps
synthetic evidence safe for logs/exports while preserving full
reproducibility (seed + neighbor lineage + formula are recorded).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest

SMOTE_FORMULA = "x_new = x_i + lambda * (x_nn - x_i)"


class SyntheticGenerationMethod(StrEnum):
    """Supported synthetic generation methods.

    The MVP implements ``SMOTE`` only. Other methods are reserved here
    for downstream tasks (Gaussian Copula, Borderline-SMOTE, ADASYN,
    CTGAN, etc.) so the contract shape stays stable.
    """

    SMOTE = "smote"


class SyntheticSampleLineage(BaseModel):
    """Lineage block for one generated synthetic sample.

    Each synthetic row records the seed real object, the neighbor real
    object, the sampled ``lambda`` (uniform in [0, 1]), and the formula
    used. Together with ``random_seed`` from the report, this makes the
    sample reproducible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    synthetic_object_id: NonEmptyStr
    seed_object_id: NonEmptyStr
    neighbor_object_id: NonEmptyStr
    lambda_value: float = Field(ge=0.0, le=1.0)
    formula: NonEmptyStr = SMOTE_FORMULA
    rare_class_label: NonEmptyStr


class SyntheticClassStats(BaseModel):
    """Per-class generation counts for the synthetic report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonEmptyStr
    real_count_in_source_split: int = Field(ge=0)
    real_count_in_majority_split: int = Field(ge=0)
    target_count_after_augmentation: int = Field(ge=0)
    generated_count: int = Field(ge=0)
    achieved_ratio: Score


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

    Carries reproducibility parameters (``random_seed``, ``k_neighbors``,
    ``sampling_strategy``), per-class generation stats, the
    SMOTE formula, and a bounded sample of per-row lineage entries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "synthetic_dataset_report.v1"
    method: SyntheticGenerationMethod
    method_version: NonEmptyStr
    formula: NonEmptyStr = SMOTE_FORMULA
    target_column: NonEmptyStr
    rare_class_label: NonEmptyStr
    source_split: NonEmptyStr = "train"
    random_seed: int
    k_neighbors: int = Field(ge=1)
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
    lineage: SyntheticDatasetLineage
    generated_at: datetime


__all__ = [
    "SMOTE_FORMULA",
    "SyntheticClassStats",
    "SyntheticDatasetLineage",
    "SyntheticDatasetReport",
    "SyntheticGenerationMethod",
    "SyntheticSampleLineage",
]
