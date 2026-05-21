"""Tabular profile contracts.

These contracts describe the output of the tabular profiler. The profiler
turns a validated tabular manifest plus its source CSV into a structured
report that other Decision Core / kernel components can consume.

The report is intentionally narrow at TASK-023 scope:

* schema inference (column type hints, nullability);
* row/column counts;
* role detection: target/label, group key, id-like, PII-like;
* link to the validated manifest artifact and the dataset version.

TASK-024 extends the report with missingness diagnostics:

* per-column ``missing_rate``;
* per-column missingness conditioned on the target column;
* per-column missingness conditioned on a segment column;
* a top-level ``target_column_missing`` flag plus ``missing_target_count``
  used by Decision Core to raise a hard-blocker candidate when the
  target column has any missing values.

Subsequent tasks add duplicates, outliers, leakage and business-rule
signals on top of this base contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest


class ColumnType(StrEnum):
    """Inferred column type hints used by the tabular profiler."""

    NUMERIC_INTEGER = "numeric_integer"
    NUMERIC_FLOAT = "numeric_float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"
    TEXT = "text"
    UNKNOWN = "unknown"


class ColumnRole(StrEnum):
    """Detected logical role of a column.

    A column may have at most one role; ``UNKNOWN`` is the default. The
    profiler must detect a target column, group key, id-like columns and
    PII-like columns separately because policy gates and Decision Core
    treat them differently.
    """

    UNKNOWN = "unknown"
    TARGET = "target"
    GROUP_KEY = "group_key"
    ID = "id"
    PII_LIKE = "pii_like"
    LEAKAGE_CANDIDATE = "leakage_candidate"


class ColumnProfile(BaseModel):
    """Profile entry for one column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyStr
    type: ColumnType
    role: ColumnRole
    nullable: bool
    null_count: int = Field(ge=0)
    null_ratio: Score
    distinct_count: int = Field(ge=0)
    sample_value: str | None = None
    is_constant: bool


class TabularProfileLineage(BaseModel):
    """Lineage block linking the profile back to its inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_manifest_artifact: ArtifactRef
    source_artifact_id: NonEmptyStr


class MissingnessGroupStats(BaseModel):
    """Missing/total counts and ratio for one group bucket."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    missing_ratio: Score


class MissingnessByGroup(BaseModel):
    """Per-group missingness breakdown for one column.

    The block describes how missingness for a single column is distributed
    over the values of a grouping column (typically the target or a
    customer segment). For each observed group value we record the share
    of missing entries and the absolute counts so Decision Core can decide
    whether the missingness pattern is segment-dependent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_column: NonEmptyStr
    groups: dict[str, MissingnessGroupStats]


class ColumnMissingness(BaseModel):
    """Aggregate missingness signals for one column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: NonEmptyStr
    missing_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    missing_rate: Score
    by_target: MissingnessByGroup | None = None
    by_segment: MissingnessByGroup | None = None


class MissingnessDiagnostics(BaseModel):
    """Top-level missingness diagnostics block.

    The block carries:

    * per-column ``missing_rate`` (also exposed in ``ColumnProfile``,
      duplicated here for symmetry with ``by_target`` / ``by_segment``);
    * grouped missingness for the most informative slices;
    * a flag and count proving whether the target column itself has
      missing values, which Decision Core treats as a hard-blocker
      candidate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_column: str | None = None
    target_column_missing: bool
    missing_target_count: int = Field(ge=0)
    columns: tuple[ColumnMissingness, ...]
    segment_column: str | None = None


class DuplicateDiagnostics(BaseModel):
    """Exact duplicate-row diagnostics.

    Two rows are considered duplicates when the canonical concatenation of
    all non-id columns is byte-equal. The diagnostics block reports the
    total number of duplicate row pairs (groups of size > 1 expanded to
    pairs), the affected ``object_id`` set, and the keys used to compute
    the duplicate signature so the downstream ActionPlan can reproduce
    the detection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    duplicate_pair_count: int = Field(ge=0)
    duplicate_group_count: int = Field(ge=0)
    affected_object_ids: tuple[NonEmptyStr, ...]
    signature_columns: tuple[NonEmptyStr, ...]
    id_column: str | None = None


class ColumnOutlierStats(BaseModel):
    """Per-column outlier statistics using the IQR rule.

    Outliers are defined as values strictly outside ``[Q1 - 1.5*IQR,
    Q3 + 1.5*IQR]``. Counts and the affected ``object_id`` sample are
    reported so Decision Core can plan winsorization or row-level
    review without re-reading the dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: NonEmptyStr
    outlier_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    outlier_rate: Score
    q1: float | None = None
    q3: float | None = None
    iqr_lower_bound: float | None = None
    iqr_upper_bound: float | None = None
    affected_object_ids: tuple[NonEmptyStr, ...] = ()


class OutlierDiagnostics(BaseModel):
    """Top-level outlier diagnostics block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: NonEmptyStr = "iqr_1.5"
    columns: tuple[ColumnOutlierStats, ...]


class ClassCount(BaseModel):
    """Count entry for one observed class label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonEmptyStr
    count: int = Field(ge=0)


class ClassImbalanceDiagnostics(BaseModel):
    """Class-imbalance diagnostics for the target column.

    The block decomposes the imbalance signal into machine-readable
    components so Decision Core can apply policy gates without
    recomputing them:

    * ``minority_share = min_c n_c / N`` — share of the smallest class.
    * ``imbalance_ratio = max_c n_c / min_c n_c`` — ratio between the
      largest and smallest class.
    * ``balance_score = 1 / log(1 + imbalance_ratio)`` — bounded score
      that drops as imbalance grows. Documented alternative
      ``min_c(n_c) / mean_c(n_c)`` is reported in
      ``balance_score_alternative`` for cross-validation.
    * ``effective_number_of_samples`` for the rare class follows
      ``E_n = (1 - β^n) / (1 - β)`` (Cui et al., 2019). It is stored as
      a per-class map so Decision Core can compare effective sample
      counts without recomputing β.

    ``rare_class_label`` and ``rare_class_ratio`` keep backward
    compatibility with TASK-017 fixture metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_column: NonEmptyStr
    total_samples: int = Field(ge=0)
    class_counts: tuple[ClassCount, ...]
    rare_class_label: NonEmptyStr
    rare_class_count: int = Field(ge=0)
    rare_class_ratio: Score
    minority_class_label: NonEmptyStr
    minority_class_share: Score
    imbalance_ratio: float = Field(ge=1.0)
    balance_score: Score
    balance_score_formula: NonEmptyStr = "1 / log(1 + imbalance_ratio)"
    balance_score_alternative: Score
    balance_score_alternative_formula: NonEmptyStr = "min_c(n_c) / mean_c(n_c)"
    effective_number_beta: float = Field(gt=0.0, lt=1.0)
    effective_number_of_samples: dict[str, float]


class LeakageCandidate(BaseModel):
    """One leakage candidate column reported by the profiler.

    The candidate carries a stable ``reason_code`` so Decision Core and
    the report layer can render the same signal.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: NonEmptyStr
    reason_code: NonEmptyStr
    target_match_rate: Score | None = None
    notes: NonEmptyStr | None = None


class LeakageDiagnostics(BaseModel):
    """Top-level leakage diagnostics block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[LeakageCandidate, ...] = ()


class TabularProfileReport(BaseModel):
    """Tabular profile report contract.

    The report is contract-shaped so it can be referenced as a
    ``DataForgeReport.detail_artifacts`` ``ArtifactRef`` once the tabular
    profile artifact is registered (see :func:`save_tabular_profile_report`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: NonEmptyStr
    profile_schema_version: NonEmptyStr = "tabular_profile_report.v1"
    source_system: NonEmptyStr
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: tuple[ColumnProfile, ...]
    target_column: str | None = None
    group_key_columns: tuple[NonEmptyStr, ...] = ()
    id_columns: tuple[NonEmptyStr, ...] = ()
    pii_like_columns: tuple[NonEmptyStr, ...] = ()
    missingness: MissingnessDiagnostics | None = None
    duplicates: DuplicateDiagnostics | None = None
    outliers: OutlierDiagnostics | None = None
    class_imbalance: ClassImbalanceDiagnostics | None = None
    leakage: LeakageDiagnostics | None = None
    lineage: TabularProfileLineage
    generated_at: datetime


__all__ = [
    "ClassCount",
    "ClassImbalanceDiagnostics",
    "ColumnMissingness",
    "ColumnOutlierStats",
    "ColumnProfile",
    "ColumnRole",
    "ColumnType",
    "DuplicateDiagnostics",
    "LeakageCandidate",
    "LeakageDiagnostics",
    "MissingnessByGroup",
    "MissingnessDiagnostics",
    "MissingnessGroupStats",
    "OutlierDiagnostics",
    "TabularProfileLineage",
    "TabularProfileReport",
]
