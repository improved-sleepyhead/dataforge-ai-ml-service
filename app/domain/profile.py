"""Tabular profile contracts.

These contracts describe the output of the tabular profiler. The profiler
turns a validated tabular manifest plus its source CSV into a structured
report that other Decision Core / kernel components can consume.

The report is intentionally narrow at TASK-023 scope:

* schema inference (column type hints, nullability);
* row/column counts;
* role detection: target/label, group key, id-like, PII-like;
* link to the validated manifest artifact and the dataset version.

Subsequent tasks add missingness, duplicates, outliers, leakage, and
business-rule signals on top of this base contract.
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
    lineage: TabularProfileLineage
    generated_at: datetime


__all__ = [
    "ColumnProfile",
    "ColumnRole",
    "ColumnType",
    "TabularProfileLineage",
    "TabularProfileReport",
]
