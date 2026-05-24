"""Contracts for safe tabular imputation execution artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Sha256Digest


class ImputationMethod(StrEnum):
    """Supported safe imputation methods for MVP tabular actions."""

    MEDIAN = "median"
    MODE = "mode"
    GROUP_MEDIAN = "group_median"
    MISSINGNESS_INDICATOR = "missingness_indicator"


class ImputationColumnReport(BaseModel):
    """Before/after imputation metrics for one column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: NonEmptyStr
    method: ImputationMethod
    indicator_column: str | None = None
    group_key: str | None = None
    before_missing_count: int = Field(ge=0)
    after_missing_count: int = Field(ge=0)
    imputed_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    fill_value: str | None = None
    group_imputed_counts: dict[str, int] = Field(default_factory=dict)


class TabularImputationLineage(BaseModel):
    """Lineage linking an imputation report to ActionPlan and artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact: ArtifactRef
    candidate_artifact: ArtifactRef


class TabularImputationReport(BaseModel):
    """Machine-readable report for safe tabular imputation execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "tabular_imputation_report.v1"
    action_plan_id: NonEmptyStr
    step_id: NonEmptyStr
    target_column: NonEmptyStr
    target_unchanged: bool
    before_row_count: int = Field(ge=0)
    after_row_count: int = Field(ge=0)
    before_missing_total: int = Field(ge=0)
    after_missing_total: int = Field(ge=0)
    columns: tuple[ImputationColumnReport, ...]
    candidate_artifact: ArtifactRef
    lineage: TabularImputationLineage
    generated_at: datetime


__all__ = [
    "ImputationColumnReport",
    "ImputationMethod",
    "TabularImputationLineage",
    "TabularImputationReport",
]
