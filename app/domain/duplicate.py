"""Contracts for duplicate marking/removal action artifacts.

The duplicate action executor produces two artifacts:

- a candidate tabular dataset CSV (``candidate_tabular_dataset``) that
  carries duplicate-group markers in MARK mode or has duplicate rows
  removed in REMOVE_CANDIDATE mode;
- a contract-valid ``duplicate_action_report`` artifact carrying
  per-group object_id sets, before/after row/pair counts, and lineage
  refs to the immutable source artifact.

The report never echoes raw row payloads — only ``object_id`` values,
group hashes, signature columns, and counts. The action never rewrites
the source artifact: REMOVE_CANDIDATE writes a new candidate CSV and
does not mutate the raw dataset version.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Sha256Digest


class DuplicateActionMode(StrEnum):
    """Supported MVP duplicate action modes.

    ``MARK`` adds duplicate-group marker columns to the candidate
    dataset without removing any rows. ``REMOVE_CANDIDATE`` keeps one
    canonical row per duplicate group and removes the rest from the
    candidate dataset only.
    """

    MARK = "MARK_DUPLICATE_CANDIDATES"
    REMOVE_CANDIDATE = "REMOVE_DUPLICATES"


class DuplicateGroupSummary(BaseModel):
    """Per-duplicate-group summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: NonEmptyStr
    signature_hash: Sha256Digest
    affected_object_ids: tuple[NonEmptyStr, ...]
    kept_object_id: NonEmptyStr | None = None
    removed_object_ids: tuple[NonEmptyStr, ...] = ()
    marked_object_ids: tuple[NonEmptyStr, ...] = ()


class DuplicateActionLineage(BaseModel):
    """Lineage linking the duplicate-action report to its inputs/outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact: ArtifactRef
    candidate_artifact: ArtifactRef


class DuplicateActionReport(BaseModel):
    """Top-level duplicate-action report contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "duplicate_action_report.v1"
    mode: DuplicateActionMode
    id_column: NonEmptyStr
    signature_columns: tuple[NonEmptyStr, ...]
    before_row_count: int = Field(ge=0)
    after_row_count: int = Field(ge=0)
    before_duplicate_pair_count: int = Field(ge=0)
    before_duplicate_group_count: int = Field(ge=0)
    after_duplicate_pair_count: int = Field(ge=0)
    after_duplicate_group_count: int = Field(ge=0)
    marked_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    groups: tuple[DuplicateGroupSummary, ...]
    raw_artifact_hash: Sha256Digest
    raw_artifact_unchanged: bool
    lineage: DuplicateActionLineage
    generated_at: datetime


__all__ = [
    "DuplicateActionLineage",
    "DuplicateActionMode",
    "DuplicateActionReport",
    "DuplicateGroupSummary",
]
