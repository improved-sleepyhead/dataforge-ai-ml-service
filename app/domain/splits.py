"""Contracts for supervised tabular split creation artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest
from app.domain.manifest import DataSplit


class SplitStrategy(StrEnum):
    """Supported MVP split strategies."""

    GROUP_STRATIFIED = "group_stratified"
    STRATIFIED = "stratified"
    RANDOM = "random"


class SplitAssignment(BaseModel):
    """One object assignment in a split manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    split: DataSplit
    label: NonEmptyStr
    group_value: str | None = None


class SplitClassDistribution(BaseModel):
    """Class distribution for one split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: DataSplit
    total_count: int = Field(ge=0)
    class_counts: dict[NonEmptyStr, int]
    class_ratios: dict[NonEmptyStr, Score]


class SplitManifestLineage(BaseModel):
    """Lineage linking split creation to source data and policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact: ArtifactRef


class SplitManifest(BaseModel):
    """Machine-readable split manifest for supervised tabular classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_manifest_id: NonEmptyStr
    split_schema_version: NonEmptyStr = "split_manifest.v1"
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    target_column: NonEmptyStr
    strategy: SplitStrategy
    seed: int
    group_key: NonEmptyStr | None = None
    policy_version: NonEmptyStr
    split_ratios: dict[DataSplit, Score]
    assignments: tuple[SplitAssignment, ...]
    class_distribution: tuple[SplitClassDistribution, ...]
    lineage: SplitManifestLineage
    generated_at: datetime


__all__ = [
    "SplitAssignment",
    "SplitClassDistribution",
    "SplitManifest",
    "SplitManifestLineage",
    "SplitStrategy",
]
