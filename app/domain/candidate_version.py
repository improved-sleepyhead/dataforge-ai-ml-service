"""Contracts for proposed candidate dataset version artifacts.

A *candidate dataset version* is the artifact that DataForge AI emits at
the end of an ``APPLY_SELECTED_ACTIONS`` Dagster run. It bundles all
references that the control plane needs to decide whether to promote
the run into a finalized dataset version:

- the immutable raw source version that was used as the parent;
- the action-plan id that was approved;
- the policy versions in force at the moment of execution;
- every output artifact ref produced by the run;
- the validation-gates summary;
- synthetic-method metadata when the candidate carries synthetic rows;
- explicit ``status`` (proposed / blocked / failed).

The Python compute plane never writes the *final* lifecycle state of a
dataset version into the platform database. It only persists the
candidate metadata as an immutable artifact in object storage and emits
a ``DATASET_VERSION_PROPOSED`` audit event so the platform can pick the
proposal up and apply its own approval workflow. The compute plane
also never overwrites a previously persisted candidate-version artifact:
re-running with the same content hashes returns the existing record,
and a different content hash creates a new artifact rather than
mutating the old one.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Sha256Digest

CANDIDATE_DATASET_VERSION_SCHEMA_VERSION = "candidate_dataset_version.v1"


class CandidateVersionStatus(StrEnum):
    """High-level outcome state for a candidate dataset version.

    Notes:
        ``PROPOSED`` — gates passed and the platform may now run its
        approval workflow. The compute plane never sets ``PROMOTED``;
        promotion is a control-plane lifecycle action.

        ``BLOCKED`` — execution finished, but at least one validation
        gate marked the candidate ``validation_failed``. The candidate
        artifacts remain in storage for audit, but the candidate must
        not be promoted as-is.

        ``FAILED`` — execution itself failed before validation could
        complete. The candidate is recorded so the platform can surface
        the failure with its evidence.
    """

    PROPOSED = "proposed"
    BLOCKED = "blocked"
    FAILED = "failed"


class CandidatePolicyVersions(BaseModel):
    """Versioned policies referenced by a candidate version artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_policy_version: NonEmptyStr
    decision_policy_version: NonEmptyStr
    score_policy_version: NonEmptyStr
    method_policy_version: NonEmptyStr
    validation_gates_policy_version: NonEmptyStr | None = None


class CandidateActionStepSummary(BaseModel):
    """Compact record of one ActionPlan step that produced this candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: NonEmptyStr
    step_type: NonEmptyStr
    method_id: NonEmptyStr
    plugin_id: NonEmptyStr
    plugin_version: NonEmptyStr
    config_hash: Sha256Digest
    output_artifact_kind: NonEmptyStr
    random_seed: int | None = None


class SyntheticCandidateMetadata(BaseModel):
    """Synthetic-method metadata required for synthetic candidates.

    Mirrors PRD §11.13 and the acceptance criteria from TASK-049: every
    synthetic candidate must capture the method, plugin, plugin
    version, random seed, config hash, the train split / cohort that
    seeded the generator, source object ids when it is privacy-safe to
    record them, generated row count, sampling strategy, policy
    version, the validation report reference and (when available) the
    model-impact report reference plus a lineage block.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: NonEmptyStr
    plugin_id: NonEmptyStr
    plugin_version: NonEmptyStr
    random_seed: int
    config_hash: Sha256Digest
    source_split: NonEmptyStr
    source_cohort: str | None = None
    source_object_ids: tuple[NonEmptyStr, ...] = ()
    source_object_ids_truncated: bool = False
    full_source_object_ids_count: int = Field(ge=0, default=0)
    generated_count: int = Field(ge=0)
    sampling_strategy: float = Field(ge=0.0)
    policy_version: NonEmptyStr
    validation_report: ArtifactRef
    model_impact_report: ArtifactRef | None = None
    synthetic_dataset_report: ArtifactRef


class CandidateDatasetVersionLineage(BaseModel):
    """Lineage envelope for the candidate dataset version artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    proposed_version_name: NonEmptyStr
    action_plan_id: NonEmptyStr
    decision_report_id: NonEmptyStr | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    input_artifact_hashes: tuple[Sha256Digest, ...]
    output_artifact_hashes: tuple[Sha256Digest, ...]
    input_manifest_hash: Sha256Digest | None = None
    output_manifest_hash: Sha256Digest | None = None


class CandidateDatasetVersion(BaseModel):
    """Immutable proposed candidate dataset version metadata artifact.

    The artifact is emitted at the end of ``APPLY_SELECTED_ACTIONS``.
    The compute plane writes it into object storage and signals the
    platform via a ``DATASET_VERSION_PROPOSED`` audit event. The
    platform then runs its own approval workflow before promoting the
    candidate to a finalized dataset version.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_version_id: NonEmptyStr
    candidate_version_schema_version: NonEmptyStr = (
        CANDIDATE_DATASET_VERSION_SCHEMA_VERSION
    )
    status: CandidateVersionStatus
    policy_versions: CandidatePolicyVersions
    lineage: CandidateDatasetVersionLineage
    candidate_artifacts: tuple[ArtifactRef, ...]
    primary_dataset_artifact: ArtifactRef
    validation_gates_report: ArtifactRef | None = None
    validation_gates_summary: dict[str, str | bool | int | float] = Field(
        default_factory=dict
    )
    block_export: bool
    block_model_evaluation: bool
    block_training: bool
    blocker_reason_codes: tuple[NonEmptyStr, ...] = ()
    action_plan_steps: tuple[CandidateActionStepSummary, ...]
    synthetic_metadata: SyntheticCandidateMetadata | None = None
    proposed_at: datetime


__all__ = [
    "CANDIDATE_DATASET_VERSION_SCHEMA_VERSION",
    "CandidateActionStepSummary",
    "CandidateDatasetVersion",
    "CandidateDatasetVersionLineage",
    "CandidatePolicyVersions",
    "CandidateVersionStatus",
    "SyntheticCandidateMetadata",
]
