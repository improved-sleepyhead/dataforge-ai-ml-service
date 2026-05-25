"""Contracts for the ``lineage.json`` artifact (TASK-055).

PRD §20 requires every output to link to its inputs through a
machine-readable lineage record carrying:

- ``parent_version_id`` of the immutable raw source;
- ``job_id`` of the compute run that produced the artifact;
- ``input_artifact_hashes`` and ``output_artifact_hashes``;
- algorithm/plugin names + versions;
- ``config_hash`` and ``policy_version``;
- ``random_seed`` when applicable;
- ``created_at``.

The lineage report is shaped so its JSON serialization can be mapped
to OpenLineage Job/Run/Dataset concepts in the future without a
contract change. The ``openlineage`` envelope already carries Job /
Run / Input/Output Dataset facets in OpenLineage-compatible naming.

The artifact never carries raw rows, raw text, raw PII or signed
request bodies — only ids, hashes, version strings, counts and
timestamps.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Sha256Digest

LINEAGE_REPORT_SCHEMA_VERSION = "lineage_report.v1"


class LineageAlgorithm(BaseModel):
    """One algorithm/plugin invocation that contributed to the output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: NonEmptyStr
    step_type: NonEmptyStr
    algorithm_name: NonEmptyStr
    algorithm_version: NonEmptyStr
    plugin_id: NonEmptyStr
    plugin_version: NonEmptyStr
    config_hash: Sha256Digest
    random_seed: int | None = None


class LineagePolicyVersions(BaseModel):
    """Policy versions in force when the output was produced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_policy_version: NonEmptyStr | None = None
    decision_policy_version: NonEmptyStr | None = None
    score_policy_version: NonEmptyStr | None = None
    method_policy_version: NonEmptyStr | None = None
    validation_gates_policy_version: NonEmptyStr | None = None
    privacy_policy_version: NonEmptyStr | None = None
    export_policy_version: NonEmptyStr | None = None


class OpenLineageDataset(BaseModel):
    """OpenLineage-compatible dataset descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: NonEmptyStr
    name: NonEmptyStr
    facets: dict[str, str | int | float | bool] = Field(default_factory=dict)


class OpenLineageRun(BaseModel):
    """OpenLineage-compatible run descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: NonEmptyStr
    job_namespace: NonEmptyStr
    job_name: NonEmptyStr
    started_at: datetime
    completed_at: datetime
    facets: dict[str, str | int | float | bool] = Field(default_factory=dict)


class OpenLineageEnvelope(BaseModel):
    """Future-compatible OpenLineage payload included in lineage.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: OpenLineageRun
    inputs: tuple[OpenLineageDataset, ...]
    outputs: tuple[OpenLineageDataset, ...]


class LineageReport(BaseModel):
    """Top-level ``lineage.json`` artifact contract.

    The required PRD §20.1 fields are exposed at the top level for
    direct mapping; an ``algorithms`` list captures multi-step
    compositions (split → impute → SMOTE), and ``openlineage`` carries
    the OpenLineage-compatible envelope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lineage_report_id: NonEmptyStr
    lineage_schema_version: NonEmptyStr = LINEAGE_REPORT_SCHEMA_VERSION
    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    output_dataset_version_id: NonEmptyStr
    job_id: NonEmptyStr
    input_artifact_hashes: tuple[Sha256Digest, ...]
    output_artifact_hashes: tuple[Sha256Digest, ...]
    algorithm_name: NonEmptyStr
    algorithm_version: NonEmptyStr
    config_hash: Sha256Digest
    policy_version: NonEmptyStr
    random_seed: int | None = None
    action_plan_id: NonEmptyStr | None = None
    decision_report_id: NonEmptyStr | None = None
    candidate_version_artifact: ArtifactRef | None = None
    export_package_artifact: ArtifactRef | None = None
    algorithms: tuple[LineageAlgorithm, ...]
    policy_versions: LineagePolicyVersions
    input_artifact_refs: tuple[ArtifactRef, ...] = ()
    output_artifact_refs: tuple[ArtifactRef, ...] = ()
    openlineage: OpenLineageEnvelope
    created_at: datetime


__all__ = [
    "LINEAGE_REPORT_SCHEMA_VERSION",
    "LineageAlgorithm",
    "LineagePolicyVersions",
    "LineageReport",
    "OpenLineageDataset",
    "OpenLineageEnvelope",
    "OpenLineageRun",
]
