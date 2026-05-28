"""Contracts for split leakage check artifacts.

The split leakage report is produced after split creation. It carries the
results of post-split leakage checks (exact hash, group key, target
leakage candidate scan) so Decision Core and the Action Plan validation
gates can decide whether model-impact evaluation is allowed.

The report is the authoritative artifact for ``BLOCK_MODEL_EVALUATION``
and ``BLOCK_TRAINING`` decisions related to splits. It does not expose
raw row payloads — only object ids, group keys, column names, and
counts.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest


class LeakageCheckType(StrEnum):
    """Supported MVP post-split leakage checks."""

    EXACT_HASH = "exact_hash"
    GROUP_KEY = "group_key"
    TARGET_LEAKAGE_CANDIDATE = "target_leakage_candidate"


class LeakageCheckStatus(StrEnum):
    """Outcome of a single leakage check."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class LeakageCheckSeverity(StrEnum):
    """Severity carried by a leakage check finding."""

    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class LeakageCrossSplitFinding(BaseModel):
    """One concrete finding across two or more splits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: NonEmptyStr
    splits: tuple[NonEmptyStr, ...]
    object_ids: tuple[NonEmptyStr, ...] = ()
    group_value: str | None = None
    column: NonEmptyStr | None = None
    notes: str | None = None


class LeakageCheckResult(BaseModel):
    """Result of one post-split leakage check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_type: LeakageCheckType
    status: LeakageCheckStatus
    severity: LeakageCheckSeverity
    reason_code: NonEmptyStr
    block_action: NonEmptyStr | None = None
    findings_count: int = Field(ge=0, default=0)
    affected_object_ids: tuple[NonEmptyStr, ...] = ()
    affected_groups: tuple[NonEmptyStr, ...] = ()
    affected_columns: tuple[NonEmptyStr, ...] = ()
    findings: tuple[LeakageCrossSplitFinding, ...] = ()
    notes: str | None = None


class SplitLeakageLineage(BaseModel):
    """Lineage linking the leakage report back to its split manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr | None = None
    step_id: NonEmptyStr | None = None
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact: ArtifactRef
    split_manifest: ArtifactRef


class SplitLeakageReport(BaseModel):
    """Machine-readable post-split leakage report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "split_leakage_report.v1"
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    split_manifest_id: NonEmptyStr
    target_column: NonEmptyStr
    policy_version: NonEmptyStr
    leakage_detected: bool
    leakage_risk_score: Score
    block_model_evaluation: bool
    block_training: bool
    checks: tuple[LeakageCheckResult, ...]
    lineage: SplitLeakageLineage
    generated_at: datetime


__all__ = [
    "LeakageCheckResult",
    "LeakageCheckSeverity",
    "LeakageCheckStatus",
    "LeakageCheckType",
    "LeakageCrossSplitFinding",
    "SplitLeakageLineage",
    "SplitLeakageReport",
]
