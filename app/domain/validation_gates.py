"""Contracts for candidate-artifact validation gates.

The validation-gates report is the unified artifact that records whether
a candidate artifact (imputation, duplicate action, redaction, synthetic
generation, ...) passes the gates required before it can be promoted to
a candidate dataset version, before model-impact evaluation, and before
controlled export.

Design notes
------------

- Decision Core / export gates consume only the normalized
  ``ValidationGatesReport`` block. They never re-read raw plugin output
  or raw row payloads.
- Each gate carries a stable ``gate_type``, ``status``,
  ``severity``, ``reason_code`` and ``block_action``. ``not_applicable``
  must be returned with an explicit reason instead of silently omitting
  the gate.
- DCR ("distance to closest record") is recorded with the explicit
  formula

  ``DCR(x_synth) = min_{x_real ∈ D_real} distance(x_synth, x_real)``

  so reviewers can audit the metric without reading code.
- TSTR/TRTS/SHAP/TabSynDex are represented in the contract; in the MVP
  profile they may resolve to ``not_applicable`` with a reason such as
  ``model_impact_pipeline_not_available``.
- Raw row payloads are never exposed. Findings carry only ``object_id``
  values, column names, group identifiers, hashes, and counts.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.artifact import ArtifactRef
from app.domain.common import NonEmptyStr, Score, Sha256Digest

DCR_FORMULA = "DCR(x_synth) = min_{x_real in D_real} distance(x_synth, x_real)"
"""Distance-to-Closest-Record metric formula (PRD §11.12 / DATASETS.md §8)."""


class ValidationGateType(StrEnum):
    """Validation gates supported by the MVP executor.

    The list enumerates the gates the PRD requires for candidate
    artifacts (PRD §3, §11.12, DATASETS.md §8). Synthetic-only gates
    that do not apply to a non-synthetic candidate artifact must be
    represented explicitly with ``status=not_applicable`` and a
    reason code.
    """

    SCHEMA_VALIDATION = "schema_validation"
    BUSINESS_RULES = "business_rules"
    PRIVACY_CHECK = "privacy_check"
    SPLIT_LEAKAGE_CHECK = "split_leakage_check"
    SYNTHETIC_DCR_CHECK = "synthetic_dcr_check"
    SYNTHETIC_EXACT_DUPLICATE_TO_REAL = "synthetic_exact_duplicate_to_real"
    SYNTHETIC_NEAREST_NEIGHBOR_PRIVACY = "synthetic_nearest_neighbor_privacy"
    SYNTHETIC_DISTRIBUTION_SIMILARITY = "synthetic_distribution_similarity"
    SYNTHETIC_CORRELATION_PRESERVATION = "synthetic_correlation_preservation"
    SYNTHETIC_MISSINGNESS_COMPARISON = "synthetic_missingness_comparison"
    SYNTHETIC_TSTR = "synthetic_tstr"
    SYNTHETIC_TRTS = "synthetic_trts"
    SYNTHETIC_SHAP_CONSISTENCY = "synthetic_shap_consistency"
    SYNTHETIC_TABSYNDEX = "synthetic_tabsyndex"
    RAW_ARTIFACT_IMMUTABILITY = "raw_artifact_immutability"


class ValidationGateStatus(StrEnum):
    """Status of a single validation gate."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class ValidationGateSeverity(StrEnum):
    """Severity carried by a gate finding."""

    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class ValidationGateFinding(BaseModel):
    """One concrete finding produced by a gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: NonEmptyStr
    object_ids: tuple[NonEmptyStr, ...] = ()
    columns: tuple[NonEmptyStr, ...] = ()
    notes: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    metric_threshold: float | None = None


class ValidationGateMetric(BaseModel):
    """Per-gate numeric metric (DCR, TSTR/TRTS scores, etc.)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyStr
    value: float | None = None
    threshold: float | None = None
    formula: str | None = None
    notes: str | None = None


class CandidateValidationGate(BaseModel):
    """Result of one validation gate against a candidate artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_type: ValidationGateType
    status: ValidationGateStatus
    severity: ValidationGateSeverity
    reason_code: NonEmptyStr
    block_action: NonEmptyStr | None = None
    findings_count: int = Field(ge=0, default=0)
    pass_rate: Score | None = None
    metrics: tuple[ValidationGateMetric, ...] = ()
    findings: tuple[ValidationGateFinding, ...] = ()
    notes: str | None = None
    not_applicable_reason: str | None = None


class CandidateArtifactStatus(StrEnum):
    """High-level outcome for the candidate artifact under validation."""

    OK = "ok"
    VALIDATION_FAILED = "validation_failed"
    REVIEW_REQUIRED = "review_required"


class ValidationGatesLineage(BaseModel):
    """Lineage linking the gates report to its inputs/outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr | None = None
    step_id: NonEmptyStr | None = None
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact: ArtifactRef
    candidate_artifact: ArtifactRef
    split_manifest: ArtifactRef | None = None
    split_leakage_report: ArtifactRef | None = None
    synthetic_dataset_report: ArtifactRef | None = None


class ValidationGatesReport(BaseModel):
    """Machine-readable validation-gates report for a candidate artifact.

    The report is the single source of truth that downstream stages
    (Decision Core, export gates, model-impact eligibility) consume to
    decide whether the candidate artifact may proceed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: NonEmptyStr
    report_schema_version: NonEmptyStr = "validation_gates_report.v1"
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    candidate_artifact_kind: NonEmptyStr
    policy_version: NonEmptyStr
    overall_status: ValidationGateStatus
    candidate_status: CandidateArtifactStatus
    raw_artifact_unchanged: bool
    block_export: bool
    block_model_evaluation: bool
    block_training: bool
    blocker_present: bool
    blocker_gate_types: tuple[ValidationGateType, ...] = ()
    gates: tuple[CandidateValidationGate, ...]
    lineage: ValidationGatesLineage
    generated_at: datetime


__all__ = [
    "CandidateArtifactStatus",
    "CandidateValidationGate",
    "DCR_FORMULA",
    "ValidationGateFinding",
    "ValidationGateMetric",
    "ValidationGateSeverity",
    "ValidationGateStatus",
    "ValidationGateType",
    "ValidationGatesLineage",
    "ValidationGatesReport",
]
