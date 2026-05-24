"""Validation gates executor for candidate artifacts.

This module implements TASK-048: a unified validation pass that runs the
gates required by PRD §3 / §11.12 / DATASETS.md §8 against a candidate
artifact and persists a contract-valid ``ValidationGatesReport``.

Supported gates
---------------

- ``schema_validation``: verify the candidate CSV preserves the
  declared schema columns and types.
- ``business_rules``: re-run the rule engine against the candidate
  rows. A critical violation marks the candidate ``review_required``
  and blocks export.
- ``privacy_check``: scan candidate rows for PII-like tokens (emails,
  phone numbers, passport-like ids). Blocks export when restricted PII
  appears in a candidate that is supposed to be redacted.
- ``split_leakage_check``: read a previously persisted
  ``SplitLeakageReport`` and bubble up its blockers (BLOCK_TRAINING /
  BLOCK_MODEL_EVALUATION).
- ``synthetic_dcr_check``: nearest-neighbor DCR using the formula
  ``DCR(x_synth) = min_{x_real ∈ D_real} distance(x_synth, x_real)``.
  When the candidate has no synthetic rows the gate reports
  ``not_applicable`` with a reason.
- ``synthetic_exact_duplicate_to_real``: blocks a synthetic row whose
  feature signature reproduces a real row exactly.
- ``synthetic_nearest_neighbor_privacy``: warn (or block) when a
  synthetic row is too close to its nearest real neighbor below a
  configurable epsilon.
- TSTR / TRTS / SHAP / TabSynDex / distribution similarity /
  correlation preservation / missingness comparison are represented in
  the contract; in MVP they resolve to ``not_applicable`` with explicit
  ``not_applicable_reason`` strings (model-impact pipelines and
  full distribution stats land in TASK-050+).
- ``raw_artifact_immutability``: re-read the source artifact bytes from
  storage and verify their sha256 is unchanged. The gate fails closed
  if the source could not be re-read or if the hash drifted.

The executor never overwrites the candidate or the source artifact.
The report contains only object_ids, column names, hashes, counts and
metric values; it does not echo raw row payloads.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter, ObjectStorageError
from app.domain import (
    DCR_FORMULA,
    ArtifactRef,
    CandidateArtifactStatus,
    CandidateValidationGate,
    ErrorCode,
    SplitLeakageReport,
    SyntheticDatasetReport,
    ValidationGateFinding,
    ValidationGateMetric,
    ValidationGateSeverity,
    ValidationGatesLineage,
    ValidationGatesReport,
    ValidationGateStatus,
    ValidationGateType,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.plugins.tabular.rules import BusinessRule, BusinessRuleEvaluator

VALIDATION_GATES_REPORT_KIND = "validation_gates_report"
VALIDATION_GATES_REPORT_FORMAT = "json"
VALIDATION_GATES_REPORT_MEDIA_TYPE = "application/json"
VALIDATION_GATES_REPORT_SCHEMA_VERSION = "validation_gates_report.v1"
DEFAULT_VALIDATION_GATES_POLICY_VERSION = "validation_gates_policy_v0"

_BLOCK_EXPORT = "BLOCK_EXPORT"
_BLOCK_MODEL_EVALUATION = "BLOCK_MODEL_EVALUATION"
_BLOCK_TRAINING = "BLOCK_TRAINING"

_IS_SYNTHETIC_COLUMN = "is_synthetic"
_SYNTHETIC_SPLIT_COLUMN = "synthetic_source_split"

# Regex patterns for the privacy check. They mirror the patterns used
# by the privacy logger (TASK-011) and the text/OCR PII plugin so the
# gate is consistent with the rest of the platform. They intentionally
# only flag presence; the matched substrings are never echoed into the
# report.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (
        "phone",
        # Require either a ``+`` country code prefix or at least one
        # non-digit separator (space, dash, parenthesis) so that pure
        # numeric columns like ``monthly_income`` or ``amount`` do not
        # falsely match the phone pattern.
        re.compile(r"(?:\+\d[\d\s\-()]{5,}\d|\d[\d]*[\s\-()][\d\s\-()]{4,}\d)"),
    ),
    (
        "passport_like",
        re.compile(r"\b\d{4}\s\d{6}\b"),
    ),
)


class ValidationGatesError(ValueError):
    """Raised when the gates executor cannot run safely against the inputs."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION_GATE_FAILED,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class DcrThresholds(BaseModel):
    """Thresholds for DCR / nearest-neighbor synthetic gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nearest_neighbor_epsilon: float = Field(default=1e-6, ge=0.0)
    """Synthetic rows closer than this to a real row are treated as
    near-duplicates. The default value is intentionally tight; raise it
    via configuration to enforce a stronger privacy budget.
    """

    minimum_dcr_threshold: float = Field(default=0.0, ge=0.0)
    """Minimum acceptable mean DCR over the synthetic rows. Set above
    zero to require a stronger separation from the real distribution.
    """


class RunValidationGatesRequest(BaseModel):
    """Inputs for :func:`run_validation_gates`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    candidate_artifact: ArtifactRef
    source_artifact: ArtifactRef
    candidate_artifact_kind: NonEmptyStr
    schema_columns: tuple[NonEmptyStr, ...]
    numeric_columns: tuple[NonEmptyStr, ...] = ()
    business_rules: tuple[BusinessRule, ...] = ()
    rules_version: NonEmptyStr = "tabular_business_rules.v1"
    rules_config_hash: Sha256Digest | None = None
    pii_restricted: bool = False
    split_leakage_report: SplitLeakageReport | None = None
    split_leakage_report_artifact: ArtifactRef | None = None
    split_manifest_artifact: ArtifactRef | None = None
    synthetic_dataset_report: SyntheticDatasetReport | None = None
    synthetic_dataset_report_artifact: ArtifactRef | None = None
    dcr_thresholds: DcrThresholds = Field(default_factory=DcrThresholds)
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    policy_version: NonEmptyStr = DEFAULT_VALIDATION_GATES_POLICY_VERSION
    action_plan_id: NonEmptyStr | None = None
    step_id: NonEmptyStr | None = None
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class RunValidationGatesResult:
    """Persisted gates report and its registry record."""

    report: ValidationGatesReport
    report_artifact: RegisteredArtifact


def run_validation_gates(
    request: RunValidationGatesRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> RunValidationGatesResult:
    """Run the validation gates against ``request.candidate_artifact``.

    The function reads the candidate CSV (and the source CSV for
    immutability/DCR checks). It does not modify either artifact.
    """
    candidate_rows, candidate_columns = _read_csv(
        storage=storage, artifact=request.candidate_artifact, role="candidate"
    )
    source_check, raw_unchanged = _raw_artifact_immutability_gate(
        storage=storage, source_artifact=request.source_artifact
    )

    schema_check = _schema_gate(
        request=request,
        candidate_columns=candidate_columns,
        candidate_rows=candidate_rows,
    )
    business_rules_check = _business_rules_gate(
        request=request,
        candidate_rows=candidate_rows,
    )
    privacy_check = _privacy_gate(
        candidate_rows=candidate_rows,
        candidate_columns=candidate_columns,
        pii_restricted=request.pii_restricted,
        numeric_columns=request.numeric_columns,
    )
    split_leakage_check = _split_leakage_gate(
        report=request.split_leakage_report,
    )

    # Synthetic-only gates need both the synthetic report and the source
    # rows for DCR / near-duplicate checks. They resolve to
    # not_applicable when the candidate is not a synthetic artifact.
    synthetic_rows, real_rows = _split_real_and_synthetic_rows(candidate_rows)
    if request.synthetic_dataset_report is None or not synthetic_rows:
        dcr_check = _synthetic_not_applicable_gate(
            ValidationGateType.SYNTHETIC_DCR_CHECK,
            reason="synthetic_artifact_not_provided"
            if request.synthetic_dataset_report is None
            else "candidate_has_no_synthetic_rows",
        )
        exact_duplicate_check = _synthetic_not_applicable_gate(
            ValidationGateType.SYNTHETIC_EXACT_DUPLICATE_TO_REAL,
            reason="synthetic_artifact_not_provided"
            if request.synthetic_dataset_report is None
            else "candidate_has_no_synthetic_rows",
        )
        nearest_check = _synthetic_not_applicable_gate(
            ValidationGateType.SYNTHETIC_NEAREST_NEIGHBOR_PRIVACY,
            reason="synthetic_artifact_not_provided"
            if request.synthetic_dataset_report is None
            else "candidate_has_no_synthetic_rows",
        )
    else:
        # Use real_rows from the candidate as the comparison reference;
        # if the caller supplies numeric columns we restrict DCR to that
        # intersection (categorical features cannot use Euclidean
        # distance). Otherwise fall back to the synthetic report's
        # declared feature columns.
        synthetic_feature_columns = request.synthetic_dataset_report.feature_columns
        if request.numeric_columns:
            feature_columns = tuple(
                column
                for column in synthetic_feature_columns
                if column in set(request.numeric_columns)
            )
            if not feature_columns:
                feature_columns = synthetic_feature_columns
        else:
            feature_columns = synthetic_feature_columns
        dcr_metrics = _compute_dcr(
            synthetic_rows=synthetic_rows,
            real_rows=real_rows,
            feature_columns=feature_columns,
        )
        dcr_check = _dcr_gate(
            metrics=dcr_metrics,
            thresholds=request.dcr_thresholds,
        )
        exact_duplicate_check = _synthetic_exact_duplicate_gate(
            synthetic_rows=synthetic_rows,
            real_rows=real_rows,
            feature_columns=feature_columns,
        )
        nearest_check = _synthetic_nearest_neighbor_privacy_gate(
            metrics=dcr_metrics,
            synthetic_rows=synthetic_rows,
            thresholds=request.dcr_thresholds,
        )

    distribution_similarity = _synthetic_not_applicable_gate(
        ValidationGateType.SYNTHETIC_DISTRIBUTION_SIMILARITY,
        reason="distribution_similarity_pipeline_not_available",
    )
    correlation_preservation = _synthetic_not_applicable_gate(
        ValidationGateType.SYNTHETIC_CORRELATION_PRESERVATION,
        reason="correlation_preservation_pipeline_not_available",
    )
    missingness_comparison = _synthetic_not_applicable_gate(
        ValidationGateType.SYNTHETIC_MISSINGNESS_COMPARISON,
        reason="missingness_comparison_pipeline_not_available",
    )
    tstr_check = _synthetic_not_applicable_gate(
        ValidationGateType.SYNTHETIC_TSTR,
        reason="model_impact_pipeline_not_available",
        notes=(
            "TSTR policy: TSTR_score < baseline_score - threshold => "
            "synthetic_method_status rejected; available once model "
            "impact eligibility lands."
        ),
    )
    trts_check = _synthetic_not_applicable_gate(
        ValidationGateType.SYNTHETIC_TRTS,
        reason="model_impact_pipeline_not_available",
        notes="TRTS_score is unstable => requires_review.",
    )
    shap_check = _synthetic_not_applicable_gate(
        ValidationGateType.SYNTHETIC_SHAP_CONSISTENCY,
        reason="shap_pipeline_not_available",
    )
    tabsyndex_check = _synthetic_not_applicable_gate(
        ValidationGateType.SYNTHETIC_TABSYNDEX,
        reason="tabsyndex_pipeline_not_available",
    )

    gates: tuple[CandidateValidationGate, ...] = (
        schema_check,
        business_rules_check,
        privacy_check,
        split_leakage_check,
        dcr_check,
        exact_duplicate_check,
        nearest_check,
        distribution_similarity,
        correlation_preservation,
        missingness_comparison,
        tstr_check,
        trts_check,
        shap_check,
        tabsyndex_check,
        source_check,
    )

    overall_status = _overall_status(gates)
    blockers = tuple(
        gate.gate_type
        for gate in gates
        if gate.severity is ValidationGateSeverity.BLOCKER
        and gate.status is ValidationGateStatus.FAILED
    )
    block_export = any(
        gate.block_action == _BLOCK_EXPORT for gate in gates if _gate_is_blocking(gate)
    )
    block_model_evaluation = any(
        gate.block_action == _BLOCK_MODEL_EVALUATION for gate in gates if _gate_is_blocking(gate)
    )
    block_training = any(
        gate.block_action == _BLOCK_TRAINING for gate in gates if _gate_is_blocking(gate)
    )

    candidate_status = (
        CandidateArtifactStatus.OK
        if overall_status is not ValidationGateStatus.FAILED and not blockers
        else (
            CandidateArtifactStatus.VALIDATION_FAILED
            if blockers
            else CandidateArtifactStatus.REVIEW_REQUIRED
        )
    )
    if not raw_unchanged:
        candidate_status = CandidateArtifactStatus.VALIDATION_FAILED

    report = ValidationGatesReport(
        report_id=request.report_id or f"validation_gates_report_{uuid.uuid4().hex[:16]}",
        report_schema_version=VALIDATION_GATES_REPORT_SCHEMA_VERSION,
        dataset_id=request.dataset_id,
        source_dataset_version_id=request.source_dataset_version_id,
        candidate_dataset_version_id=request.candidate_dataset_version_id,
        candidate_artifact_kind=request.candidate_artifact_kind,
        policy_version=request.policy_version,
        overall_status=overall_status,
        candidate_status=candidate_status,
        raw_artifact_unchanged=raw_unchanged,
        block_export=block_export or candidate_status is CandidateArtifactStatus.VALIDATION_FAILED,
        block_model_evaluation=block_model_evaluation,
        block_training=block_training,
        blocker_present=bool(blockers),
        blocker_gate_types=blockers,
        gates=gates,
        lineage=ValidationGatesLineage(
            action_plan_id=request.action_plan_id,
            step_id=request.step_id,
            source_dataset_version_id=request.source_dataset_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_artifact=request.source_artifact,
            candidate_artifact=request.candidate_artifact,
            split_manifest=request.split_manifest_artifact,
            split_leakage_report=request.split_leakage_report_artifact,
            synthetic_dataset_report=request.synthetic_dataset_report_artifact,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )

    artifact = registry.save_artifact(
        artifact_kind=VALIDATION_GATES_REPORT_KIND,
        data=_serialize_report(report),
        artifact_format=VALIDATION_GATES_REPORT_FORMAT,
        media_type=VALIDATION_GATES_REPORT_MEDIA_TYPE,
        schema_version=VALIDATION_GATES_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "candidate-artifact-hash": request.candidate_artifact.hash,
            "candidate-artifact-kind": request.candidate_artifact_kind,
            "source-artifact-hash": request.source_artifact.hash,
            "policy-version": request.policy_version,
            "overall-status": overall_status.value,
            "candidate-status": candidate_status.value,
            "raw-artifact-unchanged": "true" if raw_unchanged else "false",
            "block-export": "true" if report.block_export else "false",
            "block-model-evaluation": "true" if block_model_evaluation else "false",
            "block-training": "true" if block_training else "false",
            **(
                {"action-plan-id": request.action_plan_id}
                if request.action_plan_id is not None
                else {}
            ),
            **({"step-id": request.step_id} if request.step_id is not None else {}),
        },
    )
    return RunValidationGatesResult(report=report, report_artifact=artifact)


# ---------------------------------------------------------------------------
# CSV / row helpers
# ---------------------------------------------------------------------------


def _read_csv(
    *,
    storage: MinioObjectStorageAdapter,
    artifact: ArtifactRef,
    role: str,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    try:
        stored = storage.get(artifact.uri)
    except ObjectStorageError as exc:
        raise ValidationGatesError(
            reason_code=f"{role}_artifact_unreadable",
            message=f"Could not read {role} artifact from object storage.",
            details={"uri": artifact.uri},
        ) from exc
    text = stored.data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = tuple(name for name in (reader.fieldnames or ()) if name)
    if not columns:
        raise ValidationGatesError(
            reason_code=f"empty_{role}_csv",
            message=f"{role.capitalize()} CSV has no header columns.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    rows: list[dict[str, str]] = []
    for index, raw_row in enumerate(reader):
        row = {
            column: ("" if raw_row.get(column) is None else str(raw_row.get(column)))
            for column in columns
        }
        if not row.get("object_id"):
            row["object_id"] = f"row_{index:06d}"
        rows.append(row)
    return tuple(rows), columns


def _split_real_and_synthetic_rows(
    rows: Sequence[Mapping[str, str]],
) -> tuple[tuple[Mapping[str, str], ...], tuple[Mapping[str, str], ...]]:
    synthetic = tuple(row for row in rows if row.get(_IS_SYNTHETIC_COLUMN) == "1")
    real = tuple(row for row in rows if row.get(_IS_SYNTHETIC_COLUMN) != "1")
    return synthetic, real


# ---------------------------------------------------------------------------
# Schema gate
# ---------------------------------------------------------------------------


def _schema_gate(
    *,
    request: RunValidationGatesRequest,
    candidate_columns: tuple[str, ...],
    candidate_rows: Sequence[Mapping[str, str]],
) -> CandidateValidationGate:
    declared = set(request.schema_columns)
    actual = set(candidate_columns)
    missing = sorted(declared - actual)
    if missing:
        return CandidateValidationGate(
            gate_type=ValidationGateType.SCHEMA_VALIDATION,
            status=ValidationGateStatus.FAILED,
            severity=ValidationGateSeverity.BLOCKER,
            reason_code="schema_columns_missing",
            block_action=_BLOCK_EXPORT,
            findings_count=len(missing),
            findings=(
                ValidationGateFinding(
                    finding_id="schema_missing_columns",
                    columns=tuple(missing),
                    notes="declared schema columns are absent from candidate CSV",
                ),
            ),
            notes=f"{len(missing)} declared column(s) missing from candidate",
        )
    findings_count = 0
    sample_findings: list[ValidationGateFinding] = []
    for column in request.numeric_columns:
        if column not in actual:
            continue
        for row in candidate_rows:
            value = row.get(column, "")
            if value == "":
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                findings_count += 1
                if len(sample_findings) < 5:
                    sample_findings.append(
                        ValidationGateFinding(
                            finding_id=f"non_numeric_{column}_{row.get('object_id', 'row')}",
                            object_ids=(row.get("object_id", ""),) if row.get("object_id") else (),
                            columns=(column,),
                            notes="non-numeric value in declared numeric column",
                        )
                    )
    if findings_count:
        return CandidateValidationGate(
            gate_type=ValidationGateType.SCHEMA_VALIDATION,
            status=ValidationGateStatus.FAILED,
            severity=ValidationGateSeverity.WARNING,
            reason_code="non_numeric_values_in_numeric_columns",
            block_action=None,
            findings_count=findings_count,
            findings=tuple(sample_findings),
            notes=f"{findings_count} non-numeric value(s) detected in declared numeric columns",
        )
    return CandidateValidationGate(
        gate_type=ValidationGateType.SCHEMA_VALIDATION,
        status=ValidationGateStatus.PASSED,
        severity=ValidationGateSeverity.INFO,
        reason_code="schema_match",
        findings_count=0,
        pass_rate=1.0,
    )


# ---------------------------------------------------------------------------
# Business rules gate
# ---------------------------------------------------------------------------


def _business_rules_gate(
    *,
    request: RunValidationGatesRequest,
    candidate_rows: Sequence[Mapping[str, str]],
) -> CandidateValidationGate:
    if not request.business_rules:
        return CandidateValidationGate(
            gate_type=ValidationGateType.BUSINESS_RULES,
            status=ValidationGateStatus.NOT_APPLICABLE,
            severity=ValidationGateSeverity.INFO,
            reason_code="no_business_rules_configured",
            findings_count=0,
            not_applicable_reason="no business rules supplied for this candidate",
        )
    evaluator = BusinessRuleEvaluator(
        request.business_rules,
        rules_version=request.rules_version,
        rules_config_hash=request.rules_config_hash,
    )
    for row in candidate_rows:
        evaluator.observe_row(row, row.get("object_id"))
    report = evaluator.build_report()
    blocker_ids = report.blocker_candidate_rule_ids
    findings_count = sum(rule.violation_count for rule in report.rules)
    pass_rate = (
        sum(rule.pass_rate for rule in report.rules) / len(report.rules)
        if report.rules
        else 1.0
    )
    sample_findings: list[ValidationGateFinding] = []
    for violation in report.sample_violations[:10]:
        sample_findings.append(
            ValidationGateFinding(
                finding_id=f"rule_{violation.rule_id}_{violation.object_id or 'unknown'}",
                object_ids=(violation.object_id,) if violation.object_id else (),
                columns=(violation.column,) if violation.column else (),
                notes=violation.message,
            )
        )
    if findings_count == 0:
        return CandidateValidationGate(
            gate_type=ValidationGateType.BUSINESS_RULES,
            status=ValidationGateStatus.PASSED,
            severity=ValidationGateSeverity.INFO,
            reason_code="business_rules_passed",
            findings_count=0,
            pass_rate=pass_rate,
        )
    severity = (
        ValidationGateSeverity.BLOCKER if blocker_ids else ValidationGateSeverity.WARNING
    )
    block_action = _BLOCK_EXPORT if blocker_ids else None
    reason_code = "business_rule_failure" if blocker_ids else "business_rule_warning"
    return CandidateValidationGate(
        gate_type=ValidationGateType.BUSINESS_RULES,
        status=ValidationGateStatus.FAILED,
        severity=severity,
        reason_code=reason_code,
        block_action=block_action,
        findings_count=findings_count,
        pass_rate=pass_rate,
        findings=tuple(sample_findings),
        notes=(
            f"{findings_count} row violation(s); {len(blocker_ids)} critical rule(s) failed"
        ),
    )


# ---------------------------------------------------------------------------
# Privacy gate
# ---------------------------------------------------------------------------


def _privacy_gate(
    *,
    candidate_rows: Sequence[Mapping[str, str]],
    candidate_columns: Sequence[str],
    pii_restricted: bool,
    numeric_columns: Sequence[str] = (),
) -> CandidateValidationGate:
    metrics: dict[str, int] = {category: 0 for category, _ in _PII_PATTERNS}
    findings_count = 0
    sample_findings: list[ValidationGateFinding] = []
    numeric_set = set(numeric_columns)
    for row in candidate_rows:
        for column in candidate_columns:
            if column in numeric_set:
                continue
            value = row.get(column, "")
            if not value:
                continue
            for category, pattern in _PII_PATTERNS:
                if pattern.search(value):
                    metrics[category] += 1
                    findings_count += 1
                    if len(sample_findings) < 5:
                        sample_findings.append(
                            ValidationGateFinding(
                                finding_id=(
                                    f"pii_{category}_"
                                    f"{row.get('object_id', 'row')}_{column}"
                                ),
                                object_ids=(
                                    (row.get("object_id", ""),)
                                    if row.get("object_id")
                                    else ()
                                ),
                                columns=(column,),
                                notes=f"{category} pattern matched in {column}",
                            )
                        )
                    break  # one match per cell is enough
    if findings_count == 0:
        return CandidateValidationGate(
            gate_type=ValidationGateType.PRIVACY_CHECK,
            status=ValidationGateStatus.PASSED,
            severity=ValidationGateSeverity.INFO,
            reason_code="no_pii_detected",
            findings_count=0,
            pass_rate=1.0,
            metrics=tuple(
                ValidationGateMetric(name=name, value=float(value))
                for name, value in metrics.items()
            ),
        )
    if pii_restricted:
        return CandidateValidationGate(
            gate_type=ValidationGateType.PRIVACY_CHECK,
            status=ValidationGateStatus.FAILED,
            severity=ValidationGateSeverity.BLOCKER,
            reason_code="pii_in_restricted_candidate",
            block_action=_BLOCK_EXPORT,
            findings_count=findings_count,
            metrics=tuple(
                ValidationGateMetric(name=name, value=float(value))
                for name, value in metrics.items()
            ),
            findings=tuple(sample_findings),
            notes=f"{findings_count} PII-like token(s) found in restricted candidate",
        )
    return CandidateValidationGate(
        gate_type=ValidationGateType.PRIVACY_CHECK,
        status=ValidationGateStatus.FAILED,
        severity=ValidationGateSeverity.WARNING,
        reason_code="pii_detected",
        block_action=None,
        findings_count=findings_count,
        metrics=tuple(
            ValidationGateMetric(name=name, value=float(value))
            for name, value in metrics.items()
        ),
        findings=tuple(sample_findings),
        notes=f"{findings_count} PII-like token(s) found",
    )


# ---------------------------------------------------------------------------
# Split leakage gate
# ---------------------------------------------------------------------------


def _split_leakage_gate(
    *,
    report: SplitLeakageReport | None,
) -> CandidateValidationGate:
    if report is None:
        return CandidateValidationGate(
            gate_type=ValidationGateType.SPLIT_LEAKAGE_CHECK,
            status=ValidationGateStatus.NOT_APPLICABLE,
            severity=ValidationGateSeverity.INFO,
            reason_code="split_leakage_report_not_provided",
            findings_count=0,
            not_applicable_reason="candidate is not a split-aware artifact",
        )
    if not report.leakage_detected:
        return CandidateValidationGate(
            gate_type=ValidationGateType.SPLIT_LEAKAGE_CHECK,
            status=ValidationGateStatus.PASSED,
            severity=ValidationGateSeverity.INFO,
            reason_code="no_split_leakage",
            findings_count=0,
            pass_rate=1.0,
        )
    block_action = (
        _BLOCK_TRAINING
        if report.block_training
        else (_BLOCK_MODEL_EVALUATION if report.block_model_evaluation else None)
    )
    return CandidateValidationGate(
        gate_type=ValidationGateType.SPLIT_LEAKAGE_CHECK,
        status=ValidationGateStatus.FAILED,
        severity=ValidationGateSeverity.BLOCKER if block_action else ValidationGateSeverity.WARNING,
        reason_code="split_leakage_detected",
        block_action=block_action,
        findings_count=sum(check.findings_count for check in report.checks),
        notes=(
            "split leakage report flagged "
            f"{sum(1 for check in report.checks if check.status.value == 'failed')} check(s)"
        ),
        metrics=(
            ValidationGateMetric(
                name="leakage_risk_score",
                value=float(report.leakage_risk_score),
                threshold=0.5,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic gates
# ---------------------------------------------------------------------------


def _synthetic_not_applicable_gate(
    gate_type: ValidationGateType,
    *,
    reason: str,
    notes: str | None = None,
) -> CandidateValidationGate:
    return CandidateValidationGate(
        gate_type=gate_type,
        status=ValidationGateStatus.NOT_APPLICABLE,
        severity=ValidationGateSeverity.INFO,
        reason_code=reason,
        findings_count=0,
        not_applicable_reason=reason,
        notes=notes,
    )


@dataclass(frozen=True)
class _DcrMetrics:
    distances: tuple[float, ...]
    minimum: float
    mean: float
    median: float
    below_epsilon: int


def _compute_dcr(
    *,
    synthetic_rows: Sequence[Mapping[str, str]],
    real_rows: Sequence[Mapping[str, str]],
    feature_columns: Sequence[str],
) -> _DcrMetrics:
    """Compute DCR metrics using the formula

    ``DCR(x_synth) = min_{x_real ∈ D_real} distance(x_synth, x_real)``

    where ``distance`` is the Euclidean distance over the configured
    feature columns. Non-numeric features are skipped by the parsing
    step and contribute zero to the squared distance to keep the metric
    well-defined when the synthetic generator could not emit a number.
    """
    feature_columns = tuple(feature_columns)
    real_vectors = [
        _row_vector(row, feature_columns) for row in real_rows
    ]
    distances: list[float] = []
    below = 0
    for synthetic_row in synthetic_rows:
        synthetic_vector = _row_vector(synthetic_row, feature_columns)
        nearest = math.inf
        for real_vector in real_vectors:
            distance = _euclidean(synthetic_vector, real_vector)
            if distance < nearest:
                nearest = distance
                if nearest == 0.0:
                    break
        if math.isinf(nearest):
            nearest = 0.0
        distances.append(nearest)
        if nearest <= 1e-9:
            below += 1
    if not distances:
        return _DcrMetrics(
            distances=(),
            minimum=0.0,
            mean=0.0,
            median=0.0,
            below_epsilon=0,
        )
    sorted_distances = sorted(distances)
    midpoint = len(sorted_distances) // 2
    if len(sorted_distances) % 2 == 1:
        median = sorted_distances[midpoint]
    else:
        median = (sorted_distances[midpoint - 1] + sorted_distances[midpoint]) / 2
    return _DcrMetrics(
        distances=tuple(distances),
        minimum=min(distances),
        mean=sum(distances) / len(distances),
        median=median,
        below_epsilon=below,
    )


def _dcr_gate(
    *,
    metrics: _DcrMetrics,
    thresholds: DcrThresholds,
) -> CandidateValidationGate:
    base_metrics = (
        ValidationGateMetric(
            name="dcr_min",
            value=metrics.minimum,
            threshold=thresholds.minimum_dcr_threshold,
            formula=DCR_FORMULA,
        ),
        ValidationGateMetric(name="dcr_mean", value=metrics.mean, formula=DCR_FORMULA),
        ValidationGateMetric(name="dcr_median", value=metrics.median, formula=DCR_FORMULA),
        ValidationGateMetric(
            name="below_epsilon_count",
            value=float(metrics.below_epsilon),
            threshold=0.0,
        ),
    )
    if metrics.minimum < thresholds.minimum_dcr_threshold or metrics.below_epsilon > 0:
        return CandidateValidationGate(
            gate_type=ValidationGateType.SYNTHETIC_DCR_CHECK,
            status=ValidationGateStatus.FAILED,
            severity=ValidationGateSeverity.BLOCKER,
            reason_code="dcr_threshold_violated",
            block_action=_BLOCK_EXPORT,
            findings_count=max(metrics.below_epsilon, 1),
            metrics=base_metrics,
            notes=(
                f"min DCR {metrics.minimum:.6g} below threshold "
                f"{thresholds.minimum_dcr_threshold:.6g}"
                if metrics.minimum < thresholds.minimum_dcr_threshold
                else f"{metrics.below_epsilon} synthetic row(s) collide with real rows"
            ),
        )
    return CandidateValidationGate(
        gate_type=ValidationGateType.SYNTHETIC_DCR_CHECK,
        status=ValidationGateStatus.PASSED,
        severity=ValidationGateSeverity.INFO,
        reason_code="dcr_passed",
        findings_count=0,
        pass_rate=1.0,
        metrics=base_metrics,
    )


def _synthetic_exact_duplicate_gate(
    *,
    synthetic_rows: Sequence[Mapping[str, str]],
    real_rows: Sequence[Mapping[str, str]],
    feature_columns: Sequence[str],
) -> CandidateValidationGate:
    real_signatures = {
        _row_signature(row, feature_columns) for row in real_rows
    }
    findings: list[ValidationGateFinding] = []
    for row in synthetic_rows:
        signature = _row_signature(row, feature_columns)
        if signature in real_signatures and len(findings) < 10:
            findings.append(
                ValidationGateFinding(
                    finding_id=f"exact_dup_{row.get('object_id', signature[:12])}",
                    object_ids=(row.get("object_id", ""),) if row.get("object_id") else (),
                    notes="synthetic row reproduces a real row exactly",
                )
            )
    findings_count = sum(
        1
        for row in synthetic_rows
        if _row_signature(row, feature_columns) in real_signatures
    )
    if findings_count == 0:
        return CandidateValidationGate(
            gate_type=ValidationGateType.SYNTHETIC_EXACT_DUPLICATE_TO_REAL,
            status=ValidationGateStatus.PASSED,
            severity=ValidationGateSeverity.INFO,
            reason_code="no_exact_duplicate_to_real",
            findings_count=0,
            pass_rate=1.0,
        )
    return CandidateValidationGate(
        gate_type=ValidationGateType.SYNTHETIC_EXACT_DUPLICATE_TO_REAL,
        status=ValidationGateStatus.FAILED,
        severity=ValidationGateSeverity.BLOCKER,
        reason_code="synthetic_exact_duplicate_to_real",
        block_action=_BLOCK_EXPORT,
        findings_count=findings_count,
        findings=tuple(findings),
        notes=f"{findings_count} synthetic row(s) duplicate a real row",
    )


def _synthetic_nearest_neighbor_privacy_gate(
    *,
    metrics: _DcrMetrics,
    synthetic_rows: Sequence[Mapping[str, str]],
    thresholds: DcrThresholds,
) -> CandidateValidationGate:
    findings: list[ValidationGateFinding] = []
    findings_count = 0
    for distance, row in zip(metrics.distances, synthetic_rows, strict=True):
        if distance <= thresholds.nearest_neighbor_epsilon:
            findings_count += 1
            if len(findings) < 5:
                findings.append(
                    ValidationGateFinding(
                        finding_id=f"near_neighbor_{row.get('object_id', 'row')}",
                        object_ids=(
                            (row.get("object_id", ""),)
                            if row.get("object_id")
                            else ()
                        ),
                        notes=(
                            f"distance {distance:.6g} ≤ epsilon "
                            f"{thresholds.nearest_neighbor_epsilon:.6g}"
                        ),
                        metric_name="nearest_neighbor_distance",
                        metric_value=distance,
                        metric_threshold=thresholds.nearest_neighbor_epsilon,
                    )
                )
    if findings_count == 0:
        return CandidateValidationGate(
            gate_type=ValidationGateType.SYNTHETIC_NEAREST_NEIGHBOR_PRIVACY,
            status=ValidationGateStatus.PASSED,
            severity=ValidationGateSeverity.INFO,
            reason_code="nearest_neighbor_privacy_ok",
            findings_count=0,
            pass_rate=1.0,
            metrics=(
                ValidationGateMetric(
                    name="nearest_neighbor_epsilon",
                    value=thresholds.nearest_neighbor_epsilon,
                ),
            ),
        )
    return CandidateValidationGate(
        gate_type=ValidationGateType.SYNTHETIC_NEAREST_NEIGHBOR_PRIVACY,
        status=ValidationGateStatus.FAILED,
        severity=ValidationGateSeverity.BLOCKER,
        reason_code="synthetic_too_close_to_real",
        block_action=_BLOCK_EXPORT,
        findings_count=findings_count,
        metrics=(
            ValidationGateMetric(
                name="nearest_neighbor_epsilon",
                value=thresholds.nearest_neighbor_epsilon,
            ),
            ValidationGateMetric(
                name="below_epsilon_count",
                value=float(findings_count),
            ),
        ),
        findings=tuple(findings),
        notes=f"{findings_count} synthetic row(s) below nearest-neighbor epsilon",
    )


# ---------------------------------------------------------------------------
# Raw artifact immutability gate
# ---------------------------------------------------------------------------


def _raw_artifact_immutability_gate(
    *,
    storage: MinioObjectStorageAdapter,
    source_artifact: ArtifactRef,
) -> tuple[CandidateValidationGate, bool]:
    """Re-read the source artifact and verify its sha256 is unchanged.

    The gate fails closed when the source cannot be re-read or when the
    hash drifted. Returning ``raw_unchanged=False`` causes the report to
    flip the candidate into ``validation_failed`` and block export.
    """
    try:
        stored = storage.get(source_artifact.uri)
    except ObjectStorageError:
        return (
            CandidateValidationGate(
                gate_type=ValidationGateType.RAW_ARTIFACT_IMMUTABILITY,
                status=ValidationGateStatus.FAILED,
                severity=ValidationGateSeverity.BLOCKER,
                reason_code="source_artifact_unreadable",
                block_action=_BLOCK_EXPORT,
                findings_count=1,
                notes="source artifact could not be re-read from storage",
            ),
            False,
        )
    digest = _sha256(stored.data)
    if digest != source_artifact.hash:
        return (
            CandidateValidationGate(
                gate_type=ValidationGateType.RAW_ARTIFACT_IMMUTABILITY,
                status=ValidationGateStatus.FAILED,
                severity=ValidationGateSeverity.BLOCKER,
                reason_code="raw_artifact_mutated",
                block_action=_BLOCK_EXPORT,
                findings_count=1,
                metrics=(
                    ValidationGateMetric(
                        name="expected_hash",
                        value=None,
                        notes=source_artifact.hash,
                    ),
                    ValidationGateMetric(
                        name="actual_hash",
                        value=None,
                        notes=digest,
                    ),
                ),
                notes="source artifact sha256 has drifted since action execution",
            ),
            False,
        )
    return (
        CandidateValidationGate(
            gate_type=ValidationGateType.RAW_ARTIFACT_IMMUTABILITY,
            status=ValidationGateStatus.PASSED,
            severity=ValidationGateSeverity.INFO,
            reason_code="raw_artifact_unchanged",
            findings_count=0,
            pass_rate=1.0,
        ),
        True,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _overall_status(gates: Iterable[CandidateValidationGate]) -> ValidationGateStatus:
    saw_failed = False
    saw_applicable = False
    for gate in gates:
        if gate.status is ValidationGateStatus.NOT_APPLICABLE:
            continue
        saw_applicable = True
        if gate.status is ValidationGateStatus.FAILED:
            saw_failed = True
    if not saw_applicable:
        return ValidationGateStatus.NOT_APPLICABLE
    return ValidationGateStatus.FAILED if saw_failed else ValidationGateStatus.PASSED


def _gate_is_blocking(gate: CandidateValidationGate) -> bool:
    return (
        gate.status is ValidationGateStatus.FAILED
        and gate.severity is ValidationGateSeverity.BLOCKER
    )


def _row_vector(row: Mapping[str, str], feature_columns: Iterable[str]) -> tuple[float, ...]:
    vector: list[float] = []
    for column in feature_columns:
        raw = row.get(column, "")
        try:
            vector.append(float(raw))
        except (TypeError, ValueError):
            vector.append(0.0)
    return tuple(vector)


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        # Defensive check; should not happen given consistent feature
        # column lists. We return 0.0 rather than raising so the gate
        # surfaces a finding instead of crashing the workflow.
        return 0.0
    total = 0.0
    for left, right in zip(a, b, strict=True):
        delta = left - right
        total += delta * delta
    return math.sqrt(total)


def _row_signature(row: Mapping[str, str], feature_columns: Iterable[str]) -> str:
    payload = {column: row.get(column, "") for column in sorted(feature_columns)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _serialize_report(report: ValidationGatesReport) -> bytes:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2).encode("utf-8")


__all__ = [
    "DEFAULT_VALIDATION_GATES_POLICY_VERSION",
    "DcrThresholds",
    "RunValidationGatesRequest",
    "RunValidationGatesResult",
    "VALIDATION_GATES_REPORT_FORMAT",
    "VALIDATION_GATES_REPORT_KIND",
    "VALIDATION_GATES_REPORT_MEDIA_TYPE",
    "VALIDATION_GATES_REPORT_SCHEMA_VERSION",
    "ValidationGatesError",
    "run_validation_gates",
]
