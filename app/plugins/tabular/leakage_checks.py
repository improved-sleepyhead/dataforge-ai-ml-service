"""Post-split leakage checks for supervised tabular classification.

The executor consumes:

- the immutable source CSV referenced by the split manifest;
- the persisted ``SplitManifest`` artifact;
- an optional ``TabularProfileReport`` carrying ``LeakageDiagnostics`` from
  the analyze pass (used for the target leakage candidate scan).

It produces a contract-valid ``split_leakage_report`` artifact recording
the outcome of three MVP checks: exact hash leakage, group key leakage,
and target leakage candidate scan. Cross-split duplicates and
group-key collisions block ``BLOCK_MODEL_EVALUATION``; high-target-match
leakage candidates block ``BLOCK_TRAINING``.

The module is intentionally stdlib-only and never echoes raw row
payloads. It records ``object_id`` values, group keys, column names, and
counts only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ArtifactRef,
    DataSplit,
    ErrorCode,
    LeakageCheckResult,
    LeakageCheckSeverity,
    LeakageCheckStatus,
    LeakageCheckType,
    LeakageCrossSplitFinding,
    SplitLeakageLineage,
    SplitLeakageReport,
    SplitManifest,
    TabularProfileReport,
)
from app.domain.common import NonEmptyStr, Sha256Digest

SPLIT_LEAKAGE_REPORT_KIND = "split_leakage_report"
SPLIT_LEAKAGE_REPORT_FORMAT = "json"
SPLIT_LEAKAGE_REPORT_MEDIA_TYPE = "application/json"
SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION = "split_leakage_report.v1"
DEFAULT_SPLIT_LEAKAGE_POLICY_VERSION = "split_leakage_policy_v0"

_BLOCK_MODEL_EVALUATION = "BLOCK_MODEL_EVALUATION"
_BLOCK_TRAINING = "BLOCK_TRAINING"

# Reason codes follow the Decision Core reason-code registry. The split
# leakage gate uses ``split_leakage`` for cross-split duplicates and
# group-key collisions; ``target_leakage_candidate`` is reused for
# target-like feature scans so the existing hard gate consumes the
# report without remapping.
_REASON_NO_EXACT_HASH = "no_exact_hash_leakage"
_REASON_NO_GROUP_KEY_LEAKAGE = "no_group_key_leakage"
_REASON_NO_TARGET_LEAKAGE = "no_target_leakage_candidate"
_REASON_GROUP_KEY_NOT_APPLICABLE = "group_key_not_configured"
_REASON_TARGET_LEAKAGE_NOT_APPLICABLE = "tabular_profile_report_not_provided"

_REASON_EXACT_HASH_LEAKAGE = "split_leakage"
_REASON_GROUP_KEY_LEAKAGE = "split_leakage"
_REASON_TARGET_LEAKAGE_CANDIDATE = "target_leakage_candidate"

# We block on target leakage candidates that have either a confirmed
# high target match rate (set in TASK-026) or a name-pattern hit when no
# numeric match rate is available. Pure name-pattern hits without target
# match data fall back to a warning, mirroring the ``warning by policy``
# clause from the PRD §7.5.4 / §3 acceptance criterion.
_HIGH_TARGET_MATCH_REASON = "high_target_match_rate"


class SplitLeakageCheckError(ValueError):
    """Raised when leakage checks cannot run safely against the inputs."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.LEAKAGE_DETECTED,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class RunSplitLeakageChecksRequest(BaseModel):
    """Inputs for the post-split leakage check executor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    split_manifest: SplitManifest
    split_manifest_artifact: ArtifactRef
    source_artifact: ArtifactRef
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    policy_version: NonEmptyStr = DEFAULT_SPLIT_LEAKAGE_POLICY_VERSION
    tabular_profile_report: TabularProfileReport | None = None
    action_plan_id: NonEmptyStr | None = None
    step_id: NonEmptyStr | None = None
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class RunSplitLeakageChecksResult:
    """Persisted leakage report and its registry record."""

    report: SplitLeakageReport
    report_artifact: RegisteredArtifact


def run_split_leakage_checks(
    request: RunSplitLeakageChecksRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> RunSplitLeakageChecksResult:
    """Run MVP post-split leakage checks and persist the resulting report."""
    manifest = request.split_manifest
    if not manifest.assignments:
        raise SplitLeakageCheckError(
            reason_code="empty_split_manifest",
            message="Split manifest does not contain any assignments.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    rows = _read_rows(
        storage=storage,
        source_artifact=request.source_artifact,
        target_column=manifest.target_column,
    )
    assignments = _index_assignments(manifest)
    _verify_assignments_match_rows(assignments=assignments, rows=rows)

    exact_hash_check = _exact_hash_check(
        rows=rows,
        assignments=assignments,
        target_column=manifest.target_column,
    )
    group_key_check = _group_key_check(
        rows=rows,
        assignments=assignments,
        group_key=manifest.group_key,
    )
    target_leakage_check = _target_leakage_candidate_check(
        profile=request.tabular_profile_report,
        target_column=manifest.target_column,
        manifest=manifest,
    )

    checks = (exact_hash_check, group_key_check, target_leakage_check)
    leakage_detected = any(
        check.status is LeakageCheckStatus.FAILED
        and check.severity is LeakageCheckSeverity.BLOCKER
        for check in checks
    )
    block_model_evaluation = any(
        check.status is LeakageCheckStatus.FAILED
        and check.severity is LeakageCheckSeverity.BLOCKER
        and check.block_action == _BLOCK_MODEL_EVALUATION
        for check in checks
    )
    block_training = any(
        check.status is LeakageCheckStatus.FAILED
        and check.severity is LeakageCheckSeverity.BLOCKER
        and check.block_action == _BLOCK_TRAINING
        for check in checks
    )

    leakage_risk_score = _leakage_risk_score(checks)

    report = SplitLeakageReport(
        report_id=request.report_id or f"split_leakage_report_{uuid.uuid4().hex[:16]}",
        report_schema_version=SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION,
        dataset_id=request.dataset_id,
        source_dataset_version_id=request.source_dataset_version_id,
        candidate_dataset_version_id=request.candidate_dataset_version_id,
        split_manifest_id=manifest.split_manifest_id,
        target_column=manifest.target_column,
        policy_version=request.policy_version,
        leakage_detected=leakage_detected,
        leakage_risk_score=leakage_risk_score,
        block_model_evaluation=block_model_evaluation,
        block_training=block_training,
        checks=checks,
        lineage=SplitLeakageLineage(
            action_plan_id=request.action_plan_id,
            step_id=request.step_id,
            source_dataset_version_id=request.source_dataset_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_artifact=request.source_artifact,
            split_manifest=request.split_manifest_artifact,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )

    artifact = registry.save_artifact(
        artifact_kind=SPLIT_LEAKAGE_REPORT_KIND,
        data=_serialize_report(report),
        artifact_format=SPLIT_LEAKAGE_REPORT_FORMAT,
        media_type=SPLIT_LEAKAGE_REPORT_MEDIA_TYPE,
        schema_version=SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "split-manifest-id": manifest.split_manifest_id,
            "split-manifest-hash": request.split_manifest_artifact.hash,
            "source-artifact-hash": request.source_artifact.hash,
            "target-column": manifest.target_column,
            "policy-version": request.policy_version,
            "leakage-detected": "true" if leakage_detected else "false",
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
    return RunSplitLeakageChecksResult(report=report, report_artifact=artifact)


def _read_rows(
    *,
    storage: MinioObjectStorageAdapter,
    source_artifact: ArtifactRef,
    target_column: str,
) -> tuple[dict[str, str], ...]:
    stored = storage.get(source_artifact.uri)
    text = stored.data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = tuple(name for name in (reader.fieldnames or ()) if name)
    if not columns:
        raise SplitLeakageCheckError(
            reason_code="empty_tabular_source",
            message="Source CSV has no header columns.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    if target_column not in columns:
        raise SplitLeakageCheckError(
            reason_code="target_column_not_found",
            message="Split target column is not present in source CSV.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            details={"target_column": target_column},
        )
    rows: list[dict[str, str]] = []
    for index, raw_row in enumerate(reader):
        row = {column: "" if raw_row.get(column) is None else str(raw_row.get(column))
               for column in columns}
        if not row.get("object_id"):
            row["object_id"] = f"row_{index:06d}"
        rows.append(row)
    if not rows:
        raise SplitLeakageCheckError(
            reason_code="empty_tabular_source",
            message="Source CSV has no data rows.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    return tuple(rows)


def _index_assignments(manifest: SplitManifest) -> dict[str, tuple[DataSplit, str | None]]:
    return {
        assignment.object_id: (assignment.split, assignment.group_value)
        for assignment in manifest.assignments
    }


def _verify_assignments_match_rows(
    *,
    assignments: Mapping[str, tuple[DataSplit, str | None]],
    rows: Iterable[Mapping[str, str]],
) -> None:
    row_ids = {row["object_id"] for row in rows}
    missing_in_rows = sorted(set(assignments) - row_ids)
    if missing_in_rows:
        raise SplitLeakageCheckError(
            reason_code="split_manifest_object_id_missing_in_source",
            message="Split manifest references object_ids that are not present in the source CSV.",
            details={"missing_object_ids": missing_in_rows[:5]},
        )


def _exact_hash_check(
    *,
    rows: tuple[dict[str, str], ...],
    assignments: Mapping[str, tuple[DataSplit, str | None]],
    target_column: str,
) -> LeakageCheckResult:
    """Detect exact row-content duplicates that span multiple splits.

    The signature excludes ``object_id`` so that exact duplicates of the
    same row content created by upstream ingestion (different row keys
    but identical payload) are still detected. The target column is
    included because identical (features, label) rows in two splits are
    the leakage scenario the PRD calls out.
    """
    signature_to_splits: dict[str, dict[DataSplit, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        object_id = row["object_id"]
        if object_id not in assignments:
            continue
        split, _ = assignments[object_id]
        signature = _row_signature(row, target_column=target_column)
        signature_to_splits[signature][split].append(object_id)

    findings: list[LeakageCrossSplitFinding] = []
    affected_object_ids: set[str] = set()
    for index, (signature, by_split) in enumerate(sorted(signature_to_splits.items())):
        if len(by_split) < 2:
            continue
        finding_object_ids = sorted(
            object_id for ids in by_split.values() for object_id in ids
        )
        affected_object_ids.update(finding_object_ids)
        findings.append(
            LeakageCrossSplitFinding(
                finding_id=f"exact_hash_{index:04d}_{signature[:16]}",
                splits=tuple(sorted(split.value for split in by_split)),
                object_ids=tuple(finding_object_ids),
                group_value=None,
                column=None,
                notes="row content identical across splits",
            )
        )

    if not findings:
        return LeakageCheckResult(
            check_type=LeakageCheckType.EXACT_HASH,
            status=LeakageCheckStatus.PASSED,
            severity=LeakageCheckSeverity.INFO,
            reason_code=_REASON_NO_EXACT_HASH,
            block_action=None,
            findings_count=0,
            notes="no exact row content overlap across splits",
        )
    return LeakageCheckResult(
        check_type=LeakageCheckType.EXACT_HASH,
        status=LeakageCheckStatus.FAILED,
        severity=LeakageCheckSeverity.BLOCKER,
        reason_code=_REASON_EXACT_HASH_LEAKAGE,
        block_action=_BLOCK_MODEL_EVALUATION,
        findings_count=len(findings),
        affected_object_ids=tuple(sorted(affected_object_ids)),
        findings=tuple(findings),
        notes=f"{len(findings)} row signature(s) span multiple splits",
    )


def _group_key_check(
    *,
    rows: tuple[dict[str, str], ...],
    assignments: Mapping[str, tuple[DataSplit, str | None]],
    group_key: str | None,
) -> LeakageCheckResult:
    """Detect group-key values that appear in more than one split."""
    if group_key is None:
        return LeakageCheckResult(
            check_type=LeakageCheckType.GROUP_KEY,
            status=LeakageCheckStatus.NOT_APPLICABLE,
            severity=LeakageCheckSeverity.INFO,
            reason_code=_REASON_GROUP_KEY_NOT_APPLICABLE,
            block_action=None,
            findings_count=0,
            notes="split manifest has no group_key configured",
        )

    group_to_splits: dict[str, dict[DataSplit, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        object_id = row["object_id"]
        if object_id not in assignments:
            continue
        split, _assigned_group = assignments[object_id]
        # Source-of-truth for the group value is the row content. The
        # split assignment's group_value is recorded for transparency
        # but the row's actual value is what could leak across splits.
        group_value = row.get(group_key, "")
        if not group_value:
            continue
        group_to_splits[group_value][split].append(object_id)

    findings: list[LeakageCrossSplitFinding] = []
    affected_object_ids: set[str] = set()
    affected_groups: set[str] = set()
    for group_value, by_split in sorted(group_to_splits.items()):
        if len(by_split) < 2:
            continue
        finding_object_ids = sorted(
            object_id for ids in by_split.values() for object_id in ids
        )
        affected_object_ids.update(finding_object_ids)
        affected_groups.add(group_value)
        findings.append(
            LeakageCrossSplitFinding(
                finding_id=f"group_leakage_{group_value}",
                splits=tuple(sorted(split.value for split in by_split)),
                object_ids=tuple(finding_object_ids),
                group_value=group_value,
                column=group_key,
                notes=f"group key appears in {len(by_split)} splits",
            )
        )

    if not findings:
        return LeakageCheckResult(
            check_type=LeakageCheckType.GROUP_KEY,
            status=LeakageCheckStatus.PASSED,
            severity=LeakageCheckSeverity.INFO,
            reason_code=_REASON_NO_GROUP_KEY_LEAKAGE,
            block_action=None,
            findings_count=0,
            affected_columns=(group_key,),
            notes=f"all {group_key} values stay in a single split",
        )
    return LeakageCheckResult(
        check_type=LeakageCheckType.GROUP_KEY,
        status=LeakageCheckStatus.FAILED,
        severity=LeakageCheckSeverity.BLOCKER,
        reason_code=_REASON_GROUP_KEY_LEAKAGE,
        block_action=_BLOCK_MODEL_EVALUATION,
        findings_count=len(findings),
        affected_object_ids=tuple(sorted(affected_object_ids)),
        affected_groups=tuple(sorted(affected_groups)),
        affected_columns=(group_key,),
        findings=tuple(findings),
        notes=f"{len(findings)} {group_key} value(s) span multiple splits",
    )


def _target_leakage_candidate_check(
    *,
    profile: TabularProfileReport | None,
    target_column: str,
    manifest: SplitManifest,
) -> LeakageCheckResult:
    """Map TabularProfileReport leakage candidates to a hard gate result.

    The check is a normalized read of the analyze-pass leakage
    diagnostics. It does not re-scan the dataset, so the report stays
    deterministic and avoids double work.
    """
    if profile is None or profile.leakage is None or not profile.leakage.candidates:
        return LeakageCheckResult(
            check_type=LeakageCheckType.TARGET_LEAKAGE_CANDIDATE,
            status=LeakageCheckStatus.NOT_APPLICABLE if profile is None
            else LeakageCheckStatus.PASSED,
            severity=LeakageCheckSeverity.INFO,
            reason_code=(
                _REASON_TARGET_LEAKAGE_NOT_APPLICABLE
                if profile is None
                else _REASON_NO_TARGET_LEAKAGE
            ),
            block_action=None,
            findings_count=0,
            notes=(
                "no tabular profile report supplied"
                if profile is None
                else "no leakage candidates in tabular profile"
            ),
        )

    blocker_candidates = [
        candidate
        for candidate in profile.leakage.candidates
        if candidate.reason_code == _HIGH_TARGET_MATCH_REASON
    ]
    warning_candidates = [
        candidate
        for candidate in profile.leakage.candidates
        if candidate.reason_code != _HIGH_TARGET_MATCH_REASON
    ]

    if not blocker_candidates and not warning_candidates:
        return LeakageCheckResult(
            check_type=LeakageCheckType.TARGET_LEAKAGE_CANDIDATE,
            status=LeakageCheckStatus.PASSED,
            severity=LeakageCheckSeverity.INFO,
            reason_code=_REASON_NO_TARGET_LEAKAGE,
            block_action=None,
            findings_count=0,
        )

    splits = tuple(sorted(split.value for split in _splits_in_manifest(manifest)))
    findings: list[LeakageCrossSplitFinding] = []
    affected_columns: list[str] = []
    if blocker_candidates:
        for candidate in blocker_candidates:
            findings.append(
                LeakageCrossSplitFinding(
                    finding_id=f"target_leakage_{candidate.column}",
                    splits=splits or ("train", "test"),
                    object_ids=(),
                    group_value=None,
                    column=candidate.column,
                    notes=candidate.reason_code,
                )
            )
            affected_columns.append(candidate.column)
        return LeakageCheckResult(
            check_type=LeakageCheckType.TARGET_LEAKAGE_CANDIDATE,
            status=LeakageCheckStatus.FAILED,
            severity=LeakageCheckSeverity.BLOCKER,
            reason_code=_REASON_TARGET_LEAKAGE_CANDIDATE,
            block_action=_BLOCK_TRAINING,
            findings_count=len(findings),
            affected_columns=tuple(sorted(set(affected_columns))),
            findings=tuple(findings),
            notes=(
                f"{len(blocker_candidates)} column(s) flagged with "
                f"{_HIGH_TARGET_MATCH_REASON}"
            ),
        )

    # Warning-only path: name-pattern match without high target match
    # rate. We surface the candidates so the validation gate UI can
    # warn but allow training to proceed by default.
    for candidate in warning_candidates:
        findings.append(
            LeakageCrossSplitFinding(
                finding_id=f"target_leakage_warning_{candidate.column}",
                splits=splits or ("train", "test"),
                object_ids=(),
                group_value=None,
                column=candidate.column,
                notes=candidate.reason_code,
            )
        )
        affected_columns.append(candidate.column)
    _ = target_column  # documented input retained for future rule evolution
    return LeakageCheckResult(
        check_type=LeakageCheckType.TARGET_LEAKAGE_CANDIDATE,
        status=LeakageCheckStatus.FAILED,
        severity=LeakageCheckSeverity.WARNING,
        reason_code=_REASON_TARGET_LEAKAGE_CANDIDATE,
        block_action=None,
        findings_count=len(findings),
        affected_columns=tuple(sorted(set(affected_columns))),
        findings=tuple(findings),
        notes=(
            f"{len(warning_candidates)} target leakage candidate(s) without "
            f"confirmed target match rate"
        ),
    )


def _row_signature(row: Mapping[str, str], *, target_column: str) -> str:
    """Return a stable sha256 over the row content excluding object_id.

    The target column is included so identical (features, label) tuples
    are flagged. The hash is computed over a canonical JSON of sorted
    column names so column order changes do not affect the result.
    """
    payload = {
        column: row.get(column, "")
        for column in sorted(row)
        if column != "object_id"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    # ``target_column`` is referenced as a defensive guard: callers must
    # provide it so that future signature variants can salt by target
    # column name without breaking determinism.
    _ = target_column
    return digest


def _splits_in_manifest(manifest: SplitManifest) -> tuple[DataSplit, ...]:
    return tuple(
        sorted(
            {assignment.split for assignment in manifest.assignments},
            key=lambda split: split.value,
        )
    )


def _leakage_risk_score(checks: Iterable[LeakageCheckResult]) -> float:
    """Compute leakage_risk_score in [0, 1] from per-check severities."""
    score = 0.0
    for check in checks:
        if check.status is not LeakageCheckStatus.FAILED:
            continue
        if check.severity is LeakageCheckSeverity.BLOCKER:
            return 1.0
        if check.severity is LeakageCheckSeverity.WARNING:
            score = max(score, 0.5)
        else:
            score = max(score, 0.25)
    return score


def _serialize_report(report: SplitLeakageReport) -> bytes:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2).encode("utf-8")


__all__ = [
    "DEFAULT_SPLIT_LEAKAGE_POLICY_VERSION",
    "RunSplitLeakageChecksRequest",
    "RunSplitLeakageChecksResult",
    "SPLIT_LEAKAGE_REPORT_FORMAT",
    "SPLIT_LEAKAGE_REPORT_KIND",
    "SPLIT_LEAKAGE_REPORT_MEDIA_TYPE",
    "SPLIT_LEAKAGE_REPORT_SCHEMA_VERSION",
    "SplitLeakageCheckError",
    "run_split_leakage_checks",
]
