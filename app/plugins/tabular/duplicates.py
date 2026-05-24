"""Duplicate marking and removal executor for approved ActionPlan steps.

This executor implements TASK-042 from the MVP backlog. Two modes are
supported:

- ``MARK_DUPLICATE_CANDIDATES`` adds duplicate-group marker columns
  (``is_duplicate_candidate``, ``duplicate_group_id``,
  ``is_duplicate_kept``) to the candidate dataset. No row is removed
  in this mode; the marker columns let downstream review queues
  surface duplicate candidates without deciding which row to keep.
- ``REMOVE_DUPLICATES`` keeps one canonical row per duplicate group
  (the lexicographically smallest ``object_id``) in the candidate
  dataset and removes the rest from the candidate CSV only. Row
  removal is candidate-only — the source dataset version artifact is
  never overwritten.

Reports never carry raw row payloads — only ``object_id`` values,
canonical signature hashes, and aggregate counts. The action records
``raw_artifact_hash`` and ``raw_artifact_unchanged`` so audit gates can
prove the immutability of the source artifact.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    DuplicateActionLineage,
    DuplicateActionMode,
    DuplicateActionReport,
    DuplicateGroupSummary,
    ErrorCode,
)
from app.domain.common import NonEmptyStr, Sha256Digest

DUPLICATE_REPORT_KIND = "duplicate_action_report"
DUPLICATE_REPORT_FORMAT = "json"
DUPLICATE_REPORT_MEDIA_TYPE = "application/json"
DUPLICATE_REPORT_SCHEMA_VERSION = "duplicate_action_report.v1"
CANDIDATE_TABULAR_DATASET_KIND = "candidate_tabular_dataset"
CANDIDATE_TABULAR_DATASET_FORMAT = "csv"
CANDIDATE_TABULAR_DATASET_MEDIA_TYPE = "text/csv"
CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION = "tabular_dataset.v1"

IS_DUPLICATE_CANDIDATE_COLUMN = "is_duplicate_candidate"
DUPLICATE_GROUP_ID_COLUMN = "duplicate_group_id"
IS_DUPLICATE_KEPT_COLUMN = "is_duplicate_kept"

_SUPPORTED_STEP_TYPES = {
    DuplicateActionMode.MARK.value,
    DuplicateActionMode.REMOVE_CANDIDATE.value,
}
_DEFAULT_ID_COLUMN = "object_id"


class DuplicateActionError(ValueError):
    """Raised when a duplicate ActionPlan step is unsafe or invalid."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.ACTION_PLAN_PRECONDITION_FAILED,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class ExecuteTabularDuplicatesRequest(BaseModel):
    """Inputs for executing one approved duplicate ActionPlan step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step: ActionPlanStep
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    source_artifact: ArtifactRef
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    id_column: NonEmptyStr = _DEFAULT_ID_COLUMN
    signature_columns: tuple[NonEmptyStr, ...] | None = None
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class ExecuteTabularDuplicatesResult:
    """Persisted artifacts and parsed report from a duplicate action."""

    report: DuplicateActionReport
    candidate_artifact: RegisteredArtifact
    report_artifact: RegisteredArtifact


def execute_tabular_duplicates_action(
    request: ExecuteTabularDuplicatesRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> ExecuteTabularDuplicatesResult:
    """Run an approved duplicate ActionPlan step and persist the results."""
    mode = _resolve_mode(request.step)
    source = storage.get(request.source_artifact.uri)
    raw_hash_before = _sha256(source.data)
    if raw_hash_before != request.source_artifact.hash:
        raise DuplicateActionError(
            reason_code="source_artifact_hash_mismatch",
            message="Source artifact content does not match the request hash.",
            code=ErrorCode.CONTRACT_VALIDATION_FAILED,
            details={
                "expected_hash": request.source_artifact.hash,
                "actual_hash": raw_hash_before,
            },
        )

    rows, columns = _read_csv(source.data)
    id_column = request.id_column
    if id_column not in columns:
        raise DuplicateActionError(
            reason_code="id_column_not_found",
            message="Configured id_column is not present in source CSV.",
            details={"id_column": id_column},
        )
    signature_columns = _resolve_signature_columns(
        columns=columns,
        explicit=request.signature_columns,
        id_column=id_column,
    )
    if not signature_columns:
        raise DuplicateActionError(
            reason_code="no_signature_columns",
            message=(
                "Duplicate detection requires at least one non-id signature column."
            ),
        )

    groups_by_signature = _group_by_signature(
        rows=rows,
        id_column=id_column,
        signature_columns=signature_columns,
    )
    duplicate_groups = [
        (signature, members)
        for signature, members in sorted(groups_by_signature.items())
        if len(members) >= 2
    ]
    before_pair_count = sum(len(members) - 1 for _, members in duplicate_groups)
    before_group_count = len(duplicate_groups)

    if mode is DuplicateActionMode.MARK:
        candidate_columns = _candidate_columns_for_mark(columns)
        candidate_rows, group_summaries, marked_count = _build_mark_output(
            rows=rows,
            id_column=id_column,
            duplicate_groups=duplicate_groups,
        )
        removed_count = 0
        after_pair_count = before_pair_count
        after_group_count = before_group_count
    else:
        candidate_columns = columns  # row removal does not change the schema
        (
            candidate_rows,
            group_summaries,
            removed_count,
        ) = _build_remove_output(
            rows=rows,
            id_column=id_column,
            duplicate_groups=duplicate_groups,
        )
        marked_count = 0
        after_pair_count = 0
        after_group_count = 0

    candidate_payload = _write_csv(rows=candidate_rows, columns=candidate_columns)
    candidate_artifact = registry.save_artifact(
        artifact_kind=CANDIDATE_TABULAR_DATASET_KIND,
        data=candidate_payload,
        artifact_format=CANDIDATE_TABULAR_DATASET_FORMAT,
        media_type=CANDIDATE_TABULAR_DATASET_MEDIA_TYPE,
        schema_version=CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": request.step.step_id,
            "source-artifact-hash": request.source_artifact.hash,
            "duplicate-mode": mode.value,
            "before-row-count": str(len(rows)),
            "after-row-count": str(len(candidate_rows)),
            "marked-count": str(marked_count),
            "removed-count": str(removed_count),
        },
    )

    raw_after = storage.get(request.source_artifact.uri)
    raw_hash_after = _sha256(raw_after.data)
    raw_unchanged = raw_hash_after == raw_hash_before

    report = DuplicateActionReport(
        report_id=request.report_id or f"duplicate_action_report_{uuid.uuid4().hex[:16]}",
        report_schema_version=DUPLICATE_REPORT_SCHEMA_VERSION,
        mode=mode,
        id_column=id_column,
        signature_columns=signature_columns,
        before_row_count=len(rows),
        after_row_count=len(candidate_rows),
        before_duplicate_pair_count=before_pair_count,
        before_duplicate_group_count=before_group_count,
        after_duplicate_pair_count=after_pair_count,
        after_duplicate_group_count=after_group_count,
        marked_count=marked_count,
        removed_count=removed_count,
        groups=tuple(group_summaries),
        raw_artifact_hash=raw_hash_after,
        raw_artifact_unchanged=raw_unchanged,
        lineage=DuplicateActionLineage(
            action_plan_id=request.action_plan_id,
            step_id=request.step.step_id,
            source_dataset_version_id=request.source_dataset_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_artifact=request.source_artifact,
            candidate_artifact=candidate_artifact.artifact_ref,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )
    report_artifact = registry.save_artifact(
        artifact_kind=DUPLICATE_REPORT_KIND,
        data=_serialize_report(report),
        artifact_format=DUPLICATE_REPORT_FORMAT,
        media_type=DUPLICATE_REPORT_MEDIA_TYPE,
        schema_version=DUPLICATE_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": request.step.step_id,
            "duplicate-mode": mode.value,
            "before-duplicate-pair-count": str(before_pair_count),
            "before-duplicate-group-count": str(before_group_count),
            "marked-count": str(marked_count),
            "removed-count": str(removed_count),
            "raw-artifact-unchanged": "true" if raw_unchanged else "false",
            "candidate-artifact-hash": candidate_artifact.hash,
        },
    )
    return ExecuteTabularDuplicatesResult(
        report=report,
        candidate_artifact=candidate_artifact,
        report_artifact=report_artifact,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_mode(step: ActionPlanStep) -> DuplicateActionMode:
    if step.type not in _SUPPORTED_STEP_TYPES:
        raise DuplicateActionError(
            reason_code="unsupported_action_step_type",
            message=(
                "Duplicate executor only handles MARK_DUPLICATE_CANDIDATES "
                "and REMOVE_DUPLICATES steps."
            ),
            details={"step_id": step.step_id, "step_type": step.type},
        )
    if step.type == DuplicateActionMode.MARK.value:
        return DuplicateActionMode.MARK
    return DuplicateActionMode.REMOVE_CANDIDATE


def _read_csv(data: bytes) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = tuple(name for name in (reader.fieldnames or ()) if name)
    if not columns:
        raise DuplicateActionError(
            reason_code="empty_tabular_source",
            message="Source CSV has no header columns.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    rows = [
        {column: ("" if row.get(column) is None else str(row.get(column))) for column in columns}
        for row in reader
    ]
    if not rows:
        raise DuplicateActionError(
            reason_code="empty_tabular_source",
            message="Source CSV has no data rows.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    return rows, columns


def _resolve_signature_columns(
    *,
    columns: tuple[str, ...],
    explicit: tuple[str, ...] | None,
    id_column: str,
) -> tuple[str, ...]:
    if explicit is not None:
        # Explicit list filters to known columns and excludes the id column.
        seen: set[str] = set()
        ordered: list[str] = []
        for column in explicit:
            if column not in columns or column == id_column or column in seen:
                continue
            seen.add(column)
            ordered.append(column)
        return tuple(ordered)
    return tuple(sorted(column for column in columns if column != id_column))


def _group_by_signature(
    *,
    rows: Sequence[Mapping[str, str]],
    id_column: str,
    signature_columns: tuple[str, ...],
) -> dict[str, list[tuple[int, str]]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for index, row in enumerate(rows):
        signature = _row_signature(row=row, signature_columns=signature_columns)
        members = groups.setdefault(signature, [])
        object_id = row.get(id_column, "")
        if not object_id:
            object_id = f"row_{index:06d}"
        members.append((index, object_id))
    return groups


def _row_signature(
    *,
    row: Mapping[str, str],
    signature_columns: tuple[str, ...],
) -> str:
    payload = {column: row.get(column, "") for column in signature_columns}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _candidate_columns_for_mark(columns: tuple[str, ...]) -> tuple[str, ...]:
    extras: list[str] = []
    if IS_DUPLICATE_CANDIDATE_COLUMN not in columns:
        extras.append(IS_DUPLICATE_CANDIDATE_COLUMN)
    if DUPLICATE_GROUP_ID_COLUMN not in columns:
        extras.append(DUPLICATE_GROUP_ID_COLUMN)
    if IS_DUPLICATE_KEPT_COLUMN not in columns:
        extras.append(IS_DUPLICATE_KEPT_COLUMN)
    return tuple([*columns, *extras])


def _build_mark_output(
    *,
    rows: Sequence[Mapping[str, str]],
    id_column: str,
    duplicate_groups: list[tuple[str, list[tuple[int, str]]]],
) -> tuple[list[dict[str, str]], list[DuplicateGroupSummary], int]:
    materialized = [dict(row) for row in rows]
    for row in materialized:
        row[IS_DUPLICATE_CANDIDATE_COLUMN] = "0"
        row[DUPLICATE_GROUP_ID_COLUMN] = ""
        row[IS_DUPLICATE_KEPT_COLUMN] = ""

    summaries: list[DuplicateGroupSummary] = []
    marked_object_ids_total: list[str] = []
    for group_index, (signature, members) in enumerate(duplicate_groups):
        canonical_index, canonical_object_id = _canonical_member(members)
        group_id = f"duplicate_group_{group_index:04d}"
        affected_object_ids: list[str] = []
        marked_for_group: list[str] = []
        for member_index, object_id in members:
            row = materialized[member_index]
            row[IS_DUPLICATE_CANDIDATE_COLUMN] = "1"
            row[DUPLICATE_GROUP_ID_COLUMN] = group_id
            row[IS_DUPLICATE_KEPT_COLUMN] = (
                "1" if object_id == canonical_object_id else "0"
            )
            affected_object_ids.append(object_id)
            if object_id != canonical_object_id:
                marked_for_group.append(object_id)
                marked_object_ids_total.append(object_id)
        # Affected ids carry every member of the group (canonical + duplicates).
        # Marked ids surface only the candidates that the user must review.
        summaries.append(
            DuplicateGroupSummary(
                group_id=group_id,
                signature_hash=signature,
                affected_object_ids=tuple(sorted(set(affected_object_ids))),
                kept_object_id=canonical_object_id,
                removed_object_ids=(),
                marked_object_ids=tuple(sorted(set(marked_for_group))),
            )
        )
        _ = canonical_index  # canonical_index recorded via canonical_object_id
    return materialized, summaries, len(set(marked_object_ids_total))


def _build_remove_output(
    *,
    rows: Sequence[Mapping[str, str]],
    id_column: str,
    duplicate_groups: list[tuple[str, list[tuple[int, str]]]],
) -> tuple[list[dict[str, str]], list[DuplicateGroupSummary], int]:
    drop_indices: set[int] = set()
    summaries: list[DuplicateGroupSummary] = []
    removed_object_ids_total: list[str] = []
    for group_index, (signature, members) in enumerate(duplicate_groups):
        canonical_index, canonical_object_id = _canonical_member(members)
        group_id = f"duplicate_group_{group_index:04d}"
        affected_object_ids: list[str] = []
        removed_for_group: list[str] = []
        for member_index, object_id in members:
            affected_object_ids.append(object_id)
            if member_index != canonical_index:
                drop_indices.add(member_index)
                removed_for_group.append(object_id)
                removed_object_ids_total.append(object_id)
        summaries.append(
            DuplicateGroupSummary(
                group_id=group_id,
                signature_hash=signature,
                affected_object_ids=tuple(sorted(set(affected_object_ids))),
                kept_object_id=canonical_object_id,
                removed_object_ids=tuple(sorted(set(removed_for_group))),
                marked_object_ids=(),
            )
        )
    candidate_rows = [
        dict(row) for index, row in enumerate(rows) if index not in drop_indices
    ]
    return candidate_rows, summaries, len(removed_object_ids_total)


def _canonical_member(members: list[tuple[int, str]]) -> tuple[int, str]:
    """Return the canonical (kept) member of a duplicate group.

    The canonical row is the one with the lexicographically smallest
    ``object_id``. Ties are broken by row index so the choice is fully
    deterministic.
    """
    return sorted(members, key=lambda item: (item[1], item[0]))[0]


def _write_csv(*, rows: Sequence[Mapping[str, str]], columns: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        materialized = {column: row.get(column, "") for column in columns}
        writer.writerow(materialized)
    return buffer.getvalue().encode("utf-8")


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _serialize_report(report: DuplicateActionReport) -> bytes:
    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
    ).encode("utf-8")


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


__all__ = [
    "DUPLICATE_GROUP_ID_COLUMN",
    "DUPLICATE_REPORT_FORMAT",
    "DUPLICATE_REPORT_KIND",
    "DUPLICATE_REPORT_MEDIA_TYPE",
    "DUPLICATE_REPORT_SCHEMA_VERSION",
    "DuplicateActionError",
    "ExecuteTabularDuplicatesRequest",
    "ExecuteTabularDuplicatesResult",
    "IS_DUPLICATE_CANDIDATE_COLUMN",
    "IS_DUPLICATE_KEPT_COLUMN",
    "execute_tabular_duplicates_action",
]
