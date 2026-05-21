"""Tabular profiler: schema inference and base profile report.

TASK-023 scope:

* compute ``row_count``, ``column_count``;
* infer column type and nullability;
* detect ``target_column``, group keys, id-like columns and PII-like columns;
* produce a contract-shaped :class:`TabularProfileReport`;
* persist the JSON report through :class:`ArtifactRegistry` so the
  ``DataForgeReport.detail_artifacts`` slot can hold an
  :class:`ArtifactRef` to the registered report.

Privacy rules honored here:

* the profiler never logs raw row payloads or column samples;
* ``sample_value`` is captured only for non-PII-like columns; PII-like
  columns surface ``sample_value=None`` so the profile artifact does not
  carry raw PII even by accident;
* ``pii_like_columns`` are detected by column name heuristics for the
  MVP. Deep PII detection is the responsibility of the text/OCR plugin
  and dedicated tabular PII detectors (later tasks).

Implementation details:

* the profiler uses stdlib ``csv`` and processes rows lazily; rows are
  consumed once and per-column statistics are accumulated incrementally.
* column type inference uses a deterministic precedence: identifier ->
  boolean -> numeric_integer -> numeric_float -> datetime -> categorical
  -> text. The decision is made from the union of all observed non-null
  values.
* role detection is conservative: the profiler picks at most one
  target column from a small allowlist (``is_fraud``, ``label``,
  ``target``) and at most one group key from common business keys.
"""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ArtifactRef,
    ColumnProfile,
    ColumnRole,
    ColumnType,
    TabularProfileLineage,
    TabularProfileReport,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.ingestion.archive_reader import (
    ArchiveEntryKind,
    ArchiveFileDescriptor,
    ArchiveReader,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    pass

PROFILE_REPORT_KIND = "tabular_profile_report"
PROFILE_REPORT_FORMAT = "json"
PROFILE_REPORT_MEDIA_TYPE = "application/json"
PROFILE_REPORT_SCHEMA_VERSION = "tabular_profile_report.v1"

_TARGET_COLUMN_NAMES: tuple[str, ...] = ("is_fraud", "label", "target")
_GROUP_KEY_NAMES: tuple[str, ...] = (
    "customer_id_hash",
    "customer_id",
    "case_id",
    "case_id_hash",
    "household_id",
    "group_id",
)
_ID_COLUMN_NAMES: tuple[str, ...] = (
    "object_id",
    "row_id",
    "transaction_id",
    "document_id",
    "support_ticket_id",
)
# Column-name heuristics for PII-like columns. Detection by content is
# delegated to the text/OCR plugin and dedicated tabular PII detectors.
_PII_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|_)e?mail($|_)", re.IGNORECASE),
    re.compile(r"(^|_)phone($|_)", re.IGNORECASE),
    re.compile(r"(^|_)passport($|_)", re.IGNORECASE),
    re.compile(r"(^|_)ssn($|_)", re.IGNORECASE),
    re.compile(r"(^|_)full_?name($|_)", re.IGNORECASE),
    re.compile(r"(^|_)address($|_)", re.IGNORECASE),
)
_LEAKAGE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"manual_review", re.IGNORECASE),
    re.compile(r"is_fraud_predict", re.IGNORECASE),
    re.compile(r"target_leak", re.IGNORECASE),
)

_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
)


class ProfileBuildRequest(BaseModel):
    """Inputs the tabular profiler requires from the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact_id: NonEmptyStr
    source_system: NonEmptyStr = "transactions"


@dataclass(frozen=True)
class BuildProfileResult:
    """Result of a tabular profile build."""

    profile_report: TabularProfileReport
    artifact: RegisteredArtifact


def infer_tabular_profile(
    rows: Iterable[dict[str, str]],
    *,
    columns: tuple[str, ...],
    request: ProfileBuildRequest,
    source_manifest_artifact: ArtifactRef,
    profile_id: str | None = None,
    generated_at: datetime | None = None,
) -> TabularProfileReport:
    """Build a ``TabularProfileReport`` from an iterable of CSV row dicts.

    This function is plugin-internal and never persists the report; use
    :func:`build_tabular_profile_report` for the full build/persist flow.
    The split keeps the inference logic easy to test against in-memory
    rows (no archive, no storage, no registry).
    """
    if not columns:
        raise ValueError("tabular profile requires at least one column")

    aggregators = {column: _ColumnAggregator(name=column) for column in columns}
    row_count = 0
    for row in rows:
        row_count += 1
        for column in columns:
            value = row.get(column, "")
            aggregators[column].observe(value)

    column_profiles: list[ColumnProfile] = []
    target_column: str | None = None
    group_keys: list[str] = []
    id_columns: list[str] = []
    pii_columns: list[str] = []
    for column in columns:
        agg = aggregators[column]
        is_pii_like = _column_name_looks_pii(column)
        role = _detect_role(
            column,
            is_pii_like=is_pii_like,
            target_already_chosen=target_column is not None,
        )
        if role is ColumnRole.TARGET and target_column is None:
            target_column = column
        if role is ColumnRole.GROUP_KEY:
            group_keys.append(column)
        if role is ColumnRole.ID:
            id_columns.append(column)
        if role is ColumnRole.PII_LIKE:
            pii_columns.append(column)

        sample_value = None if is_pii_like else agg.first_non_null
        column_profiles.append(
            ColumnProfile(
                name=column,
                type=agg.infer_type(),
                role=role,
                nullable=agg.null_count > 0,
                null_count=agg.null_count,
                null_ratio=(agg.null_count / row_count) if row_count else 0.0,
                distinct_count=len(agg.distinct_values),
                sample_value=sample_value,
                is_constant=row_count > 0 and len(agg.distinct_values) <= 1,
            )
        )

    return TabularProfileReport(
        profile_id=profile_id or f"tabular_profile_{uuid.uuid4().hex[:16]}",
        profile_schema_version=PROFILE_REPORT_SCHEMA_VERSION,
        source_system=request.source_system,
        row_count=row_count,
        column_count=len(columns),
        columns=tuple(column_profiles),
        target_column=target_column,
        group_key_columns=tuple(group_keys),
        id_columns=tuple(id_columns),
        pii_like_columns=tuple(pii_columns),
        lineage=TabularProfileLineage(
            dataset_id=request.dataset_id,
            version_id=request.version_id,
            parent_version_id=request.parent_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_manifest_artifact=source_manifest_artifact,
            source_artifact_id=request.source_artifact_id,
        ),
        generated_at=generated_at or datetime.now(UTC),
    )


def build_tabular_profile_report(
    archive_reader: ArchiveReader,
    *,
    request: ProfileBuildRequest,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    source_manifest_artifact: ArtifactRef,
    profile_id: str | None = None,
    generated_at: datetime | None = None,
) -> BuildProfileResult:
    """Build the tabular profile and persist it as an immutable artifact.

    ``archive_reader`` must produce at least one ``transactions.csv``
    descriptor (matching the demo archive contract). The profile is built
    by streaming rows through the archive descriptor; rows are not stored
    in memory beyond the per-column aggregators.

    The persisted artifact is JSON, content-addressed under
    ``artifact_kind=tabular_profile_report``, ``schema_version=tabular_profile_report.v1``.
    Its :class:`ArtifactRef` is contract-compatible with
    ``DataForgeReport.detail_artifacts`` slots.
    """
    descriptor = _resolve_transactions_descriptor(archive_reader)
    with descriptor.open() as handle:
        text_stream = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        reader = csv.DictReader(text_stream)
        fieldnames = reader.fieldnames or ()
        columns = tuple(name for name in fieldnames if name)
        rows_iter = (
            {key: ("" if value is None else value) for key, value in row.items()}
            for row in reader
        )
        report = infer_tabular_profile(
            rows_iter,
            columns=columns,
            request=request,
            source_manifest_artifact=source_manifest_artifact,
            profile_id=profile_id,
            generated_at=generated_at,
        )
        text_stream.close()
    payload = _serialize_report(report)
    artifact = registry.save_artifact(
        artifact_kind=PROFILE_REPORT_KIND,
        data=payload,
        artifact_format=PROFILE_REPORT_FORMAT,
        media_type=PROFILE_REPORT_MEDIA_TYPE,
        schema_version=PROFILE_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "row_count": str(report.row_count),
            "column_count": str(report.column_count),
            "source_system": report.source_system,
        },
    )
    # storage parameter is required by tests/orchestrator wiring to keep the
    # signature symmetrical with build_validated_manifest; the concrete put
    # is delegated to the registry via the same scoped adapter.
    _ = storage
    return BuildProfileResult(profile_report=report, artifact=artifact)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_transactions_descriptor(reader: ArchiveReader) -> ArchiveFileDescriptor:
    descriptors = [
        descriptor
        for descriptor in reader.descriptors()
        if descriptor.kind is ArchiveEntryKind.TRANSACTIONS
    ]
    if not descriptors:
        raise ValueError(
            "tabular profile requires a transactions.csv entry in the archive"
        )
    if len(descriptors) > 1:
        raise ValueError(
            "tabular profile expects exactly one transactions.csv entry"
        )
    return descriptors[0]


def _detect_role(
    column: str,
    *,
    is_pii_like: bool,
    target_already_chosen: bool,
) -> ColumnRole:
    if is_pii_like:
        return ColumnRole.PII_LIKE
    if any(
        pattern.search(column) is not None for pattern in _LEAKAGE_NAME_PATTERNS
    ):
        return ColumnRole.LEAKAGE_CANDIDATE
    if column in _ID_COLUMN_NAMES:
        return ColumnRole.ID
    if column in _GROUP_KEY_NAMES:
        return ColumnRole.GROUP_KEY
    if not target_already_chosen and column in _TARGET_COLUMN_NAMES:
        return ColumnRole.TARGET
    return ColumnRole.UNKNOWN


def _column_name_looks_pii(column: str) -> bool:
    return any(pattern.search(column) is not None for pattern in _PII_NAME_PATTERNS)


def _serialize_report(report: TabularProfileReport) -> bytes:
    payload = report.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# column aggregator
# ---------------------------------------------------------------------------


@dataclass
class _ColumnAggregator:
    """Streaming aggregator for one CSV column."""

    name: str
    null_count: int = 0
    distinct_values: set[str] = None  # type: ignore[assignment]
    first_non_null: str | None = None
    saw_int: bool = False
    saw_float: bool = False
    saw_bool: bool = False
    saw_datetime: bool = False
    saw_text_only: bool = False
    saw_identifier_only: bool = True
    total_non_null: int = 0

    def __post_init__(self) -> None:
        self.distinct_values = set()

    def observe(self, value: str) -> None:
        if value == "" or value is None:
            self.null_count += 1
            return
        self.total_non_null += 1
        self.distinct_values.add(value)
        if self.first_non_null is None:
            self.first_non_null = value
        token = value.strip()
        is_bool = token.lower() in {"true", "false", "yes", "no"}
        is_int = _is_integer(token)
        is_float = _is_float(token) and not is_int
        is_datetime = _is_datetime(token)
        if is_bool:
            self.saw_bool = True
        if is_int:
            self.saw_int = True
        if is_float:
            self.saw_float = True
        if is_datetime:
            self.saw_datetime = True
        if not (is_bool or is_int or is_float or is_datetime):
            self.saw_text_only = True
        if not _looks_like_identifier(token):
            self.saw_identifier_only = False

    def infer_type(self) -> ColumnType:
        if self.total_non_null == 0:
            return ColumnType.UNKNOWN
        # 0/1 numeric integers across the column count as boolean if the
        # set of distinct values is exactly {0, 1}.
        if self.saw_bool and not (self.saw_text_only or self.saw_float or self.saw_datetime):
            return ColumnType.BOOLEAN
        if (
            self.saw_int
            and not self.saw_text_only
            and not self.saw_float
            and not self.saw_datetime
        ):
            if self.distinct_values <= {"0", "1"} and len(self.distinct_values) <= 2:
                return ColumnType.BOOLEAN
            return ColumnType.NUMERIC_INTEGER
        if (self.saw_float or self.saw_int) and not self.saw_text_only and not self.saw_datetime:
            return ColumnType.NUMERIC_FLOAT
        if self.saw_datetime and not (self.saw_text_only or self.saw_int or self.saw_float):
            return ColumnType.DATETIME
        if self.saw_identifier_only and self.total_non_null > 0:
            distinct_ratio = len(self.distinct_values) / self.total_non_null
            if distinct_ratio >= 0.9:
                return ColumnType.IDENTIFIER
        if self.saw_text_only:
            distinct_ratio = (
                len(self.distinct_values) / self.total_non_null if self.total_non_null else 0.0
            )
            if distinct_ratio < 0.5 and len(self.distinct_values) <= 50:
                return ColumnType.CATEGORICAL
            return ColumnType.TEXT
        return ColumnType.UNKNOWN


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    if value.lower() in {"inf", "-inf", "nan"}:
        return False
    return True


def _is_datetime(value: str) -> bool:
    for fmt in _DATETIME_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:\-]*$")


def _looks_like_identifier(value: str) -> bool:
    return _IDENTIFIER_PATTERN.match(value) is not None


__all__ = [
    "BuildProfileResult",
    "PROFILE_REPORT_FORMAT",
    "PROFILE_REPORT_KIND",
    "PROFILE_REPORT_MEDIA_TYPE",
    "PROFILE_REPORT_SCHEMA_VERSION",
    "ProfileBuildRequest",
    "build_tabular_profile_report",
    "infer_tabular_profile",
]
