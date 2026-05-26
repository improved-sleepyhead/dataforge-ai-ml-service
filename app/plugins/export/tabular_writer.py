"""Tabular Parquet/CSV export writer (TASK-054).

The writer reads an immutable candidate tabular CSV from object
storage, optionally filters out blocked ``object_id`` values, and
materializes:

- a Parquet artifact (``train.parquet``-style primary export);
- an optional CSV artifact (``train.csv`` compatibility export);
- per-split Parquet (and optional CSV) artifacts when a
  ``SplitManifest`` is supplied.

All outputs are registered as immutable artifacts via the
``ArtifactRegistry`` so the ExportPackage can carry deterministic
content-addressed hashes (acceptance criterion: "Artifact hashes
сохраняются в ExportPackage").
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import ArtifactRef, DataSplit, ErrorCode, SplitManifest
from app.domain.common import NonEmptyStr, Sha256Digest

TABULAR_EXPORT_PARQUET_KIND = "tabular_export_parquet"
TABULAR_EXPORT_PARQUET_FORMAT = "parquet"
TABULAR_EXPORT_PARQUET_MEDIA_TYPE = "application/x-parquet"
TABULAR_EXPORT_PARQUET_SCHEMA_VERSION = "tabular_export.v1"

TABULAR_EXPORT_CSV_KIND = "tabular_export_csv"
TABULAR_EXPORT_CSV_FORMAT = "csv"
TABULAR_EXPORT_CSV_MEDIA_TYPE = "text/csv"
TABULAR_EXPORT_CSV_SCHEMA_VERSION = "tabular_export.v1"

_OBJECT_ID_COLUMN = "object_id"


class TabularExportWriterError(ValueError):
    """Raised when the tabular export writer cannot run safely."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.INVALID_JOB_PAYLOAD,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class TabularExportRequest(BaseModel):
    """Inputs for :func:`write_tabular_export`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    source_artifact: ArtifactRef
    split_manifest: SplitManifest | None = None
    split_manifest_artifact: ArtifactRef | None = None
    blocked_object_ids: tuple[NonEmptyStr, ...] = ()
    write_csv: bool = True
    write_per_split: bool = True
    object_id_column: NonEmptyStr = _OBJECT_ID_COLUMN
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    export_name_prefix: NonEmptyStr = Field(default="export")
    generated_at: datetime | None = None


@dataclass(frozen=True)
class TabularExportPerSplitArtifact:
    """Parquet (and optional CSV) artifacts for a single split."""

    split: DataSplit
    parquet_artifact: RegisteredArtifact
    csv_artifact: RegisteredArtifact | None
    row_count: int


@dataclass(frozen=True)
class TabularExportArtifacts:
    """Result of :func:`write_tabular_export`."""

    parquet_artifact: RegisteredArtifact
    csv_artifact: RegisteredArtifact | None
    per_split: tuple[TabularExportPerSplitArtifact, ...]
    split_manifest_artifact: ArtifactRef | None
    columns: tuple[str, ...]
    included_row_count: int
    excluded_blocked_count: int

    def all_artifact_refs(self) -> tuple[ArtifactRef, ...]:
        """Return every artifact ref this writer produced."""
        refs: list[ArtifactRef] = [self.parquet_artifact.artifact_ref]
        if self.csv_artifact is not None:
            refs.append(self.csv_artifact.artifact_ref)
        if self.split_manifest_artifact is not None:
            refs.append(self.split_manifest_artifact)
        for entry in self.per_split:
            refs.append(entry.parquet_artifact.artifact_ref)
            if entry.csv_artifact is not None:
                refs.append(entry.csv_artifact.artifact_ref)
        return tuple(refs)


def write_tabular_export(
    request: TabularExportRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> TabularExportArtifacts:
    """Write Parquet / CSV / per-split artifacts for a tabular candidate."""
    rows, columns = _read_candidate_rows(
        storage=storage, artifact=request.source_artifact
    )
    if request.object_id_column not in columns:
        raise TabularExportWriterError(
            reason_code="object_id_column_missing_in_source",
            message=(
                f"object_id column {request.object_id_column!r} not found "
                "in candidate tabular source columns."
            ),
            details={"columns": list(columns)},
        )

    blocked_ids = set(request.blocked_object_ids)
    filtered_rows = [
        row for row in rows if row.get(request.object_id_column) not in blocked_ids
    ]
    excluded = len(rows) - len(filtered_rows)

    parquet_artifact = _persist_parquet(
        rows=filtered_rows,
        columns=columns,
        split=None,
        request=request,
        registry=registry,
    )
    csv_artifact = (
        _persist_csv(
            rows=filtered_rows,
            columns=columns,
            split=None,
            request=request,
            registry=registry,
        )
        if request.write_csv
        else None
    )

    per_split: tuple[TabularExportPerSplitArtifact, ...] = ()
    if request.write_per_split and request.split_manifest is not None:
        per_split = _write_per_split_artifacts(
            request=request,
            registry=registry,
            rows=filtered_rows,
            columns=columns,
        )

    return TabularExportArtifacts(
        parquet_artifact=parquet_artifact,
        csv_artifact=csv_artifact,
        per_split=per_split,
        split_manifest_artifact=request.split_manifest_artifact,
        columns=tuple(columns),
        included_row_count=len(filtered_rows),
        excluded_blocked_count=excluded,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_candidate_rows(
    *,
    storage: MinioObjectStorageAdapter,
    artifact: ArtifactRef,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    stored = storage.get(artifact.uri)
    text = stored.data.decode("utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    rows: list[dict[str, str]] = []
    columns: tuple[str, ...] = ()
    for index, raw in enumerate(reader):
        if index == 0:
            columns = tuple(raw)
            continue
        row = {
            column: ("" if value is None else str(value))
            for column, value in zip(columns, raw, strict=False)
        }
        rows.append(row)
    if not columns:
        raise TabularExportWriterError(
            reason_code="empty_tabular_source",
            message="Candidate tabular source is empty (missing CSV header).",
        )
    return rows, columns


def _persist_parquet(
    *,
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
    split: DataSplit | None,
    request: TabularExportRequest,
    registry: ArtifactRegistry,
) -> RegisteredArtifact:
    table = _rows_to_arrow(rows=rows, columns=columns)
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    payload = sink.getvalue()
    return registry.save_artifact(
        artifact_kind=TABULAR_EXPORT_PARQUET_KIND,
        data=payload,
        artifact_format=TABULAR_EXPORT_PARQUET_FORMAT,
        media_type=TABULAR_EXPORT_PARQUET_MEDIA_TYPE,
        schema_version=TABULAR_EXPORT_PARQUET_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata=_artifact_metadata(
            request=request,
            split=split,
            row_count=len(rows),
            columns=columns,
            artifact_format=TABULAR_EXPORT_PARQUET_FORMAT,
        ),
    )


def _persist_csv(
    *,
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
    split: DataSplit | None,
    request: TabularExportRequest,
    registry: ArtifactRegistry,
) -> RegisteredArtifact:
    sink = io.StringIO()
    writer = csv.DictWriter(sink, fieldnames=list(columns))
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    payload = sink.getvalue().encode("utf-8")
    return registry.save_artifact(
        artifact_kind=TABULAR_EXPORT_CSV_KIND,
        data=payload,
        artifact_format=TABULAR_EXPORT_CSV_FORMAT,
        media_type=TABULAR_EXPORT_CSV_MEDIA_TYPE,
        schema_version=TABULAR_EXPORT_CSV_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata=_artifact_metadata(
            request=request,
            split=split,
            row_count=len(rows),
            columns=columns,
            artifact_format=TABULAR_EXPORT_CSV_FORMAT,
        ),
    )


def _write_per_split_artifacts(
    *,
    request: TabularExportRequest,
    registry: ArtifactRegistry,
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
) -> tuple[TabularExportPerSplitArtifact, ...]:
    manifest = request.split_manifest
    assert manifest is not None
    by_split: dict[DataSplit, list[dict[str, str]]] = {}
    assignment_map = {
        assignment.object_id: assignment.split for assignment in manifest.assignments
    }
    blocked_ids = set(request.blocked_object_ids)
    for row in rows:
        object_id = row.get(request.object_id_column)
        if not object_id or object_id in blocked_ids:
            continue
        split = assignment_map.get(object_id)
        if split is None:
            continue
        by_split.setdefault(split, []).append(row)

    results: list[TabularExportPerSplitArtifact] = []
    for split in sorted(by_split, key=lambda value: value.value):
        split_rows = by_split[split]
        parquet_artifact = _persist_parquet(
            rows=split_rows,
            columns=columns,
            split=split,
            request=request,
            registry=registry,
        )
        csv_artifact = (
            _persist_csv(
                rows=split_rows,
                columns=columns,
                split=split,
                request=request,
                registry=registry,
            )
            if request.write_csv
            else None
        )
        results.append(
            TabularExportPerSplitArtifact(
                split=split,
                parquet_artifact=parquet_artifact,
                csv_artifact=csv_artifact,
                row_count=len(split_rows),
            )
        )
    return tuple(results)


def _rows_to_arrow(
    *,
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
) -> pa.Table:
    column_arrays: dict[str, pa.Array] = {}
    for column in columns:
        values = [row.get(column, "") for row in rows]
        coerced, dtype = _coerce_column_values(values)
        column_arrays[column] = pa.array(coerced, type=dtype)
    return pa.table(column_arrays)


def _coerce_column_values(
    values: Iterable[str],
) -> tuple[list[Any], pa.DataType]:
    """Coerce CSV string values into Arrow-typed values.

    Strategy:

    * empty string -> ``None`` (preserves missingness through the Arrow
      column without losing nullability);
    * if every non-null value parses as an integer -> int64;
    * else if every non-null value parses as a float -> float64;
    * otherwise -> string.

    The strategy mirrors how downstream ML tooling (pandas/Polars,
    scikit-learn) reads the CSV, so the exported Parquet does not
    silently change semantics.
    """
    raw = list(values)
    cleaned: list[str | None] = [None if value == "" else value for value in raw]

    if all(value is None for value in cleaned):
        return cleaned, pa.string()

    int_values: list[int | None] = []
    is_int = True
    for value in cleaned:
        if value is None:
            int_values.append(None)
            continue
        try:
            int_values.append(int(value))
        except ValueError:
            is_int = False
            break
    if is_int:
        return int_values, pa.int64()

    float_values: list[float | None] = []
    is_float = True
    for value in cleaned:
        if value is None:
            float_values.append(None)
            continue
        try:
            float_values.append(float(value))
        except ValueError:
            is_float = False
            break
    if is_float:
        return float_values, pa.float64()

    return cleaned, pa.string()


def _artifact_metadata(
    *,
    request: TabularExportRequest,
    split: DataSplit | None,
    row_count: int,
    columns: Sequence[str],
    artifact_format: str,
) -> dict[str, str]:
    metadata: dict[str, str] = {
        "export-name": request.export_name_prefix,
        "row-count": str(row_count),
        "column-count": str(len(columns)),
        "source-artifact-hash": request.source_artifact.hash,
        "candidate-version-id": request.candidate_dataset_version_id,
        "artifact-format": artifact_format,
        **(
            {"created-at": request.generated_at.isoformat()}
            if request.generated_at is not None
            else {}
        ),
    }
    if split is not None:
        metadata["split"] = split.value
    else:
        metadata["split"] = "all"
    return metadata


__all__ = [
    "TABULAR_EXPORT_CSV_FORMAT",
    "TABULAR_EXPORT_CSV_KIND",
    "TABULAR_EXPORT_CSV_MEDIA_TYPE",
    "TABULAR_EXPORT_CSV_SCHEMA_VERSION",
    "TABULAR_EXPORT_PARQUET_FORMAT",
    "TABULAR_EXPORT_PARQUET_KIND",
    "TABULAR_EXPORT_PARQUET_MEDIA_TYPE",
    "TABULAR_EXPORT_PARQUET_SCHEMA_VERSION",
    "TabularExportArtifacts",
    "TabularExportPerSplitArtifact",
    "TabularExportRequest",
    "TabularExportWriterError",
    "write_tabular_export",
]
