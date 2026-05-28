"""Asset Manifest builder for the DataForge AI ingestion layer.

This module turns a validated archive (TASK-018/019) and the identity helpers
(TASK-020) into a contract-shaped Asset Manifest. The manifest is a versioned
JSONL artifact whose rows are :class:`ManifestRow` records. The builder is
contract-first: every row is validated through the pydantic model before
being serialized, so structurally invalid rows never reach the manifest.

Constraints honored here:

* the manifest is **never** a raw mutation of source data; it is a new
  immutable derived artifact written through :class:`ArtifactRegistry`;
* every row carries identity (``object_id``, ``dataset_id``, ``version_id``,
  ``modality``, ``hash``, ``metadata``, ``lineage``);
* link keys (``case_id``, ``customer_id_hash``, ``document_id``,
  ``transaction_id``, ``support_ticket_id``) are surfaced into row metadata
  when the source record exposes them, so downstream multimodal joins are
  possible without re-reading raw archives;
* unsupported entries (``ArchiveEntryKind.OTHER``) are not silently dropped:
  they are recorded in :class:`BuildManifestResult.unsupported_entries`;
* nothing here writes raw text, secrets, or full record payloads into logs.
"""

from __future__ import annotations

import csv
import io
import json
import posixpath
from collections.abc import Iterable
from dataclasses import dataclass
from typing import IO, Any

from pydantic import BaseModel, ConfigDict

from app.adapters.artifact_registry import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    ArtifactRef,
    DataModality,
    ManifestLineage,
    ManifestRow,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.ingestion.archive_reader import (
    ArchiveEntryKind,
    ArchiveFileDescriptor,
    ArchiveReader,
)
from app.ingestion.identity import (
    compute_record_sha256,
    derive_object_id,
)

_KNOWN_LINK_KEYS: tuple[str, ...] = (
    "case_id",
    "customer_id_hash",
    "document_id",
    "transaction_id",
    "support_ticket_id",
)
_TABULAR_TARGET_KEYS: tuple[str, ...] = ("is_fraud", "label", "target")
_TABULAR_SPLIT_KEYS: tuple[str, ...] = ("split", "data_split")


class BuildManifestRequest(BaseModel):
    """Inputs the manifest builder requires from the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest


@dataclass(frozen=True)
class UnsupportedEntry:
    """One archive entry that the builder did not classify as a known modality."""

    name: str
    kind: ArchiveEntryKind
    file_size: int
    line_number: int | None = None
    reason_code: str = "unsupported_entry"


@dataclass(frozen=True)
class BuildManifestResult:
    """Result of a manifest build."""

    manifest_artifact: RegisteredArtifact
    rows_by_modality: dict[DataModality, int]
    unsupported_entries: tuple[UnsupportedEntry, ...]
    total_rows: int


def build_asset_manifest(
    reader: ArchiveReader,
    *,
    request: BuildManifestRequest,
    registry: ArtifactRegistry,
) -> BuildManifestResult:
    """Build and persist the versioned Asset Manifest for a validated archive.

    The function reads the archive lazily (line-by-line for tabular and JSONL
    payloads), constructs :class:`ManifestRow` records through pydantic, and
    writes the resulting JSONL bytes through the artifact registry. The
    written artifact is returned together with row counts and a list of
    unsupported entries for the report layer.
    """
    rows: list[ManifestRow] = []
    rows_by_modality: dict[DataModality, int] = {}
    unsupported: list[UnsupportedEntry] = []

    source_artifact_id = _source_artifact_id_for_archive(reader)

    for descriptor in reader.descriptors():
        if descriptor.kind is ArchiveEntryKind.TRANSACTIONS:
            for row in _build_tabular_rows(
                descriptor=descriptor,
                request=request,
                source_artifact_id=source_artifact_id,
            ):
                rows.append(row)
                rows_by_modality[row.modality] = (
                    rows_by_modality.get(row.modality, 0) + 1
                )
        elif descriptor.kind is ArchiveEntryKind.SUPPORT_MESSAGES:
            for row in _build_jsonl_rows(
                descriptor=descriptor,
                request=request,
                source_artifact_id=source_artifact_id,
                modality=DataModality.TEXT,
                source_system="support_messages",
                unsupported=unsupported,
            ):
                rows.append(row)
                rows_by_modality[row.modality] = (
                    rows_by_modality.get(row.modality, 0) + 1
                )
        elif descriptor.kind is ArchiveEntryKind.OCR_RECORDS:
            for row in _build_jsonl_rows(
                descriptor=descriptor,
                request=request,
                source_artifact_id=source_artifact_id,
                modality=DataModality.DOCUMENT_OCR,
                source_system="ocr_records",
                unsupported=unsupported,
            ):
                rows.append(row)
                rows_by_modality[row.modality] = (
                    rows_by_modality.get(row.modality, 0) + 1
                )
        elif descriptor.kind is ArchiveEntryKind.IMAGE_MANIFEST:
            for row in _build_jsonl_rows(
                descriptor=descriptor,
                request=request,
                source_artifact_id=source_artifact_id,
                modality=DataModality.IMAGE,
                source_system="image_manifest",
                unsupported=unsupported,
            ):
                rows.append(row)
                rows_by_modality[row.modality] = (
                    rows_by_modality.get(row.modality, 0) + 1
                )
        elif descriptor.kind is ArchiveEntryKind.ANNOTATIONS:
            # Annotation samples are bundled documents; expose them as a
            # single skeleton manifest row so downstream plugins know they
            # exist without expanding their internal structure.
            row = _build_skeleton_row(
                descriptor=descriptor,
                request=request,
                source_artifact_id=source_artifact_id,
                modality=DataModality.MULTIMODAL,
                source_system="annotations_sample",
            )
            rows.append(row)
            rows_by_modality[row.modality] = (
                rows_by_modality.get(row.modality, 0) + 1
            )
        elif descriptor.kind is ArchiveEntryKind.PREDICTIONS:
            # Predictions are joined to manifest rows by object_id later;
            # they are not themselves manifest rows. Skipping is intentional.
            continue
        elif descriptor.kind is ArchiveEntryKind.README:
            continue
        else:
            unsupported.append(
                UnsupportedEntry(
                    name=descriptor.name,
                    kind=descriptor.kind,
                    file_size=descriptor.file_size,
                )
            )

    payload = _serialize_rows(rows)
    artifact = registry.save_artifact(
        artifact_kind="asset_manifest",
        data=payload,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="manifest_row.v1",
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={"row_count": str(len(rows))},
    )
    return BuildManifestResult(
        manifest_artifact=artifact,
        rows_by_modality=dict(rows_by_modality),
        unsupported_entries=tuple(unsupported),
        total_rows=len(rows),
    )


def manifest_rows_from_artifact(artifact_data: bytes) -> tuple[ManifestRow, ...]:
    """Re-parse manifest bytes back into ManifestRow records (test helper)."""
    rows: list[ManifestRow] = []
    for line in artifact_data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(ManifestRow.model_validate_json(line))
    return tuple(rows)


# ---------------------------------------------------------------------------
# tabular
# ---------------------------------------------------------------------------


def _build_tabular_rows(
    *,
    descriptor: ArchiveFileDescriptor,
    request: BuildManifestRequest,
    source_artifact_id: str,
) -> Iterable[ManifestRow]:
    asset_uri_prefix = _manifest_asset_uri_prefix(request, source_system="transactions")
    with descriptor.open() as handle:
        reader = csv.DictReader(_text_stream(handle))
        if reader.fieldnames is None:
            return
        for row in reader:
            content_hash = compute_record_sha256(row)
            row_key = row.get("object_id") or _fallback_row_key("transactions", row)
            object_id = derive_object_id(
                dataset_version_id=request.version_id,
                row_key=row_key,
                content_hash=content_hash,
            )
            metadata = _tabular_metadata(row)
            label, split = _label_and_split_from_row(row)
            manifest_row = ManifestRow(
                object_id=object_id,
                dataset_id=request.dataset_id,
                version_id=request.version_id,
                modality=DataModality.TABULAR,
                asset_uri=f"{asset_uri_prefix}/{object_id}",
                hash=content_hash,
                metadata=metadata,
                lineage=ManifestLineage(
                    source_artifact_id=source_artifact_id,
                    parent_version_id=request.parent_version_id,
                    created_by_job_id=request.created_by_job_id,
                    config_hash=request.config_hash,
                ),
                label=label,
                split=split,
                source_system="transactions",
            )
            yield manifest_row


def _tabular_metadata(row: dict[str, str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _KNOWN_LINK_KEYS:
        value = row.get(key)
        if value:
            metadata[key] = value
    if "customer_segment" in row and row["customer_segment"]:
        metadata["customer_segment"] = row["customer_segment"]
    source_object_id = row.get("object_id")
    if source_object_id:
        metadata["source_object_id"] = source_object_id
    return metadata


def _label_and_split_from_row(row: dict[str, str]) -> tuple[str | None, None]:
    for key in _TABULAR_TARGET_KEYS:
        if key in row and row[key] != "":
            return _canonical_label(key=key, value=row[key]), None
    return None, None


def _canonical_label(*, key: str, value: str) -> str:
    if key == "is_fraud":
        if value == "1":
            return "fraud"
        if value == "0":
            return "not_fraud"
    return value


# ---------------------------------------------------------------------------
# JSONL modalities
# ---------------------------------------------------------------------------


def _build_jsonl_rows(
    *,
    descriptor: ArchiveFileDescriptor,
    request: BuildManifestRequest,
    source_artifact_id: str,
    modality: DataModality,
    source_system: str,
    unsupported: list[UnsupportedEntry],
) -> Iterable[ManifestRow]:
    asset_uri_prefix = _manifest_asset_uri_prefix(request, source_system=source_system)
    with descriptor.open() as handle:
        for line_number, raw_line in enumerate(_text_stream(handle), start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                unsupported.append(
                    UnsupportedEntry(
                        name=descriptor.name,
                        kind=descriptor.kind,
                        file_size=descriptor.file_size,
                        line_number=line_number,
                        reason_code="invalid_jsonl",
                    )
                )
                continue
            if not isinstance(record, dict):
                unsupported.append(
                    UnsupportedEntry(
                        name=descriptor.name,
                        kind=descriptor.kind,
                        file_size=descriptor.file_size,
                        line_number=line_number,
                        reason_code="jsonl_row_not_object",
                    )
                )
                continue
            content_hash = compute_record_sha256(record)
            row_key = (
                record.get("object_id")
                or _fallback_row_key(source_system, record, line_number=line_number)
            )
            object_id = derive_object_id(
                dataset_version_id=request.version_id,
                row_key=row_key,
                content_hash=content_hash,
            )
            metadata = _jsonl_metadata(record)
            yield ManifestRow(
                object_id=object_id,
                dataset_id=request.dataset_id,
                version_id=request.version_id,
                modality=modality,
                asset_uri=f"{asset_uri_prefix}/{object_id}",
                hash=content_hash,
                metadata=metadata,
                lineage=ManifestLineage(
                    source_artifact_id=source_artifact_id,
                    parent_version_id=request.parent_version_id,
                    created_by_job_id=request.created_by_job_id,
                    config_hash=request.config_hash,
                ),
                label=None,
                split=None,
                source_system=source_system,
            )


def _jsonl_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _KNOWN_LINK_KEYS:
        value = record.get(key)
        if value not in (None, "", []):
            metadata[key] = value
    if "language" in record and record["language"]:
        metadata["language"] = record["language"]
    if "page" in record and record["page"] is not None:
        metadata["page"] = record["page"]
    source_object_id = record.get("object_id")
    if source_object_id:
        metadata["source_object_id"] = str(source_object_id)
    return metadata


# ---------------------------------------------------------------------------
# skeleton modalities
# ---------------------------------------------------------------------------


def _build_skeleton_row(
    *,
    descriptor: ArchiveFileDescriptor,
    request: BuildManifestRequest,
    source_artifact_id: str,
    modality: DataModality,
    source_system: str,
) -> ManifestRow:
    file_hash = descriptor.sha256()
    object_id = derive_object_id(
        dataset_version_id=request.version_id,
        row_key=f"{source_system}:{posixpath.basename(descriptor.name)}",
        content_hash=file_hash,
    )
    asset_uri_prefix = _manifest_asset_uri_prefix(request, source_system=source_system)
    return ManifestRow(
        object_id=object_id,
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        modality=modality,
        asset_uri=f"{asset_uri_prefix}/{object_id}",
        hash=file_hash,
        metadata={
            "source_artifact_name": descriptor.name,
            "skeleton": True,
        },
        lineage=ManifestLineage(
            source_artifact_id=source_artifact_id,
            parent_version_id=request.parent_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
        ),
        label=None,
        split=None,
        source_system=source_system,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _source_artifact_id_for_archive(reader: ArchiveReader) -> str:
    # The reader does not currently surface the originating ArtifactRef; we
    # derive a stable handle from the safety report so lineage points at
    # this archive bundle deterministically. The orchestrator will swap this
    # for a real ArtifactRef when ingestion is wired into Dagster.
    sample = ",".join(reader.contents.safety_report.safe_entries)
    digest = compute_record_sha256(
        {
            "compressed_bytes": reader.contents.safety_report.compressed_bytes,
            "uncompressed_bytes": reader.contents.safety_report.uncompressed_bytes,
            "safe_entries": sample,
        }
    )
    return f"raw_archive:{digest.removeprefix('sha256:')[:16]}"


def _manifest_asset_uri_prefix(
    request: BuildManifestRequest,
    *,
    source_system: str,
) -> str:
    return (
        "s3://dataforge-manifest/"
        f"versions/{request.version_id}/"
        f"objects/{source_system}"
    )


def _serialize_rows(rows: list[ManifestRow]) -> bytes:
    if not rows:
        return b""
    serialized = "\n".join(_row_to_json(row) for row in rows) + "\n"
    return serialized.encode("utf-8")


def _row_to_json(row: ManifestRow) -> str:
    payload = row.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _text_stream(handle: IO[bytes]) -> io.TextIOWrapper:
    return io.TextIOWrapper(handle, encoding="utf-8", newline="")


def _fallback_row_key(
    source_system: str,
    record: dict[str, Any],
    *,
    line_number: int | None = None,
) -> str:
    if line_number is not None:
        return f"{source_system}:{line_number}"
    return f"{source_system}:" + json.dumps(record, sort_keys=True, separators=(",", ":"))


__all__ = [
    "BuildManifestRequest",
    "BuildManifestResult",
    "UnsupportedEntry",
    "build_asset_manifest",
    "manifest_rows_from_artifact",
]


# Make ArtifactRef importable for downstream consumers; not strictly used here
# but the attribute keeps the public layer-aware import from dragging the
# adapters package into kernel-level modules later.
_ = ArtifactRef
