"""Archive reader for the DataForge AI ingestion layer.

The reader sits behind :mod:`app.ingestion.archive_safety` and turns a
validated archive into a list of :class:`ArchiveFileDescriptor` objects
classified by stable contract kinds. It is deliberately small:

* it does not extract files to disk;
* it does not load entries into memory eagerly. Each descriptor exposes
  ``open()`` that returns an ``IO[bytes]`` backed by ``zipfile.ZipFile.open``
  so callers can stream-read CSV/JSONL/Parquet content lazily.
* it is contract-aware: it knows the canonical demo-archive entries
  (transactions, predictions, support messages, OCR records, image manifest,
  annotations sample, README) and tags them with stable kinds; unknown
  files become ``"other"`` descriptors so manifest-builder code can decide
  what to do with them.
* it enforces a hard requirement: ``transactions.csv`` must be present.
  Otherwise the archive is considered invalid and
  :class:`ArchiveSafetyError` is raised with
  :class:`ErrorCode.INVALID_ARCHIVE_STRUCTURE`.

The reader takes ownership of the underlying ``zipfile.ZipFile`` and exposes
itself as a context manager so the file handle is closed deterministically.
"""

from __future__ import annotations

import hashlib
import io
import posixpath
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import IO

from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import ErrorCode
from app.ingestion.archive_safety import (
    ArchiveSafetyError,
    ArchiveSafetyPolicy,
    ArchiveSafetyReport,
    validate_archive_artifact,
    validate_archive_bytes,
    validate_archive_path,
)

_REQUIRED_ENTRY = "transactions.csv"
_STREAM_CHUNK_SIZE = 64 * 1024


class ArchiveEntryKind(StrEnum):
    """Stable contract kinds assigned to recognized archive entries."""

    TRANSACTIONS = "transactions"
    PREDICTIONS = "predictions"
    SUPPORT_MESSAGES = "support_messages"
    OCR_RECORDS = "ocr_records"
    IMAGE_MANIFEST = "image_manifest"
    ANNOTATIONS = "annotations"
    README = "readme"
    OTHER = "other"


_NAME_TO_KIND: dict[str, ArchiveEntryKind] = {
    "transactions.csv": ArchiveEntryKind.TRANSACTIONS,
    "predictions.jsonl": ArchiveEntryKind.PREDICTIONS,
    "predictions.csv": ArchiveEntryKind.PREDICTIONS,
    "support_messages.jsonl": ArchiveEntryKind.SUPPORT_MESSAGES,
    "ocr_records.jsonl": ArchiveEntryKind.OCR_RECORDS,
    "image_manifest.jsonl": ArchiveEntryKind.IMAGE_MANIFEST,
    "annotations_sample.json": ArchiveEntryKind.ANNOTATIONS,
    "readme.md": ArchiveEntryKind.README,
}


@dataclass(frozen=True)
class ArchiveFileDescriptor:
    """Safe descriptor for one archive entry without loading payload bytes."""

    name: str
    kind: ArchiveEntryKind
    media_type: str
    file_size: int
    compressed_size: int
    archive_member: zipfile.ZipInfo
    _zip: zipfile.ZipFile

    @contextmanager
    def open(self) -> Iterator[IO[bytes]]:
        """Open the entry as a streaming binary file handle.

        Memory footprint is bounded by ``zipfile.ZipFile.open`` chunked reads;
        the whole entry is never materialized at once.
        """
        with self._zip.open(self.archive_member, mode="r") as handle:
            yield handle

    def read_bytes(self) -> bytes:
        """Read the entry into memory.

        Use this only for small, expected-tiny entries (README, summary
        manifests). For tabular and JSONL data prefer ``open()`` plus a
        streaming reader so large datasets do not blow up memory.
        """
        with self.open() as handle:
            return handle.read()

    def stream_chunks(self, *, chunk_size: int = _STREAM_CHUNK_SIZE) -> Iterator[bytes]:
        """Yield raw bytes from the entry in fixed-size chunks."""
        with self.open() as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    return
                yield chunk

    def sha256(self) -> str:
        """Compute a content sha256 by streaming the entry, never loading it whole."""
        digest = hashlib.sha256()
        for chunk in self.stream_chunks():
            digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class ArchiveContents:
    """Discovered archive entries and the underlying safety report."""

    descriptors: tuple[ArchiveFileDescriptor, ...]
    safety_report: ArchiveSafetyReport

    def descriptors_by_kind(self, kind: ArchiveEntryKind) -> tuple[ArchiveFileDescriptor, ...]:
        return tuple(d for d in self.descriptors if d.kind is kind)

    def find_required_transactions(self) -> ArchiveFileDescriptor:
        for descriptor in self.descriptors:
            if descriptor.kind is ArchiveEntryKind.TRANSACTIONS:
                return descriptor
        raise ArchiveSafetyError(
            code=ErrorCode.INVALID_ARCHIVE_STRUCTURE,
            reason_code="missing_required_entry",
            message="Archive does not contain the required transactions.csv entry.",
        )


class ArchiveReader:
    """Context-managed lazy archive reader.

    The reader owns one ``zipfile.ZipFile`` instance for the lifetime of the
    context. ``ArchiveFileDescriptor`` objects are valid only while the
    reader is open; using them after ``__exit__`` raises ``ValueError`` from
    ``zipfile``.
    """

    def __init__(
        self,
        *,
        archive: zipfile.ZipFile,
        contents: ArchiveContents,
    ) -> None:
        self._archive = archive
        self._contents = contents

    @property
    def contents(self) -> ArchiveContents:
        return self._contents

    def descriptors(self) -> tuple[ArchiveFileDescriptor, ...]:
        return self._contents.descriptors

    def find_required_transactions(self) -> ArchiveFileDescriptor:
        return self._contents.find_required_transactions()

    def __enter__(self) -> ArchiveReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._archive.close()


def open_archive_path(
    archive_path: Path,
    *,
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveReader:
    """Validate and open an archive on disk for streaming reads."""
    safety_report = validate_archive_path(archive_path, policy=policy)
    archive = zipfile.ZipFile(archive_path, mode="r")
    return _build_reader(archive=archive, safety_report=safety_report)


def open_archive_bytes(
    archive_bytes: bytes,
    *,
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveReader:
    """Validate and open in-memory archive bytes for streaming reads."""
    safety_report = validate_archive_bytes(archive_bytes, policy=policy)
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r")
    return _build_reader(archive=archive, safety_report=safety_report)


def open_archive_artifact(
    *,
    storage: MinioObjectStorageAdapter,
    artifact_uri: str,
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveReader:
    """Validate and open an archive identified by a scoped object-storage URI."""
    safety_report = validate_archive_artifact(
        storage=storage,
        artifact_uri=artifact_uri,
        policy=policy,
    )
    stored = storage.get(artifact_uri)
    archive = zipfile.ZipFile(io.BytesIO(stored.data), mode="r")
    return _build_reader(archive=archive, safety_report=safety_report)


def _build_reader(
    *,
    archive: zipfile.ZipFile,
    safety_report: ArchiveSafetyReport,
) -> ArchiveReader:
    descriptors = tuple(
        _descriptor_for_member(archive, info)
        for info in archive.infolist()
        if not info.is_dir()
    )
    if not any(d.kind is ArchiveEntryKind.TRANSACTIONS for d in descriptors):
        archive.close()
        raise ArchiveSafetyError(
            code=ErrorCode.INVALID_ARCHIVE_STRUCTURE,
            reason_code="missing_required_entry",
            message="Archive does not contain the required transactions.csv entry.",
        )
    contents = ArchiveContents(descriptors=descriptors, safety_report=safety_report)
    return ArchiveReader(archive=archive, contents=contents)


def _descriptor_for_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> ArchiveFileDescriptor:
    name = posixpath.normpath(info.filename)
    base = posixpath.basename(name).lower()
    kind = _NAME_TO_KIND.get(base, ArchiveEntryKind.OTHER)
    return ArchiveFileDescriptor(
        name=name,
        kind=kind,
        media_type=_media_type_for_name(base),
        file_size=info.file_size,
        compressed_size=info.compress_size,
        archive_member=info,
        _zip=archive,
    )


def _media_type_for_name(base: str) -> str:
    if base.endswith(".csv") or base.endswith(".tsv"):
        return "text/csv"
    if base.endswith(".jsonl") or base.endswith(".ndjson"):
        return "application/jsonl"
    if base.endswith(".json"):
        return "application/json"
    if base.endswith(".parquet"):
        return "application/x-parquet"
    if base.endswith(".md"):
        return "text/markdown"
    if base.endswith(".yaml") or base.endswith(".yml"):
        return "application/yaml"
    if base.endswith(".xml"):
        return "application/xml"
    return "application/octet-stream"


__all__ = [
    "ArchiveContents",
    "ArchiveEntryKind",
    "ArchiveFileDescriptor",
    "ArchiveReader",
    "open_archive_artifact",
    "open_archive_bytes",
    "open_archive_path",
]
