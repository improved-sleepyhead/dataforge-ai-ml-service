"""Ingestion layer for the DataForge AI compute plane."""

from app.ingestion.archive_reader import (
    ArchiveContents,
    ArchiveEntryKind,
    ArchiveFileDescriptor,
    ArchiveReader,
    open_archive_artifact,
    open_archive_bytes,
    open_archive_path,
)
from app.ingestion.archive_safety import (
    ArchiveSafetyError,
    ArchiveSafetyPolicy,
    ArchiveSafetyReport,
    validate_archive_artifact,
    validate_archive_bytes,
    validate_archive_path,
)
from app.ingestion.identity import (
    compute_content_sha256,
    compute_record_sha256,
    derive_object_id,
)

__all__ = [
    "ArchiveContents",
    "ArchiveEntryKind",
    "ArchiveFileDescriptor",
    "ArchiveReader",
    "ArchiveSafetyError",
    "ArchiveSafetyPolicy",
    "ArchiveSafetyReport",
    "compute_content_sha256",
    "compute_record_sha256",
    "derive_object_id",
    "open_archive_artifact",
    "open_archive_bytes",
    "open_archive_path",
    "validate_archive_artifact",
    "validate_archive_bytes",
    "validate_archive_path",
]
