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
from app.ingestion.manifest_builder import (
    BuildManifestRequest,
    BuildManifestResult,
    UnsupportedEntry,
    build_asset_manifest,
    manifest_rows_from_artifact,
)
from app.ingestion.manifest_validation import (
    MANIFEST_ROW_SCHEMA_NAME,
    VALIDATED_MANIFEST_FORMAT,
    VALIDATED_MANIFEST_KIND,
    VALIDATED_MANIFEST_MEDIA_TYPE,
    VALIDATED_MANIFEST_SCHEMA_VERSION,
    BuildValidatedManifestResult,
    ManifestContractValidationError,
    ManifestValidationReport,
    build_validated_manifest,
    iter_manifest_rows,
    validate_manifest_jsonl,
)

__all__ = [
    "ArchiveContents",
    "ArchiveEntryKind",
    "ArchiveFileDescriptor",
    "ArchiveReader",
    "ArchiveSafetyError",
    "ArchiveSafetyPolicy",
    "ArchiveSafetyReport",
    "BuildManifestRequest",
    "BuildManifestResult",
    "BuildValidatedManifestResult",
    "MANIFEST_ROW_SCHEMA_NAME",
    "ManifestContractValidationError",
    "ManifestValidationReport",
    "UnsupportedEntry",
    "VALIDATED_MANIFEST_FORMAT",
    "VALIDATED_MANIFEST_KIND",
    "VALIDATED_MANIFEST_MEDIA_TYPE",
    "VALIDATED_MANIFEST_SCHEMA_VERSION",
    "build_asset_manifest",
    "build_validated_manifest",
    "compute_content_sha256",
    "compute_record_sha256",
    "derive_object_id",
    "iter_manifest_rows",
    "manifest_rows_from_artifact",
    "open_archive_artifact",
    "open_archive_bytes",
    "open_archive_path",
    "validate_archive_artifact",
    "validate_archive_bytes",
    "validate_archive_path",
    "validate_manifest_jsonl",
]
