"""Archive safety validation for the DataForge AI ingestion layer.

This module enforces hard safety rules on user-provided archives before any
plugin or asset can read their contents. It is intentionally pure: it
inspects archive bytes/paths through stdlib only, never executes archive
entries, never follows symlinks, and never copies raw payload content into
logs or HTTP responses.

The policy enforces:

* maximum compressed archive size on disk;
* maximum total uncompressed payload size (zip-bomb guard);
* maximum number of files inside the archive;
* allowlisted file extensions (case-insensitive);
* path safety: no absolute paths, no path traversal (``..``), no symlinks,
  no Windows drive letters, no NUL bytes.

Violations raise :class:`ArchiveSafetyError` with a stable
:class:`ErrorCode` so callers can map errors to safe ``ErrorResponse``
bodies without leaking raw archive contents.
"""

from __future__ import annotations

import posixpath
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ConfigDict, Field

from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import ErrorCode

# Conservative MVP defaults. Production deployments override these from
# config policy files; the defaults here are tuned for the demo archive
# generator (TASK-017) plus a generous safety margin.
_DEFAULT_MAX_COMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MiB
_DEFAULT_MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  # 200 MiB
_DEFAULT_MAX_FILE_COUNT = 5_000
_DEFAULT_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "csv",
        "tsv",
        "parquet",
        "json",
        "jsonl",
        "ndjson",
        "txt",
        "md",
        "yaml",
        "yml",
        "xml",
    }
)


class ArchiveSafetyError(ValueError):
    """Raised when an archive violates structure or safety rules.

    ``code`` is a stable :class:`ErrorCode` for API mapping. ``reason_code``
    is a finer-grained machine-readable label used by tests and audit, and
    must not contain raw payload content.
    """

    def __init__(
        self,
        *,
        code: ErrorCode,
        reason_code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason_code = reason_code


class ArchiveSafetyPolicy(BaseModel):
    """Bounded safety policy applied before any archive content is read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_compressed_bytes: int = Field(
        default=_DEFAULT_MAX_COMPRESSED_BYTES,
        ge=1,
    )
    max_uncompressed_bytes: int = Field(
        default=_DEFAULT_MAX_UNCOMPRESSED_BYTES,
        ge=1,
    )
    max_file_count: int = Field(
        default=_DEFAULT_MAX_FILE_COUNT,
        ge=1,
    )
    allowed_extensions: frozenset[str] = Field(
        default=_DEFAULT_ALLOWED_EXTENSIONS,
    )


@dataclass(frozen=True)
class ArchiveSafetyReport:
    """Validated archive metadata suitable for safe logging."""

    file_count: int
    compressed_bytes: int
    uncompressed_bytes: int
    safe_entries: tuple[str, ...]


def validate_archive_path(
    archive_path: Path,
    *,
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Validate a zip archive at ``archive_path`` against the safety policy."""
    effective_policy = policy or ArchiveSafetyPolicy()

    if not archive_path.is_file():
        raise ArchiveSafetyError(
            code=ErrorCode.INVALID_ARCHIVE_STRUCTURE,
            reason_code="archive_missing",
            message="Archive file is missing or is not a regular file.",
        )

    compressed_size = archive_path.stat().st_size
    if compressed_size > effective_policy.max_compressed_bytes:
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="compressed_size_exceeded",
            message="Archive compressed size exceeds the safety policy limit.",
        )

    try:
        with archive_path.open("rb") as handle:
            return _validate_zip_stream(
                handle,
                policy=effective_policy,
                compressed_size=compressed_size,
            )
    except zipfile.BadZipFile as exc:
        raise ArchiveSafetyError(
            code=ErrorCode.INVALID_ARCHIVE_STRUCTURE,
            reason_code="bad_zip_file",
            message="Archive is not a valid zip file.",
        ) from exc


def validate_archive_bytes(
    archive_bytes: bytes,
    *,
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Validate raw archive bytes against the safety policy.

    Useful for streaming validation when bytes are obtained from object
    storage and have not been written to disk yet.
    """
    import io

    return validate_archive_seekable(
        io.BytesIO(archive_bytes),
        compressed_size=len(archive_bytes),
        policy=policy,
    )


def validate_archive_seekable(
    stream: IO[bytes],
    *,
    compressed_size: int,
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Validate a seekable zip stream against the safety policy."""
    effective_policy = policy or ArchiveSafetyPolicy()
    if compressed_size > effective_policy.max_compressed_bytes:
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="compressed_size_exceeded",
            message="Archive compressed size exceeds the safety policy limit.",
        )

    try:
        return _validate_zip_stream(
            stream,
            policy=effective_policy,
            compressed_size=compressed_size,
        )
    except zipfile.BadZipFile as exc:
        raise ArchiveSafetyError(
            code=ErrorCode.INVALID_ARCHIVE_STRUCTURE,
            reason_code="bad_zip_file",
            message="Archive is not a valid zip file.",
        ) from exc


def validate_archive_artifact(
    *,
    storage: MinioObjectStorageAdapter,
    artifact_uri: str,
    policy: ArchiveSafetyPolicy | None = None,
) -> ArchiveSafetyReport:
    """Validate an archive identified by its scoped object-storage URI.

    The adapter enforces tenant/project/dataset prefix scoping, so passing
    out-of-scope URIs already fails before this function reads any bytes.
    """
    stored = storage.download_to_seekable(artifact_uri)
    try:
        return validate_archive_seekable(
            stored.file,
            compressed_size=stored.info.size_bytes,
            policy=policy,
        )
    finally:
        stored.file.close()


def _validate_zip_stream(
    stream: IO[bytes],
    *,
    policy: ArchiveSafetyPolicy,
    compressed_size: int,
) -> ArchiveSafetyReport:
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) > policy.max_file_count:
            raise ArchiveSafetyError(
                code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
                reason_code="file_count_exceeded",
                message="Archive contains too many files.",
            )

        total_uncompressed = 0
        safe_entries: list[str] = []
        for info in infos:
            _validate_entry_path(info)
            _validate_entry_attributes(info)
            if info.is_dir():
                # Directory entries are skipped from extension/size accounting.
                continue
            _validate_entry_extension(info, policy=policy)

            total_uncompressed += info.file_size
            if total_uncompressed > policy.max_uncompressed_bytes:
                raise ArchiveSafetyError(
                    code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
                    reason_code="uncompressed_size_exceeded",
                    message="Archive uncompressed size exceeds the safety policy limit.",
                )
            safe_entries.append(_normalized_entry_name(info))

        return ArchiveSafetyReport(
            file_count=len(safe_entries),
            compressed_bytes=compressed_size,
            uncompressed_bytes=total_uncompressed,
            safe_entries=tuple(safe_entries),
        )


def _validate_entry_path(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not name:
        raise ArchiveSafetyError(
            code=ErrorCode.INVALID_ARCHIVE_STRUCTURE,
            reason_code="empty_entry_name",
            message="Archive contains an entry with an empty name.",
        )
    if "\x00" in name:
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="nul_byte_in_entry",
            message="Archive entry contains an invalid null byte.",
        )
    if name.startswith("/"):
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="absolute_path",
            message="Archive entry uses an absolute path.",
        )
    if len(name) >= 3 and name[1] == ":" and name[0].isalpha():
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="windows_drive_path",
            message="Archive entry uses a Windows drive path.",
        )

    normalized = posixpath.normpath(name)
    if normalized.startswith("..") or "/../" in f"/{normalized}/":
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="path_traversal",
            message="Archive entry attempts path traversal outside the archive root.",
        )


def _validate_entry_attributes(info: zipfile.ZipInfo) -> None:
    # External attribute upper 16 bits encode the file mode under Unix.
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="symlink_entry",
            message="Archive entry is a symbolic link.",
        )


def _validate_entry_extension(
    info: zipfile.ZipInfo,
    *,
    policy: ArchiveSafetyPolicy,
) -> None:
    name = posixpath.basename(info.filename)
    if "." not in name:
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="missing_extension",
            message="Archive entry has no file extension.",
        )
    extension = name.rsplit(".", 1)[1].lower()
    if extension not in policy.allowed_extensions:
        raise ArchiveSafetyError(
            code=ErrorCode.ARCHIVE_SAFETY_VIOLATION,
            reason_code="forbidden_extension",
            message="Archive entry uses a file extension that is not allowlisted.",
        )


def _normalized_entry_name(info: zipfile.ZipInfo) -> str:
    return posixpath.normpath(info.filename)


__all__ = [
    "ArchiveSafetyError",
    "ArchiveSafetyPolicy",
    "ArchiveSafetyReport",
    "validate_archive_artifact",
    "validate_archive_bytes",
    "validate_archive_path",
    "validate_archive_seekable",
]
