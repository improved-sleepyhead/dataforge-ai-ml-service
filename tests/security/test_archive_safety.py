"""Archive safety validation tests for TASK-018.

Acceptance:

* compressed/uncompressed/file-count limits enforced;
* path traversal, zip slip, symlink and forbidden extensions rejected;
* invalid archives raise INVALID_ARCHIVE_STRUCTURE or ARCHIVE_SAFETY_VIOLATION;
* validation works against locally generated demo archives and
  object-storage artifact refs (via the in-memory MinIO fake).
"""

from __future__ import annotations

import io
import stat
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import S3CompatibleClient
from app.domain import ErrorCode
from app.ingestion import (
    ArchiveSafetyError,
    ArchiveSafetyPolicy,
    validate_archive_artifact,
    validate_archive_bytes,
    validate_archive_path,
)
from tests.fixtures.demo_archive import build_demo_archive


def test_demo_archive_passes_safety_validation(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)

    report = validate_archive_path(built.archive_path)

    assert report.file_count == 5
    assert report.compressed_bytes == built.archive_path.stat().st_size
    assert report.uncompressed_bytes > 0
    assert "transactions.csv" in report.safe_entries


def test_archive_with_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive_bytes = _build_zip(
        {
            "../escape.csv": b"object_id\nrow_1\n",
        }
    )

    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(archive_bytes)

    assert exc_info.value.code is ErrorCode.ARCHIVE_SAFETY_VIOLATION
    assert exc_info.value.reason_code == "path_traversal"


def test_archive_with_absolute_path_is_rejected() -> None:
    archive_bytes = _build_zip(
        {
            "/etc/passwd.txt": b"root:x:0:0:root:/root:/bin/bash\n",
        }
    )

    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(archive_bytes)

    assert exc_info.value.code is ErrorCode.ARCHIVE_SAFETY_VIOLATION
    assert exc_info.value.reason_code == "absolute_path"


def test_archive_with_forbidden_extension_is_rejected() -> None:
    archive_bytes = _build_zip(
        {
            "transactions.csv": b"object_id\nrow_1\n",
            "malicious_payload.exe": b"\x00fake-binary",
        }
    )

    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(archive_bytes)

    assert exc_info.value.code is ErrorCode.ARCHIVE_SAFETY_VIOLATION
    assert exc_info.value.reason_code == "forbidden_extension"


def test_archive_with_symlink_entry_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo("link_to_etc.csv")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "/etc/passwd")
    archive_bytes = buffer.getvalue()

    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(archive_bytes)

    assert exc_info.value.code is ErrorCode.ARCHIVE_SAFETY_VIOLATION
    assert exc_info.value.reason_code == "symlink_entry"


def test_archive_compressed_size_limit_is_enforced() -> None:
    archive_bytes = _build_zip(
        {
            "transactions.csv": b"object_id\nrow_1\n",
        }
    )
    policy = ArchiveSafetyPolicy(max_compressed_bytes=10)

    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(archive_bytes, policy=policy)

    assert exc_info.value.code is ErrorCode.ARCHIVE_SAFETY_VIOLATION
    assert exc_info.value.reason_code == "compressed_size_exceeded"


def test_archive_uncompressed_size_limit_is_enforced() -> None:
    archive_bytes = _build_zip(
        {
            "transactions.csv": b"a" * 10_000,
        }
    )
    policy = ArchiveSafetyPolicy(max_uncompressed_bytes=1_000)

    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(archive_bytes, policy=policy)

    assert exc_info.value.code is ErrorCode.ARCHIVE_SAFETY_VIOLATION
    assert exc_info.value.reason_code == "uncompressed_size_exceeded"


def test_archive_file_count_limit_is_enforced() -> None:
    files = {f"part_{idx:03d}.csv": b"object_id\n" for idx in range(20)}
    archive_bytes = _build_zip(files)
    policy = ArchiveSafetyPolicy(max_file_count=5)

    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(archive_bytes, policy=policy)

    assert exc_info.value.code is ErrorCode.ARCHIVE_SAFETY_VIOLATION
    assert exc_info.value.reason_code == "file_count_exceeded"


def test_invalid_zip_bytes_return_invalid_archive_structure_error() -> None:
    with pytest.raises(ArchiveSafetyError) as exc_info:
        validate_archive_bytes(b"this is not a zip archive")

    assert exc_info.value.code is ErrorCode.INVALID_ARCHIVE_STRUCTURE
    assert exc_info.value.reason_code == "bad_zip_file"


def test_validate_archive_artifact_uses_object_storage_scope(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)
    archive_bytes = built.archive_path.read_bytes()

    storage_client = _InMemoryS3Client()
    storage = MinioObjectStorageAdapter(
        client=storage_client,
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_test",
            project_id="project_test",
            dataset_id="dataset_test",
        ),
    )
    artifact_ref = storage.put(
        object_name="versions/dataset_version_1/raw/demo_archive.zip",
        data=archive_bytes,
        kind="raw_archive",
        media_type="application/zip",
        schema_version="raw_archive.v1",
        parent_version_id="dataset_version_1",
        job_id="compute_run_001",
        config_hash="sha256:" + "a" * 64,
    )

    report = validate_archive_artifact(storage=storage, artifact_uri=artifact_ref.uri)

    assert report.file_count == 5


def _build_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            info = zipfile.ZipInfo(filename=name)
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


class _InMemoryS3Client(S3CompatibleClient):
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
            "LastModified": datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._objects[(Bucket, Key)]
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._objects[(Bucket, Key)]
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(record["Body"])}
                for (bucket, key), record in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }
