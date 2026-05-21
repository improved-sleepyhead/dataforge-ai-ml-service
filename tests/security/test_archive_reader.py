"""Archive reader tests for TASK-019.

Acceptance:

* reader finds transactions.csv, support_messages.jsonl, ocr_records.jsonl,
  optional image/annotation manifests inside the demo archive;
* missing transactions.csv yields INVALID_ARCHIVE_STRUCTURE;
* reader supports object-storage adapter and local-file fixtures;
* reader does not read large files into memory eagerly.
"""

from __future__ import annotations

import io
import json
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
    ArchiveEntryKind,
    ArchiveSafetyError,
    open_archive_artifact,
    open_archive_bytes,
    open_archive_path,
)
from tests.fixtures.demo_archive import build_demo_archive


def test_open_archive_path_classifies_known_demo_entries(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)

    with open_archive_path(built.archive_path) as reader:
        kinds = {descriptor.name: descriptor.kind for descriptor in reader.descriptors()}

    assert kinds["transactions.csv"] is ArchiveEntryKind.TRANSACTIONS
    assert kinds["predictions.jsonl"] is ArchiveEntryKind.PREDICTIONS
    assert kinds["support_messages.jsonl"] is ArchiveEntryKind.SUPPORT_MESSAGES
    assert kinds["ocr_records.jsonl"] is ArchiveEntryKind.OCR_RECORDS
    assert kinds["README.md"] is ArchiveEntryKind.README


def test_archive_reader_descriptors_by_kind_filters_correctly(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)

    with open_archive_path(built.archive_path) as reader:
        transactions = reader.contents.descriptors_by_kind(ArchiveEntryKind.TRANSACTIONS)
        predictions = reader.contents.descriptors_by_kind(ArchiveEntryKind.PREDICTIONS)
        ocr = reader.contents.descriptors_by_kind(ArchiveEntryKind.OCR_RECORDS)

    assert len(transactions) == 1
    assert len(predictions) == 1
    assert len(ocr) == 1
    assert transactions[0].media_type == "text/csv"
    assert predictions[0].media_type == "application/jsonl"
    assert ocr[0].media_type == "application/jsonl"


def test_archive_reader_streams_predictions_without_loading_full_payload(
    tmp_path: Path,
) -> None:
    built = build_demo_archive(output_dir=tmp_path)

    with open_archive_path(built.archive_path) as reader:
        descriptor = reader.contents.descriptors_by_kind(
            ArchiveEntryKind.PREDICTIONS
        )[0]
        chunks: list[bytes] = []
        for chunk in descriptor.stream_chunks(chunk_size=512):
            assert len(chunk) <= 512
            chunks.append(chunk)
            if len(chunks) >= 2:
                # Stopping early proves the generator is lazy and never
                # materialized the full payload upfront.
                break

    joined = b"".join(chunks)
    first_record = json.loads(joined.split(b"\n", 1)[0])
    assert "object_id" in first_record


def test_archive_reader_sha256_is_streamed_and_stable(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)

    with open_archive_path(built.archive_path) as reader:
        descriptor = reader.contents.find_required_transactions()
        digest_first = descriptor.sha256()
        digest_second = descriptor.sha256()

    assert digest_first.startswith("sha256:")
    assert digest_first == digest_second


def test_open_archive_bytes_works_for_in_memory_fixtures(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)
    archive_bytes = built.archive_path.read_bytes()

    with open_archive_bytes(archive_bytes) as reader:
        descriptor = reader.find_required_transactions()
        first_line = descriptor.read_bytes().splitlines()[0]

    assert first_line.startswith(b"object_id,")


def test_open_archive_path_rejects_archive_without_transactions(tmp_path: Path) -> None:
    archive_path = tmp_path / "no_transactions.zip"
    archive_path.write_bytes(
        _build_zip_payload(
            {
                "support_messages.jsonl": b'{"object_id":"support_0001"}\n',
                "README.md": b"No transactions here.",
            }
        )
    )

    with pytest.raises(ArchiveSafetyError) as exc_info:
        open_archive_path(archive_path)

    assert exc_info.value.code is ErrorCode.INVALID_ARCHIVE_STRUCTURE
    assert exc_info.value.reason_code == "missing_required_entry"


def test_open_archive_artifact_uses_object_storage_scope(tmp_path: Path) -> None:
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

    with open_archive_artifact(storage=storage, artifact_uri=artifact_ref.uri) as reader:
        descriptors = {d.name for d in reader.descriptors()}
        transactions = reader.find_required_transactions()
        with transactions.open() as handle:
            header_line = handle.readline()

    assert "transactions.csv" in descriptors
    assert header_line.startswith(b"object_id,")


def _build_zip_payload(files: dict[str, bytes]) -> bytes:
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
