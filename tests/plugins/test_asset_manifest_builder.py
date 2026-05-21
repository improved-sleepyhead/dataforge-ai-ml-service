"""Asset Manifest builder tests for TASK-021.

Acceptance:

* builder produces a versioned JSONL manifest artifact;
* every row carries object_id, dataset_id, version_id, modality, hash,
  metadata, and lineage;
* link keys (case_id, customer_id_hash, document_id, ...) are surfaced
  into row metadata when the source record exposes them;
* unsupported assets are reported, not silently dropped;
* tabular, text, document_ocr, and skeleton modalities are produced.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters import (
    ArtifactRegistry,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import DataModality, ErrorCode, ManifestRow
from app.ingestion import (
    ArchiveEntryKind,
    BuildManifestRequest,
    build_asset_manifest,
    manifest_rows_from_artifact,
    open_archive_path,
)
from tests.fixtures.demo_archive import build_demo_archive

_REQUEST = BuildManifestRequest(
    dataset_id="dataset_demo",
    version_id="version_demo",
    parent_version_id="version_demo_parent",
    created_by_job_id="compute_run_test",
    config_hash="sha256:" + "a" * 64,
)


def test_build_asset_manifest_for_demo_archive_yields_versioned_jsonl_artifact(
    tmp_path: Path,
) -> None:
    archive_path, registry, storage = _archive_and_registry(tmp_path)

    with open_archive_path(archive_path) as reader:
        result = build_asset_manifest(reader, request=_REQUEST, registry=registry)

    assert result.manifest_artifact.format == "jsonl"
    assert result.manifest_artifact.schema_version == "manifest_row.v1"
    assert result.manifest_artifact.created_by_job_id == "compute_run_test"
    assert result.total_rows == sum(result.rows_by_modality.values())
    assert result.rows_by_modality[DataModality.TABULAR] >= 1
    assert result.rows_by_modality[DataModality.TEXT] >= 1
    assert result.rows_by_modality[DataModality.DOCUMENT_OCR] >= 1
    assert DataModality.MULTIMODAL not in result.rows_by_modality

    stored = storage.get(result.manifest_artifact.uri)
    rows = manifest_rows_from_artifact(stored.data)
    assert len(rows) == result.total_rows


def test_manifest_rows_carry_required_identity_and_lineage_fields(tmp_path: Path) -> None:
    archive_path, registry, storage = _archive_and_registry(tmp_path)

    with open_archive_path(archive_path) as reader:
        result = build_asset_manifest(reader, request=_REQUEST, registry=registry)

    rows = manifest_rows_from_artifact(storage.get(result.manifest_artifact.uri).data)
    assert all(isinstance(row, ManifestRow) for row in rows)
    assert all(row.dataset_id == "dataset_demo" for row in rows)
    assert all(row.version_id == "version_demo" for row in rows)
    assert all(row.lineage.parent_version_id == "version_demo_parent" for row in rows)
    assert all(row.lineage.created_by_job_id == "compute_run_test" for row in rows)
    assert all(row.lineage.config_hash.startswith("sha256:") for row in rows)
    assert all(row.hash.startswith("sha256:") for row in rows)
    assert all(row.object_id.startswith("obj_") for row in rows)


def test_tabular_rows_surface_link_keys_into_metadata(tmp_path: Path) -> None:
    archive_path, registry, storage = _archive_and_registry(tmp_path)

    with open_archive_path(archive_path) as reader:
        result = build_asset_manifest(reader, request=_REQUEST, registry=registry)

    rows = manifest_rows_from_artifact(storage.get(result.manifest_artifact.uri).data)
    tabular_rows = [row for row in rows if row.modality is DataModality.TABULAR]
    assert tabular_rows, "demo archive must produce at least one tabular row"

    sample = tabular_rows[0]
    assert "case_id" in sample.metadata
    assert "customer_id_hash" in sample.metadata
    assert sample.label in {"0", "1"}


def test_text_and_ocr_rows_use_correct_modalities_and_source_system(
    tmp_path: Path,
) -> None:
    archive_path, registry, storage = _archive_and_registry(tmp_path)

    with open_archive_path(archive_path) as reader:
        result = build_asset_manifest(reader, request=_REQUEST, registry=registry)

    rows = manifest_rows_from_artifact(storage.get(result.manifest_artifact.uri).data)
    text_rows = [row for row in rows if row.modality is DataModality.TEXT]
    ocr_rows = [row for row in rows if row.modality is DataModality.DOCUMENT_OCR]

    assert all(row.source_system == "support_messages" for row in text_rows)
    assert all(row.source_system == "ocr_records" for row in ocr_rows)
    assert all("language" in row.metadata for row in text_rows)


def test_unsupported_archive_entries_are_reported_not_dropped(tmp_path: Path) -> None:
    base = build_demo_archive(output_dir=tmp_path / "demo")
    augmented_path = tmp_path / "demo_with_extra.zip"

    augmented_path.write_bytes(
        _augment_archive(
            base.archive_path.read_bytes(),
            extra_files={"extra/payload.csv": b"id,value\n1,foo\n"},
        )
    )

    archive_path, registry, storage = _archive_and_registry(tmp_path)
    # Build manifest from augmented archive instead of the canonical demo one.
    with open_archive_path(augmented_path) as reader:
        result = build_asset_manifest(reader, request=_REQUEST, registry=registry)

    assert any(
        entry.kind is ArchiveEntryKind.OTHER and entry.name.endswith("payload.csv")
        for entry in result.unsupported_entries
    )
    # Unsupported entries do not appear in the manifest itself.
    rows = manifest_rows_from_artifact(storage.get(result.manifest_artifact.uri).data)
    assert all("extra/payload.csv" not in row.asset_uri for row in rows)


def test_repeated_manifest_build_is_idempotent(tmp_path: Path) -> None:
    archive_path, registry, storage = _archive_and_registry(tmp_path)

    with open_archive_path(archive_path) as reader:
        first = build_asset_manifest(reader, request=_REQUEST, registry=registry)
    with open_archive_path(archive_path) as reader:
        second = build_asset_manifest(reader, request=_REQUEST, registry=registry)

    assert first.manifest_artifact.uri == second.manifest_artifact.uri
    assert first.manifest_artifact.hash == second.manifest_artifact.hash
    assert first.total_rows == second.total_rows


def _archive_and_registry(
    tmp_path: Path,
) -> tuple[Path, ArtifactRegistry, MinioObjectStorageAdapter]:
    built = build_demo_archive(output_dir=tmp_path)
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_test",
            project_id="project_test",
            dataset_id="dataset_demo",
        ),
    )
    registry = ArtifactRegistry(storage=storage)
    return built.archive_path, registry, storage


def _augment_archive(archive_bytes: bytes, *, extra_files: dict[str, bytes]) -> bytes:
    """Append additional zip entries to an existing zip archive."""
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as src:
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as dst:
            for member in src.infolist():
                dst.writestr(member, src.read(member.filename))
            for name, payload in extra_files.items():
                dst.writestr(name, payload)
    return output.getvalue()


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
        record = self._object(Bucket, Key)
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
        record = self._object(Bucket, Key)
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

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
