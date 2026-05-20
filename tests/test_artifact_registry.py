"""Artifact registry tests for immutable compute artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError
from app.domain import ErrorCode


def test_artifact_registry_saves_manifest_artifact_ref_and_metadata() -> None:
    client = InMemoryS3Client()
    storage = _storage(client)
    registry = ArtifactRegistry(storage=storage)
    data = b'{"object_id":"obj_1","dataset_id":"dataset_1"}\n'

    record = registry.save_artifact(
        artifact_kind="asset_manifest",
        data=data,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="manifest_row.v1",
        dataset_version_id="version_1",
        created_by_job_id="compute_run_001",
        config_hash="sha256:" + "c" * 64,
    )

    assert record.artifact_kind == "asset_manifest"
    assert record.format == "jsonl"
    assert record.schema_version == "manifest_row.v1"
    assert record.created_by_job_id == "compute_run_001"
    assert record.hash == (
        "sha256:bd73541c6fb8406fc2b895f379066606a4a8f03085af2a7e89bb0cb960a2f9b0"
    )
    assert record.uri.endswith(
        "/versions/version_1/jobs/compute_run_001/artifacts/asset_manifest/"
        "manifest_row.v1/bd73541c6fb8406fc2b895f379066606a4a8f03085af2a7e89bb0cb960a2f9b0.jsonl"
    )

    head = storage.head(record.uri)
    assert head.hash == record.hash
    assert head.metadata["artifact-kind"] == "asset_manifest"
    assert head.metadata["artifact-format"] == "jsonl"
    assert head.metadata["schema-version"] == "manifest_row.v1"
    assert head.metadata["created-by-job-id"] == "compute_run_001"
    assert head.metadata["dataset-version-id"] == "version_1"
    assert client.object_count == 1


def test_artifact_registry_repeated_save_of_same_immutable_artifact_is_idempotent() -> None:
    client = InMemoryS3Client()
    registry = ArtifactRegistry(storage=_storage(client))
    data = b'{"object_id":"obj_1"}\n'

    first = registry.save_artifact(
        artifact_kind="asset_manifest",
        data=data,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="manifest_row.v1",
        dataset_version_id="version_1",
        created_by_job_id="compute_run_001",
        config_hash="sha256:" + "c" * 64,
    )
    second = registry.save_artifact(
        artifact_kind="asset_manifest",
        data=data,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="manifest_row.v1",
        dataset_version_id="version_1",
        created_by_job_id="compute_run_001",
        config_hash="sha256:" + "c" * 64,
    )

    assert second.uri == first.uri
    assert second.hash == first.hash
    assert second.artifact_ref.size_bytes == first.artifact_ref.size_bytes
    assert client.object_count == 1


def _storage(client: InMemoryS3Client) -> MinioObjectStorageAdapter:
    return MinioObjectStorageAdapter(
        client=client,
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
        ),
    )


class InMemoryS3Client:
    """Small MinIO/S3-compatible fake for object operations only."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

    @property
    def object_count(self) -> int:
        return len(self._objects)

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

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        stored = self._object(Bucket, Key)
        body = stored["Body"]
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        stored = self._object(Bucket, Key)
        body = stored["Body"]
        return {
            "ContentLength": len(body),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> dict[str, Any]:
        contents = [
            {"Key": key, "Size": len(stored["Body"])}
            for (bucket, key), stored in sorted(self._objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents}

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
