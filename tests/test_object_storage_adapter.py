"""Object storage adapter tests using an in-memory MinIO/S3-compatible fake."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import pytest

from app.adapters import MinioObjectStorageAdapter, ObjectStorageError, ObjectStorageScope
from app.domain import ErrorCode


def test_minio_object_storage_adapter_put_get_head_and_list_scoped_artifact() -> None:
    client = InMemoryS3Client()
    adapter = _adapter(client)
    data = b'{"object_id":"obj_1"}\n'

    artifact_ref = adapter.put(
        object_name="dataset_version_1/manifests/asset_manifest.jsonl",
        data=data,
        kind="asset_manifest",
        media_type="application/jsonl",
        schema_version="manifest_row.v1",
        parent_version_id="dataset_version_1",
        job_id="compute_run_001",
        config_hash="sha256:" + "b" * 64,
        metadata={"source": "unit-test"},
    )

    assert artifact_ref.uri.startswith(
        "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/"
    )
    assert artifact_ref.hash == "sha256:" + (
        "0226a06c29714e757154459b1efddd379c37cba9b6a4f2df95206aa08e9b45c9"
    )
    assert artifact_ref.size_bytes == len(data)

    stored = adapter.get(artifact_ref.uri)
    head = adapter.head(artifact_ref.uri)
    listed = adapter.list(prefix="dataset_version_1/manifests")

    assert stored.data == data
    assert stored.info == head
    assert stored.info.content_type == "application/jsonl"
    assert stored.info.metadata["source"] == "unit-test"
    assert stored.info.metadata["artifact-kind"] == "asset_manifest"
    assert listed == (head,)


def test_object_uri_outside_project_scope_is_rejected() -> None:
    adapter = _adapter(InMemoryS3Client())

    with pytest.raises(ObjectStorageError) as exc_info:
        adapter.get("s3://dataforge-local/dataforge/org_1/project_2/dataset_1/file.jsonl")

    assert exc_info.value.code is ErrorCode.ARTIFACT_OUT_OF_SCOPE


def test_object_name_path_traversal_is_rejected() -> None:
    adapter = _adapter(InMemoryS3Client())

    with pytest.raises(ObjectStorageError) as exc_info:
        adapter.put(
            object_name="../project_2/leak.json",
            data=b"{}",
            kind="report",
            media_type="application/json",
            schema_version="report.v1",
            parent_version_id="dataset_version_1",
            job_id="compute_run_001",
            config_hash="sha256:" + "b" * 64,
        )

    assert exc_info.value.code is ErrorCode.ARTIFACT_OUT_OF_SCOPE


def test_adapter_does_not_expose_bucket_admin_operations() -> None:
    adapter = _adapter(InMemoryS3Client())

    assert not hasattr(adapter, "create_bucket")
    assert not hasattr(adapter, "delete_bucket")
    assert not hasattr(adapter, "make_bucket")
    assert not hasattr(adapter, "remove_bucket")


def _adapter(client: InMemoryS3Client) -> MinioObjectStorageAdapter:
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
