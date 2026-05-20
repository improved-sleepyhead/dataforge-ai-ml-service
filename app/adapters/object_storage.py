"""Project-scoped object storage adapter for MinIO/S3-compatible clients."""

from __future__ import annotations

import hashlib
import posixpath
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from app.domain import ArtifactLineage, ArtifactRef, ErrorCode
from app.domain.common import NonEmptyStr, S3Uri, Sha256Digest


class ObjectStorageError(ValueError):
    """Raised when object storage access violates scope or object state."""

    def __init__(self, *, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class S3Body(Protocol):
    """Readable body returned by boto3-compatible get_object calls."""

    def read(self) -> bytes: ...


class S3CompatibleClient(Protocol):
    """Subset of boto3 S3 client methods used against MinIO-compatible storage."""

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]: ...

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]: ...

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]: ...

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
    ) -> Mapping[str, Any]: ...


class ObjectStorageScope(BaseModel):
    """Organization/project/dataset prefix scope supplied by signed platform context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr


class StoredObjectInfo(BaseModel):
    """Safe object metadata returned by object storage operations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: S3Uri
    hash: Sha256Digest
    size_bytes: int = Field(ge=0)
    content_type: NonEmptyStr
    updated_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)


class StoredObject(BaseModel):
    """Object bytes plus safe storage metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: bytes
    info: StoredObjectInfo


class MinioObjectStorageAdapter:
    """Adapter for MinIO/S3-compatible object operations within one dataset scope."""

    def __init__(
        self,
        *,
        client: S3CompatibleClient,
        bucket_name: str,
        prefix_root: str,
        scope: ObjectStorageScope,
    ) -> None:
        self._client = client
        self._bucket_name = bucket_name
        self._prefix_root = _safe_path_part(prefix_root)
        self._scope = scope

    @property
    def allowed_prefix(self) -> str:
        """Allowed key prefix for the scoped organization/project/dataset."""
        return _join_key(
            self._prefix_root,
            self._scope.organization_id,
            self._scope.project_id,
            self._scope.dataset_id,
        )

    def put(
        self,
        *,
        object_name: str,
        data: bytes,
        kind: str,
        media_type: str,
        schema_version: str,
        parent_version_id: str,
        job_id: str,
        config_hash: str,
        metadata: Mapping[str, str] | None = None,
    ) -> ArtifactRef:
        """Write an immutable compute artifact and return its contract ArtifactRef."""
        key = self._key_for_object_name(object_name)
        digest = _sha256(data)
        safe_metadata = _string_metadata(
            {
                **({} if metadata is None else dict(metadata)),
                "artifact-kind": kind,
                "schema-version": schema_version,
                "sha256": digest,
            }
        )
        self._client.put_object(
            Bucket=self._bucket_name,
            Key=key,
            Body=data,
            ContentType=media_type,
            Metadata=safe_metadata,
        )
        return ArtifactRef(
            artifact_id=f"{kind}:{digest.removeprefix('sha256:')[:16]}",
            kind=kind,
            uri=self._uri_for_key(key),
            hash=digest,
            media_type=media_type,
            size_bytes=len(data),
            schema_version=schema_version,
            lineage=ArtifactLineage(
                parent_version_id=parent_version_id,
                job_id=job_id,
                config_hash=config_hash,
                created_at=datetime.now(UTC),
            ),
        )

    def get(self, uri: str) -> StoredObject:
        """Read an object by scoped s3 URI."""
        key = self._key_for_uri(uri)
        response = self._client.get_object(Bucket=self._bucket_name, Key=key)
        body = response.get("Body")
        if not hasattr(body, "read"):
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object body is unavailable",
            )
        data = cast(S3Body, body).read()
        return StoredObject(data=data, info=self._info_from_response(key, response, data=data))

    def head(self, uri: str) -> StoredObjectInfo:
        """Return safe object metadata by scoped s3 URI."""
        key = self._key_for_uri(uri)
        response = self._client.head_object(Bucket=self._bucket_name, Key=key)
        return self._info_from_response(key, response)

    def list(self, *, prefix: str = "") -> tuple[StoredObjectInfo, ...]:
        """List objects below the scoped dataset prefix."""
        scoped_prefix = self._key_for_object_name(prefix) if prefix else f"{self.allowed_prefix}/"
        response = self._client.list_objects_v2(Bucket=self._bucket_name, Prefix=scoped_prefix)
        infos: list[StoredObjectInfo] = []
        for item in response.get("Contents", ()):
            key = item.get("Key")
            if not isinstance(key, str):
                continue
            if not self._is_key_in_scope(key):
                continue
            try:
                infos.append(self.head(self._uri_for_key(key)))
            except ObjectStorageError:
                continue
        return tuple(infos)

    def uri_for_object_name(self, object_name: str) -> str:
        """Return the scoped s3 URI that would be used for a relative object name."""
        return self._uri_for_key(self._key_for_object_name(object_name))

    def _key_for_object_name(self, object_name: str) -> str:
        object_path = _safe_relative_path(object_name)
        key = _join_key(self.allowed_prefix, object_path)
        self._require_key_in_scope(key)
        return key

    def _key_for_uri(self, uri: str) -> str:
        prefix = f"s3://{self._bucket_name}/"
        if not uri.startswith(prefix):
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_OUT_OF_SCOPE,
                message="Object URI is outside configured bucket",
            )
        key = uri.removeprefix(prefix)
        self._require_key_in_scope(key)
        return key

    def _require_key_in_scope(self, key: str) -> None:
        if not self._is_key_in_scope(key):
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_OUT_OF_SCOPE,
                message="Object URI is outside organization/project/dataset scope",
            )

    def _is_key_in_scope(self, key: str) -> bool:
        prefix = f"{self.allowed_prefix}/"
        return key.startswith(prefix) and ".." not in key.split("/")

    def _uri_for_key(self, key: str) -> str:
        return f"s3://{self._bucket_name}/{key}"

    def _info_from_response(
        self,
        key: str,
        response: Mapping[str, Any],
        *,
        data: bytes | None = None,
    ) -> StoredObjectInfo:
        metadata = _string_metadata(response.get("Metadata", {}))
        digest = metadata.get("sha256")
        if data is not None:
            digest = _sha256(data)
        if not digest:
            digest = "sha256:" + "0" * 64
        updated_at = response.get("LastModified")
        if not isinstance(updated_at, datetime):
            updated_at = datetime.now(UTC)
        return StoredObjectInfo(
            uri=self._uri_for_key(key),
            hash=digest,
            size_bytes=_content_length(response, data),
            content_type=str(response.get("ContentType") or "application/octet-stream"),
            updated_at=updated_at,
            metadata=metadata,
        )


def _safe_relative_path(value: str) -> str:
    stripped = value.strip().lstrip("/")
    normalized = posixpath.normpath(stripped)
    if not stripped or normalized in {"", "."}:
        raise ObjectStorageError(
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            message="Object name must be a non-empty relative path",
        )
    if normalized.startswith("../") or normalized == ".." or "/../" in f"/{normalized}/":
        raise ObjectStorageError(
            code=ErrorCode.ARTIFACT_OUT_OF_SCOPE,
            message="Object name cannot traverse outside scoped prefix",
        )
    return normalized


def _safe_path_part(value: str) -> str:
    return _safe_relative_path(value).strip("/")


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _string_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key).lower(): str(item) for key, item in value.items()}


def _content_length(response: Mapping[str, Any], data: bytes | None) -> int:
    if data is not None:
        return len(data)
    value = response.get("ContentLength", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
