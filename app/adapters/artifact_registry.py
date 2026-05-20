"""Immutable compute artifact registry backed by scoped object storage."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from app.adapters.object_storage import MinioObjectStorageAdapter, ObjectStorageError
from app.domain import ArtifactLineage, ArtifactRef, ErrorCode
from app.domain.common import NonEmptyStr, S3Uri, Sha256Digest


class RegisteredArtifact(BaseModel):
    """Registry record for one immutable compute artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_ref: ArtifactRef
    artifact_kind: NonEmptyStr
    uri: S3Uri
    hash: Sha256Digest
    format: NonEmptyStr
    schema_version: NonEmptyStr
    created_by_job_id: NonEmptyStr


class ArtifactRegistry:
    """Stores immutable compute artifacts in object storage, never in a database."""

    def __init__(self, *, storage: MinioObjectStorageAdapter) -> None:
        self._storage = storage

    def save_artifact(
        self,
        *,
        artifact_kind: str,
        data: bytes,
        artifact_format: str,
        media_type: str,
        schema_version: str,
        dataset_version_id: str,
        created_by_job_id: str,
        config_hash: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RegisteredArtifact:
        """Save an artifact under a deterministic content-addressed path."""
        kind = _safe_component(artifact_kind, name="artifact_kind")
        output_format = _safe_component(artifact_format, name="artifact_format")
        schema = _safe_component(schema_version, name="schema_version")
        dataset_version = _safe_component(dataset_version_id, name="dataset_version_id")
        job_id = _safe_component(created_by_job_id, name="created_by_job_id")
        digest = _sha256(data)
        object_name = _object_name(
            dataset_version_id=dataset_version,
            created_by_job_id=job_id,
            artifact_kind=kind,
            schema_version=schema,
            digest=digest,
            artifact_format=output_format,
        )
        uri = self._storage.uri_for_object_name(object_name)
        existing = self._existing_artifact(
            uri=uri,
            expected_hash=digest,
            artifact_kind=kind,
            artifact_format=output_format,
            schema_version=schema,
            dataset_version_id=dataset_version,
            created_by_job_id=job_id,
            config_hash=config_hash,
        )
        if existing is not None:
            return existing

        artifact_ref = self._storage.put(
            object_name=object_name,
            data=data,
            kind=kind,
            media_type=media_type,
            schema_version=schema,
            parent_version_id=dataset_version,
            job_id=job_id,
            config_hash=config_hash,
            metadata={
                **({} if metadata is None else dict(metadata)),
                "artifact-format": output_format,
                "created-by-job-id": job_id,
                "dataset-version-id": dataset_version,
            },
        )
        return _record_from_ref(
            artifact_ref=artifact_ref,
            artifact_kind=kind,
            artifact_format=output_format,
            created_by_job_id=job_id,
        )

    def _existing_artifact(
        self,
        *,
        uri: str,
        expected_hash: str,
        artifact_kind: str,
        artifact_format: str,
        schema_version: str,
        dataset_version_id: str,
        created_by_job_id: str,
        config_hash: str,
    ) -> RegisteredArtifact | None:
        try:
            info = self._storage.head(uri)
        except ObjectStorageError as exc:
            if exc.code is ErrorCode.ARTIFACT_NOT_FOUND:
                return None
            raise
        if (
            info.hash != expected_hash
            or info.metadata.get("sha256") != expected_hash
            or info.metadata.get("artifact-kind") != artifact_kind
            or info.metadata.get("artifact-format") != artifact_format
            or info.metadata.get("schema-version") != schema_version
            or info.metadata.get("created-by-job-id") != created_by_job_id
            or info.metadata.get("dataset-version-id") != dataset_version_id
        ):
            raise ObjectStorageError(
                code=ErrorCode.CONTRACT_VALIDATION_FAILED,
                message="Existing artifact metadata does not match registry request",
            )
        artifact_ref = ArtifactRef(
            artifact_id=f"{artifact_kind}:{info.hash.removeprefix('sha256:')[:16]}",
            kind=artifact_kind,
            uri=info.uri,
            hash=info.hash,
            media_type=info.content_type,
            size_bytes=info.size_bytes,
            schema_version=schema_version,
            lineage=ArtifactLineage(
                parent_version_id=dataset_version_id,
                job_id=created_by_job_id,
                config_hash=config_hash,
                created_at=info.updated_at,
            ),
        )
        return _record_from_ref(
            artifact_ref=artifact_ref,
            artifact_kind=artifact_kind,
            artifact_format=artifact_format,
            created_by_job_id=created_by_job_id,
        )


def _record_from_ref(
    *,
    artifact_ref: ArtifactRef,
    artifact_kind: str,
    artifact_format: str,
    created_by_job_id: str,
) -> RegisteredArtifact:
    return RegisteredArtifact(
        artifact_ref=artifact_ref,
        artifact_kind=artifact_kind,
        uri=artifact_ref.uri,
        hash=artifact_ref.hash,
        format=artifact_format,
        schema_version=artifact_ref.schema_version,
        created_by_job_id=created_by_job_id,
    )


def _object_name(
    *,
    dataset_version_id: str,
    created_by_job_id: str,
    artifact_kind: str,
    schema_version: str,
    digest: str,
    artifact_format: str,
) -> str:
    digest_hex = digest.removeprefix("sha256:")
    return (
        f"versions/{dataset_version_id}/jobs/{created_by_job_id}/artifacts/"
        f"{artifact_kind}/{schema_version}/{digest_hex}.{artifact_format}"
    )


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _safe_component(value: str, *, name: str) -> str:
    stripped = value.strip()
    if not stripped or stripped in {".", ".."} or "/" in stripped or "\\" in stripped:
        raise ObjectStorageError(
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            message=f"{name} must be a non-empty path component",
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stripped):
        raise ObjectStorageError(
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            message=f"{name} contains unsupported characters",
        )
    return stripped
