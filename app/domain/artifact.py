"""Artifact reference contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.common import NonEmptyStr, S3Uri, Sha256Digest


class ArtifactLineage(BaseModel):
    """Lineage metadata for immutable artifact references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_version_id: NonEmptyStr
    job_id: NonEmptyStr
    config_hash: Sha256Digest
    created_at: datetime


class ArtifactRef(BaseModel):
    """Immutable object-storage pointer used instead of inline raw payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: NonEmptyStr
    kind: NonEmptyStr
    uri: S3Uri
    hash: Sha256Digest
    media_type: NonEmptyStr
    size_bytes: int = Field(ge=0)
    schema_version: NonEmptyStr
    lineage: ArtifactLineage
