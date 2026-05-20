"""Ports for object storage, platform metadata, vector stores, and external systems."""

from app.adapters.artifact_registry import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import (
    MinioObjectStorageAdapter,
    ObjectStorageError,
    ObjectStorageScope,
    StoredObject,
    StoredObjectInfo,
)
from app.adapters.platform_metadata import (
    AuditEventType,
    FakePlatformMetadataClient,
    FakePlatformState,
    PlatformAuditEvent,
    PlatformJobEvent,
    create_fake_platform_app,
)

__all__ = [
    "AuditEventType",
    "ArtifactRegistry",
    "FakePlatformMetadataClient",
    "FakePlatformState",
    "MinioObjectStorageAdapter",
    "ObjectStorageError",
    "ObjectStorageScope",
    "PlatformAuditEvent",
    "PlatformJobEvent",
    "RegisteredArtifact",
    "StoredObject",
    "StoredObjectInfo",
    "create_fake_platform_app",
]
