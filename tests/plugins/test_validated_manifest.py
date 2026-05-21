"""Tests for TASK-022: validated_manifest asset and manifest contract validation.

Acceptance criteria covered:

* manifest JSONL validates against the contract schema (happy path);
* manifest with a missing required field (``hash``) fails with
  ``CONTRACT_VALIDATION_FAILED``;
* validated manifest is saved as an immutable artifact with a stable hash;
* compute audit event carries both ``input_hash`` and ``output_hash``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    AuditEventType,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.artifact_registry import RegisteredArtifact
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import ErrorCode
from app.ingestion import (
    BuildManifestRequest,
    ManifestContractValidationError,
    build_asset_manifest,
    build_validated_manifest,
    open_archive_path,
    validate_manifest_jsonl,
)
from app.validation.contracts import load_contract_pack
from tests.fixtures.demo_archive import build_demo_archive

_REQUEST = BuildManifestRequest(
    dataset_id="dataset_demo",
    version_id="version_demo",
    parent_version_id="version_demo_parent",
    created_by_job_id="compute_run_validated",
    config_hash="sha256:" + "a" * 64,
)


def _build_demo_manifest_artifacts(
    tmp_path: Path,
) -> tuple[MinioObjectStorageAdapter, ArtifactRegistry, RegisteredArtifact]:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
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
    with open_archive_path(built.archive_path) as reader:
        manifest_result = build_asset_manifest(
            reader, request=_REQUEST, registry=registry
        )
    return storage, registry, manifest_result.manifest_artifact


# ---------------------------------------------------------------------------
# Step 1: validate a correct manifest
# ---------------------------------------------------------------------------


def test_validate_correct_manifest_jsonl_passes_contract(tmp_path: Path) -> None:
    storage, _registry, raw_artifact = _build_demo_manifest_artifacts(tmp_path)

    raw_bytes = storage.get(raw_artifact.uri).data
    report = validate_manifest_jsonl(raw_bytes)

    assert report.row_count > 0
    assert "tabular" in report.rows_by_modality
    assert report.input_hash == raw_artifact.hash
    assert report.schema_name == "manifest_row"


def test_build_validated_manifest_persists_immutable_artifact(tmp_path: Path) -> None:
    storage, registry, raw_artifact = _build_demo_manifest_artifacts(tmp_path)
    fake_platform = FakePlatformMetadataClient()

    result = build_validated_manifest(
        raw_artifact,
        storage=storage,
        registry=registry,
        dataset_version_id=_REQUEST.version_id,
        parent_version_id=_REQUEST.parent_version_id,
        created_by_job_id=_REQUEST.created_by_job_id,
        config_hash=_REQUEST.config_hash,
        audit_sink=fake_platform,
        organization_id="org_test",
        project_id="project_test",
    )

    # Step 3: validated_manifest has its own ArtifactRef + hash, distinct kind.
    assert result.validated_manifest.artifact_kind == "validated_manifest"
    assert result.validated_manifest.uri.startswith("s3://dataforge-local/dataforge/")
    assert result.validated_manifest.hash == raw_artifact.hash  # bytes preserved
    assert result.validated_manifest.uri != raw_artifact.uri

    stored = storage.get(result.validated_manifest.uri)
    # Bytes are byte-for-byte the raw manifest payload.
    assert stored.data == storage.get(raw_artifact.uri).data
    # Stored object metadata reflects the validated artifact contract.
    assert stored.info.metadata["artifact-kind"] == "validated_manifest"
    assert stored.info.metadata["schema-version"] == "manifest_row.v1"


def test_validated_manifest_audit_event_has_input_and_output_hashes(
    tmp_path: Path,
) -> None:
    storage, registry, raw_artifact = _build_demo_manifest_artifacts(tmp_path)
    fake_platform = FakePlatformMetadataClient()

    result = build_validated_manifest(
        raw_artifact,
        storage=storage,
        registry=registry,
        dataset_version_id=_REQUEST.version_id,
        parent_version_id=_REQUEST.parent_version_id,
        created_by_job_id=_REQUEST.created_by_job_id,
        config_hash=_REQUEST.config_hash,
        audit_sink=fake_platform,
        organization_id="org_test",
        project_id="project_test",
    )

    snapshot = fake_platform.snapshot()
    assert len(snapshot.audit_events) == 1
    audit_event = snapshot.audit_events[0]
    assert audit_event.event_type is AuditEventType.MANIFEST_VALIDATED
    metadata = audit_event.metadata
    assert metadata["input_hash"] == raw_artifact.hash
    assert metadata["output_hash"] == result.validated_manifest.hash
    assert metadata["input_artifact_uri"] == raw_artifact.uri
    assert metadata["output_artifact_uri"] == result.validated_manifest.uri
    assert metadata["row_count"] == result.validation_report.row_count
    assert metadata["schema_version"] == "manifest_row.v1"


def test_repeated_validation_is_idempotent(tmp_path: Path) -> None:
    storage, registry, raw_artifact = _build_demo_manifest_artifacts(tmp_path)
    fake_platform = FakePlatformMetadataClient()

    first = build_validated_manifest(
        raw_artifact,
        storage=storage,
        registry=registry,
        dataset_version_id=_REQUEST.version_id,
        parent_version_id=_REQUEST.parent_version_id,
        created_by_job_id=_REQUEST.created_by_job_id,
        config_hash=_REQUEST.config_hash,
        audit_sink=fake_platform,
        organization_id="org_test",
        project_id="project_test",
    )
    second = build_validated_manifest(
        raw_artifact,
        storage=storage,
        registry=registry,
        dataset_version_id=_REQUEST.version_id,
        parent_version_id=_REQUEST.parent_version_id,
        created_by_job_id=_REQUEST.created_by_job_id,
        config_hash=_REQUEST.config_hash,
        audit_sink=fake_platform,
        organization_id="org_test",
        project_id="project_test",
    )

    assert first.validated_manifest.uri == second.validated_manifest.uri
    assert first.validated_manifest.hash == second.validated_manifest.hash


# ---------------------------------------------------------------------------
# Step 2: manifest with a missing required field fails CONTRACT_VALIDATION_FAILED
# ---------------------------------------------------------------------------


def _broken_manifest_without_hash() -> bytes:
    pack = load_contract_pack()
    example = next(
        ex for ex in pack.examples if ex.name == "manifest_row.tabular"
    )
    payload = dict(example.payload)
    del payload["hash"]
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def test_manifest_without_hash_raises_contract_validation_failed() -> None:
    payload = _broken_manifest_without_hash()

    with pytest.raises(ManifestContractValidationError) as excinfo:
        validate_manifest_jsonl(payload)

    err = excinfo.value
    assert err.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    assert err.reason_code == "missing_required_field"
    assert err.line_number == 1


def test_manifest_with_invalid_hash_pattern_raises_contract_validation_failed() -> None:
    pack = load_contract_pack()
    example = next(
        ex for ex in pack.examples if ex.name == "manifest_row.tabular"
    )
    payload = dict(example.payload)
    payload["hash"] = "not-a-sha256-digest"
    bad_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")

    with pytest.raises(ManifestContractValidationError) as excinfo:
        validate_manifest_jsonl(bad_bytes)

    err = excinfo.value
    assert err.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    # Hash pattern violation is a schema-level violation, not a missing field.
    assert err.reason_code == "schema_violation"


def test_manifest_with_invalid_json_raises_invalid_json_reason_code() -> None:
    bad_bytes = b"{not-valid-json\n"

    with pytest.raises(ManifestContractValidationError) as excinfo:
        validate_manifest_jsonl(bad_bytes)

    assert excinfo.value.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    assert excinfo.value.reason_code == "invalid_json"
    assert excinfo.value.line_number == 1


def test_empty_manifest_raises_empty_manifest_reason_code() -> None:
    with pytest.raises(ManifestContractValidationError) as excinfo:
        validate_manifest_jsonl(b"")

    assert excinfo.value.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    assert excinfo.value.reason_code == "empty_manifest"


def test_build_validated_manifest_does_not_persist_when_validation_fails(
    tmp_path: Path,
) -> None:
    storage, registry, raw_artifact = _build_demo_manifest_artifacts(tmp_path)
    s3_client = _client_for(storage)
    fake_platform = FakePlatformMetadataClient()

    # Replace the raw manifest object body with a payload missing `hash`.
    broken_bytes = _broken_manifest_without_hash()
    s3_client.put_object(
        Bucket="dataforge-local",
        Key=raw_artifact.uri.removeprefix("s3://dataforge-local/"),
        Body=broken_bytes,
        ContentType="application/jsonl",
        Metadata={
            "artifact-kind": "asset_manifest",
            "schema-version": "manifest_row.v1",
            "sha256": "sha256:" + "0" * 64,  # mismatch will be detected first
        },
    )
    # Object body changed; build_validated_manifest must detect the mismatch
    # before contract validation runs.
    with pytest.raises(ManifestContractValidationError) as excinfo:
        build_validated_manifest(
            raw_artifact,
            storage=storage,
            registry=registry,
            dataset_version_id=_REQUEST.version_id,
            parent_version_id=_REQUEST.parent_version_id,
            created_by_job_id=_REQUEST.created_by_job_id,
            config_hash=_REQUEST.config_hash,
            audit_sink=fake_platform,
            organization_id="org_test",
            project_id="project_test",
        )

    assert excinfo.value.code is ErrorCode.CONTRACT_VALIDATION_FAILED
    assert excinfo.value.reason_code == "raw_hash_mismatch"
    assert fake_platform.snapshot().audit_events == ()


# ---------------------------------------------------------------------------
# In-memory S3-compatible client used by the TASK-022 tests
# ---------------------------------------------------------------------------


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


def _client_for(storage: MinioObjectStorageAdapter) -> _InMemoryS3Client:
    """Reach into the adapter once for the in-memory S3 fake used in tests."""
    client = storage._client  # noqa: SLF001 - test-only access to inject body bytes
    assert isinstance(client, _InMemoryS3Client)
    return client
