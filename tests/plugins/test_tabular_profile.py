"""Tests for TASK-023: tabular schema inference and base profile report.

Acceptance criteria covered:

* profiler computes ``row_count``, ``column_count``, schema, nullability;
* detects ``target_column``, ``group_key_columns``, id-like and PII-like
  columns;
* report saved as ``tabular_profile_report`` immutable artifact;
* report ``ArtifactRef`` is contract-compatible with
  ``DataForgeReport.detail_artifacts``.
"""

from __future__ import annotations

import io
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
from app.domain import (
    ColumnRole,
    ColumnType,
    ErrorCode,
    TabularProfileReport,
)
from app.ingestion import (
    BuildManifestRequest,
    build_asset_manifest,
    build_validated_manifest,
    open_archive_path,
)
from app.plugins.tabular import (
    PROFILE_REPORT_KIND,
    PROFILE_REPORT_SCHEMA_VERSION,
    ProfileBuildRequest,
    build_tabular_profile_report,
    infer_tabular_profile,
)
from app.validation.contracts import (
    load_contract_pack,
    validate_contract_payload,
)
from tests.fixtures.demo_archive import build_demo_archive

_MANIFEST_REQUEST = BuildManifestRequest(
    dataset_id="dataset_demo",
    version_id="version_demo",
    parent_version_id="version_demo_parent",
    created_by_job_id="compute_run_profile",
    config_hash="sha256:" + "a" * 64,
)
_PROFILE_REQUEST = ProfileBuildRequest(
    dataset_id="dataset_demo",
    version_id="version_demo",
    parent_version_id="version_demo_parent",
    created_by_job_id="compute_run_profile",
    config_hash="sha256:" + "a" * 64,
    source_artifact_id="raw_archive:demo",
    source_system="transactions",
)


# ---------------------------------------------------------------------------
# Step 1: run profiler on demo transactions.csv
# ---------------------------------------------------------------------------


def test_profiler_runs_on_demo_archive_and_produces_artifact(
    tmp_path: Path,
) -> None:
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )

    report = result.profile_report
    assert isinstance(report, TabularProfileReport)
    assert report.row_count == 200
    assert report.column_count == 8
    assert report.profile_schema_version == "tabular_profile_report.v1"
    assert report.source_system == "transactions"
    assert result.artifact.artifact_kind == PROFILE_REPORT_KIND
    assert result.artifact.schema_version == PROFILE_REPORT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Step 2: target = is_fraud, group key = customer_id_hash, schema/nullability
# ---------------------------------------------------------------------------


def test_profiler_detects_target_group_key_and_pii_like(tmp_path: Path) -> None:
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        result = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
        )

    report = result.profile_report
    assert report.target_column == "is_fraud"
    assert "customer_id_hash" in report.group_key_columns
    assert "object_id" in report.id_columns
    # Demo transactions.csv has no PII-like column names.
    assert report.pii_like_columns == ()

    columns_by_name = {col.name: col for col in report.columns}
    target = columns_by_name["is_fraud"]
    assert target.role is ColumnRole.TARGET
    assert target.type is ColumnType.BOOLEAN

    customer_hash = columns_by_name["customer_id_hash"]
    assert customer_hash.role is ColumnRole.GROUP_KEY

    monthly_income = columns_by_name["monthly_income"]
    assert monthly_income.nullable is True
    assert monthly_income.null_count > 0
    assert 0.0 < monthly_income.null_ratio < 1.0
    assert monthly_income.type is ColumnType.NUMERIC_FLOAT

    leakage = columns_by_name["manual_review_flag"]
    assert leakage.role is ColumnRole.LEAKAGE_CANDIDATE
    assert leakage.type is ColumnType.BOOLEAN


def test_pii_like_column_names_are_detected_and_sample_value_redacted() -> None:
    rows = [
        {"object_id": "u1", "email": "alex@example.test", "phone": "+10000001234"},
        {"object_id": "u2", "email": "lena@example.test", "phone": "+10000005678"},
    ]
    report = infer_tabular_profile(
        iter(rows),
        columns=("object_id", "email", "phone"),
        request=_PROFILE_REQUEST,
        source_manifest_artifact=_synthetic_manifest_artifact_ref(),
        profile_id="tabular_profile_pii_test",
        generated_at=datetime(2026, 5, 20, tzinfo=UTC),
    )

    pii_columns = {col.name for col in report.columns if col.role is ColumnRole.PII_LIKE}
    assert pii_columns == {"email", "phone"}
    for col in report.columns:
        if col.role is ColumnRole.PII_LIKE:
            assert col.sample_value is None
        else:
            assert col.sample_value is not None


# ---------------------------------------------------------------------------
# Step 3: artifact saved with hash and contract-compatible
# ---------------------------------------------------------------------------


def test_profile_artifact_is_contract_compatible_and_idempotent(
    tmp_path: Path,
) -> None:
    storage, registry, _validated, archive_path = _setup_demo(tmp_path)
    validated = _validated_manifest(storage, registry, archive_path)

    with open_archive_path(archive_path) as reader:
        first = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
            profile_id="tabular_profile_demo",
            generated_at=datetime(2026, 5, 20, tzinfo=UTC),
        )
    with open_archive_path(archive_path) as reader:
        second = build_tabular_profile_report(
            reader,
            request=_PROFILE_REQUEST,
            storage=storage,
            registry=registry,
            source_manifest_artifact=validated.artifact_ref,
            profile_id="tabular_profile_demo",
            generated_at=datetime(2026, 5, 20, tzinfo=UTC),
        )

    assert first.artifact.uri == second.artifact.uri
    assert first.artifact.hash == second.artifact.hash

    stored = storage.get(first.artifact.uri)
    assert stored.info.metadata["artifact-kind"] == "tabular_profile_report"
    assert stored.info.metadata["schema-version"] == "tabular_profile_report.v1"

    pack = load_contract_pack()
    payload = first.profile_report.model_dump(mode="json")
    validate_contract_payload(pack, "tabular_profile_report", payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_demo(
    tmp_path: Path,
) -> tuple[
    MinioObjectStorageAdapter,
    ArtifactRegistry,
    Path,
    Path,
]:
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
    return storage, registry, built.archive_path, built.archive_path


def _validated_manifest(
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    archive_path: Path,
) -> Any:
    with open_archive_path(archive_path) as reader:
        manifest_result = build_asset_manifest(
            reader, request=_MANIFEST_REQUEST, registry=registry
        )
    raw_artifact = manifest_result.manifest_artifact
    return build_validated_manifest(
        raw_artifact,
        storage=storage,
        registry=registry,
        dataset_version_id=_MANIFEST_REQUEST.version_id,
        parent_version_id=_MANIFEST_REQUEST.parent_version_id,
        created_by_job_id=_MANIFEST_REQUEST.created_by_job_id,
        config_hash=_MANIFEST_REQUEST.config_hash,
        organization_id="org_test",
        project_id="project_test",
    ).validated_manifest


def _synthetic_manifest_artifact_ref() -> Any:
    """Return a synthetic ArtifactRef for PII-name unit tests (no archive)."""
    from app.domain import ArtifactLineage, ArtifactRef

    return ArtifactRef(
        artifact_id="validated_manifest:synthetic",
        kind="validated_manifest",
        uri="s3://dataforge-local/dataforge/org_test/project_test/dataset_demo/manifest.jsonl",
        hash="sha256:" + "f" * 64,
        media_type="application/jsonl",
        size_bytes=1024,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="version_demo",
            job_id="compute_run_profile",
            config_hash="sha256:" + "a" * 64,
            created_at=datetime(2026, 5, 20, tzinfo=UTC),
        ),
    )


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
