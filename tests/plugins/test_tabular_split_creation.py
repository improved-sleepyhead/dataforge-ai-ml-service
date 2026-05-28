"""Tests for TASK-044 supervised tabular split creation."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import ActionPlanStep, ArtifactRef, DataSplit, ErrorCode, RetryPolicy, SplitManifest
from app.ingestion import open_archive_path
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    ExecuteTabularSplitRequest,
    execute_tabular_split_action,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "e" * 64
_CREATED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)


def test_group_stratified_split_persists_manifest_and_keeps_customers_together(
    tmp_path: Path,
) -> None:
    """Steps 1-3: create demo split, check group isolation and class distribution."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _source_transactions_artifact(registry=registry, data=transactions)
    source_rows = _read_csv(transactions)

    result = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id="action_plan_split_001",
            step=_split_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            target_column="is_fraud",
            seed=42,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    manifest = result.manifest
    assert isinstance(manifest, SplitManifest)
    assert manifest.strategy == "group_stratified"
    assert manifest.seed == 42
    assert manifest.group_key == "customer_id_hash"
    assert manifest.policy_version == "split_policy_v0"
    assert manifest.lineage.source_artifact == source_artifact
    assert len(manifest.assignments) == len(source_rows)

    assignments_by_object = {
        assignment.object_id: assignment for assignment in manifest.assignments
    }
    splits_by_customer: dict[str, set[DataSplit]] = defaultdict(set)
    for row in source_rows:
        assignment = assignments_by_object[row["object_id"]]
        splits_by_customer[row["customer_id_hash"]].add(assignment.split)
        assert assignment.label == row["is_fraud"]
        assert assignment.group_value == row["customer_id_hash"]
    assert all(len(splits) == 1 for splits in splits_by_customer.values())

    distribution = {item.split: item for item in manifest.class_distribution}
    assert set(distribution) == {DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST}
    assert sum(item.total_count for item in distribution.values()) == len(source_rows)
    rare_counts = {split: item.class_counts.get("1", 0) for split, item in distribution.items()}
    assert sum(rare_counts.values()) == sum(1 for row in source_rows if row["is_fraud"] == "1")
    assert all(count >= 1 for count in rare_counts.values())

    assert result.split_artifact.artifact_kind == SPLIT_MANIFEST_KIND
    assert result.split_artifact.schema_version == SPLIT_MANIFEST_SCHEMA_VERSION
    assert result.split_artifact.artifact_ref.lineage.parent_version_id == (
        "dataset_version_2_candidate"
    )
    stored = storage.get(result.split_artifact.uri)
    assert stored.info.metadata["seed"] == "42"
    assert stored.info.metadata["strategy"] == "group_stratified"
    assert stored.info.metadata["group-key"] == "customer_id_hash"
    assert stored.info.metadata["policy-version"] == "split_policy_v0"
    assert stored.info.metadata["assignment-count"] == str(len(source_rows))
    validate_contract_payload(
        load_contract_pack(),
        "split_manifest",
        manifest.model_dump(mode="json"),
    )


def _split_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="create_split_tabular",
        type="CREATE_SPLIT",
        depends_on=(),
        idempotency_key="sha256:" + "1" * 64,
        method_id="group_stratified",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="split_policy_v0",
        validation_gates=("schema_validation", "split_policy_check"),
        preconditions=("source_version_is_immutable", "split_before_train_only_augmentation"),
        input_artifacts=("s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",),
        output_artifact_kind=SPLIT_MANIFEST_KIND,
        config={
            "strategy": "group_stratified",
            "group_key": "customer_id_hash",
            "target_column": "is_fraud",
        },
        random_seed=42,
        retry_policy=RetryPolicy(max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _source_transactions_artifact(*, registry: ArtifactRegistry, data: bytes) -> ArtifactRef:
    return registry.save_artifact(
        artifact_kind="raw_transactions",
        data=data,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


def _read_transactions(tmp_path: Path) -> bytes:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        return reader.find_required_transactions().read_bytes()


def _read_csv(data: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    return [dict(row) for row in reader]


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
        ),
    )
    return storage, ArtifactRegistry(storage=storage)


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
            "LastModified": _CREATED_AT,
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "Body": io.BytesIO(stored["Body"]),
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(stored["Body"])}
                for (bucket, key), stored in sorted(self._objects.items())
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
