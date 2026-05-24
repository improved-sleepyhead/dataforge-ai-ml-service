"""Tests for TASK-041 safe tabular imputation executor."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlan,
    ArtifactRef,
    ErrorCode,
    MethodRecommendation,
    TabularImputationReport,
    TabularProfileReport,
)
from app.ingestion import open_archive_path
from app.kernel import (
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    build_action_plan_preview,
    build_method_recommendations,
)
from app.plugins.tabular import (
    IMPUTATION_REPORT_KIND,
    IMPUTATION_REPORT_SCHEMA_VERSION,
    ExecuteTabularImputationRequest,
    ImputationExecutionError,
    execute_tabular_imputation_action,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "c" * 64


def test_group_median_imputes_monthly_income_and_persists_report(tmp_path: Path) -> None:
    """Step 1+3: execute impute monthly_income and verify before/after metadata."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    source_rows = _read_csv(storage.get(source_artifact.uri).data)
    source_target = [row["is_fraud"] for row in source_rows]
    before_missing = sum(1 for row in source_rows if row["monthly_income"] == "")
    assert before_missing > 0

    plan = _imputation_action_plan()
    result = execute_tabular_imputation_action(
        ExecuteTabularImputationRequest(
            action_plan_id=plan.action_plan_id,
            step=plan.steps[0],
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            target_column="is_fraud",
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            report_id="tabular_imputation_report_test",
            generated_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
        ),
        storage=storage,
        registry=registry,
    )

    candidate_rows = _read_csv(storage.get(result.candidate_artifact.uri).data)
    assert [row["is_fraud"] for row in candidate_rows] == source_target
    assert sum(1 for row in candidate_rows if row["monthly_income"] == "") == 0
    assert "monthly_income_was_missing" in candidate_rows[0]
    assert sum(int(row["monthly_income_was_missing"]) for row in candidate_rows) == before_missing

    report = result.report
    assert isinstance(report, TabularImputationReport)
    assert report.target_unchanged is True
    assert report.before_row_count == len(source_rows)
    assert report.after_row_count == len(source_rows)
    assert report.before_missing_total == before_missing
    assert report.after_missing_total == 0
    assert report.candidate_artifact == result.candidate_artifact.artifact_ref
    assert report.lineage.source_artifact == source_artifact
    assert report.lineage.candidate_artifact == result.candidate_artifact.artifact_ref

    column_report = report.columns[0]
    assert column_report.column == "monthly_income"
    assert column_report.method == "group_median"
    assert column_report.group_key == "customer_segment"
    assert column_report.indicator_column == "monthly_income_was_missing"
    assert column_report.before_missing_count == before_missing
    assert column_report.after_missing_count == 0
    assert column_report.imputed_count == before_missing
    assert sum(column_report.group_imputed_counts.values()) == before_missing

    assert result.report_artifact.artifact_kind == IMPUTATION_REPORT_KIND
    assert result.report_artifact.schema_version == IMPUTATION_REPORT_SCHEMA_VERSION
    stored_report = storage.get(result.report_artifact.uri)
    assert stored_report.info.metadata["before-missing-count"] == str(before_missing)
    assert stored_report.info.metadata["after-missing-count"] == "0"
    validate_contract_payload(
        load_contract_pack(),
        "tabular_imputation_report",
        report.model_dump(mode="json"),
    )


def test_target_column_auto_imputation_is_hard_blocked(tmp_path: Path) -> None:
    """Step 2: target column is not changed and target imputation is blocked."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    step = _imputation_action_plan().steps[0].model_copy(
        update={"method_id": "median", "config": {"column": "is_fraud", "method": "median"}}
    )

    with pytest.raises(ImputationExecutionError) as error:
        execute_tabular_imputation_action(
            ExecuteTabularImputationRequest(
                action_plan_id="action_plan_target_block",
                step=step,
                source_dataset_version_id="dataset_version_1",
                candidate_dataset_version_id="dataset_version_2_candidate",
                target_column="is_fraud",
                source_artifact=source_artifact,
                created_by_job_id="compute_run_apply_001",
                config_hash=_CONFIG_HASH,
            ),
            storage=storage,
            registry=registry,
        )

    assert error.value.code is ErrorCode.POLICY_BLOCKED
    assert error.value.reason_code == "target_column_auto_imputation_forbidden"


def test_missingness_indicator_adds_metadata_without_filling_values() -> None:
    storage, registry = _storage_and_registry()
    source = registry.save_artifact(
        artifact_kind="raw_transactions",
        data=b"object_id,is_fraud,value\nr1,0,\nr2,1,12\n",
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref
    step = _imputation_action_plan().steps[0].model_copy(
        update={
            "method_id": "missingness_indicator",
            "config": {"column": "value", "method": "missingness_indicator"},
        }
    )

    result = execute_tabular_imputation_action(
        ExecuteTabularImputationRequest(
            action_plan_id="action_plan_indicator",
            step=step,
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            target_column="is_fraud",
            source_artifact=source,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            generated_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
        ),
        storage=storage,
        registry=registry,
    )

    rows = _read_csv(storage.get(result.candidate_artifact.uri).data)
    assert [row["value"] for row in rows] == ["", "12"]
    assert [row["value_was_missing"] for row in rows] == ["1", "0"]
    assert result.report.columns[0].imputed_count == 0
    assert result.report.columns[0].before_missing_count == 1
    assert result.report.columns[0].after_missing_count == 1


def _imputation_action_plan() -> ActionPlan:
    recommendations = _method_recommendations()
    imputation = next(
        recommendation
        for recommendation in recommendations
        if recommendation.action_type == "IMPUTE_MISSING_VALUES"
    )
    return build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_001",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(imputation.recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(imputation,),
            created_by_user_id="platform_user_123",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
            ),
            target_version_name="dataset_version_2_candidate",
            created_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
        )
    )


def _method_recommendations() -> tuple[MethodRecommendation, ...]:
    return build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example for example in pack.examples if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def _source_transactions_artifact(tmp_path: Path, registry: ArtifactRegistry) -> ArtifactRef:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        transactions = reader.find_required_transactions().read_bytes()
    return registry.save_artifact(
        artifact_kind="raw_transactions",
        data=transactions,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


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
            "LastModified": datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
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
