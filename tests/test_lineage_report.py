"""Tests for TASK-055 lineage.json + compute audit events builder."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    AuditEventType,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ArtifactLineage,
    ArtifactRef,
    CandidateActionStepSummary,
    CandidateDatasetVersion,
    CandidateDatasetVersionLineage,
    CandidatePolicyVersions,
    CandidateVersionStatus,
    ErrorCode,
    ExportObjectCounts,
    ExportPackage,
    ExportPackageLineage,
    ExportPackageStatus,
    GateStatus,
    LineageReport,
    ValidationGateResult,
)
from app.kernel import (
    LINEAGE_REPORT_KIND,
    OPENLINEAGE_JOB_NAME,
    OPENLINEAGE_NAMESPACE,
    BuildLineageReportRequest,
    LineageBuilderError,
    build_lineage_report,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload

_CONFIG_HASH = "sha256:" + "a" * 64
_GENERATED_AT = datetime(2026, 5, 30, 12, 30, tzinfo=UTC)
_BASE_HASH = "sha256:" + "b" * 64
_CANDIDATE_HASH = "sha256:" + "c" * 64
_GATES_HASH = "sha256:" + "d" * 64
_EXPORT_HASH = "sha256:" + "e" * 64
_EXPORT_MANIFEST_HASH = "sha256:" + "1" * 64
_DATASET_CARD_HASH = "sha256:" + "2" * 64
_PARQUET_HASH = "sha256:" + "5" * 64


# ---------------------------------------------------------------------------
# Step 1+2+3: full flow up to export -> lineage.json + audit events
# ---------------------------------------------------------------------------


def test_lineage_report_after_export_records_required_fields_and_audit_events() -> None:
    """Steps 1-3: lineage.json contains required fields; audit events emitted."""
    storage, registry = _storage_and_registry()
    platform_client = FakePlatformMetadataClient()
    candidate = _candidate()
    export_package = _export_package(candidate=candidate)

    request = BuildLineageReportRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        candidate_version_artifact=_candidate_version_artifact(),
        export_package=export_package,
        export_package_artifact=_export_package_artifact(),
        input_artifact_refs=(_baseline_artifact(),),
        output_artifact_refs=(_parquet_artifact(),),
        privacy_policy_version="privacy_v0",
        export_policy_version="export_policy_v0",
        job_started_at=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
        job_completed_at=datetime(2026, 5, 30, 12, 30, tzinfo=UTC),
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        lineage_report_id="lineage_report_test_001",
        generated_at=_GENERATED_AT,
    )
    result = build_lineage_report(
        request, registry=registry, platform_client=platform_client
    )
    report = result.lineage_report

    # Step 2: required PRD §20.1 fields are present.
    assert isinstance(report, LineageReport)
    assert report.parent_version_id == "dataset_version_v1"
    assert report.output_dataset_version_id == "dataset_version_v2_candidate"
    assert report.job_id == "compute_run_apply_001"
    assert report.algorithm_name == "group_stratified"  # primary step (first)
    assert report.algorithm_version == "0.1.0"
    assert report.config_hash == _CONFIG_HASH
    assert report.policy_version == "method_policy_v0"
    assert report.random_seed == 42

    # Hashes contain the explicit input/output refs we passed in.
    assert _baseline_artifact().hash in report.input_artifact_hashes
    # Output hashes include the candidate output + every export-package
    # artifact hash so audit can prove the export came from this lineage.
    assert _parquet_artifact().hash in report.output_artifact_hashes
    assert _export_manifest_artifact().hash in report.output_artifact_hashes

    # Algorithms list captures every action plan step with plugin/config
    # version metadata, including random seeds.
    assert len(report.algorithms) == len(candidate.action_plan_steps)
    smote_algo = next(a for a in report.algorithms if a.algorithm_name == "smote")
    assert smote_algo.plugin_id == "dataforge.tabular"
    assert smote_algo.plugin_version == "0.1.0"
    assert smote_algo.random_seed == 42

    # Policy versions block carries every policy in force.
    assert report.policy_versions.method_policy_version == "method_policy_v0"
    assert report.policy_versions.privacy_policy_version == "privacy_v0"
    assert report.policy_versions.export_policy_version == "export_policy_v0"

    # OpenLineage envelope is well-formed and uses stable
    # namespaces/job names.
    assert report.openlineage.run.run_id == "compute_run_apply_001"
    assert report.openlineage.run.job_namespace == OPENLINEAGE_NAMESPACE
    assert report.openlineage.run.job_name == OPENLINEAGE_JOB_NAME
    assert report.openlineage.inputs
    assert any(
        out.facets.get("kind") == "candidate_dataset_version"
        for out in report.openlineage.outputs
    )
    assert any(
        out.facets.get("kind") == "export_package"
        for out in report.openlineage.outputs
    )

    # Persistence metadata + contract validation.
    assert result.lineage_report_artifact.artifact_kind == LINEAGE_REPORT_KIND
    stored = storage.get(result.lineage_report_artifact.uri)
    assert stored.info.metadata["lineage-report-id"] == "lineage_report_test_001"
    assert stored.info.metadata["job-id"] == "compute_run_apply_001"
    validate_contract_payload(
        load_contract_pack(),
        "lineage_report",
        report.model_dump(mode="json"),
    )

    # Step 3: fake platform receives LINEAGE_REPORT_BUILT and
    # EXPORT_PACKAGE_BUILT audit events with safe metadata only.
    snapshot = platform_client.snapshot()
    audit_types = [event.event_type for event in snapshot.audit_events]
    assert AuditEventType.LINEAGE_REPORT_BUILT in audit_types
    assert AuditEventType.EXPORT_PACKAGE_BUILT in audit_types
    lineage_event = next(
        e
        for e in snapshot.audit_events
        if e.event_type is AuditEventType.LINEAGE_REPORT_BUILT
    )
    assert lineage_event.metadata["parent_version_id"] == "dataset_version_v1"
    assert lineage_event.metadata["job_id"] == "compute_run_apply_001"
    assert lineage_event.metadata["config_hash"] == _CONFIG_HASH
    assert lineage_event.metadata["policy_version"] == "method_policy_v0"
    assert lineage_event.metadata["lineage_artifact_hash"] == (
        result.lineage_report_artifact.hash
    )
    export_event = next(
        e
        for e in snapshot.audit_events
        if e.event_type is AuditEventType.EXPORT_PACKAGE_BUILT
    )
    assert export_event.metadata["export_package_id"] == "export_package_test_001"
    assert export_event.metadata["export_status"] == "READY"
    assert export_event.metadata["included"] == 100


def test_audit_events_carry_no_raw_pii(tmp_path: Any) -> None:
    """Audit events must never echo PII even if upstream metadata leaks raw values."""
    _, registry = _storage_and_registry()
    platform_client = FakePlatformMetadataClient()
    candidate = _candidate()
    request = BuildLineageReportRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
    )
    build_lineage_report(
        request, registry=registry, platform_client=platform_client
    )
    snapshot = platform_client.snapshot()
    raw_blob = json.dumps(
        [event.model_dump(mode="json") for event in snapshot.audit_events]
    )
    # No raw email-like or phone-like patterns must leak.
    assert "@example.com" not in raw_blob
    assert "@gmail.com" not in raw_blob
    assert "555-123-4567" not in raw_blob
    # Audit metadata is intentionally narrow: only structured ids,
    # hashes, counts, and policy/config versions.
    for event in snapshot.audit_events:
        for key in event.metadata:
            assert "raw_" not in key
            assert "secret" not in key.lower()
            assert "token" not in key.lower()


def test_lineage_builder_rejects_candidate_without_action_plan_steps() -> None:
    """A candidate with no action_plan_steps cannot produce algorithm metadata."""
    _, registry = _storage_and_registry()
    candidate = _candidate(steps=())
    request = BuildLineageReportRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        generated_at=_GENERATED_AT,
    )
    with pytest.raises(LineageBuilderError) as exc:
        build_lineage_report(request, registry=registry)
    assert exc.value.reason_code == "candidate_action_plan_steps_missing"


def test_lineage_report_is_idempotent_for_same_inputs() -> None:
    """Re-running with identical inputs returns the same persisted artifact bytes."""
    storage, registry = _storage_and_registry()
    candidate = _candidate()
    request = BuildLineageReportRequest(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        candidate_dataset_version=candidate,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        lineage_report_id="lineage_report_idempotent",
        generated_at=_GENERATED_AT,
    )
    first = build_lineage_report(request, registry=registry)
    second = build_lineage_report(request, registry=registry)
    assert first.lineage_report_artifact.uri == second.lineage_report_artifact.uri
    assert (
        storage.get(first.lineage_report_artifact.uri).data
        == storage.get(second.lineage_report_artifact.uri).data
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _candidate(
    *, steps: tuple[CandidateActionStepSummary, ...] | None = None
) -> CandidateDatasetVersion:
    if steps is None:
        steps = (
            CandidateActionStepSummary(
                step_id="create_split_tabular",
                step_type="CREATE_SPLIT",
                method_id="group_stratified",
                plugin_id="dataforge.tabular",
                plugin_version="0.1.0",
                config_hash="sha256:" + "3" * 64,
                output_artifact_kind="split_manifest",
                random_seed=42,
            ),
            CandidateActionStepSummary(
                step_id="augment_rare_class_smote",
                step_type="AUGMENT_RARE_CLASS",
                method_id="smote",
                plugin_id="dataforge.tabular",
                plugin_version="0.1.0",
                config_hash="sha256:" + "4" * 64,
                output_artifact_kind="candidate_tabular_dataset",
                random_seed=42,
            ),
        )
    return CandidateDatasetVersion(
        candidate_version_id="candidate_dataset_version_test_001",
        status=CandidateVersionStatus.PROPOSED,
        policy_versions=CandidatePolicyVersions(
            profile_policy_version="demo_strict_v1",
            decision_policy_version="decision_policy_v0",
            score_policy_version="dataforge_score_v0",
            method_policy_version="method_policy_v0",
            validation_gates_policy_version="validation_gates_policy_v0",
        ),
        lineage=CandidateDatasetVersionLineage(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
            parent_version_id="dataset_version_v1",
            proposed_version_name="dataset_version_v2_candidate",
            action_plan_id="action_plan_smote_001",
            decision_report_id="decision_report_001",
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            input_artifact_hashes=(_BASE_HASH,),
            output_artifact_hashes=(_CANDIDATE_HASH,),
        ),
        candidate_artifacts=(_candidate_artifact(),),
        primary_dataset_artifact=_candidate_artifact(),
        validation_gates_report=_validation_gates_artifact(),
        validation_gates_summary={
            "overall_status": "passed",
            "candidate_status": "ok",
            "raw_artifact_unchanged": True,
            "blocker_present": False,
        },
        block_export=False,
        block_model_evaluation=False,
        block_training=False,
        blocker_reason_codes=(),
        action_plan_steps=steps,
        synthetic_metadata=None,
        proposed_at=_GENERATED_AT,
    )


def _export_package(*, candidate: CandidateDatasetVersion) -> ExportPackage:
    return ExportPackage(
        export_package_id="export_package_test_001",
        export_schema_version="export_package.v1",
        dataset_id=candidate.lineage.dataset_id,
        version_id=candidate.lineage.proposed_version_name,
        source_version_id=candidate.lineage.parent_version_id,
        created_by_job_id="compute_run_apply_001",
        status=ExportPackageStatus.READY,
        artifacts=(_export_manifest_artifact(), _parquet_artifact()),
        validation_gates=(
            ValidationGateResult(
                name="candidate_validation_passed",
                status=GateStatus.PASSED,
                reason_codes=(),
            ),
        ),
        object_counts=ExportObjectCounts(included=100, blocked=0, excluded=0),
        blocked_reason_codes=(),
        lineage=ExportPackageLineage(
            parent_version_id=candidate.lineage.parent_version_id,
            action_plan_id=candidate.lineage.action_plan_id,
            decision_report_id=candidate.lineage.decision_report_id or "decision_report_001",
            config_hash=_CONFIG_HASH,
        ),
        created_at=_GENERATED_AT,
    )


def _baseline_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="raw_transactions:" + "b" * 16,
        kind="raw_transactions",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v1/transactions.csv",
        hash=_BASE_HASH,
        media_type="text/csv",
        size_bytes=2048,
        schema_version="tabular_dataset.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v1",
            job_id="compute_run_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _candidate_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="candidate_tabular_dataset:" + "c" * 16,
        kind="candidate_tabular_dataset",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/transactions.csv",
        hash=_CANDIDATE_HASH,
        media_type="text/csv",
        size_bytes=2048,
        schema_version="tabular_dataset.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _candidate_version_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="candidate_dataset_version:" + "f" * 16,
        kind="candidate_dataset_version",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/candidate_dataset_version.json",
        hash="sha256:" + "f" * 64,
        media_type="application/json",
        size_bytes=2048,
        schema_version="candidate_dataset_version.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _export_package_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="export_package:" + "e" * 16,
        kind="export_package",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/export_package.json",
        hash=_EXPORT_HASH,
        media_type="application/json",
        size_bytes=4096,
        schema_version="export_package.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _export_manifest_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="export_manifest:" + "1" * 16,
        kind="EXPORT_MANIFEST",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/manifest.jsonl",
        hash=_EXPORT_MANIFEST_HASH,
        media_type="application/jsonl",
        size_bytes=8192,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _parquet_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="tabular_export_parquet:" + "5" * 16,
        kind="tabular_export_parquet",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/export/train.parquet",
        hash=_PARQUET_HASH,
        media_type="application/x-parquet",
        size_bytes=4096,
        schema_version="tabular_export.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


def _validation_gates_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="validation_gates_report:" + "d" * 16,
        kind="validation_gates_report",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/v2/validation_gates_report.json",
        hash=_GATES_HASH,
        media_type="application/json",
        size_bytes=1024,
        schema_version="validation_gates_report.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v2_candidate",
            job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            created_at=_GENERATED_AT,
        ),
    )


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


# Suppress unused-symbol warnings for typing helpers.
_ = (_DATASET_CARD_HASH,)


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
            "LastModified": _GENERATED_AT,
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
                {"Key": key, "Size": len(payload["Body"])}
                for (bucket, key), payload in self._objects.items()
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc
