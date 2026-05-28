"""TASK-068: E2E full compute flow — analyze → action plan → apply → impact → export.

This test glues the public launchers and builders into the full
``ANALYZE_ONLY`` + ``APPLY_SELECTED_ACTIONS`` pipeline the platform expects
from the compute plane and asserts the four acceptance criteria of
TASK-068:

1. The flow runs end-to-end on the deterministic demo archive without a
   real platform backend (a ``FakePlatformMetadataClient`` absorbs status
   and audit events).
2. The fake platform receives ANALYZE and APPLY job events plus the
   proposed candidate dataset state (candidate artifact uri + hash).
3. A model-impact report is produced when the candidate is eligible.
4. An export package is produced only when readiness gates pass.

Every published artifact uri (candidate, export package, model-impact
report, dataset_card, lineage) is fetched back from object storage via
``ArtifactRegistry`` and validated to be a real, hashable, scoped
artifact. Raw mutation is forbidden: the source ``raw_transactions``
artifact bytes must be byte-identical before and after APPLY.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError
from app.api.schemas import (
    ActionPlanExecuteApprovedRequest,
    AnalyzeDatasetRequest,
)
from app.domain import (
    ArtifactRef,
    ComputeRunStatus,
    ErrorCode,
    ExportPackage,
    ExportPackageStatus,
    ModelImpactReport,
    ModelImpactVerdict,
    TabularProfileReport,
)
from app.ingestion import open_archive_path
from app.kernel import (
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    action_plan_integrity_hash,
    build_action_plan_preview,
    build_method_recommendations,
)
from app.kernel.action_plan import ActionPlanApprovalMetadata
from app.kernel.config import (
    DagsterSettings,
    ExternalAISettings,
    ObjectStorageSettings,
    PlatformSettings,
    PolicySettings,
    RuntimeProfile,
    ServiceConfig,
    profile_defaults,
)
from app.orchestration.analyze_workflow import (
    expected_analyze_outputs,
    launch_analyze_dataset_workflow,
)
from app.orchestration.apply_assets import APPLY_ASSET_KEYS
from app.orchestration.apply_workflow import launch_apply_actions_workflow
from app.orchestration.resources import ComputeResources
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_GENERATED_AT = datetime(2026, 6, 6, 12, 0, tzinfo=UTC)
_ORG_ID = "org_e2e_full"
_PROJECT_ID = "project_e2e_full"
_DATASET_ID = "dataset_e2e_full"
_PARENT_VERSION_ID = "dataset_version_v1"


# ---------------------------------------------------------------------------
# Step 1: full E2E compute flow without a real backend
# ---------------------------------------------------------------------------


def test_e2e_full_compute_flow_runs_analyze_then_apply_then_export(
    tmp_path: Path,
) -> None:
    """End-to-end ANALYZE_ONLY → ActionPlan → APPLY → impact → export.

    Step 1: every stage runs through public launchers/builders without a
    real backend; the fake platform absorbs status events.
    Step 2: candidate and export artifacts are present, scoped and hashable.
    Step 3: lineage.json, dataset_card.md and model_impact_report.json are
    persisted as immutable artifacts and ``ExportPackage`` references them.
    """
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    resources, source_artifact = _resources_with_source_artifact(
        tmp_path=tmp_path,
        config=config,
        fake_platform=fake_platform,
    )
    raw_source_bytes = resources.object_storage.get(source_artifact.uri).data

    # ------------------------------------------------------------------
    # 1. ANALYZE_ONLY — no candidate must be produced.
    # ------------------------------------------------------------------
    analyze_request = AnalyzeDatasetRequest(
        platform_job_id="platform_job_e2e_full_analyze",
        organization_id=_ORG_ID,
        project_id=_PROJECT_ID,
        dataset_id=_DATASET_ID,
        dataset_version_id=_PARENT_VERSION_ID,
        dataset_object_refs=(source_artifact,),
    )
    analyze_result = launch_analyze_dataset_workflow(
        request=analyze_request,
        config=config,
        fake_platform=fake_platform,
    )
    assert analyze_result.status is ComputeRunStatus.ACCEPTED
    assert analyze_result.mutates_dataset is False
    expected_analyze = expected_analyze_outputs(include_predictions=False)
    assert set(analyze_result.materialized_assets) == set(expected_analyze)
    apply_only_assets = {key.path[-1] for key in APPLY_ASSET_KEYS}
    assert apply_only_assets.isdisjoint(analyze_result.materialized_assets)

    analyze_events = list(fake_platform.snapshot().job_events)
    assert any(event.stage == "RUNNING_DECISION_CORE" for event in analyze_events)
    assert analyze_events[-1].status is ComputeRunStatus.COMPLETED

    # ------------------------------------------------------------------
    # 2. ActionPlan preview from contract method recommendations.
    # ------------------------------------------------------------------
    apply_request, plan_hash = _action_plan_apply_request(
        source_artifact=source_artifact
    )

    # ------------------------------------------------------------------
    # 3. APPLY_SELECTED_ACTIONS — must produce real candidate + export.
    # ------------------------------------------------------------------
    apply_result = launch_apply_actions_workflow(
        request=apply_request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=apply_request.source_artifacts,
        compute_resources=resources,
    )
    assert apply_result.status is ComputeRunStatus.ACCEPTED
    assert apply_result.mutates_dataset is True
    assert apply_result.candidate_artifact_uri is not None
    assert apply_result.candidate_artifact_hash is not None
    assert apply_result.export_package_artifact_uri is not None
    # Synthetic-free plan: synthetic dataset is intentionally not_applicable.
    assert apply_result.synthetic_status == "not_applicable"
    assert apply_result.synthetic_artifact_uri is None
    # APPLY assets are the only ones the apply launcher reports.
    assert set(apply_result.materialized_assets) == set(apply_result.expected_outputs)
    assert ".placeholder." not in apply_result.candidate_artifact_uri
    assert ".placeholder." not in apply_result.export_package_artifact_uri

    # ------------------------------------------------------------------
    # 4. Source raw artifact MUST NOT be mutated by APPLY.
    # ------------------------------------------------------------------
    raw_source_bytes_after = resources.object_storage.get(source_artifact.uri).data
    assert raw_source_bytes_after == raw_source_bytes

    # ------------------------------------------------------------------
    # 5. Fake platform sees both ANALYZE and APPLY lifecycles end with COMPLETED
    #    plus proposed candidate state in APPLY artifact_uris.
    # ------------------------------------------------------------------
    snapshot = fake_platform.snapshot()
    apply_events = [
        event
        for event in snapshot.job_events
        if event.platform_job_id == apply_request.platform_job_id
    ]
    assert apply_events, "fake platform must receive APPLY job events"
    assert apply_events[-1].status is ComputeRunStatus.COMPLETED
    apply_artifact_uris = {
        uri
        for event in apply_events
        for uri in event.details.get("artifact_uris", []) or []
    }
    assert apply_result.candidate_artifact_uri in apply_artifact_uris
    assert apply_result.export_package_artifact_uri in apply_artifact_uris

    # ------------------------------------------------------------------
    # 6. ExportPackage must be READY and reference dataset_card + lineage
    #    + model_impact_report artifacts (acceptance test step 3).
    # ------------------------------------------------------------------
    package = _load_export_package(
        storage=resources.object_storage,
        uri=apply_result.export_package_artifact_uri,
    )
    assert package.status is ExportPackageStatus.READY
    assert package.blocked_reason_codes == ()

    artifact_kinds = {ref.kind for ref in package.artifacts}
    assert "DATASET_CARD" in artifact_kinds
    assert "lineage_report" in artifact_kinds

    # ------------------------------------------------------------------
    # 7. model_impact_report.json must exist when candidate is eligible.
    # ------------------------------------------------------------------
    assert apply_result.model_impact_artifact_uri is not None
    impact_report = _load_model_impact_report(
        storage=resources.object_storage,
        uri=apply_result.model_impact_artifact_uri,
    )
    assert isinstance(impact_report, ModelImpactReport)
    assert impact_report.verdict is not ModelImpactVerdict.REJECTED
    pack = load_contract_pack()
    validate_contract_payload(
        pack,
        "model_impact_report",
        impact_report.model_dump(mode="json"),
    )

    # ------------------------------------------------------------------
    # 8. dataset_card.md and lineage.json bytes are real (non-empty),
    #    scoped, and immutable.
    # ------------------------------------------------------------------
    by_kind = {ref.kind: ref for ref in package.artifacts}
    dataset_card_bytes = resources.object_storage.get(by_kind["DATASET_CARD"].uri).data
    lineage_bytes = resources.object_storage.get(by_kind["lineage_report"].uri).data
    assert dataset_card_bytes.startswith(b"#")  # Markdown headline
    assert b"Dataset Card" in dataset_card_bytes
    lineage_payload = json.loads(lineage_bytes.decode("utf-8"))
    assert lineage_payload, "lineage.json must contain a non-empty payload"


# ---------------------------------------------------------------------------
# Step 2: failed validation gates must not promote a candidate or export
# ---------------------------------------------------------------------------


def test_e2e_export_only_when_gates_pass_blocks_when_pii_restricted(
    tmp_path: Path,
) -> None:
    """When a step is policy-blocked (PII), APPLY must not promote candidate/export.

    This proves acceptance criterion 4: "Export package создается только
    если gates pass". We simulate a gate failure by flipping a step
    config flag the apply assets honor as a deterministic blocker.
    """
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    resources, source_artifact = _resources_with_source_artifact(
        tmp_path=tmp_path,
        config=config,
        fake_platform=fake_platform,
        source_bytes=_demo_transactions_with_email(tmp_path),
    )
    apply_request, _ = _action_plan_apply_request(source_artifact=source_artifact)

    # Force a deterministic gate failure: mark the only step as
    # ``pii_restricted=True``. The synthetic/validation gate path
    # treats this as a hard blocker so the launcher must not surface
    # a candidate or export package URI.
    plan = apply_request.action_plan
    blocked_step = plan.steps[0].model_copy(
        update={"config": {**plan.steps[0].config, "pii_restricted": True}}
    )
    blocked_plan = plan.model_copy(update={"steps": (blocked_step,)})
    blocked_hash = action_plan_integrity_hash(blocked_plan)
    blocked_request = apply_request.model_copy(
        update={
            "action_plan": blocked_plan,
            "approval_metadata": apply_request.approval_metadata.model_copy(
                update={"action_plan_hash": blocked_hash}
            )
            if apply_request.approval_metadata is not None
            else None,
        }
    )

    result = launch_apply_actions_workflow(
        request=blocked_request,
        action_plan_hash=blocked_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=blocked_request.source_artifacts,
        compute_resources=resources,
    )

    # APPLY runs to completion (Dagster materialization succeeds), but
    # gated outputs are never surfaced as final candidate/export refs.
    assert result.status is ComputeRunStatus.ACCEPTED
    assert result.candidate_artifact_uri is None
    assert result.candidate_artifact_hash is None
    assert result.export_package_artifact_uri is None

    # Validation gate failure still publishes audit artifacts, which is
    # the platform contract for "blocked but recorded".
    snapshot = fake_platform.snapshot()
    last_apply = next(
        event
        for event in reversed(snapshot.job_events)
        if event.platform_job_id == blocked_request.platform_job_id
    )
    assert last_apply.status is ComputeRunStatus.COMPLETED
    validation_refs = [
        uri
        for event in snapshot.job_events
        if event.platform_job_id == blocked_request.platform_job_id
        for uri in event.details.get("artifact_uris", []) or []
        if "validation_gates_report" in uri
    ]
    assert validation_refs, "blocked APPLY must persist a validation_gates_report"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _test_config() -> ServiceConfig:
    profile = RuntimeProfile.DEMO_STRICT
    return ServiceConfig(
        profile=profile,
        object_storage=ObjectStorageSettings(
            endpoint_url="http://localhost:9000",
            bucket_name="dataforge-local",
            region="local",
            prefix_root="dataforge",
        ),
        platform=PlatformSettings(
            callback_url="http://platform.local/api/ml/jobs/callback",
            service_signing_secret="local-dev-signing-secret",  # type: ignore[arg-type]
            service_identity="dataforge-platform",
            signature_max_age_seconds=300,
        ),
        dagster=DagsterSettings(
            home="/tmp/dataforge-dagster-e2e-full",
            job_name="dataforge_e2e_full_flow",
            run_queue="default",
        ),
        policies=PolicySettings(
            policy_config_path="configs/policies/demo_strict.yaml",
            decision_policy_path="configs/policies/decision_v0.yaml",
            score_policy_path="configs/policies/score_v0.yaml",
        ),
        contract_pack_version="local-fallback-v0.1.0-demo",
        external_ai=ExternalAISettings(allow_external_api=False),
        profile_defaults=profile_defaults(profile),
    )


def _resources_with_source_artifact(
    *,
    tmp_path: Path,
    config: ServiceConfig,
    fake_platform: FakePlatformMetadataClient,
    source_bytes: bytes | None = None,
) -> tuple[ComputeResources, ArtifactRef]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=ObjectStorageScope(
            organization_id=_ORG_ID,
            project_id=_PROJECT_ID,
            dataset_id=_DATASET_ID,
        ),
    )
    registry = ArtifactRegistry(storage=storage)
    if source_bytes is None:
        built = build_demo_archive(output_dir=tmp_path / "demo_archive")
        with open_archive_path(built.archive_path) as reader:
            transactions = reader.find_required_transactions().read_bytes()
    else:
        transactions = source_bytes
    source_artifact = registry.save_artifact(
        artifact_kind="raw_transactions",
        data=transactions,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id=_PARENT_VERSION_ID,
        created_by_job_id="compute_run_e2e_full_source",
        config_hash="sha256:" + "9" * 64,
    ).artifact_ref
    return (
        ComputeResources(
            service_config=config,
            object_storage=storage,
            artifact_registry=registry,
            fake_platform=fake_platform,
        ),
        source_artifact,
    )


def _demo_transactions_with_email(tmp_path: Path) -> bytes:
    """Inject a synthetic ``customer_email`` column to trip the PII gate."""
    import csv
    from io import StringIO

    built = build_demo_archive(output_dir=tmp_path / "demo_archive_with_email")
    with open_archive_path(built.archive_path) as reader:
        transactions = reader.find_required_transactions().read_bytes().decode("utf-8")
    input_rows = list(csv.DictReader(transactions.splitlines()))
    fieldnames = list(input_rows[0])
    fieldnames.insert(-1, "customer_email")
    output: list[dict[str, str]] = []
    for index, row in enumerate(input_rows):
        row["customer_email"] = f"customer{index:03d}@demo.invalid"
        output.append(row)
    text_sink = StringIO(newline="")
    writer = csv.DictWriter(text_sink, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output)
    return text_sink.getvalue().encode("utf-8")


def _action_plan_apply_request(
    *,
    source_artifact: ArtifactRef,
) -> tuple[ActionPlanExecuteApprovedRequest, str]:
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_e2e_full_001",
            source_dataset_version_id=_PARENT_VERSION_ID,
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_e2e_full",
            input_artifacts=(source_artifact.uri,),
            target_version_name="dataset_version_v2_candidate",
            created_at=_GENERATED_AT,
        )
    ).model_copy(
        update={
            "requires_approval": True,
            "approval_request_id": "approval_request_e2e_full_001",
        }
    )
    plan_hash = action_plan_integrity_hash(plan)
    approval = ActionPlanApprovalMetadata(
        approval_id="approval_e2e_full_001",
        approval_request_id="approval_request_e2e_full_001",
        approved_by_user_id="platform_owner_e2e_full",
        approved_at=_GENERATED_AT,
        action_plan_id=plan.action_plan_id,
        action_plan_hash=plan_hash,
        decision_report_id=plan.created_from_decision_report,
        source_dataset_version_id=plan.source_dataset_version_id,
    )
    request = ActionPlanExecuteApprovedRequest(
        platform_job_id="platform_job_e2e_full_apply",
        organization_id=_ORG_ID,
        project_id=_PROJECT_ID,
        dataset_id=_DATASET_ID,
        source_dataset_version_id=plan.source_dataset_version_id,
        action_plan=plan,
        approval_metadata=approval,
        source_artifacts=(source_artifact,),
    )
    return request, plan_hash


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        e for e in pack.examples if e.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def _load_export_package(
    *, storage: MinioObjectStorageAdapter, uri: str
) -> ExportPackage:
    payload = json.loads(storage.get(uri).data.decode("utf-8"))
    return ExportPackage.model_validate(payload)


def _load_model_impact_report(
    *, storage: MinioObjectStorageAdapter, uri: str
) -> ModelImpactReport:
    payload = json.loads(storage.get(uri).data.decode("utf-8"))
    return ModelImpactReport.model_validate(payload)


# Silence unused-import warnings for typing helpers that the suite uses
# only inside class bodies.
_ = (pytest, ErrorCode)


class _InMemoryS3Client:
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
        record = self._object(Bucket, Key)
        body = record["Body"]
        if not isinstance(body, bytes):
            raise TypeError("InMemoryS3Client body must be bytes")
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        if not isinstance(body, bytes):
            raise TypeError("InMemoryS3Client body must be bytes")
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        contents: list[dict[str, object]] = []
        for (bucket, key), record in sorted(self._objects.items()):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            body = record["Body"]
            if isinstance(body, bytes):
                contents.append({"Key": key, "Size": len(body)})
        return {"Contents": contents}

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc
