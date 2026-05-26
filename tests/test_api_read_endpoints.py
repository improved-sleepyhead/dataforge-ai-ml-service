"""TASK-058 read endpoints: jobs, action plans, reports, issues, compare.

The compute-side test API exposes the most recently registered ANALYZE_ONLY
and APPLY_SELECTED_ACTIONS results for the platform/UI test harness. The
endpoints under test are:

* ``GET /api/v1/jobs/{job_id}``
* ``GET /api/v1/action-plans/{action_plan_id}``
* ``GET /api/v1/reports/{report_id}``
* ``GET /api/v1/reports/{report_id}/issues``
* ``GET /api/v1/compare/{compare_id}``

The tests run analyze + execute through the existing POST endpoints first,
then follow the test API to read back the same artifacts. ``ReviewQueue``
and ``VersionCompareReport`` artifacts are seeded into the in-memory
compute store directly because their builders are exercised by other test
modules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.adapters import FakePlatformMetadataClient
from app.api.main import create_app
from app.api.security import (
    ORGANIZATION_ID_HEADER,
    PROJECT_ID_HEADER,
    SERVICE_IDENTITY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    build_service_signature,
)
from app.api.store import ComputeResultStore
from app.domain import (
    ArtifactLineage,
    ArtifactRef,
    ComputeRunStatus,
    DataModality,
    ReviewExportPolicy,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    TabularProfileReport,
    WorkflowType,
)
from app.kernel import (
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    action_plan_integrity_hash,
    build_action_plan_preview,
    build_method_recommendations,
)
from app.kernel.config import ServiceConfig, load_config
from app.validation.contracts import load_contract_pack
from tests.test_apply_workflow import _compute_resources_with_source_artifact

_GENERATED_AT = datetime(2026, 6, 4, 12, 0, tzinfo=UTC)


def test_read_endpoints_return_recorded_analyze_apply_state(tmp_path: Path) -> None:
    """Steps 1+2: run analyze/execute, read artifacts back through GET endpoints."""
    config = _test_config()
    resources, source_artifact = _compute_resources_with_source_artifact(
        tmp_path=tmp_path,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )
    app = create_app(config=config, compute_resources=resources)
    client = TestClient(app, raise_server_exceptions=False)

    # Step 1: analyze flow records a job.
    analyze_payload = _analyze_payload()
    analyze_body = _body_bytes(analyze_payload)
    analyze_resp = client.post(
        "/api/v1/jobs/analyze-dataset",
        content=analyze_body,
        headers=_signed_headers(config=config, body=analyze_body, payload=analyze_payload),
    )
    assert analyze_resp.status_code == 202
    analyze_job_id = analyze_resp.json()["job_id"]

    # Step 1: preview + execute records both an action plan and a job.
    preview_payload = _action_plan_preview_payload()
    preview_body = _body_bytes(preview_payload)
    preview_resp = client.post(
        "/api/v1/action-plans/preview",
        content=preview_body,
        headers=_signed_headers(config=config, body=preview_body, payload=preview_payload),
    )
    assert preview_resp.status_code == 200
    preview_plan = preview_resp.json()["action_plan"]
    preview_action_plan_id = preview_plan["action_plan_id"]

    execute_payload = _action_plan_execute_payload(source_artifacts=[source_artifact])
    execute_body = _body_bytes(execute_payload)
    execute_resp = client.post(
        "/api/v1/action-plans/execute-approved",
        content=execute_body,
        headers=_signed_headers(config=config, body=execute_body, payload=execute_payload),
    )
    assert execute_resp.status_code == 202
    execute_action_plan_id = execute_resp.json()["action_plan_id"]
    execute_job_id = execute_resp.json()["job_id"]
    expected_action_plan_hash = execute_resp.json()["action_plan_hash"]

    # Step 2: GET /jobs/{job_id} returns analyze run summary.
    headers = _signed_headers(
        config=config,
        body=b"",
        payload={"organization_id": "org_1", "project_id": "project_1"},
    )
    job_resp = client.get(f"/api/v1/jobs/{analyze_job_id}", headers=headers)
    assert job_resp.status_code == 200
    job_body = job_resp.json()
    assert job_body["job_id"] == analyze_job_id
    assert job_body["workflow_type"] == WorkflowType.ANALYZE_ONLY
    assert job_body["status"] == ComputeRunStatus.ACCEPTED
    assert job_body["mutates_dataset"] is False
    assert job_body["materialized_assets"]
    # raw payloads/PII never present.
    _assert_safe_response_text(job_resp.text)

    # Step 2: GET /jobs/{job_id} returns apply run summary with real gated refs.
    apply_job_resp = client.get(f"/api/v1/jobs/{execute_job_id}", headers=headers)
    assert apply_job_resp.status_code == 200
    apply_job_body = apply_job_resp.json()
    assert apply_job_body["workflow_type"] == WorkflowType.APPLY_SELECTED_ACTIONS
    assert apply_job_body["mutates_dataset"] is True
    assert apply_job_body["action_plan_id"] == execute_action_plan_id
    assert apply_job_body["action_plan_hash"] == expected_action_plan_hash
    assert apply_job_body["candidate_artifact_uri"] is not None
    assert apply_job_body["export_package_artifact_uri"] is not None
    assert ".placeholder." not in apply_job_body["candidate_artifact_uri"]
    assert ".placeholder." not in apply_job_body["export_package_artifact_uri"]
    _assert_safe_response_text(apply_job_resp.text)

    # Step 2: GET /action-plans/{id} returns preview-only metadata for previewed plan
    # and full execution metadata for the executed plan.
    preview_get = client.get(
        f"/api/v1/action-plans/{preview_action_plan_id}", headers=headers
    )
    assert preview_get.status_code == 200
    preview_get_body = preview_get.json()
    assert preview_get_body["mutates_dataset"] is False
    assert preview_get_body["accepted_step_ids"] == []
    assert preview_get_body["action_plan_hash"] is None
    assert preview_get_body["action_plan"]["action_plan_id"] == preview_action_plan_id

    execute_get = client.get(
        f"/api/v1/action-plans/{execute_action_plan_id}", headers=headers
    )
    assert execute_get.status_code == 200
    execute_get_body = execute_get.json()
    assert execute_get_body["mutates_dataset"] is True
    assert execute_get_body["action_plan_hash"] == expected_action_plan_hash
    assert execute_get_body["workflow_type"] == WorkflowType.APPLY_SELECTED_ACTIONS
    assert execute_get_body["accepted_step_ids"]
    assert execute_get_body["action_plan"]["action_plan_id"] == execute_action_plan_id

    # Step 2/3: GET /reports/{id} and /reports/{id}/issues return seeded artifacts.
    store: ComputeResultStore = app.state.compute_store
    decision_report = _decision_report_fixture()
    store.record_decision_report(decision_report)
    review_queues = _review_queues_fixture(decision_report.decision_report_id)
    store.record_review_queues(
        report_id=decision_report.decision_report_id, queues=review_queues
    )

    report_resp = client.get(
        f"/api/v1/reports/{decision_report.decision_report_id}", headers=headers
    )
    assert report_resp.status_code == 200
    report_body = report_resp.json()
    assert report_body["decision_report_id"] == decision_report.decision_report_id
    assert report_body["dataset_decision"] == "READY_FOR_TRAINING"
    _assert_safe_response_text(report_resp.text)

    issues_resp = client.get(
        f"/api/v1/reports/{decision_report.decision_report_id}/issues", headers=headers
    )
    assert issues_resp.status_code == 200
    issues_body = issues_resp.json()
    assert issues_body["report_id"] == decision_report.decision_report_id
    summaries = {
        entry["queue_type"]: entry for entry in issues_body["review_queue_summary"]
    }
    assert ReviewQueueType.LABEL_REVIEW in summaries
    assert ReviewQueueType.PRIVACY_REVIEW in summaries
    assert summaries[ReviewQueueType.PRIVACY_REVIEW]["raw_pii_allowed"] is False
    assert summaries[ReviewQueueType.PRIVACY_REVIEW]["redacted_only"] is True
    assert (
        issues_body["total_item_count"]
        == sum(len(queue.objects) for queue in review_queues)
    )
    _assert_safe_response_text(issues_resp.text)

    # Step 2: GET /compare/{compare_id} returns the seeded VersionCompareReport.
    compare_report = _version_compare_fixture()
    store.record_version_compare(compare_report)
    compare_resp = client.get(
        f"/api/v1/compare/{compare_report.report_id}", headers=headers
    )
    assert compare_resp.status_code == 200
    compare_body = compare_resp.json()
    assert compare_body["report_id"] == compare_report.report_id
    assert compare_body["report_schema_version"] == compare_report.report_schema_version
    _assert_safe_response_text(compare_resp.text)


def test_read_endpoints_404_for_unknown_ids() -> None:
    """Unknown ids return a stable 404 ErrorResponse without raw details."""
    config = _test_config()
    app = create_app(config=config)
    client = TestClient(app, raise_server_exceptions=False)
    headers = _signed_headers(
        config=config,
        body=b"",
        payload={"organization_id": "org_1", "project_id": "project_1"},
    )

    for path in (
        "/api/v1/jobs/missing_job",
        "/api/v1/action-plans/missing_plan",
        "/api/v1/reports/missing_report",
        "/api/v1/reports/missing_report/issues",
        "/api/v1/compare/missing_compare",
    ):
        response = client.get(path, headers=headers)
        assert response.status_code == 404, f"{path} should return 404"
        body = response.json()
        assert "error" in body
        assert body["error"]["recoverable"] is True


def test_read_endpoints_require_signed_service_identity() -> None:
    """Read endpoints reject anonymous requests with a 401-class error."""
    config = _test_config()
    app = create_app(config=config)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/v1/jobs/anything")
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["stage"].startswith("api.security")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_safe_response_text(body: str) -> None:
    """Read endpoints must never echo raw secrets, raw PII or raw row content."""
    forbidden_substrings = (
        "@example.com",
        "secret",
        "password",
        "token",
        "+10000001234",
    )
    lower = body.lower()
    for marker in forbidden_substrings:
        assert marker not in lower, f"read endpoint leaked {marker!r}"


def _decision_report_fixture() -> Any:
    from app.domain import (
        DatasetDecision,
        DatasetReadiness,
        DecisionReport,
        PolicyVersions,
        ReadinessAssessment,
    )

    return DecisionReport(
        decision_report_id="decision_report_read_001",
        dataset_id="dataset_1",
        version_id="dataset_version_1",
        decision_schema_version="decision_report.v1",
        dataset_decision=DatasetDecision.READY_FOR_TRAINING,
        readiness=ReadinessAssessment(
            status=DatasetReadiness.READY_FOR_EXPORT,
            score=0.85,
            reason_codes=("schema_valid",),
        ),
        critical_blockers=(),
        safe_actions_available=True,
        recommended_next_job=None,
        object_decisions=(),
        recommended_actions=(),
        policy_versions=PolicyVersions(
            decision_policy="decision_v0",
            score_policy="score_v0",
            privacy_policy="privacy_v0",
            method_policy="method_selection_v0",
        ),
        generated_at=_GENERATED_AT,
    )


def _review_queues_fixture(report_id: str) -> tuple[ReviewQueue, ...]:
    label_queue = ReviewQueue(
        review_queue_id=f"{report_id}_label",
        queue_schema_version="review_queue.v1",
        queue_type=ReviewQueueType.LABEL_REVIEW,
        target_tool="label_studio",
        dataset_id="dataset_1",
        version_id="dataset_version_1",
        created_by_job_id="compute_run_read_001",
        objects=(
            ReviewQueueItem(
                object_id="obj_label_1",
                modality=DataModality.TEXT,
                object_type="text_record",
                reason_codes=("ambiguous_object",),
                priority=0.8,
                safe_preview_uri=(
                    "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/"
                    "previews/obj_label_1.json"
                ),
                evidence_refs=(),
            ),
        ),
        export_policy=ReviewExportPolicy(
            raw_pii_allowed=False,
            redacted_only=False,
            allow_external_tool=True,
        ),
        created_at=_GENERATED_AT,
    )
    privacy_queue = ReviewQueue(
        review_queue_id=f"{report_id}_privacy",
        queue_schema_version="review_queue.v1",
        queue_type=ReviewQueueType.PRIVACY_REVIEW,
        target_tool=None,
        dataset_id="dataset_1",
        version_id="dataset_version_1",
        created_by_job_id="compute_run_read_001",
        objects=(
            ReviewQueueItem(
                object_id="obj_privacy_1",
                modality=DataModality.TEXT,
                object_type="text_record",
                reason_codes=("pii_detected",),
                priority=0.9,
                safe_preview_uri=(
                    "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/"
                    "previews/obj_privacy_1.json"
                ),
                evidence_refs=(),
            ),
        ),
        export_policy=ReviewExportPolicy(
            raw_pii_allowed=False,
            redacted_only=True,
            allow_external_tool=False,
        ),
        created_at=_GENERATED_AT,
    )
    return (label_queue, privacy_queue)


def _version_compare_fixture() -> Any:
    pack = load_contract_pack()
    example = next(
        example
        for example in pack.examples
        if example.name.startswith("version_compare_report")
    )
    from app.domain import VersionCompareReport

    return VersionCompareReport.model_validate(example.payload)


def _test_config() -> ServiceConfig:
    return load_config(
        {
            "DATAFORGE_PROFILE": "demo_strict",
            "DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL": "http://localhost:9000",
            "DATAFORGE_OBJECT_STORAGE_BUCKET": "dataforge-local",
            "DATAFORGE_PLATFORM_CALLBACK_URL": "http://platform.local/api/ml/jobs/callback",
            "DATAFORGE_SERVICE_SIGNING_SECRET": "local-dev-signing-secret",
            "DATAFORGE_DAGSTER_HOME": "/tmp/dataforge-dagster",
            "DATAFORGE_POLICY_CONFIG_PATH": "configs/policies/demo_strict.yaml",
            "DATAFORGE_DECISION_POLICY_PATH": "configs/policies/decision_v0.yaml",
            "DATAFORGE_SCORE_POLICY_PATH": "configs/policies/score_v0.yaml",
        }
    )


def _analyze_payload() -> dict[str, object]:
    return {
        "platform_job_id": "platform_job_read_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "dataset_version_id": "dataset_version_1",
        "dataset_object_refs": [_artifact_ref("raw_archive_1", "raw_dataset_archive")],
        "prediction_artifact_refs": [],
    }


def _action_plan_preview_payload() -> dict[str, object]:
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )
    return {
        "platform_job_id": "platform_job_read_preview_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "source_dataset_version_id": "dataset_version_1",
        "decision_report_id": "decision_report_preview_only_001",
        "selected_decision_ids": [recommendations[0].recommendation_id],
        "selected_method_overrides": {},
        "method_recommendations": [
            recommendations[0].model_dump(mode="json"),
        ],
        "created_by_user_id": "platform_user_read_001",
        "input_artifacts": [
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl"
        ],
        "target_version_name": "dataset_version_2_preview_read",
    }


def _action_plan_execute_payload(
    *,
    source_artifacts: list[ArtifactRef] | None = None,
) -> dict[str, object]:
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )
    input_artifacts = (
        tuple(artifact.uri for artifact in source_artifacts)
        if source_artifacts
        else (
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
        )
    )
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_001",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_read_apply",
            input_artifacts=input_artifacts,
            target_version_name="dataset_version_2_candidate_read",
            created_at=_GENERATED_AT,
        )
    ).model_copy(
        update={
            "requires_approval": True,
            "approval_request_id": "approval_request_read_001",
        }
    )
    action_plan = plan.model_dump(mode="json")
    return {
        "platform_job_id": "platform_job_read_apply_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "source_dataset_version_id": "dataset_version_1",
        "action_plan": action_plan,
        "approval_metadata": {
            "approval_id": "approval_read_001",
            "approval_request_id": "approval_request_read_001",
            "approved_by_user_id": "platform_owner_read_001",
            "approved_at": "2026-06-04T12:05:00Z",
            "action_plan_id": action_plan["action_plan_id"],
            "action_plan_hash": action_plan_integrity_hash(plan),
            "decision_report_id": action_plan["created_from_decision_report"],
            "source_dataset_version_id": action_plan["source_dataset_version_id"],
        },
        "source_artifacts": []
        if source_artifacts is None
        else [artifact.model_dump(mode="json") for artifact in source_artifacts],
    }


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example
        for example in pack.examples
        if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def _artifact_ref(artifact_id: str, kind: str) -> dict[str, object]:
    artifact = ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=f"s3://dataforge-local/dataforge/org_1/project_1/dataset_1/{artifact_id}.json",
        hash="sha256:" + "a" * 64,
        media_type="application/json",
        size_bytes=128,
        schema_version="v0.1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_1",
            job_id="platform_job_read_001",
            config_hash="sha256:" + "b" * 64,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    return artifact.model_dump(mode="json")


def _body_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _signed_headers(
    *,
    config: ServiceConfig,
    body: bytes,
    payload: dict[str, object],
) -> dict[str, str]:
    signed_at = datetime.now(UTC).isoformat()
    service_identity = config.platform.service_identity
    organization_id = str(payload["organization_id"])
    project_id = str(payload["project_id"])
    return {
        SERVICE_IDENTITY_HEADER: service_identity,
        TIMESTAMP_HEADER: signed_at,
        ORGANIZATION_ID_HEADER: organization_id,
        PROJECT_ID_HEADER: project_id,
        SIGNATURE_HEADER: build_service_signature(
            secret=config.platform.service_signing_secret.get_secret_value(),
            service_identity=service_identity,
            timestamp=signed_at,
            organization_id=organization_id,
            project_id=project_id,
            body=body,
        ),
        "content-type": "application/json",
    }
