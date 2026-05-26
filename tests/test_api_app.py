"""FastAPI app boundary tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.security import (
    ORGANIZATION_ID_HEADER,
    PROJECT_ID_HEADER,
    SERVICE_IDENTITY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    build_service_signature,
)
from app.domain import (
    ArtifactLineage,
    ArtifactRef,
    ComputeRunStatus,
    ErrorCode,
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
from app.orchestration.analyze_workflow import expected_analyze_outputs
from app.validation.contracts import load_contract_pack, validate_contract_payload


def test_health_returns_service_and_contract_versions() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "service_version": "0.1.0",
        "contract_pack_version": load_contract_pack().version,
    }


def test_unhandled_exception_returns_safe_error_response_without_raw_details() -> None:
    client = TestClient(create_app(include_test_error_route=True), raise_server_exceptions=False)

    response = client.get("/__test__/unhandled-error")

    assert response.status_code == 500
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.PLUGIN_EXECUTION_FAILED
    assert payload["error"]["message"] == "Internal compute service error."
    assert payload["error"]["stage"] == "api"
    assert "demo@example.com" not in response.text
    assert "raw secret token" not in response.text


def test_validation_error_returns_error_response() -> None:
    client = TestClient(create_app(include_test_error_route=True), raise_server_exceptions=False)

    response = client.get("/__test__/validation-error", params={"limit": 0})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == ErrorCode.INVALID_JOB_PAYLOAD
    assert payload["error"]["recoverable"] is True
    assert payload["error"]["details"] == {"error_count": 1}


def test_openapi_generates_health_and_error_schemas() -> None:
    client = TestClient(create_app())

    response = client.get("/api/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/jobs/analyze-dataset" in schema["paths"]
    assert "/api/v1/action-plans/execute-approved" in schema["paths"]
    assert "HealthResponse" in schema["components"]["schemas"]
    assert "AnalyzeDatasetAcceptedResponse" in schema["components"]["schemas"]
    assert "ActionPlanPreviewResponse" in schema["components"]["schemas"]
    assert "ActionPlanExecuteApprovedResponse" in schema["components"]["schemas"]
    assert "ErrorResponse" in schema["components"]["schemas"]


def test_analyze_dataset_endpoint_accepts_signed_request_and_materializes_base_assets() -> None:
    config = _test_config()
    app = create_app(config=config)
    payload = _analyze_payload()
    body = _body_bytes(payload)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/jobs/analyze-dataset",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == ComputeRunStatus.ACCEPTED
    assert data["job_id"] == payload["platform_job_id"]
    assert data["status_url"] == f"/api/v1/jobs/{payload['platform_job_id']}/status"
    assert data["expected_outputs"] == list(expected_analyze_outputs(include_predictions=False))
    assert set(data["materialized_assets"]) == set(data["expected_outputs"])
    assert data["mutates_dataset"] is False

    snapshot = app.state.fake_platform_client.snapshot()
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED
    assert all(event.details.get("dataset_id") == "dataset_1" for event in snapshot.job_events)
    assert "candidate_dataset" not in response.text


def test_analyze_dataset_endpoint_materializes_prediction_outputs_when_refs_provided() -> None:
    config = _test_config()
    app = create_app(config=config)
    payload = _analyze_payload(
        prediction_artifact_refs=[_artifact_ref("prediction_manifest_1", "prediction_manifest")]
    )
    body = _body_bytes(payload)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/jobs/analyze-dataset",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 202
    data = response.json()
    expected = expected_analyze_outputs(include_predictions=True)
    assert data["expected_outputs"] == list(expected)
    assert set(data["materialized_assets"]) == set(expected)
    assert {
        "prediction_manifest",
        "prediction_validation_report",
        "model_error_analysis_report",
        "ambiguous_object_candidates",
        "probable_label_error_candidates",
    }.issubset(data["materialized_assets"])
    assert data["mutates_dataset"] is False
    snapshot = app.state.fake_platform_client.snapshot()
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED


def test_action_plan_preview_endpoint_builds_steps_and_validation_gates() -> None:
    config = _test_config()
    payload = _action_plan_preview_payload()
    body = _body_bytes(payload)
    client = TestClient(create_app(config=config), raise_server_exceptions=False)

    response = client.post(
        "/api/v1/action-plans/preview",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PREVIEW_READY"
    assert data["job_id"] == "platform_job_001"
    assert data["mutates_dataset"] is False

    action_plan = data["action_plan"]
    validate_contract_payload(load_contract_pack(), "action_plan", action_plan)
    assert action_plan["created_from_decision_report"] == "decision_report_001"
    assert action_plan["execution_mode"] == "PREVIEW_ACTION_PLAN"
    assert action_plan["selected_decision_ids"] == payload["selected_decision_ids"]
    assert action_plan["requires_approval"] is False

    steps = action_plan["steps"]
    assert [step["method_id"] for step in steps] == ["group_median", "class_weights"]
    assert steps[0]["depends_on"] == []
    assert steps[1]["depends_on"] == [steps[0]["step_id"]]
    for step in steps:
        assert step["idempotency_key"].startswith("sha256:")
        assert step["config_hash"].startswith("sha256:")
        assert step["plugin_id"] == "dataforge.tabular"
        assert step["plugin_version"] == "0.1.0"
    assert {"schema_validation", "business_rules", "privacy_check"}.issubset(
        steps[0]["validation_gates"]
    )
    assert "model_impact_check" in steps[1]["validation_gates"]


def test_execute_approved_rejects_unsigned_request() -> None:
    """TASK-040 step 1: execute-approved without platform signature is rejected."""
    config = _test_config()
    payload = _action_plan_execute_payload()
    body = _body_bytes(payload)
    client = TestClient(create_app(config=config), raise_server_exceptions=False)

    response = client.post(
        "/api/v1/action-plans/execute-approved",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert error["details"]["reason_code"] == "missing_service_identity"


def test_execute_approved_accepts_valid_approval_metadata() -> None:
    """TASK-040 step 2-3 + TASK-057: fake valid approval metadata yields accepted state.

    The endpoint must also launch the APPLY_SELECTED_ACTIONS Dagster
    materialization (TASK-057), expose status_url/expected_outputs and
    surface candidate / model impact / export artifact URIs so the
    platform UI can pin them to the platform job record.
    """
    config = _test_config()
    payload = _action_plan_execute_payload()
    body = _body_bytes(payload)
    client = TestClient(create_app(config=config), raise_server_exceptions=False)

    response = client.post(
        "/api/v1/action-plans/execute-approved",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 202
    data = response.json()
    action_plan = cast(dict[str, object], payload["action_plan"])
    approval_metadata = cast(dict[str, object], payload["approval_metadata"])
    steps = cast(list[dict[str, object]], action_plan["steps"])
    assert data["status"] == ComputeRunStatus.ACCEPTED
    assert data["job_id"] == "platform_job_001"
    assert data["workflow_type"] == WorkflowType.APPLY_SELECTED_ACTIONS
    assert data["action_plan_id"] == action_plan["action_plan_id"]
    assert data["action_plan_hash"] == approval_metadata["action_plan_hash"]
    assert data["accepted_step_ids"] == [step["step_id"] for step in steps]
    assert data["mutates_dataset"] is True

    # TASK-057: APPLY workflow materialization wiring.
    assert data["status_url"] == "/api/v1/jobs/platform_job_001/status"
    expected = list(data["expected_outputs"])
    materialized = list(data["materialized_assets"])
    assert sorted(expected) == sorted(
        [
            "action_plan",
            "remediation_execution_report",
            "prepared_dataset",
            "synthetic_dataset",
            "model_impact_report",
            "export_package",
        ]
    )
    assert sorted(materialized) == sorted(expected)
    # The current APPLY graph materializes placeholder audit/progress
    # artifacts only. They must not be exposed as final candidate,
    # model-impact, or export refs until real builders/gates replace them.
    assert data["candidate_artifact_uri"] is None
    assert data["candidate_artifact_hash"] is None
    assert data["model_impact_artifact_uri"] is None
    assert data["export_package_artifact_uri"] is None
    # The fixture uses an imputation-only ActionPlan so synthetic_status
    # must be not_applicable and no final synthetic artifact is surfaced.
    assert data["synthetic_status"] == "not_applicable"
    assert data["synthetic_artifact_uri"] is None


def test_execute_approved_rejects_approval_hash_mismatch() -> None:
    config = _test_config()
    payload = _action_plan_execute_payload()
    approval_metadata = dict(cast(dict[str, object], payload["approval_metadata"]))
    approval_metadata["action_plan_hash"] = "sha256:" + "e" * 64
    payload["approval_metadata"] = approval_metadata
    body = _body_bytes(payload)
    client = TestClient(create_app(config=config), raise_server_exceptions=False)

    response = client.post(
        "/api/v1/action-plans/execute-approved",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert error["details"]["reason_code"] == "approval_metadata_integrity_mismatch"
    assert "raw" not in response.text.lower()


def test_action_plan_preview_rejects_disabled_ctgan_override() -> None:
    config = _test_config()
    recommendations = _method_recommendations()
    rare_class = next(
        recommendation
        for recommendation in recommendations
        if recommendation["action_type"] == "AUGMENT_RARE_CLASS"
    )
    rare_class_recommendation_id = str(rare_class["recommendation_id"])
    payload = _action_plan_preview_payload(
        selected_decision_ids=[rare_class_recommendation_id],
        selected_method_overrides={rare_class_recommendation_id: "ctgan"},
        method_recommendations=recommendations,
    )
    body = _body_bytes(payload)
    client = TestClient(create_app(config=config), raise_server_exceptions=False)

    response = client.post(
        "/api/v1/action-plans/preview",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == ErrorCode.POLICY_BLOCKED
    assert error["details"]["reason_code"] == "method_not_selectable"
    assert error["details"]["method_id"] == "ctgan"
    assert "raw" not in response.text.lower()


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


def _analyze_payload(
    *,
    prediction_artifact_refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "platform_job_id": "platform_job_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "dataset_version_id": "dataset_version_1",
        "dataset_object_refs": [_artifact_ref("raw_archive_1", "raw_dataset_archive")],
        "prediction_artifact_refs": []
        if prediction_artifact_refs is None
        else prediction_artifact_refs,
    }


def _action_plan_execute_payload() -> dict[str, object]:
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_001",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_123",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
            ),
            target_version_name="dataset_version_2_preview",
            created_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
        )
    ).model_copy(
        update={"requires_approval": True, "approval_request_id": "approval_request_001"}
    )
    action_plan = plan.model_dump(mode="json")
    return {
        "platform_job_id": "platform_job_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "source_dataset_version_id": "dataset_version_1",
        "action_plan": action_plan,
        "approval_metadata": {
            "approval_id": "approval_001",
            "approval_request_id": "approval_request_001",
            "approved_by_user_id": "platform_owner_001",
            "approved_at": "2026-05-24T12:05:00Z",
            "action_plan_id": action_plan["action_plan_id"],
            "action_plan_hash": action_plan_integrity_hash(plan),
            "decision_report_id": action_plan["created_from_decision_report"],
            "source_dataset_version_id": action_plan["source_dataset_version_id"],
        },
    }


def _action_plan_preview_payload(
    *,
    selected_decision_ids: list[str] | None = None,
    selected_method_overrides: dict[str, str] | None = None,
    method_recommendations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    recommendations = (
        _method_recommendations()
        if method_recommendations is None
        else method_recommendations
    )
    selected = (
        [str(recommendation["recommendation_id"]) for recommendation in recommendations]
        if selected_decision_ids is None
        else selected_decision_ids
    )
    return {
        "platform_job_id": "platform_job_001",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "source_dataset_version_id": "dataset_version_1",
        "decision_report_id": "decision_report_001",
        "selected_decision_ids": selected,
        "selected_method_overrides": {}
        if selected_method_overrides is None
        else selected_method_overrides,
        "method_recommendations": recommendations,
        "created_by_user_id": "platform_user_123",
        "input_artifacts": [
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl"
        ],
        "target_version_name": "dataset_version_2_preview",
    }


def _method_recommendations() -> list[dict[str, object]]:
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )
    return [recommendation.model_dump(mode="json") for recommendation in recommendations]


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
            job_id="platform_job_001",
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
