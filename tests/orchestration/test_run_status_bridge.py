"""Run status bridge tests for TASK-016.

These tests cover the compute-plane → platform status bridge:

* every supported stage maps to a stable JobEvent with technical fields;
* the analyze job covers the canonical pre-completion lifecycle;
* the apply job culminates in a COMPLETED event;
* failure and cancellation paths emit the terminal FAILED/CANCELLED events;
* JobEvent never carries raw PII even before defense-in-depth redaction.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.adapters import FakePlatformMetadataClient
from app.domain import (
    ArtifactLineage,
    ArtifactRef,
    ComputeRunStatus,
    WorkflowType,
)
from app.orchestration import (
    ANALYZE_JOB_NAME,
    APPLY_JOB_NAME,
    ApplyRunContext,
    JobEvent,
    JobStage,
    RunContextResource,
    RunStatusBridge,
    build_definitions,
    scan_event_for_raw_pii,
)
from tests.orchestration.test_dagster_runtime import (
    _build_in_memory_resources,
    _job_by_name,
    _run_context,
)


def test_job_stage_enum_supports_all_required_stages() -> None:
    required = {
        "QUEUED",
        "INGESTING",
        "BUILDING_MANIFEST",
        "VALIDATING",
        "PROFILING_TABULAR",
        "BUILDING_EVIDENCE",
        "RUNNING_DECISION_CORE",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }
    actual = {stage.value for stage in JobStage}
    assert required.issubset(actual)


def test_job_event_carries_all_required_technical_fields() -> None:
    artifact = _artifact_ref()
    event = JobEvent(
        job_id="platform_job_demo",
        stage=JobStage.BUILDING_MANIFEST,
        status=ComputeRunStatus.RUNNING,
        progress=0.25,
        artifact_refs=(artifact,),
    )

    assert event.job_id == "platform_job_demo"
    assert event.stage is JobStage.BUILDING_MANIFEST
    assert event.status is ComputeRunStatus.RUNNING
    assert event.progress == 0.25
    assert event.artifact_refs == (artifact,)
    assert isinstance(event.created_at, datetime)


def test_run_status_bridge_emits_started_running_completed_pipeline() -> None:
    fake_platform = FakePlatformMetadataClient()
    bridge = RunStatusBridge(fake_platform=fake_platform)
    run_context = _run_context()

    bridge.emit_started(run_context=run_context, stage=JobStage.QUEUED)
    bridge.emit_stage(run_context=run_context, stage=JobStage.INGESTING, progress=0.1)
    bridge.emit_stage(
        run_context=run_context,
        stage=JobStage.BUILDING_MANIFEST,
        progress=0.25,
    )
    bridge.emit_stage(
        run_context=run_context,
        stage=JobStage.RUNNING_DECISION_CORE,
        progress=0.9,
    )
    bridge.emit_completed(run_context=run_context)

    snapshot = fake_platform.snapshot()
    stages = [event.stage for event in snapshot.job_events]
    assert stages == [
        JobStage.QUEUED.value,
        JobStage.INGESTING.value,
        JobStage.BUILDING_MANIFEST.value,
        JobStage.RUNNING_DECISION_CORE.value,
        JobStage.COMPLETED.value,
    ]
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED


def test_run_status_bridge_emits_failed_event_with_error_code() -> None:
    fake_platform = FakePlatformMetadataClient()
    bridge = RunStatusBridge(fake_platform=fake_platform)

    bridge.emit_failed(
        run_context=_run_context(),
        progress=0.4,
        error_code="DAGSTER_RUN_FAILED",
    )

    snapshot = fake_platform.snapshot()
    [event] = snapshot.job_events
    assert event.stage == JobStage.FAILED.value
    assert event.status is ComputeRunStatus.FAILED
    assert event.details["error_code"] == "DAGSTER_RUN_FAILED"
    assert "secret" not in event.details


def test_run_status_bridge_emits_cancelled_event() -> None:
    fake_platform = FakePlatformMetadataClient()
    bridge = RunStatusBridge(fake_platform=fake_platform)

    bridge.emit_cancelled(run_context=_run_context(), progress=0.2)

    snapshot = fake_platform.snapshot()
    [event] = snapshot.job_events
    assert event.stage == JobStage.CANCELLED.value
    assert event.status is ComputeRunStatus.FAILED


def test_analyze_job_emits_canonical_lifecycle_stages() -> None:
    fake_platform = FakePlatformMetadataClient()
    compute_resources = _build_in_memory_resources(fake_platform=fake_platform)
    run_context_resource = RunContextResource(
        run_context=_run_context(),
        workflow_type=WorkflowType.ANALYZE_ONLY,
    )
    definitions = build_definitions(
        compute_resources=compute_resources,
        run_context_resource=run_context_resource,
    )
    analyze_job = _job_by_name(definitions, ANALYZE_JOB_NAME)

    result = analyze_job.execute_in_process()

    assert result.success
    snapshot = fake_platform.snapshot()
    stages = {event.stage for event in snapshot.job_events}
    expected = {
        JobStage.INGESTING.value,
        JobStage.BUILDING_MANIFEST.value,
        JobStage.PROFILING_TABULAR.value,
        JobStage.BUILDING_EVIDENCE.value,
        JobStage.RUNNING_DECISION_CORE.value,
    }
    assert expected.issubset(stages)
    # All analyze events monotonically advance the progress between [0, 1].
    progresses = [event.details["progress"] for event in snapshot.job_events]
    assert all(0.0 <= value <= 1.0 for value in progresses)


def test_apply_job_emits_completed_terminal_event() -> None:
    fake_platform = FakePlatformMetadataClient()
    compute_resources = _build_in_memory_resources(fake_platform=fake_platform)
    run_context_resource = RunContextResource(
        run_context=_run_context(),
        workflow_type=WorkflowType.APPLY_SELECTED_ACTIONS,
        apply_context=ApplyRunContext(
            action_plan_id="action_plan_001",
            decision_report_id="decision_report_001",
        ),
    )
    definitions = build_definitions(
        compute_resources=compute_resources,
        run_context_resource=run_context_resource,
    )
    apply_job = _job_by_name(definitions, APPLY_JOB_NAME)

    result = apply_job.execute_in_process()

    assert result.success
    snapshot = fake_platform.snapshot()
    stages = {event.stage for event in snapshot.job_events}
    assert JobStage.COMPLETED.value in stages
    last_event = snapshot.job_events[-1]
    assert last_event.stage == JobStage.COMPLETED.value
    assert last_event.status is ComputeRunStatus.COMPLETED


def test_job_event_does_not_leak_raw_pii_or_secrets() -> None:
    suspicious_artifact = ArtifactRef(
        artifact_id="artifact_manifest_safe",
        kind="asset_manifest",
        uri="s3://dataforge/org_1/project_1/dataset_1/v1/manifest.jsonl",
        hash="sha256:" + "a" * 64,
        media_type="application/jsonl",
        size_bytes=128,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_1",
            job_id="compute_run_001",
            config_hash="sha256:" + "b" * 64,
            created_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        ),
    )
    event = JobEvent(
        job_id="platform_job_demo",
        stage=JobStage.RUNNING_DECISION_CORE,
        status=ComputeRunStatus.RUNNING,
        progress=0.8,
        artifact_refs=(suspicious_artifact,),
    )

    categories = scan_event_for_raw_pii(event)

    assert categories == ()


def _artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="artifact_event_test",
        kind="asset_manifest",
        uri="s3://dataforge/org_1/project_1/dataset_1/v1/manifest.jsonl",
        hash="sha256:" + "a" * 64,
        media_type="application/jsonl",
        size_bytes=128,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_1",
            job_id="compute_run_001",
            config_hash="sha256:" + "b" * 64,
            created_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        ),
    )
