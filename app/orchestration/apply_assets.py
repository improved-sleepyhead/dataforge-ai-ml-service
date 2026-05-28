"""APPLY_SELECTED_ACTIONS assets for the DataForge AI compute plane.

These assets run only after the platform backend has supplied an approved
ActionPlan. They execute existing plugin/kernel builders and register immutable
candidate, validation, model-impact, lineage and export artifacts. Raw source
artifacts are read from object storage and are never overwritten.

This module intentionally does not use ``from __future__ import annotations``
because Dagster validates the ``context`` parameter type via runtime
annotations on decorated asset functions.
"""

from typing import Any

from dagster import AssetExecutionContext, AssetKey, MaterializeResult, asset

from app.adapters import ArtifactRegistry, FakePlatformMetadataClient, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import ArtifactRef, WorkflowType
from app.orchestration.apply_runtime import (
    ACTION_PLAN_ARTIFACT_KIND,
    ACTION_PLAN_ARTIFACT_SCHEMA_VERSION,
    EXPORT_MANIFEST_KIND,
    REMEDIATION_EXECUTION_REPORT_KIND,
    build_final_apply_outputs,
    ensure_model_impact_report,
    execute_remediation_plan,
    final_candidate_ref,
    final_export_ref,
    persist_action_plan_artifact,
    synthetic_status,
)
from app.orchestration.job_event import JobStage
from app.orchestration.run_context import ApplyRunContext, RunContextResource
from app.orchestration.status_bridge import RunContext, RunStatusBridge
from app.telemetry import (
    METRIC_EXPORT_BLOCKED_COUNT,
    METRIC_JOB_DURATION_MS,
    MetricsRegistry,
    TracingRegistry,
)

APPLY_GROUP = "apply_selected_actions"

APPLY_ASSET_KEYS: tuple[AssetKey, ...] = (
    AssetKey("action_plan"),
    AssetKey("remediation_execution_report"),
    AssetKey("prepared_dataset"),
    AssetKey("synthetic_dataset"),
    AssetKey("model_impact_report"),
    AssetKey("export_package"),
)

APPLY_ARTIFACT_KINDS: dict[str, str] = {
    "action_plan": ACTION_PLAN_ARTIFACT_KIND,
    "remediation_execution_report": REMEDIATION_EXECUTION_REPORT_KIND,
    "prepared_dataset": "candidate_tabular_dataset",
    "synthetic_dataset": "synthetic_dataset_report",
    "model_impact_report": "model_impact_report",
    "export_package": "export_package",
}

APPLY_ARTIFACT_SCHEMA_VERSIONS: dict[str, str] = {
    "action_plan": ACTION_PLAN_ARTIFACT_SCHEMA_VERSION,
    "remediation_execution_report": "remediation_execution_report.v1",
    "prepared_dataset": "tabular_dataset.v1",
    "synthetic_dataset": "synthetic_dataset_report.v1",
    "model_impact_report": "model_impact_report.v1",
    "export_package": "export_package.v1",
}

APPLY_ARTIFACT_FORMAT = "json"
APPLY_ARTIFACT_MEDIA_TYPE = "application/json"

_APPLY_STAGE_BY_ASSET: dict[str, tuple[JobStage, float]] = {
    "action_plan": (JobStage.RUNNING_DECISION_CORE, 0.30),
    "remediation_execution_report": (JobStage.RUNNING_DECISION_CORE, 0.50),
    "prepared_dataset": (JobStage.RUNNING_DECISION_CORE, 0.65),
    "synthetic_dataset": (JobStage.RUNNING_DECISION_CORE, 0.75),
    "model_impact_report": (JobStage.RUNNING_DECISION_CORE, 0.90),
    "export_package": (JobStage.COMPLETED, 1.0),
}

_APPLY_RESOURCE_KEYS = {
    "run_context",
    "fake_platform",
    "artifact_registry",
    "object_storage",
    "metrics",
    "tracing",
}


def _require_apply(
    run_context: RunContext,
    workflow_type: WorkflowType,
    apply_context: ApplyRunContext | None,
) -> ApplyRunContext:
    if workflow_type is not WorkflowType.APPLY_SELECTED_ACTIONS:
        raise ValueError(
            "APPLY assets cannot run under workflow_type "
            f"{workflow_type.value!r}; expected APPLY_SELECTED_ACTIONS"
        )
    if apply_context is None:
        raise ValueError("APPLY assets require an ApplyRunContext with action_plan_id")
    if not apply_context.action_plan_id or not apply_context.decision_report_id:
        raise ValueError("ApplyRunContext must include action_plan_id and decision_report_id")
    if not apply_context.source_dataset_version_id:
        raise ValueError("ApplyRunContext must include source_dataset_version_id")
    if not apply_context.proposed_version_name:
        raise ValueError("ApplyRunContext must include proposed_version_name")
    if not run_context.dataset_version_id:
        raise ValueError("RunContext must include dataset_version_id for apply runs")
    return apply_context


def _resources(
    context: AssetExecutionContext,
) -> tuple[
    RunContext,
    ApplyRunContext,
    FakePlatformMetadataClient,
    ArtifactRegistry,
    MinioObjectStorageAdapter,
]:
    run_context_resource: RunContextResource = context.resources.run_context
    run_context = run_context_resource.run_context
    apply_context = _require_apply(
        run_context,
        run_context_resource.workflow_type,
        run_context_resource.apply_context,
    )
    return (
        run_context,
        apply_context,
        context.resources.fake_platform,
        context.resources.artifact_registry,
        context.resources.object_storage,
    )


def _telemetry(context: AssetExecutionContext) -> tuple[MetricsRegistry, TracingRegistry]:
    return context.resources.metrics, context.resources.tracing


def _metadata(
    *,
    asset_name: str,
    run_context: RunContext,
    apply_context: ApplyRunContext,
    stage: JobStage,
    progress: float,
    status: str,
    artifact: RegisteredArtifact | None = None,
    artifact_refs: tuple[ArtifactRef, ...] = (),
    extras: dict[str, object] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "compute_run_id": run_context.compute_run_id,
        "platform_job_id": run_context.platform_job_id,
        "organization_id": run_context.organization_id,
        "project_id": run_context.project_id,
        "dataset_id": run_context.dataset_id,
        "dataset_version_id": run_context.dataset_version_id,
        "workflow_type": WorkflowType.APPLY_SELECTED_ACTIONS.value,
        "action_plan_id": apply_context.action_plan_id,
        "decision_report_id": apply_context.decision_report_id,
        "source_dataset_version_id": apply_context.source_dataset_version_id,
        "proposed_version_name": apply_context.proposed_version_name,
        "stage": stage.value,
        "progress": progress,
        "asset_name": asset_name,
        "asset_status": status,
        "artifact_kind": APPLY_ARTIFACT_KINDS[asset_name],
        "schema_version": APPLY_ARTIFACT_SCHEMA_VERSIONS[asset_name],
        "artifact_uris": [ref.uri for ref in artifact_refs],
    }
    if artifact is not None:
        payload["artifact_uri"] = artifact.uri
        payload["artifact_hash"] = artifact.hash
    if extras:
        payload.update(extras)
    return payload


def _emit_and_result(
    *,
    asset_name: str,
    run_context: RunContext,
    apply_context: ApplyRunContext,
    fake_platform: FakePlatformMetadataClient,
    status: str,
    artifact: RegisteredArtifact | None = None,
    artifact_refs: tuple[ArtifactRef, ...] = (),
    extras: dict[str, object] | None = None,
) -> MaterializeResult[None]:
    stage, progress = _APPLY_STAGE_BY_ASSET[asset_name]
    bridge = RunStatusBridge(fake_platform=fake_platform)
    if stage is JobStage.COMPLETED:
        bridge.emit_completed(run_context=run_context, artifact_refs=artifact_refs)
    else:
        bridge.emit_stage(
            run_context=run_context,
            stage=stage,
            progress=progress,
            artifact_refs=artifact_refs,
        )
    return MaterializeResult(
        metadata=_metadata(
            asset_name=asset_name,
            run_context=run_context,
            apply_context=apply_context,
            stage=stage,
            progress=progress,
            status=status,
            artifact=artifact,
            artifact_refs=artifact_refs,
            extras=extras,
        )
    )


@asset(
    name="action_plan",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    description="Approved ActionPlan loaded from the platform and pinned to this run.",
)
def action_plan(context: AssetExecutionContext) -> MaterializeResult[None]:
    run_context, apply_context, fake_platform, registry, _storage = _resources(context)
    artifact = persist_action_plan_artifact(
        apply_context=apply_context,
        run_context=run_context,
        registry=registry,
    )
    return _emit_and_result(
        asset_name="action_plan",
        run_context=run_context,
        apply_context=apply_context,
        fake_platform=fake_platform,
        status="registered",
        artifact=artifact,
        artifact_refs=(artifact.artifact_ref,),
        extras={
            "approved_step_count": len(apply_context.action_plan.steps)
            if apply_context.action_plan is not None
            else 0,
        },
    )


@asset(
    name="remediation_execution_report",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("action_plan")],
    description="Execution report for approved ActionPlan remediation steps.",
)
def remediation_execution_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    run_context, apply_context, fake_platform, registry, storage = _resources(context)
    artifact = execute_remediation_plan(
        apply_context=apply_context,
        run_context=run_context,
        storage=storage,
        registry=registry,
    )
    state = apply_context.execution_state
    refs = tuple(
        artifact.artifact_ref
        for artifact in (
            state.remediation_report_artifact,
            state.validation_gates_report_artifact,
        )
        if artifact is not None
    )
    return _emit_and_result(
        asset_name="remediation_execution_report",
        run_context=run_context,
        apply_context=apply_context,
        fake_platform=fake_platform,
        status="executed",
        artifact=artifact,
        artifact_refs=refs,
        extras={
            "validation_status": state.validation_gates_report.candidate_status.value
            if state.validation_gates_report is not None
            else "not_available",
            "block_export": state.validation_gates_report.block_export
            if state.validation_gates_report is not None
            else True,
        },
    )


@asset(
    name="prepared_dataset",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("remediation_execution_report")],
    description="Prepared candidate dataset artifact produced by approved steps.",
)
def prepared_dataset(context: AssetExecutionContext) -> MaterializeResult[None]:
    run_context, apply_context, fake_platform, registry, storage = _resources(context)
    execute_remediation_plan(
        apply_context=apply_context,
        run_context=run_context,
        storage=storage,
        registry=registry,
    )
    artifact = apply_context.execution_state.primary_candidate_artifact
    refs = (artifact.artifact_ref,) if artifact is not None else ()
    return _emit_and_result(
        asset_name="prepared_dataset",
        run_context=run_context,
        apply_context=apply_context,
        fake_platform=fake_platform,
        status="materialized" if artifact is not None else "missing",
        artifact=artifact,
        artifact_refs=refs,
    )


@asset(
    name="synthetic_dataset",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("prepared_dataset")],
    description="Synthetic dataset report and candidate rows when a synthetic step is selected.",
)
def synthetic_dataset(context: AssetExecutionContext) -> MaterializeResult[None]:
    run_context, apply_context, fake_platform, registry, storage = _resources(context)
    execute_remediation_plan(
        apply_context=apply_context,
        run_context=run_context,
        storage=storage,
        registry=registry,
    )
    state = apply_context.execution_state
    artifact = state.synthetic_dataset_report_artifact
    refs = (artifact.artifact_ref,) if artifact is not None else ()
    return _emit_and_result(
        asset_name="synthetic_dataset",
        run_context=run_context,
        apply_context=apply_context,
        fake_platform=fake_platform,
        status=synthetic_status(state, apply_context),
        artifact=artifact,
        artifact_refs=refs,
    )


@asset(
    name="model_impact_report",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("prepared_dataset"), AssetKey("synthetic_dataset")],
    description="Baseline vs candidate model impact report when eligible.",
)
def model_impact_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    run_context, apply_context, fake_platform, registry, storage = _resources(context)
    metrics, tracing = _telemetry(context)
    with tracing.span(
        "model_impact.evaluate",
        attributes={"stage": "model_impact.evaluate", "job_type": "APPLY_SELECTED_ACTIONS"},
    ):
        artifact = ensure_model_impact_report(
            apply_context=apply_context,
            run_context=run_context,
            storage=storage,
            registry=registry,
        )
    metrics.observe(
        METRIC_JOB_DURATION_MS,
        value=0.0,
        labels={"stage": "model_impact.evaluate", "job_type": "APPLY_SELECTED_ACTIONS"},
    )
    refs = (artifact.artifact_ref,) if artifact is not None else ()
    return _emit_and_result(
        asset_name="model_impact_report",
        run_context=run_context,
        apply_context=apply_context,
        fake_platform=fake_platform,
        status="materialized" if artifact is not None else "not_applicable",
        artifact=artifact,
        artifact_refs=refs,
    )


@asset(
    name="export_package",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("model_impact_report")],
    description="Gated ExportPackage built only from immutable candidate artifacts.",
)
def export_package(context: AssetExecutionContext) -> MaterializeResult[None]:
    run_context, apply_context, fake_platform, registry, storage = _resources(context)
    metrics, tracing = _telemetry(context)
    with tracing.span(
        "export.build",
        attributes={"stage": "export.build", "job_type": "APPLY_SELECTED_ACTIONS"},
    ):
        artifact = build_final_apply_outputs(
            apply_context=apply_context,
            run_context=run_context,
            storage=storage,
            registry=registry,
            platform_client=fake_platform,
        )
    state = apply_context.execution_state
    final_candidate = final_candidate_ref(state)
    final_export = final_export_ref(state)
    refs = tuple(
        ref
        for ref in (
            final_candidate.artifact_ref if final_candidate is not None else None,
            final_export.artifact_ref if final_export is not None else None,
            state.validation_gates_report_artifact.artifact_ref
            if state.validation_gates_report_artifact is not None
            else None,
        )
        if ref is not None
    )
    candidate = state.candidate_dataset_version
    package = state.export_package
    if package is not None and package.blocked_reason_codes:
        metrics.increment(
            METRIC_EXPORT_BLOCKED_COUNT,
            labels={"stage": "export.build", "reason_code": "blocked"},
        )
    metrics.observe(
        METRIC_JOB_DURATION_MS,
        value=0.0,
        labels={"stage": "export.build", "job_type": "APPLY_SELECTED_ACTIONS"},
    )
    return _emit_and_result(
        asset_name="export_package",
        run_context=run_context,
        apply_context=apply_context,
        fake_platform=fake_platform,
        status=package.status.value if package is not None else "missing",
        artifact=artifact,
        artifact_refs=refs,
        extras={
            "candidate_status": candidate.status.value if candidate is not None else None,
            "candidate_artifact_uri": (
                state.candidate_version_artifact.uri
                if state.candidate_version_artifact is not None
                else None
            ),
            "candidate_artifact_hash": (
                state.candidate_version_artifact.hash
                if state.candidate_version_artifact is not None
                else None
            ),
            "final_candidate_artifact_uri": (
                final_candidate.uri if final_candidate is not None else None
            ),
            "final_candidate_artifact_hash": (
                final_candidate.hash if final_candidate is not None else None
            ),
            "final_export_package_uri": (
                final_export.uri if final_export is not None else None
            ),
            "final_export_package_hash": (
                final_export.hash if final_export is not None else None
            ),
            "export_package_status": package.status.value if package is not None else None,
        },
    )


APPLY_ASSETS = (
    action_plan,
    remediation_execution_report,
    prepared_dataset,
    synthetic_dataset,
    model_impact_report,
    export_package,
)


def asset_kind_for(asset_name: str) -> str:
    """Return the registered artifact kind for an APPLY asset name."""
    return APPLY_ARTIFACT_KINDS[asset_name]


def schema_version_for(asset_name: str) -> str:
    """Return the schema_version emitted by an APPLY asset."""
    return APPLY_ARTIFACT_SCHEMA_VERSIONS[asset_name]


__all__ = [
    "APPLY_ARTIFACT_FORMAT",
    "APPLY_ARTIFACT_KINDS",
    "APPLY_ARTIFACT_MEDIA_TYPE",
    "APPLY_ARTIFACT_SCHEMA_VERSIONS",
    "APPLY_ASSETS",
    "APPLY_ASSET_KEYS",
    "APPLY_GROUP",
    "EXPORT_MANIFEST_KIND",
    "action_plan",
    "asset_kind_for",
    "export_package",
    "model_impact_report",
    "prepared_dataset",
    "remediation_execution_report",
    "schema_version_for",
    "synthetic_dataset",
]
