"""Skeleton APPLY_SELECTED_ACTIONS assets for the DataForge AI compute plane.

Apply assets only run after the platform backend has produced an approved
ActionPlan. They are the only path that may produce dataset-changing
artifacts (prepared/redacted/synthetic/candidate datasets, export packages).

This skeleton:

* refuses to materialize unless ``workflow_type == APPLY_SELECTED_ACTIONS``
  and an ``apply_context`` is present in the run context resource;
* produces only safe metadata, never raw mutation;
* emits stage events through the status bridge to the fake platform client.

Real implementations will write candidate datasets via
``ArtifactRegistry.save_artifact`` with parent_version_id lineage, plug into
validation gates and synthetic privacy checks, and never overwrite raw
artifacts.

This module intentionally does not use ``from __future__ import annotations``
because Dagster validates the ``context`` parameter type via runtime
annotations on the decorated asset functions.
"""

from typing import Any

from dagster import AssetExecutionContext, AssetKey, MaterializeResult, asset

from app.adapters import FakePlatformMetadataClient
from app.domain import WorkflowType
from app.orchestration.job_event import JobStage
from app.orchestration.run_context import ApplyRunContext, RunContextResource
from app.orchestration.status_bridge import RunContext, RunStatusBridge

APPLY_GROUP = "apply_selected_actions"

APPLY_ASSET_KEYS: tuple[AssetKey, ...] = (
    AssetKey("action_plan"),
    AssetKey("remediation_execution_report"),
    AssetKey("prepared_dataset"),
    AssetKey("synthetic_dataset"),
    AssetKey("model_impact_report"),
    AssetKey("export_package"),
)

# Apply stages all run after the analyze pipeline has completed; the bridge
# uses RUNNING_DECISION_CORE for the apply preamble and COMPLETED only at the
# end of the export stage. Progress monotonically increases.
_APPLY_STAGE_BY_ASSET: dict[str, tuple[JobStage, float]] = {
    "action_plan": (JobStage.RUNNING_DECISION_CORE, 0.30),
    "remediation_execution_report": (JobStage.RUNNING_DECISION_CORE, 0.50),
    "prepared_dataset": (JobStage.RUNNING_DECISION_CORE, 0.65),
    "synthetic_dataset": (JobStage.RUNNING_DECISION_CORE, 0.75),
    "model_impact_report": (JobStage.RUNNING_DECISION_CORE, 0.90),
    "export_package": (JobStage.COMPLETED, 1.0),
}

_APPLY_RESOURCE_KEYS = {"run_context", "fake_platform"}


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
    if not run_context.dataset_version_id:
        raise ValueError("RunContext must include dataset_version_id for apply runs")
    return apply_context


def _apply_metadata(
    *,
    run_context: RunContext,
    apply_context: ApplyRunContext,
    stage: JobStage,
    progress: float,
) -> dict[str, Any]:
    return {
        "compute_run_id": run_context.compute_run_id,
        "platform_job_id": run_context.platform_job_id,
        "organization_id": run_context.organization_id,
        "project_id": run_context.project_id,
        "dataset_id": run_context.dataset_id,
        "dataset_version_id": run_context.dataset_version_id,
        "workflow_type": WorkflowType.APPLY_SELECTED_ACTIONS.value,
        "action_plan_id": apply_context.action_plan_id,
        "decision_report_id": apply_context.decision_report_id,
        "stage": stage.value,
        "progress": progress,
        "skeleton": True,
        # Skeleton emits no candidate artifact; flag stays False until a real
        # ArtifactRegistry write happens in the implementation phase.
        "produced_candidate_artifact": False,
    }


def _materialize_apply_skeleton(
    context: AssetExecutionContext,
    *,
    asset_name: str,
) -> MaterializeResult[None]:
    run_context_resource: RunContextResource = context.resources.run_context
    fake_platform: FakePlatformMetadataClient = context.resources.fake_platform

    run_context = run_context_resource.run_context
    workflow_type = run_context_resource.workflow_type
    resolved_apply = _require_apply(
        run_context, workflow_type, run_context_resource.apply_context
    )

    stage, progress = _APPLY_STAGE_BY_ASSET[asset_name]
    bridge = RunStatusBridge(fake_platform=fake_platform)
    if stage is JobStage.COMPLETED:
        bridge.emit_completed(run_context=run_context)
    else:
        bridge.emit_stage(run_context=run_context, stage=stage, progress=progress)

    return MaterializeResult(
        metadata=_apply_metadata(
            run_context=run_context,
            apply_context=resolved_apply,
            stage=stage,
            progress=progress,
        )
    )


@asset(
    name="action_plan",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    description="Skeleton: approved ActionPlan loaded from the platform.",
)
def action_plan(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_skeleton(context, asset_name="action_plan")


@asset(
    name="remediation_execution_report",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("action_plan")],
    description="Skeleton: report produced by ActionPlan step execution.",
)
def remediation_execution_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_skeleton(context, asset_name="remediation_execution_report")


@asset(
    name="prepared_dataset",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("remediation_execution_report")],
    description="Skeleton: prepared candidate dataset artifact reference (placeholder).",
)
def prepared_dataset(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_skeleton(context, asset_name="prepared_dataset")


@asset(
    name="synthetic_dataset",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("prepared_dataset")],
    description="Skeleton: synthetic candidate dataset artifact reference (placeholder).",
)
def synthetic_dataset(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_skeleton(context, asset_name="synthetic_dataset")


@asset(
    name="model_impact_report",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("prepared_dataset"), AssetKey("synthetic_dataset")],
    description="Skeleton: baseline vs candidate model impact report (placeholder).",
)
def model_impact_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_skeleton(context, asset_name="model_impact_report")


@asset(
    name="export_package",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("model_impact_report")],
    description="Skeleton: export package after readiness gates (placeholder).",
)
def export_package(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_skeleton(context, asset_name="export_package")


APPLY_ASSETS = (
    action_plan,
    remediation_execution_report,
    prepared_dataset,
    synthetic_dataset,
    model_impact_report,
    export_package,
)


__all__ = [
    "APPLY_ASSETS",
    "APPLY_ASSET_KEYS",
    "APPLY_GROUP",
    "action_plan",
    "export_package",
    "model_impact_report",
    "prepared_dataset",
    "remediation_execution_report",
    "synthetic_dataset",
]
