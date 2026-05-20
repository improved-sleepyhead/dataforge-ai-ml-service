"""Skeleton ANALYZE_ONLY assets for the DataForge AI compute plane.

These assets are intentionally skeleton. They do not run real ML/data
algorithms yet; their purpose is to wire up the Dagster runtime, expose the
asset graph mandated by the orchestration steering rules, and prove that:

* Dagster definitions load without errors;
* a skeleton analyze job materializes against in-memory adapters;
* the fake platform client receives stage events through the status bridge.

Strict constraints honored here:

* ``ANALYZE_ONLY`` must not mutate source dataset artifacts. None of these
  assets write to source paths or call ``put`` on object storage in this
  skeleton.
* Materialization metadata only contains stable technical fields. No raw
  PII, raw text, file contents, or secrets.
* Decision Core consumes normalized evidence in real implementations; here
  we just propagate placeholders so downstream assets can be wired up.

This module intentionally does not use ``from __future__ import annotations``
because Dagster validates the ``context`` parameter type via runtime
annotations on the decorated asset functions.
"""

from typing import Any

from dagster import AssetExecutionContext, AssetKey, MaterializeResult, asset

from app.adapters import FakePlatformMetadataClient
from app.domain import WorkflowType
from app.orchestration.job_event import JobStage
from app.orchestration.run_context import RunContextResource
from app.orchestration.status_bridge import RunContext, RunStatusBridge

ANALYZE_GROUP = "analyze_only"

ANALYZE_ASSET_KEYS: tuple[AssetKey, ...] = (
    AssetKey("raw_manifest"),
    AssetKey("validated_manifest"),
    AssetKey("tabular_profile_report"),
    AssetKey("object_analytics_passports"),
    AssetKey("evidence_bundle"),
    AssetKey("decision_report"),
    AssetKey("recommended_actions"),
    AssetKey("review_queue"),
)

# Map asset name -> (JobStage, normalized progress at the end of the stage).
# Progress moves through canonical platform stages so the UI can render a
# stable timeline. Stages outside the canonical lifecycle (recommended_actions,
# review_queue) reuse RUNNING_DECISION_CORE because they are produced from
# the Decision Core output.
_ANALYZE_STAGE_BY_ASSET: dict[str, tuple[JobStage, float]] = {
    "raw_manifest": (JobStage.INGESTING, 0.10),
    "validated_manifest": (JobStage.BUILDING_MANIFEST, 0.25),
    "tabular_profile_report": (JobStage.PROFILING_TABULAR, 0.45),
    "object_analytics_passports": (JobStage.PROFILING_TABULAR, 0.55),
    "evidence_bundle": (JobStage.BUILDING_EVIDENCE, 0.70),
    "decision_report": (JobStage.RUNNING_DECISION_CORE, 0.80),
    "recommended_actions": (JobStage.RUNNING_DECISION_CORE, 0.90),
    "review_queue": (JobStage.RUNNING_DECISION_CORE, 0.95),
}

_ANALYZE_RESOURCE_KEYS = {"run_context", "fake_platform"}


def _require_analyze_only(run_context: RunContext, workflow_type: WorkflowType) -> None:
    if workflow_type is not WorkflowType.ANALYZE_ONLY:
        raise ValueError(
            "ANALYZE_ONLY assets cannot run under workflow_type "
            f"{workflow_type.value!r}; this prevents accidental dataset mutation"
        )
    if not run_context.dataset_version_id:
        raise ValueError("RunContext must include dataset_version_id for analyze runs")


def _materialization_metadata(
    *,
    run_context: RunContext,
    workflow_type: WorkflowType,
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
        "workflow_type": workflow_type.value,
        "mutates_dataset": False,
        "stage": stage.value,
        "progress": progress,
        "skeleton": True,
    }


def _materialize_skeleton(
    context: AssetExecutionContext,
    *,
    asset_name: str,
) -> MaterializeResult[None]:
    run_context_resource: RunContextResource = context.resources.run_context
    fake_platform: FakePlatformMetadataClient = context.resources.fake_platform

    run_context = run_context_resource.run_context
    workflow_type = run_context_resource.workflow_type
    _require_analyze_only(run_context, workflow_type)

    stage, progress = _ANALYZE_STAGE_BY_ASSET[asset_name]

    bridge = RunStatusBridge(fake_platform=fake_platform)
    bridge.emit_stage(run_context=run_context, stage=stage, progress=progress)

    metadata = _materialization_metadata(
        run_context=run_context,
        workflow_type=workflow_type,
        stage=stage,
        progress=progress,
    )
    return MaterializeResult(metadata=metadata)


@asset(
    name="raw_manifest",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    description="Skeleton: raw asset manifest assembled from immutable raw archive refs.",
)
def raw_manifest(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="raw_manifest")


@asset(
    name="validated_manifest",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("raw_manifest")],
    description="Skeleton: validated manifest after schema/contract checks.",
)
def validated_manifest(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="validated_manifest")


@asset(
    name="tabular_profile_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("validated_manifest")],
    description="Skeleton: tabular profile/EDA report (placeholder for tabular plugin).",
)
def tabular_profile_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="tabular_profile_report")


@asset(
    name="object_analytics_passports",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("tabular_profile_report")],
    description="Skeleton: per-object analytical passports (placeholder).",
)
def object_analytics_passports(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="object_analytics_passports")


@asset(
    name="evidence_bundle",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("object_analytics_passports")],
    description="Skeleton: normalized EvidenceBundle for Decision Core (placeholder).",
)
def evidence_bundle(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="evidence_bundle")


@asset(
    name="decision_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("evidence_bundle")],
    description="Skeleton: Decision Core dataset-level report (placeholder).",
)
def decision_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="decision_report")


@asset(
    name="recommended_actions",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("decision_report")],
    description="Skeleton: recommended actions surfaced to the platform UI.",
)
def recommended_actions(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="recommended_actions")


@asset(
    name="review_queue",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("decision_report")],
    description="Skeleton: review queue for label/privacy/duplicate review.",
)
def review_queue(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, asset_name="review_queue")


ANALYZE_ASSETS = (
    raw_manifest,
    validated_manifest,
    tabular_profile_report,
    object_analytics_passports,
    evidence_bundle,
    decision_report,
    recommended_actions,
    review_queue,
)


__all__ = [
    "ANALYZE_ASSET_KEYS",
    "ANALYZE_ASSETS",
    "ANALYZE_GROUP",
    "decision_report",
    "evidence_bundle",
    "object_analytics_passports",
    "raw_manifest",
    "recommended_actions",
    "review_queue",
    "tabular_profile_report",
    "validated_manifest",
]
