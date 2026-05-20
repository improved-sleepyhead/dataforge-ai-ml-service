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
from app.domain import ComputeRunStatus, WorkflowType
from app.orchestration.run_context import RunContextResource
from app.orchestration.status_bridge import RunContext, emit_stage_event

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

_ANALYZE_RESOURCE_KEYS = {"run_context", "fake_platform"}


def _require_analyze_only(run_context: RunContext, workflow_type: WorkflowType) -> None:
    if workflow_type is not WorkflowType.ANALYZE_ONLY:
        raise ValueError(
            "ANALYZE_ONLY assets cannot run under workflow_type "
            f"{workflow_type.value!r}; this prevents accidental dataset mutation"
        )
    # Defensive sanity: dataset/version identifiers must be present.
    if not run_context.dataset_version_id:
        raise ValueError("RunContext must include dataset_version_id for analyze runs")


def _materialization_metadata(
    *,
    run_context: RunContext,
    workflow_type: WorkflowType,
    stage: str,
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
        "stage": stage,
        "skeleton": True,
    }


def _emit_stage(
    *,
    fake_platform: FakePlatformMetadataClient,
    run_context: RunContext,
    stage: str,
) -> None:
    emit_stage_event(
        fake_platform=fake_platform,
        run_context=run_context,
        stage=stage,
        status=ComputeRunStatus.RUNNING,
    )


def _materialize_skeleton(
    context: AssetExecutionContext,
    *,
    stage: str,
    extra_metadata: dict[str, Any] | None = None,
) -> MaterializeResult[None]:
    run_context_resource: RunContextResource = context.resources.run_context
    fake_platform: FakePlatformMetadataClient = context.resources.fake_platform

    run_context = run_context_resource.run_context
    workflow_type = run_context_resource.workflow_type
    _require_analyze_only(run_context, workflow_type)
    _emit_stage(fake_platform=fake_platform, run_context=run_context, stage=stage)

    metadata = _materialization_metadata(
        run_context=run_context,
        workflow_type=workflow_type,
        stage=stage,
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    return MaterializeResult(metadata=metadata)


@asset(
    name="raw_manifest",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    description="Skeleton: raw asset manifest assembled from immutable raw archive refs.",
)
def raw_manifest(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.raw_manifest")


@asset(
    name="validated_manifest",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("raw_manifest")],
    description="Skeleton: validated manifest after schema/contract checks.",
)
def validated_manifest(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.validated_manifest")


@asset(
    name="tabular_profile_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("validated_manifest")],
    description="Skeleton: tabular profile/EDA report (placeholder for tabular plugin).",
)
def tabular_profile_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.tabular_profile_report")


@asset(
    name="object_analytics_passports",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("tabular_profile_report")],
    description="Skeleton: per-object analytical passports (placeholder).",
)
def object_analytics_passports(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.object_analytics_passports")


@asset(
    name="evidence_bundle",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("object_analytics_passports")],
    description="Skeleton: normalized EvidenceBundle for Decision Core (placeholder).",
)
def evidence_bundle(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.evidence_bundle")


@asset(
    name="decision_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("evidence_bundle")],
    description="Skeleton: Decision Core dataset-level report (placeholder).",
)
def decision_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.decision_report")


@asset(
    name="recommended_actions",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("decision_report")],
    description="Skeleton: recommended actions surfaced to the platform UI.",
)
def recommended_actions(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.recommended_actions")


@asset(
    name="review_queue",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("decision_report")],
    description="Skeleton: review queue for label/privacy/duplicate review.",
)
def review_queue(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_skeleton(context, stage="analyze.review_queue")


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
