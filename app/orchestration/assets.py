"""ANALYZE_ONLY assets for the DataForge AI compute plane.

These assets run the existing contract-compatible builders and register
immutable analysis artifacts through scoped object storage. The graph is
strictly non-mutating: it reads source/prediction artifacts and writes derived
reports, evidence, recommendations, and review queues only.

Strict constraints honored here:

* ``ANALYZE_ONLY`` must not mutate source dataset artifacts.
* Materialization metadata only contains stable technical fields. No raw
  PII, raw text, file contents, or secrets.
* Decision Core consumes normalized evidence; plugin-specific outputs are
  persisted as artifacts and referenced by hash/URI only.

This module intentionally does not use ``from __future__ import annotations``
because Dagster validates the ``context`` parameter type via runtime
annotations on the decorated asset functions.
"""

from typing import Any

from dagster import AssetExecutionContext, AssetKey, MaterializeResult, asset

from app.adapters import ArtifactRegistry, FakePlatformMetadataClient
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import WorkflowType
from app.orchestration.analyze_runtime import (
    artifact_for,
    count_for,
    ensure_analyze_outputs,
    status_for,
)
from app.orchestration.job_event import JobStage
from app.orchestration.run_context import AnalyzeRunContext, RunContextResource
from app.orchestration.status_bridge import RunContext, RunStatusBridge
from app.telemetry import MetricsRegistry, TracingRegistry

ANALYZE_GROUP = "analyze_only"

BASE_ANALYZE_ASSET_KEYS: tuple[AssetKey, ...] = (
    AssetKey("raw_manifest"),
    AssetKey("validated_manifest"),
    AssetKey("tabular_profile_report"),
    AssetKey("object_analytics_passports"),
    AssetKey("evidence_bundle"),
    AssetKey("decision_report"),
    AssetKey("recommended_actions"),
    AssetKey("review_queue"),
)

PREDICTION_ANALYZE_ASSET_KEYS: tuple[AssetKey, ...] = (
    AssetKey("prediction_manifest"),
    AssetKey("prediction_validation_report"),
    AssetKey("model_error_analysis_report"),
    AssetKey("ambiguous_object_candidates"),
    AssetKey("probable_label_error_candidates"),
)

ANALYZE_ASSET_KEYS: tuple[AssetKey, ...] = (
    *BASE_ANALYZE_ASSET_KEYS,
    *PREDICTION_ANALYZE_ASSET_KEYS,
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
    "prediction_manifest": (JobStage.BUILDING_MANIFEST, 0.35),
    "prediction_validation_report": (JobStage.BUILDING_EVIDENCE, 0.62),
    "model_error_analysis_report": (JobStage.BUILDING_EVIDENCE, 0.68),
    "ambiguous_object_candidates": (JobStage.BUILDING_EVIDENCE, 0.72),
    "probable_label_error_candidates": (JobStage.BUILDING_EVIDENCE, 0.74),
    "decision_report": (JobStage.RUNNING_DECISION_CORE, 0.80),
    "recommended_actions": (JobStage.RUNNING_DECISION_CORE, 0.90),
    "review_queue": (JobStage.RUNNING_DECISION_CORE, 0.95),
}

_ANALYZE_RESOURCE_KEYS = {
    "run_context",
    "fake_platform",
    "artifact_registry",
    "object_storage",
    "metrics",
    "tracing",
}


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
        "analysis_mode": "artifact_materialization",
    }


def _materialize_analyze_asset(
    context: AssetExecutionContext,
    *,
    asset_name: str,
) -> MaterializeResult[None]:
    run_context_resource: RunContextResource = context.resources.run_context
    fake_platform: FakePlatformMetadataClient = context.resources.fake_platform
    registry: ArtifactRegistry = context.resources.artifact_registry
    storage: MinioObjectStorageAdapter = context.resources.object_storage
    metrics: MetricsRegistry = context.resources.metrics
    tracing: TracingRegistry = context.resources.tracing

    run_context = run_context_resource.run_context
    workflow_type = run_context_resource.workflow_type
    analyze_context = _require_analyze_context(run_context_resource.analyze_context)
    _require_analyze_only(run_context, workflow_type)

    ensure_analyze_outputs(
        analyze_context=analyze_context,
        run_context=run_context,
        storage=storage,
        registry=registry,
        metrics=metrics,
        tracing=tracing,
    )

    stage, progress = _ANALYZE_STAGE_BY_ASSET[asset_name]

    bridge = RunStatusBridge(fake_platform=fake_platform)
    bridge.emit_stage(run_context=run_context, stage=stage, progress=progress)

    metadata = _materialization_metadata(
        run_context=run_context,
        workflow_type=workflow_type,
        stage=stage,
        progress=progress,
    )
    artifact = artifact_for(analyze_context=analyze_context, asset_name=asset_name)
    if artifact is not None:
        metadata["artifact_uri"] = artifact.uri
        metadata["artifact_hash"] = artifact.hash
        metadata["artifact_kind"] = artifact.artifact_kind
        metadata["schema_version"] = artifact.schema_version
        metadata["artifact_uris"] = [artifact.uri]
    manifest_count = count_for(analyze_context=analyze_context, name="manifest_row_count")
    review_size = count_for(analyze_context=analyze_context, name="review_queue_size")
    analysis_mode = status_for(analyze_context=analyze_context, name="analysis_mode")
    if manifest_count is not None:
        metadata["manifest_row_count"] = manifest_count
    if review_size is not None:
        metadata["review_queue_size"] = review_size
    if analysis_mode is not None:
        metadata["analysis_mode"] = analysis_mode
    return MaterializeResult(metadata=metadata)


def _require_analyze_context(context: AnalyzeRunContext | None) -> AnalyzeRunContext:
    if context is None:
        raise ValueError("ANALYZE_ONLY assets require AnalyzeRunContext")
    return context


@asset(
    name="raw_manifest",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    description="Raw asset manifest assembled from immutable raw archive refs.",
)
def raw_manifest(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="raw_manifest")


@asset(
    name="validated_manifest",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("raw_manifest")],
    description="Validated manifest after schema/contract checks.",
)
def validated_manifest(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="validated_manifest")


@asset(
    name="tabular_profile_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("validated_manifest")],
    description="Tabular profile/EDA report produced by the tabular plugin.",
)
def tabular_profile_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="tabular_profile_report")


@asset(
    name="object_analytics_passports",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("tabular_profile_report")],
    description="Per-object analytical passports.",
)
def object_analytics_passports(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="object_analytics_passports")


@asset(
    name="evidence_bundle",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("object_analytics_passports")],
    description="Normalized EvidenceBundle artifacts for Decision Core.",
)
def evidence_bundle(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="evidence_bundle")


@asset(
    name="prediction_manifest",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("validated_manifest")],
    description="Normalized optional predictions manifest.",
)
def prediction_manifest(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="prediction_manifest")


@asset(
    name="prediction_validation_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("prediction_manifest")],
    description="Prediction-manifest validation report.",
)
def prediction_validation_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="prediction_validation_report")


@asset(
    name="model_error_analysis_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("prediction_validation_report")],
    description="Model error analysis derived from predictions.",
)
def model_error_analysis_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="model_error_analysis_report")


@asset(
    name="ambiguous_object_candidates",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("model_error_analysis_report")],
    description="Ambiguous-object review candidates derived from predictions.",
)
def ambiguous_object_candidates(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="ambiguous_object_candidates")


@asset(
    name="probable_label_error_candidates",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("model_error_analysis_report")],
    description="Probable-label-error review candidates derived from predictions.",
)
def probable_label_error_candidates(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="probable_label_error_candidates")


@asset(
    name="decision_report",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("evidence_bundle")],
    description="Decision Core dataset-level report.",
)
def decision_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="decision_report")


@asset(
    name="recommended_actions",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("decision_report")],
    description="Recommended actions surfaced to the platform UI.",
)
def recommended_actions(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="recommended_actions")


@asset(
    name="review_queue",
    group_name=ANALYZE_GROUP,
    required_resource_keys=_ANALYZE_RESOURCE_KEYS,
    deps=[AssetKey("decision_report")],
    description="Review queue for label/privacy/duplicate review.",
)
def review_queue(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_analyze_asset(context, asset_name="review_queue")


ANALYZE_ASSETS = (
    raw_manifest,
    validated_manifest,
    tabular_profile_report,
    object_analytics_passports,
    evidence_bundle,
    prediction_manifest,
    prediction_validation_report,
    model_error_analysis_report,
    ambiguous_object_candidates,
    probable_label_error_candidates,
    decision_report,
    recommended_actions,
    review_queue,
)


__all__ = [
    "ANALYZE_ASSET_KEYS",
    "ANALYZE_ASSETS",
    "ANALYZE_GROUP",
    "BASE_ANALYZE_ASSET_KEYS",
    "PREDICTION_ANALYZE_ASSET_KEYS",
    "ambiguous_object_candidates",
    "decision_report",
    "evidence_bundle",
    "model_error_analysis_report",
    "object_analytics_passports",
    "prediction_manifest",
    "prediction_validation_report",
    "probable_label_error_candidates",
    "raw_manifest",
    "recommended_actions",
    "review_queue",
    "tabular_profile_report",
    "validated_manifest",
]
