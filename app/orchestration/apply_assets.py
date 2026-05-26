"""APPLY_SELECTED_ACTIONS assets for the DataForge AI compute plane (TASK-057).

Apply assets only run after the platform backend has produced an approved
ActionPlan. They are the only path that may emit dataset-changing artifact
references (prepared/redacted/synthetic/candidate datasets, export packages),
and even then they never overwrite raw artifacts — every asset registers
new immutable artifacts via :class:`ArtifactRegistry`.

This module is intentionally a *thin orchestration layer*. The deep
algorithms (imputation, SMOTE, Gaussian Copula, duplicate marking, redaction,
candidate version assembly, model impact, export package) live in
``app.kernel`` and ``app.plugins``. Apply assets:

* validate the ``RunContextResource`` (workflow type + apply context);
* register a small, deterministic JSON artifact per asset that captures
  the asset's status, a stable schema_version and lineage references;
* emit a stage event through the status bridge so the platform can
  surface progress without learning Dagster internals;
* skip synthetic-only material when no synthetic step is selected, but
  still emit the ``synthetic_dataset`` asset with status
  ``not_applicable`` so the asset graph stays stable;
* mark the job timeline as ``COMPLETED`` once the placeholder graph is
  observable, while keeping placeholder artifacts explicitly non-final.

The placeholder JSON payloads make the assets observable from tests and
from the platform UI: each artifact ref carries a content hash, the
schema_version (``apply_*.placeholder.v1``), and a lineage block. This
gives the candidate-version, model-impact and export builders something
real to consume in TASK-058+ once they are wired in.

These placeholders are audit/progress artifacts only. They must not be
surfaced as final candidate dataset, model-impact, or export package refs
until real builders and validation gates replace the placeholder payloads.

This module intentionally does not use ``from __future__ import annotations``
because Dagster validates the ``context`` parameter type via runtime
annotations on the decorated asset functions.
"""

import json
from typing import Any

from dagster import AssetExecutionContext, AssetKey, MaterializeResult, asset

from app.adapters import ArtifactRegistry, FakePlatformMetadataClient, RegisteredArtifact
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

# Stable artifact-kind tokens for the placeholder artifacts each apply
# asset registers. Schema versions are deliberately suffixed with
# ``placeholder.v1`` so a future TASK-058+ replacement can lift the
# suffix while keeping the same artifact_kind on the registry path.
APPLY_ARTIFACT_KINDS: dict[str, str] = {
    "action_plan": "action_plan_artifact",
    "remediation_execution_report": "remediation_execution_report",
    "prepared_dataset": "prepared_dataset_apply_placeholder",
    "synthetic_dataset": "synthetic_dataset_report_apply_placeholder",
    "model_impact_report": "model_impact_report_apply_placeholder",
    "export_package": "export_package_apply_placeholder",
}

APPLY_ARTIFACT_SCHEMA_VERSIONS: dict[str, str] = {
    "action_plan": "action_plan_artifact.v1",
    "remediation_execution_report": "remediation_execution_report.placeholder.v1",
    "prepared_dataset": "prepared_dataset.placeholder.v1",
    "synthetic_dataset": "synthetic_dataset.placeholder.v1",
    "model_impact_report": "model_impact_report.placeholder.v1",
    "export_package": "export_package.placeholder.v1",
}

APPLY_ARTIFACT_FORMAT = "json"
APPLY_ARTIFACT_MEDIA_TYPE = "application/json"


# Apply stages monotonically advance through canonical platform lifecycle
# stages. The skeleton keeps everything in RUNNING_DECISION_CORE and
# transitions to COMPLETED only at the export stage so the platform UI
# renders a stable timeline without revealing Dagster internals.
_APPLY_STAGE_BY_ASSET: dict[str, tuple[JobStage, float]] = {
    "action_plan": (JobStage.RUNNING_DECISION_CORE, 0.30),
    "remediation_execution_report": (JobStage.RUNNING_DECISION_CORE, 0.50),
    "prepared_dataset": (JobStage.RUNNING_DECISION_CORE, 0.65),
    "synthetic_dataset": (JobStage.RUNNING_DECISION_CORE, 0.75),
    "model_impact_report": (JobStage.RUNNING_DECISION_CORE, 0.90),
    "export_package": (JobStage.COMPLETED, 1.0),
}

_APPLY_RESOURCE_KEYS = {"run_context", "fake_platform", "artifact_registry"}


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


def _placeholder_payload(
    *,
    asset_name: str,
    run_context: RunContext,
    apply_context: ApplyRunContext,
    extras: dict[str, object] | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "asset_name": asset_name,
        "schema_version": APPLY_ARTIFACT_SCHEMA_VERSIONS[asset_name],
        "compute_run_id": run_context.compute_run_id,
        "platform_job_id": run_context.platform_job_id,
        "organization_id": run_context.organization_id,
        "project_id": run_context.project_id,
        "dataset_id": run_context.dataset_id,
        "source_dataset_version_id": apply_context.source_dataset_version_id,
        "proposed_version_name": apply_context.proposed_version_name,
        "action_plan_id": apply_context.action_plan_id,
        "decision_report_id": apply_context.decision_report_id,
        "config_hash": apply_context.config_hash,
        "policy_versions": {
            "profile": apply_context.policy_versions.profile_policy_version,
            "decision": apply_context.policy_versions.decision_policy_version,
            "score": apply_context.policy_versions.score_policy_version,
            "method": apply_context.policy_versions.method_policy_version,
            "validation_gates": (
                apply_context.policy_versions.validation_gates_policy_version
            ),
        },
        "synthetic_step_ids": list(apply_context.synthetic_step_ids),
        "input_artifact_uris": [ref.uri for ref in apply_context.input_artifacts],
    }
    if extras:
        payload.update(extras)
    return json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")


def _register_artifact(
    *,
    registry: ArtifactRegistry,
    asset_name: str,
    payload: bytes,
    run_context: RunContext,
    apply_context: ApplyRunContext,
) -> RegisteredArtifact:
    kind = APPLY_ARTIFACT_KINDS[asset_name]
    schema_version = APPLY_ARTIFACT_SCHEMA_VERSIONS[asset_name]
    return registry.save_artifact(
        artifact_kind=kind,
        data=payload,
        artifact_format=APPLY_ARTIFACT_FORMAT,
        media_type=APPLY_ARTIFACT_MEDIA_TYPE,
        schema_version=schema_version,
        dataset_version_id=apply_context.proposed_version_name,
        created_by_job_id=run_context.compute_run_id,
        config_hash=apply_context.config_hash,
        metadata={
            "apply-asset-name": asset_name,
            "apply-action-plan-id": apply_context.action_plan_id,
            "apply-source-version-id": apply_context.source_dataset_version_id,
            "apply-proposed-version-name": apply_context.proposed_version_name,
        },
    )


def _apply_metadata(
    *,
    asset_name: str,
    run_context: RunContext,
    apply_context: ApplyRunContext,
    stage: JobStage,
    progress: float,
    artifact: RegisteredArtifact,
    status: str,
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
        "artifact_uri": artifact.uri,
        "artifact_hash": artifact.hash,
        "schema_version": APPLY_ARTIFACT_SCHEMA_VERSIONS[asset_name],
    }
    if extras:
        payload.update(extras)
    return payload


def _materialize_apply_asset(
    context: AssetExecutionContext,
    *,
    asset_name: str,
    extras: dict[str, object] | None = None,
    artifact_status: str = "provisional_placeholder",
) -> MaterializeResult[None]:
    run_context_resource: RunContextResource = context.resources.run_context
    fake_platform: FakePlatformMetadataClient = context.resources.fake_platform
    artifact_registry: ArtifactRegistry = context.resources.artifact_registry

    run_context = run_context_resource.run_context
    workflow_type = run_context_resource.workflow_type
    apply_context = _require_apply(
        run_context, workflow_type, run_context_resource.apply_context
    )

    payload = _placeholder_payload(
        asset_name=asset_name,
        run_context=run_context,
        apply_context=apply_context,
        extras=extras,
    )
    artifact = _register_artifact(
        registry=artifact_registry,
        asset_name=asset_name,
        payload=payload,
        run_context=run_context,
        apply_context=apply_context,
    )

    stage, progress = _APPLY_STAGE_BY_ASSET[asset_name]
    bridge = RunStatusBridge(fake_platform=fake_platform)
    if stage is JobStage.COMPLETED:
        bridge.emit_completed(
            run_context=run_context,
            artifact_refs=(artifact.artifact_ref,),
        )
    else:
        bridge.emit_stage(
            run_context=run_context,
            stage=stage,
            progress=progress,
            artifact_refs=(artifact.artifact_ref,),
        )

    return MaterializeResult(
        metadata=_apply_metadata(
            asset_name=asset_name,
            run_context=run_context,
            apply_context=apply_context,
            stage=stage,
            progress=progress,
            artifact=artifact,
            status=artifact_status,
            extras=extras,
        )
    )


def _action_plan_extras(apply_context: ApplyRunContext) -> dict[str, object]:
    plan = apply_context.action_plan
    if plan is None:
        return {
            "approved_step_count": 0,
            "approved_step_ids": [],
            "approved_step_types": [],
            "validation_gates": [],
        }
    return {
        "approved_step_count": len(plan.steps),
        "approved_step_ids": [step.step_id for step in plan.steps],
        "approved_step_types": sorted({step.type for step in plan.steps}),
        "validation_gates": list(plan.validation_gates),
        "execution_mode": plan.execution_mode.value,
    }


def _remediation_extras(apply_context: ApplyRunContext) -> dict[str, object]:
    plan = apply_context.action_plan
    if plan is None:
        return {
            "executed_step_count": 0,
            "executed_steps": [],
            "synthetic_steps_present": apply_context.has_synthetic,
        }
    return {
        "executed_step_count": len(plan.steps),
        "executed_steps": [
            {
                "step_id": step.step_id,
                "action_type": step.type,
                "method_id": step.method_id,
                "plugin_id": step.plugin_id,
                "plugin_version": step.plugin_version,
            }
            for step in plan.steps
        ],
        "synthetic_steps_present": apply_context.has_synthetic,
        "synthetic_step_ids": list(apply_context.synthetic_step_ids),
    }


def _prepared_dataset_extras(apply_context: ApplyRunContext) -> dict[str, object]:
    non_synthetic_steps = (
        []
        if apply_context.action_plan is None
        else [
            step.step_id
            for step in apply_context.action_plan.steps
            if step.step_id not in apply_context.synthetic_step_ids
        ]
    )
    return {
        "prepared_step_ids": non_synthetic_steps,
        "input_artifact_count": len(apply_context.input_artifacts),
    }


def _synthetic_dataset_extras(apply_context: ApplyRunContext) -> dict[str, object]:
    if not apply_context.has_synthetic:
        return {
            "applicable": False,
            "status": "not_applicable",
            "reason_code": "no_synthetic_step_selected",
            "synthetic_step_ids": [],
        }
    return {
        "applicable": True,
        "status": "provisional_placeholder",
        "synthetic_step_ids": list(apply_context.synthetic_step_ids),
    }


def _model_impact_extras(apply_context: ApplyRunContext) -> dict[str, object]:
    return {
        "applicable": True,
        "status": "provisional_placeholder",
        "require_eligibility": apply_context.require_model_impact_eligibility,
        "synthetic_aware": apply_context.has_synthetic,
    }


def _export_package_extras(apply_context: ApplyRunContext) -> dict[str, object]:
    return {
        "status": "PROVISIONAL_PLACEHOLDER",
        "export_gates_status": "not_evaluated",
        "reason_code": "real_export_package_not_materialized",
        "synthetic_aware": apply_context.has_synthetic,
        "require_model_impact_eligibility": (
            apply_context.require_model_impact_eligibility
        ),
    }


# ---------------------------------------------------------------------------
# Asset definitions
# ---------------------------------------------------------------------------


@asset(
    name="action_plan",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    description="Approved ActionPlan loaded from the platform and pinned to the run.",
)
def action_plan(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_asset(
        context,
        asset_name="action_plan",
        extras=_action_plan_extras(context.resources.run_context.apply_context),
        artifact_status="registered",
    )


@asset(
    name="remediation_execution_report",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("action_plan")],
    description="Report produced by ActionPlan step execution (placeholder for TASK-058).",
)
def remediation_execution_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_asset(
        context,
        asset_name="remediation_execution_report",
        extras=_remediation_extras(context.resources.run_context.apply_context),
    )


@asset(
    name="prepared_dataset",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("remediation_execution_report")],
    description=(
        "Prepared candidate dataset artifact (placeholder; real Parquet "
        "writes land in TASK-054 export writers)."
    ),
)
def prepared_dataset(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_asset(
        context,
        asset_name="prepared_dataset",
        extras=_prepared_dataset_extras(context.resources.run_context.apply_context),
    )


@asset(
    name="synthetic_dataset",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("prepared_dataset")],
    description=(
        "Synthetic candidate dataset artifact (skipped with status "
        "not_applicable if no synthetic step is selected)."
    ),
)
def synthetic_dataset(context: AssetExecutionContext) -> MaterializeResult[None]:
    apply_context = context.resources.run_context.apply_context
    extras = _synthetic_dataset_extras(apply_context)
    artifact_status = (
        "not_applicable"
        if not (apply_context and apply_context.has_synthetic)
        else "provisional_placeholder"
    )
    return _materialize_apply_asset(
        context,
        asset_name="synthetic_dataset",
        extras=extras,
        artifact_status=artifact_status,
    )


@asset(
    name="model_impact_report",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("prepared_dataset"), AssetKey("synthetic_dataset")],
    description="Baseline vs candidate model impact report (placeholder for TASK-051).",
)
def model_impact_report(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_asset(
        context,
        asset_name="model_impact_report",
        extras=_model_impact_extras(context.resources.run_context.apply_context),
    )


@asset(
    name="export_package",
    group_name=APPLY_GROUP,
    required_resource_keys=_APPLY_RESOURCE_KEYS,
    deps=[AssetKey("model_impact_report")],
    description="Export package after readiness gates (placeholder for TASK-053).",
)
def export_package(context: AssetExecutionContext) -> MaterializeResult[None]:
    return _materialize_apply_asset(
        context,
        asset_name="export_package",
        extras=_export_package_extras(context.resources.run_context.apply_context),
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
    """Return the schema_version emitted by an APPLY asset placeholder."""
    return APPLY_ARTIFACT_SCHEMA_VERSIONS[asset_name]


__all__ = [
    "APPLY_ARTIFACT_FORMAT",
    "APPLY_ARTIFACT_KINDS",
    "APPLY_ARTIFACT_MEDIA_TYPE",
    "APPLY_ARTIFACT_SCHEMA_VERSIONS",
    "APPLY_ASSETS",
    "APPLY_ASSET_KEYS",
    "APPLY_GROUP",
    "action_plan",
    "asset_kind_for",
    "export_package",
    "model_impact_report",
    "prepared_dataset",
    "remediation_execution_report",
    "schema_version_for",
    "synthetic_dataset",
]
