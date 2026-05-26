"""Runtime helpers for real APPLY_SELECTED_ACTIONS materializations.

The functions in this module are called by the Dagster apply assets. They
execute approved ActionPlan steps through existing plugin/kernel builders and
store only immutable derived artifacts in object storage. Raw source artifacts
are read, never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    RegisteredArtifact,
)
from app.adapters.object_storage import MinioObjectStorageAdapter, ObjectStorageError
from app.domain import (
    ActionPlan,
    ActionPlanStep,
    ArtifactLineage,
    ArtifactRef,
    CandidateVersionStatus,
    DecisionAction,
    ErrorCode,
    ExportPackageStatus,
    SplitManifest,
    ValidationGatesReport,
)
from app.domain.common import Sha256Digest
from app.kernel import (
    BuildCandidateVersionRequest,
    BuildExportPackageRequest,
    BuildLineageReportRequest,
    RunModelImpactRequest,
    build_candidate_dataset_version,
    build_export_package,
    build_lineage_report,
    run_model_impact,
)
from app.orchestration.run_context import ApplyExecutionState, ApplyRunContext
from app.orchestration.status_bridge import RunContext
from app.plugins.export import TabularExportRequest, write_tabular_export
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    ExecuteSmoteAugmentationRequest,
    ExecuteTabularDuplicatesRequest,
    ExecuteTabularImputationRequest,
    ExecuteTabularSplitRequest,
    execute_smote_augmentation_action,
    execute_tabular_duplicates_action,
    execute_tabular_imputation_action,
    execute_tabular_split_action,
)
from app.plugins.tabular.rules import BusinessRule, compute_rules_config_hash, parse_business_rules
from app.plugins.validation import DcrThresholds, RunValidationGatesRequest, run_validation_gates
from app.reports.dataset_card import BuildDatasetCardRequest, build_dataset_card_artifact

ACTION_PLAN_ARTIFACT_KIND = "action_plan_artifact"
ACTION_PLAN_ARTIFACT_FORMAT = "json"
ACTION_PLAN_ARTIFACT_MEDIA_TYPE = "application/json"
ACTION_PLAN_ARTIFACT_SCHEMA_VERSION = "action_plan.v1"

REMEDIATION_EXECUTION_REPORT_KIND = "remediation_execution_report"
REMEDIATION_EXECUTION_REPORT_FORMAT = "json"
REMEDIATION_EXECUTION_REPORT_MEDIA_TYPE = "application/json"
REMEDIATION_EXECUTION_REPORT_SCHEMA_VERSION = "remediation_execution_report.v1"

EXPORT_MANIFEST_KIND = "EXPORT_MANIFEST"
EXPORT_MANIFEST_FORMAT = "json"
EXPORT_MANIFEST_MEDIA_TYPE = "application/json"
EXPORT_MANIFEST_SCHEMA_VERSION = "export_manifest.v1"

_DEFAULT_TARGET_COLUMN = "is_fraud"
_DEFAULT_RARE_CLASS_LABEL = "1"
_SYNTHETIC_STEP_TYPES = {
    DecisionAction.AUGMENT_RARE_CLASS.value,
    DecisionAction.GENERATE_SYNTHETIC_CANDIDATE.value,
}


class ApplyRuntimeError(ValueError):
    """Raised when APPLY cannot safely produce real gated artifacts."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.ACTION_PLAN_PRECONDITION_FAILED,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


def persist_action_plan_artifact(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    registry: ArtifactRegistry,
) -> RegisteredArtifact:
    """Persist the approved ActionPlan payload as a real immutable artifact."""
    state = apply_context.execution_state
    if state.action_plan_artifact is not None:
        return state.action_plan_artifact
    plan = _require_plan(apply_context)
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True, indent=2).encode(
        "utf-8"
    )
    artifact = registry.save_artifact(
        artifact_kind=ACTION_PLAN_ARTIFACT_KIND,
        data=payload,
        artifact_format=ACTION_PLAN_ARTIFACT_FORMAT,
        media_type=ACTION_PLAN_ARTIFACT_MEDIA_TYPE,
        schema_version=ACTION_PLAN_ARTIFACT_SCHEMA_VERSION,
        dataset_version_id=apply_context.proposed_version_name,
        created_by_job_id=run_context.compute_run_id,
        config_hash=apply_context.config_hash,
        metadata={
            "action-plan-id": plan.action_plan_id,
            "source-version-id": apply_context.source_dataset_version_id,
            "proposed-version-name": apply_context.proposed_version_name,
            "created-at": _generated_at(plan).isoformat(),
        },
    )
    state.action_plan_artifact = artifact
    return artifact


def execute_remediation_plan(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> RegisteredArtifact:
    """Execute selected ActionPlan steps and run candidate validation gates."""
    state = apply_context.execution_state
    if state.remediation_report_artifact is not None:
        return state.remediation_report_artifact
    plan = _require_plan(apply_context)
    source_artifact = _resolve_source_artifact(
        apply_context=apply_context,
        run_context=run_context,
        storage=storage,
    )
    current_artifact = source_artifact
    step_summaries: list[dict[str, object]] = []

    for step in plan.steps:
        candidate_artifact, produced = _execute_step(
            step=step,
            current_artifact=current_artifact,
            plan=plan,
            apply_context=apply_context,
            run_context=run_context,
            storage=storage,
            registry=registry,
        )
        current_artifact = candidate_artifact.artifact_ref
        state.primary_candidate_artifact = candidate_artifact
        state.step_output_artifacts.extend(produced)
        step_summaries.append(
            {
                "step_id": step.step_id,
                "step_type": step.type,
                "method_id": step.method_id,
                "plugin_id": step.plugin_id,
                "plugin_version": step.plugin_version,
                "output_artifact_uris": [artifact.uri for artifact in produced],
                "output_artifact_hashes": [artifact.hash for artifact in produced],
            }
        )

    if state.primary_candidate_artifact is None:
        raise ApplyRuntimeError(
            reason_code="action_plan_produced_no_candidate_artifact",
            message="APPLY ActionPlan did not produce a candidate dataset artifact.",
        )

    gates = _run_validation_gates(
        apply_context=apply_context,
        run_context=run_context,
        storage=storage,
        registry=registry,
    )
    report_payload = {
        "report_schema_version": REMEDIATION_EXECUTION_REPORT_SCHEMA_VERSION,
        "action_plan_id": plan.action_plan_id,
        "source_dataset_version_id": apply_context.source_dataset_version_id,
        "candidate_dataset_version_id": apply_context.proposed_version_name,
        "created_by_job_id": run_context.compute_run_id,
        "step_count": len(plan.steps),
        "steps": step_summaries,
        "primary_candidate_artifact": (
            state.primary_candidate_artifact.artifact_ref.model_dump(mode="json")
        ),
        "validation_gates_report": gates.model_dump(mode="json"),
        "validation_status": gates.candidate_status.value,
        "block_export": gates.block_export,
        "generated_at": _generated_at(plan).isoformat(),
    }
    artifact = registry.save_artifact(
        artifact_kind=REMEDIATION_EXECUTION_REPORT_KIND,
        data=json.dumps(report_payload, sort_keys=True, indent=2).encode("utf-8"),
        artifact_format=REMEDIATION_EXECUTION_REPORT_FORMAT,
        media_type=REMEDIATION_EXECUTION_REPORT_MEDIA_TYPE,
        schema_version=REMEDIATION_EXECUTION_REPORT_SCHEMA_VERSION,
        dataset_version_id=apply_context.proposed_version_name,
        created_by_job_id=run_context.compute_run_id,
        config_hash=apply_context.config_hash,
        metadata={
            "action-plan-id": plan.action_plan_id,
            "validation-status": gates.candidate_status.value,
            "block-export": "true" if gates.block_export else "false",
            "created-at": _generated_at(plan).isoformat(),
        },
    )
    state.remediation_report_artifact = artifact
    return artifact


def ensure_model_impact_report(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> RegisteredArtifact | None:
    """Run model impact when the tabular candidate is eligible."""
    state = apply_context.execution_state
    if state.model_impact_report_artifact is not None:
        return state.model_impact_report_artifact
    if state.primary_candidate_artifact is None:
        execute_remediation_plan(
            apply_context=apply_context,
            run_context=run_context,
            storage=storage,
            registry=registry,
        )
    if (
        state.source_artifact is None
        or state.primary_candidate_artifact is None
        or state.validation_gates_report is None
    ):
        return None

    try:
        split, split_artifact = _ensure_split_manifest(
            apply_context=apply_context,
            run_context=run_context,
            source_artifact=state.source_artifact,
            storage=storage,
            registry=registry,
        )
        candidate_split = state.split_manifest or split
        candidate_split_artifact = state.split_artifact or split_artifact
        feature_columns = _numeric_feature_columns(
            storage=storage,
            artifact=state.primary_candidate_artifact.artifact_ref,
            target_column=_target_column(_require_plan(apply_context)),
        )
        if not feature_columns:
            return None
        result = run_model_impact(
            RunModelImpactRequest(
                dataset_id=run_context.dataset_id,
                parent_version_id=apply_context.source_dataset_version_id,
                candidate_dataset_version_id=apply_context.proposed_version_name,
                organization_id=run_context.organization_id,
                project_id=run_context.project_id,
                source_artifact=state.source_artifact,
                candidate_artifact=state.primary_candidate_artifact.artifact_ref,
                source_split_manifest=split,
                candidate_split_manifest=candidate_split,
                feature_columns=feature_columns,
                target_column=_target_column(_require_plan(apply_context)),
                rare_class_label=_rare_class_label(_require_plan(apply_context)),
                candidate_is_synthetic=state.synthetic_dataset_report is not None,
                source_split_manifest_artifact=split_artifact.artifact_ref,
                candidate_split_manifest_artifact=candidate_split_artifact.artifact_ref,
                validation_gates_report_artifact=(
                    state.validation_gates_report_artifact.artifact_ref
                    if state.validation_gates_report_artifact is not None
                    else None
                ),
                synthetic_dataset_report_artifact=(
                    state.synthetic_dataset_report_artifact.artifact_ref
                    if state.synthetic_dataset_report_artifact is not None
                    else None
                ),
                validation_gates_blocker_present=state.validation_gates_report.blocker_present,
                random_seed=_random_seed(_require_plan(apply_context)),
                created_by_job_id=run_context.compute_run_id,
                config_hash=apply_context.config_hash,
                report_id=_stable_id(
                    "model_impact_report",
                    _require_plan(apply_context).action_plan_id,
                    apply_context.proposed_version_name,
                    apply_context.config_hash,
                ),
                generated_at=_generated_at(_require_plan(apply_context)),
            ),
            storage=storage,
            registry=registry,
        )
    except Exception:
        if apply_context.require_model_impact_eligibility:
            raise
        return None
    state.model_impact_report = result.report
    state.model_impact_report_artifact = result.report_artifact
    return result.report_artifact


def build_final_apply_outputs(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    platform_client: FakePlatformMetadataClient,
) -> RegisteredArtifact:
    """Build candidate metadata, dataset card, lineage, export package."""
    state = apply_context.execution_state
    if state.export_package_artifact is not None:
        return state.export_package_artifact
    if state.primary_candidate_artifact is None:
        execute_remediation_plan(
            apply_context=apply_context,
            run_context=run_context,
            storage=storage,
            registry=registry,
        )
    ensure_model_impact_report(
        apply_context=apply_context,
        run_context=run_context,
        storage=storage,
        registry=registry,
    )
    if (
        state.source_artifact is None
        or state.primary_candidate_artifact is None
        or state.validation_gates_report is None
        or state.validation_gates_report_artifact is None
    ):
        raise ApplyRuntimeError(
            reason_code="candidate_validation_outputs_missing",
            message="APPLY cannot build final outputs without validation gates.",
        )

    candidate_result = build_candidate_dataset_version(
        BuildCandidateVersionRequest(
            organization_id=run_context.organization_id,
            project_id=run_context.project_id,
            dataset_id=run_context.dataset_id,
            parent_version_id=apply_context.source_dataset_version_id,
            proposed_version_name=apply_context.proposed_version_name,
            action_plan=_require_plan(apply_context),
            policy_versions=apply_context.policy_versions,
            decision_report_id=apply_context.decision_report_id,
            created_by_job_id=run_context.compute_run_id,
            config_hash=apply_context.config_hash,
            source_artifacts=(state.source_artifact,),
            candidate_artifacts=_candidate_artifact_refs(state),
            primary_dataset_artifact=state.primary_candidate_artifact.artifact_ref,
            validation_gates_report=state.validation_gates_report,
            validation_gates_report_artifact=state.validation_gates_report_artifact.artifact_ref,
            synthetic_dataset_report=state.synthetic_dataset_report,
            synthetic_dataset_report_artifact=(
                state.synthetic_dataset_report_artifact.artifact_ref
                if state.synthetic_dataset_report_artifact is not None
                else None
            ),
            synthetic_validation_report_artifact=(
                state.validation_gates_report_artifact.artifact_ref
                if state.synthetic_dataset_report is not None
                else None
            ),
            model_impact_report_artifact=(
                state.model_impact_report_artifact.artifact_ref
                if state.model_impact_report_artifact is not None
                else None
            ),
            candidate_version_id=_stable_id(
                "candidate_dataset_version",
                _require_plan(apply_context).action_plan_id,
                apply_context.proposed_version_name,
                apply_context.config_hash,
            ),
            proposed_at=_generated_at(_require_plan(apply_context)),
        ),
        registry=registry,
        platform_client=platform_client,
    )
    state.candidate_dataset_version = candidate_result.candidate_version
    state.candidate_version_artifact = candidate_result.candidate_version_artifact

    dataset_card = build_dataset_card_artifact(
        BuildDatasetCardRequest(
            organization_id=run_context.organization_id,
            project_id=run_context.project_id,
            dataset_id=run_context.dataset_id,
            candidate_dataset_version=state.candidate_dataset_version,
            validation_gates_report=state.validation_gates_report,
            model_impact_report=state.model_impact_report,
            split_manifest=state.split_manifest,
            created_by_job_id=run_context.compute_run_id,
            config_hash=apply_context.config_hash,
            object_count=_row_count(
                storage=storage,
                artifact=state.primary_candidate_artifact.artifact_ref,
            ),
            generated_at=_generated_at(_require_plan(apply_context)),
            dataset_card_id=_stable_id(
                "dataset_card",
                state.candidate_dataset_version.candidate_version_id,
                apply_context.config_hash,
            ),
        ),
        registry=registry,
    )
    state.dataset_card_artifact = dataset_card.artifact

    lineage = build_lineage_report(
        BuildLineageReportRequest(
            organization_id=run_context.organization_id,
            project_id=run_context.project_id,
            dataset_id=run_context.dataset_id,
            candidate_dataset_version=state.candidate_dataset_version,
            candidate_version_artifact=state.candidate_version_artifact.artifact_ref,
            input_artifact_refs=(state.source_artifact,),
            output_artifact_refs=_candidate_artifact_refs(state),
            created_by_job_id=run_context.compute_run_id,
            config_hash=apply_context.config_hash,
            lineage_report_id=_stable_id(
                "lineage_report",
                state.candidate_dataset_version.candidate_version_id,
                apply_context.config_hash,
            ),
            generated_at=_generated_at(_require_plan(apply_context)),
        ),
        registry=registry,
        platform_client=platform_client,
    )
    state.lineage_artifact = lineage.lineage_report_artifact

    tabular_refs: tuple[ArtifactRef, ...] = ()
    if state.candidate_dataset_version.status is not CandidateVersionStatus.BLOCKED:
        export_artifacts = write_tabular_export(
            TabularExportRequest(
                dataset_id=run_context.dataset_id,
                candidate_dataset_version_id=apply_context.proposed_version_name,
                source_artifact=state.primary_candidate_artifact.artifact_ref,
                split_manifest=state.split_manifest,
                split_manifest_artifact=(
                    state.split_artifact.artifact_ref
                    if state.split_artifact is not None
                    else None
                ),
                write_csv=True,
                write_per_split=state.split_manifest is not None,
                created_by_job_id=run_context.compute_run_id,
                config_hash=apply_context.config_hash,
                generated_at=_generated_at(_require_plan(apply_context)),
            ),
            storage=storage,
            registry=registry,
        )
        tabular_refs = export_artifacts.all_artifact_refs()
        state.tabular_export_artifacts = tabular_refs

    export_manifest = _persist_export_manifest(
        apply_context=apply_context,
        run_context=run_context,
        state=state,
        registry=registry,
    )
    state.export_manifest_artifact = export_manifest

    package_result = build_export_package(
        BuildExportPackageRequest(
            organization_id=run_context.organization_id,
            project_id=run_context.project_id,
            dataset_id=run_context.dataset_id,
            candidate_dataset_version=state.candidate_dataset_version,
            validation_gates_report=state.validation_gates_report,
            model_impact_report=state.model_impact_report,
            decision_report_id=apply_context.decision_report_id,
            export_manifest_artifact=export_manifest.artifact_ref,
            tabular_artifacts=tabular_refs,
            dataset_card_artifact=state.dataset_card_artifact.artifact_ref,
            lineage_artifact=state.lineage_artifact.artifact_ref,
            extra_artifacts=_extra_export_artifacts(state),
            included_object_count=_row_count(
                storage=storage,
                artifact=state.primary_candidate_artifact.artifact_ref,
            ),
            blocked_object_count=1
            if state.candidate_dataset_version.block_export
            else 0,
            require_model_impact_eligibility=apply_context.require_model_impact_eligibility,
            created_by_job_id=run_context.compute_run_id,
            config_hash=apply_context.config_hash,
            export_package_id=_stable_id(
                "export_package",
                state.candidate_dataset_version.candidate_version_id,
                apply_context.config_hash,
            ),
            created_at=_generated_at(_require_plan(apply_context)),
        ),
        registry=registry,
    )
    state.export_package = package_result.export_package
    state.export_package_artifact = package_result.package_artifact
    return package_result.package_artifact


def final_candidate_ref(state: ApplyExecutionState) -> RegisteredArtifact | None:
    """Return a non-blocked candidate metadata artifact suitable for API response."""
    candidate = state.candidate_dataset_version
    if candidate is None or state.candidate_version_artifact is None:
        return None
    if candidate.status in {CandidateVersionStatus.BLOCKED, CandidateVersionStatus.FAILED}:
        return None
    return state.candidate_version_artifact


def final_export_ref(state: ApplyExecutionState) -> RegisteredArtifact | None:
    """Return a READY export package artifact suitable for API response."""
    if (
        state.export_package is None
        or state.export_package_artifact is None
        or state.export_package.status is not ExportPackageStatus.READY
    ):
        return None
    return state.export_package_artifact


def synthetic_status(state: ApplyExecutionState, apply_context: ApplyRunContext) -> str:
    if not apply_context.has_synthetic:
        return "not_applicable"
    if state.synthetic_dataset_report is None:
        return "failed"
    if state.validation_gates_report is not None and state.validation_gates_report.block_export:
        return "blocked"
    return "materialized"


def _execute_step(
    *,
    step: ActionPlanStep,
    current_artifact: ArtifactRef,
    plan: ActionPlan,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> tuple[RegisteredArtifact, list[RegisteredArtifact]]:
    state = apply_context.execution_state
    if step.type == DecisionAction.IMPUTE_MISSING_VALUES.value:
        imputation_result = execute_tabular_imputation_action(
            ExecuteTabularImputationRequest(
                action_plan_id=plan.action_plan_id,
                step=step,
                source_dataset_version_id=apply_context.source_dataset_version_id,
                candidate_dataset_version_id=apply_context.proposed_version_name,
                target_column=_target_column(plan),
                source_artifact=current_artifact,
                created_by_job_id=run_context.compute_run_id,
                config_hash=step.config_hash,
                report_id=_stable_id("tabular_imputation", plan.action_plan_id, step.step_id),
                generated_at=_generated_at(plan),
            ),
            storage=storage,
            registry=registry,
        )
        return imputation_result.candidate_artifact, [
            imputation_result.candidate_artifact,
            imputation_result.report_artifact,
        ]

    if step.type in {"MARK_DUPLICATE_CANDIDATES", "REMOVE_DUPLICATES"}:
        duplicates_result = execute_tabular_duplicates_action(
            ExecuteTabularDuplicatesRequest(
                action_plan_id=plan.action_plan_id,
                step=step,
                dataset_id=run_context.dataset_id,
                source_dataset_version_id=apply_context.source_dataset_version_id,
                candidate_dataset_version_id=apply_context.proposed_version_name,
                source_artifact=current_artifact,
                created_by_job_id=run_context.compute_run_id,
                config_hash=step.config_hash,
                report_id=_stable_id("duplicate_action", plan.action_plan_id, step.step_id),
                generated_at=_generated_at(plan),
            ),
            storage=storage,
            registry=registry,
        )
        return duplicates_result.candidate_artifact, [
            duplicates_result.candidate_artifact,
            duplicates_result.report_artifact,
        ]

    if step.type in _SYNTHETIC_STEP_TYPES and step.method_id == "smote":
        split, split_artifact = _ensure_split_manifest(
            apply_context=apply_context,
            run_context=run_context,
            source_artifact=current_artifact,
            storage=storage,
            registry=registry,
        )
        smote_result = execute_smote_augmentation_action(
            ExecuteSmoteAugmentationRequest(
                action_plan_id=plan.action_plan_id,
                step=step,
                dataset_id=run_context.dataset_id,
                source_dataset_version_id=apply_context.source_dataset_version_id,
                candidate_dataset_version_id=apply_context.proposed_version_name,
                source_artifact=current_artifact,
                split_manifest=split,
                split_manifest_artifact=split_artifact.artifact_ref,
                created_by_job_id=run_context.compute_run_id,
                config_hash=step.config_hash,
                target_column=_target_column(plan),
                rare_class_label=_rare_class_label(plan),
                random_seed=step.random_seed or 42,
                k_neighbors=int(step.config.get("k_neighbors", 3)),
                sampling_strategy=float(step.config.get("sampling_strategy", 0.20)),
                report_id=_stable_id(
                    "synthetic_dataset_report",
                    plan.action_plan_id,
                    step.step_id,
                    step.config_hash,
                ),
                generated_at=_generated_at(plan),
            ),
            storage=storage,
            registry=registry,
        )
        state.split_manifest = SplitManifest.model_validate_json(
            storage.get(smote_result.augmented_split_artifact.uri).data
        )
        state.split_artifact = smote_result.augmented_split_artifact
        state.synthetic_dataset_report = smote_result.report
        state.synthetic_dataset_report_artifact = smote_result.report_artifact
        state.synthetic_candidate_artifact = smote_result.candidate_artifact
        return smote_result.candidate_artifact, [
            smote_result.candidate_artifact,
            smote_result.augmented_split_artifact,
            smote_result.report_artifact,
        ]

    raise ApplyRuntimeError(
        reason_code="unsupported_action_plan_step",
        message="APPLY runtime does not support the selected ActionPlan step.",
        details={
            "step_id": step.step_id,
            "step_type": step.type,
            "method_id": step.method_id,
        },
    )


def _run_validation_gates(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> ValidationGatesReport:
    state = apply_context.execution_state
    if state.validation_gates_report is not None:
        return state.validation_gates_report
    if state.source_artifact is None or state.primary_candidate_artifact is None:
        raise ApplyRuntimeError(
            reason_code="validation_inputs_missing",
            message="Candidate validation requires source and candidate artifacts.",
        )
    plan = _require_plan(apply_context)
    rules = _business_rules(plan)
    result = run_validation_gates(
        RunValidationGatesRequest(
            dataset_id=run_context.dataset_id,
            source_dataset_version_id=apply_context.source_dataset_version_id,
            candidate_dataset_version_id=apply_context.proposed_version_name,
            candidate_artifact=state.primary_candidate_artifact.artifact_ref,
            source_artifact=state.source_artifact,
            candidate_artifact_kind=state.primary_candidate_artifact.artifact_kind,
            schema_columns=_csv_columns(storage=storage, artifact=state.source_artifact),
            numeric_columns=_numeric_feature_columns(
                storage=storage,
                artifact=state.primary_candidate_artifact.artifact_ref,
                target_column=_target_column(plan),
            ),
            business_rules=rules,
            rules_config_hash=compute_rules_config_hash(rules) if rules else None,
            pii_restricted=_pii_restricted(plan),
            split_manifest_artifact=(
                state.split_artifact.artifact_ref if state.split_artifact is not None else None
            ),
            synthetic_dataset_report=state.synthetic_dataset_report,
            synthetic_dataset_report_artifact=(
                state.synthetic_dataset_report_artifact.artifact_ref
                if state.synthetic_dataset_report_artifact is not None
                else None
            ),
            dcr_thresholds=_dcr_thresholds(plan),
            created_by_job_id=run_context.compute_run_id,
            config_hash=apply_context.config_hash,
            policy_version=apply_context.policy_versions.validation_gates_policy_version
            or "validation_gates_policy_v0",
            action_plan_id=plan.action_plan_id,
            report_id=_stable_id(
                "validation_gates_report",
                plan.action_plan_id,
                apply_context.proposed_version_name,
                state.primary_candidate_artifact.hash,
            ),
            generated_at=_generated_at(plan),
        ),
        storage=storage,
        registry=registry,
    )
    state.validation_gates_report = result.report
    state.validation_gates_report_artifact = result.report_artifact
    return result.report


def _ensure_split_manifest(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    source_artifact: ArtifactRef,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> tuple[SplitManifest, RegisteredArtifact]:
    state = apply_context.execution_state
    if state.split_manifest is not None and state.split_artifact is not None:
        return state.split_manifest, state.split_artifact
    plan = _require_plan(apply_context)
    step = _split_step(plan=plan, source_artifact=source_artifact)
    result = execute_tabular_split_action(
        ExecuteTabularSplitRequest(
            action_plan_id=plan.action_plan_id,
            step=step,
            dataset_id=run_context.dataset_id,
            source_dataset_version_id=apply_context.source_dataset_version_id,
            candidate_dataset_version_id=apply_context.proposed_version_name,
            source_artifact=source_artifact,
            created_by_job_id=run_context.compute_run_id,
            config_hash=step.config_hash,
            target_column=_target_column(plan),
            seed=step.random_seed or 42,
            split_manifest_id=_stable_id(
                "split_manifest",
                plan.action_plan_id,
                step.step_id,
                source_artifact.hash,
            ),
            generated_at=_generated_at(plan),
        ),
        storage=storage,
        registry=registry,
    )
    state.split_manifest = result.manifest
    state.split_artifact = result.split_artifact
    return result.manifest, result.split_artifact


def _split_step(*, plan: ActionPlan, source_artifact: ArtifactRef) -> ActionPlanStep:
    existing = next((step for step in plan.steps if step.type == "CREATE_SPLIT"), None)
    if existing is not None:
        return existing
    target_column = _target_column(plan)
    payload = {
        "action_plan_id": plan.action_plan_id,
        "source_artifact_hash": source_artifact.hash,
        "target_column": target_column,
    }
    config = {
        "strategy": "group_stratified",
        "target_column": target_column,
        "group_key": "customer_id_hash",
    }
    config_hash = _stable_hash({"split_config": config, **payload})
    return ActionPlanStep(
        step_id="create_split_for_apply",
        type="CREATE_SPLIT",
        depends_on=(),
        idempotency_key=_stable_hash(
            {"step_id": "create_split_for_apply", "config_hash": config_hash}
        ),
        method_id="group_stratified",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash=config_hash,
        policy_version="split_policy_v0",
        validation_gates=("schema_validation",),
        preconditions=("source_version_is_immutable",),
        input_artifacts=(source_artifact.uri,),
        output_artifact_kind=SPLIT_MANIFEST_KIND,
        config=config,
        random_seed=42,
        retry_policy=plan.steps[0].retry_policy,
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _resolve_source_artifact(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
) -> ArtifactRef:
    state = apply_context.execution_state
    if state.source_artifact is not None:
        return state.source_artifact
    plan = _require_plan(apply_context)
    if apply_context.input_artifacts:
        source = _select_source_ref(apply_context.input_artifacts, plan)
    else:
        uris = [uri for step in plan.steps for uri in step.input_artifacts]
        if not uris:
            raise ApplyRuntimeError(
                reason_code="source_artifact_required",
                message="APPLY requires an immutable source ArtifactRef or readable input URI.",
            )
        source = _artifact_ref_from_storage(
            uri=uris[0],
            apply_context=apply_context,
            run_context=run_context,
            storage=storage,
        )
    try:
        storage.get(source.uri)
    except ObjectStorageError as exc:
        raise ApplyRuntimeError(
            reason_code="source_artifact_unreadable",
            message="APPLY source artifact could not be read from object storage.",
            code=exc.code,
            details={"uri": source.uri},
        ) from exc
    state.source_artifact = source
    return source


def _select_source_ref(
    refs: tuple[ArtifactRef, ...],
    plan: ActionPlan,
) -> ArtifactRef:
    """Select the ActionPlan source ref without depending on tuple order."""
    plan_input_uris = {uri for step in plan.steps for uri in step.input_artifacts}
    for ref in refs:
        if ref.uri in plan_input_uris:
            return ref
    for ref in refs:
        if ref.kind in {"raw_transactions", "raw_dataset_archive"} or ref.kind.startswith(
            "raw_"
        ):
            return ref
    return refs[0]


def _artifact_ref_from_storage(
    *,
    uri: str,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
) -> ArtifactRef:
    info = storage.head(uri)
    kind = info.metadata.get("artifact-kind", "raw_transactions")
    schema_version = info.metadata.get("schema-version", "tabular_dataset.v1")
    parent_version = info.metadata.get(
        "dataset-version-id", apply_context.source_dataset_version_id
    )
    job_id = info.metadata.get("created-by-job-id", run_context.compute_run_id)
    config_hash = info.metadata.get("config-hash", apply_context.config_hash)
    return ArtifactRef(
        artifact_id=f"{kind}:{info.hash.removeprefix('sha256:')[:16]}",
        kind=kind,
        uri=info.uri,
        hash=info.hash,
        media_type=info.content_type,
        size_bytes=info.size_bytes,
        schema_version=schema_version,
        lineage=ArtifactLineage(
            parent_version_id=parent_version,
            job_id=job_id,
            config_hash=config_hash,
            created_at=info.updated_at,
        ),
    )


def _persist_export_manifest(
    *,
    apply_context: ApplyRunContext,
    run_context: RunContext,
    state: ApplyExecutionState,
    registry: ArtifactRegistry,
) -> RegisteredArtifact:
    if state.export_manifest_artifact is not None:
        return state.export_manifest_artifact
    payload = {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "candidate_version_artifact": (
            state.candidate_version_artifact.artifact_ref.model_dump(mode="json")
            if state.candidate_version_artifact is not None
            else None
        ),
        "candidate_status": (
            state.candidate_dataset_version.status.value
            if state.candidate_dataset_version is not None
            else "not_built"
        ),
        "validation_gates_report": (
            state.validation_gates_report_artifact.artifact_ref.model_dump(mode="json")
            if state.validation_gates_report_artifact is not None
            else None
        ),
        "tabular_export_artifacts": [
            ref.model_dump(mode="json") for ref in state.tabular_export_artifacts
        ],
        "generated_at": _generated_at(_require_plan(apply_context)).isoformat(),
    }
    return registry.save_artifact(
        artifact_kind=EXPORT_MANIFEST_KIND,
        data=json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
        artifact_format=EXPORT_MANIFEST_FORMAT,
        media_type=EXPORT_MANIFEST_MEDIA_TYPE,
        schema_version=EXPORT_MANIFEST_SCHEMA_VERSION,
        dataset_version_id=apply_context.proposed_version_name,
        created_by_job_id=run_context.compute_run_id,
        config_hash=apply_context.config_hash,
        metadata={
            "candidate-status": str(payload["candidate_status"]),
            "action-plan-id": apply_context.action_plan_id,
            "created-at": _generated_at(_require_plan(apply_context)).isoformat(),
        },
    )


def _candidate_artifact_refs(state: ApplyExecutionState) -> tuple[ArtifactRef, ...]:
    refs: list[ArtifactRef] = []
    for artifact in state.step_output_artifacts:
        refs.append(artifact.artifact_ref)
    if state.validation_gates_report_artifact is not None:
        refs.append(state.validation_gates_report_artifact.artifact_ref)
    if state.model_impact_report_artifact is not None:
        refs.append(state.model_impact_report_artifact.artifact_ref)
    return tuple(_unique_refs(refs))


def _extra_export_artifacts(state: ApplyExecutionState) -> tuple[ArtifactRef, ...]:
    refs: list[ArtifactRef] = []
    if state.candidate_version_artifact is not None:
        refs.append(state.candidate_version_artifact.artifact_ref)
    if state.validation_gates_report_artifact is not None:
        refs.append(state.validation_gates_report_artifact.artifact_ref)
    if state.model_impact_report_artifact is not None:
        refs.append(state.model_impact_report_artifact.artifact_ref)
    return tuple(_unique_refs(refs))


def _unique_refs(refs: Iterable[ArtifactRef]) -> list[ArtifactRef]:
    seen: set[str] = set()
    out: list[ArtifactRef] = []
    for ref in refs:
        if ref.uri in seen:
            continue
        seen.add(ref.uri)
        out.append(ref)
    return out


def _require_plan(apply_context: ApplyRunContext) -> ActionPlan:
    if apply_context.action_plan is None:
        raise ApplyRuntimeError(
            reason_code="action_plan_missing",
            message="APPLY requires the approved ActionPlan in ApplyRunContext.",
        )
    return apply_context.action_plan


def _target_column(plan: ActionPlan) -> str:
    for step in plan.steps:
        value = step.config.get("target_column")
        if isinstance(value, str) and value:
            return value
    return _DEFAULT_TARGET_COLUMN


def _rare_class_label(plan: ActionPlan) -> str:
    for step in plan.steps:
        value = step.config.get("rare_class_label")
        if isinstance(value, str) and value:
            return value
    return _DEFAULT_RARE_CLASS_LABEL


def _random_seed(plan: ActionPlan) -> int:
    for step in plan.steps:
        if step.random_seed is not None:
            return step.random_seed
    return 42


def _pii_restricted(plan: ActionPlan) -> bool:
    return any(step.config.get("pii_restricted") is True for step in plan.steps)


def _business_rules(plan: ActionPlan) -> tuple[BusinessRule, ...]:
    payloads: list[Mapping[str, Any]] = []
    for step in plan.steps:
        raw = step.config.get("business_rules")
        if isinstance(raw, list):
            payloads.extend(item for item in raw if isinstance(item, Mapping))
    return parse_business_rules(payloads) if payloads else ()


def _dcr_thresholds(plan: ActionPlan) -> DcrThresholds:
    for step in plan.steps:
        epsilon = step.config.get("nearest_neighbor_epsilon")
        minimum = step.config.get("minimum_dcr_threshold")
        if epsilon is not None or minimum is not None:
            return DcrThresholds(
                nearest_neighbor_epsilon=float(epsilon or 1e-6),
                minimum_dcr_threshold=float(minimum or 0.0),
            )
    return DcrThresholds()


def _csv_columns(*, storage: MinioObjectStorageAdapter, artifact: ArtifactRef) -> tuple[str, ...]:
    text = storage.get(artifact.uri).data.decode("utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        return tuple(next(reader))
    except StopIteration as exc:
        raise ApplyRuntimeError(
            reason_code="empty_tabular_source",
            message="Tabular source artifact has no CSV header.",
        ) from exc


def _row_count(*, storage: MinioObjectStorageAdapter, artifact: ArtifactRef) -> int:
    text = storage.get(artifact.uri).data.decode("utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        next(reader)
    except StopIteration:
        return 0
    return sum(1 for _ in reader)


def _numeric_feature_columns(
    *,
    storage: MinioObjectStorageAdapter,
    artifact: ArtifactRef,
    target_column: str,
) -> tuple[str, ...]:
    text = storage.get(artifact.uri).data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    if not reader.fieldnames:
        return ()
    excluded = {
        "object_id",
        "customer_id_hash",
        "case_id",
        "is_synthetic",
        "synthetic_source_split",
        target_column,
    }
    numeric: list[str] = []
    for column in reader.fieldnames:
        if column in excluded or column.endswith("_was_missing"):
            continue
        values = [row.get(column, "") for row in rows if row.get(column, "") != ""]
        if values and all(_is_float(value) for value in values):
            numeric.append(column)
    return tuple(numeric)


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _generated_at(plan: ActionPlan) -> datetime:
    return plan.created_at.astimezone(UTC)


def _stable_id(prefix: str, *parts: object) -> str:
    digest = _stable_hash({"prefix": prefix, "parts": parts}).removeprefix("sha256:")
    return f"{prefix}_{digest[:16]}"


def _stable_hash(payload: object) -> Sha256Digest:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "ACTION_PLAN_ARTIFACT_KIND",
    "ACTION_PLAN_ARTIFACT_SCHEMA_VERSION",
    "ApplyRuntimeError",
    "EXPORT_MANIFEST_KIND",
    "REMEDIATION_EXECUTION_REPORT_KIND",
    "build_final_apply_outputs",
    "ensure_model_impact_report",
    "execute_remediation_plan",
    "final_candidate_ref",
    "final_export_ref",
    "persist_action_plan_artifact",
    "synthetic_status",
]
