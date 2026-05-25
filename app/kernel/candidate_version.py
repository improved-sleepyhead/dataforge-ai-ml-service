"""Builder for proposed candidate dataset version artifacts.

This module consumes the artifacts produced by an
``APPLY_SELECTED_ACTIONS`` Dagster run and assembles a
``CandidateDatasetVersion`` artifact that the platform can pick up. The
builder:

- never writes the final lifecycle status into the platform database;
- never overwrites a previously persisted candidate-version artifact —
  re-running with the same input hashes returns the existing record;
- emits a ``DATASET_VERSION_PROPOSED`` audit event so the platform can
  enqueue its approval workflow.

Synthetic candidates require complete synthetic metadata: method,
plugin, plugin version, random seed, config hash, source split or
cohort, source object ids when safe, generated count, sampling
strategy, policy version, validation report ref and (when available)
the model-impact report ref. The acceptance criteria for TASK-049
verify these fields explicitly.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import (
    ArtifactRegistry,
    AuditEventType,
    FakePlatformMetadataClient,
    PlatformAuditEvent,
    RegisteredArtifact,
)
from app.domain import (
    CANDIDATE_DATASET_VERSION_SCHEMA_VERSION,
    ActionPlan,
    ArtifactRef,
    CandidateActionStepSummary,
    CandidateDatasetVersion,
    CandidateDatasetVersionLineage,
    CandidatePolicyVersions,
    CandidateVersionStatus,
    ErrorCode,
    SyntheticCandidateMetadata,
    SyntheticDatasetReport,
    ValidationGatesReport,
    ValidationGateStatus,
)
from app.domain.common import NonEmptyStr, Sha256Digest

CANDIDATE_DATASET_VERSION_KIND = "candidate_dataset_version"
CANDIDATE_DATASET_VERSION_FORMAT = "json"
CANDIDATE_DATASET_VERSION_MEDIA_TYPE = "application/json"

_SOURCE_OBJECT_ID_LIMIT = 25


class CandidateVersionBuilderError(ValueError):
    """Raised when the candidate-version builder cannot assemble safely."""

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


class BuildCandidateVersionRequest(BaseModel):
    """Inputs for :func:`build_candidate_dataset_version`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    proposed_version_name: NonEmptyStr
    action_plan: ActionPlan
    policy_versions: CandidatePolicyVersions
    decision_report_id: NonEmptyStr | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    candidate_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    primary_dataset_artifact: ArtifactRef
    validation_gates_report: ValidationGatesReport | None = None
    validation_gates_report_artifact: ArtifactRef | None = None
    synthetic_dataset_report: SyntheticDatasetReport | None = None
    synthetic_dataset_report_artifact: ArtifactRef | None = None
    synthetic_validation_report_artifact: ArtifactRef | None = None
    model_impact_report_artifact: ArtifactRef | None = None
    candidate_version_id: str | None = None
    proposed_at: datetime | None = None


@dataclass(frozen=True)
class BuildCandidateVersionResult:
    """Persisted candidate-version metadata + emitted audit event."""

    candidate_version: CandidateDatasetVersion
    candidate_version_artifact: RegisteredArtifact
    audit_event: PlatformAuditEvent | None


def build_candidate_dataset_version(
    request: BuildCandidateVersionRequest,
    *,
    registry: ArtifactRegistry,
    platform_client: FakePlatformMetadataClient | None = None,
) -> BuildCandidateVersionResult:
    """Assemble + persist a proposed candidate dataset version artifact.

    The function computes the candidate status from the validation
    gates summary (``validation_failed`` flips the candidate to
    ``BLOCKED``; missing report keeps it ``FAILED``; otherwise the
    candidate is ``PROPOSED``), records it in object storage as
    ``candidate_dataset_version.v1``, and (if a fake/in-process
    platform client is supplied) emits a ``DATASET_VERSION_PROPOSED``
    audit event with safe metadata only.
    """
    status = _resolve_status(request.validation_gates_report)
    block_export, block_model_evaluation, block_training = _resolve_blocks(
        request.validation_gates_report
    )
    blocker_reason_codes = _resolve_blocker_reason_codes(request.validation_gates_report)
    summary = _validation_summary(request.validation_gates_report)
    synthetic_metadata = _build_synthetic_metadata(request=request, status=status)
    steps = _summarize_steps(request.action_plan)
    lineage = CandidateDatasetVersionLineage(
        organization_id=request.organization_id,
        project_id=request.project_id,
        dataset_id=request.dataset_id,
        parent_version_id=request.parent_version_id,
        proposed_version_name=request.proposed_version_name,
        action_plan_id=request.action_plan.action_plan_id,
        decision_report_id=request.decision_report_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        input_artifact_hashes=tuple(
            artifact.hash for artifact in request.source_artifacts
        ),
        output_artifact_hashes=tuple(
            _ordered_unique(artifact.hash for artifact in request.candidate_artifacts)
        ),
    )

    candidate_version_id = (
        request.candidate_version_id
        or f"candidate_dataset_version_{uuid.uuid4().hex[:16]}"
    )
    candidate_version = CandidateDatasetVersion(
        candidate_version_id=candidate_version_id,
        candidate_version_schema_version=CANDIDATE_DATASET_VERSION_SCHEMA_VERSION,
        status=status,
        policy_versions=request.policy_versions,
        lineage=lineage,
        candidate_artifacts=request.candidate_artifacts,
        primary_dataset_artifact=request.primary_dataset_artifact,
        validation_gates_report=request.validation_gates_report_artifact,
        validation_gates_summary=summary,
        block_export=block_export,
        block_model_evaluation=block_model_evaluation,
        block_training=block_training,
        blocker_reason_codes=blocker_reason_codes,
        action_plan_steps=steps,
        synthetic_metadata=synthetic_metadata,
        proposed_at=request.proposed_at or datetime.now(UTC),
    )

    payload = _serialize(candidate_version)
    metadata_label = {
        "candidate-version-id": candidate_version_id,
        "candidate-status": status.value,
        "block-export": "true" if block_export else "false",
        "block-model-evaluation": "true" if block_model_evaluation else "false",
        "block-training": "true" if block_training else "false",
        "action-plan-id": request.action_plan.action_plan_id,
        "parent-version-id": request.parent_version_id,
        "proposed-version-name": request.proposed_version_name,
    }
    if synthetic_metadata is not None:
        metadata_label["synthetic-method-id"] = synthetic_metadata.method_id
        metadata_label["synthetic-plugin-id"] = synthetic_metadata.plugin_id

    candidate_artifact = registry.save_artifact(
        artifact_kind=CANDIDATE_DATASET_VERSION_KIND,
        data=payload,
        artifact_format=CANDIDATE_DATASET_VERSION_FORMAT,
        media_type=CANDIDATE_DATASET_VERSION_MEDIA_TYPE,
        schema_version=CANDIDATE_DATASET_VERSION_SCHEMA_VERSION,
        dataset_version_id=request.proposed_version_name,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata=metadata_label,
    )

    audit_event = _emit_audit_event(
        candidate_version=candidate_version,
        candidate_artifact_ref=candidate_artifact.artifact_ref,
        request=request,
        platform_client=platform_client,
    )
    return BuildCandidateVersionResult(
        candidate_version=candidate_version,
        candidate_version_artifact=candidate_artifact,
        audit_event=audit_event,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_status(
    report: ValidationGatesReport | None,
) -> CandidateVersionStatus:
    if report is None:
        return CandidateVersionStatus.FAILED
    if report.candidate_status.value == "validation_failed":
        return CandidateVersionStatus.BLOCKED
    if report.candidate_status.value == "review_required":
        return CandidateVersionStatus.REVIEW_REQUIRED
    return CandidateVersionStatus.PROPOSED


def _resolve_blocks(
    report: ValidationGatesReport | None,
) -> tuple[bool, bool, bool]:
    if report is None:
        return True, True, True
    return report.block_export, report.block_model_evaluation, report.block_training


def _resolve_blocker_reason_codes(
    report: ValidationGatesReport | None,
) -> tuple[NonEmptyStr, ...]:
    if report is None:
        return ()
    return tuple(
        gate.reason_code
        for gate in report.gates
        if gate.status is ValidationGateStatus.FAILED
        and gate.severity.value == "blocker"
    )


def _validation_summary(
    report: ValidationGatesReport | None,
) -> dict[str, str | bool | int | float]:
    if report is None:
        return {}
    return {
        "overall_status": report.overall_status.value,
        "candidate_status": report.candidate_status.value,
        "raw_artifact_unchanged": report.raw_artifact_unchanged,
        "blocker_present": report.blocker_present,
    }


def _summarize_steps(plan: ActionPlan) -> tuple[CandidateActionStepSummary, ...]:
    return tuple(
        CandidateActionStepSummary(
            step_id=step.step_id,
            step_type=step.type,
            method_id=step.method_id,
            plugin_id=step.plugin_id,
            plugin_version=step.plugin_version,
            config_hash=step.config_hash,
            output_artifact_kind=step.output_artifact_kind,
            random_seed=step.random_seed,
        )
        for step in plan.steps
    )


def _build_synthetic_metadata(
    *,
    request: BuildCandidateVersionRequest,
    status: CandidateVersionStatus,
) -> SyntheticCandidateMetadata | None:
    if request.synthetic_dataset_report is None:
        return None
    if request.synthetic_dataset_report_artifact is None:
        raise CandidateVersionBuilderError(
            reason_code="synthetic_dataset_report_artifact_required",
            message=(
                "synthetic_dataset_report_artifact must be provided when "
                "synthetic_dataset_report is supplied so the candidate "
                "version artifact can reference the immutable report."
            ),
        )
    validation_artifact = (
        request.synthetic_validation_report_artifact
        or request.validation_gates_report_artifact
    )
    if validation_artifact is None:
        raise CandidateVersionBuilderError(
            reason_code="synthetic_validation_report_required",
            message=(
                "synthetic candidate metadata must reference a validation "
                "report artifact (synthetic_validation_report_artifact or "
                "validation_gates_report_artifact)."
            ),
        )
    report = request.synthetic_dataset_report
    source_object_ids: list[str] = list(
        _ordered_unique(
            entry.seed_object_id
            for entry in report.sample_lineage
            if entry.seed_object_id
        )
    )
    full_count = len(source_object_ids)
    truncated = False
    if full_count > _SOURCE_OBJECT_ID_LIMIT:
        truncated = True
        source_object_ids = source_object_ids[:_SOURCE_OBJECT_ID_LIMIT]
    cohort = report.rare_class_label
    metadata = SyntheticCandidateMetadata(
        method_id=report.method.value,
        plugin_id="dataforge.tabular",
        plugin_version=report.method_version,
        random_seed=report.random_seed,
        config_hash=request.config_hash,
        source_split=report.source_split,
        source_cohort=cohort,
        source_object_ids=tuple(source_object_ids),
        source_object_ids_truncated=truncated,
        full_source_object_ids_count=full_count,
        generated_count=report.generated_count,
        sampling_strategy=float(report.sampling_strategy),
        policy_version=request.policy_versions.method_policy_version,
        validation_report=validation_artifact,
        model_impact_report=request.model_impact_report_artifact,
        synthetic_dataset_report=request.synthetic_dataset_report_artifact,
    )
    if status is CandidateVersionStatus.PROPOSED and metadata.generated_count == 0:
        # A proposed synthetic candidate must actually have generated
        # rows; surface this as an explicit precondition error rather
        # than silently proposing an empty synthetic candidate.
        raise CandidateVersionBuilderError(
            reason_code="synthetic_candidate_has_no_generated_rows",
            message="synthetic candidate metadata reports zero generated rows",
        )
    return metadata


def _emit_audit_event(
    *,
    candidate_version: CandidateDatasetVersion,
    candidate_artifact_ref: ArtifactRef,
    request: BuildCandidateVersionRequest,
    platform_client: FakePlatformMetadataClient | None,
) -> PlatformAuditEvent | None:
    if platform_client is None:
        return None
    audit_event_id = f"audit_{uuid.uuid4().hex[:16]}"
    metadata: dict[str, str | int | float | bool] = {
        "candidate_version_id": candidate_version.candidate_version_id,
        "candidate_version_artifact_uri": candidate_artifact_ref.uri,
        "candidate_version_artifact_hash": candidate_artifact_ref.hash,
        "candidate_status": candidate_version.status.value,
        "action_plan_id": candidate_version.lineage.action_plan_id,
        "parent_version_id": candidate_version.lineage.parent_version_id,
        "proposed_version_name": candidate_version.lineage.proposed_version_name,
        "block_export": candidate_version.block_export,
        "block_model_evaluation": candidate_version.block_model_evaluation,
        "block_training": candidate_version.block_training,
    }
    if candidate_version.synthetic_metadata is not None:
        metadata["synthetic_method_id"] = candidate_version.synthetic_metadata.method_id
        metadata["synthetic_generated_count"] = (
            candidate_version.synthetic_metadata.generated_count
        )
        metadata["synthetic_random_seed"] = (
            candidate_version.synthetic_metadata.random_seed
        )
    audit_event = PlatformAuditEvent(
        audit_event_id=audit_event_id,
        event_type=AuditEventType.DATASET_VERSION_PROPOSED,
        organization_id=request.organization_id,
        project_id=request.project_id,
        metadata=metadata,
    )
    return platform_client.record_audit_event(audit_event)


def _ordered_unique(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _serialize(version: CandidateDatasetVersion) -> bytes:
    return json.dumps(version.model_dump(mode="json"), sort_keys=True, indent=2).encode(
        "utf-8"
    )


__all__ = [
    "BuildCandidateVersionRequest",
    "BuildCandidateVersionResult",
    "CANDIDATE_DATASET_VERSION_FORMAT",
    "CANDIDATE_DATASET_VERSION_KIND",
    "CANDIDATE_DATASET_VERSION_MEDIA_TYPE",
    "CandidateVersionBuilderError",
    "build_candidate_dataset_version",
]
