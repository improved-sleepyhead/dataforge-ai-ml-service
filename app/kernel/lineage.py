"""Builder for the ``lineage.json`` artifact and compute audit events.

PRD §20.1 / §20.2 / §20.3 require the ML service to:

- emit a machine-readable lineage record linking every output to its
  inputs (parent_version_id, job_id, input/output artifact hashes,
  algorithm/plugin versions, config_hash, policy_version, random_seed);
- emit ML compute audit events to the platform with safe metadata
  only (no raw rows, raw text, raw PII, signed bodies);
- expose an OpenLineage-compatible shape so future Job/Run/Dataset
  facets work without a contract change.

The builder consumes the ``CandidateDatasetVersion`` (which already
carries the per-step ActionPlan summary, policy versions and lineage
hashes) plus optional ``ExportPackage``/source artifact refs, and
persists ``lineage_report.v1`` JSON via ``ArtifactRegistry``. When a
fake/in-process ``FakePlatformMetadataClient`` is supplied, it also
records ``LINEAGE_REPORT_BUILT`` and (when an export package is
present) ``EXPORT_PACKAGE_BUILT`` audit events.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.adapters import (
    ArtifactRegistry,
    AuditEventType,
    FakePlatformMetadataClient,
    PlatformAuditEvent,
    RegisteredArtifact,
)
from app.domain import (
    LINEAGE_REPORT_SCHEMA_VERSION,
    ArtifactRef,
    CandidateActionStepSummary,
    CandidateDatasetVersion,
    ErrorCode,
    ExportPackage,
    LineageAlgorithm,
    LineagePolicyVersions,
    LineageReport,
    OpenLineageDataset,
    OpenLineageEnvelope,
    OpenLineageRun,
)
from app.domain.common import NonEmptyStr, Sha256Digest

LINEAGE_REPORT_KIND = "lineage_report"
LINEAGE_REPORT_FORMAT = "json"
LINEAGE_REPORT_MEDIA_TYPE = "application/json"

OPENLINEAGE_NAMESPACE = "dataforgeai-ml-service"
OPENLINEAGE_JOB_NAME = "apply_selected_actions"


class LineageBuilderError(ValueError):
    """Raised when the lineage-report builder cannot run safely."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.INVALID_JOB_PAYLOAD,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class BuildLineageReportRequest(BaseModel):
    """Inputs for :func:`build_lineage_report`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    candidate_dataset_version: CandidateDatasetVersion
    candidate_version_artifact: ArtifactRef | None = None
    export_package: ExportPackage | None = None
    export_package_artifact: ArtifactRef | None = None
    input_artifact_refs: tuple[ArtifactRef, ...] = ()
    output_artifact_refs: tuple[ArtifactRef, ...] = ()
    privacy_policy_version: NonEmptyStr | None = None
    export_policy_version: NonEmptyStr | None = None
    job_started_at: datetime | None = None
    job_completed_at: datetime | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    lineage_report_id: str | None = None
    generated_at: datetime | None = None
    emit_export_package_audit_event: bool = True


@dataclass(frozen=True)
class BuildLineageReportResult:
    """Persisted lineage_report artifact + emitted audit events."""

    lineage_report: LineageReport
    lineage_report_artifact: RegisteredArtifact
    audit_events: tuple[PlatformAuditEvent, ...]


def build_lineage_report(
    request: BuildLineageReportRequest,
    *,
    registry: ArtifactRegistry,
    platform_client: FakePlatformMetadataClient | None = None,
) -> BuildLineageReportResult:
    """Assemble + persist a ``lineage.json`` artifact and audit events.

    The function consumes ``CandidateDatasetVersion`` (PRD §20.1
    fields are derived from its lineage envelope and action_plan
    steps) and optional ``ExportPackage``/artifact refs. It returns
    the parsed ``LineageReport``, the registered artifact and any
    audit events emitted to the fake/in-process platform client.
    """
    candidate = request.candidate_dataset_version
    if not candidate.action_plan_steps:
        raise LineageBuilderError(
            reason_code="candidate_action_plan_steps_missing",
            message=(
                "lineage builder requires CandidateDatasetVersion.action_plan_steps "
                "to attribute algorithm/plugin versions."
            ),
        )

    algorithms = tuple(_build_algorithms(candidate.action_plan_steps))
    primary_step = candidate.action_plan_steps[0]
    policy_versions = LineagePolicyVersions(
        profile_policy_version=candidate.policy_versions.profile_policy_version,
        decision_policy_version=candidate.policy_versions.decision_policy_version,
        score_policy_version=candidate.policy_versions.score_policy_version,
        method_policy_version=candidate.policy_versions.method_policy_version,
        validation_gates_policy_version=(
            candidate.policy_versions.validation_gates_policy_version
        ),
        privacy_policy_version=request.privacy_policy_version,
        export_policy_version=request.export_policy_version,
    )

    output_hashes = _resolve_output_hashes(
        candidate=candidate,
        export_package=request.export_package,
        explicit_output_refs=request.output_artifact_refs,
    )
    input_hashes = _resolve_input_hashes(
        candidate=candidate,
        explicit_input_refs=request.input_artifact_refs,
    )

    started_at = request.job_started_at or candidate.proposed_at
    completed_at = (
        request.job_completed_at
        or (
            request.export_package.created_at
            if request.export_package is not None
            else candidate.proposed_at
        )
    )

    openlineage = _build_openlineage(
        candidate=candidate,
        export_package=request.export_package,
        run_id=candidate.lineage.created_by_job_id,
        started_at=started_at,
        completed_at=completed_at,
        algorithms=algorithms,
        input_refs=request.input_artifact_refs,
        output_refs=request.output_artifact_refs,
    )

    report_id = (
        request.lineage_report_id or f"lineage_report_{uuid.uuid4().hex[:16]}"
    )
    report = LineageReport(
        lineage_report_id=report_id,
        lineage_schema_version=LINEAGE_REPORT_SCHEMA_VERSION,
        organization_id=request.organization_id,
        project_id=request.project_id,
        dataset_id=request.dataset_id,
        parent_version_id=candidate.lineage.parent_version_id,
        output_dataset_version_id=candidate.lineage.proposed_version_name,
        job_id=candidate.lineage.created_by_job_id,
        input_artifact_hashes=tuple(_ordered_unique(input_hashes)),
        output_artifact_hashes=tuple(_ordered_unique(output_hashes)),
        algorithm_name=primary_step.method_id,
        algorithm_version=primary_step.plugin_version,
        config_hash=candidate.lineage.config_hash,
        policy_version=candidate.policy_versions.method_policy_version,
        random_seed=primary_step.random_seed,
        action_plan_id=candidate.lineage.action_plan_id,
        decision_report_id=candidate.lineage.decision_report_id,
        candidate_version_artifact=request.candidate_version_artifact,
        export_package_artifact=request.export_package_artifact,
        algorithms=algorithms,
        policy_versions=policy_versions,
        input_artifact_refs=request.input_artifact_refs,
        output_artifact_refs=request.output_artifact_refs,
        openlineage=openlineage,
        created_at=request.generated_at or datetime.now(UTC),
    )

    payload = _serialize(report)
    metadata = {
        "lineage-report-id": report_id,
        "parent-version-id": report.parent_version_id,
        "output-version-id": report.output_dataset_version_id,
        "job-id": report.job_id,
        "algorithm-name": report.algorithm_name,
        "algorithm-version": report.algorithm_version,
        "input-artifact-count": str(len(report.input_artifact_hashes)),
        "output-artifact-count": str(len(report.output_artifact_hashes)),
    }
    artifact = registry.save_artifact(
        artifact_kind=LINEAGE_REPORT_KIND,
        data=payload,
        artifact_format=LINEAGE_REPORT_FORMAT,
        media_type=LINEAGE_REPORT_MEDIA_TYPE,
        schema_version=LINEAGE_REPORT_SCHEMA_VERSION,
        dataset_version_id=report.output_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata=metadata,
    )

    audit_events: list[PlatformAuditEvent] = []
    if platform_client is not None:
        audit_events.append(
            _emit_lineage_event(
                request=request,
                report=report,
                artifact_ref=artifact.artifact_ref,
                platform_client=platform_client,
            )
        )
        if (
            request.export_package is not None
            and request.emit_export_package_audit_event
        ):
            audit_events.append(
                _emit_export_event(
                    request=request,
                    export_package=request.export_package,
                    export_package_artifact=request.export_package_artifact,
                    lineage_artifact_ref=artifact.artifact_ref,
                    platform_client=platform_client,
                )
            )

    return BuildLineageReportResult(
        lineage_report=report,
        lineage_report_artifact=artifact,
        audit_events=tuple(audit_events),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_algorithms(
    steps: Iterable[CandidateActionStepSummary],
) -> list[LineageAlgorithm]:
    return [
        LineageAlgorithm(
            step_id=step.step_id,
            step_type=step.step_type,
            algorithm_name=step.method_id,
            algorithm_version=step.plugin_version,
            plugin_id=step.plugin_id,
            plugin_version=step.plugin_version,
            config_hash=step.config_hash,
            random_seed=step.random_seed,
        )
        for step in steps
    ]


def _resolve_input_hashes(
    *,
    candidate: CandidateDatasetVersion,
    explicit_input_refs: tuple[ArtifactRef, ...],
) -> tuple[Sha256Digest, ...]:
    explicit = tuple(ref.hash for ref in explicit_input_refs)
    if explicit:
        return explicit
    return candidate.lineage.input_artifact_hashes


def _resolve_output_hashes(
    *,
    candidate: CandidateDatasetVersion,
    export_package: ExportPackage | None,
    explicit_output_refs: tuple[ArtifactRef, ...],
) -> list[Sha256Digest]:
    hashes: list[Sha256Digest] = []
    if explicit_output_refs:
        hashes.extend(ref.hash for ref in explicit_output_refs)
    else:
        hashes.extend(candidate.lineage.output_artifact_hashes)
    if export_package is not None:
        hashes.extend(artifact.hash for artifact in export_package.artifacts)
    return hashes


def _build_openlineage(
    *,
    candidate: CandidateDatasetVersion,
    export_package: ExportPackage | None,
    run_id: NonEmptyStr,
    started_at: datetime,
    completed_at: datetime,
    algorithms: tuple[LineageAlgorithm, ...],
    input_refs: tuple[ArtifactRef, ...],
    output_refs: tuple[ArtifactRef, ...],
) -> OpenLineageEnvelope:
    inputs: list[OpenLineageDataset] = []
    if input_refs:
        for ref in input_refs:
            inputs.append(
                _openlineage_dataset(
                    name=_dataset_name_from_ref(ref),
                    facets={"artifact_hash": ref.hash, "kind": ref.kind},
                )
            )
    else:
        inputs.append(
            _openlineage_dataset(
                name=candidate.lineage.parent_version_id,
                facets={
                    "kind": "dataset_version",
                    "version_id": candidate.lineage.parent_version_id,
                },
            )
        )
    outputs: list[OpenLineageDataset] = [
        _openlineage_dataset(
            name=candidate.lineage.proposed_version_name,
            facets={
                "kind": "candidate_dataset_version",
                "candidate_version_id": candidate.candidate_version_id,
                "status": candidate.status.value,
            },
        )
    ]
    if export_package is not None:
        outputs.append(
            _openlineage_dataset(
                name=f"{export_package.export_package_id}",
                facets={
                    "kind": "export_package",
                    "status": export_package.status.value,
                    "included": export_package.object_counts.included,
                    "blocked": export_package.object_counts.blocked,
                    "excluded": export_package.object_counts.excluded,
                },
            )
        )
    if output_refs:
        for ref in output_refs:
            outputs.append(
                _openlineage_dataset(
                    name=_dataset_name_from_ref(ref),
                    facets={"artifact_hash": ref.hash, "kind": ref.kind},
                )
            )
    run = OpenLineageRun(
        run_id=run_id,
        job_namespace=OPENLINEAGE_NAMESPACE,
        job_name=OPENLINEAGE_JOB_NAME,
        started_at=started_at,
        completed_at=completed_at,
        facets={
            "step_count": len(algorithms),
            "policy_version": candidate.policy_versions.method_policy_version,
            "decision_policy_version": (
                candidate.policy_versions.decision_policy_version
            ),
        },
    )
    return OpenLineageEnvelope(
        run=run,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def _openlineage_dataset(
    *,
    name: str,
    facets: dict[str, str | int | float | bool],
) -> OpenLineageDataset:
    return OpenLineageDataset(
        namespace=OPENLINEAGE_NAMESPACE,
        name=name,
        facets=facets,
    )


def _dataset_name_from_ref(ref: ArtifactRef) -> str:
    if ref.uri.startswith("s3://"):
        # Use the path portion as the dataset name; this is stable
        # across runs and never includes raw payloads.
        return ref.uri[len("s3://") :]
    return ref.artifact_id


def _emit_lineage_event(
    *,
    request: BuildLineageReportRequest,
    report: LineageReport,
    artifact_ref: ArtifactRef,
    platform_client: FakePlatformMetadataClient,
) -> PlatformAuditEvent:
    metadata: dict[str, str | int | float | bool] = {
        "lineage_report_id": report.lineage_report_id,
        "lineage_artifact_uri": artifact_ref.uri,
        "lineage_artifact_hash": artifact_ref.hash,
        "parent_version_id": report.parent_version_id,
        "output_dataset_version_id": report.output_dataset_version_id,
        "job_id": report.job_id,
        "algorithm_name": report.algorithm_name,
        "algorithm_version": report.algorithm_version,
        "input_artifact_count": len(report.input_artifact_hashes),
        "output_artifact_count": len(report.output_artifact_hashes),
        "config_hash": report.config_hash,
        "policy_version": report.policy_version,
    }
    if report.action_plan_id is not None:
        metadata["action_plan_id"] = report.action_plan_id
    if report.decision_report_id is not None:
        metadata["decision_report_id"] = report.decision_report_id
    audit_event = PlatformAuditEvent(
        audit_event_id=f"audit_{uuid.uuid4().hex[:16]}",
        event_type=AuditEventType.LINEAGE_REPORT_BUILT,
        organization_id=request.organization_id,
        project_id=request.project_id,
        metadata=metadata,
    )
    return platform_client.record_audit_event(audit_event)


def _emit_export_event(
    *,
    request: BuildLineageReportRequest,
    export_package: ExportPackage,
    export_package_artifact: ArtifactRef | None,
    lineage_artifact_ref: ArtifactRef,
    platform_client: FakePlatformMetadataClient,
) -> PlatformAuditEvent:
    metadata: dict[str, str | int | float | bool] = {
        "export_package_id": export_package.export_package_id,
        "export_status": export_package.status.value,
        "version_id": export_package.version_id,
        "source_version_id": export_package.source_version_id,
        "lineage_artifact_uri": lineage_artifact_ref.uri,
        "lineage_artifact_hash": lineage_artifact_ref.hash,
        "included": export_package.object_counts.included,
        "blocked": export_package.object_counts.blocked,
        "excluded": export_package.object_counts.excluded,
        "blocker_reason_count": len(export_package.blocked_reason_codes),
    }
    if export_package_artifact is not None:
        metadata["export_package_artifact_uri"] = export_package_artifact.uri
        metadata["export_package_artifact_hash"] = export_package_artifact.hash
    audit_event = PlatformAuditEvent(
        audit_event_id=f"audit_{uuid.uuid4().hex[:16]}",
        event_type=AuditEventType.EXPORT_PACKAGE_BUILT,
        organization_id=request.organization_id,
        project_id=request.project_id,
        metadata=metadata,
    )
    return platform_client.record_audit_event(audit_event)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _serialize(report: LineageReport) -> bytes:
    return json.dumps(
        report.model_dump(mode="json"), sort_keys=True, indent=2
    ).encode("utf-8")


__all__ = [
    "BuildLineageReportRequest",
    "BuildLineageReportResult",
    "LINEAGE_REPORT_FORMAT",
    "LINEAGE_REPORT_KIND",
    "LINEAGE_REPORT_MEDIA_TYPE",
    "LineageBuilderError",
    "OPENLINEAGE_JOB_NAME",
    "OPENLINEAGE_NAMESPACE",
    "build_lineage_report",
]
