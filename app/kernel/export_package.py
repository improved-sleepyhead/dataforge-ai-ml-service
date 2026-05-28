"""Export readiness gates and ``ExportPackage`` artifact builder (TASK-053).

Per PRD §26.1 / §29.14 the export layer must:

- never export blocked objects;
- export only from immutable candidate/source artifacts;
- include a manifest, tabular/text outputs, dataset_card.md, lineage.json,
  dataforge_report.json and review_queue.jsonl where available;
- mark the export ``BLOCKED`` with stable reason codes when any
  readiness gate fails.

This module:

- evaluates a deterministic set of export readiness gates against
  normalized contract reports (CandidateDatasetVersion,
  ValidationGatesReport, ModelImpactReport, TextOcrReport);
- assembles the ExportPackage artifact including only the artifact
  refs the caller passes in (manifest, tabular/text outputs,
  dataset_card, lineage, dataforge_report, review_queue);
- excludes blocked objects from the published object_counts;
- persists the ExportPackage as an immutable JSON artifact via
  ``ArtifactRegistry``.

The builder never reads raw rows or PII payloads — only counts,
artifact refs and ids. Failed validation must not promote a candidate;
the builder respects this by surfacing every gate failure as
``status=BLOCKED`` with explicit reason codes and refusing to include
the export artifacts when ``raise_on_blocked=True`` is requested.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    ArtifactRef,
    CandidateDatasetVersion,
    ErrorCode,
    ExportObjectCounts,
    ExportPackage,
    ExportPackageLineage,
    ExportPackageStatus,
    GateStatus,
    ModelImpactReport,
    ModelImpactVerdict,
    TextOcrReport,
    ValidationGateResult,
    ValidationGatesReport,
)
from app.domain.common import NonEmptyStr, Sha256Digest

EXPORT_PACKAGE_KIND = "export_package"
EXPORT_PACKAGE_FORMAT = "json"
EXPORT_PACKAGE_MEDIA_TYPE = "application/json"
EXPORT_PACKAGE_SCHEMA_VERSION = "export_package.v1"


# Stable export readiness gate names. Keep names short and stable so
# the platform UI can map them to copy without re-reading code.
GATE_CANDIDATE_VALIDATION = "candidate_validation_passed"
GATE_RAW_IMMUTABILITY = "raw_artifact_immutability"
GATE_VALIDATION_GATES = "validation_gates_blocker_check"
GATE_PRIVACY_CHECK = "privacy_check"
GATE_BLOCKED_OBJECTS_EXCLUDED = "blocked_objects_excluded"
GATE_MODEL_IMPACT = "model_impact_eligibility"
GATE_REQUIRED_ARTIFACTS = "required_artifacts_present"


class ExportPackageBuilderError(ValueError):
    """Raised when the export-package builder cannot run safely."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.EXPORT_BLOCKED,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class BuildExportPackageRequest(BaseModel):
    """Inputs for :func:`build_export_package`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    candidate_dataset_version: CandidateDatasetVersion
    validation_gates_report: ValidationGatesReport | None = None
    model_impact_report: ModelImpactReport | None = None
    text_ocr_report: TextOcrReport | None = None
    decision_report_id: NonEmptyStr
    export_manifest_artifact: ArtifactRef
    tabular_artifacts: tuple[ArtifactRef, ...] = ()
    text_artifacts: tuple[ArtifactRef, ...] = ()
    dataset_card_artifact: ArtifactRef | None = None
    lineage_artifact: ArtifactRef | None = None
    dataforge_report_artifact: ArtifactRef | None = None
    review_queue_artifact: ArtifactRef | None = None
    extra_artifacts: tuple[ArtifactRef, ...] = ()
    included_object_count: int = Field(ge=0)
    blocked_object_count: int = Field(ge=0, default=0)
    excluded_object_count: int = Field(ge=0, default=0)
    require_model_impact_eligibility: bool = False
    require_dataset_card: bool = True
    require_lineage_artifact: bool = True
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    export_package_id: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class BuildExportPackageResult:
    """Persisted export-package artifact + parsed model + readiness gates."""

    export_package: ExportPackage
    package_artifact: RegisteredArtifact
    blocked: bool
    blocker_reason_codes: tuple[str, ...]


def build_export_package(
    request: BuildExportPackageRequest,
    *,
    registry: ArtifactRegistry,
) -> BuildExportPackageResult:
    """Evaluate readiness gates and persist an ``ExportPackage`` artifact.

    The builder always persists the package — even when ``BLOCKED`` —
    so the platform/audit can surface the blocker with evidence. The
    builder never silently drops blocked-state evidence.
    """
    gates, blocker_codes = _evaluate_export_gates(request)
    artifacts = _resolve_artifacts(request, blocked=bool(blocker_codes))
    object_counts = ExportObjectCounts(
        included=request.included_object_count if not blocker_codes else 0,
        blocked=request.blocked_object_count,
        excluded=request.excluded_object_count,
    )
    status = (
        ExportPackageStatus.BLOCKED if blocker_codes else ExportPackageStatus.READY
    )

    package_id = (
        request.export_package_id or f"export_package_{uuid.uuid4().hex[:16]}"
    )
    candidate = request.candidate_dataset_version
    package = ExportPackage(
        export_package_id=package_id,
        export_schema_version=EXPORT_PACKAGE_SCHEMA_VERSION,
        dataset_id=request.dataset_id,
        version_id=candidate.lineage.proposed_version_name,
        source_version_id=candidate.lineage.parent_version_id,
        created_by_job_id=request.created_by_job_id,
        status=status,
        artifacts=tuple(artifacts),
        validation_gates=tuple(gates),
        object_counts=object_counts,
        blocked_reason_codes=tuple(blocker_codes),
        lineage=ExportPackageLineage(
            parent_version_id=candidate.lineage.parent_version_id,
            action_plan_id=candidate.lineage.action_plan_id,
            decision_report_id=request.decision_report_id,
            config_hash=request.config_hash,
        ),
        created_at=request.created_at or datetime.now(UTC),
    )
    payload = _serialize(package)
    metadata = {
        "export-package-id": package_id,
        "export-status": status.value,
        "version-id": package.version_id,
        "source-version-id": package.source_version_id,
        "blocker-count": str(len(blocker_codes)),
        "created-at": package.created_at.isoformat(),
    }
    artifact = registry.save_artifact(
        artifact_kind=EXPORT_PACKAGE_KIND,
        data=payload,
        artifact_format=EXPORT_PACKAGE_FORMAT,
        media_type=EXPORT_PACKAGE_MEDIA_TYPE,
        schema_version=EXPORT_PACKAGE_SCHEMA_VERSION,
        dataset_version_id=package.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata=metadata,
    )
    return BuildExportPackageResult(
        export_package=package,
        package_artifact=artifact,
        blocked=bool(blocker_codes),
        blocker_reason_codes=tuple(blocker_codes),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _evaluate_export_gates(
    request: BuildExportPackageRequest,
) -> tuple[list[ValidationGateResult], list[str]]:
    candidate = request.candidate_dataset_version
    gates: list[ValidationGateResult] = []
    blocker_codes: list[str] = []

    # Gate 1 — candidate validation must allow export.
    if candidate.block_export:
        reasons = (
            tuple(candidate.blocker_reason_codes)
            or ("candidate_block_export_flag",)
        )
        blocker_codes.extend(reasons)
        gates.append(
            ValidationGateResult(
                name=GATE_CANDIDATE_VALIDATION,
                status=GateStatus.FAILED,
                reason_codes=reasons,
            )
        )
    else:
        gates.append(
            ValidationGateResult(
                name=GATE_CANDIDATE_VALIDATION,
                status=GateStatus.PASSED,
                reason_codes=(),
            )
        )

    # Gate 2 — raw artifact immutability.
    raw_unchanged = _raw_artifact_unchanged(
        candidate=candidate,
        gates_report=request.validation_gates_report,
    )
    if raw_unchanged is False:
        blocker_codes.append("raw_artifact_changed")
        gates.append(
            ValidationGateResult(
                name=GATE_RAW_IMMUTABILITY,
                status=GateStatus.FAILED,
                reason_codes=("raw_artifact_changed",),
            )
        )
    elif raw_unchanged is True:
        gates.append(
            ValidationGateResult(
                name=GATE_RAW_IMMUTABILITY,
                status=GateStatus.PASSED,
                reason_codes=(),
            )
        )
    else:
        gates.append(
            ValidationGateResult(
                name=GATE_RAW_IMMUTABILITY,
                status=GateStatus.NOT_APPLICABLE,
                reason_codes=("raw_artifact_unchanged_unknown",),
            )
        )

    # Gate 3 — validation gates blocker check.
    if request.validation_gates_report is None:
        gates.append(
            ValidationGateResult(
                name=GATE_VALIDATION_GATES,
                status=GateStatus.NOT_APPLICABLE,
                reason_codes=("validation_gates_report_not_provided",),
            )
        )
    elif request.validation_gates_report.blocker_present:
        reasons = tuple(
            gate_type.value
            for gate_type in request.validation_gates_report.blocker_gate_types
        ) or ("validation_gates_blocker_present",)
        blocker_codes.extend(reasons)
        gates.append(
            ValidationGateResult(
                name=GATE_VALIDATION_GATES,
                status=GateStatus.FAILED,
                reason_codes=reasons,
            )
        )
    else:
        gates.append(
            ValidationGateResult(
                name=GATE_VALIDATION_GATES,
                status=GateStatus.PASSED,
                reason_codes=(),
            )
        )

    # Gate 4 — privacy check (text/OCR PII must be redacted).
    privacy_gate = _privacy_gate(request.text_ocr_report)
    gates.append(privacy_gate)
    if privacy_gate.status is GateStatus.FAILED:
        blocker_codes.extend(privacy_gate.reason_codes)

    # Gate 5 — blocked objects must be excluded from the export.
    if request.blocked_object_count > 0:
        blocker_codes.append("blocked_objects_present")
        gates.append(
            ValidationGateResult(
                name=GATE_BLOCKED_OBJECTS_EXCLUDED,
                status=GateStatus.FAILED,
                reason_codes=("blocked_objects_present",),
            )
        )
    else:
        gates.append(
            ValidationGateResult(
                name=GATE_BLOCKED_OBJECTS_EXCLUDED,
                status=GateStatus.PASSED,
                reason_codes=(),
            )
        )

    # Gate 6 — model impact verdict (optional unless required by request).
    gates.append(
        _model_impact_gate(
            request.model_impact_report,
            require_eligibility=request.require_model_impact_eligibility,
            blocker_codes=blocker_codes,
        )
    )

    # Gate 7 — required artifacts are present.
    missing = _missing_required_artifacts(request)
    if missing:
        for code in missing:
            blocker_codes.append(code)
        gates.append(
            ValidationGateResult(
                name=GATE_REQUIRED_ARTIFACTS,
                status=GateStatus.FAILED,
                reason_codes=tuple(missing),
            )
        )
    else:
        gates.append(
            ValidationGateResult(
                name=GATE_REQUIRED_ARTIFACTS,
                status=GateStatus.PASSED,
                reason_codes=(),
            )
        )

    return gates, _ordered_unique(blocker_codes)


def _raw_artifact_unchanged(
    *,
    candidate: CandidateDatasetVersion,
    gates_report: ValidationGatesReport | None,
) -> bool | None:
    if gates_report is not None:
        return gates_report.raw_artifact_unchanged
    summary = candidate.validation_gates_summary
    raw = summary.get("raw_artifact_unchanged")
    if isinstance(raw, bool):
        return raw
    return None


def _privacy_gate(report: TextOcrReport | None) -> ValidationGateResult:
    if report is None:
        return ValidationGateResult(
            name=GATE_PRIVACY_CHECK,
            status=GateStatus.NOT_APPLICABLE,
            reason_codes=("text_ocr_report_not_provided",),
        )
    unredacted = max(
        0, report.total_pii_record_count - report.total_redacted_record_count
    )
    if unredacted > 0:
        return ValidationGateResult(
            name=GATE_PRIVACY_CHECK,
            status=GateStatus.FAILED,
            reason_codes=("pii_unmasked",),
        )
    return ValidationGateResult(
        name=GATE_PRIVACY_CHECK,
        status=GateStatus.PASSED,
        reason_codes=(),
    )


def _model_impact_gate(
    report: ModelImpactReport | None,
    *,
    require_eligibility: bool,
    blocker_codes: list[str],
) -> ValidationGateResult:
    if report is None:
        if require_eligibility:
            blocker_codes.append("model_impact_report_required")
            return ValidationGateResult(
                name=GATE_MODEL_IMPACT,
                status=GateStatus.FAILED,
                reason_codes=("model_impact_report_required",),
            )
        return ValidationGateResult(
            name=GATE_MODEL_IMPACT,
            status=GateStatus.NOT_APPLICABLE,
            reason_codes=("model_impact_report_not_provided",),
        )
    if report.verdict is ModelImpactVerdict.REJECTED:
        blocker_codes.append("model_impact_rejected")
        return ValidationGateResult(
            name=GATE_MODEL_IMPACT,
            status=GateStatus.FAILED,
            reason_codes=("model_impact_rejected",),
        )
    if report.verdict is ModelImpactVerdict.DEGRADED:
        blocker_codes.append("model_impact_degraded")
        return ValidationGateResult(
            name=GATE_MODEL_IMPACT,
            status=GateStatus.FAILED,
            reason_codes=("model_impact_degraded",),
        )
    return ValidationGateResult(
        name=GATE_MODEL_IMPACT,
        status=GateStatus.PASSED,
        reason_codes=(report.verdict.value,),
    )


def _missing_required_artifacts(
    request: BuildExportPackageRequest,
) -> list[str]:
    missing: list[str] = []
    if request.require_dataset_card and request.dataset_card_artifact is None:
        missing.append("dataset_card_artifact_missing")
    if request.require_lineage_artifact and request.lineage_artifact is None:
        missing.append("lineage_artifact_missing")
    return missing


def _resolve_artifacts(
    request: BuildExportPackageRequest,
    *,
    blocked: bool,
) -> list[ArtifactRef]:
    if blocked:
        # When blocked, the manifest still records the export-manifest
        # ref so the platform can surface the blocked package, but
        # downstream tabular/text outputs and dataset_card are NOT
        # included — the package is not promotable.
        return [request.export_manifest_artifact]
    artifacts: list[ArtifactRef] = [request.export_manifest_artifact]
    artifacts.extend(request.tabular_artifacts)
    artifacts.extend(request.text_artifacts)
    if request.dataset_card_artifact is not None:
        artifacts.append(request.dataset_card_artifact)
    if request.lineage_artifact is not None:
        artifacts.append(request.lineage_artifact)
    if request.dataforge_report_artifact is not None:
        artifacts.append(request.dataforge_report_artifact)
    if request.review_queue_artifact is not None:
        artifacts.append(request.review_queue_artifact)
    artifacts.extend(request.extra_artifacts)
    seen: set[str] = set()
    unique: list[ArtifactRef] = []
    for ref in artifacts:
        if ref.uri in seen:
            continue
        seen.add(ref.uri)
        unique.append(ref)
    return unique


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _serialize(package: ExportPackage) -> bytes:
    return json.dumps(
        package.model_dump(mode="json"), sort_keys=True, indent=2
    ).encode("utf-8")


__all__ = [
    "BuildExportPackageRequest",
    "BuildExportPackageResult",
    "EXPORT_PACKAGE_FORMAT",
    "EXPORT_PACKAGE_KIND",
    "EXPORT_PACKAGE_MEDIA_TYPE",
    "EXPORT_PACKAGE_SCHEMA_VERSION",
    "ExportPackageBuilderError",
    "GATE_BLOCKED_OBJECTS_EXCLUDED",
    "GATE_CANDIDATE_VALIDATION",
    "GATE_MODEL_IMPACT",
    "GATE_PRIVACY_CHECK",
    "GATE_RAW_IMMUTABILITY",
    "GATE_REQUIRED_ARTIFACTS",
    "GATE_VALIDATION_GATES",
    "build_export_package",
]
