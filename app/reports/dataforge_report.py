"""Build and persist DataForgeReport summary artifacts for ANALYZE_ONLY."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    ArtifactRef,
    DataForgeReport,
    DataForgeScore,
    DataModality,
    DecisionReport,
    EvidenceRef,
    PredictionReportSection,
    SignalStatus,
    WorkflowType,
)
from app.domain.common import NonEmptyStr, Sha256Digest

DATAFORGE_REPORT_ARTIFACT_KIND = "dataforge_report"
DATAFORGE_REPORT_ARTIFACT_FORMAT = "json"
DATAFORGE_REPORT_MEDIA_TYPE = "application/json"
DATAFORGE_REPORT_SCHEMA_VERSION = "dataforge_report.v1"
_PREDICTION_NOT_PROVIDED_REASON = "prediction_manifest_not_provided"


class BuildDataForgeReportRequest(BaseModel):
    """Inputs required to assemble an ANALYZE_ONLY summary report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    score: DataForgeScore
    decision_report: DecisionReport
    review_queue_refs: tuple[EvidenceRef, ...] = ()
    detail_artifacts: tuple[ArtifactRef, ...] = ()
    prediction_manifest_ref: EvidenceRef | None = None
    prediction_validation_report_ref: EvidenceRef | None = None
    model_error_analysis_report_ref: EvidenceRef | None = None
    object_count: int | None = Field(default=None, ge=0)
    risk_profile: str | None = None
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class BuildDataForgeReportArtifactResult:
    """DataForgeReport plus immutable artifact registry record."""

    report: DataForgeReport
    artifact: RegisteredArtifact


def build_dataforge_report_artifact(
    *,
    request: BuildDataForgeReportRequest,
    registry: ArtifactRegistry,
) -> BuildDataForgeReportArtifactResult:
    """Build and persist the high-level ANALYZE_ONLY DataForgeReport."""
    report = build_dataforge_report(request)
    artifact = registry.save_artifact(
        artifact_kind=DATAFORGE_REPORT_ARTIFACT_KIND,
        data=(report.model_dump_json() + "\n").encode("utf-8"),
        artifact_format=DATAFORGE_REPORT_ARTIFACT_FORMAT,
        media_type=DATAFORGE_REPORT_MEDIA_TYPE,
        schema_version=DATAFORGE_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "dataforge-report-id": report.report_id,
            "workflow-type": report.workflow_type.value,
            "mutates-dataset": str(report.mutates_dataset).lower(),
        },
    )
    return BuildDataForgeReportArtifactResult(report=report, artifact=artifact)


def build_dataforge_report(request: BuildDataForgeReportRequest) -> DataForgeReport:
    """Assemble a contract-valid summary report without dataset mutation."""
    prediction_section = _prediction_section(request)
    modalities = _modalities(request.decision_report)
    score_value = request.score.value
    report_id = request.report_id or (
        f"dataforge_report_{request.decision_report.decision_report_id}"
    )
    return DataForgeReport(
        report_id=report_id,
        report_schema_version=DATAFORGE_REPORT_SCHEMA_VERSION,
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        workflow_type=WorkflowType.ANALYZE_ONLY,
        mutates_dataset=False,
        overview=_overview(
            request=request,
            modalities=modalities,
            prediction_section=prediction_section,
        ),
        score=request.score,
        modality_scores={modality: score_value for modality in modalities},
        blockers=request.decision_report.critical_blockers,
        recommendations=request.decision_report.recommended_actions,
        review_queues=request.review_queue_refs,
        detail_artifacts=request.detail_artifacts,
        prediction_section=prediction_section,
        generated_at=request.generated_at or datetime.now(UTC),
    )


def _overview(
    *,
    request: BuildDataForgeReportRequest,
    modalities: tuple[DataModality, ...],
    prediction_section: PredictionReportSection,
) -> dict[str, object]:
    return {
        "object_count": request.object_count
        if request.object_count is not None
        else len(request.decision_report.object_decisions),
        "modalities": [modality.value for modality in modalities],
        "risk_profile": request.risk_profile,
        "dataset_decision": request.decision_report.dataset_decision.value,
        "readiness_status": request.decision_report.readiness.status.value,
        "blocker_count": len(request.decision_report.critical_blockers),
        "recommendation_count": len(request.decision_report.recommended_actions),
        "review_queue_count": len(request.review_queue_refs),
        "detail_artifact_count": len(request.detail_artifacts),
        "ambiguous_object_candidates": prediction_section.ambiguous_object_count,
        "probable_label_error_candidates": prediction_section.probable_label_error_count,
    }


def _prediction_section(request: BuildDataForgeReportRequest) -> PredictionReportSection:
    ambiguous_count = _decision_count(request.decision_report, "ambiguous_object")
    probable_count = _decision_count(request.decision_report, "probable_label_error")
    if (
        request.prediction_manifest_ref is None
        and request.prediction_validation_report_ref is None
        and request.model_error_analysis_report_ref is None
    ):
        return PredictionReportSection(
            status=SignalStatus.NOT_APPLICABLE,
            reason=_PREDICTION_NOT_PROVIDED_REASON,
            prediction_manifest_ref=None,
            prediction_validation_report_ref=None,
            model_error_analysis_report_ref=None,
            ambiguous_object_count=0,
            probable_label_error_count=0,
        )
    return PredictionReportSection(
        status=SignalStatus.AVAILABLE,
        reason=None,
        prediction_manifest_ref=request.prediction_manifest_ref,
        prediction_validation_report_ref=request.prediction_validation_report_ref,
        model_error_analysis_report_ref=request.model_error_analysis_report_ref,
        ambiguous_object_count=ambiguous_count,
        probable_label_error_count=probable_count,
    )


def _decision_count(decision_report: DecisionReport, reason_code: str) -> int:
    return sum(
        1
        for decision in decision_report.object_decisions
        if reason_code in decision.reasons
    )


def _modalities(decision_report: DecisionReport) -> tuple[DataModality, ...]:
    unique = {decision.modality for decision in decision_report.object_decisions}
    if not unique:
        return (DataModality.TABULAR,)
    return tuple(sorted(unique, key=lambda modality: modality.value))


def dataforge_report_ref(artifact: RegisteredArtifact) -> EvidenceRef:
    """Return a lightweight report reference for downstream report indexes."""
    return EvidenceRef(kind="DATAFORGE_REPORT", uri=artifact.uri)


def review_queue_ref(artifact: RegisteredArtifact) -> EvidenceRef:
    """Return a lightweight ReviewQueue reference for DataForgeReport.review_queues."""
    return EvidenceRef(kind="REVIEW_QUEUE", uri=artifact.uri)


def detail_artifact_refs(artifacts: Iterable[RegisteredArtifact]) -> tuple[ArtifactRef, ...]:
    """Extract immutable ArtifactRef values from registered detail artifacts."""
    return tuple(artifact.artifact_ref for artifact in artifacts)


def serialize_dataforge_report(report: DataForgeReport) -> bytes:
    """Serialize report JSON deterministically for tests and artifact comparisons."""
    payload = json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return (payload + "\n").encode("utf-8")


__all__ = [
    "DATAFORGE_REPORT_ARTIFACT_FORMAT",
    "DATAFORGE_REPORT_ARTIFACT_KIND",
    "DATAFORGE_REPORT_MEDIA_TYPE",
    "DATAFORGE_REPORT_SCHEMA_VERSION",
    "BuildDataForgeReportArtifactResult",
    "BuildDataForgeReportRequest",
    "build_dataforge_report",
    "build_dataforge_report_artifact",
    "dataforge_report_ref",
    "detail_artifact_refs",
    "review_queue_ref",
    "serialize_dataforge_report",
]
