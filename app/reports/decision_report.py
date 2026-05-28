"""Persist DecisionReport artifacts through scoped object storage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import DecisionReport, EvidenceBundle
from app.kernel import DecisionPolicy, PolicyInputEnvelope
from app.kernel.decision_report import (
    DECISION_REPORT_SCHEMA_VERSION,
    BuildDecisionReportRequest,
    build_decision_report,
)

DECISION_REPORT_ARTIFACT_KIND = "decision_report"
DECISION_REPORT_ARTIFACT_FORMAT = "json"
DECISION_REPORT_MEDIA_TYPE = "application/json"


@dataclass(frozen=True)
class BuildDecisionReportArtifactResult:
    """DecisionReport plus its immutable artifact registry record."""

    report: DecisionReport
    artifact: RegisteredArtifact


def build_decision_report_artifact(
    *,
    evidence_bundles: Iterable[EvidenceBundle],
    request: BuildDecisionReportRequest,
    registry: ArtifactRegistry,
    policy: DecisionPolicy | None = None,
    inputs: PolicyInputEnvelope | None = None,
) -> BuildDecisionReportArtifactResult:
    """Run Decision Core report assembly and persist the immutable report JSON."""
    report = build_decision_report(
        evidence_bundles=evidence_bundles,
        request=request,
        policy=policy,
        inputs=inputs,
    )
    payload = (report.model_dump_json() + "\n").encode("utf-8")
    artifact = registry.save_artifact(
        artifact_kind=DECISION_REPORT_ARTIFACT_KIND,
        data=payload,
        artifact_format=DECISION_REPORT_ARTIFACT_FORMAT,
        media_type=DECISION_REPORT_MEDIA_TYPE,
        schema_version=DECISION_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "decision-report-id": report.decision_report_id,
            "object-decision-count": str(len(report.object_decisions)),
            "recommended-action-count": str(len(report.recommended_actions)),
        },
    )
    return BuildDecisionReportArtifactResult(report=report, artifact=artifact)


__all__ = [
    "DECISION_REPORT_ARTIFACT_FORMAT",
    "DECISION_REPORT_ARTIFACT_KIND",
    "DECISION_REPORT_MEDIA_TYPE",
    "BuildDecisionReportArtifactResult",
    "build_decision_report_artifact",
]
