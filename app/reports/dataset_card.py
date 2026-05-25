"""Builder for the ``dataset_card.md`` export artifact (TASK-056).

Per PRD §26.1 / §29.14 every export package must include a human
readable ``dataset_card.md`` that describes the dataset, the actions
that produced this version, the privacy/export restrictions that
apply, the synthetic / augmented provenance (if any) and the lineage
references the platform UI can follow.

The dataset card must:

- be generated only from immutable, contract-validated upstream
  artifacts (CandidateDatasetVersion, ValidationGatesReport,
  ModelImpactReport, TextOcrReport, ExportPackage, SplitManifest,
  DecisionReport, DataForgeScore);
- never echo raw rows, raw text, raw PII or raw signed bodies — it
  carries counts, ids, statuses, hashes and uris only;
- always state the synthetic provenance explicitly. When the candidate
  has no synthetic rows, the card prints an explicit
  ``No synthetic data`` line so reviewers cannot mistake the absence
  for an oversight;
- always state privacy / export restrictions explicitly, even when
  no PII was found, so the bank reviewer knows the system actually
  evaluated the question;
- be deterministic for the same inputs (idempotent through
  ``ArtifactRegistry`` content-addressed paths).

The resulting Markdown artifact is registered with
``artifact_kind="DATASET_CARD"`` / ``artifact_format="md"`` /
``media_type="text/markdown"`` / ``schema_version="dataset_card.v1"``
so downstream ExportPackage builder can pick it up and surface it as
``DATASET_CARD`` in the package manifest.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.domain import (
    ArtifactRef,
    CandidateDatasetVersion,
    DataForgeScore,
    DataModality,
    DataSplit,
    DecisionReport,
    ExportPackage,
    ExportPackageStatus,
    GateStatus,
    ModelImpactReport,
    SplitClassDistribution,
    SplitManifest,
    SyntheticCandidateMetadata,
    TextOcrReport,
    ValidationGatesReport,
    ValidationGateStatus,
)
from app.domain.common import NonEmptyStr, Sha256Digest

DATASET_CARD_ARTIFACT_KIND = "DATASET_CARD"
DATASET_CARD_ARTIFACT_FORMAT = "md"
DATASET_CARD_MEDIA_TYPE = "text/markdown"
DATASET_CARD_SCHEMA_VERSION = "dataset_card.v1"


class DatasetCardBuilderError(ValueError):
    """Raised when the dataset_card builder receives unsafe inputs."""

    def __init__(self, *, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class BuildDatasetCardRequest(BaseModel):
    """Inputs for :func:`build_dataset_card_artifact`.

    All inputs except ``candidate_dataset_version`` are optional: the
    card always renders a complete document with explicit
    "not available" notes for missing sections. The builder never
    silently skips required sections.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    candidate_dataset_version: CandidateDatasetVersion
    decision_report: DecisionReport | None = None
    validation_gates_report: ValidationGatesReport | None = None
    model_impact_report: ModelImpactReport | None = None
    text_ocr_report: TextOcrReport | None = None
    export_package: ExportPackage | None = None
    split_manifest: SplitManifest | None = None
    dataforge_score: DataForgeScore | None = None
    declared_modalities: tuple[DataModality, ...] = ()
    additional_review_notes: tuple[str, ...] = ()
    review_queue_artifacts: tuple[ArtifactRef, ...] = ()
    detail_artifacts: tuple[ArtifactRef, ...] = ()
    privacy_policy_version: NonEmptyStr | None = None
    export_policy_version: NonEmptyStr | None = None
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    generated_at: datetime | None = None
    dataset_card_id: str | None = None
    object_count: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class BuildDatasetCardResult:
    """Persisted dataset_card.md artifact + rendered markdown body."""

    artifact: RegisteredArtifact
    markdown: str


def build_dataset_card_artifact(
    request: BuildDatasetCardRequest,
    *,
    registry: ArtifactRegistry,
) -> BuildDatasetCardResult:
    """Render and persist a ``dataset_card.md`` artifact."""
    markdown = render_dataset_card(request)
    payload = markdown.encode("utf-8")
    metadata = {
        "dataset-card-id": request.dataset_card_id
        or f"dataset_card_{request.candidate_dataset_version.candidate_version_id}",
        "version-id": (
            request.candidate_dataset_version.lineage.proposed_version_name
        ),
        "source-version-id": (
            request.candidate_dataset_version.lineage.parent_version_id
        ),
    }
    artifact = registry.save_artifact(
        artifact_kind=DATASET_CARD_ARTIFACT_KIND,
        data=payload,
        artifact_format=DATASET_CARD_ARTIFACT_FORMAT,
        media_type=DATASET_CARD_MEDIA_TYPE,
        schema_version=DATASET_CARD_SCHEMA_VERSION,
        dataset_version_id=(
            request.candidate_dataset_version.lineage.proposed_version_name
        ),
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata=metadata,
    )
    return BuildDatasetCardResult(artifact=artifact, markdown=markdown)


def render_dataset_card(request: BuildDatasetCardRequest) -> str:
    """Render dataset_card.md content deterministically.

    The function intentionally accepts only typed contract objects so
    raw rows / PII can never be embedded by accident.
    """
    candidate = request.candidate_dataset_version
    generated_at = request.generated_at or datetime.now(UTC)

    buffer = io.StringIO()
    buffer.write("# Dataset Card\n\n")

    # ---------------------------------------------------------------
    # 1. Dataset summary
    # ---------------------------------------------------------------
    buffer.write("## Dataset Summary\n\n")
    summary_rows: list[tuple[str, str]] = [
        ("Dataset id", candidate.lineage.dataset_id),
        ("Candidate version id", candidate.candidate_version_id),
        (
            "Proposed dataset version",
            candidate.lineage.proposed_version_name,
        ),
        ("Parent dataset version", candidate.lineage.parent_version_id),
        ("Candidate status", candidate.status.value),
    ]
    if request.decision_report is not None:
        summary_rows.append(
            (
                "Dataset decision",
                request.decision_report.dataset_decision.value,
            )
        )
        summary_rows.append(
            (
                "Readiness status",
                request.decision_report.readiness.status.value,
            )
        )
    if request.dataforge_score is not None:
        score = request.dataforge_score
        summary_rows.append(
            ("DataForge score", f"{score.value:.2f} (raw {score.raw_score:.2f})")
        )
        summary_rows.append(("Score policy", score.policy_version))
    summary_rows.append(("Generated at", generated_at.isoformat()))
    _write_kv_table(buffer, summary_rows)

    # ---------------------------------------------------------------
    # 2. Task type & target
    # ---------------------------------------------------------------
    buffer.write("\n## Task Type and Target\n\n")
    target_rows: list[tuple[str, str]] = []
    if request.model_impact_report is not None:
        cfg = request.model_impact_report.candidate_model_config
        target_rows.append(("Task type", "supervised_classification"))
        target_rows.append(("Target column", cfg.target_column))
        target_rows.append(("Baseline algorithm", cfg.algorithm))
        target_rows.append(
            (
                "Baseline library",
                f"{cfg.library} {cfg.library_version}",
            )
        )
        if cfg.feature_columns:
            target_rows.append(
                ("Feature columns", ", ".join(sorted(cfg.feature_columns)))
            )
    elif request.split_manifest is not None:
        target_rows.append(("Task type", "supervised_classification"))
        target_rows.append(
            ("Target column", request.split_manifest.target_column)
        )
        target_rows.append(
            ("Split strategy", request.split_manifest.strategy.value)
        )
    else:
        target_rows.append(("Task type", "not_recorded"))
        target_rows.append(("Target column", "not_recorded"))
    _write_kv_table(buffer, target_rows)

    # ---------------------------------------------------------------
    # 3. Modalities
    # ---------------------------------------------------------------
    buffer.write("\n## Modalities\n\n")
    modalities = _resolve_modalities(
        declared=request.declared_modalities,
        decision_report=request.decision_report,
        text_ocr_report=request.text_ocr_report,
    )
    for modality in modalities:
        buffer.write(f"- {modality.value}\n")

    # ---------------------------------------------------------------
    # 4. Object counts and split distribution
    # ---------------------------------------------------------------
    buffer.write("\n## Object Counts and Split Distribution\n\n")
    count_rows: list[tuple[str, str]] = []
    if request.object_count is not None:
        count_rows.append(("Object count (analyzed)", str(request.object_count)))
    if request.export_package is not None:
        counts = request.export_package.object_counts
        count_rows.append(("Included in export", str(counts.included)))
        count_rows.append(("Blocked", str(counts.blocked)))
        count_rows.append(("Excluded", str(counts.excluded)))
    if not count_rows:
        count_rows.append(("Object counts", "not_recorded"))
    _write_kv_table(buffer, count_rows)

    if request.split_manifest is not None:
        buffer.write("\n### Split Distribution\n\n")
        buffer.write("| Split | Total | Class | Count |\n")
        buffer.write("|---|---|---|---|\n")
        for entry in _split_rows(request.split_manifest.class_distribution):
            split_label, total, class_label, count = entry
            buffer.write(
                f"| {split_label} | {total} | {class_label} | {count} |\n"
            )
    else:
        buffer.write(
            "\nSplit distribution is not available for this candidate version.\n"
        )

    # ---------------------------------------------------------------
    # 5. Applied actions
    # ---------------------------------------------------------------
    buffer.write("\n## Applied Actions\n\n")
    if candidate.action_plan_steps:
        buffer.write("| # | Step type | Method | Plugin | Plugin version | Random seed |\n")
        buffer.write("|---|---|---|---|---|---|\n")
        for idx, step in enumerate(candidate.action_plan_steps, start=1):
            seed = "n/a" if step.random_seed is None else str(step.random_seed)
            buffer.write(
                f"| {idx} | {step.step_type} | {step.method_id} "
                f"| {step.plugin_id} | {step.plugin_version} | {seed} |\n"
            )
    else:
        buffer.write("No action plan steps were recorded for this candidate.\n")

    # ---------------------------------------------------------------
    # 6. Synthetic / augmented provenance — ALWAYS explicit
    # ---------------------------------------------------------------
    buffer.write("\n## Synthetic and Augmented Provenance\n\n")
    _write_synthetic_section(buffer, candidate.synthetic_metadata)

    # ---------------------------------------------------------------
    # 7. Privacy / export restrictions — ALWAYS explicit
    # ---------------------------------------------------------------
    buffer.write("\n## Privacy and Export Restrictions\n\n")
    _write_privacy_section(
        buffer,
        text_ocr_report=request.text_ocr_report,
        privacy_policy_version=request.privacy_policy_version,
        export_policy_version=request.export_policy_version,
        export_package=request.export_package,
        review_queue_count=len(request.review_queue_artifacts),
    )

    # ---------------------------------------------------------------
    # 8. Blockers & review notes
    # ---------------------------------------------------------------
    buffer.write("\n## Blockers and Review Notes\n\n")
    blockers = _collect_blockers(
        decision_report=request.decision_report,
        export_package=request.export_package,
        candidate=candidate,
    )
    if blockers:
        for code, message in blockers:
            buffer.write(f"- `{code}` — {message}\n")
    else:
        buffer.write("No critical blockers reported.\n")
    if request.additional_review_notes:
        buffer.write("\n### Additional review notes\n\n")
        for note in request.additional_review_notes:
            buffer.write(f"- {note}\n")

    # ---------------------------------------------------------------
    # 9. Validation gates summary
    # ---------------------------------------------------------------
    buffer.write("\n## Validation Gates\n\n")
    _write_validation_gates_section(
        buffer,
        gates_report=request.validation_gates_report,
        export_package=request.export_package,
    )

    # ---------------------------------------------------------------
    # 10. Model impact summary
    # ---------------------------------------------------------------
    buffer.write("\n## Model Impact Summary\n\n")
    _write_model_impact_section(buffer, report=request.model_impact_report)

    # ---------------------------------------------------------------
    # 11. Lineage references
    # ---------------------------------------------------------------
    buffer.write("\n## Lineage References\n\n")
    _write_lineage_section(
        buffer,
        request=request,
        candidate=candidate,
    )

    # Final byte: ensure trailing newline so file ends deterministically.
    text = buffer.getvalue()
    if not text.endswith("\n"):
        text += "\n"
    return text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_kv_table(
    buffer: io.StringIO, rows: Iterable[tuple[str, str]]
) -> None:
    buffer.write("| Field | Value |\n")
    buffer.write("|---|---|\n")
    for key, value in rows:
        safe_value = _escape_table_cell(value)
        buffer.write(f"| {key} | {safe_value} |\n")


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _resolve_modalities(
    *,
    declared: tuple[DataModality, ...],
    decision_report: DecisionReport | None,
    text_ocr_report: TextOcrReport | None,
) -> tuple[DataModality, ...]:
    seen: list[DataModality] = []
    for modality in declared:
        if modality not in seen:
            seen.append(modality)
    if decision_report is not None:
        for decision in decision_report.object_decisions:
            if decision.modality not in seen:
                seen.append(decision.modality)
    if text_ocr_report is not None and text_ocr_report.total_record_count > 0:
        # Text/OCR reports prove text/document_ocr modality is present
        # even if no per-object decisions exist yet.
        if DataModality.TEXT not in seen and DataModality.DOCUMENT_OCR not in seen:
            for source in text_ocr_report.sources:
                modality = (
                    DataModality.DOCUMENT_OCR
                    if source.source_kind.value == "ocr_records"
                    else DataModality.TEXT
                )
                if modality not in seen:
                    seen.append(modality)
    if not seen:
        seen.append(DataModality.TABULAR)
    return tuple(sorted(seen, key=lambda modality: modality.value))


def _split_rows(
    distributions: Iterable[SplitClassDistribution],
) -> Iterator[tuple[str, int, str, int]]:
    by_split: dict[DataSplit, SplitClassDistribution] = {
        d.split: d for d in distributions
    }
    for split in (DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST):
        entry = by_split.get(split)
        if entry is None:
            continue
        for label in sorted(entry.class_counts):
            yield (split.value, entry.total_count, label, entry.class_counts[label])


def _write_synthetic_section(
    buffer: io.StringIO,
    metadata: SyntheticCandidateMetadata | None,
) -> None:
    if metadata is None:
        buffer.write(
            "No synthetic data was generated for this candidate version.\n"
        )
        return
    rows: list[tuple[str, str]] = [
        ("Synthetic method", metadata.method_id),
        ("Plugin", f"{metadata.plugin_id} {metadata.plugin_version}"),
        ("Random seed", str(metadata.random_seed)),
        ("Source split", metadata.source_split),
        ("Generated row count", str(metadata.generated_count)),
        (
            "Sampling strategy",
            f"{metadata.sampling_strategy:.4f}",
        ),
        ("Synthetic policy", metadata.policy_version),
        ("Synthetic config hash", metadata.config_hash),
    ]
    if metadata.source_cohort is not None:
        rows.append(("Source cohort", metadata.source_cohort))
    rows.append(
        (
            "Source object ids referenced",
            str(metadata.full_source_object_ids_count),
        )
    )
    rows.append(
        (
            "Source object ids inline",
            str(len(metadata.source_object_ids))
            + (
                " (truncated)"
                if metadata.source_object_ids_truncated
                else ""
            ),
        )
    )
    rows.append(
        ("Validation report", metadata.validation_report.uri)
    )
    rows.append(
        ("Synthetic dataset report", metadata.synthetic_dataset_report.uri)
    )
    if metadata.model_impact_report is not None:
        rows.append(
            ("Model impact report", metadata.model_impact_report.uri)
        )
    _write_kv_table(buffer, rows)


def _write_privacy_section(
    buffer: io.StringIO,
    *,
    text_ocr_report: TextOcrReport | None,
    privacy_policy_version: NonEmptyStr | None,
    export_policy_version: NonEmptyStr | None,
    export_package: ExportPackage | None,
    review_queue_count: int,
) -> None:
    if text_ocr_report is None:
        pii_records = 0
        redacted_records = 0
        text_record_count = 0
        pii_token_count = 0
    else:
        pii_records = text_ocr_report.total_pii_record_count
        redacted_records = text_ocr_report.total_redacted_record_count
        text_record_count = text_ocr_report.total_record_count
        pii_token_count = text_ocr_report.total_pii_token_count
    unredacted = max(0, pii_records - redacted_records)
    rows: list[tuple[str, str]] = [
        ("Text/OCR records analyzed", str(text_record_count)),
        ("Records with detected PII", str(pii_records)),
        ("PII tokens detected", str(pii_token_count)),
        ("Records redacted", str(redacted_records)),
        ("Records with unredacted PII", str(unredacted)),
        (
            "Privacy policy",
            privacy_policy_version
            if privacy_policy_version is not None
            else "not_recorded",
        ),
        (
            "Export policy",
            export_policy_version
            if export_policy_version is not None
            else "not_recorded",
        ),
        ("Privacy review queues", str(review_queue_count)),
    ]
    _write_kv_table(buffer, rows)
    buffer.write("\n### Restrictions\n\n")
    buffer.write(
        "- Raw text and raw PII must not be present in any export "
        "artifact; only redacted text/OCR JSONL records may be "
        "exported.\n"
    )
    buffer.write(
        "- Blocked objects are excluded from the export and from the "
        "object_id manifest.\n"
    )
    buffer.write(
        "- High-risk privacy objects are routed through the privacy "
        "review queue before any downstream training or sharing.\n"
    )
    if export_package is not None:
        buffer.write(
            f"- Export package status at card generation: "
            f"`{export_package.status.value}`.\n"
        )
        if export_package.status is ExportPackageStatus.BLOCKED:
            for code in export_package.blocked_reason_codes:
                buffer.write(f"  - export blocker: `{code}`\n")


def _collect_blockers(
    *,
    decision_report: DecisionReport | None,
    export_package: ExportPackage | None,
    candidate: CandidateDatasetVersion,
) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    if decision_report is not None:
        for blocker in decision_report.critical_blockers:
            seen.setdefault(blocker.code, blocker.message)
    if export_package is not None:
        for code in export_package.blocked_reason_codes:
            seen.setdefault(code, "Reported by export readiness gates.")
    for code in candidate.blocker_reason_codes:
        seen.setdefault(code, "Reported by candidate validation gates.")
    return [(code, seen[code]) for code in sorted(seen)]


def _write_validation_gates_section(
    buffer: io.StringIO,
    *,
    gates_report: ValidationGatesReport | None,
    export_package: ExportPackage | None,
) -> None:
    if gates_report is None and export_package is None:
        buffer.write("No validation gates report attached to this candidate.\n")
        return
    if gates_report is not None:
        buffer.write(
            f"Candidate validation overall status: "
            f"`{gates_report.overall_status.value}`.\n\n"
        )
        if gates_report.gates:
            buffer.write("| Gate | Status | Severity | Reason code |\n")
            buffer.write("|---|---|---|---|\n")
            for candidate_gate in gates_report.gates:
                reason = (
                    candidate_gate.reason_code
                    if candidate_gate.reason_code is not None
                    else "-"
                )
                buffer.write(
                    f"| {candidate_gate.gate_type.value} | {candidate_gate.status.value} "
                    f"| {candidate_gate.severity.value} | {reason} |\n"
                )
        else:
            buffer.write("No individual gate results recorded.\n")
    if export_package is not None and export_package.validation_gates:
        buffer.write("\n### Export readiness gates\n\n")
        buffer.write("| Gate | Status | Reason codes |\n")
        buffer.write("|---|---|---|\n")
        for export_gate in export_package.validation_gates:
            reasons = (
                ", ".join(f"`{code}`" for code in export_gate.reason_codes)
                if export_gate.reason_codes
                else "-"
            )
            buffer.write(
                f"| {export_gate.name} | {_gate_status_label(export_gate.status)} "
                f"| {reasons} |\n"
            )


def _gate_status_label(status: GateStatus | ValidationGateStatus) -> str:
    return status.value


def _write_model_impact_section(
    buffer: io.StringIO,
    *,
    report: ModelImpactReport | None,
) -> None:
    if report is None:
        buffer.write(
            "Model impact report is not attached to this candidate version.\n"
        )
        return
    rows: list[tuple[str, str]] = [
        ("Verdict", report.verdict.value),
        ("Metric library", f"{report.metric_library} {report.metric_library_version}"),
        (
            "Rare-class recall",
            f"{report.rare_class_recall_before:.4f} -> "
            f"{report.rare_class_recall_after:.4f} "
            f"(Δ {report.rare_class_recall_delta:+.4f})",
        ),
        (
            "Macro F1",
            f"{report.macro_f1_before:.4f} -> "
            f"{report.macro_f1_after:.4f} "
            f"(Δ {report.macro_f1_delta:+.4f})",
        ),
        (
            "Weighted F1",
            f"{report.weighted_f1_before:.4f} -> "
            f"{report.weighted_f1_after:.4f} "
            f"(Δ {report.weighted_f1_delta:+.4f})",
        ),
    ]
    if (
        report.pr_auc_status.value == "available"
        and report.pr_auc_before is not None
        and report.pr_auc_after is not None
        and report.pr_auc_delta is not None
    ):
        rows.append(
            (
                "PR-AUC",
                f"{report.pr_auc_before:.4f} -> "
                f"{report.pr_auc_after:.4f} "
                f"(Δ {report.pr_auc_delta:+.4f})",
            )
        )
    else:
        rows.append(
            (
                "PR-AUC",
                f"not_applicable ({report.pr_auc_reason or 'reason_not_recorded'})",
            )
        )
    rows.append(
        (
            "Synthetic utility status",
            report.synthetic_utility_status.value,
        )
    )
    _write_kv_table(buffer, rows)


def _write_lineage_section(
    buffer: io.StringIO,
    *,
    request: BuildDatasetCardRequest,
    candidate: CandidateDatasetVersion,
) -> None:
    rows: list[tuple[str, str]] = [
        ("Action plan", candidate.lineage.action_plan_id),
        (
            "Decision report",
            candidate.lineage.decision_report_id or "not_recorded",
        ),
        ("Compute job id", candidate.lineage.created_by_job_id),
        ("Config hash", candidate.lineage.config_hash),
        (
            "Profile policy",
            candidate.policy_versions.profile_policy_version,
        ),
        (
            "Decision policy",
            candidate.policy_versions.decision_policy_version,
        ),
        (
            "Score policy",
            candidate.policy_versions.score_policy_version,
        ),
        (
            "Method policy",
            candidate.policy_versions.method_policy_version,
        ),
    ]
    if candidate.policy_versions.validation_gates_policy_version is not None:
        rows.append(
            (
                "Validation gates policy",
                candidate.policy_versions.validation_gates_policy_version,
            )
        )
    rows.append(
        ("Primary dataset artifact", candidate.primary_dataset_artifact.uri)
    )
    if candidate.validation_gates_report is not None:
        rows.append(
            (
                "Validation gates report",
                candidate.validation_gates_report.uri,
            )
        )
    if request.export_package is not None:
        rows.append(
            (
                "Export package version id",
                request.export_package.version_id,
            )
        )
        rows.append(
            (
                "Export package source version id",
                request.export_package.source_version_id,
            )
        )
    _write_kv_table(buffer, rows)

    if request.review_queue_artifacts:
        buffer.write("\n### Review queues\n\n")
        for ref in request.review_queue_artifacts:
            buffer.write(f"- `{ref.kind}`: {ref.uri}\n")
    if request.detail_artifacts:
        buffer.write("\n### Detail artifacts\n\n")
        for ref in request.detail_artifacts:
            buffer.write(f"- `{ref.kind}`: {ref.uri}\n")


def _enforce_no_inline_pii(text: str) -> None:
    """Defensive guard: dataset_card must never echo raw PII tokens.

    The card builder only consumes typed contract objects, so this
    guard is mostly belt-and-braces for refactors. It is invoked from
    tests through ``serialize_dataset_card`` if/when reviewers want a
    second wall against accidental PII echo.
    """
    forbidden = ("@example.com", "@gmail.com", "555-")
    matches = [token for token in forbidden if token in text]
    if matches:
        raise DatasetCardBuilderError(
            reason_code="dataset_card_contains_raw_pii_tokens",
            message=(
                "dataset_card.md must not echo raw PII tokens; "
                f"detected: {sorted(set(matches))}"
            ),
        )


def serialize_dataset_card(markdown: str) -> bytes:
    """Encode dataset_card markdown for storage tests."""
    _enforce_no_inline_pii(markdown)
    return markdown.encode("utf-8")


__all__ = [
    "BuildDatasetCardRequest",
    "BuildDatasetCardResult",
    "DATASET_CARD_ARTIFACT_FORMAT",
    "DATASET_CARD_ARTIFACT_KIND",
    "DATASET_CARD_MEDIA_TYPE",
    "DATASET_CARD_SCHEMA_VERSION",
    "DatasetCardBuilderError",
    "build_dataset_card_artifact",
    "render_dataset_card",
    "serialize_dataset_card",
]
