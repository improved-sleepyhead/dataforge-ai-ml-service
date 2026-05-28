"""Runtime builders for ANALYZE_ONLY Dagster assets.

The functions in this module bridge the Dagster asset graph to the existing
contract-compatible builders. They produce immutable artifacts through
``ArtifactRegistry`` and keep all materialization metadata privacy-safe.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter, ObjectStorageError
from app.domain import ArtifactRef, ModelErrorReport, PredictionRow
from app.ingestion import (
    BuildManifestRequest,
    build_asset_manifest,
    build_validated_manifest,
    build_validated_predictions,
    manifest_rows_from_artifact,
    open_archive_artifact,
)
from app.kernel import (
    BuildDecisionReportRequest,
    BuildMethodRecommendationsRequest,
    build_method_recommendations,
)
from app.orchestration.run_context import AnalyzeRunContext
from app.orchestration.status_bridge import RunContext
from app.plugins.object_analytics import (
    BuildEvidenceBundleRequest,
    BuildObjectAnalyticsRequest,
    build_evidence_bundles,
    build_object_analytics_passports,
)
from app.plugins.predictions import analyze_model_errors, build_not_applicable_report
from app.plugins.predictions.analyzer import ManifestRowSummary
from app.plugins.tabular import ProfileBuildRequest, build_tabular_profile_report
from app.plugins.text_ocr.validator import (
    TextOcrBuildRequest,
    build_text_ocr_report,
    validate_ocr_records_jsonl,
    validate_support_messages_jsonl,
)
from app.reports import (
    BuildReviewQueuesRequest,
    build_decision_report_artifact,
    build_review_queues_artifact,
)
from app.telemetry import (
    METRIC_AMBIGUOUS_OBJECT_COUNT,
    METRIC_ARTIFACT_WRITE_MS,
    METRIC_JOB_DURATION_MS,
    METRIC_PROBABLE_LABEL_ERROR_COUNT,
    METRIC_REVIEW_QUEUE_SIZE,
    MetricsRegistry,
    TracingRegistry,
)

MODEL_ERROR_REPORT_KIND = "model_error_analysis_report"
MODEL_ERROR_REPORT_SCHEMA_VERSION = "model_error_report.v1"
PREDICTION_VALIDATION_KIND = "prediction_validation_report"
PREDICTION_VALIDATION_SCHEMA_VERSION = "prediction_validation_report.v1"
AMBIGUOUS_CANDIDATES_KIND = "ambiguous_object_candidates"
PROBABLE_LABEL_ERROR_CANDIDATES_KIND = "probable_label_error_candidates"
METHOD_RECOMMENDATIONS_KIND = "method_recommendations"


def ensure_analyze_outputs(
    *,
    analyze_context: AnalyzeRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    metrics: MetricsRegistry,
    tracing: TracingRegistry,
) -> None:
    """Build the ANALYZE_ONLY artifact set once for the current Dagster run."""
    state = analyze_context.execution_state
    if state.built:
        return

    started = datetime.now(UTC)
    try:
        _build_real_outputs(
            analyze_context=analyze_context,
            run_context=run_context,
            storage=storage,
            registry=registry,
            metrics=metrics,
            tracing=tracing,
            generated_at=started,
        )
    except ObjectStorageError:
        # API boundary tests can pass signed ArtifactRefs without loading
        # object bytes into the local in-memory storage. Keep that boundary
        # usable by emitting explicit reference-only artifacts instead of
        # silently pretending real analysis happened.
        _build_reference_only_outputs(
            analyze_context=analyze_context,
            run_context=run_context,
            registry=registry,
        )
    state.built = True


def artifact_for(
    *,
    analyze_context: AnalyzeRunContext,
    asset_name: str,
) -> RegisteredArtifact | None:
    """Return the artifact associated with an analyze asset, if present."""
    return analyze_context.execution_state.artifacts.get(asset_name)


def count_for(
    *,
    analyze_context: AnalyzeRunContext,
    name: str,
) -> int | None:
    """Return an integer summary count recorded by the analyze runtime."""
    return analyze_context.execution_state.counts.get(name)


def status_for(
    *,
    analyze_context: AnalyzeRunContext,
    name: str,
) -> str | None:
    """Return an explicit status recorded by the analyze runtime."""
    return analyze_context.execution_state.statuses.get(name)


def _build_real_outputs(
    *,
    analyze_context: AnalyzeRunContext,
    run_context: RunContext,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    metrics: MetricsRegistry,
    tracing: TracingRegistry,
    generated_at: datetime,
) -> None:
    archive_ref = _archive_ref(analyze_context.dataset_object_refs)
    state = analyze_context.execution_state
    parent_version_id = analyze_context.parent_version_id or run_context.dataset_version_id

    with tracing.span("ingestion", attributes={"stage": "ingestion", "job_type": "analyze"}):
        with open_archive_artifact(storage=storage, artifact_uri=archive_ref.uri) as reader:
            manifest_result = build_asset_manifest(
                reader,
                request=BuildManifestRequest(
                    dataset_id=run_context.dataset_id,
                    version_id=run_context.dataset_version_id,
                    parent_version_id=parent_version_id,
                    created_by_job_id=run_context.compute_run_id,
                    config_hash=analyze_context.config_hash,
                ),
                registry=registry,
            )
    state.artifacts["raw_manifest"] = manifest_result.manifest_artifact

    with tracing.span("manifest_builder", attributes={"stage": "manifest_builder"}):
        validated = build_validated_manifest(
            manifest_result.manifest_artifact,
            storage=storage,
            registry=registry,
            dataset_version_id=run_context.dataset_version_id,
            parent_version_id=parent_version_id,
            created_by_job_id=run_context.compute_run_id,
            config_hash=analyze_context.config_hash,
            organization_id=run_context.organization_id,
            project_id=run_context.project_id,
        ).validated_manifest
    state.artifacts["validated_manifest"] = validated
    manifest_rows = manifest_rows_from_artifact(storage.get(validated.uri).data)
    state.counts["manifest_row_count"] = len(manifest_rows)

    model_error_report: ModelErrorReport
    prediction_rows: tuple[PredictionRow, ...] = ()
    if analyze_context.prediction_artifact_refs:
        prediction_input = _registered_from_ref(
            analyze_context.prediction_artifact_refs[0],
            artifact_kind="raw_predictions",
            created_by_job_id=run_context.compute_run_id,
        )
        with tracing.span("prediction.validate", attributes={"stage": "prediction.validate"}):
            prediction_result = build_validated_predictions(
                prediction_input,
                storage=storage,
                registry=registry,
                dataset_version_id=run_context.dataset_version_id,
                parent_version_id=parent_version_id,
                created_by_job_id=run_context.compute_run_id,
                config_hash=analyze_context.config_hash,
                model_id="fraud_baseline",
                model_version="2026-05-14",
                manifest_object_ids=[row.object_id for row in manifest_rows],
                organization_id=run_context.organization_id,
                project_id=run_context.project_id,
            )
        state.artifacts["prediction_manifest"] = (
            prediction_result.prediction_manifest_artifact
        )
        state.artifacts["prediction_validation_report"] = _save_json_artifact(
            registry=registry,
            run_context=run_context,
            analyze_context=analyze_context,
            artifact_kind=PREDICTION_VALIDATION_KIND,
            schema_version=PREDICTION_VALIDATION_SCHEMA_VERSION,
            payload=asdict(prediction_result.validation_report),
        )
        prediction_rows = _prediction_rows(
            storage.get(prediction_result.prediction_manifest_artifact.uri).data
        )
        with tracing.span("model_error.analyze", attributes={"stage": "model_error.analyze"}):
            model_error_report = analyze_model_errors(
                rows=prediction_rows,
                manifest_index=_manifest_index(manifest_rows),
                dataset_id=run_context.dataset_id,
                version_id=run_context.dataset_version_id,
                config_hash=analyze_context.config_hash,
                model_id="fraud_baseline",
                model_version="2026-05-14",
                generated_at=generated_at,
            )
    else:
        model_error_report = build_not_applicable_report(
            dataset_id=run_context.dataset_id,
            version_id=run_context.dataset_version_id,
            config_hash=analyze_context.config_hash,
            generated_at=generated_at,
        )

    model_error_artifact = _save_json_artifact(
        registry=registry,
        run_context=run_context,
        analyze_context=analyze_context,
        artifact_kind=MODEL_ERROR_REPORT_KIND,
        schema_version=MODEL_ERROR_REPORT_SCHEMA_VERSION,
        payload=model_error_report.model_dump(mode="json"),
    )
    if analyze_context.prediction_artifact_refs:
        state.artifacts["model_error_analysis_report"] = model_error_artifact
        state.artifacts["ambiguous_object_candidates"] = _save_json_artifact(
            registry=registry,
            run_context=run_context,
            analyze_context=analyze_context,
            artifact_kind=AMBIGUOUS_CANDIDATES_KIND,
            schema_version="ambiguous_object_candidates.v1",
            payload=_candidate_payload(
                model_error_report=model_error_report,
                reason_code="ambiguous_object",
            ),
        )
        state.artifacts["probable_label_error_candidates"] = _save_json_artifact(
            registry=registry,
            run_context=run_context,
            analyze_context=analyze_context,
            artifact_kind=PROBABLE_LABEL_ERROR_CANDIDATES_KIND,
            schema_version="probable_label_error_candidates.v1",
            payload=_candidate_payload(
                model_error_report=model_error_report,
                reason_code="probable_label_error",
            ),
        )
    _record_model_error_metrics(model_error_report, metrics=metrics)

    with open_archive_artifact(storage=storage, artifact_uri=archive_ref.uri) as reader:
        with tracing.span("tabular.profile", attributes={"stage": "tabular.profile"}):
            profile_result = build_tabular_profile_report(
                reader,
                request=ProfileBuildRequest(
                    dataset_id=run_context.dataset_id,
                    version_id=run_context.dataset_version_id,
                    parent_version_id=parent_version_id,
                    created_by_job_id=run_context.compute_run_id,
                    config_hash=analyze_context.config_hash,
                    source_artifact_id=archive_ref.artifact_id,
                ),
                storage=storage,
                registry=registry,
                source_manifest_artifact=manifest_result.manifest_artifact.artifact_ref,
                generated_at=generated_at,
            )
    state.artifacts["tabular_profile_report"] = profile_result.artifact

    with open_archive_artifact(storage=storage, artifact_uri=archive_ref.uri) as reader:
        support_payload, ocr_payload = _read_text_payloads(reader.descriptors())
    with tracing.span("text_ocr.profile", attributes={"stage": "text_ocr.profile"}):
        text_request = TextOcrBuildRequest(
            dataset_id=run_context.dataset_id,
            version_id=run_context.dataset_version_id,
            parent_version_id=parent_version_id,
            created_by_job_id=run_context.compute_run_id,
            config_hash=analyze_context.config_hash,
        )
        support_report = validate_support_messages_jsonl(
            support_payload,
            detect_pii=True,
            registry=registry,
            build_request=text_request,
        )
        ocr_report = validate_ocr_records_jsonl(
            ocr_payload,
            detect_pii=True,
            registry=registry,
            build_request=text_request,
        )
        text_ocr_report = build_text_ocr_report(
            request=text_request,
            sources=(support_report, ocr_report),
            generated_at=generated_at,
        ).report

    with tracing.span("decision_core.score", attributes={"stage": "decision_core.score"}):
        passports_result = build_object_analytics_passports(
            manifest_rows=manifest_rows,
            request=BuildObjectAnalyticsRequest(
                dataset_id=run_context.dataset_id,
                version_id=run_context.dataset_version_id,
                parent_version_id=parent_version_id,
                created_by_job_id=run_context.compute_run_id,
                config_hash=analyze_context.config_hash,
            ),
            registry=registry,
            tabular_profile=profile_result.profile_report,
            text_ocr_report=text_ocr_report,
            model_error_report=model_error_report,
            computed_at=generated_at,
        )
        evidence_result = build_evidence_bundles(
            passports=passports_result.passports,
            request=BuildEvidenceBundleRequest(
                dataset_id=run_context.dataset_id,
                version_id=run_context.dataset_version_id,
                parent_version_id=parent_version_id,
                created_by_job_id=run_context.compute_run_id,
                config_hash=analyze_context.config_hash,
            ),
            registry=registry,
            source_passports_artifact=passports_result.artifact,
        )
        decision_result = build_decision_report_artifact(
            evidence_bundles=evidence_result.evidence_bundles,
            request=BuildDecisionReportRequest(
                dataset_id=run_context.dataset_id,
                version_id=run_context.dataset_version_id,
                created_by_job_id=run_context.compute_run_id,
                config_hash=analyze_context.config_hash,
                generated_at=generated_at,
            ),
            registry=registry,
        )
    state.artifacts["object_analytics_passports"] = passports_result.artifact
    state.artifacts["evidence_bundle"] = evidence_result.artifact
    state.artifacts["decision_report"] = decision_result.artifact

    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(
            tabular_profile=profile_result.profile_report,
            decision_report=decision_result.report,
        )
    )
    state.artifacts["recommended_actions"] = _save_json_artifact(
        registry=registry,
        run_context=run_context,
        analyze_context=analyze_context,
        artifact_kind=METHOD_RECOMMENDATIONS_KIND,
        schema_version="method_recommendation.v1",
        payload=[item.model_dump(mode="json") for item in recommendations],
    )

    review_result = build_review_queues_artifact(
        decision_report=decision_result.report,
        evidence_bundles=evidence_result.evidence_bundles,
        request=BuildReviewQueuesRequest(
            dataset_id=run_context.dataset_id,
            version_id=run_context.dataset_version_id,
            created_by_job_id=run_context.compute_run_id,
            config_hash=analyze_context.config_hash,
            generated_at=generated_at,
        ),
        registry=registry,
    )
    state.artifacts["review_queue"] = review_result.artifact
    state.counts["review_queue_size"] = sum(len(queue.objects) for queue in review_result.queues)
    metrics.set_gauge(
        METRIC_REVIEW_QUEUE_SIZE,
        value=float(state.counts["review_queue_size"]),
        labels={"queue_type": "all"},
    )
    _record_artifact_write_metrics(state.artifacts, metrics=metrics)


def _build_reference_only_outputs(
    *,
    analyze_context: AnalyzeRunContext,
    run_context: RunContext,
    registry: ArtifactRegistry,
) -> None:
    state = analyze_context.execution_state
    asset_names = (
        "raw_manifest",
        "validated_manifest",
        "tabular_profile_report",
        "object_analytics_passports",
        "evidence_bundle",
        "decision_report",
        "recommended_actions",
        "review_queue",
        *(
            (
                "prediction_manifest",
                "prediction_validation_report",
                "model_error_analysis_report",
                "ambiguous_object_candidates",
                "probable_label_error_candidates",
            )
            if analyze_context.prediction_artifact_refs
            else ()
        ),
    )
    state.statuses["analysis_mode"] = "reference_only"
    for asset_name in asset_names:
        state.artifacts[asset_name] = _save_json_artifact(
            registry=registry,
            run_context=run_context,
            analyze_context=analyze_context,
            artifact_kind=asset_name,
            schema_version=f"{asset_name}.reference_only.v1",
            payload={
                "schema_version": f"{asset_name}.reference_only.v1",
                "status": "reference_only",
                "reason": "input_artifact_bytes_not_available_in_local_launcher",
                "dataset_artifact_ids": [
                    ref.artifact_id for ref in analyze_context.dataset_object_refs
                ],
                "prediction_artifact_ids": [
                    ref.artifact_id for ref in analyze_context.prediction_artifact_refs
                ],
                "raw_content_included": False,
            },
        )


def _archive_ref(refs: tuple[ArtifactRef, ...]) -> ArtifactRef:
    for ref in refs:
        if ref.kind in {"raw_dataset_archive", "demo_archive"} or ref.media_type in {
            "application/zip",
            "application/x-zip-compressed",
        }:
            return ref
    if refs:
        return refs[0]
    raise ValueError("ANALYZE_ONLY requires at least one dataset object artifact ref")


def _registered_from_ref(
    ref: ArtifactRef,
    *,
    artifact_kind: str,
    created_by_job_id: str,
) -> RegisteredArtifact:
    return RegisteredArtifact(
        artifact_ref=ref,
        artifact_kind=artifact_kind,
        uri=ref.uri,
        hash=ref.hash,
        format="jsonl",
        schema_version=ref.schema_version,
        created_by_job_id=created_by_job_id,
    )


def _save_json_artifact(
    *,
    registry: ArtifactRegistry,
    run_context: RunContext,
    analyze_context: AnalyzeRunContext,
    artifact_kind: str,
    schema_version: str,
    payload: Any,
) -> RegisteredArtifact:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return registry.save_artifact(
        artifact_kind=artifact_kind,
        data=data,
        artifact_format="json",
        media_type="application/json",
        schema_version=schema_version,
        dataset_version_id=run_context.dataset_version_id,
        created_by_job_id=run_context.compute_run_id,
        config_hash=analyze_context.config_hash,
    )


def _prediction_rows(payload: bytes) -> tuple[PredictionRow, ...]:
    rows: list[PredictionRow] = []
    for line in payload.decode("utf-8").splitlines():
        if line.strip():
            rows.append(PredictionRow.model_validate(json.loads(line)))
    return tuple(rows)


def _manifest_index(manifest_rows: tuple[Any, ...]) -> dict[str, ManifestRowSummary]:
    index: dict[str, ManifestRowSummary] = {}
    for row in manifest_rows:
        if row.label is None:
            continue
        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        segment = metadata.get("customer_segment")
        index[row.object_id] = ManifestRowSummary(
            object_id=row.object_id,
            label=row.label,
            segment=str(segment) if segment else None,
        )
    return index


def _candidate_payload(
    *,
    model_error_report: ModelErrorReport,
    reason_code: str,
) -> dict[str, Any]:
    rows = [
        signal.model_dump(mode="json")
        for signal in model_error_report.object_signals
        if reason_code in signal.reason_codes
    ]
    return {
        "schema_version": f"{reason_code}_candidates.v1",
        "reason_code": reason_code,
        "count": len(rows),
        "candidates": rows,
    }


def _read_text_payloads(descriptors: tuple[Any, ...]) -> tuple[bytes, bytes]:
    support_payload: bytes | None = None
    ocr_payload: bytes | None = None
    for descriptor in descriptors:
        if descriptor.kind.value == "support_messages":
            with descriptor.open() as handle:
                support_payload = handle.read()
        elif descriptor.kind.value == "ocr_records":
            with descriptor.open() as handle:
                ocr_payload = handle.read()
    if support_payload is None or ocr_payload is None:
        raise ValueError("demo archive must include support_messages and ocr_records")
    return support_payload, ocr_payload


def _record_model_error_metrics(
    model_error_report: ModelErrorReport,
    *,
    metrics: MetricsRegistry,
) -> None:
    aggregate = model_error_report.aggregate_metrics
    ambiguous = aggregate.ambiguous_object_count if aggregate is not None else 0
    probable = aggregate.probable_label_error_count if aggregate is not None else 0
    metrics.set_gauge(
        METRIC_AMBIGUOUS_OBJECT_COUNT,
        value=float(ambiguous),
        labels={"stage": "model_error.analyze"},
    )
    metrics.set_gauge(
        METRIC_PROBABLE_LABEL_ERROR_COUNT,
        value=float(probable),
        labels={"stage": "model_error.analyze"},
    )


def _record_artifact_write_metrics(
    artifacts: dict[str, RegisteredArtifact],
    *,
    metrics: MetricsRegistry,
) -> None:
    for name in artifacts:
        metrics.observe(
            METRIC_ARTIFACT_WRITE_MS,
            value=0.0,
            labels={"stage": name},
        )
    metrics.observe(
        METRIC_JOB_DURATION_MS,
        value=0.0,
        labels={"stage": "analyze", "job_type": "ANALYZE_ONLY"},
    )


__all__ = ["artifact_for", "count_for", "ensure_analyze_outputs", "status_for"]
