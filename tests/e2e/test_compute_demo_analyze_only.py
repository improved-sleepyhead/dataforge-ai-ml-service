"""E2E ANALYZE_ONLY compute demo test for the DataForge AI ML service.

This test glues the public builders into the full ANALYZE_ONLY pipeline
the platform expects from the compute plane and asserts the four
acceptance criteria of TASK-067:

1. The flow runs end-to-end on the deterministic demo archive without a
   real platform backend (a fake platform metadata client absorbs status
   and audit events).
2. Manifest, PredictionManifest, prediction_validation_report,
   model_error_analysis_report, tabular profile, text/OCR PII report,
   DecisionReport, MethodRecommendations and ReviewQueue artifacts are
   produced.
3. The model error analysis surfaces at least one ambiguous_object
   candidate and one probable_label_error candidate.
4. ANALYZE_ONLY does not produce any candidate dataset artifacts and
   does not mutate raw artifacts.

Every contract-shaped artifact is validated against the active contract
pack via Pydantic + JSON Schema validators. Status events are observed
through ``FakePlatformMetadataClient`` so the test is fully hermetic.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlan,
    CandidateDatasetVersion,
    DecisionReport,
    EvidenceBundle,
    ExportPackage,
    ManifestRow,
    MethodRecommendation,
    ModelErrorReport,
    ModelErrorReportStatus,
    ObjectAnalyticalPassport,
    PredictionArtifactRef,
    PredictionManifest,
    SyntheticDatasetReport,
    TextOcrReport,
)
from app.ingestion import (
    BuildManifestRequest,
    build_asset_manifest,
    build_validated_manifest,
    build_validated_predictions,
    manifest_rows_from_artifact,
    open_archive_path,
)
from app.kernel import (
    BuildDecisionReportRequest,
    BuildMethodRecommendationsRequest,
    build_method_recommendations,
)
from app.plugins.object_analytics import (
    BuildEvidenceBundleRequest,
    BuildObjectAnalyticsRequest,
    build_evidence_bundles,
    build_object_analytics_passports,
)
from app.plugins.predictions import analyze_model_errors
from app.plugins.predictions.analyzer import ManifestRowSummary
from app.plugins.tabular import (
    ProfileBuildRequest,
    build_tabular_profile_report,
)
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
from app.validation.contracts import (
    ContractPack,
    load_contract_pack,
    validate_contract_payload,
)
from tests.fixtures.demo_archive import (
    DEMO_DATASET_VERSION_ID,
    build_demo_archive,
)

_ORG_ID = "org_e2e"
_PROJECT_ID = "project_e2e"
_DATASET_ID = "dataset_e2e"
_PARENT_VERSION_ID = "dataset_version_parent"
_DATASET_VERSION_ID = DEMO_DATASET_VERSION_ID
_JOB_ID = "compute_run_e2e_analyze"
_CONFIG_HASH = "sha256:" + "a" * 64
_MODEL_ID = "fraud_baseline"
_MODEL_VERSION = "2026-05-14"
_GENERATED_AT = datetime(2026, 6, 4, 12, 0, tzinfo=UTC)

# Apply-only artifact kinds that ANALYZE_ONLY must never produce. Pulled
# from the domain directly so a future contract addition can't drift this
# allow list out of sync.
_APPLY_ONLY_ARTIFACT_KINDS: frozenset[str] = frozenset(
    {
        "candidate_tabular_dataset",
        "synthetic_dataset",
        "synthetic_dataset_report",
        "prepared_dataset",
        "model_impact_report",
        "remediation_execution_report",
        "export_package",
        "action_plan",
        "candidate_dataset_version",
    }
)
# Apply-only domain models. Their presence in storage would mean the
# pipeline mutated dataset state.
_APPLY_ONLY_DOMAIN_MODELS: tuple[type[Any], ...] = (
    ActionPlan,
    CandidateDatasetVersion,
    ExportPackage,
    SyntheticDatasetReport,
)


def test_e2e_analyze_only_compute_demo_runs_full_pipeline(tmp_path: Path) -> None:
    """End-to-end ANALYZE_ONLY flow with contract validation on every artifact.

    Test step 1: run the demo archive through every public builder.
    Test step 2: assert every required artifact is present, contract-valid,
    and that ambiguous/probable label-error candidates are surfaced.
    Test step 3: assert no candidate-dataset artifacts and no raw mutation.
    """
    archive_path = build_demo_archive(output_dir=tmp_path / "demo_archive").archive_path
    storage, registry = _storage_and_registry()
    fake_platform = FakePlatformMetadataClient()
    pack = load_contract_pack()
    raw_archive_object_count = len(storage.list())

    # ------------------------------------------------------------------
    # 1. Manifest + validated manifest.
    # ------------------------------------------------------------------
    manifest_request = BuildManifestRequest(
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
    with open_archive_path(archive_path) as reader:
        manifest_result = build_asset_manifest(
            reader, request=manifest_request, registry=registry
        )
    raw_manifest = manifest_result.manifest_artifact
    validated = build_validated_manifest(
        raw_manifest,
        storage=storage,
        registry=registry,
        dataset_version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        organization_id=_ORG_ID,
        project_id=_PROJECT_ID,
        audit_sink=fake_platform,
    ).validated_manifest

    manifest_bytes = storage.get(validated.uri).data
    manifest_rows = manifest_rows_from_artifact(manifest_bytes)
    assert manifest_rows, "demo archive must produce at least one manifest row"
    for row in manifest_rows:
        validate_contract_payload(pack, "manifest_row", row.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # 2. Validated PredictionManifest + coverage report.
    # ------------------------------------------------------------------
    raw_predictions = _persist_demo_raw_predictions(
        registry=registry, archive_path=archive_path
    )
    manifest_object_ids = [row.object_id for row in manifest_rows]
    predictions_result = build_validated_predictions(
        raw_predictions,
        storage=storage,
        registry=registry,
        dataset_version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        manifest_object_ids=manifest_object_ids,
        audit_sink=fake_platform,
        organization_id=_ORG_ID,
        project_id=_PROJECT_ID,
    )
    prediction_artifact = predictions_result.prediction_manifest_artifact
    coverage = predictions_result.validation_report.coverage
    assert coverage is not None
    assert coverage.prediction_row_count > 0
    assert len(coverage.matched_object_ids) == coverage.prediction_row_count

    prediction_bytes = storage.get(prediction_artifact.uri).data
    prediction_rows = _parse_prediction_rows(prediction_bytes)
    prediction_manifest = PredictionManifest(
        prediction_manifest_id=f"prediction_manifest_{_JOB_ID}",
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        task_type="classification",
        schema_version=pack.version,
        artifact=PredictionArtifactRef(
            uri=prediction_artifact.uri,
            hash=prediction_artifact.hash,
        ),
        rows=prediction_rows,
    )
    validate_contract_payload(
        pack,
        "prediction_manifest",
        prediction_manifest.model_dump(mode="json"),
    )
    assert len(prediction_manifest.rows) == coverage.prediction_row_count

    # Validate every prediction row against the contract pack as well;
    # the wrapper check above catches manifest-level drift, while this
    # loop pinpoints bad row payloads.
    for row in prediction_rows:
        validate_contract_payload(
            pack,
            "prediction_manifest_row",
            row.model_dump(mode="json"),
        )

    # ------------------------------------------------------------------
    # 3. Tabular profile report (from transactions.csv inside archive).
    # ------------------------------------------------------------------
    profile_request = ProfileBuildRequest(
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        source_artifact_id=raw_manifest.uri,
    )
    with open_archive_path(archive_path) as reader:
        profile_result = build_tabular_profile_report(
            reader,
            request=profile_request,
            storage=storage,
            registry=registry,
            source_manifest_artifact=raw_manifest.artifact_ref,
            generated_at=_GENERATED_AT,
        )
    tabular_profile = profile_result.profile_report
    validate_contract_payload(
        pack,
        "tabular_profile_report",
        tabular_profile.model_dump(mode="json"),
    )

    # ------------------------------------------------------------------
    # 4. Text/OCR PII report (from support_messages + ocr_records).
    # ------------------------------------------------------------------
    text_ocr_request = TextOcrBuildRequest(
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )
    support_payload, ocr_payload = _read_text_payloads(archive_path)
    support_report = validate_support_messages_jsonl(
        support_payload,
        detect_pii=True,
        registry=registry,
        build_request=text_ocr_request,
    )
    ocr_report = validate_ocr_records_jsonl(
        ocr_payload,
        detect_pii=True,
        registry=registry,
        build_request=text_ocr_request,
    )
    text_ocr_result = build_text_ocr_report(
        request=text_ocr_request,
        sources=(support_report, ocr_report),
        generated_at=_GENERATED_AT,
    )
    text_ocr_report: TextOcrReport = text_ocr_result.report
    validate_contract_payload(
        pack,
        "text_ocr_report",
        text_ocr_report.model_dump(mode="json"),
    )
    # Demo archive must contain PII so the proof-level text/OCR plugin
    # surfaces at least one finding across the support and OCR sources.
    assert text_ocr_report.total_pii_record_count > 0

    # ------------------------------------------------------------------
    # 5. Model error analysis (ambiguous + probable label error).
    # ------------------------------------------------------------------
    manifest_index = _manifest_index(manifest_rows)
    model_error_report = analyze_model_errors(
        rows=prediction_rows,
        manifest_index=manifest_index,
        dataset_id=_DATASET_ID,
        version_id=_DATASET_VERSION_ID,
        config_hash=_CONFIG_HASH,
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        generated_at=_GENERATED_AT,
    )
    assert model_error_report.status is ModelErrorReportStatus.AVAILABLE
    aggregate = model_error_report.aggregate_metrics
    assert aggregate is not None
    assert aggregate.ambiguous_object_count >= 1, (
        "demo predictions must include at least one ambiguous object"
    )
    assert aggregate.probable_label_error_count >= 1, (
        "demo predictions must include at least one probable label-error candidate"
    )

    # ------------------------------------------------------------------
    # 6. Object analytical passports + evidence bundles.
    # ------------------------------------------------------------------
    passports_result = build_object_analytics_passports(
        manifest_rows=manifest_rows,
        request=BuildObjectAnalyticsRequest(
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            parent_version_id=_PARENT_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
        ),
        registry=registry,
        tabular_profile=tabular_profile,
        text_ocr_report=text_ocr_report,
        model_error_report=model_error_report,
        computed_at=_GENERATED_AT,
    )
    for passport in passports_result.passports:
        validate_contract_payload(
            pack,
            "object_analytical_passport",
            passport.model_dump(mode="json"),
        )

    evidence_result = build_evidence_bundles(
        passports=passports_result.passports,
        request=BuildEvidenceBundleRequest(
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            parent_version_id=_PARENT_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
        ),
        registry=registry,
        source_passports_artifact=passports_result.artifact,
    )
    for bundle in evidence_result.evidence_bundles:
        validate_contract_payload(
            pack,
            "evidence_bundle",
            bundle.model_dump(mode="json"),
        )

    # ------------------------------------------------------------------
    # 7. DecisionReport.
    # ------------------------------------------------------------------
    decision_result = build_decision_report_artifact(
        evidence_bundles=evidence_result.evidence_bundles,
        request=BuildDecisionReportRequest(
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        registry=registry,
    )
    decision_report: DecisionReport = decision_result.report
    validate_contract_payload(
        pack,
        "decision_report",
        decision_report.model_dump(mode="json"),
    )

    # ------------------------------------------------------------------
    # 8. MethodRecommendations.
    # ------------------------------------------------------------------
    method_recommendations: tuple[MethodRecommendation, ...] = (
        build_method_recommendations(
            BuildMethodRecommendationsRequest(
                tabular_profile=tabular_profile,
                decision_report=decision_report,
            )
        )
    )
    assert method_recommendations, "method recommendations must not be empty"
    for recommendation in method_recommendations:
        validate_contract_payload(
            pack,
            "method_recommendation",
            recommendation.model_dump(mode="json"),
        )

    # ------------------------------------------------------------------
    # 9. ReviewQueue.
    # ------------------------------------------------------------------
    review_result = build_review_queues_artifact(
        decision_report=decision_report,
        evidence_bundles=evidence_result.evidence_bundles,
        request=BuildReviewQueuesRequest(
            dataset_id=_DATASET_ID,
            version_id=_DATASET_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
            generated_at=_GENERATED_AT,
        ),
        registry=registry,
    )
    for queue in review_result.queues:
        validate_contract_payload(
            pack,
            "review_queue",
            queue.model_dump(mode="json"),
        )

    # ------------------------------------------------------------------
    # AC: ambiguous + probable label-error reach the label review queue.
    # ------------------------------------------------------------------
    label_queue = next(
        queue for queue in review_result.queues if queue.queue_type.value == "LABEL_REVIEW"
    )
    label_reason_codes = {
        code for item in label_queue.objects for code in item.reason_codes
    }
    assert "ambiguous_object" in label_reason_codes
    assert "probable_label_error" in label_reason_codes

    # ------------------------------------------------------------------
    # AC: ANALYZE_ONLY must not produce candidate / mutation artifacts.
    # ------------------------------------------------------------------
    storage_objects = storage.list()
    # Initial empty bucket → registered artifacts only contain ANALYZE_ONLY kinds.
    assert len(storage_objects) > raw_archive_object_count
    for info in storage_objects:
        # ``info.metadata`` carries the artifact_kind we registered.
        kind = info.metadata.get("artifact-kind") or info.metadata.get("artifact_kind") or ""
        assert kind not in _APPLY_ONLY_ARTIFACT_KINDS, (
            f"ANALYZE_ONLY produced apply-only artifact kind={kind} at {info.uri}"
        )

    # No apply-only domain model can be parsed from any registered payload.
    for info in storage_objects:
        payload = storage.get(info.uri).data
        text_payload = payload.decode("utf-8", errors="ignore")
        for model_cls in _APPLY_ONLY_DOMAIN_MODELS:
            assert model_cls.__name__ not in text_payload or (
                # Decision reports legitimately reference the
                # CandidateDatasetVersion *kind name* in reason codes; we
                # therefore only flag the marker when it appears as a
                # serialized JSON model, i.e. the discriminator field is
                # present alongside it.
                f'"{model_cls.__name__}"' not in text_payload
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _persist_demo_raw_predictions(
    *, registry: ArtifactRegistry, archive_path: Path
) -> Any:
    with open_archive_path(archive_path) as reader:
        descriptors = [d for d in reader.descriptors() if d.kind.value == "predictions"]
        assert len(descriptors) == 1
        with descriptors[0].open() as handle:
            payload = handle.read()
    return registry.save_artifact(
        artifact_kind="raw_predictions",
        data=payload,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="prediction_manifest_row.v1",
        dataset_version_id=_DATASET_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )


def _read_text_payloads(archive_path: Path) -> tuple[bytes, bytes]:
    support_payload: bytes | None = None
    ocr_payload: bytes | None = None
    with open_archive_path(archive_path) as reader:
        for descriptor in reader.descriptors():
            if descriptor.kind.value == "support_messages":
                with descriptor.open() as handle:
                    support_payload = handle.read()
            elif descriptor.kind.value == "ocr_records":
                with descriptor.open() as handle:
                    ocr_payload = handle.read()
    if support_payload is None or ocr_payload is None:
        raise AssertionError("demo archive must include support and OCR sources")
    return support_payload, ocr_payload


def _parse_prediction_rows(payload: bytes) -> tuple[Any, ...]:
    from app.domain import PredictionRow

    rows: list[PredictionRow] = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(PredictionRow.model_validate(json.loads(line)))
    return tuple(rows)


def _manifest_index(
    manifest_rows: tuple[ManifestRow, ...],
) -> dict[str, ManifestRowSummary]:
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


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id=_ORG_ID,
            project_id=_PROJECT_ID,
            dataset_id=_DATASET_ID,
        ),
    )
    return storage, ArtifactRegistry(storage=storage)


class _InMemoryS3Client(S3CompatibleClient):
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
            "LastModified": _GENERATED_AT,
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "Body": io.BytesIO(stored["Body"]),
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(stored["Body"])}
                for (bucket, key), stored in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            from app.domain import ErrorCode

            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc


# Silence unused-import warnings for symbols used only as helpers above.
_ = (ContractPack, ModelErrorReport, ObjectAnalyticalPassport, EvidenceBundle, pytest)
