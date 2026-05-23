"""Tests for TASK-029: Object Analytical Passport builder."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ArtifactLineage,
    ArtifactRef,
    ClassCount,
    ClassImbalanceDiagnostics,
    DataModality,
    DuplicateDiagnostics,
    ErrorCode,
    EvidenceRef,
    ManifestLineage,
    ManifestRow,
    ModelErrorAggregateMetrics,
    ModelErrorReport,
    ModelErrorReportStatus,
    ModelErrorThresholds,
    ObjectAnalyticalPassport,
    ObjectModelErrorSignals,
    PiiCategory,
    PiiFinding,
    SignalStatus,
    TabularProfileLineage,
    TabularProfileReport,
    TextDuplicateGroup,
    TextOcrReport,
    TextOcrSourceKind,
    TextOcrSourceReport,
    TextPiiFindingsForRecord,
)
from app.plugins.object_analytics import (
    OBJECT_ANALYTICS_ARTIFACT_KIND,
    OBJECT_ANALYTICS_SCHEMA_VERSION,
    BuildObjectAnalyticsRequest,
    build_object_analytics_passports,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload

_DATASET_ID = "dataset_demo"
_VERSION_ID = "version_demo"
_PARENT_VERSION_ID = "version_parent"
_JOB_ID = "compute_run_object_analytics"
_CONFIG_HASH = "sha256:" + "a" * 64
_COMPUTED_AT = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)


def test_object_analytics_stage_persists_contract_valid_passports() -> None:
    """Step 1+2+3: build stage, check count, validate persisted passport artifact."""
    storage, registry = _storage_and_registry()
    rows = _manifest_rows()
    result = build_object_analytics_passports(
        manifest_rows=rows,
        request=_request(),
        registry=registry,
        tabular_profile=_tabular_profile(),
        text_ocr_report=_text_ocr_report(),
        model_error_report=_model_error_report(),
        evidence_refs=(
            EvidenceRef(kind="tabular_profile_report", uri=_artifact_ref().uri),
            EvidenceRef(kind="text_ocr_report", uri=_text_report_ref().uri),
        ),
        computed_at=_COMPUTED_AT,
    )

    assert len(result.passports) == len(rows)
    assert result.artifact.artifact_kind == OBJECT_ANALYTICS_ARTIFACT_KIND
    assert result.artifact.schema_version == OBJECT_ANALYTICS_SCHEMA_VERSION

    stored = storage.get(result.artifact.uri)
    assert stored.info.metadata["passport_count"] == str(len(rows))
    parsed = [
        ObjectAnalyticalPassport.model_validate(json.loads(line))
        for line in stored.data.decode("utf-8").splitlines()
    ]
    assert len(parsed) == len(rows)

    pack = load_contract_pack()
    for passport in parsed:
        validate_contract_payload(
            pack,
            "object_analytical_passport",
            passport.model_dump(mode="json"),
        )


def test_passports_merge_tabular_text_prediction_and_skeleton_signals() -> None:
    storage, registry = _storage_and_registry()
    rows = _manifest_rows()
    result = build_object_analytics_passports(
        manifest_rows=rows,
        request=_request(),
        registry=registry,
        tabular_profile=_tabular_profile(),
        text_ocr_report=_text_ocr_report(),
        model_error_report=_model_error_report(),
        computed_at=_COMPUTED_AT,
    )

    passports = {passport.object_id: passport for passport in result.passports}

    tabular = passports["obj_tabular_1"]
    assert tabular.identity.metadata["source_object_id"] == "txn_raw_1"
    assert tabular.technical_quality.status is SignalStatus.AVAILABLE
    assert tabular.duplicate_signals.duplicate_score == 1.0
    assert tabular.learning_value.rare_segment_score == 0.98
    assert tabular.learning_value.ambiguous_object_score == 0.76
    assert tabular.learning_value.probable_label_error_score == 0.0
    assert tabular.prediction is not None
    assert tabular.prediction.model_id == "fraud_baseline"
    assert "duplicate_candidate" in tabular.decision.reason_codes
    assert "ambiguous_object" in tabular.decision.reason_codes

    text = passports["obj_text_1"]
    assert text.privacy.pii_detected is True
    assert text.privacy.export_eligibility == "redacted_only"
    assert text.privacy.pii_types == ("email",)
    assert text.duplicate_signals.duplicate_score == 1.0
    assert text.duplicate_signals.duplicate_cluster_id == "sha256:" + "d" * 64
    assert "pii_detected" in text.decision.reason_codes

    skeleton = passports["obj_image_1"]
    assert skeleton.technical_quality.status is SignalStatus.NOT_APPLICABLE
    assert skeleton.privacy.status is SignalStatus.NOT_APPLICABLE
    assert skeleton.duplicate_signals.status is SignalStatus.NOT_APPLICABLE
    assert skeleton.learning_value.missing_reason == "skeleton_modality_not_profiled"
    assert skeleton.decision.reason_codes == ("validated_skeleton_only",)


def _request() -> BuildObjectAnalyticsRequest:
    return BuildObjectAnalyticsRequest(
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
    )


def _manifest_rows() -> tuple[ManifestRow, ...]:
    return (
        _manifest_row(
            object_id="obj_tabular_1",
            source_object_id="txn_raw_1",
            modality=DataModality.TABULAR,
            label="fraud",
            source_system="transactions",
        ),
        _manifest_row(
            object_id="obj_text_1",
            source_object_id="msg_raw_1",
            modality=DataModality.TEXT,
            label=None,
            source_system="support_messages",
        ),
        _manifest_row(
            object_id="obj_image_1",
            source_object_id="image_raw_1",
            modality=DataModality.IMAGE,
            label=None,
            source_system="image_manifest",
        ),
    )


def _manifest_row(
    *,
    object_id: str,
    source_object_id: str,
    modality: DataModality,
    label: str | None,
    source_system: str,
) -> ManifestRow:
    return ManifestRow(
        object_id=object_id,
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        modality=modality,
        asset_uri=f"s3://dataforge-manifest/versions/{_VERSION_ID}/objects/{object_id}",
        hash="sha256:" + object_id[-1] * 64,
        metadata={"source_object_id": source_object_id},
        lineage=ManifestLineage(
            source_artifact_id="raw_archive:demo",
            parent_version_id=_PARENT_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
        ),
        label=label,
        source_system=source_system,
    )


def _tabular_profile() -> TabularProfileReport:
    return TabularProfileReport(
        profile_id="tabular_profile_demo",
        source_system="transactions",
        row_count=2,
        column_count=2,
        columns=(),
        duplicates=DuplicateDiagnostics(
            duplicate_pair_count=1,
            duplicate_group_count=1,
            affected_object_ids=("txn_raw_1",),
            signature_columns=("amount",),
            id_column="object_id",
        ),
        class_imbalance=ClassImbalanceDiagnostics(
            target_column="is_fraud",
            total_samples=50,
            class_counts=(
                ClassCount(label="0", count=49),
                ClassCount(label="1", count=1),
            ),
            rare_class_label="1",
            rare_class_count=1,
            rare_class_ratio=0.02,
            minority_class_label="1",
            minority_class_share=0.02,
            imbalance_ratio=49.0,
            balance_score=0.25,
            balance_score_alternative=0.04,
            effective_number_beta=0.999,
            effective_number_of_samples={"0": 47.0, "1": 1.0},
        ),
        lineage=TabularProfileLineage(
            dataset_id=_DATASET_ID,
            version_id=_VERSION_ID,
            parent_version_id=_PARENT_VERSION_ID,
            created_by_job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
            source_manifest_artifact=_artifact_ref(),
            source_artifact_id="raw_archive:demo",
        ),
        generated_at=_COMPUTED_AT,
    )


def _text_ocr_report() -> TextOcrReport:
    source = TextOcrSourceReport(
        source_kind=TextOcrSourceKind.SUPPORT_MESSAGES,
        source_name="support_messages.jsonl",
        record_count=2,
        valid_record_count=2,
        issue_count=0,
        duplicate_groups=(
            TextDuplicateGroup(
                text_sha256="sha256:" + "d" * 64,
                object_ids=("msg_raw_1", "msg_raw_2"),
                count=2,
            ),
        ),
        duplicate_record_count=2,
        duplicate_object_ids=("msg_raw_1", "msg_raw_2"),
        average_text_length=24.0,
        min_text_length=12,
        max_text_length=36,
        pii_findings=(
            TextPiiFindingsForRecord(
                object_id="msg_raw_1",
                findings=(PiiFinding(category=PiiCategory.EMAIL, occurrence_count=1),),
                pii_token_count=1,
                pii_risk_score=0.5,
                redacted_text_sha256="sha256:" + "e" * 64,
            ),
        ),
        pii_token_count=1,
        pii_record_count=1,
        redacted_record_count=1,
        review_queue_object_ids=("msg_raw_1",),
    )
    return TextOcrReport(
        report_id="text_ocr_report_demo",
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        parent_version_id=_PARENT_VERSION_ID,
        created_by_job_id=_JOB_ID,
        config_hash=_CONFIG_HASH,
        sources=(source,),
        total_record_count=2,
        total_valid_record_count=2,
        total_issue_count=0,
        total_duplicate_group_count=1,
        total_duplicate_record_count=2,
        total_pii_record_count=1,
        total_pii_token_count=1,
        total_redacted_record_count=1,
        review_queue_object_ids=("msg_raw_1",),
        generated_at=_COMPUTED_AT,
    )


def _model_error_report() -> ModelErrorReport:
    return ModelErrorReport(
        report_id="model_error_report_demo",
        status=ModelErrorReportStatus.AVAILABLE,
        dataset_id=_DATASET_ID,
        version_id=_VERSION_ID,
        model_id="fraud_baseline",
        model_version="2026-05-14",
        classes=("fraud", "not_fraud"),
        aggregate_metrics=ModelErrorAggregateMetrics(
            accuracy=0.9,
            label_conflict_count=0,
            high_confidence_error_count=0,
            ambiguous_object_count=1,
            probable_label_error_count=0,
        ),
        object_signals=(
            ObjectModelErrorSignals(
                object_id="obj_tabular_1",
                true_label="fraud",
                predicted_label="fraud",
                confidence=0.52,
                margin=0.04,
                entropy=0.69,
                normalized_entropy=0.99,
                label_conflict=False,
                ambiguous_object_score=0.76,
                probable_label_error_score=0.0,
                reason_codes=("ambiguous_object", "low_prediction_margin"),
            ),
        ),
        thresholds=ModelErrorThresholds(),
        config_hash=_CONFIG_HASH,
        generated_at=_COMPUTED_AT,
    )


def _artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="validated_manifest:test",
        kind="validated_manifest",
        uri="s3://dataforge-local/dataforge/org_test/project_test/dataset_demo/manifest.jsonl",
        hash="sha256:" + "b" * 64,
        media_type="application/jsonl",
        size_bytes=100,
        schema_version="manifest_row.v1",
        lineage=ArtifactLineage(
            parent_version_id=_VERSION_ID,
            job_id=_JOB_ID,
            config_hash=_CONFIG_HASH,
            created_at=_COMPUTED_AT,
        ),
    )


def _text_report_ref() -> ArtifactRef:
    ref = _artifact_ref()
    return ref.model_copy(
        update={
            "artifact_id": "text_ocr_report:test",
            "kind": "text_ocr_report",
            "uri": "s3://dataforge-local/dataforge/org_test/project_test/dataset_demo/text.json",
            "schema_version": "text_ocr_report.v1",
        }
    )


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_test",
            project_id="project_test",
            dataset_id="dataset_demo",
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
            "LastModified": _COMPUTED_AT,
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        assert isinstance(body, bytes)
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(record["Body"])}
                for (bucket, key), record in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
