"""Tests for TASK-043 REDACT_PII text/OCR action executor."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    ErrorCode,
    EvidenceRef,
    RetryPolicy,
    TextOcrSourceKind,
)
from app.ingestion import open_archive_path
from app.plugins.text_ocr import (
    REDACT_PII_STEP_TYPE,
    TEXT_OCR_REDACTED_ARTIFACT_KIND,
    TEXT_OCR_REDACTED_SCHEMA_VERSION,
    ExecuteTextOcrRedactionRequest,
    TextOcrExportPolicyStatus,
    evaluate_text_ocr_export_policy,
    execute_text_ocr_redaction_action,
)
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "d" * 64
_CREATED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("source_name", "source_kind", "raw_tokens"),
    [
        (
            "support_messages.jsonl",
            TextOcrSourceKind.SUPPORT_MESSAGES,
            ("alex@example.test", "+10000001234"),
        ),
        (
            "ocr_records.jsonl",
            TextOcrSourceKind.OCR_RECORDS,
            ("0001 100200",),
        ),
    ],
)
def test_redact_pii_action_writes_redacted_artifact_without_mutating_source(
    tmp_path: Path,
    source_name: str,
    source_kind: TextOcrSourceKind,
    raw_tokens: tuple[str, ...],
) -> None:
    """Steps 1-3: execute REDACT_PII, verify safe JSONL and lineage."""
    storage, registry = _storage_and_registry()
    source_bytes = _read_archive_entry(tmp_path, source_name)
    source_artifact = _source_artifact(
        registry=registry,
        data=source_bytes,
        artifact_kind=source_kind.value,
    )
    evidence = EvidenceRef(
        kind="text_ocr_report",
        uri="s3://dataforge-local/dataforge/org_1/project_1/dataset_1/reports/text_ocr.json",
    )

    result = execute_text_ocr_redaction_action(
        ExecuteTextOcrRedactionRequest(
            action_plan_id="action_plan_redact_pii_001",
            step=_redact_step(source_name=source_name, source_kind=source_kind),
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_kind=source_kind,
            source_name=source_name,
            source_artifact=source_artifact,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            evidence_refs=(evidence,),
        ),
        storage=storage,
        registry=registry,
    )

    assert storage.get(source_artifact.uri).data == source_bytes
    assert result.redacted_artifact.artifact_kind == TEXT_OCR_REDACTED_ARTIFACT_KIND
    assert result.redacted_artifact.schema_version == TEXT_OCR_REDACTED_SCHEMA_VERSION
    assert result.redacted_artifact.artifact_ref.lineage.parent_version_id == (
        "dataset_version_2_candidate"
    )
    assert result.redacted_artifact.artifact_ref.lineage.job_id == "compute_run_apply_001"

    stored = storage.get(result.redacted_artifact.uri)
    output_text = stored.data.decode("utf-8")
    for token in raw_tokens:
        assert token not in output_text
    assert "[REDACTED_" in output_text

    output_lines = [json.loads(line) for line in output_text.splitlines() if line]
    assert output_lines
    assert len(output_lines) == len(result.redacted_records)
    expected_keys = {"object_id", "pii_token_count", "redacted_text", "redacted_text_sha256"}
    assert all(set(line) == expected_keys for line in output_lines)
    assert all("text" not in line for line in output_lines)
    assert all(raw not in json.dumps(result.pii_findings, default=str) for raw in raw_tokens)

    metadata = stored.info.metadata
    assert metadata["action-plan-id"] == "action_plan_redact_pii_001"
    assert metadata["step-id"] == "redact_pii_text_ocr"
    assert metadata["source-artifact-hash"] == source_artifact.hash
    assert metadata["source-kind"] == source_kind.value
    assert metadata["source-name"] == source_name
    assert metadata["raw-export-policy"] == "redacted_only"
    assert int(metadata["pii-record-count"]) == result.pii_record_count
    assert int(metadata["pii-token-count"]) == result.pii_token_count
    assert json.loads(metadata["evidence-refs"]) == [evidence.model_dump(mode="json")]

    assert result.export_policy.status is TextOcrExportPolicyStatus.READY
    assert result.export_policy.raw_artifact_allowed is False
    assert result.export_policy.redacted_artifact_required is True
    assert result.export_policy.export_artifact == result.redacted_artifact.artifact_ref


def test_export_policy_blocks_raw_restricted_text_ocr_without_redaction() -> None:
    storage, registry = _storage_and_registry()
    source_artifact = _source_artifact(
        registry=registry,
        data=_jsonl([{"object_id": "message_1", "text": "email alex@example.test"}]),
        artifact_kind=TextOcrSourceKind.SUPPORT_MESSAGES.value,
    )

    blocked = evaluate_text_ocr_export_policy(
        source_artifact=source_artifact,
        pii_record_count=1,
    )

    assert blocked.status is TextOcrExportPolicyStatus.BLOCKED
    assert blocked.raw_artifact_allowed is False
    assert blocked.redacted_artifact_required is True
    assert blocked.export_artifact is None
    assert "raw_restricted_text_ocr_requires_redaction" in blocked.reason_codes
    assert "PII_UNMASKED" in blocked.reason_codes

    allowed = evaluate_text_ocr_export_policy(
        source_artifact=source_artifact,
        pii_record_count=0,
    )
    assert allowed.status is TextOcrExportPolicyStatus.READY
    assert allowed.raw_artifact_allowed is True
    assert allowed.export_artifact == source_artifact


def _redact_step(*, source_name: str, source_kind: TextOcrSourceKind) -> ActionPlanStep:
    return ActionPlanStep(
        step_id="redact_pii_text_ocr",
        type=REDACT_PII_STEP_TYPE,
        depends_on=(),
        idempotency_key="sha256:" + "1" * 64,
        method_id="deterministic_pii_redaction",
        plugin_id="dataforge.text_ocr",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="method_policy_v0",
        validation_gates=("privacy_check", "schema_validation"),
        preconditions=("source_version_is_immutable", "raw_text_ocr_requires_redaction"),
        input_artifacts=("s3://dataforge-local/dataforge/org_1/project_1/dataset_1/source.jsonl",),
        output_artifact_kind=TEXT_OCR_REDACTED_ARTIFACT_KIND,
        config={
            "source_kind": source_kind.value,
            "source_name": source_name,
            "include_clean_records": True,
        },
        random_seed=None,
        retry_policy=RetryPolicy(max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _source_artifact(
    *,
    registry: ArtifactRegistry,
    data: bytes,
    artifact_kind: str,
) -> ArtifactRef:
    return registry.save_artifact(
        artifact_kind=artifact_kind,
        data=data,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="text_ocr_source_jsonl.v1",
        dataset_version_id="dataset_version_1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


def _read_archive_entry(tmp_path: Path, name: str) -> bytes:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        for descriptor in reader.descriptors():
            if descriptor.name == name:
                return descriptor.read_bytes()
    raise AssertionError("archive entry " + repr(name) + " not found")


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
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
            "LastModified": _CREATED_AT,
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
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
