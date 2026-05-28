"""TASK-065: connected security/privacy test suite for the ML compute plane.

This suite proves the five acceptance criteria from ``tasks.json`` end-to-end
on real code paths (not mocks):

1. Tests verify no raw PII in logs.
2. Strict profile blocks external API access.
3. Object URI outside the signed project prefix is rejected.
4. Blocked objects are excluded from export artifacts.
5. Unsigned compute requests are rejected.

Each test wires together the real components — ``StructuredLogEvent`` +
log scanner, ``ServiceConfig`` + the external-AI policy gate,
``MinioObjectStorageAdapter`` + signed scope, the tabular export writer +
``ArtifactRegistry``, and the FastAPI signature dependency — so a regression
in any one of them surfaces here.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.adapters import (
    ArtifactRegistry,
    MinioObjectStorageAdapter,
    ObjectStorageError,
    ObjectStorageScope,
)
from app.adapters.object_storage import S3CompatibleClient
from app.api.main import create_app
from app.domain import ArtifactLineage, ArtifactRef, ErrorCode
from app.kernel.config import (
    DagsterSettings,
    ExternalAISettings,
    ObjectStorageSettings,
    PlatformSettings,
    PolicySettings,
    RuntimeProfile,
    ServiceConfig,
    load_config,
    profile_defaults,
)
from app.kernel.external_api_policy import (
    ExternalApiBlockedError,
    is_external_api_allowed,
    require_external_api_allowed,
)
from app.plugins.export.tabular_writer import (
    TabularExportRequest,
    write_tabular_export,
)
from app.telemetry import (
    LogStatus,
    StructuredLogEvent,
    emit_structured_log,
    scan_log_text,
)

# ---------------------------------------------------------------------------
# Acceptance criterion 1 — no raw PII in logs
# ---------------------------------------------------------------------------


def test_no_raw_pii_token_reaches_structured_compute_log() -> None:
    """Realistic compute-plane log carrying PII-like demo tokens must redact them."""
    logger, stream = _memory_logger("tests.security.privacy.no_pii_in_logs")

    emit_structured_log(
        logger,
        StructuredLogEvent(
            job_id="compute_run_security_001",
            project_id="project_1",
            dataset_id="dataset_1",
            version_id="dataset_version_v1",
            plugin="text_ocr_plugin",
            job_type="ANALYZE_ONLY",
            stage="text_ocr.profile.input_preview",
            duration_ms=128,
            status=LogStatus.COMPLETED,
            metadata={
                # Demo PII tokens that must NEVER reach logs.
                "preview_text": (
                    "Customer alice.example@bank.test phone +7 999 123 45 67 "
                    "passport 1234 567890 token=top-secret-bearer-xyz"
                ),
                "ocr_block": {
                    "lines": [
                        "card 4111 1111 1111 1111",
                        "email bob.example@bank.test",
                    ]
                },
                "row_count": 200,
            },
        ),
    )

    log_text = stream.getvalue()
    parsed = json.loads(log_text)

    # The structured event keeps technical fields...
    assert parsed["job_id"] == "compute_run_security_001"
    assert parsed["plugin"] == "text_ocr_plugin"
    assert parsed["stage"] == "text_ocr.profile.input_preview"
    assert parsed["status"] == "COMPLETED"
    assert parsed["metadata"]["row_count"] == 200

    # ...but every PII-like field is redacted before it reaches the stream.
    assert parsed["metadata"]["preview_text"] == "[REDACTED]"
    # Nested OCR block: the container shape may be preserved, but every
    # leaf containing a PII-like token must be redacted.
    nested_lines = parsed["metadata"]["ocr_block"]["lines"]
    assert nested_lines == ["card [REDACTED]", "email [REDACTED]"]

    # Defense-in-depth: scanner must agree that the rendered log line has
    # no raw PII tokens at all (no email, no phone, no passport, no card).
    assert "alice.example@bank.test" not in log_text
    assert "bob.example@bank.test" not in log_text
    assert "+7 999 123 45 67" not in log_text
    assert "1234 567890" not in log_text
    assert "4111 1111 1111 1111" not in log_text
    assert "top-secret-bearer-xyz" not in log_text
    assert scan_log_text(log_text).passed is True


def test_log_scanner_flags_raw_pii_tokens_when_redaction_is_skipped() -> None:
    """If a regression bypasses redaction, the scanner must catch the PII tokens."""
    raw = (
        "leaked log: customer analyst@example.com phone +1 415 555 0199 "
        "passport 1234 567890"
    )
    result = scan_log_text(raw)

    assert result.passed is False
    # The scanner returns stable, technical category names — never the raw values.
    assert "email" in result.violations
    assert "phone" in result.violations
    assert "passport" in result.violations


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — strict profile blocks external API
# ---------------------------------------------------------------------------


def test_banking_strict_profile_blocks_external_api_even_when_flag_is_true() -> None:
    """banking_strict must keep external API disabled even if the flag was flipped."""
    config = _build_config(
        profile=RuntimeProfile.BANKING_STRICT,
        allow_external_api=True,
    )

    assert is_external_api_allowed(config) is False
    with pytest.raises(ExternalApiBlockedError) as exc_info:
        require_external_api_allowed(config, provider="azure-openai")

    assert exc_info.value.code is ErrorCode.EXTERNAL_API_BLOCKED
    assert exc_info.value.reason_code == "banking_strict_profile_blocks_external_api"


def test_demo_strict_profile_blocks_external_api_by_default_flag() -> None:
    """demo_strict with the default ``allow_external_api=False`` must also block."""
    config = _build_config(
        profile=RuntimeProfile.DEMO_STRICT,
        allow_external_api=False,
    )

    assert is_external_api_allowed(config) is False
    with pytest.raises(ExternalApiBlockedError) as exc_info:
        require_external_api_allowed(config)

    assert exc_info.value.code is ErrorCode.EXTERNAL_API_BLOCKED
    assert exc_info.value.reason_code == "external_api_disabled_by_policy"


def test_demo_strict_profile_allows_external_api_when_explicitly_enabled() -> None:
    """When demo_strict explicitly enables external API, the gate lets traffic through."""
    config = _build_config(
        profile=RuntimeProfile.DEMO_STRICT,
        allow_external_api=True,
    )

    assert is_external_api_allowed(config) is True
    # No exception expected.
    require_external_api_allowed(config, provider="openai")


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — object URI outside project prefix is rejected
# ---------------------------------------------------------------------------


def test_object_uri_outside_signed_project_prefix_is_rejected() -> None:
    """Crossing project_id in an S3 URI must surface ARTIFACT_OUT_OF_SCOPE."""
    adapter = _scoped_adapter(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
    )

    with pytest.raises(ObjectStorageError) as exc_info:
        adapter.get(
            "s3://dataforge-local/dataforge/org_1/project_2/dataset_1/leak.json"
        )

    assert exc_info.value.code is ErrorCode.ARTIFACT_OUT_OF_SCOPE


def test_object_uri_outside_signed_organization_prefix_is_rejected() -> None:
    """Crossing organization_id is also out of scope."""
    adapter = _scoped_adapter(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
    )

    with pytest.raises(ObjectStorageError) as exc_info:
        adapter.head(
            "s3://dataforge-local/dataforge/org_other/project_1/dataset_1/file.json"
        )

    assert exc_info.value.code is ErrorCode.ARTIFACT_OUT_OF_SCOPE


def test_path_traversal_in_object_name_is_rejected() -> None:
    """Path traversal must never escape the dataset prefix."""
    adapter = _scoped_adapter(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
    )

    with pytest.raises(ObjectStorageError) as exc_info:
        adapter.put(
            object_name="../project_2/leak.json",
            data=b"{}",
            kind="report",
            media_type="application/json",
            schema_version="report.v1",
            parent_version_id="dataset_version_v1",
            job_id="compute_run_security_002",
            config_hash="sha256:" + "b" * 64,
        )

    assert exc_info.value.code is ErrorCode.ARTIFACT_OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — blocked objects are excluded from export
# ---------------------------------------------------------------------------


def test_tabular_export_writer_drops_blocked_object_ids_from_artifacts() -> None:
    """Blocked object_ids must NEVER reach the tabular export Parquet/CSV artifacts."""
    storage, registry = _storage_and_registry()
    candidate_artifact = _seed_candidate_csv(
        storage,
        rows=[
            {"object_id": "txn_001", "amount": "100", "is_fraud": "0"},
            {"object_id": "txn_002", "amount": "250", "is_fraud": "0"},
            {"object_id": "txn_003", "amount": "9999", "is_fraud": "1"},
            {"object_id": "txn_004", "amount": "75", "is_fraud": "0"},
            {"object_id": "txn_005", "amount": "5000", "is_fraud": "1"},
        ],
    )

    request = TabularExportRequest(
        dataset_id="dataset_1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=candidate_artifact,
        # Two objects are policy-blocked (e.g. PII / leakage / privacy review).
        blocked_object_ids=("txn_003", "txn_005"),
        write_csv=True,
        write_per_split=False,
        created_by_job_id="compute_run_security_export_001",
        config_hash="sha256:" + "c" * 64,
    )

    artifacts = write_tabular_export(request, storage=storage, registry=registry)

    # The writer reports it filtered exactly the two blocked rows out.
    assert artifacts.excluded_blocked_count == 2
    assert artifacts.included_row_count == 3

    # Re-read the published CSV bytes from object storage and confirm
    # blocked object_ids are NOT present anywhere in the exported file.
    assert artifacts.csv_artifact is not None
    stored_csv = storage.get(artifacts.csv_artifact.artifact_ref.uri)
    csv_text = stored_csv.data.decode("utf-8")
    csv_object_ids = _csv_object_ids(csv_text)
    assert "txn_003" not in csv_object_ids
    assert "txn_005" not in csv_object_ids
    assert csv_object_ids == ("txn_001", "txn_002", "txn_004")

    # Re-read Parquet bytes and confirm blocked object_ids are not there.
    # Parquet uses columnar/dictionary encoding so we cannot rely on bytes
    # search for present values, but the blocked tokens are unique demo
    # strings that must be entirely absent from every page/dictionary.
    parquet_bytes = storage.get(artifacts.parquet_artifact.artifact_ref.uri).data
    assert b"txn_003" not in parquet_bytes
    assert b"txn_005" not in parquet_bytes
    # Decoding the Parquet table gives the authoritative view of included rows.
    parquet_object_ids = _parquet_object_ids(parquet_bytes)
    assert "txn_003" not in parquet_object_ids
    assert "txn_005" not in parquet_object_ids
    assert parquet_object_ids == ("txn_001", "txn_002", "txn_004")


# ---------------------------------------------------------------------------
# Acceptance criterion 5 — unsigned compute requests are rejected
# ---------------------------------------------------------------------------


def test_unsigned_protected_compute_request_is_rejected_with_signature_error() -> None:
    """Protected compute endpoint must reject any request lacking platform signature."""
    config = _load_demo_config()
    client = TestClient(
        create_app(
            include_test_protected_route=True,
            config=config,
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/__test__/protected-compute-request",
        content=b'{"platform_job_id":"job","organization_id":"org_1","project_id":"project_1"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert body["error"]["details"] == {"reason_code": "missing_service_identity"}


def test_user_jwt_authorization_header_is_not_treated_as_service_identity() -> None:
    """A user Authorization Bearer header must not authorize protected compute calls."""
    client = TestClient(
        create_app(
            include_test_protected_route=True,
            config=_load_demo_config(),
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/__test__/protected-compute-request",
        content=b'{"platform_job_id":"job","organization_id":"org_1","project_id":"project_1"}',
        headers={
            "authorization": "Bearer user-jwt-must-not-authorize-compute",
            "content-type": "application/json",
        },
    )

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    # The user JWT must not leak through the response body, either as raw
    # text or inside the structured error details.
    assert "user-jwt-must-not-authorize-compute" not in response.text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _memory_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, stream


def _build_config(
    *,
    profile: RuntimeProfile,
    allow_external_api: bool,
) -> ServiceConfig:
    return ServiceConfig(
        profile=profile,
        object_storage=ObjectStorageSettings(
            endpoint_url="http://localhost:9000",
            bucket_name="dataforge-local",
        ),
        platform=PlatformSettings(
            callback_url="http://platform.local/api/ml/jobs/callback",
            service_signing_secret=SecretStr("local-dev-signing-secret"),
        ),
        dagster=DagsterSettings(home="/tmp/dataforge-dagster"),
        policies=PolicySettings(
            policy_config_path="configs/policies/demo_strict.yaml",
            decision_policy_path="configs/policies/decision_v0.yaml",
            score_policy_path="configs/policies/score_v0.yaml",
        ),
        contract_pack_version="local-fallback-v0.1.0-demo",
        external_ai=ExternalAISettings(allow_external_api=allow_external_api),
        profile_defaults=profile_defaults(profile),
    )


def _load_demo_config() -> ServiceConfig:
    return load_config(
        {
            "DATAFORGE_PROFILE": "demo_strict",
            "DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL": "http://localhost:9000",
            "DATAFORGE_OBJECT_STORAGE_BUCKET": "dataforge-local",
            "DATAFORGE_PLATFORM_CALLBACK_URL": "http://platform.local/api/ml/jobs/callback",
            "DATAFORGE_SERVICE_SIGNING_SECRET": "local-dev-signing-secret",
            "DATAFORGE_DAGSTER_HOME": "/tmp/dataforge-dagster",
            "DATAFORGE_POLICY_CONFIG_PATH": "configs/policies/demo_strict.yaml",
            "DATAFORGE_DECISION_POLICY_PATH": "configs/policies/decision_v0.yaml",
            "DATAFORGE_SCORE_POLICY_PATH": "configs/policies/score_v0.yaml",
        }
    )


def _scoped_adapter(
    *,
    organization_id: str,
    project_id: str,
    dataset_id: str,
) -> MinioObjectStorageAdapter:
    return MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id=organization_id,
            project_id=project_id,
            dataset_id=dataset_id,
        ),
    )


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = _scoped_adapter(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
    )
    return storage, ArtifactRegistry(storage=storage)


def _seed_candidate_csv(
    storage: MinioObjectStorageAdapter,
    *,
    rows: list[dict[str, str]],
) -> ArtifactRef:
    """Persist a candidate tabular CSV inside the signed scope."""
    columns = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    payload = buffer.getvalue().encode("utf-8")
    config_hash = "sha256:" + "c" * 64
    job_id = "compute_run_security_export_seed"
    artifact = storage.put(
        object_name="dataset_version_v2_candidate/candidate/transactions.csv",
        data=payload,
        kind="candidate_tabular_dataset",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        parent_version_id="dataset_version_v2_candidate",
        job_id=job_id,
        config_hash=config_hash,
    )
    return artifact.model_copy(
        update={
            "lineage": ArtifactLineage(
                parent_version_id="dataset_version_v2_candidate",
                job_id=job_id,
                config_hash=config_hash,
                created_at=datetime.now(UTC),
            ),
            "schema_version": "tabular_dataset.v1",
        }
    )


def _csv_object_ids(text: str) -> tuple[str, ...]:
    reader = csv.DictReader(io.StringIO(text))
    return tuple(row["object_id"] for row in reader)


def _parquet_object_ids(payload: bytes) -> tuple[str, ...]:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(payload))
    column = table.column("object_id").to_pylist()
    return tuple(str(value) for value in column)


class _InMemoryS3Client(S3CompatibleClient):
    """Minimal S3-compatible fake for the test suite (no real MinIO needed)."""

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
            "LastModified": datetime.now(UTC),
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

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
    ) -> Mapping[str, Any]:
        contents: list[Mapping[str, Any]] = []
        for (bucket, key), stored in sorted(self._objects.items()):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            contents.append(
                {
                    "Key": key,
                    "Size": len(stored["Body"]),
                    "LastModified": stored["LastModified"],
                }
            )
        return {"Contents": contents}

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
