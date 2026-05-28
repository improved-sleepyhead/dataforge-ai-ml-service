"""Validated manifest assembly for the DataForge AI ingestion layer.

TASK-022 wires three pieces together:

* a manifest contract validator that runs JSON Schema + pydantic against
  every JSONL row of an Asset Manifest and surfaces a
  ``CONTRACT_VALIDATION_FAILED`` error when a row does not match the
  contract;
* a ``build_validated_manifest`` function that consumes the raw
  ``RegisteredArtifact`` produced by :func:`build_asset_manifest`, runs
  contract validation, and saves the same bytes through
  :class:`ArtifactRegistry` under the immutable ``validated_manifest``
  artifact kind so the validated manifest itself is content-addressed and
  carries lineage to the parent dataset version;
* a tiny compute audit publisher that records input/output artifact hashes
  through the platform adapter so audit can prove which raw manifest was
  validated and which validated manifest came out of it.

Strict constraints honored here:

* the source JSONL bytes are never mutated; the validated manifest is a
  new immutable artifact;
* errors are raised as :class:`ManifestContractValidationError` with the
  stable :class:`ErrorCode.CONTRACT_VALIDATION_FAILED` and a machine
  readable ``reason_code`` so the API layer can return a safe
  :class:`ErrorResponse` without exposing raw payload content;
* compute audit events carry technical IDs and hashes only, never the row
  payload itself;
* the validator is independent of any plugin and can be reused by future
  tasks (text/OCR, image, etc.) that emit their own JSONL row contracts.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from app.adapters import (
    ArtifactRegistry,
    AuditEventType,
    FakePlatformMetadataClient,
    PlatformAuditEvent,
    RegisteredArtifact,
)
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import ErrorCode, ManifestRow
from app.domain.common import NonEmptyStr, Sha256Digest
from app.validation.contracts import (
    ContractPack,
    ContractValidationError,
    load_contract_pack,
    validate_contract_payload,
)

MANIFEST_ROW_SCHEMA_NAME = "manifest_row"
VALIDATED_MANIFEST_KIND = "validated_manifest"
VALIDATED_MANIFEST_SCHEMA_VERSION = "manifest_row.v1"
VALIDATED_MANIFEST_FORMAT = "jsonl"
VALIDATED_MANIFEST_MEDIA_TYPE = "application/jsonl"


class ManifestContractValidationError(ValueError):
    """Raised when manifest JSONL fails contract validation.

    The exception carries the stable :class:`ErrorCode.CONTRACT_VALIDATION_FAILED`
    so the API boundary can build a safe :class:`ErrorResponse` from it. The
    ``reason_code`` field is a short machine-readable token (for example
    ``"missing_required_field"`` or ``"row_not_object"``) so the API layer can
    pick an appropriate ``remediation_hint`` without echoing raw payloads.
    """

    code = ErrorCode.CONTRACT_VALIDATION_FAILED

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        line_number: int | None = None,
        field_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.line_number = line_number
        self.field_path = field_path


@dataclass(frozen=True)
class ManifestValidationReport:
    """Result of a successful manifest contract validation pass."""

    row_count: int
    rows_by_modality: dict[str, int]
    input_hash: Sha256Digest
    schema_name: NonEmptyStr


@dataclass(frozen=True)
class BuildValidatedManifestResult:
    """Result of a successful ``build_validated_manifest`` call."""

    validated_manifest: RegisteredArtifact
    validation_report: ManifestValidationReport
    audit_event: PlatformAuditEvent


def validate_manifest_jsonl(
    data: bytes,
    *,
    contract_pack: ContractPack | None = None,
) -> ManifestValidationReport:
    """Validate manifest JSONL bytes against the manifest contract.

    Each non-empty line of ``data`` must be a JSON object that:

    * parses as JSON (``invalid_json``);
    * is a JSON object (``row_not_object``);
    * matches the ``manifest_row`` JSON Schema (``schema_violation``);
    * matches the :class:`ManifestRow` pydantic model
      (``pydantic_violation``); the pydantic validator carries domain-level
      invariants that JSON Schema alone cannot express (for example the
      hash regex, the strict ``extra="forbid"`` policy, and the lineage
      sub-model contract).

    The function returns a :class:`ManifestValidationReport` carrying the
    sha256 digest of the input bytes and the row count. It does not
    persist anything; callers (typically :func:`build_validated_manifest`)
    are responsible for writing the validated manifest as an immutable
    artifact.
    """
    pack = contract_pack if contract_pack is not None else load_contract_pack()

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestContractValidationError(
            reason_code="not_utf8_jsonl",
            message="manifest payload must be UTF-8 encoded JSONL",
        ) from exc

    if not text.strip():
        raise ManifestContractValidationError(
            reason_code="empty_manifest",
            message="manifest payload is empty",
        )

    rows_by_modality: dict[str, int] = {}
    row_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestContractValidationError(
                reason_code="invalid_json",
                message=f"manifest line {line_number} is not valid JSON",
                line_number=line_number,
            ) from exc

        if not isinstance(payload, dict):
            raise ManifestContractValidationError(
                reason_code="row_not_object",
                message=f"manifest line {line_number} is not a JSON object",
                line_number=line_number,
            )

        try:
            validate_contract_payload(pack, MANIFEST_ROW_SCHEMA_NAME, payload)
        except ContractValidationError as exc:
            raise ManifestContractValidationError(
                reason_code=_schema_reason_code(payload),
                message=f"manifest line {line_number} {exc}",
                line_number=line_number,
                field_path=_field_path_from_message(str(exc)),
            ) from exc

        try:
            row = ManifestRow.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            field_path = ".".join(str(part) for part in first.get("loc", ()))
            raise ManifestContractValidationError(
                reason_code="pydantic_violation",
                message=(
                    f"manifest line {line_number} failed pydantic validation: "
                    f"{first.get('msg', '')}"
                ),
                line_number=line_number,
                field_path=field_path or None,
            ) from exc

        rows_by_modality[row.modality.value] = (
            rows_by_modality.get(row.modality.value, 0) + 1
        )
        row_count += 1

    if row_count == 0:
        raise ManifestContractValidationError(
            reason_code="empty_manifest",
            message="manifest payload contained only blank lines",
        )

    return ManifestValidationReport(
        row_count=row_count,
        rows_by_modality=dict(rows_by_modality),
        input_hash=_sha256_of_bytes(data),
        schema_name=MANIFEST_ROW_SCHEMA_NAME,
    )


def build_validated_manifest(
    raw_manifest: RegisteredArtifact,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    dataset_version_id: str,
    parent_version_id: str,
    created_by_job_id: str,
    config_hash: str,
    contract_pack: ContractPack | None = None,
    audit_sink: FakePlatformMetadataClient | None = None,
    organization_id: str = "",
    project_id: str = "",
) -> BuildValidatedManifestResult:
    """Validate a raw Asset Manifest artifact and persist a validated artifact.

    Strict guarantees:

    * if any row fails validation, no validated artifact is written and a
      :class:`ManifestContractValidationError` is raised;
    * on success the validated manifest is saved with the same bytes
      through :class:`ArtifactRegistry` under the immutable
      ``validated_manifest`` kind, so the validated artifact has its own
      content-addressed URI, hash, and lineage;
    * a compute audit event is recorded carrying both ``input_hash`` (the
      raw manifest digest) and ``output_hash`` (the validated manifest
      digest), so audit can prove which manifest was validated into which.
    """
    raw_object = storage.get(raw_manifest.uri)
    raw_bytes = raw_object.data
    if raw_object.info.hash != raw_manifest.hash:
        raise ManifestContractValidationError(
            reason_code="raw_hash_mismatch",
            message="raw manifest object hash does not match registry record",
        )

    report = validate_manifest_jsonl(raw_bytes, contract_pack=contract_pack)

    validated_artifact = registry.save_artifact(
        artifact_kind=VALIDATED_MANIFEST_KIND,
        data=raw_bytes,
        artifact_format=VALIDATED_MANIFEST_FORMAT,
        media_type=VALIDATED_MANIFEST_MEDIA_TYPE,
        schema_version=VALIDATED_MANIFEST_SCHEMA_VERSION,
        dataset_version_id=dataset_version_id,
        created_by_job_id=created_by_job_id,
        config_hash=config_hash,
        metadata={
            "row_count": str(report.row_count),
            "validated_from_artifact_kind": raw_manifest.artifact_kind,
        },
    )

    audit_event = PlatformAuditEvent(
        audit_event_id=f"audit_{uuid.uuid4().hex}",
        event_type=AuditEventType.MANIFEST_VALIDATED,
        organization_id=organization_id or "org_compute_plane",
        project_id=project_id or "project_compute_plane",
        metadata={
            "compute_run_job_id": created_by_job_id,
            "dataset_version_id": dataset_version_id,
            "parent_version_id": parent_version_id,
            "config_hash": config_hash,
            "input_artifact_kind": raw_manifest.artifact_kind,
            "input_artifact_uri": raw_manifest.uri,
            "input_hash": raw_manifest.hash,
            "output_artifact_kind": validated_artifact.artifact_kind,
            "output_artifact_uri": validated_artifact.uri,
            "output_hash": validated_artifact.hash,
            "schema_version": VALIDATED_MANIFEST_SCHEMA_VERSION,
            "row_count": report.row_count,
            "rows_by_modality": dict(report.rows_by_modality),
        },
    )
    if audit_sink is not None:
        audit_event = audit_sink.record_audit_event(audit_event)

    return BuildValidatedManifestResult(
        validated_manifest=validated_artifact,
        validation_report=report,
        audit_event=audit_event,
    )


def iter_manifest_rows(data: bytes) -> Iterable[ManifestRow]:
    """Yield :class:`ManifestRow` records for already-validated manifest bytes.

    This helper is intentionally separate from :func:`validate_manifest_jsonl`
    so callers that have already validated a manifest can stream rows
    without paying for validation twice.
    """
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        yield ManifestRow.model_validate_json(line)


def _schema_reason_code(payload: dict[str, object]) -> str:
    required = (
        "object_id",
        "dataset_id",
        "version_id",
        "modality",
        "asset_uri",
        "hash",
        "metadata",
        "lineage",
    )
    for field in required:
        if field not in payload:
            return "missing_required_field"
    return "schema_violation"


def _field_path_from_message(message: str) -> str | None:
    marker = "validation failed at "
    if marker not in message:
        return None
    tail = message.split(marker, 1)[1]
    return tail.split(":", 1)[0] or None


def _sha256_of_bytes(data: bytes) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


__all__ = [
    "BuildValidatedManifestResult",
    "MANIFEST_ROW_SCHEMA_NAME",
    "ManifestContractValidationError",
    "ManifestValidationReport",
    "VALIDATED_MANIFEST_FORMAT",
    "VALIDATED_MANIFEST_KIND",
    "VALIDATED_MANIFEST_MEDIA_TYPE",
    "VALIDATED_MANIFEST_SCHEMA_VERSION",
    "build_validated_manifest",
    "iter_manifest_rows",
    "validate_manifest_jsonl",
]
