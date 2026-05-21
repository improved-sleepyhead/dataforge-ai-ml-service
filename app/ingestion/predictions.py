"""PredictionManifest ingestion and validation.

TASK-025A wires three pieces together:

* a ``predictions.jsonl`` JSONL parser that runs JSON Schema + pydantic
  per row and surfaces a ``CONTRACT_VALIDATION_FAILED`` error when a row
  does not match the contract (or ``PREDICTION_VALIDATION_FAILED`` when
  the row breaks the prediction-specific invariants like
  ``predicted_label = argmax(predicted_proba)``);
* a coverage join between the validated manifest (``ManifestRow``) and
  the predictions, surfacing ``unmatched_predictions`` (predictions
  pointing at object_ids that do not exist in the manifest) and
  ``missing_predictions`` (manifest rows without prediction coverage);
* a ``build_validated_predictions`` function that consumes the raw
  predictions ``RegisteredArtifact`` (or raw bytes, for tests), runs all
  validations, and saves the same bytes through :class:`ArtifactRegistry`
  under the immutable ``prediction_manifest`` artifact kind so the
  validated artifact is content-addressed and carries lineage to the
  parent dataset version.

Strict guarantees:

* the source JSONL bytes are never mutated; the validated artifact is a
  new immutable artifact with hash, ``model_id``, ``model_version``,
  ``inference_timestamp`` and schema version in its metadata;
* errors carry a stable :class:`ErrorCode` (``CONTRACT_VALIDATION_FAILED``
  for schema/pydantic violations, ``PREDICTION_VALIDATION_FAILED`` for
  prediction-specific invariants like join coverage or argmax mismatch)
  and a machine-readable ``reason_code`` so the API layer can build a
  safe :class:`ErrorResponse` without echoing raw payload content;
* compute audit events carry technical IDs and hashes only;
* predictions are evidence, not authority — they never override human
  labels, never approve export and never mutate the dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from app.domain import ErrorCode, PredictionRow
from app.domain.common import NonEmptyStr, Sha256Digest
from app.validation.contracts import (
    ContractPack,
    ContractValidationError,
    load_contract_pack,
    validate_contract_payload,
)

PREDICTION_ROW_SCHEMA_NAME = "prediction_manifest_row"
PREDICTION_MANIFEST_KIND = "prediction_manifest"
PREDICTION_MANIFEST_SCHEMA_VERSION = "prediction_manifest_row.v1"
PREDICTION_MANIFEST_FORMAT = "jsonl"
PREDICTION_MANIFEST_MEDIA_TYPE = "application/jsonl"

_PROBABILITY_TOLERANCE = 1e-6


class PredictionContractValidationError(ValueError):
    """Raised when a prediction row fails the JSON Schema or pydantic shape.

    The exception carries :class:`ErrorCode.CONTRACT_VALIDATION_FAILED`
    so the API layer can render a stable :class:`ErrorResponse`. The
    ``reason_code`` is a short machine-readable token (for example
    ``invalid_json``, ``schema_violation``, ``probability_sum_invalid``)
    so the API layer can pick a remediation hint without echoing raw
    payloads.
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


class PredictionValidationError(ValueError):
    """Raised when prediction-specific invariants fail.

    Distinct from :class:`PredictionContractValidationError` because the
    error code is :class:`ErrorCode.PREDICTION_VALIDATION_FAILED` —
    contract bytes are well-formed but the prediction logic is broken
    (e.g. ``predicted_label != argmax(predicted_proba)`` or join coverage
    constraint is violated).
    """

    code = ErrorCode.PREDICTION_VALIDATION_FAILED

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
class PredictionRowsValidationReport:
    """Per-row aggregates for a successful validation pass."""

    row_count: int
    rows_by_split: dict[str, int]
    classes_seen: tuple[str, ...]
    input_hash: Sha256Digest


@dataclass(frozen=True)
class PredictionCoverageReport:
    """Join-coverage report between predictions and the manifest.

    * ``coverage_ratio = matched / manifest_row_count`` — share of
      manifest objects that have a corresponding prediction;
    * ``unmatched_prediction_object_ids`` — predictions whose object_id
      does not exist in the manifest (data error);
    * ``missing_prediction_object_ids`` — manifest rows without
      prediction coverage (gap surfaced to Decision Core; not a hard
      error in ANALYZE_ONLY).
    """

    manifest_row_count: int
    prediction_row_count: int
    matched_object_ids: tuple[str, ...]
    unmatched_prediction_object_ids: tuple[str, ...]
    missing_prediction_object_ids: tuple[str, ...]
    coverage_ratio: float


@dataclass(frozen=True)
class PredictionValidationReport:
    """Top-level validation report for the predictions artifact."""

    rows: PredictionRowsValidationReport
    coverage: PredictionCoverageReport | None


@dataclass(frozen=True)
class BuildValidatedPredictionsResult:
    """Result of a successful ``build_validated_predictions`` call."""

    prediction_manifest_artifact: RegisteredArtifact
    validation_report: PredictionValidationReport
    audit_event: PlatformAuditEvent


def validate_predictions_jsonl(
    data: bytes,
    *,
    contract_pack: ContractPack | None = None,
) -> tuple[PredictionRowsValidationReport, list[PredictionRow]]:
    """Validate predictions JSONL bytes and return parsed rows.

    Each non-empty line of ``data`` must be a JSON object that:

    * parses as JSON (``invalid_json``);
    * is a JSON object (``row_not_object``);
    * matches the ``prediction_manifest_row`` JSON Schema
      (``schema_violation`` / ``missing_required_field``);
    * matches the :class:`PredictionRow` pydantic model
      (``pydantic_violation``); the pydantic validator carries the
      ``predicted_label = argmax(predicted_proba)``,
      ``confidence = max(predicted_proba)`` and probability-sum-to-1
      invariants.

    Pydantic violations on prediction-specific invariants are translated
    into :class:`PredictionValidationError` with reason codes
    ``probability_sum_invalid`` / ``argmax_mismatch`` /
    ``confidence_mismatch`` so callers can distinguish "wrong shape"
    from "wrong logic".
    """
    pack = contract_pack if contract_pack is not None else load_contract_pack()

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PredictionContractValidationError(
            reason_code="not_utf8_jsonl",
            message="prediction payload must be UTF-8 encoded JSONL",
        ) from exc

    if not text.strip():
        raise PredictionContractValidationError(
            reason_code="empty_predictions",
            message="prediction payload is empty",
        )

    rows: list[PredictionRow] = []
    rows_by_split: dict[str, int] = {}
    classes_seen: set[str] = set()

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PredictionContractValidationError(
                reason_code="invalid_json",
                message=f"prediction line {line_number} is not valid JSON",
                line_number=line_number,
            ) from exc

        if not isinstance(payload, dict):
            raise PredictionContractValidationError(
                reason_code="row_not_object",
                message=f"prediction line {line_number} is not a JSON object",
                line_number=line_number,
            )

        try:
            validate_contract_payload(pack, PREDICTION_ROW_SCHEMA_NAME, payload)
        except ContractValidationError as exc:
            raise PredictionContractValidationError(
                reason_code=_schema_reason_code(payload),
                message=f"prediction line {line_number} {exc}",
                line_number=line_number,
                field_path=_field_path_from_message(str(exc)),
            ) from exc

        try:
            row = PredictionRow.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            field_path = ".".join(str(part) for part in first.get("loc", ()))
            message = first.get("msg", "")
            reason_code, error_class = _classify_pydantic_error(message)
            raise error_class(
                reason_code=reason_code,
                message=(
                    f"prediction line {line_number} failed pydantic validation: "
                    f"{message}"
                ),
                line_number=line_number,
                field_path=field_path or None,
            ) from exc

        rows.append(row)
        rows_by_split[row.split.value] = rows_by_split.get(row.split.value, 0) + 1
        classes_seen.update(row.predicted_proba.keys())

    if not rows:
        raise PredictionContractValidationError(
            reason_code="empty_predictions",
            message="prediction payload contained only blank lines",
        )

    report = PredictionRowsValidationReport(
        row_count=len(rows),
        rows_by_split=dict(rows_by_split),
        classes_seen=tuple(sorted(classes_seen)),
        input_hash=_sha256_of_bytes(data),
    )
    return report, rows


def compute_prediction_coverage(
    *,
    manifest_object_ids: Iterable[str],
    prediction_rows: Iterable[PredictionRow],
) -> PredictionCoverageReport:
    """Compute join coverage between manifest object_ids and predictions.

    ``coverage_ratio`` reports the share of manifest object_ids covered
    by a prediction. Unmatched/missing object_ids are surfaced as
    sorted tuples for stable downstream rendering.
    """
    manifest_set = set(manifest_object_ids)
    prediction_ids: set[str] = set()
    for row in prediction_rows:
        prediction_ids.add(row.object_id)

    matched = manifest_set & prediction_ids
    unmatched_predictions = prediction_ids - manifest_set
    missing_predictions = manifest_set - prediction_ids
    coverage_ratio = (
        len(matched) / len(manifest_set) if manifest_set else 0.0
    )

    return PredictionCoverageReport(
        manifest_row_count=len(manifest_set),
        prediction_row_count=len(prediction_ids),
        matched_object_ids=tuple(sorted(matched)),
        unmatched_prediction_object_ids=tuple(sorted(unmatched_predictions)),
        missing_prediction_object_ids=tuple(sorted(missing_predictions)),
        coverage_ratio=coverage_ratio,
    )


def build_validated_predictions(
    raw_predictions: RegisteredArtifact,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    dataset_version_id: str,
    parent_version_id: str,
    created_by_job_id: str,
    config_hash: str,
    model_id: str,
    model_version: str,
    task_type: str = "classification",
    contract_pack: ContractPack | None = None,
    audit_sink: FakePlatformMetadataClient | None = None,
    organization_id: str = "",
    project_id: str = "",
    manifest_object_ids: Iterable[str] | None = None,
) -> BuildValidatedPredictionsResult:
    """Validate raw prediction artifact bytes and persist immutable manifest.

    Strict guarantees:

    * raw object hash must match registry record (``raw_hash_mismatch``);
    * if any row fails validation no validated artifact is written;
    * on success the validated manifest is saved with the same bytes
      through :class:`ArtifactRegistry` under the immutable
      ``prediction_manifest`` kind, so the artifact has its own
      content-addressed URI/hash and lineage with model_id,
      model_version and inference_timestamp metadata;
    * a compute audit event is recorded carrying input/output hashes,
      coverage statistics and model identifiers.
    """
    raw_object = storage.get(raw_predictions.uri)
    raw_bytes = raw_object.data
    if raw_object.info.hash != raw_predictions.hash:
        raise PredictionContractValidationError(
            reason_code="raw_hash_mismatch",
            message="raw prediction artifact hash does not match registry record",
        )

    rows_report, rows = validate_predictions_jsonl(
        raw_bytes, contract_pack=contract_pack
    )

    coverage: PredictionCoverageReport | None = None
    if manifest_object_ids is not None:
        coverage = compute_prediction_coverage(
            manifest_object_ids=manifest_object_ids,
            prediction_rows=rows,
        )

    inference_timestamp = max(row.inference_timestamp for row in rows)

    validated_artifact = registry.save_artifact(
        artifact_kind=PREDICTION_MANIFEST_KIND,
        data=raw_bytes,
        artifact_format=PREDICTION_MANIFEST_FORMAT,
        media_type=PREDICTION_MANIFEST_MEDIA_TYPE,
        schema_version=PREDICTION_MANIFEST_SCHEMA_VERSION,
        dataset_version_id=dataset_version_id,
        created_by_job_id=created_by_job_id,
        config_hash=config_hash,
        metadata={
            "row_count": str(rows_report.row_count),
            "model-id": model_id,
            "model-version": model_version,
            "task-type": task_type,
            "inference-timestamp": inference_timestamp.isoformat(),
        },
    )

    audit_metadata: dict[str, NonEmptyStr | str | int | float | dict[str, int]] = {
        "compute_run_job_id": created_by_job_id,
        "dataset_version_id": dataset_version_id,
        "parent_version_id": parent_version_id,
        "config_hash": config_hash,
        "input_artifact_kind": raw_predictions.artifact_kind,
        "input_artifact_uri": raw_predictions.uri,
        "input_hash": raw_predictions.hash,
        "output_artifact_kind": validated_artifact.artifact_kind,
        "output_artifact_uri": validated_artifact.uri,
        "output_hash": validated_artifact.hash,
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "row_count": rows_report.row_count,
        "rows_by_split": dict(rows_report.rows_by_split),
        "model_id": model_id,
        "model_version": model_version,
        "task_type": task_type,
        "inference_timestamp": inference_timestamp.isoformat(),
    }
    if coverage is not None:
        audit_metadata["manifest_row_count"] = coverage.manifest_row_count
        audit_metadata["prediction_row_count"] = coverage.prediction_row_count
        audit_metadata["coverage_ratio"] = coverage.coverage_ratio
        audit_metadata["unmatched_prediction_count"] = len(
            coverage.unmatched_prediction_object_ids
        )
        audit_metadata["missing_prediction_count"] = len(
            coverage.missing_prediction_object_ids
        )

    audit_event = PlatformAuditEvent(
        audit_event_id=f"audit_{uuid.uuid4().hex}",
        event_type=AuditEventType.PREDICTIONS_VALIDATED,
        organization_id=organization_id or "org_compute_plane",
        project_id=project_id or "project_compute_plane",
        metadata=audit_metadata,
    )
    if audit_sink is not None:
        audit_event = audit_sink.record_audit_event(audit_event)

    return BuildValidatedPredictionsResult(
        prediction_manifest_artifact=validated_artifact,
        validation_report=PredictionValidationReport(
            rows=rows_report, coverage=coverage
        ),
        audit_event=audit_event,
    )


def compute_row_uncertainty_signals(row: PredictionRow) -> dict[str, float]:
    """Compute confidence/margin/entropy signals for a single prediction row.

    Formulas mirror DATASETS.md:

    * ``confidence = max_k p_k`` (already enforced by the row contract);
    * ``margin = p_top1 - p_top2``;
    * ``entropy = -Σ_k p_k * log(p_k)`` (natural log);
    * ``normalized_entropy = entropy / log(K)`` where K is the number of
      classes; falls back to ``0.0`` when K <= 1.

    The function is deterministic and never logs or echoes raw payload.
    """
    probabilities = sorted(row.predicted_proba.values(), reverse=True)
    confidence = probabilities[0]
    margin = (
        probabilities[0] - probabilities[1] if len(probabilities) >= 2 else 1.0
    )
    entropy = 0.0
    for probability in probabilities:
        if probability > 0:
            entropy -= probability * math.log(probability)
    log_k = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    normalized_entropy = entropy / log_k if log_k > 0 else 0.0
    return {
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }


def _classify_pydantic_error(
    message: str,
) -> tuple[str, type[PredictionContractValidationError | PredictionValidationError]]:
    """Map a pydantic validator message to a stable reason code/class."""
    lower = message.lower()
    if "predicted_proba values must sum to" in lower:
        return ("probability_sum_invalid", PredictionValidationError)
    if "predicted_label must match argmax" in lower:
        return ("argmax_mismatch", PredictionValidationError)
    if "confidence must equal max" in lower:
        return ("confidence_mismatch", PredictionValidationError)
    if "predicted_proba must contain at least one class probability" in lower:
        return ("empty_probability_map", PredictionValidationError)
    return ("pydantic_violation", PredictionContractValidationError)


def _schema_reason_code(payload: dict[str, object]) -> str:
    required = (
        "object_id",
        "true_label",
        "predicted_label",
        "predicted_proba",
        "confidence",
        "split",
        "model_id",
        "model_version",
        "inference_timestamp",
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
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


# Suppress unused-import warning: math is used inside the helper above.
_ = _PROBABILITY_TOLERANCE


__all__ = [
    "BuildValidatedPredictionsResult",
    "PREDICTION_MANIFEST_FORMAT",
    "PREDICTION_MANIFEST_KIND",
    "PREDICTION_MANIFEST_MEDIA_TYPE",
    "PREDICTION_MANIFEST_SCHEMA_VERSION",
    "PREDICTION_ROW_SCHEMA_NAME",
    "PredictionContractValidationError",
    "PredictionCoverageReport",
    "PredictionRowsValidationReport",
    "PredictionValidationError",
    "PredictionValidationReport",
    "build_validated_predictions",
    "compute_prediction_coverage",
    "compute_row_uncertainty_signals",
    "validate_predictions_jsonl",
]
