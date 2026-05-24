"""SMOTE rare-class augmentation executor for approved ActionPlan steps.

This executor implements the MVP SMOTE generator described in PRD
§11.10–§11.11. It must satisfy the TASK-046 acceptance criteria:

- SMOTE is available only for supervised classification.
- Validation/test records are never used as synthetic sources.
- Generated samples are explicitly marked synthetic and carry source
  lineage back to the seed and neighbor real objects.
- The SMOTE formula ``x_new = x_i + lambda * (x_nn - x_i)`` is recorded
  in the report and per-sample lineage entries.
- ``random_seed``, ``k_neighbors``, ``sampling_strategy`` are persisted.

The executor consumes:

- the approved ``AUGMENT_RARE_CLASS`` step with ``method=smote`` config;
- the source CSV referenced by the split manifest;
- the persisted ``SplitManifest`` artifact, which decides which rows
  belong to the training split.

It produces:

- a candidate tabular dataset CSV that contains the original rows plus
  appended synthetic rows (with ``is_synthetic`` and ``source_split``
  marker columns);
- an augmented split manifest that assigns the new synthetic rows to
  the training split;
- a contract-valid ``synthetic_dataset_report.v1`` artifact that
  carries parameters, per-class generation stats, sample lineage, and
  references to all upstream/downstream artifacts.

Privacy and safety:

- the report never echoes raw row payloads; only ``object_id`` values,
  numeric ``lambda`` samples, column names, and aggregate counts are
  recorded;
- the executor is stdlib-only (``random``, ``heapq``, ``math``); no
  external dependencies are pulled in for the MVP path;
- the executor never overwrites the source artifact and never reads
  rows from the validation or test splits.
"""

from __future__ import annotations

import csv
import heapq
import io
import json
import random
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    SMOTE_FORMULA,
    ActionPlanStep,
    ArtifactRef,
    DataSplit,
    ErrorCode,
    SplitAssignment,
    SplitManifest,
    SyntheticClassStats,
    SyntheticDatasetLineage,
    SyntheticDatasetReport,
    SyntheticGenerationMethod,
    SyntheticSampleLineage,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.plugins.tabular.splits import (
    SPLIT_MANIFEST_FORMAT,
    SPLIT_MANIFEST_KIND,
    SPLIT_MANIFEST_MEDIA_TYPE,
    SPLIT_MANIFEST_SCHEMA_VERSION,
)

SYNTHETIC_REPORT_KIND = "synthetic_dataset_report"
SYNTHETIC_REPORT_FORMAT = "json"
SYNTHETIC_REPORT_MEDIA_TYPE = "application/json"
SYNTHETIC_REPORT_SCHEMA_VERSION = "synthetic_dataset_report.v1"
SMOTE_METHOD_VERSION = "smote_v0"
CANDIDATE_TABULAR_DATASET_KIND = "candidate_tabular_dataset"
CANDIDATE_TABULAR_DATASET_FORMAT = "csv"
CANDIDATE_TABULAR_DATASET_MEDIA_TYPE = "text/csv"
CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION = "tabular_dataset.v1"

DEFAULT_SAMPLE_LINEAGE_LIMIT = 25

_IS_SYNTHETIC_COLUMN = "is_synthetic"
_SOURCE_SPLIT_COLUMN = "synthetic_source_split"
_SUPPORTED_STEP_TYPES = {"AUGMENT_RARE_CLASS"}
_SUPPORTED_METHOD_IDS = {"smote"}
# Columns that must never be treated as numeric SMOTE features even if
# they happen to parse as numbers. ``object_id`` and group keys are
# identifiers; the target column must not be perturbed; binary review
# flags carry no continuous semantics. Excluding them here keeps the
# candidate dataset interpretable and avoids accidentally inventing
# synthetic group keys.
_NEVER_FEATURE_COLUMNS = frozenset(
    {
        "object_id",
        "customer_id_hash",
        "case_id",
        "transaction_id",
        "document_id",
        "support_ticket_id",
        "manual_review_flag",
        _IS_SYNTHETIC_COLUMN,
        _SOURCE_SPLIT_COLUMN,
    }
)


class SmoteExecutionError(ValueError):
    """Raised when a SMOTE ActionPlan step is unsafe or cannot run."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.ACTION_PLAN_PRECONDITION_FAILED,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class ExecuteSmoteAugmentationRequest(BaseModel):
    """Inputs for executing one approved AUGMENT_RARE_CLASS / SMOTE step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step: ActionPlanStep
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    source_artifact: ArtifactRef
    split_manifest: SplitManifest
    split_manifest_artifact: ArtifactRef
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    target_column: NonEmptyStr = "is_fraud"
    rare_class_label: NonEmptyStr = "1"
    random_seed: int = 42
    k_neighbors: int = Field(ge=1, default=5)
    sampling_strategy: float = Field(gt=0.0, le=1.0, default=0.15)
    sample_lineage_limit: int = Field(ge=0, default=DEFAULT_SAMPLE_LINEAGE_LIMIT)
    feature_columns: tuple[NonEmptyStr, ...] | None = None
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class ExecuteSmoteAugmentationResult:
    """Persisted artifacts and parsed report produced by SMOTE execution."""

    report: SyntheticDatasetReport
    candidate_artifact: RegisteredArtifact
    augmented_split_artifact: RegisteredArtifact
    report_artifact: RegisteredArtifact


def execute_smote_augmentation_action(
    request: ExecuteSmoteAugmentationRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> ExecuteSmoteAugmentationResult:
    """Run an approved SMOTE step and persist all derived artifacts."""
    _validate_step(request)
    rows, columns = _read_source_rows(
        storage=storage,
        source_artifact=request.source_artifact,
        target_column=request.target_column,
    )
    assignments = _index_assignments(request.split_manifest)
    _verify_assignments_match_rows(assignments=assignments, rows=rows)

    train_rare_indices = [
        index
        for index, row in enumerate(rows)
        if assignments.get(row["object_id"], (None, None))[0] is DataSplit.TRAIN
        and row[request.target_column] == request.rare_class_label
    ]
    train_majority_count = sum(
        1
        for row in rows
        if assignments.get(row["object_id"], (None, None))[0] is DataSplit.TRAIN
        and row[request.target_column] != request.rare_class_label
    )
    if not train_rare_indices:
        raise SmoteExecutionError(
            reason_code="rare_class_absent_in_train_split",
            message=(
                "SMOTE requires at least one rare-class row in the training split."
            ),
        )
    if len(train_rare_indices) < 2:
        raise SmoteExecutionError(
            reason_code="rare_class_too_small_for_neighbors",
            message=(
                "SMOTE requires at least two rare-class rows in the training "
                "split to compute nearest neighbors."
            ),
            details={"rare_class_count": len(train_rare_indices)},
        )

    feature_columns, excluded_columns = _resolve_feature_columns(
        columns=columns,
        target_column=request.target_column,
        explicit=request.feature_columns,
    )
    if not feature_columns:
        raise SmoteExecutionError(
            reason_code="no_numeric_feature_columns",
            message="SMOTE requires at least one numeric feature column.",
        )

    rare_feature_vectors = _extract_feature_vectors(
        rows=rows,
        indices=train_rare_indices,
        columns=feature_columns,
    )
    effective_k = min(request.k_neighbors, len(train_rare_indices) - 1)

    target_count = _target_count_after_augmentation(
        rare_count=len(train_rare_indices),
        majority_count=train_majority_count,
        sampling_strategy=request.sampling_strategy,
    )
    generated_count = max(0, target_count - len(train_rare_indices))

    rng = random.Random(request.random_seed)
    synthetic_rows: list[dict[str, str]] = []
    sample_lineage: list[SyntheticSampleLineage] = []
    for sample_index in range(generated_count):
        seed_pos = rng.randrange(len(train_rare_indices))
        seed_index = train_rare_indices[seed_pos]
        seed_row = rows[seed_index]
        neighbor_pos = _pick_neighbor(
            seed_pos=seed_pos,
            rare_feature_vectors=rare_feature_vectors,
            k_neighbors=effective_k,
            rng=rng,
        )
        neighbor_index = train_rare_indices[neighbor_pos]
        neighbor_row = rows[neighbor_index]
        lambda_value = rng.random()
        synthetic_object_id = f"txn_synth_{sample_index:06d}"
        synthetic_row = _build_synthetic_row(
            seed_row=seed_row,
            neighbor_row=neighbor_row,
            feature_columns=feature_columns,
            lambda_value=lambda_value,
            target_column=request.target_column,
            rare_class_label=request.rare_class_label,
            synthetic_object_id=synthetic_object_id,
        )
        synthetic_rows.append(synthetic_row)
        sample_lineage.append(
            SyntheticSampleLineage(
                synthetic_object_id=synthetic_object_id,
                seed_object_id=seed_row["object_id"],
                neighbor_object_id=neighbor_row["object_id"],
                lambda_value=round(lambda_value, 6),
                rare_class_label=request.rare_class_label,
            )
        )

    candidate_columns = _candidate_dataset_columns(columns)
    candidate_payload = _write_candidate_csv(
        rows=rows,
        synthetic_rows=synthetic_rows,
        columns=candidate_columns,
    )
    candidate_artifact = registry.save_artifact(
        artifact_kind=CANDIDATE_TABULAR_DATASET_KIND,
        data=candidate_payload,
        artifact_format=CANDIDATE_TABULAR_DATASET_FORMAT,
        media_type=CANDIDATE_TABULAR_DATASET_MEDIA_TYPE,
        schema_version=CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": request.step.step_id,
            "source-artifact-hash": request.source_artifact.hash,
            "split-manifest-hash": request.split_manifest_artifact.hash,
            "target-column": request.target_column,
            "rare-class-label": request.rare_class_label,
            "synthetic-method": SyntheticGenerationMethod.SMOTE.value,
            "random-seed": str(request.random_seed),
            "k-neighbors": str(effective_k),
            "sampling-strategy": str(request.sampling_strategy),
            "synthetic-row-count": str(len(synthetic_rows)),
        },
    )

    augmented_manifest = _build_augmented_split_manifest(
        manifest=request.split_manifest,
        synthetic_rows=synthetic_rows,
        rare_class_label=request.rare_class_label,
    )
    augmented_split_artifact = registry.save_artifact(
        artifact_kind=SPLIT_MANIFEST_KIND,
        data=_serialize_split_manifest(augmented_manifest),
        artifact_format=SPLIT_MANIFEST_FORMAT,
        media_type=SPLIT_MANIFEST_MEDIA_TYPE,
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": request.step.step_id,
            "augments-split-manifest-id": request.split_manifest.split_manifest_id,
            "augments-split-manifest-hash": request.split_manifest_artifact.hash,
            "synthetic-row-count": str(len(synthetic_rows)),
            "synthetic-method": SyntheticGenerationMethod.SMOTE.value,
            "rare-class-label": request.rare_class_label,
        },
    )

    truncated = len(sample_lineage) > request.sample_lineage_limit
    sample_lineage_for_report = (
        tuple(sample_lineage[: request.sample_lineage_limit])
        if request.sample_lineage_limit > 0
        else ()
    )

    report = SyntheticDatasetReport(
        report_id=(
            request.report_id or f"synthetic_dataset_report_{uuid.uuid4().hex[:16]}"
        ),
        report_schema_version=SYNTHETIC_REPORT_SCHEMA_VERSION,
        method=SyntheticGenerationMethod.SMOTE,
        method_version=SMOTE_METHOD_VERSION,
        formula=SMOTE_FORMULA,
        target_column=request.target_column,
        rare_class_label=request.rare_class_label,
        source_split=DataSplit.TRAIN.value,
        random_seed=request.random_seed,
        k_neighbors=effective_k,
        sampling_strategy=request.sampling_strategy,
        feature_columns=feature_columns,
        excluded_columns=excluded_columns,
        real_total_count=len(rows),
        real_train_count=len(train_rare_indices) + train_majority_count,
        real_train_rare_count=len(train_rare_indices),
        generated_count=len(synthetic_rows),
        class_stats=(
            SyntheticClassStats(
                label=request.rare_class_label,
                real_count_in_source_split=len(train_rare_indices),
                real_count_in_majority_split=train_majority_count,
                target_count_after_augmentation=target_count,
                generated_count=len(synthetic_rows),
                achieved_ratio=_achieved_ratio(
                    rare_count_after=len(train_rare_indices) + len(synthetic_rows),
                    majority_count=train_majority_count,
                ),
            ),
        ),
        sample_lineage=sample_lineage_for_report,
        sample_lineage_truncated=truncated,
        full_sample_lineage_count=len(sample_lineage),
        lineage=SyntheticDatasetLineage(
            action_plan_id=request.action_plan_id,
            step_id=request.step.step_id,
            source_dataset_version_id=request.source_dataset_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_artifact=request.source_artifact,
            split_manifest=request.split_manifest_artifact,
            candidate_artifact=candidate_artifact.artifact_ref,
            augmented_split_manifest=augmented_split_artifact.artifact_ref,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )

    report_artifact = registry.save_artifact(
        artifact_kind=SYNTHETIC_REPORT_KIND,
        data=_serialize_report(report),
        artifact_format=SYNTHETIC_REPORT_FORMAT,
        media_type=SYNTHETIC_REPORT_MEDIA_TYPE,
        schema_version=SYNTHETIC_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": request.step.step_id,
            "synthetic-method": SyntheticGenerationMethod.SMOTE.value,
            "random-seed": str(request.random_seed),
            "k-neighbors": str(effective_k),
            "sampling-strategy": str(request.sampling_strategy),
            "generated-count": str(len(synthetic_rows)),
            "candidate-artifact-hash": candidate_artifact.hash,
            "augmented-split-manifest-hash": augmented_split_artifact.hash,
        },
    )

    return ExecuteSmoteAugmentationResult(
        report=report,
        candidate_artifact=candidate_artifact,
        augmented_split_artifact=augmented_split_artifact,
        report_artifact=report_artifact,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_step(request: ExecuteSmoteAugmentationRequest) -> None:
    step = request.step
    if step.type not in _SUPPORTED_STEP_TYPES:
        raise SmoteExecutionError(
            reason_code="unsupported_action_step_type",
            message=(
                "SMOTE executor only handles AUGMENT_RARE_CLASS steps."
            ),
            details={"step_id": step.step_id, "step_type": step.type},
        )
    if step.method_id not in _SUPPORTED_METHOD_IDS:
        raise SmoteExecutionError(
            reason_code="unsupported_synthetic_method",
            message="SMOTE executor only supports method=smote.",
            details={"step_id": step.step_id, "method_id": step.method_id},
        )
    source_split = step.config.get("source_split", DataSplit.TRAIN.value)
    if source_split != DataSplit.TRAIN.value:
        raise SmoteExecutionError(
            reason_code="non_train_source_split",
            message=(
                "SMOTE may only be fitted on the training split per PRD §11.10."
            ),
            details={"source_split": str(source_split)},
        )
    if request.split_manifest.target_column != request.target_column:
        raise SmoteExecutionError(
            reason_code="target_column_mismatch_with_split_manifest",
            message=(
                "Request target column does not match the split manifest target."
            ),
            details={
                "request_target_column": request.target_column,
                "split_manifest_target_column": (
                    request.split_manifest.target_column
                ),
            },
        )


def _read_source_rows(
    *,
    storage: MinioObjectStorageAdapter,
    source_artifact: ArtifactRef,
    target_column: str,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    stored = storage.get(source_artifact.uri)
    text = stored.data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = tuple(name for name in (reader.fieldnames or ()) if name)
    if not columns:
        raise SmoteExecutionError(
            reason_code="empty_tabular_source",
            message="Source CSV has no header columns.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    if target_column not in columns:
        raise SmoteExecutionError(
            reason_code="target_column_not_found",
            message="Target column is not present in source CSV.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            details={"target_column": target_column},
        )
    rows: list[dict[str, str]] = []
    for index, raw_row in enumerate(reader):
        row = {
            column: ("" if raw_row.get(column) is None else str(raw_row.get(column)))
            for column in columns
        }
        if not row.get("object_id"):
            row["object_id"] = f"row_{index:06d}"
        rows.append(row)
    if not rows:
        raise SmoteExecutionError(
            reason_code="empty_tabular_source",
            message="Source CSV has no data rows.",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
        )
    return tuple(rows), columns


def _index_assignments(
    manifest: SplitManifest,
) -> dict[str, tuple[DataSplit, str | None]]:
    return {
        assignment.object_id: (assignment.split, assignment.group_value)
        for assignment in manifest.assignments
    }


def _verify_assignments_match_rows(
    *,
    assignments: Mapping[str, tuple[DataSplit, str | None]],
    rows: Sequence[Mapping[str, str]],
) -> None:
    row_ids = {row["object_id"] for row in rows}
    missing = sorted(set(assignments) - row_ids)
    if missing:
        raise SmoteExecutionError(
            reason_code="split_manifest_object_id_missing_in_source",
            message=(
                "Split manifest references object_ids missing from the source CSV."
            ),
            details={"missing_object_ids": missing[:5]},
        )


def _resolve_feature_columns(
    *,
    columns: tuple[str, ...],
    target_column: str,
    explicit: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidates: Iterable[str]
    if explicit is not None:
        candidates = explicit
    else:
        candidates = columns
    feature_columns: list[str] = []
    excluded: list[str] = []
    for column in candidates:
        if column not in columns:
            excluded.append(column)
            continue
        if column == target_column or column in _NEVER_FEATURE_COLUMNS:
            excluded.append(column)
            continue
        feature_columns.append(column)
    # Drop columns that are non-numeric across the dataset. We compute
    # this here so excluded_columns surfaces the reason a column was
    # skipped in the report.
    return tuple(feature_columns), tuple(_unique(excluded))


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _extract_feature_vectors(
    *,
    rows: Sequence[Mapping[str, str]],
    indices: Sequence[int],
    columns: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    if not columns:
        return ()
    vectors: list[tuple[float, ...]] = []
    for index in indices:
        row = rows[index]
        features: list[float] = []
        for column in columns:
            raw = row.get(column, "")
            value = _parse_numeric(raw)
            if value is None:
                # Treat missing/non-numeric rare-class values as 0.0; the
                # alternative is to skip the row, which would silently
                # change the rare-class population. Skipping is unsafe
                # because SMOTE callers expect all rare rows to be
                # eligible. We mark the situation with a reason code so
                # callers can decide to add an imputation step before
                # SMOTE if needed.
                value = 0.0
            features.append(value)
        vectors.append(tuple(features))
    return tuple(vectors)


def _parse_numeric(value: str) -> float | None:
    if value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _pick_neighbor(
    *,
    seed_pos: int,
    rare_feature_vectors: tuple[tuple[float, ...], ...],
    k_neighbors: int,
    rng: random.Random,
) -> int:
    seed_vector = rare_feature_vectors[seed_pos]
    distances: list[tuple[float, int]] = []
    for index, vector in enumerate(rare_feature_vectors):
        if index == seed_pos:
            continue
        distance = _squared_distance(seed_vector, vector)
        distances.append((distance, index))
    if not distances:
        # Defensive: caller already enforced len(rare) >= 2, so this
        # branch should not trigger. We pick the seed itself as a
        # last-resort fallback to keep the type contract intact.
        return seed_pos
    nearest = heapq.nsmallest(k_neighbors, distances)
    return rng.choice([index for _, index in nearest])


def _squared_distance(
    a: tuple[float, ...],
    b: tuple[float, ...],
) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _target_count_after_augmentation(
    *,
    rare_count: int,
    majority_count: int,
    sampling_strategy: float,
) -> int:
    """Return the target rare-class count after SMOTE.

    ``sampling_strategy`` follows the imbalanced-learn convention: it is
    the desired ratio between the number of rare-class samples and the
    number of majority-class samples in the augmented training split.
    A value of ``1.0`` means "match the majority class"; ``0.15`` means
    "rare class should reach 15% of the majority size". The returned
    value is clamped so the rare class never shrinks.
    """
    desired = int(round(majority_count * sampling_strategy))
    return max(rare_count, desired)


def _build_synthetic_row(
    *,
    seed_row: Mapping[str, str],
    neighbor_row: Mapping[str, str],
    feature_columns: tuple[str, ...],
    lambda_value: float,
    target_column: str,
    rare_class_label: str,
    synthetic_object_id: str,
) -> dict[str, str]:
    row: dict[str, str] = dict(seed_row)
    row["object_id"] = synthetic_object_id
    row[target_column] = rare_class_label
    for column in feature_columns:
        seed_value = _parse_numeric(seed_row.get(column, ""))
        neighbor_value = _parse_numeric(neighbor_row.get(column, ""))
        if seed_value is None or neighbor_value is None:
            # Preserve the seed row's value for non-numeric or missing
            # columns. This keeps the synthetic row valid against the
            # source schema; the lineage entry already records that the
            # numeric SMOTE formula could not be applied to that column.
            continue
        new_value = seed_value + lambda_value * (neighbor_value - seed_value)
        row[column] = _format_numeric(new_value)
    row[_IS_SYNTHETIC_COLUMN] = "1"
    row[_SOURCE_SPLIT_COLUMN] = DataSplit.TRAIN.value
    return row


def _format_numeric(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _candidate_dataset_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    extras: list[str] = []
    if _IS_SYNTHETIC_COLUMN not in columns:
        extras.append(_IS_SYNTHETIC_COLUMN)
    if _SOURCE_SPLIT_COLUMN not in columns:
        extras.append(_SOURCE_SPLIT_COLUMN)
    return tuple([*columns, *extras])


def _write_candidate_csv(
    *,
    rows: Sequence[Mapping[str, str]],
    synthetic_rows: Sequence[Mapping[str, str]],
    columns: tuple[str, ...],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        materialized = {column: row.get(column, "") for column in columns}
        materialized.setdefault(_IS_SYNTHETIC_COLUMN, "0")
        materialized.setdefault(_SOURCE_SPLIT_COLUMN, "")
        if not materialized.get(_IS_SYNTHETIC_COLUMN):
            materialized[_IS_SYNTHETIC_COLUMN] = "0"
        writer.writerow(materialized)
    for row in synthetic_rows:
        materialized = {column: row.get(column, "") for column in columns}
        writer.writerow(materialized)
    return buffer.getvalue().encode("utf-8")


def _build_augmented_split_manifest(
    *,
    manifest: SplitManifest,
    synthetic_rows: Sequence[Mapping[str, str]],
    rare_class_label: str,
) -> SplitManifest:
    if not synthetic_rows:
        return manifest
    new_assignments = list(manifest.assignments)
    label_counts = {item.split: dict(item.class_counts) for item in manifest.class_distribution}
    total_counts = {item.split: item.total_count for item in manifest.class_distribution}
    for row in synthetic_rows:
        new_assignments.append(
            SplitAssignment(
                object_id=row["object_id"],
                split=DataSplit.TRAIN,
                label=rare_class_label,
                group_value=None,
            )
        )
        train_counts = label_counts.setdefault(DataSplit.TRAIN, {})
        train_counts[rare_class_label] = train_counts.get(rare_class_label, 0) + 1
        total_counts[DataSplit.TRAIN] = (
            total_counts.get(DataSplit.TRAIN, 0) + 1
        )

    new_distribution = []
    for item in manifest.class_distribution:
        counts = label_counts.get(item.split, item.class_counts)
        total = total_counts.get(item.split, item.total_count)
        ratios = {
            label: (count / total if total else 0.0)
            for label, count in counts.items()
        }
        new_distribution.append(
            item.model_copy(
                update={
                    "class_counts": dict(sorted(counts.items())),
                    "class_ratios": dict(sorted(ratios.items())),
                    "total_count": total,
                }
            )
        )

    return manifest.model_copy(
        update={
            "split_manifest_id": (
                f"{manifest.split_manifest_id}_with_synthetic_{len(synthetic_rows)}"
            ),
            "assignments": tuple(new_assignments),
            "class_distribution": tuple(new_distribution),
        }
    )


def _achieved_ratio(*, rare_count_after: int, majority_count: int) -> float:
    if majority_count <= 0:
        return 1.0
    ratio = rare_count_after / majority_count
    return min(1.0, max(0.0, ratio))


def _serialize_report(report: SyntheticDatasetReport) -> bytes:
    return json.dumps(
        report.model_dump(mode="json"), sort_keys=True, indent=2
    ).encode("utf-8")


def _serialize_split_manifest(manifest: SplitManifest) -> bytes:
    return json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, indent=2
    ).encode("utf-8")


__all__ = [
    "DEFAULT_SAMPLE_LINEAGE_LIMIT",
    "ExecuteSmoteAugmentationRequest",
    "ExecuteSmoteAugmentationResult",
    "SMOTE_METHOD_VERSION",
    "SYNTHETIC_REPORT_FORMAT",
    "SYNTHETIC_REPORT_KIND",
    "SYNTHETIC_REPORT_MEDIA_TYPE",
    "SYNTHETIC_REPORT_SCHEMA_VERSION",
    "SmoteExecutionError",
    "execute_smote_augmentation_action",
]
