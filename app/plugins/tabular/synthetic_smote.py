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

Persistence (candidate dataset, augmented split manifest, report
artifact) is delegated to :mod:`app.plugins.tabular._synthetic_common`
so SMOTE and Gaussian Copula share an identical artifact contract.

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

import heapq
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
    SplitManifest,
    SyntheticAugmentationKind,
    SyntheticClassStats,
    SyntheticDatasetLineage,
    SyntheticDatasetReport,
    SyntheticGenerationMethod,
    SyntheticSampleLineage,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.plugins.tabular._synthetic_common import (
    IS_SYNTHETIC_COLUMN,
    SOURCE_SPLIT_COLUMN,
    SYNTHETIC_REPORT_SCHEMA_VERSION,
    evaluate_synthetic_validation,
    format_numeric,
    index_assignments,
    parse_numeric,
    persist_synthetic_artifacts,
    persist_synthetic_report,
    read_source_rows,
    verify_assignments_match_rows,
)

SMOTE_METHOD_VERSION = "smote_v0"
DEFAULT_SAMPLE_LINEAGE_LIMIT = 25

_SUPPORTED_STEP_TYPES = {"AUGMENT_RARE_CLASS"}
_SUPPORTED_METHOD_IDS = {"smote"}
# Columns that must never be treated as numeric SMOTE features even if
# they happen to parse as numbers. ``object_id`` and group keys are
# identifiers; the target column must not be perturbed; binary review
# flags carry no continuous semantics.
_NEVER_FEATURE_COLUMNS = frozenset(
    {
        "object_id",
        "customer_id_hash",
        "case_id",
        "transaction_id",
        "document_id",
        "support_ticket_id",
        "manual_review_flag",
        IS_SYNTHETIC_COLUMN,
        SOURCE_SPLIT_COLUMN,
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


def _smote_error_factory(
    reason_code: str,
    message: str,
    code: ErrorCode,
    details: dict[str, object],
) -> SmoteExecutionError:
    return SmoteExecutionError(
        reason_code=reason_code,
        message=message,
        code=code,
        details=details,
    )


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
    rows, columns = read_source_rows(
        storage=storage,
        source_artifact=request.source_artifact,
        target_column=request.target_column,
        error_factory=_smote_error_factory,
    )
    assignments = index_assignments(request.split_manifest)
    verify_assignments_match_rows(
        assignments=assignments,
        rows=rows,
        error_factory=_smote_error_factory,
    )

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
                method=SyntheticGenerationMethod.SMOTE,
                formula=SMOTE_FORMULA,
                seed_object_id=seed_row["object_id"],
                neighbor_object_id=neighbor_row["object_id"],
                lambda_value=round(lambda_value, 6),
                rare_class_label=request.rare_class_label,
            )
        )

    request_metadata = {
        "action-plan-id": request.action_plan_id,
        "step-id": request.step.step_id,
        "source-artifact-hash": request.source_artifact.hash,
        "split-manifest-hash": request.split_manifest_artifact.hash,
        "target-column": request.target_column,
        "rare-class-label": request.rare_class_label,
        "random-seed": str(request.random_seed),
        "k-neighbors": str(effective_k),
        "sampling-strategy": str(request.sampling_strategy),
        **(
            {"created-at": request.generated_at.isoformat()}
            if request.generated_at is not None
            else {}
        ),
    }
    persistence = persist_synthetic_artifacts(
        method=SyntheticGenerationMethod.SMOTE,
        request_metadata=request_metadata,
        real_rows=rows,
        synthetic_rows=synthetic_rows,
        columns=columns,
        split_manifest=request.split_manifest,
        target_column=request.target_column,
        candidate_dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        storage=storage,
        registry=registry,
    )

    truncated = len(sample_lineage) > request.sample_lineage_limit
    sample_lineage_for_report = (
        tuple(sample_lineage[: request.sample_lineage_limit])
        if request.sample_lineage_limit > 0
        else ()
    )

    validation_report = evaluate_synthetic_validation(
        real_rows=rows,
        synthetic_rows=synthetic_rows,
        feature_columns=feature_columns,
    )

    report = SyntheticDatasetReport(
        report_id=(
            request.report_id or f"synthetic_dataset_report_{uuid.uuid4().hex[:16]}"
        ),
        report_schema_version=SYNTHETIC_REPORT_SCHEMA_VERSION,
        method=SyntheticGenerationMethod.SMOTE,
        method_version=SMOTE_METHOD_VERSION,
        augmentation_kind=SyntheticAugmentationKind.TARGETED_RARE_CLASS,
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
        synthetic_validation=validation_report,
        gaussian_copula_artifacts=None,
        lineage=SyntheticDatasetLineage(
            action_plan_id=request.action_plan_id,
            step_id=request.step.step_id,
            source_dataset_version_id=request.source_dataset_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_artifact=request.source_artifact,
            split_manifest=request.split_manifest_artifact,
            candidate_artifact=persistence.candidate_artifact.artifact_ref,
            augmented_split_manifest=persistence.augmented_split_artifact.artifact_ref,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )

    report_artifact = persist_synthetic_report(
        report=report,
        candidate_dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        base_metadata=request_metadata,
        candidate_artifact_hash=persistence.candidate_artifact.hash,
        augmented_split_artifact_hash=persistence.augmented_split_artifact.hash,
        registry=registry,
    )

    return ExecuteSmoteAugmentationResult(
        report=report,
        candidate_artifact=persistence.candidate_artifact,
        augmented_split_artifact=persistence.augmented_split_artifact,
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
            message="SMOTE executor only handles AUGMENT_RARE_CLASS steps.",
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
            value = parse_numeric(raw)
            if value is None:
                value = 0.0
            features.append(value)
        vectors.append(tuple(features))
    return tuple(vectors)


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
        seed_value = parse_numeric(seed_row.get(column, ""))
        neighbor_value = parse_numeric(neighbor_row.get(column, ""))
        if seed_value is None or neighbor_value is None:
            continue
        new_value = seed_value + lambda_value * (neighbor_value - seed_value)
        row[column] = format_numeric(new_value)
    row[IS_SYNTHETIC_COLUMN] = "1"
    row[SOURCE_SPLIT_COLUMN] = DataSplit.TRAIN.value
    return row


def _achieved_ratio(*, rare_count_after: int, majority_count: int) -> float:
    if majority_count <= 0:
        return 1.0
    ratio = rare_count_after / majority_count
    return min(1.0, max(0.0, ratio))


__all__ = [
    "DEFAULT_SAMPLE_LINEAGE_LIMIT",
    "ExecuteSmoteAugmentationRequest",
    "ExecuteSmoteAugmentationResult",
    "SMOTE_METHOD_VERSION",
    "SmoteExecutionError",
    "execute_smote_augmentation_action",
]
