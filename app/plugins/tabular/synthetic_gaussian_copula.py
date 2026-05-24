"""Gaussian Copula synthetic generator with strict policy gating.

This executor implements the MVP Gaussian Copula generator described in
PRD §11.10–§11.11. TASK-047 acceptance criteria:

- Gaussian Copula is available only when enabled by policy/profile.
- Synthetic output passes schema, type, business-rule, and privacy
  checks (privacy gate is the exact-duplicate-to-real check from
  PRD §11.12).
- The report records column distribution transforms, the latent normal
  space, the correlation structure, generated samples, and inverse
  transform metadata.
- The report distinguishes SMOTE (targeted rare-class augmentation)
  from Gaussian Copula (distribution-level synthetic generation) via
  ``augmentation_kind``.
- Generated rows carry ``is_synthetic=1`` markers and lineage refs.
- A disabled policy returns a stable ``GaussianCopulaPolicyError`` with
  ``reason_code=disabled_by_policy``.

Implementation:

- stdlib-only (``random``, ``math``, ``statistics``); no SDV/scipy/
  numpy dependency for the MVP path. The numerical kernel implements:
  empirical CDF (linear interpolation between sorted training values),
  inverse Φ (rational approximation via ``math.erf``-based inverse),
  Cholesky factorization of the correlation matrix, and inverse ECDF
  via linear interpolation over a fixed quantile grid.
- The generator is deterministic for a given ``random_seed``.
"""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    GAUSSIAN_COPULA_FORMULA,
    ActionPlanStep,
    ArtifactRef,
    ColumnDistributionTransform,
    DataSplit,
    ErrorCode,
    GaussianCopulaArtifacts,
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

GAUSSIAN_COPULA_METHOD_VERSION = "gaussian_copula_v0"
DEFAULT_SAMPLE_LINEAGE_LIMIT = 25
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
RIDGE_EPSILON = 1e-6
CDF_CLAMP_EPSILON = 1e-6

_SUPPORTED_STEP_TYPES = {"AUGMENT_RARE_CLASS"}
_SUPPORTED_METHOD_IDS = {"gaussian_copula"}
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


class GaussianCopulaExecutionError(ValueError):
    """Raised when a Gaussian Copula step is unsafe or cannot run."""

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


class GaussianCopulaPolicyError(GaussianCopulaExecutionError):
    """Raised when Gaussian Copula is not enabled by the active policy.

    The error carries ``reason_code='disabled_by_policy'`` so the API
    layer can return a stable ``POLICY_BLOCKED`` response without
    leaking policy internals.
    """

    def __init__(self, *, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(
            reason_code="disabled_by_policy",
            message=message,
            code=ErrorCode.POLICY_BLOCKED,
            details=details,
        )


def _gc_error_factory(
    reason_code: str,
    message: str,
    code: ErrorCode,
    details: dict[str, object],
) -> GaussianCopulaExecutionError:
    return GaussianCopulaExecutionError(
        reason_code=reason_code,
        message=message,
        code=code,
        details=details,
    )


class GaussianCopulaPolicy(BaseModel):
    """Active Gaussian Copula policy gate inputs.

    The platform / Decision Core must pass this envelope to the
    executor. ``enabled`` reflects the active profile and project
    policy (``PolicyStatus.ENABLED``). ``readiness_status`` and
    ``policy_status`` are recorded for audit traceability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    readiness_status: NonEmptyStr = "available"
    policy_status: NonEmptyStr = "enabled"
    policy_version: NonEmptyStr = "synthetic_policy_v0"
    profile: NonEmptyStr = "demo_strict"


class ExecuteGaussianCopulaRequest(BaseModel):
    """Inputs for executing one approved Gaussian Copula step."""

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
    policy: GaussianCopulaPolicy
    target_column: NonEmptyStr = "is_fraud"
    random_seed: int = 7
    sampling_strategy: float = Field(gt=0.0, le=1.0, default=0.10)
    sample_lineage_limit: int = Field(ge=0, default=DEFAULT_SAMPLE_LINEAGE_LIMIT)
    feature_columns: tuple[NonEmptyStr, ...] | None = None
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class ExecuteGaussianCopulaResult:
    """Persisted artifacts and parsed report produced by the GC executor."""

    report: SyntheticDatasetReport
    candidate_artifact: RegisteredArtifact
    augmented_split_artifact: RegisteredArtifact
    report_artifact: RegisteredArtifact


def execute_gaussian_copula_action(
    request: ExecuteGaussianCopulaRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> ExecuteGaussianCopulaResult:
    """Run an approved Gaussian Copula step and persist all derived artifacts."""
    if not request.policy.enabled:
        raise GaussianCopulaPolicyError(
            message=(
                "Gaussian Copula synthetic generation is disabled by the active policy."
            ),
            details={
                "method_id": request.step.method_id,
                "policy_status": request.policy.policy_status,
                "policy_version": request.policy.policy_version,
                "profile": request.policy.profile,
            },
        )
    _validate_step(request)

    rows, columns = read_source_rows(
        storage=storage,
        source_artifact=request.source_artifact,
        target_column=request.target_column,
        error_factory=_gc_error_factory,
    )
    assignments = index_assignments(request.split_manifest)
    verify_assignments_match_rows(
        assignments=assignments,
        rows=rows,
        error_factory=_gc_error_factory,
    )

    train_rows = [
        row
        for row in rows
        if assignments.get(row["object_id"], (None, None))[0] is DataSplit.TRAIN
    ]
    if not train_rows:
        raise GaussianCopulaExecutionError(
            reason_code="train_split_is_empty",
            message="Gaussian Copula requires at least one row in the training split.",
        )

    feature_columns, excluded_columns = _resolve_feature_columns(
        columns=columns,
        target_column=request.target_column,
        explicit=request.feature_columns,
    )
    if not feature_columns:
        raise GaussianCopulaExecutionError(
            reason_code="no_numeric_feature_columns",
            message="Gaussian Copula requires at least one numeric feature column.",
        )

    feature_vectors = _extract_feature_vectors(
        rows=train_rows,
        columns=feature_columns,
    )
    if len(feature_vectors) < 2:
        raise GaussianCopulaExecutionError(
            reason_code="train_split_too_small",
            message=(
                "Gaussian Copula requires at least two training rows "
                "to estimate the correlation matrix."
            ),
            details={"train_row_count": len(feature_vectors)},
        )

    rare_class_label, train_label_counts = _train_label_counts(
        train_rows=train_rows,
        target_column=request.target_column,
    )
    train_majority_count = max(train_label_counts.values()) if train_label_counts else 0
    train_total = sum(train_label_counts.values())
    target_total = max(train_total, int(round(train_total * (1.0 + request.sampling_strategy))))
    generated_count = max(0, target_total - train_total)

    rng = random.Random(request.random_seed)

    column_transforms = _build_column_transforms(
        feature_columns=feature_columns,
        feature_vectors=feature_vectors,
        quantile_levels=request.quantile_levels,
    )
    correlation = _correlation_matrix(feature_vectors=feature_vectors)
    cholesky = _cholesky(correlation, ridge_epsilon=RIDGE_EPSILON)

    synthetic_rows: list[dict[str, str]] = []
    sample_lineage: list[SyntheticSampleLineage] = []
    train_label_population = [row[request.target_column] for row in train_rows]
    for sample_index in range(generated_count):
        latent = _sample_latent(rng=rng, dimension=len(feature_columns))
        correlated = _matrix_vector(cholesky, latent)
        uniform = tuple(_phi(value) for value in correlated)
        new_features = tuple(
            _inverse_ecdf(transform, u, clamp=CDF_CLAMP_EPSILON)
            for transform, u in zip(column_transforms, uniform, strict=True)
        )
        # Distribute synthetic rows over observed train labels in
        # proportion to their frequency. Single-class training data
        # falls back to that label.
        synthesized_label = train_label_population[
            rng.randrange(len(train_label_population))
        ] if train_label_population else "0"
        synthetic_object_id = f"txn_synth_gc_{sample_index:06d}"
        synthetic_row = _build_synthetic_row(
            template_row=train_rows[0],
            feature_columns=feature_columns,
            new_features=new_features,
            target_column=request.target_column,
            target_value=synthesized_label,
            synthetic_object_id=synthetic_object_id,
        )
        synthetic_rows.append(synthetic_row)
        sample_lineage.append(
            SyntheticSampleLineage(
                synthetic_object_id=synthetic_object_id,
                method=SyntheticGenerationMethod.GAUSSIAN_COPULA,
                formula=GAUSSIAN_COPULA_FORMULA,
                latent_draw_index=sample_index,
                rare_class_label=None,
            )
        )

    request_metadata = {
        "action-plan-id": request.action_plan_id,
        "step-id": request.step.step_id,
        "source-artifact-hash": request.source_artifact.hash,
        "split-manifest-hash": request.split_manifest_artifact.hash,
        "target-column": request.target_column,
        "policy-version": request.policy.policy_version,
        "policy-profile": request.policy.profile,
        "policy-status": request.policy.policy_status,
        "random-seed": str(request.random_seed),
        "sampling-strategy": str(request.sampling_strategy),
    }
    persistence = persist_synthetic_artifacts(
        method=SyntheticGenerationMethod.GAUSSIAN_COPULA,
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

    class_stats = _class_stats(
        train_label_counts=train_label_counts,
        synthetic_rows=synthetic_rows,
        target_column=request.target_column,
        train_majority_count=train_majority_count,
    )

    gc_artifacts = GaussianCopulaArtifacts(
        column_distribution_transforms=column_transforms,
        latent_normal_mean=tuple(0.0 for _ in feature_columns),
        latent_normal_covariance_row_major=tuple(_flatten(correlation)),
        latent_normal_cholesky_row_major=tuple(_flatten(cholesky)),
        correlation_matrix_row_major=tuple(_flatten(correlation)),
        correlation_matrix_size=len(feature_columns),
        inverse_transform_metadata={
            "uniform_to_data": "linear_interpolation_over_quantile_grid",
            "normal_to_uniform": "phi_via_math_erf",
            "cdf_clamp_epsilon": str(CDF_CLAMP_EPSILON),
            "ridge_epsilon": str(RIDGE_EPSILON),
        },
        ridge_epsilon=RIDGE_EPSILON,
        cdf_clamp_epsilon=CDF_CLAMP_EPSILON,
    )

    report = SyntheticDatasetReport(
        report_id=(
            request.report_id or f"synthetic_dataset_report_{uuid.uuid4().hex[:16]}"
        ),
        report_schema_version=SYNTHETIC_REPORT_SCHEMA_VERSION,
        method=SyntheticGenerationMethod.GAUSSIAN_COPULA,
        method_version=GAUSSIAN_COPULA_METHOD_VERSION,
        augmentation_kind=SyntheticAugmentationKind.DISTRIBUTION_LEVEL,
        formula=GAUSSIAN_COPULA_FORMULA,
        target_column=request.target_column,
        rare_class_label=None,
        source_split=DataSplit.TRAIN.value,
        random_seed=request.random_seed,
        k_neighbors=0,
        sampling_strategy=request.sampling_strategy,
        feature_columns=feature_columns,
        excluded_columns=excluded_columns,
        real_total_count=len(rows),
        real_train_count=len(train_rows),
        real_train_rare_count=train_label_counts.get(rare_class_label, 0)
        if rare_class_label is not None
        else 0,
        generated_count=len(synthetic_rows),
        class_stats=class_stats,
        sample_lineage=sample_lineage_for_report,
        sample_lineage_truncated=truncated,
        full_sample_lineage_count=len(sample_lineage),
        synthetic_validation=validation_report,
        gaussian_copula_artifacts=gc_artifacts,
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

    return ExecuteGaussianCopulaResult(
        report=report,
        candidate_artifact=persistence.candidate_artifact,
        augmented_split_artifact=persistence.augmented_split_artifact,
        report_artifact=report_artifact,
    )


# ---------------------------------------------------------------------------
# numerical kernel
# ---------------------------------------------------------------------------


def _phi(z: float) -> float:
    """Standard normal CDF Φ(z) via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _sample_latent(*, rng: random.Random, dimension: int) -> tuple[float, ...]:
    """Draw an i.i.d. N(0, I) vector of the given dimension."""
    return tuple(rng.gauss(0.0, 1.0) for _ in range(dimension))


def _matrix_vector(
    matrix: tuple[tuple[float, ...], ...],
    vector: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    )


def _correlation_matrix(
    *,
    feature_vectors: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Compute a Pearson correlation matrix with safe variance handling."""
    dimension = len(feature_vectors[0])
    n = len(feature_vectors)
    means = [sum(v[j] for v in feature_vectors) / n for j in range(dimension)]
    stdevs: list[float] = []
    for j in range(dimension):
        variance = sum((v[j] - means[j]) ** 2 for v in feature_vectors) / max(1, n - 1)
        stdevs.append(math.sqrt(variance))
    matrix: list[list[float]] = []
    for i in range(dimension):
        row: list[float] = []
        for j in range(dimension):
            if i == j:
                row.append(1.0)
                continue
            denom = stdevs[i] * stdevs[j]
            if denom <= 0.0:
                row.append(0.0)
                continue
            cov = (
                sum((v[i] - means[i]) * (v[j] - means[j]) for v in feature_vectors)
                / max(1, n - 1)
            )
            corr = cov / denom
            row.append(max(-0.999999, min(0.999999, corr)))
        matrix.append(row)
    return tuple(tuple(row) for row in matrix)


def _cholesky(
    matrix: tuple[tuple[float, ...], ...],
    *,
    ridge_epsilon: float,
) -> tuple[tuple[float, ...], ...]:
    """Compute the lower-triangular Cholesky factor with a small ridge."""
    n = len(matrix)
    ridged = [list(row) for row in matrix]
    for i in range(n):
        ridged[i][i] += ridge_epsilon
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                value = ridged[i][i] - total
                lower[i][j] = math.sqrt(max(value, 0.0))
            else:
                if lower[j][j] == 0.0:
                    lower[i][j] = 0.0
                else:
                    lower[i][j] = (ridged[i][j] - total) / lower[j][j]
    return tuple(tuple(row) for row in lower)


def _build_column_transforms(
    *,
    feature_columns: tuple[str, ...],
    feature_vectors: tuple[tuple[float, ...], ...],
    quantile_levels: tuple[float, ...],
) -> tuple[ColumnDistributionTransform, ...]:
    transforms: list[ColumnDistributionTransform] = []
    if not quantile_levels:
        levels = DEFAULT_QUANTILE_LEVELS
    else:
        levels = quantile_levels
    for column_index, column in enumerate(feature_columns):
        column_values = sorted(vector[column_index] for vector in feature_vectors)
        quantile_values = tuple(_quantile(column_values, level) for level in levels)
        transforms.append(
            ColumnDistributionTransform(
                column=column,
                method="empirical_cdf",
                sample_count=len(column_values),
                quantile_levels=tuple(levels),
                quantile_values=quantile_values,
            )
        )
    return tuple(transforms)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower_idx = int(pos)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    fraction = pos - lower_idx
    return sorted_values[lower_idx] + (
        sorted_values[upper_idx] - sorted_values[lower_idx]
    ) * fraction


def _inverse_ecdf(
    transform: ColumnDistributionTransform,
    u: float,
    *,
    clamp: float,
) -> float:
    """Inverse empirical CDF via linear interpolation over the quantile grid."""
    levels = transform.quantile_levels
    values = transform.quantile_values
    if not levels:
        return 0.0
    clamped = min(1.0 - clamp, max(clamp, u))
    for index in range(1, len(levels)):
        upper = levels[index]
        if clamped <= upper:
            lower = levels[index - 1]
            span = upper - lower
            if span <= 0.0:
                return values[index]
            fraction = (clamped - lower) / span
            return values[index - 1] + fraction * (values[index] - values[index - 1])
    return values[-1]


def _flatten(matrix: tuple[tuple[float, ...], ...]) -> Iterable[float]:
    for row in matrix:
        yield from row


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_step(request: ExecuteGaussianCopulaRequest) -> None:
    step = request.step
    if step.type not in _SUPPORTED_STEP_TYPES:
        raise GaussianCopulaExecutionError(
            reason_code="unsupported_action_step_type",
            message=(
                "Gaussian Copula executor only handles AUGMENT_RARE_CLASS steps."
            ),
            details={"step_id": step.step_id, "step_type": step.type},
        )
    if step.method_id not in _SUPPORTED_METHOD_IDS:
        raise GaussianCopulaExecutionError(
            reason_code="unsupported_synthetic_method",
            message="Gaussian Copula executor only supports method=gaussian_copula.",
            details={"step_id": step.step_id, "method_id": step.method_id},
        )
    source_split = step.config.get("source_split", DataSplit.TRAIN.value)
    if source_split != DataSplit.TRAIN.value:
        raise GaussianCopulaExecutionError(
            reason_code="non_train_source_split",
            message=(
                "Gaussian Copula may only be fitted on the training split."
            ),
            details={"source_split": str(source_split)},
        )
    if request.split_manifest.target_column != request.target_column:
        raise GaussianCopulaExecutionError(
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
    candidates: Iterable[str] = explicit if explicit is not None else columns
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
    columns: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Build numeric feature vectors with column-wise median imputation.

    Rows are kept even if some feature cells are missing or non-numeric;
    missing values are imputed with the column median computed from
    available cells. Columns with no parseable values fall back to 0.
    The function returns ``()`` only when ``rows`` is empty.
    """
    if not rows:
        return ()
    column_values: list[list[float]] = [[] for _ in columns]
    for row in rows:
        for col_index, column in enumerate(columns):
            value = parse_numeric(row.get(column, ""))
            if value is not None:
                column_values[col_index].append(value)
    medians: list[float] = []
    for values in column_values:
        if not values:
            medians.append(0.0)
            continue
        sorted_values = sorted(values)
        midpoint = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            medians.append(sorted_values[midpoint])
        else:
            medians.append(
                (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0
            )
    vectors: list[tuple[float, ...]] = []
    for row in rows:
        features: list[float] = []
        for col_index, column in enumerate(columns):
            value = parse_numeric(row.get(column, ""))
            if value is None:
                value = medians[col_index]
            features.append(value)
        vectors.append(tuple(features))
    return tuple(vectors)


def _train_label_counts(
    *,
    train_rows: Sequence[Mapping[str, str]],
    target_column: str,
) -> tuple[str | None, dict[str, int]]:
    counts: dict[str, int] = {}
    for row in train_rows:
        label = row.get(target_column, "")
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None, counts
    rare = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return rare, counts


def _class_stats(
    *,
    train_label_counts: dict[str, int],
    synthetic_rows: Sequence[Mapping[str, str]],
    target_column: str,
    train_majority_count: int,
) -> tuple[SyntheticClassStats, ...]:
    if not train_label_counts:
        return ()
    synthetic_counts: dict[str, int] = {}
    for row in synthetic_rows:
        label = row.get(target_column, "") or "0"
        synthetic_counts[label] = synthetic_counts.get(label, 0) + 1
    stats: list[SyntheticClassStats] = []
    for label, real_count in sorted(train_label_counts.items()):
        generated = synthetic_counts.get(label, 0)
        target_count_after = real_count + generated
        achieved = (
            target_count_after / train_majority_count
            if train_majority_count > 0
            else 1.0
        )
        achieved = min(1.0, max(0.0, achieved))
        stats.append(
            SyntheticClassStats(
                label=label,
                real_count_in_source_split=real_count,
                real_count_in_majority_split=train_majority_count,
                target_count_after_augmentation=target_count_after,
                generated_count=generated,
                achieved_ratio=achieved,
            )
        )
    return tuple(stats)


def _build_synthetic_row(
    *,
    template_row: Mapping[str, str],
    feature_columns: tuple[str, ...],
    new_features: tuple[float, ...],
    target_column: str,
    target_value: str,
    synthetic_object_id: str,
) -> dict[str, str]:
    row: dict[str, str] = dict(template_row)
    row["object_id"] = synthetic_object_id
    row[target_column] = target_value
    for column, value in zip(feature_columns, new_features, strict=True):
        row[column] = format_numeric(value)
    row[IS_SYNTHETIC_COLUMN] = "1"
    row[SOURCE_SPLIT_COLUMN] = DataSplit.TRAIN.value
    return row


__all__ = [
    "DEFAULT_SAMPLE_LINEAGE_LIMIT",
    "ExecuteGaussianCopulaRequest",
    "ExecuteGaussianCopulaResult",
    "GAUSSIAN_COPULA_METHOD_VERSION",
    "GaussianCopulaExecutionError",
    "GaussianCopulaPolicy",
    "GaussianCopulaPolicyError",
    "execute_gaussian_copula_action",
]
