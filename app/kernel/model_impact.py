"""Sklearn baseline model-impact runner.

Computes baseline-vs-candidate metrics for a tabular supervised
classification candidate dataset version. PRD §21.1.1 / §21.2 specify
the baseline (LogisticRegression / RandomForest) and the metric set
(rare-class recall, macro F1, weighted F1, PR-AUC, confusion matrix).
DATASETS.md §11 specifies TSTR/TRTS rules; the runner emits both when
the candidate is a synthetic candidate.

Determinism notes:

- ``numpy.random.RandomState(seed)`` and ``sklearn``'s ``random_state``
  are always seeded.
- the train split rows are sorted by ``object_id`` before fitting so
  re-runs with the same artifact bytes produce the same model.
- the report records the baseline model algorithm, hyperparameters,
  feature columns, target column, library version and seed so audit
  can reproduce the result.

Privacy notes:

- the report carries only aggregate metrics, confusion-matrix counts
  and column names; raw rows never enter the artifact;
- when sklearn cannot fit (single class, empty split, non-numeric
  feature), the metric resolves to ``not_applicable`` with an explicit
  reason rather than crashing the workflow.
"""

from __future__ import annotations

import csv
import io
import json
import math
import uuid
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    MODEL_IMPACT_REPORT_SCHEMA_VERSION,
    ArtifactRef,
    BaselineModelConfig,
    ClassificationMetrics,
    ConfusionMatrixCell,
    DataSplit,
    ErrorCode,
    MetricStatus,
    ModelImpactReport,
    ModelImpactReportLineage,
    ModelImpactVerdict,
    SplitManifest,
    SyntheticUtilityStatus,
    TstrTrtsMetrics,
)
from app.domain.common import NonEmptyStr, Sha256Digest

MODEL_IMPACT_REPORT_KIND = "model_impact_report"
MODEL_IMPACT_REPORT_FORMAT = "json"
MODEL_IMPACT_REPORT_MEDIA_TYPE = "application/json"

DEFAULT_BASELINE_ALGORITHM = "LogisticRegression"
DEFAULT_RANDOM_SEED = 42
DEFAULT_TSTR_MACRO_F1_THRESHOLD = 0.10
"""Drop in macro_f1 from baseline that flips synthetic_method_status to rejected."""

DEFAULT_TRTS_UNSTABLE_THRESHOLD = 0.20
"""TRTS macro_f1 deviation from candidate that flips status to requires_review."""

DEFAULT_RARE_CLASS_RECALL_DEGRADED_DROP = 0.05
"""Drop in rare-class recall that marks the candidate degraded."""

DEFAULT_MACRO_F1_DEGRADED_DROP = 0.03
"""Drop in macro_f1 that marks the candidate degraded."""

DEFAULT_WEIGHTED_F1_DEGRADED_DROP = 0.03
"""Drop in weighted_f1 that marks the candidate degraded."""

DEFAULT_PR_AUC_DEGRADED_DROP = 0.03
"""Drop in PR-AUC that marks the candidate degraded when PR-AUC is available."""

_IS_SYNTHETIC_COLUMN = "is_synthetic"


class ModelImpactRunnerError(ValueError):
    """Raised when the model-impact runner cannot run safely."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        code: ErrorCode = ErrorCode.MODEL_IMPACT_NOT_ELIGIBLE,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.code = code
        self.details = {} if details is None else details


class RunModelImpactRequest(BaseModel):
    """Inputs for :func:`run_model_impact`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    organization_id: NonEmptyStr | None = None
    project_id: NonEmptyStr | None = None
    source_artifact: ArtifactRef
    candidate_artifact: ArtifactRef
    source_split_manifest: SplitManifest
    candidate_split_manifest: SplitManifest
    feature_columns: tuple[NonEmptyStr, ...] = Field(min_length=1)
    target_column: NonEmptyStr
    rare_class_label: NonEmptyStr
    candidate_is_synthetic: bool = False
    candidate_version_artifact: ArtifactRef | None = None
    eligibility_report_artifact: ArtifactRef | None = None
    source_split_manifest_artifact: ArtifactRef | None = None
    candidate_split_manifest_artifact: ArtifactRef | None = None
    validation_gates_report_artifact: ArtifactRef | None = None
    synthetic_dataset_report_artifact: ArtifactRef | None = None
    validation_gates_blocker_present: bool = False
    random_seed: int = DEFAULT_RANDOM_SEED
    algorithm: NonEmptyStr = DEFAULT_BASELINE_ALGORITHM
    tstr_macro_f1_threshold: float = DEFAULT_TSTR_MACRO_F1_THRESHOLD
    trts_unstable_threshold: float = DEFAULT_TRTS_UNSTABLE_THRESHOLD
    rare_class_recall_degraded_drop: float = DEFAULT_RARE_CLASS_RECALL_DEGRADED_DROP
    macro_f1_degraded_drop: float = DEFAULT_MACRO_F1_DEGRADED_DROP
    weighted_f1_degraded_drop: float = DEFAULT_WEIGHTED_F1_DEGRADED_DROP
    pr_auc_degraded_drop: float = DEFAULT_PR_AUC_DEGRADED_DROP
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class RunModelImpactResult:
    """Persisted model-impact report and its registry record."""

    report: ModelImpactReport
    report_artifact: RegisteredArtifact


def run_model_impact(
    request: RunModelImpactRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> RunModelImpactResult:
    """Fit a baseline classifier on source / candidate splits and emit the report."""
    source_rows = _read_csv_rows(storage=storage, artifact=request.source_artifact)
    candidate_rows = _read_csv_rows(
        storage=storage, artifact=request.candidate_artifact
    )

    feature_columns = tuple(request.feature_columns)
    excluded_columns = _resolve_excluded_columns(
        all_columns=_columns_of(candidate_rows),
        feature_columns=feature_columns,
        target_column=request.target_column,
    )

    baseline_metrics, baseline_status, baseline_reason = _train_and_evaluate(
        train_rows=_select_split_rows(source_rows, request.source_split_manifest, DataSplit.TRAIN),
        test_rows=_select_split_rows(source_rows, request.source_split_manifest, DataSplit.TEST)
        or _select_split_rows(
            source_rows, request.source_split_manifest, DataSplit.VALIDATION
        ),
        feature_columns=feature_columns,
        target_column=request.target_column,
        rare_class_label=request.rare_class_label,
        random_seed=request.random_seed,
    )
    candidate_train_rows = _select_split_rows(
        candidate_rows, request.candidate_split_manifest, DataSplit.TRAIN
    )
    candidate_test_rows = _select_split_rows(
        candidate_rows, request.candidate_split_manifest, DataSplit.TEST
    ) or _select_split_rows(
        candidate_rows, request.candidate_split_manifest, DataSplit.VALIDATION
    )
    candidate_metrics, candidate_status, candidate_reason = _train_and_evaluate(
        train_rows=candidate_train_rows,
        test_rows=candidate_test_rows,
        feature_columns=feature_columns,
        target_column=request.target_column,
        rare_class_label=request.rare_class_label,
        random_seed=request.random_seed,
    )

    if baseline_status is not MetricStatus.AVAILABLE:
        raise ModelImpactRunnerError(
            reason_code=baseline_reason or "baseline_metrics_unavailable",
            message=baseline_reason
            or "Could not fit baseline classifier on source split.",
        )
    if candidate_status is not MetricStatus.AVAILABLE:
        raise ModelImpactRunnerError(
            reason_code=candidate_reason or "candidate_metrics_unavailable",
            message=candidate_reason
            or "Could not fit baseline classifier on candidate split.",
        )

    rare_class_recall_delta = (
        candidate_metrics.rare_class_recall - baseline_metrics.rare_class_recall
    )
    macro_f1_delta = candidate_metrics.macro_f1 - baseline_metrics.macro_f1
    weighted_f1_delta = candidate_metrics.weighted_f1 - baseline_metrics.weighted_f1
    pr_auc_delta = (
        (candidate_metrics.pr_auc - baseline_metrics.pr_auc)
        if (
            candidate_metrics.pr_auc is not None
            and baseline_metrics.pr_auc is not None
        )
        else None
    )

    tstr_trts = _compute_tstr_trts(
        candidate_train_rows=candidate_train_rows,
        candidate_test_rows=candidate_test_rows,
        source_train_rows=_select_split_rows(
            source_rows, request.source_split_manifest, DataSplit.TRAIN
        ),
        source_test_rows=_select_split_rows(
            source_rows, request.source_split_manifest, DataSplit.TEST
        )
        or _select_split_rows(
            source_rows, request.source_split_manifest, DataSplit.VALIDATION
        ),
        feature_columns=feature_columns,
        target_column=request.target_column,
        rare_class_label=request.rare_class_label,
        random_seed=request.random_seed,
        candidate_is_synthetic=request.candidate_is_synthetic,
        baseline_macro_f1=baseline_metrics.macro_f1,
        candidate_macro_f1=candidate_metrics.macro_f1,
        tstr_macro_f1_threshold=request.tstr_macro_f1_threshold,
        trts_unstable_threshold=request.trts_unstable_threshold,
    )

    verdict, verdict_reasons = _classify_verdict(
        rare_class_recall_delta=rare_class_recall_delta,
        macro_f1_delta=macro_f1_delta,
        weighted_f1_delta=weighted_f1_delta,
        pr_auc_delta=pr_auc_delta,
        rare_class_recall_drop_threshold=request.rare_class_recall_degraded_drop,
        macro_f1_drop_threshold=request.macro_f1_degraded_drop,
        weighted_f1_drop_threshold=request.weighted_f1_degraded_drop,
        pr_auc_drop_threshold=request.pr_auc_degraded_drop,
        validation_gates_blocker_present=request.validation_gates_blocker_present,
    )
    synthetic_status, synthetic_reasons = _classify_synthetic_utility(
        candidate_is_synthetic=request.candidate_is_synthetic,
        verdict=verdict,
        rare_class_recall_delta=rare_class_recall_delta,
        macro_f1_delta=macro_f1_delta,
        weighted_f1_delta=weighted_f1_delta,
        pr_auc_delta=pr_auc_delta,
        weighted_f1_drop_threshold=request.weighted_f1_degraded_drop,
        pr_auc_drop_threshold=request.pr_auc_degraded_drop,
        tstr_trts=tstr_trts,
        validation_gates_blocker_present=request.validation_gates_blocker_present,
    )

    baseline_config = _model_config(
        algorithm=request.algorithm,
        feature_columns=feature_columns,
        target_column=request.target_column,
        excluded_columns=excluded_columns,
        random_seed=request.random_seed,
    )
    candidate_config = baseline_config  # same algorithm/seed/feature set per request

    report = ModelImpactReport(
        report_id=request.report_id or f"model_impact_report_{uuid.uuid4().hex[:16]}",
        report_schema_version=MODEL_IMPACT_REPORT_SCHEMA_VERSION,
        metric_library="scikit-learn",
        metric_library_version=sklearn.__version__,
        verdict=verdict,
        verdict_reason_codes=tuple(verdict_reasons),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        rare_class_recall_before=baseline_metrics.rare_class_recall,
        rare_class_recall_after=candidate_metrics.rare_class_recall,
        rare_class_recall_delta=rare_class_recall_delta,
        macro_f1_before=baseline_metrics.macro_f1,
        macro_f1_after=candidate_metrics.macro_f1,
        macro_f1_delta=macro_f1_delta,
        weighted_f1_before=baseline_metrics.weighted_f1,
        weighted_f1_after=candidate_metrics.weighted_f1,
        weighted_f1_delta=weighted_f1_delta,
        pr_auc_before=baseline_metrics.pr_auc,
        pr_auc_after=candidate_metrics.pr_auc,
        pr_auc_delta=pr_auc_delta,
        pr_auc_status=baseline_metrics.pr_auc_status
        if baseline_metrics.pr_auc_status is candidate_metrics.pr_auc_status
        else MetricStatus.NOT_APPLICABLE,
        pr_auc_reason=(
            baseline_metrics.pr_auc_reason or candidate_metrics.pr_auc_reason
        ),
        tstr_trts=tstr_trts,
        synthetic_utility_status=synthetic_status,
        synthetic_utility_reason_codes=tuple(synthetic_reasons),
        baseline_model_config=baseline_config,
        candidate_model_config=candidate_config,
        lineage=ModelImpactReportLineage(
            organization_id=request.organization_id,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            parent_version_id=request.parent_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            candidate_version_artifact=request.candidate_version_artifact,
            eligibility_report_artifact=request.eligibility_report_artifact,
            source_split_manifest=request.source_split_manifest_artifact,
            candidate_split_manifest=request.candidate_split_manifest_artifact,
            validation_gates_report_artifact=request.validation_gates_report_artifact,
            synthetic_dataset_report_artifact=request.synthetic_dataset_report_artifact,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )

    artifact = registry.save_artifact(
        artifact_kind=MODEL_IMPACT_REPORT_KIND,
        data=_serialize(report),
        artifact_format=MODEL_IMPACT_REPORT_FORMAT,
        media_type=MODEL_IMPACT_REPORT_MEDIA_TYPE,
        schema_version=MODEL_IMPACT_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "verdict": verdict.value,
            "synthetic-utility-status": synthetic_status.value,
            "rare-class-recall-before": _format_score(baseline_metrics.rare_class_recall),
            "rare-class-recall-after": _format_score(candidate_metrics.rare_class_recall),
            "macro-f1-before": _format_score(baseline_metrics.macro_f1),
            "macro-f1-after": _format_score(candidate_metrics.macro_f1),
            "metric-library-version": sklearn.__version__,
            "random-seed": str(request.random_seed),
            "algorithm": request.algorithm,
        },
    )
    return RunModelImpactResult(report=report, report_artifact=artifact)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_csv_rows(
    *,
    storage: MinioObjectStorageAdapter,
    artifact: ArtifactRef,
) -> tuple[dict[str, str], ...]:
    stored = storage.get(artifact.uri)
    text = stored.data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = []
    for index, row in enumerate(reader):
        rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
        if not rows[-1].get("object_id"):
            rows[-1]["object_id"] = f"row_{index:06d}"
    return tuple(rows)


def _columns_of(rows: Sequence[dict[str, str]]) -> tuple[str, ...]:
    if not rows:
        return ()
    return tuple(rows[0].keys())


def _resolve_excluded_columns(
    *,
    all_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[str, ...]:
    feature_set = set(feature_columns)
    excluded: list[str] = []
    for column in all_columns:
        if column in feature_set:
            continue
        if column == target_column:
            continue
        excluded.append(column)
    return tuple(excluded)


def _select_split_rows(
    rows: Sequence[dict[str, str]],
    manifest: SplitManifest,
    split: DataSplit,
) -> tuple[dict[str, str], ...]:
    target_ids = {
        assignment.object_id
        for assignment in manifest.assignments
        if assignment.split is split
    }
    if not target_ids:
        return ()
    by_id = {row["object_id"]: row for row in rows if row.get("object_id")}
    selected = [
        by_id[object_id]
        for object_id in sorted(target_ids)
        if object_id in by_id
    ]
    return tuple(selected)


def _to_xy(
    *,
    rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, len(feature_columns)), dtype=float), np.zeros((0,), dtype=str)
    features: list[list[float]] = []
    labels: list[str] = []
    for row in rows:
        vector: list[float] = []
        for column in feature_columns:
            raw = row.get(column, "")
            try:
                vector.append(float(raw) if raw != "" else 0.0)
            except (TypeError, ValueError):
                vector.append(0.0)
        features.append(vector)
        labels.append(str(row.get(target_column, "")))
    return np.array(features, dtype=float), np.array(labels, dtype=str)


def _train_and_evaluate(
    *,
    train_rows: Sequence[dict[str, str]],
    test_rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    target_column: str,
    rare_class_label: str,
    random_seed: int,
) -> tuple[ClassificationMetrics, MetricStatus, str | None]:
    if not train_rows:
        return _zero_metrics(rare_class_label), MetricStatus.NOT_APPLICABLE, "empty_train_split"
    if not test_rows:
        return _zero_metrics(rare_class_label), MetricStatus.NOT_APPLICABLE, "empty_test_split"
    x_train, y_train = _to_xy(
        rows=train_rows, feature_columns=feature_columns, target_column=target_column
    )
    x_test, y_test = _to_xy(
        rows=test_rows, feature_columns=feature_columns, target_column=target_column
    )
    if len(set(y_train)) < 2:
        return (
            _zero_metrics(rare_class_label),
            MetricStatus.NOT_APPLICABLE,
            "single_class_in_train_split",
        )
    if len(y_test) == 0:
        return (
            _zero_metrics(rare_class_label),
            MetricStatus.NOT_APPLICABLE,
            "empty_test_split",
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = LogisticRegression(
            random_state=random_seed,
            max_iter=500,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        try:
            y_proba = model.predict_proba(x_test)
            classes = list(model.classes_)
            if rare_class_label in classes:
                rare_proba = y_proba[:, classes.index(rare_class_label)]
            else:
                rare_proba = None
        except (AttributeError, ValueError):
            rare_proba = None

    rare_recall = float(
        recall_score(
            y_test,
            y_pred,
            labels=[rare_class_label],
            average=None,
            zero_division=0,
        )[0]
    )
    rare_precision = float(
        precision_score(
            y_test,
            y_pred,
            labels=[rare_class_label],
            average=None,
            zero_division=0,
        )[0]
    )
    macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    pr_auc: float | None = None
    pr_auc_status = MetricStatus.AVAILABLE
    pr_auc_reason: str | None = None
    if rare_proba is None:
        pr_auc_status = MetricStatus.NOT_APPLICABLE
        pr_auc_reason = "rare_class_proba_not_available"
    elif rare_class_label not in set(y_test):
        pr_auc_status = MetricStatus.NOT_APPLICABLE
        pr_auc_reason = "rare_class_absent_in_test_split"
    else:
        y_test_binary = (y_test == rare_class_label).astype(int)
        try:
            pr_auc = float(average_precision_score(y_test_binary, rare_proba))
        except ValueError:
            pr_auc_status = MetricStatus.NOT_APPLICABLE
            pr_auc_reason = "pr_auc_computation_failed"

    classes_for_matrix = sorted({*y_test.tolist(), *y_pred.tolist()})
    matrix = confusion_matrix(y_test, y_pred, labels=classes_for_matrix)
    cells: list[ConfusionMatrixCell] = []
    for true_idx, true_label in enumerate(classes_for_matrix):
        for predicted_idx, predicted_label in enumerate(classes_for_matrix):
            cells.append(
                ConfusionMatrixCell(
                    true_label=true_label,
                    predicted_label=predicted_label,
                    count=int(matrix[true_idx][predicted_idx]),
                )
            )

    metrics = ClassificationMetrics(
        rare_class_label=rare_class_label,
        rare_class_recall=_clamp01(rare_recall),
        rare_class_precision=_clamp01(rare_precision),
        macro_f1=_clamp01(macro),
        weighted_f1=_clamp01(weighted),
        pr_auc=_clamp01(pr_auc) if pr_auc is not None else None,
        pr_auc_status=pr_auc_status,
        pr_auc_reason=pr_auc_reason,
        confusion_matrix=tuple(cells),
        sample_count=len(y_test),
    )
    return metrics, MetricStatus.AVAILABLE, None


def _zero_metrics(rare_class_label: str) -> ClassificationMetrics:
    return ClassificationMetrics(
        rare_class_label=rare_class_label,
        rare_class_recall=0.0,
        rare_class_precision=0.0,
        macro_f1=0.0,
        weighted_f1=0.0,
        pr_auc=None,
        pr_auc_status=MetricStatus.NOT_APPLICABLE,
        pr_auc_reason="metrics_unavailable",
        confusion_matrix=(),
        sample_count=0,
    )


def _compute_tstr_trts(
    *,
    candidate_train_rows: Sequence[dict[str, str]],
    candidate_test_rows: Sequence[dict[str, str]],
    source_train_rows: Sequence[dict[str, str]],
    source_test_rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    target_column: str,
    rare_class_label: str,
    random_seed: int,
    candidate_is_synthetic: bool,
    baseline_macro_f1: float,
    candidate_macro_f1: float,
    tstr_macro_f1_threshold: float,
    trts_unstable_threshold: float,
) -> TstrTrtsMetrics:
    if not candidate_is_synthetic:
        return TstrTrtsMetrics(
            status=MetricStatus.NOT_APPLICABLE,
            reason="candidate_is_not_synthetic",
        )
    synthetic_train_rows = tuple(
        row for row in candidate_train_rows if row.get(_IS_SYNTHETIC_COLUMN) == "1"
    )
    if not synthetic_train_rows:
        return TstrTrtsMetrics(
            status=MetricStatus.NOT_APPLICABLE,
            reason="candidate_has_no_synthetic_train_rows",
        )

    # TSTR is strict per DATASETS.md §11.1: train on synthetic rows
    # only, then test on real source rows. Targeted rare-class methods
    # such as SMOTE often produce a single synthetic class; in that case
    # TSTR is explicitly not_applicable instead of being silently
    # replaced with an augmented-train metric.
    tstr_metrics, tstr_status, tstr_reason = _train_and_evaluate(
        train_rows=synthetic_train_rows,
        test_rows=source_test_rows,
        feature_columns=feature_columns,
        target_column=target_column,
        rare_class_label=rare_class_label,
        random_seed=random_seed,
    )
    # TRTS: train on real source train split, test on synthetic-only
    # rows. PRD §11.2 requires a fixed real-train side.
    trts_metrics, trts_status, trts_reason = _train_and_evaluate(
        train_rows=source_train_rows,
        test_rows=synthetic_train_rows,
        feature_columns=feature_columns,
        target_column=target_column,
        rare_class_label=rare_class_label,
        random_seed=random_seed,
    )
    tstr_macro_f1_drop = (
        baseline_macro_f1 - tstr_metrics.macro_f1
        if tstr_status is MetricStatus.AVAILABLE
        else None
    )
    trts_macro_f1_delta = (
        candidate_macro_f1 - trts_metrics.macro_f1
        if trts_status is MetricStatus.AVAILABLE
        else None
    )
    if tstr_status is not MetricStatus.AVAILABLE or trts_status is not MetricStatus.AVAILABLE:
        unavailable: list[str] = []
        if tstr_status is not MetricStatus.AVAILABLE:
            unavailable.append(f"tstr_{tstr_reason or 'unavailable'}")
        if trts_status is not MetricStatus.AVAILABLE:
            unavailable.append(f"trts_{trts_reason or 'unavailable'}")
        return TstrTrtsMetrics(
            status=MetricStatus.NOT_APPLICABLE,
            reason=";".join(unavailable) or "tstr_trts_unavailable",
            tstr_metrics=(
                tstr_metrics if tstr_status is MetricStatus.AVAILABLE else None
            ),
            trts_metrics=(
                trts_metrics if trts_status is MetricStatus.AVAILABLE else None
            ),
            tstr_macro_f1_drop=tstr_macro_f1_drop,
            tstr_macro_f1_threshold=tstr_macro_f1_threshold,
            trts_macro_f1_delta=trts_macro_f1_delta,
            trts_unstable_threshold=trts_unstable_threshold,
        )
    return TstrTrtsMetrics(
        status=MetricStatus.AVAILABLE,
        tstr_metrics=tstr_metrics,
        trts_metrics=trts_metrics,
        tstr_macro_f1_drop=tstr_macro_f1_drop,
        tstr_macro_f1_threshold=tstr_macro_f1_threshold,
        trts_macro_f1_delta=trts_macro_f1_delta,
        trts_unstable_threshold=trts_unstable_threshold,
    )


def _classify_verdict(
    *,
    rare_class_recall_delta: float,
    macro_f1_delta: float,
    weighted_f1_delta: float,
    pr_auc_delta: float | None,
    rare_class_recall_drop_threshold: float,
    macro_f1_drop_threshold: float,
    weighted_f1_drop_threshold: float,
    pr_auc_drop_threshold: float,
    validation_gates_blocker_present: bool,
) -> tuple[ModelImpactVerdict, list[str]]:
    reasons: list[str] = []
    if validation_gates_blocker_present:
        reasons.append("validation_gates_blocker_present")
        return ModelImpactVerdict.REJECTED, reasons
    degraded_reasons: list[str] = []
    if rare_class_recall_delta <= -rare_class_recall_drop_threshold:
        degraded_reasons.append("rare_class_recall_degraded")
    if macro_f1_delta <= -macro_f1_drop_threshold:
        degraded_reasons.append("macro_f1_degraded")
    if weighted_f1_delta <= -weighted_f1_drop_threshold:
        degraded_reasons.append("weighted_f1_degraded")
    if pr_auc_delta is not None and pr_auc_delta <= -pr_auc_drop_threshold:
        degraded_reasons.append("pr_auc_degraded")
    if degraded_reasons:
        reasons.append("metrics_degraded")
        reasons.extend(degraded_reasons)
        return ModelImpactVerdict.DEGRADED, reasons
    pr_auc_non_degraded = pr_auc_delta is None or pr_auc_delta >= 0
    if (
        rare_class_recall_delta > 0
        and macro_f1_delta >= 0
        and weighted_f1_delta >= 0
        and pr_auc_non_degraded
    ):
        reasons.append("rare_class_recall_improved")
        if macro_f1_delta > 0:
            reasons.append("macro_f1_improved")
        if weighted_f1_delta > 0:
            reasons.append("weighted_f1_improved")
        if pr_auc_delta is not None and pr_auc_delta > 0:
            reasons.append("pr_auc_improved")
        return ModelImpactVerdict.IMPROVED, reasons
    if (
        math.isclose(rare_class_recall_delta, 0.0, abs_tol=1e-9)
        and math.isclose(macro_f1_delta, 0.0, abs_tol=1e-9)
        and math.isclose(weighted_f1_delta, 0.0, abs_tol=1e-9)
        and (pr_auc_delta is None or math.isclose(pr_auc_delta, 0.0, abs_tol=1e-9))
    ):
        reasons.append("metrics_unchanged")
        return ModelImpactVerdict.REQUIRES_REVIEW, reasons
    reasons.append("mixed_metric_movement")
    return ModelImpactVerdict.REQUIRES_REVIEW, reasons


def _classify_synthetic_utility(
    *,
    candidate_is_synthetic: bool,
    verdict: ModelImpactVerdict,
    rare_class_recall_delta: float,
    macro_f1_delta: float,
    weighted_f1_delta: float,
    pr_auc_delta: float | None,
    weighted_f1_drop_threshold: float,
    pr_auc_drop_threshold: float,
    tstr_trts: TstrTrtsMetrics,
    validation_gates_blocker_present: bool,
) -> tuple[SyntheticUtilityStatus, list[str]]:
    if not candidate_is_synthetic:
        return SyntheticUtilityStatus.NOT_APPLICABLE, []
    if validation_gates_blocker_present:
        return SyntheticUtilityStatus.REJECTED, ["validation_gates_blocker_present"]
    degraded_reasons: list[str] = []
    if verdict is ModelImpactVerdict.DEGRADED:
        degraded_reasons.append("metrics_degraded")
    if weighted_f1_delta <= -weighted_f1_drop_threshold:
        degraded_reasons.append("weighted_f1_degraded")
    if pr_auc_delta is not None and pr_auc_delta <= -pr_auc_drop_threshold:
        degraded_reasons.append("pr_auc_degraded")
    if degraded_reasons:
        return SyntheticUtilityStatus.REJECTED, degraded_reasons
    if tstr_trts.status is not MetricStatus.AVAILABLE:
        return (
            SyntheticUtilityStatus.REQUIRES_REVIEW,
            [tstr_trts.reason or "tstr_trts_not_available"],
        )
    reasons: list[str] = []
    if (
        tstr_trts.tstr_macro_f1_drop is not None
        and tstr_trts.tstr_macro_f1_threshold is not None
        and tstr_trts.tstr_macro_f1_drop > tstr_trts.tstr_macro_f1_threshold
    ):
        reasons.append("tstr_macro_f1_drop_above_threshold")
        return SyntheticUtilityStatus.REJECTED, reasons
    if (
        tstr_trts.trts_macro_f1_delta is not None
        and tstr_trts.trts_unstable_threshold is not None
        and abs(tstr_trts.trts_macro_f1_delta) > tstr_trts.trts_unstable_threshold
    ):
        reasons.append("trts_macro_f1_unstable")
        return SyntheticUtilityStatus.REQUIRES_REVIEW, reasons
    if (
        verdict is ModelImpactVerdict.IMPROVED
        and rare_class_recall_delta > 0
        and macro_f1_delta >= 0
        and weighted_f1_delta >= 0
        and (pr_auc_delta is None or pr_auc_delta >= 0)
    ):
        reasons.append("tstr_within_threshold")
        reasons.append("rare_class_recall_improved")
        return SyntheticUtilityStatus.RECOMMENDED, reasons
    reasons.append("metrics_inconclusive")
    return SyntheticUtilityStatus.REQUIRES_REVIEW, reasons


def _model_config(
    *,
    algorithm: str,
    feature_columns: Sequence[str],
    target_column: str,
    excluded_columns: Sequence[str],
    random_seed: int,
) -> BaselineModelConfig:
    return BaselineModelConfig(
        algorithm=algorithm,
        library="scikit-learn",
        library_version=sklearn.__version__,
        hyperparameters={
            "max_iter": 500,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        feature_columns=tuple(feature_columns),
        target_column=target_column,
        excluded_columns=tuple(excluded_columns),
        random_seed=random_seed,
    )


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def _format_score(value: float) -> str:
    return f"{value:.6f}"


def _serialize(report: ModelImpactReport) -> bytes:
    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
    ).encode("utf-8")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.append(value)
    return tuple(seen)


__all__ = [
    "DEFAULT_BASELINE_ALGORITHM",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_MACRO_F1_DEGRADED_DROP",
    "DEFAULT_PR_AUC_DEGRADED_DROP",
    "DEFAULT_RARE_CLASS_RECALL_DEGRADED_DROP",
    "DEFAULT_TRTS_UNSTABLE_THRESHOLD",
    "DEFAULT_TSTR_MACRO_F1_THRESHOLD",
    "DEFAULT_WEIGHTED_F1_DEGRADED_DROP",
    "MODEL_IMPACT_REPORT_FORMAT",
    "MODEL_IMPACT_REPORT_KIND",
    "MODEL_IMPACT_REPORT_MEDIA_TYPE",
    "ModelImpactRunnerError",
    "RunModelImpactRequest",
    "RunModelImpactResult",
    "run_model_impact",
]
