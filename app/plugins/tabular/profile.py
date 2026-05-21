"""Tabular profiler: schema inference and base profile report.

TASK-023 scope:

* compute ``row_count``, ``column_count``;
* infer column type and nullability;
* detect ``target_column``, group keys, id-like columns and PII-like columns;
* produce a contract-shaped :class:`TabularProfileReport`;
* persist the JSON report through :class:`ArtifactRegistry` so the
  ``DataForgeReport.detail_artifacts`` slot can hold an
  :class:`ArtifactRef` to the registered report.

TASK-024 extension:

* compute per-column missingness (``missing_rate``);
* compute missingness conditioned on the detected target column and
  optionally on a segment column (``customer_segment`` for the demo
  archive, configurable via :attr:`ProfileBuildRequest.segment_column`);
* expose ``target_column_missing`` + ``missing_target_count`` so Decision
  Core can raise a hard-blocker candidate when the dataset has missing
  target values (training cannot proceed without target labels).

TASK-025 extension:

* exact duplicate-row detection over a deterministic signature (all
  columns except the id column) with affected ``object_id`` set;
* IQR-based outlier detection for numeric columns;
* class-imbalance diagnostics for the target column with explicit
  decomposition: ``minority_class_share = min_c n_c / N``,
  ``imbalance_ratio = max_c n_c / min_c n_c``,
  ``balance_score = 1 / log(1 + imbalance_ratio)``, alternative
  ``balance_score_alternative = min_c(n_c) / mean_c(n_c)``, and
  ``effective_number_of_samples = (1 - β^n) / (1 - β)`` (Cui 2019);
* leakage candidate detection by column-name heuristic and by
  observed target match rate.

Privacy rules honored here:

* the profiler never logs raw row payloads or column samples;
* ``sample_value`` is captured only for non-PII-like columns; PII-like
  columns surface ``sample_value=None`` so the profile artifact does not
  carry raw PII even by accident;
* ``pii_like_columns`` are detected by column name heuristics for the
  MVP. Deep PII detection is the responsibility of the text/OCR plugin
  and dedicated tabular PII detectors (later tasks).

Implementation details:

* the profiler uses stdlib ``csv`` and processes rows lazily; rows are
  consumed once and per-column statistics are accumulated incrementally.
* column type inference uses a deterministic precedence: identifier ->
  boolean -> numeric_integer -> numeric_float -> datetime -> categorical
  -> text. The decision is made from the union of all observed non-null
  values.
* role detection is conservative: the profiler picks at most one
  target column from a small allowlist (``is_fraud``, ``label``,
  ``target``) and at most one group key from common business keys.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ArtifactRef,
    BusinessRulesReport,
    ClassCount,
    ClassImbalanceDiagnostics,
    ColumnMissingness,
    ColumnOutlierStats,
    ColumnProfile,
    ColumnRole,
    ColumnType,
    DuplicateDiagnostics,
    LeakageCandidate,
    LeakageDiagnostics,
    MissingnessByGroup,
    MissingnessDiagnostics,
    MissingnessGroupStats,
    OutlierDiagnostics,
    TabularProfileLineage,
    TabularProfileReport,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.ingestion.archive_reader import (
    ArchiveEntryKind,
    ArchiveFileDescriptor,
    ArchiveReader,
)
from app.plugins.tabular.rules import BusinessRule, BusinessRuleEvaluator

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    pass

PROFILE_REPORT_KIND = "tabular_profile_report"
PROFILE_REPORT_FORMAT = "json"
PROFILE_REPORT_MEDIA_TYPE = "application/json"
PROFILE_REPORT_SCHEMA_VERSION = "tabular_profile_report.v1"

_TARGET_COLUMN_NAMES: tuple[str, ...] = ("is_fraud", "label", "target")
_GROUP_KEY_NAMES: tuple[str, ...] = (
    "customer_id_hash",
    "customer_id",
    "case_id",
    "case_id_hash",
    "household_id",
    "group_id",
)
_ID_COLUMN_NAMES: tuple[str, ...] = (
    "object_id",
    "row_id",
    "transaction_id",
    "document_id",
    "support_ticket_id",
)
# Column-name heuristics for PII-like columns. Detection by content is
# delegated to the text/OCR plugin and dedicated tabular PII detectors.
_PII_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|_)e?mail($|_)", re.IGNORECASE),
    re.compile(r"(^|_)phone($|_)", re.IGNORECASE),
    re.compile(r"(^|_)passport($|_)", re.IGNORECASE),
    re.compile(r"(^|_)ssn($|_)", re.IGNORECASE),
    re.compile(r"(^|_)full_?name($|_)", re.IGNORECASE),
    re.compile(r"(^|_)address($|_)", re.IGNORECASE),
)
_LEAKAGE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"manual_review", re.IGNORECASE),
    re.compile(r"is_fraud_predict", re.IGNORECASE),
    re.compile(r"target_leak", re.IGNORECASE),
)

_OUTLIER_METHOD = "iqr_1.5"
_OUTLIER_IQR_MULTIPLIER = 1.5
_OUTLIER_AFFECTED_ID_LIMIT = 25
_DUPLICATE_AFFECTED_ID_LIMIT = 100
_LEAKAGE_TARGET_MATCH_THRESHOLD = 0.5
_CLASS_IMBALANCE_BETA = 0.999
_CLASS_IMBALANCE_RATIO_THRESHOLD = 1.0001

_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
)


class ProfileBuildRequest(BaseModel):
    """Inputs the tabular profiler requires from the orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    parent_version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    source_artifact_id: NonEmptyStr
    source_system: NonEmptyStr = "transactions"
    segment_column: str | None = "customer_segment"
    id_column: str = "object_id"
    outlier_columns: tuple[str, ...] | None = None
    business_rules: tuple[BusinessRule, ...] = ()
    business_rules_version: str = "tabular_business_rules.v1"
    business_rules_config_hash: Sha256Digest | None = None


@dataclass(frozen=True)
class BuildProfileResult:
    """Result of a tabular profile build."""

    profile_report: TabularProfileReport
    artifact: RegisteredArtifact


def infer_tabular_profile(
    rows: Iterable[dict[str, str]],
    *,
    columns: tuple[str, ...],
    request: ProfileBuildRequest,
    source_manifest_artifact: ArtifactRef,
    profile_id: str | None = None,
    generated_at: datetime | None = None,
) -> TabularProfileReport:
    """Build a ``TabularProfileReport`` from an iterable of CSV row dicts.

    This function is plugin-internal and never persists the report; use
    :func:`build_tabular_profile_report` for the full build/persist flow.
    The split keeps the inference logic easy to test against in-memory
    rows (no archive, no storage, no registry).
    """
    if not columns:
        raise ValueError("tabular profile requires at least one column")

    target_column = _select_target_column(columns)
    segment_column = _select_segment_column(columns, request.segment_column)
    id_column = request.id_column if request.id_column in columns else None
    signature_columns = tuple(
        sorted(c for c in columns if c != id_column)
    )
    outlier_columns = _resolve_outlier_columns(columns, request.outlier_columns)
    rule_evaluator: BusinessRuleEvaluator | None = None
    if request.business_rules:
        rule_evaluator = BusinessRuleEvaluator(
            request.business_rules,
            rules_version=request.business_rules_version,
            rules_config_hash=request.business_rules_config_hash,
        )

    aggregators = {column: _ColumnAggregator(name=column) for column in columns}
    target_missing_count = 0
    # Per-column, per-target-value missing/total counters.
    by_target_counters: dict[str, dict[str, _MissingTotal]] = {
        column: {} for column in columns
    }
    by_segment_counters: dict[str, dict[str, _MissingTotal]] = {
        column: {} for column in columns
    }

    # Duplicate signature -> list of object ids (with that signature).
    signature_to_object_ids: dict[tuple[str, ...], list[str]] = {}
    # Per numeric column: list of (value, object_id) tuples for IQR.
    numeric_samples: dict[str, list[tuple[float, str]]] = {
        column: [] for column in outlier_columns
    }
    # Per target-class: count of rows.
    target_class_counts: dict[str, int] = {}
    # Leakage detection: per candidate column, count of rows where
    # the column value equals the target value (string equality).
    leakage_candidate_columns = tuple(
        column
        for column in columns
        if any(
            pattern.search(column) is not None for pattern in _LEAKAGE_NAME_PATTERNS
        )
    )
    leakage_match_counters: dict[str, _MissingTotal] = {
        column: _MissingTotal() for column in leakage_candidate_columns
    }

    rule_evaluator = (
        BusinessRuleEvaluator(
            request.business_rules,
            rules_version=request.business_rules_version,
            rules_config_hash=request.business_rules_config_hash,
        )
        if request.business_rules
        else None
    )

    row_count = 0
    for row in rows:
        row_count += 1
        target_raw = row.get(target_column, "") if target_column else ""
        segment_raw = row.get(segment_column, "") if segment_column else ""
        target_value = target_raw if target_raw != "" else None
        segment_value = segment_raw if segment_raw != "" else None
        row_object_id = row.get(id_column, "") if id_column else ""

        if target_column is not None:
            if target_raw == "":
                target_missing_count += 1
            else:
                target_class_counts[target_raw] = (
                    target_class_counts.get(target_raw, 0) + 1
                )

        # Duplicate signature: tuple of values for non-id columns,
        # ordered by the canonical ``signature_columns`` tuple so the
        # signature is independent of row column ordering.
        if signature_columns:
            signature = tuple(row.get(col, "") for col in signature_columns)
            signature_to_object_ids.setdefault(signature, []).append(row_object_id)

        for column in columns:
            value = row.get(column, "")
            aggregators[column].observe(value)
            is_missing = value == ""
            if target_column is not None and target_value is not None:
                bucket = by_target_counters[column].setdefault(
                    target_value, _MissingTotal()
                )
                bucket.total += 1
                if is_missing:
                    bucket.missing += 1
            if segment_column is not None and segment_value is not None:
                bucket = by_segment_counters[column].setdefault(
                    segment_value, _MissingTotal()
                )
                bucket.total += 1
                if is_missing:
                    bucket.missing += 1

            if column in numeric_samples and value != "":
                numeric_value = _try_parse_float(value)
                if numeric_value is not None:
                    numeric_samples[column].append((numeric_value, row_object_id))

        # Leakage candidate: string equality with target value.
        if target_column is not None and target_value is not None:
            for candidate in leakage_candidate_columns:
                value = row.get(candidate, "")
                if value == "":
                    continue
                counter = leakage_match_counters[candidate]
                counter.total += 1
                if value == target_value:
                    counter.missing += 1  # reuse "missing" slot for matches

        if rule_evaluator is not None:
            rule_evaluator.observe_row(row, row_object_id or None)

    column_profiles: list[ColumnProfile] = []
    group_keys: list[str] = []
    id_columns: list[str] = []
    pii_columns: list[str] = []
    column_missingness: list[ColumnMissingness] = []
    target_chosen = False

    for column in columns:
        agg = aggregators[column]
        is_pii_like = _column_name_looks_pii(column)
        role = _detect_role(
            column,
            is_pii_like=is_pii_like,
            target_already_chosen=target_chosen,
            target_column=target_column,
        )
        if role is ColumnRole.TARGET:
            target_chosen = True
        if role is ColumnRole.GROUP_KEY:
            group_keys.append(column)
        if role is ColumnRole.ID:
            id_columns.append(column)
        if role is ColumnRole.PII_LIKE:
            pii_columns.append(column)

        sample_value = None if is_pii_like else agg.first_non_null
        column_profiles.append(
            ColumnProfile(
                name=column,
                type=agg.infer_type(),
                role=role,
                nullable=agg.null_count > 0,
                null_count=agg.null_count,
                null_ratio=(agg.null_count / row_count) if row_count else 0.0,
                distinct_count=len(agg.distinct_values),
                sample_value=sample_value,
                is_constant=row_count > 0 and len(agg.distinct_values) <= 1,
            )
        )

        column_missingness.append(
            ColumnMissingness(
                column=column,
                missing_count=agg.null_count,
                total_count=row_count,
                missing_rate=(agg.null_count / row_count) if row_count else 0.0,
                by_target=_build_group_block(
                    group_column=target_column,
                    counters=by_target_counters[column],
                ),
                by_segment=_build_group_block(
                    group_column=segment_column,
                    counters=by_segment_counters[column],
                ),
            )
        )

    missingness = MissingnessDiagnostics(
        target_column=target_column,
        target_column_missing=(target_missing_count > 0) if target_column else False,
        missing_target_count=target_missing_count,
        columns=tuple(column_missingness),
        segment_column=segment_column,
    )

    duplicates = _build_duplicate_diagnostics(
        signature_to_object_ids=signature_to_object_ids,
        signature_columns=signature_columns,
        id_column=id_column,
    )
    outliers = _build_outlier_diagnostics(numeric_samples)
    class_imbalance = _build_class_imbalance_diagnostics(
        target_column=target_column,
        target_class_counts=target_class_counts,
    )
    leakage = _build_leakage_diagnostics(
        candidate_columns=leakage_candidate_columns,
        match_counters=leakage_match_counters,
    )
    business_rules_report: BusinessRulesReport | None = (
        rule_evaluator.build_report() if rule_evaluator is not None else None
    )

    return TabularProfileReport(
        profile_id=profile_id or f"tabular_profile_{uuid.uuid4().hex[:16]}",
        profile_schema_version=PROFILE_REPORT_SCHEMA_VERSION,
        source_system=request.source_system,
        row_count=row_count,
        column_count=len(columns),
        columns=tuple(column_profiles),
        target_column=target_column,
        group_key_columns=tuple(group_keys),
        id_columns=tuple(id_columns),
        pii_like_columns=tuple(pii_columns),
        missingness=missingness,
        duplicates=duplicates,
        outliers=outliers,
        class_imbalance=class_imbalance,
        leakage=leakage,
        business_rules=business_rules_report,
        lineage=TabularProfileLineage(
            dataset_id=request.dataset_id,
            version_id=request.version_id,
            parent_version_id=request.parent_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_manifest_artifact=source_manifest_artifact,
            source_artifact_id=request.source_artifact_id,
        ),
        generated_at=generated_at or datetime.now(UTC),
    )


def build_tabular_profile_report(
    archive_reader: ArchiveReader,
    *,
    request: ProfileBuildRequest,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
    source_manifest_artifact: ArtifactRef,
    profile_id: str | None = None,
    generated_at: datetime | None = None,
) -> BuildProfileResult:
    """Build the tabular profile and persist it as an immutable artifact.

    ``archive_reader`` must produce at least one ``transactions.csv``
    descriptor (matching the demo archive contract). The profile is built
    by streaming rows through the archive descriptor; rows are not stored
    in memory beyond the per-column aggregators.

    The persisted artifact is JSON, content-addressed under
    ``artifact_kind=tabular_profile_report``, ``schema_version=tabular_profile_report.v1``.
    Its :class:`ArtifactRef` is contract-compatible with
    ``DataForgeReport.detail_artifacts`` slots.
    """
    descriptor = _resolve_transactions_descriptor(archive_reader)
    with descriptor.open() as handle:
        text_stream = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        reader = csv.DictReader(text_stream)
        fieldnames = reader.fieldnames or ()
        columns = tuple(name for name in fieldnames if name)
        rows_iter = (
            {key: ("" if value is None else value) for key, value in row.items()}
            for row in reader
        )
        report = infer_tabular_profile(
            rows_iter,
            columns=columns,
            request=request,
            source_manifest_artifact=source_manifest_artifact,
            profile_id=profile_id,
            generated_at=generated_at,
        )
        text_stream.close()
    payload = _serialize_report(report)
    artifact = registry.save_artifact(
        artifact_kind=PROFILE_REPORT_KIND,
        data=payload,
        artifact_format=PROFILE_REPORT_FORMAT,
        media_type=PROFILE_REPORT_MEDIA_TYPE,
        schema_version=PROFILE_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "row_count": str(report.row_count),
            "column_count": str(report.column_count),
            "source_system": report.source_system,
        },
    )
    # storage parameter is required by tests/orchestrator wiring to keep the
    # signature symmetrical with build_validated_manifest; the concrete put
    # is delegated to the registry via the same scoped adapter.
    _ = storage
    return BuildProfileResult(profile_report=report, artifact=artifact)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_transactions_descriptor(reader: ArchiveReader) -> ArchiveFileDescriptor:
    descriptors = [
        descriptor
        for descriptor in reader.descriptors()
        if descriptor.kind is ArchiveEntryKind.TRANSACTIONS
    ]
    if not descriptors:
        raise ValueError(
            "tabular profile requires a transactions.csv entry in the archive"
        )
    if len(descriptors) > 1:
        raise ValueError(
            "tabular profile expects exactly one transactions.csv entry"
        )
    return descriptors[0]


def _detect_role(
    column: str,
    *,
    is_pii_like: bool,
    target_already_chosen: bool,
    target_column: str | None,
) -> ColumnRole:
    if is_pii_like:
        return ColumnRole.PII_LIKE
    if any(
        pattern.search(column) is not None for pattern in _LEAKAGE_NAME_PATTERNS
    ):
        return ColumnRole.LEAKAGE_CANDIDATE
    if column in _ID_COLUMN_NAMES:
        return ColumnRole.ID
    if column in _GROUP_KEY_NAMES:
        return ColumnRole.GROUP_KEY
    if column == target_column and not target_already_chosen:
        return ColumnRole.TARGET
    return ColumnRole.UNKNOWN


def _select_target_column(columns: tuple[str, ...]) -> str | None:
    for candidate in _TARGET_COLUMN_NAMES:
        if candidate in columns:
            return candidate
    return None


def _select_segment_column(
    columns: tuple[str, ...],
    requested: str | None,
) -> str | None:
    if requested is None or requested == "":
        return None
    return requested if requested in columns else None


def _build_group_block(
    *,
    group_column: str | None,
    counters: dict[str, _MissingTotal],
) -> MissingnessByGroup | None:
    if group_column is None or not counters:
        return None
    groups = {
        value: MissingnessGroupStats(
            missing_count=stats.missing,
            total_count=stats.total,
            missing_ratio=(stats.missing / stats.total) if stats.total else 0.0,
        )
        for value, stats in counters.items()
    }
    return MissingnessByGroup(group_column=group_column, groups=groups)


@dataclass
class _MissingTotal:
    """Mutable missing/total counter for one group bucket."""

    missing: int = 0
    total: int = 0


def _resolve_outlier_columns(
    columns: tuple[str, ...],
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested is None:
        return tuple(c for c in ("amount", "transaction_amount", "monthly_income") if c in columns)
    return tuple(c for c in requested if c in columns)


def _try_parse_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _build_duplicate_diagnostics(
    *,
    signature_to_object_ids: dict[tuple[str, ...], list[str]],
    signature_columns: tuple[str, ...],
    id_column: str | None,
) -> DuplicateDiagnostics:
    duplicate_pair_count = 0
    duplicate_group_count = 0
    affected: list[str] = []
    for object_ids in signature_to_object_ids.values():
        if len(object_ids) < 2:
            continue
        duplicate_group_count += 1
        # Number of duplicate "pairs" within a group is len-1 (the canonical
        # row plus len-1 duplicates of it).
        duplicate_pair_count += len(object_ids) - 1
        for oid in object_ids:
            if oid:
                affected.append(oid)
    affected_unique = tuple(sorted(set(affected)))
    if len(affected_unique) > _DUPLICATE_AFFECTED_ID_LIMIT:
        affected_unique = affected_unique[:_DUPLICATE_AFFECTED_ID_LIMIT]
    return DuplicateDiagnostics(
        duplicate_pair_count=duplicate_pair_count,
        duplicate_group_count=duplicate_group_count,
        affected_object_ids=affected_unique,
        signature_columns=signature_columns,
        id_column=id_column,
    )


def _build_outlier_diagnostics(
    numeric_samples: dict[str, list[tuple[float, str]]],
) -> OutlierDiagnostics | None:
    if not numeric_samples:
        return None
    columns: list[ColumnOutlierStats] = []
    for column, samples in numeric_samples.items():
        if not samples:
            columns.append(
                ColumnOutlierStats(
                    column=column,
                    outlier_count=0,
                    total_count=0,
                    outlier_rate=0.0,
                )
            )
            continue
        values = sorted(value for value, _ in samples)
        q1 = _quantile(values, 0.25)
        q3 = _quantile(values, 0.75)
        iqr = q3 - q1
        lower = q1 - _OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + _OUTLIER_IQR_MULTIPLIER * iqr
        affected: list[str] = []
        outlier_count = 0
        for value, oid in samples:
            if value < lower or value > upper:
                outlier_count += 1
                if oid:
                    affected.append(oid)
        affected_unique = tuple(sorted(set(affected))[:_OUTLIER_AFFECTED_ID_LIMIT])
        total = len(samples)
        columns.append(
            ColumnOutlierStats(
                column=column,
                outlier_count=outlier_count,
                total_count=total,
                outlier_rate=outlier_count / total if total else 0.0,
                q1=q1,
                q3=q3,
                iqr_lower_bound=lower,
                iqr_upper_bound=upper,
                affected_object_ids=affected_unique,
            )
        )
    return OutlierDiagnostics(method=_OUTLIER_METHOD, columns=tuple(columns))


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


def _build_class_imbalance_diagnostics(
    *,
    target_column: str | None,
    target_class_counts: dict[str, int],
) -> ClassImbalanceDiagnostics | None:
    if target_column is None or not target_class_counts:
        return None
    total = sum(target_class_counts.values())
    if total == 0:
        return None
    sorted_counts = sorted(target_class_counts.items(), key=lambda kv: (kv[1], kv[0]))
    rare_label, rare_count = sorted_counts[0]
    max_count = max(target_class_counts.values())
    minority_share = rare_count / total
    imbalance_ratio = (max_count / rare_count) if rare_count > 0 else float(max_count)
    # 1 / log(1 + imbalance_ratio) saturates at 1 when imbalance is exactly 1
    # (perfectly balanced); clamp to [0, 1].
    log_term = math.log1p(imbalance_ratio)
    balance_score = min(1.0, 1.0 / log_term) if log_term > 0 else 0.0
    mean_count = total / len(target_class_counts)
    balance_score_alternative = (
        min(1.0, rare_count / mean_count) if mean_count > 0 else 0.0
    )
    beta = _CLASS_IMBALANCE_BETA
    effective = {
        label: (1.0 - (beta ** count)) / (1.0 - beta)
        for label, count in target_class_counts.items()
    }
    return ClassImbalanceDiagnostics(
        target_column=target_column,
        total_samples=total,
        class_counts=tuple(
            ClassCount(label=label, count=count)
            for label, count in sorted(target_class_counts.items())
        ),
        rare_class_label=rare_label,
        rare_class_count=rare_count,
        rare_class_ratio=minority_share,
        minority_class_label=rare_label,
        minority_class_share=minority_share,
        imbalance_ratio=imbalance_ratio if imbalance_ratio >= 1.0 else 1.0,
        balance_score=balance_score,
        balance_score_alternative=balance_score_alternative,
        effective_number_beta=beta,
        effective_number_of_samples=effective,
    )


def _build_leakage_diagnostics(
    *,
    candidate_columns: tuple[str, ...],
    match_counters: dict[str, _MissingTotal],
) -> LeakageDiagnostics:
    candidates: list[LeakageCandidate] = []
    for column in candidate_columns:
        counter = match_counters.get(column)
        match_rate: float | None = None
        if counter is not None and counter.total > 0:
            match_rate = counter.missing / counter.total
        # Flag the column either when its name matches a known leakage
        # pattern (already filtered into candidate_columns) or when the
        # observed target match rate is high. Both signals carry the
        # same stable reason_code; ``target_match_rate`` is reported when
        # it can be computed, otherwise it stays ``None``.
        reason_code = "name_pattern_leakage_candidate"
        if match_rate is not None and match_rate >= _LEAKAGE_TARGET_MATCH_THRESHOLD:
            reason_code = "high_target_match_rate"
        candidates.append(
            LeakageCandidate(
                column=column,
                reason_code=reason_code,
                target_match_rate=match_rate,
            )
        )
    return LeakageDiagnostics(candidates=tuple(candidates))


def _column_name_looks_pii(column: str) -> bool:
    return any(pattern.search(column) is not None for pattern in _PII_NAME_PATTERNS)


def _serialize_report(report: TabularProfileReport) -> bytes:
    payload = report.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# column aggregator
# ---------------------------------------------------------------------------


@dataclass
class _ColumnAggregator:
    """Streaming aggregator for one CSV column."""

    name: str
    null_count: int = 0
    distinct_values: set[str] = None  # type: ignore[assignment]
    first_non_null: str | None = None
    saw_int: bool = False
    saw_float: bool = False
    saw_bool: bool = False
    saw_datetime: bool = False
    saw_text_only: bool = False
    saw_identifier_only: bool = True
    total_non_null: int = 0

    def __post_init__(self) -> None:
        self.distinct_values = set()

    def observe(self, value: str) -> None:
        if value == "" or value is None:
            self.null_count += 1
            return
        self.total_non_null += 1
        self.distinct_values.add(value)
        if self.first_non_null is None:
            self.first_non_null = value
        token = value.strip()
        is_bool = token.lower() in {"true", "false", "yes", "no"}
        is_int = _is_integer(token)
        is_float = _is_float(token) and not is_int
        is_datetime = _is_datetime(token)
        if is_bool:
            self.saw_bool = True
        if is_int:
            self.saw_int = True
        if is_float:
            self.saw_float = True
        if is_datetime:
            self.saw_datetime = True
        if not (is_bool or is_int or is_float or is_datetime):
            self.saw_text_only = True
        if not _looks_like_identifier(token):
            self.saw_identifier_only = False

    def infer_type(self) -> ColumnType:
        if self.total_non_null == 0:
            return ColumnType.UNKNOWN
        # 0/1 numeric integers across the column count as boolean if the
        # set of distinct values is exactly {0, 1}.
        if self.saw_bool and not (self.saw_text_only or self.saw_float or self.saw_datetime):
            return ColumnType.BOOLEAN
        if (
            self.saw_int
            and not self.saw_text_only
            and not self.saw_float
            and not self.saw_datetime
        ):
            if self.distinct_values <= {"0", "1"} and len(self.distinct_values) <= 2:
                return ColumnType.BOOLEAN
            return ColumnType.NUMERIC_INTEGER
        if (self.saw_float or self.saw_int) and not self.saw_text_only and not self.saw_datetime:
            return ColumnType.NUMERIC_FLOAT
        if self.saw_datetime and not (self.saw_text_only or self.saw_int or self.saw_float):
            return ColumnType.DATETIME
        if self.saw_identifier_only and self.total_non_null > 0:
            distinct_ratio = len(self.distinct_values) / self.total_non_null
            if distinct_ratio >= 0.9:
                return ColumnType.IDENTIFIER
        if self.saw_text_only:
            distinct_ratio = (
                len(self.distinct_values) / self.total_non_null if self.total_non_null else 0.0
            )
            if distinct_ratio < 0.5 and len(self.distinct_values) <= 50:
                return ColumnType.CATEGORICAL
            return ColumnType.TEXT
        return ColumnType.UNKNOWN


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    if value.lower() in {"inf", "-inf", "nan"}:
        return False
    return True


def _is_datetime(value: str) -> bool:
    for fmt in _DATETIME_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:\-]*$")


def _looks_like_identifier(value: str) -> bool:
    return _IDENTIFIER_PATTERN.match(value) is not None


__all__ = [
    "BuildProfileResult",
    "PROFILE_REPORT_FORMAT",
    "PROFILE_REPORT_KIND",
    "PROFILE_REPORT_MEDIA_TYPE",
    "PROFILE_REPORT_SCHEMA_VERSION",
    "ProfileBuildRequest",
    "build_tabular_profile_report",
    "infer_tabular_profile",
]
