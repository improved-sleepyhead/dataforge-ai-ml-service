"""Safe tabular imputation executor for approved ActionPlan steps."""

from __future__ import annotations

import csv
import io
import json
import math
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    ErrorCode,
    ImputationColumnReport,
    ImputationMethod,
    TabularImputationLineage,
    TabularImputationReport,
)
from app.domain.common import NonEmptyStr, Sha256Digest

IMPUTATION_REPORT_KIND = "imputation_report"
IMPUTATION_REPORT_FORMAT = "json"
IMPUTATION_REPORT_MEDIA_TYPE = "application/json"
IMPUTATION_REPORT_SCHEMA_VERSION = "tabular_imputation_report.v1"
CANDIDATE_TABULAR_DATASET_KIND = "candidate_tabular_dataset"
CANDIDATE_TABULAR_DATASET_FORMAT = "csv"
CANDIDATE_TABULAR_DATASET_MEDIA_TYPE = "text/csv"
CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION = "tabular_dataset.v1"
_SUPPORTED_STEP_TYPES = {"IMPUTE_MISSING_VALUES"}


class ImputationExecutionError(ValueError):
    """Raised when an imputation ActionPlan step is unsafe or invalid."""

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


class ExecuteTabularImputationRequest(BaseModel):
    """Inputs for executing one approved tabular imputation step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step: ActionPlanStep
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    target_column: NonEmptyStr
    source_artifact: ArtifactRef
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class ExecuteTabularImputationResult:
    """Artifacts and parsed report produced by imputation execution."""

    report: TabularImputationReport
    candidate_artifact: RegisteredArtifact
    report_artifact: RegisteredArtifact


def execute_tabular_imputation_action(
    request: ExecuteTabularImputationRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> ExecuteTabularImputationResult:
    """Execute one safe tabular imputation step against a source CSV artifact.

    The function never overwrites the source artifact. It writes a candidate
    CSV artifact and a separate immutable before/after imputation report.
    """
    step = request.step
    if step.type not in _SUPPORTED_STEP_TYPES:
        raise ImputationExecutionError(
            reason_code="unsupported_action_step_type",
            message="Only IMPUTE_MISSING_VALUES steps can be executed by the imputation executor.",
            details={"step_id": step.step_id, "step_type": step.type},
        )
    method = _resolve_method(step.method_id)
    column = _required_config_str(step.config, "column")
    if column == request.target_column:
        raise ImputationExecutionError(
            reason_code="target_column_auto_imputation_forbidden",
            message="Target column auto-imputation is blocked by policy.",
            code=ErrorCode.POLICY_BLOCKED,
            details={"step_id": step.step_id, "column": column},
        )

    source = storage.get(request.source_artifact.uri)
    rows, columns = _read_csv(source.data)
    if column not in columns:
        raise ImputationExecutionError(
            reason_code="imputation_column_not_found",
            message="Imputation column is not present in source CSV.",
            details={"step_id": step.step_id, "column": column},
        )
    if request.target_column not in columns:
        raise ImputationExecutionError(
            reason_code="target_column_not_found",
            message="Target column is not present in source CSV.",
            details={"target_column": request.target_column},
        )

    target_before = tuple(row[request.target_column] for row in rows)
    before_missing = _missing_count(rows, column)
    indicator_column = (
        _indicator_column_name(column, columns)
        if _should_add_indicator(step, method)
        else None
    )
    group_key = _group_key(step, columns) if method is ImputationMethod.GROUP_MEDIAN else None
    fill_value, group_fill_values = _fit_fill_values(
        rows=rows,
        column=column,
        method=method,
        group_key=group_key,
    )

    output_columns = list(columns)
    if indicator_column is not None:
        output_columns.append(indicator_column)

    group_imputed_counts: dict[str, int] = {}
    imputed_count = 0
    for row in rows:
        was_missing = row[column] == ""
        if indicator_column is not None:
            row[indicator_column] = "1" if was_missing else "0"
        if was_missing and method is not ImputationMethod.MISSINGNESS_INDICATOR:
            replacement = group_fill_values.get(row[group_key or ""], fill_value)
            row[column] = replacement
            imputed_count += 1
            if group_key is not None:
                group_value = row[group_key]
                group_imputed_counts[group_value] = group_imputed_counts.get(group_value, 0) + 1

    target_after = tuple(row[request.target_column] for row in rows)
    if target_before != target_after:
        raise ImputationExecutionError(
            reason_code="target_column_changed",
            message="Imputation attempted to modify target values.",
            code=ErrorCode.POLICY_BLOCKED,
            details={"target_column": request.target_column},
        )

    after_missing = _missing_count(rows, column)
    candidate_payload = _write_csv(rows, tuple(output_columns))
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
            "step-id": step.step_id,
            "source-artifact-hash": request.source_artifact.hash,
            "source-dataset-version-id": request.source_dataset_version_id,
            "imputation-column": column,
            "imputation-method": method.value,
            **(
                {"created-at": request.generated_at.isoformat()}
                if request.generated_at is not None
                else {}
            ),
        },
    )

    report = TabularImputationReport(
        report_id=request.report_id or f"tabular_imputation_{uuid.uuid4().hex[:16]}",
        report_schema_version=IMPUTATION_REPORT_SCHEMA_VERSION,
        action_plan_id=request.action_plan_id,
        step_id=step.step_id,
        target_column=request.target_column,
        target_unchanged=True,
        before_row_count=len(rows),
        after_row_count=len(rows),
        before_missing_total=before_missing,
        after_missing_total=after_missing,
        columns=(
            ImputationColumnReport(
                column=column,
                method=method,
                indicator_column=indicator_column,
                group_key=group_key,
                before_missing_count=before_missing,
                after_missing_count=after_missing,
                imputed_count=imputed_count,
                total_count=len(rows),
                fill_value=None if method is ImputationMethod.GROUP_MEDIAN else fill_value,
                group_imputed_counts=group_imputed_counts,
            ),
        ),
        candidate_artifact=candidate_artifact.artifact_ref,
        lineage=TabularImputationLineage(
            action_plan_id=request.action_plan_id,
            step_id=step.step_id,
            source_dataset_version_id=request.source_dataset_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_artifact=request.source_artifact,
            candidate_artifact=candidate_artifact.artifact_ref,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )
    report_payload = _serialize_report(report)
    report_artifact = registry.save_artifact(
        artifact_kind=IMPUTATION_REPORT_KIND,
        data=report_payload,
        artifact_format=IMPUTATION_REPORT_FORMAT,
        media_type=IMPUTATION_REPORT_MEDIA_TYPE,
        schema_version=IMPUTATION_REPORT_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": step.step_id,
            "candidate-artifact-hash": candidate_artifact.hash,
            "imputation-column": column,
            "imputation-method": method.value,
            "before-missing-count": str(before_missing),
            "after-missing-count": str(after_missing),
            **(
                {"created-at": request.generated_at.isoformat()}
                if request.generated_at is not None
                else {}
            ),
        },
    )
    return ExecuteTabularImputationResult(
        report=report,
        candidate_artifact=candidate_artifact,
        report_artifact=report_artifact,
    )


def _resolve_method(method_id: str) -> ImputationMethod:
    try:
        return ImputationMethod(method_id)
    except ValueError as exc:
        raise ImputationExecutionError(
            reason_code="unsupported_imputation_method",
            message="Unsupported imputation method for MVP executor.",
            details={"method_id": method_id},
        ) from exc


def _required_config_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or value == "":
        raise ImputationExecutionError(
            reason_code="invalid_imputation_config",
            message="Imputation step config is missing a required string field.",
            details={"field": key},
        )
    return value


def _read_csv(data: bytes) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = tuple(name for name in (reader.fieldnames or ()) if name)
    if not columns:
        raise ImputationExecutionError(
            reason_code="empty_tabular_source",
            message="Source CSV has no header columns.",
        )
    rows = [
        {column: "" if row.get(column) is None else str(row.get(column)) for column in columns}
        for row in reader
    ]
    if not rows:
        raise ImputationExecutionError(
            reason_code="empty_tabular_source",
            message="Source CSV has no data rows.",
        )
    return rows, columns


def _write_csv(rows: list[dict[str, str]], columns: tuple[str, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue().encode("utf-8")


def _missing_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if row[column] == "")


def _should_add_indicator(step: ActionPlanStep, method: ImputationMethod) -> bool:
    configured = step.config.get("add_missingness_indicator")
    return method is ImputationMethod.MISSINGNESS_INDICATOR or configured is True


def _indicator_column_name(column: str, columns: tuple[str, ...]) -> str:
    indicator = f"{column}_was_missing"
    if indicator in columns:
        raise ImputationExecutionError(
            reason_code="indicator_column_already_exists",
            message="Missingness indicator column already exists in source CSV.",
            details={"indicator_column": indicator},
        )
    return indicator


def _group_key(step: ActionPlanStep, columns: tuple[str, ...]) -> str:
    group_key = step.config.get("group_key", "customer_segment")
    if not isinstance(group_key, str) or group_key == "" or group_key not in columns:
        raise ImputationExecutionError(
            reason_code="group_key_not_found",
            message="group_median requires an existing group_key column.",
            details={"group_key": group_key if isinstance(group_key, str) else None},
        )
    return group_key


def _fit_fill_values(
    *,
    rows: list[dict[str, str]],
    column: str,
    method: ImputationMethod,
    group_key: str | None,
) -> tuple[str, dict[str, str]]:
    if method is ImputationMethod.MISSINGNESS_INDICATOR:
        return "", {}
    if method is ImputationMethod.MODE:
        return _mode(row[column] for row in rows if row[column] != ""), {}

    global_median = _median(row[column] for row in rows if row[column] != "")
    if method is ImputationMethod.MEDIAN:
        return global_median, {}
    if group_key is None:
        return global_median, {}
    grouped_values: dict[str, list[str]] = {}
    for row in rows:
        if row[column] == "":
            continue
        grouped_values.setdefault(row[group_key], []).append(row[column])
    group_medians = {
        group: _median(values)
        for group, values in grouped_values.items()
        if values
    }
    return global_median, group_medians


def _median(values: Iterable[str]) -> str:
    parsed = sorted(_parse_decimal(value) for value in values)
    if not parsed:
        raise ImputationExecutionError(
            reason_code="no_non_missing_numeric_values",
            message="Median imputation requires at least one non-missing numeric value.",
        )
    midpoint = len(parsed) // 2
    if len(parsed) % 2 == 1:
        return _format_decimal(parsed[midpoint])
    return _format_decimal((parsed[midpoint - 1] + parsed[midpoint]) / Decimal(2))


def _mode(values: Iterable[str]) -> str:
    counts = Counter(str(value) for value in values)
    if not counts:
        raise ImputationExecutionError(
            reason_code="no_non_missing_values",
            message="Mode imputation requires at least one non-missing value.",
        )
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _parse_decimal(value: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ImputationExecutionError(
            reason_code="non_numeric_imputation_value",
            message="Median imputation can only be fitted on numeric values.",
        ) from exc
    if not result.is_finite():
        raise ImputationExecutionError(
            reason_code="non_numeric_imputation_value",
            message="Median imputation can only be fitted on finite numeric values.",
        )
    return result


def _format_decimal(value: Decimal) -> str:
    as_float = float(value)
    if math.isfinite(as_float) and as_float.is_integer():
        return str(int(as_float))
    normalized = value.normalize()
    return format(normalized, "f")


def _serialize_report(report: TabularImputationReport) -> bytes:
    return json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2).encode("utf-8")


__all__ = [
    "CANDIDATE_TABULAR_DATASET_KIND",
    "CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION",
    "ExecuteTabularImputationRequest",
    "ExecuteTabularImputationResult",
    "IMPUTATION_REPORT_KIND",
    "IMPUTATION_REPORT_SCHEMA_VERSION",
    "ImputationExecutionError",
    "execute_tabular_imputation_action",
]
