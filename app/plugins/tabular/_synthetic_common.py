"""Shared helpers for tabular synthetic generators (SMOTE, Gaussian Copula).

Both MVP synthetic methods share the same persistence shape:

- a candidate tabular CSV with synthetic rows appended and marked
  ``is_synthetic`` / ``synthetic_source_split=train``;
- an augmented split manifest that assigns the synthetic rows to the
  training split;
- a contract-valid ``synthetic_dataset_report`` artifact with shared
  validation, lineage, sample lineage, and class stats blocks.

Keeping these helpers in one module avoids duplication and keeps the
synthetic_validation logic identical across methods, which is important
for Decision Core / export gates that read the report uniformly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ArtifactRef,
    DataSplit,
    ErrorCode,
    SplitAssignment,
    SplitManifest,
    SyntheticDatasetReport,
    SyntheticGenerationMethod,
    SyntheticValidationCheck,
    SyntheticValidationCheckStatus,
    SyntheticValidationReport,
)
from app.plugins.tabular.rules import BusinessRule, BusinessRuleEvaluator
from app.plugins.tabular.splits import (
    SPLIT_MANIFEST_FORMAT,
    SPLIT_MANIFEST_KIND,
    SPLIT_MANIFEST_MEDIA_TYPE,
    SPLIT_MANIFEST_SCHEMA_VERSION,
)

CANDIDATE_TABULAR_DATASET_KIND = "candidate_tabular_dataset"
CANDIDATE_TABULAR_DATASET_FORMAT = "csv"
CANDIDATE_TABULAR_DATASET_MEDIA_TYPE = "text/csv"
CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION = "tabular_dataset.v1"

SYNTHETIC_REPORT_KIND = "synthetic_dataset_report"
SYNTHETIC_REPORT_FORMAT = "json"
SYNTHETIC_REPORT_MEDIA_TYPE = "application/json"
SYNTHETIC_REPORT_SCHEMA_VERSION = "synthetic_dataset_report.v1"

IS_SYNTHETIC_COLUMN = "is_synthetic"
SOURCE_SPLIT_COLUMN = "synthetic_source_split"


@dataclass(frozen=True)
class SyntheticPersistenceResult:
    """Persisted artifacts produced by a synthetic generator."""

    candidate_artifact: RegisteredArtifact
    augmented_split_artifact: RegisteredArtifact
    augmented_split_manifest: SplitManifest
    report_artifact: RegisteredArtifact


# ---------------------------------------------------------------------------
# CSV / row helpers
# ---------------------------------------------------------------------------


def candidate_dataset_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Return the column list with synthetic-marker columns appended."""
    extras: list[str] = []
    if IS_SYNTHETIC_COLUMN not in columns:
        extras.append(IS_SYNTHETIC_COLUMN)
    if SOURCE_SPLIT_COLUMN not in columns:
        extras.append(SOURCE_SPLIT_COLUMN)
    return tuple([*columns, *extras])


def write_candidate_csv(
    *,
    rows: Sequence[Mapping[str, str]],
    synthetic_rows: Sequence[Mapping[str, str]],
    columns: tuple[str, ...],
) -> bytes:
    """Serialise the augmented dataset (real + synthetic) to CSV bytes.

    Real rows receive ``is_synthetic=0`` and an empty source-split
    marker; synthetic rows must already have these markers populated.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        materialized = {column: row.get(column, "") for column in columns}
        if not materialized.get(IS_SYNTHETIC_COLUMN):
            materialized[IS_SYNTHETIC_COLUMN] = "0"
        materialized.setdefault(SOURCE_SPLIT_COLUMN, "")
        writer.writerow(materialized)
    for row in synthetic_rows:
        materialized = {column: row.get(column, "") for column in columns}
        writer.writerow(materialized)
    return buffer.getvalue().encode("utf-8")


def parse_numeric(value: str) -> float | None:
    """Parse a CSV cell into a finite float or return ``None``."""
    if value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def format_numeric(value: float) -> str:
    """Format a float for stable CSV output."""
    if value == int(value):
        return str(int(value))
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


# ---------------------------------------------------------------------------
# augmented split manifest
# ---------------------------------------------------------------------------


def build_augmented_split_manifest(
    *,
    manifest: SplitManifest,
    synthetic_rows: Sequence[Mapping[str, str]],
    target_column: str,
    method: SyntheticGenerationMethod,
) -> SplitManifest:
    """Return a copy of ``manifest`` with synthetic rows assigned to TRAIN.

    The synthetic split-manifest id is suffixed with the method and
    generated count so it is easy to distinguish at audit time.
    """
    if not synthetic_rows:
        return manifest

    new_assignments = list(manifest.assignments)
    label_counts = {item.split: dict(item.class_counts) for item in manifest.class_distribution}
    total_counts = {item.split: item.total_count for item in manifest.class_distribution}
    for row in synthetic_rows:
        label = row.get(target_column, "")
        if not label:
            label = "0"
        new_assignments.append(
            SplitAssignment(
                object_id=row["object_id"],
                split=DataSplit.TRAIN,
                label=label,
                group_value=None,
            )
        )
        train_counts = label_counts.setdefault(DataSplit.TRAIN, {})
        train_counts[label] = train_counts.get(label, 0) + 1
        total_counts[DataSplit.TRAIN] = total_counts.get(DataSplit.TRAIN, 0) + 1

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
                f"{manifest.split_manifest_id}_with_{method.value}_{len(synthetic_rows)}"
            ),
            "assignments": tuple(new_assignments),
            "class_distribution": tuple(new_distribution),
        }
    )


# ---------------------------------------------------------------------------
# synthetic validation gates
# ---------------------------------------------------------------------------


def evaluate_synthetic_validation(
    *,
    real_rows: Sequence[Mapping[str, str]],
    synthetic_rows: Sequence[Mapping[str, str]],
    feature_columns: tuple[str, ...],
    business_rules: Sequence[BusinessRule] = (),
    rules_version: str = "tabular_business_rules.v1",
    rules_config_hash: str | None = None,
) -> SyntheticValidationReport:
    """Run schema/type/business-rule/privacy checks on synthetic output."""
    schema_check = _schema_check(synthetic_rows=synthetic_rows, feature_columns=feature_columns)
    type_check = _type_check(synthetic_rows=synthetic_rows, feature_columns=feature_columns)
    business_check = _business_rules_check(
        synthetic_rows=synthetic_rows,
        rules=business_rules,
        rules_version=rules_version,
        rules_config_hash=rules_config_hash,
    )
    privacy_check = _privacy_exact_duplicate_check(
        real_rows=real_rows,
        synthetic_rows=synthetic_rows,
        feature_columns=feature_columns,
    )
    checks = (schema_check, type_check, business_check, privacy_check)
    blocker_present = any(check.blocker for check in checks)
    overall_passed = not blocker_present and all(
        check.status is not SyntheticValidationCheckStatus.FAILED for check in checks
    )
    return SyntheticValidationReport(
        overall_passed=overall_passed,
        blocker_present=blocker_present,
        checks=checks,
    )


def _schema_check(
    *,
    synthetic_rows: Sequence[Mapping[str, str]],
    feature_columns: tuple[str, ...],
) -> SyntheticValidationCheck:
    if not synthetic_rows:
        return SyntheticValidationCheck(
            check="schema_match",
            status=SyntheticValidationCheckStatus.NOT_APPLICABLE,
            pass_rate=None,
            findings_count=0,
            blocker=False,
            notes="no synthetic rows generated",
        )
    findings = 0
    for row in synthetic_rows:
        for column in feature_columns:
            if column not in row:
                findings += 1
    total = len(synthetic_rows) * max(1, len(feature_columns))
    return SyntheticValidationCheck(
        check="schema_match",
        status=(
            SyntheticValidationCheckStatus.PASSED
            if findings == 0
            else SyntheticValidationCheckStatus.FAILED
        ),
        pass_rate=(total - findings) / total if total else 1.0,
        findings_count=findings,
        blocker=findings > 0,
    )


def _type_check(
    *,
    synthetic_rows: Sequence[Mapping[str, str]],
    feature_columns: tuple[str, ...],
) -> SyntheticValidationCheck:
    if not synthetic_rows:
        return SyntheticValidationCheck(
            check="type_match",
            status=SyntheticValidationCheckStatus.NOT_APPLICABLE,
            pass_rate=None,
            findings_count=0,
            blocker=False,
            notes="no synthetic rows generated",
        )
    findings = 0
    for row in synthetic_rows:
        for column in feature_columns:
            value = row.get(column, "")
            if value == "":
                # Missing numeric values in synthetic output mean the
                # generator could not produce a number for the column.
                # We do not block on this — Decision Core decides — but
                # we do count it.
                findings += 1
                continue
            if parse_numeric(value) is None:
                findings += 1
    total = len(synthetic_rows) * max(1, len(feature_columns))
    return SyntheticValidationCheck(
        check="type_match",
        status=(
            SyntheticValidationCheckStatus.PASSED
            if findings == 0
            else SyntheticValidationCheckStatus.FAILED
        ),
        pass_rate=(total - findings) / total if total else 1.0,
        findings_count=findings,
        blocker=False,
        notes=(
            None
            if findings == 0
            else "non-numeric or missing values detected in synthetic features"
        ),
    )


def _business_rules_check(
    *,
    synthetic_rows: Sequence[Mapping[str, str]],
    rules: Sequence[BusinessRule],
    rules_version: str,
    rules_config_hash: str | None,
) -> SyntheticValidationCheck:
    if not rules:
        return SyntheticValidationCheck(
            check="business_rules_pass_rate",
            status=SyntheticValidationCheckStatus.NOT_APPLICABLE,
            pass_rate=None,
            findings_count=0,
            blocker=False,
            notes="no business rules configured",
        )
    if not synthetic_rows:
        return SyntheticValidationCheck(
            check="business_rules_pass_rate",
            status=SyntheticValidationCheckStatus.NOT_APPLICABLE,
            pass_rate=None,
            findings_count=0,
            blocker=False,
            notes="no synthetic rows generated",
        )
    evaluator = BusinessRuleEvaluator(
        tuple(rules),
        rules_version=rules_version,
        rules_config_hash=rules_config_hash,
    )
    for row in synthetic_rows:
        evaluator.observe_row(row, row.get("object_id"))
    report = evaluator.build_report()
    findings = sum(rule.violation_count for rule in report.rules)
    blocker = bool(report.blocker_candidate_rule_ids)
    pass_rate = (
        sum(rule.pass_rate for rule in report.rules) / len(report.rules)
        if report.rules
        else 1.0
    )
    return SyntheticValidationCheck(
        check="business_rules_pass_rate",
        status=(
            SyntheticValidationCheckStatus.PASSED
            if findings == 0
            else SyntheticValidationCheckStatus.FAILED
        ),
        pass_rate=pass_rate,
        findings_count=findings,
        blocker=blocker,
        notes=(
            None
            if findings == 0
            else f"{findings} synthetic rows violate one or more business rules"
        ),
    )


def _privacy_exact_duplicate_check(
    *,
    real_rows: Sequence[Mapping[str, str]],
    synthetic_rows: Sequence[Mapping[str, str]],
    feature_columns: tuple[str, ...],
) -> SyntheticValidationCheck:
    """Block synthetic rows that are byte-identical (on features) with real rows.

    The privacy rule we encode here is the exact-duplicate-to-real
    check from PRD §11.12: a synthetic row that reproduces a real row's
    feature vector exactly is unsafe and must be flagged. Near-neighbor
    DCR is a follow-up task.
    """
    if not synthetic_rows:
        return SyntheticValidationCheck(
            check="privacy_exact_duplicate_to_real",
            status=SyntheticValidationCheckStatus.NOT_APPLICABLE,
            pass_rate=None,
            findings_count=0,
            blocker=False,
            notes="no synthetic rows generated",
        )
    real_signatures: Counter[str] = Counter(
        _row_signature(row, feature_columns) for row in real_rows
    )
    findings = 0
    for row in synthetic_rows:
        signature = _row_signature(row, feature_columns)
        if real_signatures.get(signature, 0) > 0:
            findings += 1
    total = len(synthetic_rows)
    return SyntheticValidationCheck(
        check="privacy_exact_duplicate_to_real",
        status=(
            SyntheticValidationCheckStatus.PASSED
            if findings == 0
            else SyntheticValidationCheckStatus.FAILED
        ),
        pass_rate=(total - findings) / total if total else 1.0,
        findings_count=findings,
        blocker=findings > 0,
        notes=(
            None
            if findings == 0
            else "synthetic rows reproduce real feature vectors exactly"
        ),
    )


def _row_signature(row: Mapping[str, str], feature_columns: Iterable[str]) -> str:
    payload = {column: row.get(column, "") for column in sorted(feature_columns)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def persist_synthetic_artifacts(
    *,
    method: SyntheticGenerationMethod,
    request_metadata: Mapping[str, str],
    real_rows: Sequence[Mapping[str, str]],
    synthetic_rows: Sequence[Mapping[str, str]],
    columns: tuple[str, ...],
    split_manifest: SplitManifest,
    target_column: str,
    candidate_dataset_version_id: str,
    created_by_job_id: str,
    config_hash: str,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> SyntheticPersistenceResult:
    """Persist candidate CSV, augmented split manifest, and report artifact."""
    _ = storage  # storage is owned by registry; signature kept symmetric.
    candidate_columns = candidate_dataset_columns(columns)
    candidate_payload = write_candidate_csv(
        rows=real_rows,
        synthetic_rows=synthetic_rows,
        columns=candidate_columns,
    )
    candidate_artifact = registry.save_artifact(
        artifact_kind=CANDIDATE_TABULAR_DATASET_KIND,
        data=candidate_payload,
        artifact_format=CANDIDATE_TABULAR_DATASET_FORMAT,
        media_type=CANDIDATE_TABULAR_DATASET_MEDIA_TYPE,
        schema_version=CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION,
        dataset_version_id=candidate_dataset_version_id,
        created_by_job_id=created_by_job_id,
        config_hash=config_hash,
        metadata={
            **dict(request_metadata),
            "synthetic-method": method.value,
            "synthetic-row-count": str(len(synthetic_rows)),
        },
    )

    augmented_manifest = build_augmented_split_manifest(
        manifest=split_manifest,
        synthetic_rows=synthetic_rows,
        target_column=target_column,
        method=method,
    )
    augmented_split_artifact = registry.save_artifact(
        artifact_kind=SPLIT_MANIFEST_KIND,
        data=_serialize_split_manifest(augmented_manifest),
        artifact_format=SPLIT_MANIFEST_FORMAT,
        media_type=SPLIT_MANIFEST_MEDIA_TYPE,
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        dataset_version_id=candidate_dataset_version_id,
        created_by_job_id=created_by_job_id,
        config_hash=config_hash,
        metadata={
            **dict(request_metadata),
            "synthetic-method": method.value,
            "synthetic-row-count": str(len(synthetic_rows)),
            "augments-split-manifest-id": split_manifest.split_manifest_id,
        },
    )

    return SyntheticPersistenceResult(
        candidate_artifact=candidate_artifact,
        augmented_split_artifact=augmented_split_artifact,
        augmented_split_manifest=augmented_manifest,
        report_artifact=candidate_artifact,  # placeholder; replaced below
    )


def persist_synthetic_report(
    *,
    report: SyntheticDatasetReport,
    candidate_dataset_version_id: str,
    created_by_job_id: str,
    config_hash: str,
    base_metadata: Mapping[str, str],
    candidate_artifact_hash: str,
    augmented_split_artifact_hash: str,
    registry: ArtifactRegistry,
) -> RegisteredArtifact:
    """Serialise and persist the synthetic_dataset_report artifact."""
    payload = json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    return registry.save_artifact(
        artifact_kind=SYNTHETIC_REPORT_KIND,
        data=payload,
        artifact_format=SYNTHETIC_REPORT_FORMAT,
        media_type=SYNTHETIC_REPORT_MEDIA_TYPE,
        schema_version=SYNTHETIC_REPORT_SCHEMA_VERSION,
        dataset_version_id=candidate_dataset_version_id,
        created_by_job_id=created_by_job_id,
        config_hash=config_hash,
        metadata={
            **dict(base_metadata),
            "synthetic-method": report.method.value,
            "augmentation-kind": report.augmentation_kind.value,
            "random-seed": str(report.random_seed),
            "k-neighbors": str(report.k_neighbors),
            "sampling-strategy": str(report.sampling_strategy),
            "generated-count": str(report.generated_count),
            "candidate-artifact-hash": candidate_artifact_hash,
            "augmented-split-manifest-hash": augmented_split_artifact_hash,
        },
    )


def index_assignments(
    manifest: SplitManifest,
) -> dict[str, tuple[DataSplit, str | None]]:
    """Index split assignments by ``object_id`` for quick lookup."""
    return {
        assignment.object_id: (assignment.split, assignment.group_value)
        for assignment in manifest.assignments
    }


ErrorFactory = Callable[[str, str, ErrorCode, dict[str, object]], BaseException]


def verify_assignments_match_rows(
    *,
    assignments: Mapping[str, tuple[DataSplit, str | None]],
    rows: Sequence[Mapping[str, str]],
    error_factory: ErrorFactory,
) -> None:
    """Raise via ``error_factory`` if the manifest references missing object_ids."""
    row_ids = {row["object_id"] for row in rows}
    missing = sorted(set(assignments) - row_ids)
    if missing:
        raise error_factory(
            "split_manifest_object_id_missing_in_source",
            "Split manifest references object_ids missing from the source CSV.",
            ErrorCode.ACTION_PLAN_PRECONDITION_FAILED,
            {"missing_object_ids": missing[:5]},
        )


def read_source_rows(
    *,
    storage: MinioObjectStorageAdapter,
    source_artifact: ArtifactRef,
    target_column: str,
    error_factory: ErrorFactory,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    """Read the source CSV referenced by ``source_artifact``."""
    stored = storage.get(source_artifact.uri)
    text = stored.data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = tuple(name for name in (reader.fieldnames or ()) if name)
    if not columns:
        raise error_factory(
            "empty_tabular_source",
            "Source CSV has no header columns.",
            ErrorCode.INVALID_JOB_PAYLOAD,
            {},
        )
    if target_column not in columns:
        raise error_factory(
            "target_column_not_found",
            "Target column is not present in source CSV.",
            ErrorCode.INVALID_JOB_PAYLOAD,
            {"target_column": target_column},
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
        raise error_factory(
            "empty_tabular_source",
            "Source CSV has no data rows.",
            ErrorCode.INVALID_JOB_PAYLOAD,
            {},
        )
    return tuple(rows), columns


def _serialize_split_manifest(manifest: SplitManifest) -> bytes:
    return json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        indent=2,
    ).encode("utf-8")


__all__ = [
    "CANDIDATE_TABULAR_DATASET_FORMAT",
    "CANDIDATE_TABULAR_DATASET_KIND",
    "CANDIDATE_TABULAR_DATASET_MEDIA_TYPE",
    "CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION",
    "IS_SYNTHETIC_COLUMN",
    "SOURCE_SPLIT_COLUMN",
    "SYNTHETIC_REPORT_FORMAT",
    "SYNTHETIC_REPORT_KIND",
    "SYNTHETIC_REPORT_MEDIA_TYPE",
    "SYNTHETIC_REPORT_SCHEMA_VERSION",
    "SyntheticPersistenceResult",
    "build_augmented_split_manifest",
    "candidate_dataset_columns",
    "evaluate_synthetic_validation",
    "format_numeric",
    "index_assignments",
    "parse_numeric",
    "persist_synthetic_artifacts",
    "persist_synthetic_report",
    "read_source_rows",
    "verify_assignments_match_rows",
    "write_candidate_csv",
]
