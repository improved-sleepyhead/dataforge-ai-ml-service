"""Supervised tabular split creation for approved ActionPlan steps."""

from __future__ import annotations

import csv
import io
import json
import random
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.adapters import ArtifactRegistry, RegisteredArtifact
from app.adapters.object_storage import MinioObjectStorageAdapter
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    DataSplit,
    ErrorCode,
    SplitAssignment,
    SplitClassDistribution,
    SplitManifest,
    SplitManifestLineage,
    SplitStrategy,
)
from app.domain.common import NonEmptyStr, Sha256Digest

SPLIT_MANIFEST_KIND = "split_manifest"
SPLIT_MANIFEST_FORMAT = "json"
SPLIT_MANIFEST_MEDIA_TYPE = "application/json"
SPLIT_MANIFEST_SCHEMA_VERSION = "split_manifest.v1"
DEFAULT_SPLIT_POLICY_VERSION = "split_policy_v0"
DEFAULT_SPLIT_RATIOS: dict[DataSplit, float] = {
    DataSplit.TRAIN: 0.70,
    DataSplit.VALIDATION: 0.15,
    DataSplit.TEST: 0.15,
}
_SUPPORTED_STEP_TYPES = {"CREATE_SPLIT", "CREATE_SUPERVISED_SPLIT"}
_DEFAULT_GROUP_KEY_CANDIDATES = ("customer_id_hash", "case_id")
_SPLIT_ORDER = (DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST)


class SplitCreationError(ValueError):
    """Raised when split creation cannot safely produce a manifest."""

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


class ExecuteTabularSplitRequest(BaseModel):
    """Inputs for executing one approved supervised tabular split step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_plan_id: NonEmptyStr
    step: ActionPlanStep
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    candidate_dataset_version_id: NonEmptyStr
    source_artifact: ArtifactRef
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    target_column: NonEmptyStr = "is_fraud"
    seed: int = 42
    split_ratios: dict[DataSplit, float] = Field(default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS))
    generated_at: datetime | None = None
    split_manifest_id: str | None = None

    @model_validator(mode="after")
    def validate_split_ratios(self) -> ExecuteTabularSplitRequest:
        splits = set(self.split_ratios)
        if splits != set(_SPLIT_ORDER):
            raise ValueError("split_ratios must define train, validation and test")
        total = sum(self.split_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0")
        if any(value <= 0.0 for value in self.split_ratios.values()):
            raise ValueError("split ratios must be positive")
        return self


@dataclass(frozen=True)
class ExecuteTabularSplitResult:
    """Persisted split manifest and parsed contract."""

    manifest: SplitManifest
    split_artifact: RegisteredArtifact


@dataclass(frozen=True)
class _TabularRow:
    object_id: str
    label: str
    group_value: str | None


@dataclass(frozen=True)
class _Group:
    group_value: str | None
    rows: tuple[_TabularRow, ...]
    label_counts: Counter[str]

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def primary_label(self) -> str:
        if "1" in self.label_counts:
            return "1"
        return sorted(self.label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def execute_tabular_split_action(
    request: ExecuteTabularSplitRequest,
    *,
    storage: MinioObjectStorageAdapter,
    registry: ArtifactRegistry,
) -> ExecuteTabularSplitResult:
    """Create and persist a group-stratified split manifest.

    The executor does not rewrite the source CSV. It emits a standalone split
    manifest artifact that downstream augmentation/model-impact steps consume.
    """
    _validate_step(request.step)
    source = storage.get(request.source_artifact.uri)
    rows, columns = _read_rows(source.data, target_column=request.target_column)
    group_key = _resolve_group_key(request.step.config, columns)
    tabular_rows = tuple(
        _to_tabular_rows(
            rows,
            target_column=request.target_column,
            group_key=group_key,
        )
    )
    assignments = _assign_splits(
        tabular_rows,
        seed=request.seed,
        split_ratios=request.split_ratios,
    )
    manifest = _build_manifest(
        request=request,
        assignments=assignments,
        group_key=group_key,
    )
    artifact = registry.save_artifact(
        artifact_kind=SPLIT_MANIFEST_KIND,
        data=_serialize_manifest(manifest),
        artifact_format=SPLIT_MANIFEST_FORMAT,
        media_type=SPLIT_MANIFEST_MEDIA_TYPE,
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        dataset_version_id=request.candidate_dataset_version_id,
        created_by_job_id=request.created_by_job_id,
        config_hash=request.config_hash,
        metadata={
            "action-plan-id": request.action_plan_id,
            "step-id": request.step.step_id,
            "source-dataset-version-id": request.source_dataset_version_id,
            "source-artifact-hash": request.source_artifact.hash,
            "target-column": request.target_column,
            "strategy": SplitStrategy.GROUP_STRATIFIED.value,
            "seed": str(request.seed),
            "group-key": group_key or "",
            "policy-version": request.step.policy_version,
            "assignment-count": str(len(assignments)),
            **(
                {"created-at": request.generated_at.isoformat()}
                if request.generated_at is not None
                else {}
            ),
        },
    )
    return ExecuteTabularSplitResult(manifest=manifest, split_artifact=artifact)


def _validate_step(step: ActionPlanStep) -> None:
    if step.type not in _SUPPORTED_STEP_TYPES:
        raise SplitCreationError(
            reason_code="unsupported_action_step_type",
            message="Only supervised split creation steps can be executed by the tabular splitter.",
            details={"step_id": step.step_id, "step_type": step.type},
        )
    strategy = step.config.get("strategy", SplitStrategy.GROUP_STRATIFIED.value)
    if strategy != SplitStrategy.GROUP_STRATIFIED.value:
        raise SplitCreationError(
            reason_code="unsupported_split_strategy",
            message="MVP split creation supports group_stratified only.",
            details={"step_id": step.step_id, "strategy": str(strategy)},
        )


def _read_rows(data: bytes, *, target_column: str) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    columns = tuple(name for name in (reader.fieldnames or ()) if name)
    if not columns:
        raise SplitCreationError(
            reason_code="empty_tabular_source",
            message="Source CSV has no header columns.",
        )
    if target_column not in columns:
        raise SplitCreationError(
            reason_code="target_column_not_found",
            message="Split target column is not present in source CSV.",
            details={"target_column": target_column},
        )
    rows = [
        {column: "" if row.get(column) is None else str(row.get(column)) for column in columns}
        for row in reader
    ]
    if not rows:
        raise SplitCreationError(
            reason_code="empty_tabular_source",
            message="Source CSV has no data rows.",
        )
    return rows, columns


def _resolve_group_key(config: Mapping[str, Any], columns: tuple[str, ...]) -> str | None:
    configured = config.get("group_key")
    if configured is not None:
        if not isinstance(configured, str) or configured not in columns:
            raise SplitCreationError(
                reason_code="group_key_not_found",
                message="Configured group_key is not present in source CSV.",
                details={"group_key": configured if isinstance(configured, str) else None},
            )
        return configured
    for candidate in _DEFAULT_GROUP_KEY_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _to_tabular_rows(
    rows: list[dict[str, str]],
    *,
    target_column: str,
    group_key: str | None,
) -> Iterable[_TabularRow]:
    for index, row in enumerate(rows):
        object_id = row.get("object_id") or f"row_{index:06d}"
        label = row[target_column]
        if label == "":
            raise SplitCreationError(
                reason_code="missing_target_label",
                message="Supervised split creation requires non-empty target labels.",
                details={"object_id": object_id},
            )
        group_value = row[group_key] if group_key is not None and row.get(group_key) else None
        yield _TabularRow(object_id=object_id, label=label, group_value=group_value)


def _assign_splits(
    rows: tuple[_TabularRow, ...],
    *,
    seed: int,
    split_ratios: Mapping[DataSplit, float],
) -> tuple[SplitAssignment, ...]:
    groups = _groups(rows)
    total_count = len(rows)
    total_label_counts = Counter(row.label for row in rows)
    target_counts = {
        split: max(1, round(total_count * ratio))
        for split, ratio in split_ratios.items()
    }
    _rebalance_targets(target_counts, total_count=total_count)
    target_label_counts = {
        split: {label: total * split_ratios[split] for label, total in total_label_counts.items()}
        for split in _SPLIT_ORDER
    }

    assignments_by_group: dict[str | None, DataSplit] = {}
    split_counts = {split: 0 for split in _SPLIT_ORDER}
    split_label_counts = {split: Counter[str]() for split in _SPLIT_ORDER}
    rng = random.Random(seed)
    ordered_groups = sorted(
        groups,
        key=lambda group: (group.primary_label != "1", group.group_value or ""),
    )
    rng.shuffle(ordered_groups)
    ordered_groups.sort(key=lambda group: (group.primary_label != "1", -group.size))

    for group in ordered_groups:
        split = _best_split(
            group=group,
            target_counts=target_counts,
            target_label_counts=target_label_counts,
            split_counts=split_counts,
            split_label_counts=split_label_counts,
        )
        assignments_by_group[group.group_value] = split
        split_counts[split] += group.size
        split_label_counts[split].update(group.label_counts)

    return tuple(
        SplitAssignment(
            object_id=row.object_id,
            split=assignments_by_group[row.group_value],
            label=row.label,
            group_value=row.group_value,
        )
        for row in rows
    )


def _groups(rows: tuple[_TabularRow, ...]) -> list[_Group]:
    grouped: dict[str | None, list[_TabularRow]] = {}
    for index, row in enumerate(rows):
        key = row.group_value if row.group_value is not None else f"__row_{index:06d}"
        grouped.setdefault(key, []).append(row)
    result: list[_Group] = []
    for stored_key, group_rows in grouped.items():
        result.append(
            _Group(
                group_value=None if str(stored_key).startswith("__row_") else stored_key,
                rows=tuple(group_rows),
                label_counts=Counter(row.label for row in group_rows),
            )
        )
    return result


def _rebalance_targets(target_counts: dict[DataSplit, int], *, total_count: int) -> None:
    while sum(target_counts.values()) > total_count:
        split = max(_SPLIT_ORDER, key=lambda item: target_counts[item])
        target_counts[split] -= 1
    while sum(target_counts.values()) < total_count:
        split = min(_SPLIT_ORDER, key=lambda item: target_counts[item])
        target_counts[split] += 1


def _best_split(
    *,
    group: _Group,
    target_counts: Mapping[DataSplit, int],
    target_label_counts: Mapping[DataSplit, Mapping[str, float]],
    split_counts: Mapping[DataSplit, int],
    split_label_counts: Mapping[DataSplit, Counter[str]],
) -> DataSplit:
    def score(split: DataSplit) -> tuple[float, int]:
        next_count = split_counts[split] + group.size
        size_penalty = abs(target_counts[split] - next_count) / max(1, target_counts[split])
        label_penalty = 0.0
        for label, count in group.label_counts.items():
            expected = target_label_counts[split].get(label, 0.0)
            next_label_count = split_label_counts[split][label] + count
            label_penalty += abs(expected - next_label_count) / max(1.0, expected)
        return size_penalty + label_penalty, _SPLIT_ORDER.index(split)

    return min(_SPLIT_ORDER, key=score)


def _build_manifest(
    *,
    request: ExecuteTabularSplitRequest,
    assignments: tuple[SplitAssignment, ...],
    group_key: str | None,
) -> SplitManifest:
    return SplitManifest(
        split_manifest_id=request.split_manifest_id or f"split_manifest_{uuid.uuid4().hex[:16]}",
        split_schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        dataset_id=request.dataset_id,
        source_dataset_version_id=request.source_dataset_version_id,
        candidate_dataset_version_id=request.candidate_dataset_version_id,
        target_column=request.target_column,
        strategy=SplitStrategy.GROUP_STRATIFIED,
        seed=request.seed,
        group_key=group_key,
        policy_version=request.step.policy_version or DEFAULT_SPLIT_POLICY_VERSION,
        split_ratios=dict(request.split_ratios),
        assignments=assignments,
        class_distribution=_class_distribution(assignments),
        lineage=SplitManifestLineage(
            action_plan_id=request.action_plan_id,
            step_id=request.step.step_id,
            source_dataset_version_id=request.source_dataset_version_id,
            candidate_dataset_version_id=request.candidate_dataset_version_id,
            created_by_job_id=request.created_by_job_id,
            config_hash=request.config_hash,
            source_artifact=request.source_artifact,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )


def _class_distribution(
    assignments: tuple[SplitAssignment, ...],
) -> tuple[SplitClassDistribution, ...]:
    result: list[SplitClassDistribution] = []
    for split in _SPLIT_ORDER:
        labels = Counter(
            assignment.label for assignment in assignments if assignment.split is split
        )
        total = sum(labels.values())
        result.append(
            SplitClassDistribution(
                split=split,
                total_count=total,
                class_counts=dict(sorted(labels.items())),
                class_ratios={
                    label: count / total if total else 0.0
                    for label, count in sorted(labels.items())
                },
            )
        )
    return tuple(result)


def _serialize_manifest(manifest: SplitManifest) -> bytes:
    return json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2).encode("utf-8")


__all__ = [
    "DEFAULT_SPLIT_POLICY_VERSION",
    "DEFAULT_SPLIT_RATIOS",
    "ExecuteTabularSplitRequest",
    "ExecuteTabularSplitResult",
    "SPLIT_MANIFEST_KIND",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SplitCreationError",
    "execute_tabular_split_action",
]
