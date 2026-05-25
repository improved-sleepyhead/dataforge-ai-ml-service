"""Tests for TASK-051 sklearn baseline model-impact runner."""

from __future__ import annotations

import io
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import app.kernel.model_impact as model_impact_module
from app.adapters import ArtifactRegistry, MinioObjectStorageAdapter, ObjectStorageScope
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.domain import (
    ActionPlanStep,
    ArtifactRef,
    DataSplit,
    ErrorCode,
    MetricStatus,
    ModelImpactReport,
    ModelImpactVerdict,
    RetryPolicy,
    SplitAssignment,
    SplitClassDistribution,
    SplitManifest,
    SplitManifestLineage,
    SplitStrategy,
    SyntheticUtilityStatus,
    TstrTrtsMetrics,
)
from app.ingestion import open_archive_path
from app.kernel import (
    RunModelImpactRequest,
    run_model_impact,
)
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    ExecuteSmoteAugmentationRequest,
    ExecuteTabularSplitRequest,
    execute_smote_augmentation_action,
    execute_tabular_split_action,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "a" * 64
_GENERATED_AT = datetime(2026, 5, 27, 14, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step 1+2+3+4: model impact on demo candidate (non-synthetic candidate)
# ---------------------------------------------------------------------------


def test_model_impact_on_demo_candidate_emits_metrics(tmp_path: Path) -> None:
    """Steps 1-4: baseline vs candidate metrics + confusion matrix + reproducibility."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split_action = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    # Use the demo source as the candidate too (non-synthetic) so the
    # baseline comparison runs deterministically. The runner still
    # records baseline-vs-candidate metrics; on identical inputs
    # rare_class_recall_before == rare_class_recall_after.
    request = _request(
        source_artifact=source_artifact,
        candidate_artifact=source_artifact,
        source_split=split_action.manifest,
        candidate_split=split_action.manifest,
        candidate_is_synthetic=False,
    )
    result = run_model_impact(request, storage=storage, registry=registry)
    report = result.report

    # Step 1: report is well-formed and validates against the contract.
    assert isinstance(report, ModelImpactReport)
    assert report.metric_library == "scikit-learn"
    assert report.metric_library_version
    validate_contract_payload(
        load_contract_pack(),
        "model_impact_report",
        report.model_dump(mode="json"),
    )

    # Step 2: rare_class_recall_before/after and macro_f1_before/after
    # are present and consistent.
    assert 0.0 <= report.rare_class_recall_before <= 1.0
    assert 0.0 <= report.rare_class_recall_after <= 1.0
    assert 0.0 <= report.macro_f1_before <= 1.0
    assert 0.0 <= report.macro_f1_after <= 1.0
    assert (
        report.rare_class_recall_after
        == report.candidate_metrics.rare_class_recall
    )
    assert (
        report.macro_f1_after == report.candidate_metrics.macro_f1
    )
    # On identical inputs the deltas must be zero.
    assert report.rare_class_recall_delta == pytest.approx(0.0, abs=1e-9)
    assert report.macro_f1_delta == pytest.approx(0.0, abs=1e-9)
    assert report.weighted_f1_delta == pytest.approx(0.0, abs=1e-9)
    # When metrics are unchanged the verdict is requires_review.
    assert report.verdict is ModelImpactVerdict.REQUIRES_REVIEW
    assert "metrics_unchanged" in report.verdict_reason_codes
    # Synthetic utility is not_applicable for non-synthetic candidates.
    assert report.synthetic_utility_status is SyntheticUtilityStatus.NOT_APPLICABLE

    # Step 4: rare_class_precision and confusion_matrix are populated.
    assert 0.0 <= report.candidate_metrics.rare_class_precision <= 1.0
    assert report.candidate_metrics.confusion_matrix
    assert sum(c.count for c in report.candidate_metrics.confusion_matrix) == (
        report.candidate_metrics.sample_count
    )
    # PR-AUC is available when the rare class is present in test split.
    assert report.pr_auc_status is MetricStatus.AVAILABLE
    assert report.pr_auc_before is not None
    assert report.pr_auc_after is not None

    # Reproducibility: model config carries algorithm, library version
    # and random seed.
    assert report.baseline_model_config.algorithm == "LogisticRegression"
    assert report.baseline_model_config.library == "scikit-learn"
    assert report.baseline_model_config.random_seed == 42

    # Persistence metadata
    stored = storage.get(result.report_artifact.uri)
    assert stored.info.metadata["verdict"] == report.verdict.value
    assert stored.info.metadata["random-seed"] == "42"


def test_model_impact_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    """Step 3: same inputs + same seed -> identical metrics."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split_action = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    request = _request(
        source_artifact=source_artifact,
        candidate_artifact=source_artifact,
        source_split=split_action.manifest,
        candidate_split=split_action.manifest,
        candidate_is_synthetic=False,
        report_id="model_impact_report_fixed_seed",
    )
    first = run_model_impact(request, storage=storage, registry=registry)
    second = run_model_impact(request, storage=storage, registry=registry)
    assert (
        first.report.candidate_metrics.macro_f1
        == second.report.candidate_metrics.macro_f1
    )
    assert (
        first.report.candidate_metrics.rare_class_recall
        == second.report.candidate_metrics.rare_class_recall
    )
    assert (
        first.report.candidate_metrics.rare_class_precision
        == second.report.candidate_metrics.rare_class_precision
    )
    # Confusion matrix counts are also stable.
    assert tuple(
        (c.true_label, c.predicted_label, c.count)
        for c in first.report.candidate_metrics.confusion_matrix
    ) == tuple(
        (c.true_label, c.predicted_label, c.count)
        for c in second.report.candidate_metrics.confusion_matrix
    )


# ---------------------------------------------------------------------------
# Step 5: synthetic SMOTE candidate -> strict TSTR/TRTS semantics
# ---------------------------------------------------------------------------


def test_smote_candidate_emits_tstr_trts_metrics(tmp_path: Path) -> None:
    """Step 5: SMOTE reports strict TSTR as N/A when synthetic train is single-class."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split_action = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    smote = execute_smote_augmentation_action(
        ExecuteSmoteAugmentationRequest(
            action_plan_id="action_plan_smote_001",
            step=_smote_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_v1",
            candidate_dataset_version_id="dataset_version_v2_candidate",
            source_artifact=source_artifact,
            split_manifest=split_action.manifest,
            split_manifest_artifact=split_action.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_001",
            config_hash=_CONFIG_HASH,
            random_seed=42,
            k_neighbors=3,
            sampling_strategy=0.30,
            generated_at=_GENERATED_AT,
        ),
        storage=storage,
        registry=registry,
    )
    augmented_payload = storage.get(smote.augmented_split_artifact.uri).data
    augmented_manifest = SplitManifest.model_validate_json(augmented_payload)
    request = _request(
        source_artifact=source_artifact,
        candidate_artifact=smote.candidate_artifact.artifact_ref,
        source_split=split_action.manifest,
        candidate_split=augmented_manifest,
        candidate_is_synthetic=True,
        synthetic_dataset_report_artifact=smote.report_artifact.artifact_ref,
    )
    result = run_model_impact(request, storage=storage, registry=registry)
    report = result.report

    # Step 5: strict TSTR trains on synthetic rows only. Targeted SMOTE
    # generates rare-class rows, so the synthetic-only training set is
    # single-class and TSTR must be explicit not_applicable instead of
    # silently using real+synthetic candidate train rows.
    assert report.tstr_trts.status is MetricStatus.NOT_APPLICABLE
    assert report.tstr_trts.reason is not None
    assert "tstr_single_class_in_train_split" in report.tstr_trts.reason
    assert report.tstr_trts.tstr_metrics is None
    assert report.tstr_trts.trts_metrics is not None
    assert 0.0 <= report.tstr_trts.trts_metrics.macro_f1 <= 1.0
    assert report.tstr_trts.tstr_macro_f1_drop is None
    assert report.tstr_trts.tstr_macro_f1_threshold == pytest.approx(0.10)
    assert report.tstr_trts.trts_macro_f1_delta is not None
    assert report.tstr_trts.trts_unstable_threshold == pytest.approx(0.20)

    # Synthetic utility verdict reflects metric movement.
    assert report.synthetic_utility_status is SyntheticUtilityStatus.REQUIRES_REVIEW
    assert "tstr_single_class_in_train_split" in " ".join(
        report.synthetic_utility_reason_codes
    )

    # Persistence: contract validates and metadata reflects the
    # synthetic verdict.
    validate_contract_payload(
        load_contract_pack(),
        "model_impact_report",
        report.model_dump(mode="json"),
    )
    stored = storage.get(result.report_artifact.uri)
    assert (
        stored.info.metadata["synthetic-utility-status"]
        == report.synthetic_utility_status.value
    )


def test_model_impact_decisions_use_weighted_f1_and_pr_auc() -> None:
    """Synthetic utility cannot ignore weighted F1 or PR-AUC degradation."""
    verdict, verdict_reasons = model_impact_module._classify_verdict(
        rare_class_recall_delta=0.10,
        macro_f1_delta=0.01,
        weighted_f1_delta=-0.04,
        pr_auc_delta=0.0,
        rare_class_recall_drop_threshold=0.05,
        macro_f1_drop_threshold=0.03,
        weighted_f1_drop_threshold=0.03,
        pr_auc_drop_threshold=0.03,
        validation_gates_blocker_present=False,
    )
    assert verdict is ModelImpactVerdict.DEGRADED
    assert "weighted_f1_degraded" in verdict_reasons

    verdict, verdict_reasons = model_impact_module._classify_verdict(
        rare_class_recall_delta=0.10,
        macro_f1_delta=0.01,
        weighted_f1_delta=0.0,
        pr_auc_delta=-0.04,
        rare_class_recall_drop_threshold=0.05,
        macro_f1_drop_threshold=0.03,
        weighted_f1_drop_threshold=0.03,
        pr_auc_drop_threshold=0.03,
        validation_gates_blocker_present=False,
    )
    assert verdict is ModelImpactVerdict.DEGRADED
    assert "pr_auc_degraded" in verdict_reasons

    utility_status, utility_reasons = model_impact_module._classify_synthetic_utility(
        candidate_is_synthetic=True,
        verdict=ModelImpactVerdict.IMPROVED,
        rare_class_recall_delta=0.10,
        macro_f1_delta=0.01,
        weighted_f1_delta=-0.04,
        pr_auc_delta=0.0,
        weighted_f1_drop_threshold=0.03,
        pr_auc_drop_threshold=0.03,
        tstr_trts=TstrTrtsMetrics(
            status=MetricStatus.AVAILABLE,
            tstr_macro_f1_drop=0.0,
            tstr_macro_f1_threshold=0.10,
            trts_macro_f1_delta=0.0,
            trts_unstable_threshold=0.20,
        ),
        validation_gates_blocker_present=False,
    )
    assert utility_status is SyntheticUtilityStatus.REJECTED
    assert "weighted_f1_degraded" in utility_reasons


# ---------------------------------------------------------------------------
# Validation gates blocker -> rejected
# ---------------------------------------------------------------------------


def test_validation_gates_blocker_rejects_candidate(tmp_path: Path) -> None:
    """validation_gates_blocker_present=True forces REJECTED verdict."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split_action = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    request = _request(
        source_artifact=source_artifact,
        candidate_artifact=source_artifact,
        source_split=split_action.manifest,
        candidate_split=split_action.manifest,
        candidate_is_synthetic=False,
        validation_gates_blocker_present=True,
    )
    result = run_model_impact(request, storage=storage, registry=registry)
    assert result.report.verdict is ModelImpactVerdict.REJECTED
    assert "validation_gates_blocker_present" in result.report.verdict_reason_codes


def test_pr_auc_falls_back_to_not_applicable_when_rare_class_absent(tmp_path: Path) -> None:
    """When the rare class is missing from the test split, PR-AUC is not_applicable."""
    storage, registry = _storage_and_registry()
    source_artifact = _source_transactions_artifact(tmp_path, registry)
    split_action = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    # Build a manifest where the test split contains only the majority
    # class. We construct it by promoting all rare-class rows to train.
    trimmed_manifest = _trim_rare_class_from_test(split_action.manifest)
    request = _request(
        source_artifact=source_artifact,
        candidate_artifact=source_artifact,
        source_split=trimmed_manifest,
        candidate_split=trimmed_manifest,
        candidate_is_synthetic=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_model_impact(request, storage=storage, registry=registry)
    assert result.report.pr_auc_status is MetricStatus.NOT_APPLICABLE
    assert result.report.pr_auc_reason is not None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _request(
    *,
    source_artifact: ArtifactRef,
    candidate_artifact: ArtifactRef,
    source_split: SplitManifest,
    candidate_split: SplitManifest,
    candidate_is_synthetic: bool,
    synthetic_dataset_report_artifact: ArtifactRef | None = None,
    validation_gates_blocker_present: bool = False,
    report_id: str | None = None,
) -> RunModelImpactRequest:
    return RunModelImpactRequest(
        dataset_id="dataset_1",
        parent_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        organization_id="org_1",
        project_id="project_1",
        source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
        source_split_manifest=source_split,
        candidate_split_manifest=candidate_split,
        feature_columns=("amount", "monthly_income"),
        target_column="is_fraud",
        rare_class_label="1",
        candidate_is_synthetic=candidate_is_synthetic,
        synthetic_dataset_report_artifact=synthetic_dataset_report_artifact,
        validation_gates_blocker_present=validation_gates_blocker_present,
        random_seed=42,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        report_id=report_id,
        generated_at=_GENERATED_AT,
    )


def _split_request(*, source_artifact: ArtifactRef) -> ExecuteTabularSplitRequest:
    split_step = ActionPlanStep(
        step_id="create_split_tabular",
        type="CREATE_SPLIT",
        depends_on=(),
        idempotency_key="sha256:" + "3" * 64,
        method_id="group_stratified",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "4" * 64,
        policy_version="split_policy_v0",
        validation_gates=("schema_validation",),
        preconditions=("source_version_is_immutable",),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind=SPLIT_MANIFEST_KIND,
        config={
            "strategy": "group_stratified",
            "group_key": "customer_id_hash",
            "target_column": "is_fraud",
        },
        random_seed=42,
        retry_policy=RetryPolicy(
            max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )
    return ExecuteTabularSplitRequest(
        action_plan_id="action_plan_model_impact_001",
        step=split_step,
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        candidate_dataset_version_id="dataset_version_v2_candidate",
        source_artifact=source_artifact,
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
        target_column="is_fraud",
        seed=42,
        generated_at=_GENERATED_AT,
    )


def _smote_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="augment_rare_class_smote",
        type="AUGMENT_RARE_CLASS",
        depends_on=("create_split_tabular",),
        idempotency_key="sha256:" + "1" * 64,
        method_id="smote",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="synthetic_policy_v0",
        validation_gates=(
            "split_leakage_check",
            "schema_validation",
            "synthetic_dcr_check",
        ),
        preconditions=(
            "source_version_is_immutable",
            "train_split_exists",
            "leakage_checks_passed",
        ),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind="CANDIDATE_DATASET_VERSION",
        config={
            "target_column": "is_fraud",
            "rare_class_label": "1",
            "method": "smote",
            "source_split": "train",
        },
        random_seed=42,
        retry_policy=RetryPolicy(
            max_attempts=2, retryable_errors=("TRANSIENT_STORAGE_ERROR",)
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _trim_rare_class_from_test(manifest: SplitManifest) -> SplitManifest:
    """Promote all rare-class rows to TRAIN so TEST has only majority class."""
    new_assignments: list[SplitAssignment] = []
    for assignment in manifest.assignments:
        if assignment.split is DataSplit.TEST and assignment.label == "1":
            new_assignments.append(
                assignment.model_copy(update={"split": DataSplit.TRAIN})
            )
        else:
            new_assignments.append(assignment)
    # Recompute class distribution counts.
    new_distribution: list[SplitClassDistribution] = []
    for item in manifest.class_distribution:
        if item.split is DataSplit.TEST:
            counts = {"0": item.class_counts.get("0", 0)}
            total = counts["0"]
            ratios = {"0": 1.0 if total else 0.0}
            new_distribution.append(
                item.model_copy(
                    update={
                        "class_counts": counts,
                        "class_ratios": ratios,
                        "total_count": total,
                    }
                )
            )
        elif item.split is DataSplit.TRAIN:
            counts = dict(item.class_counts)
            test_rare = next(
                (
                    sub.class_counts.get("1", 0)
                    for sub in manifest.class_distribution
                    if sub.split is DataSplit.TEST
                ),
                0,
            )
            counts["1"] = counts.get("1", 0) + test_rare
            counts["0"] = counts.get("0", 0)
            total = sum(counts.values())
            ratios = {label: (c / total if total else 0.0) for label, c in counts.items()}
            new_distribution.append(
                item.model_copy(
                    update={
                        "class_counts": counts,
                        "class_ratios": ratios,
                        "total_count": total,
                    }
                )
            )
        else:
            new_distribution.append(item)
    return manifest.model_copy(
        update={
            "assignments": tuple(new_assignments),
            "class_distribution": tuple(new_distribution),
        }
    )


def _source_transactions_artifact(
    tmp_path: Path, registry: ArtifactRegistry
) -> ArtifactRef:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        transactions = reader.find_required_transactions().read_bytes()
    return registry.save_artifact(
        artifact_kind="raw_transactions",
        data=transactions,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_v1",
        created_by_job_id="compute_run_apply_001",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


def _storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name="dataforge-local",
        prefix_root="dataforge",
        scope=ObjectStorageScope(
            organization_id="org_1",
            project_id="project_1",
            dataset_id="dataset_1",
        ),
    )
    return storage, ArtifactRegistry(storage=storage)


# silence unused import warnings used only for typing helpers
_ = (SplitClassDistribution, SplitManifestLineage, SplitStrategy, ErrorCode)


class _InMemoryS3Client(S3CompatibleClient):
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        Metadata: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
            "LastModified": _GENERATED_AT,
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "Body": io.BytesIO(stored["Body"]),
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        stored = self._object(Bucket, Key)
        return {
            "ContentLength": len(stored["Body"]),
            "ContentType": stored["ContentType"],
            "Metadata": stored["Metadata"],
            "LastModified": stored["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(stored["Body"])}
                for (bucket, key), stored in sorted(self._objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message="Object does not exist",
            ) from exc
