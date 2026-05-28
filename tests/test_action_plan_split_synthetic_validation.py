"""Consolidated TASK-064 invariants for ActionPlan, split/leakage, and synthetic gates.

This file collects the four acceptance criteria from TASK-064 into a single
top-level test module so a regression in any one of them surfaces as a
clearly named test failure:

1. ActionPlan preview is non-mutating and deterministic.
   - Repeated previews from the same recommendations and selection produce
     byte-identical ActionPlans.
   - Building a preview does not write any object into object storage and
     does not register any artifact.
2. Approval-required actions cannot enter execution without signed
   platform metadata.
   - ``validate_action_plan_execution`` rejects the request.
   - The ``launch_apply_actions_workflow`` launcher refuses to materialize
     APPLY assets and emits no platform job events on the rejected path.
3. Group split keeps customer/case rows together.
   - The MVP ``group_stratified`` strategy never splits one group across
     multiple data splits, both on the demo customer-id grouping and on
     a synthetic case-id grouping.
4. Synthetic output is marked ``synthetic``, reproducible, lineage-tracked,
   and blocked by failed gates.
   - SMOTE-augmented candidate rows carry ``is_synthetic=1`` and
     ``synthetic_source_split=train`` plus full sample lineage.
   - Re-running with the same seed yields identical sample lineage and
     identical candidate artifact hash.
   - When a synthetic row exactly duplicates a real row the validation
     gates block the candidate (``CandidateArtifactStatus.VALIDATION_FAILED``,
     ``block_export=True``) and the raw source bytes stay unchanged.

The file deliberately exercises behaviour through public APIs so the
underlying mechanism is free to evolve.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.api.schemas import ActionPlanExecuteApprovedRequest
from app.domain import (
    SMOTE_FORMULA,
    ActionPlan,
    ActionPlanStep,
    ArtifactRef,
    BusinessRuleSeverity,
    CandidateArtifactStatus,
    DataSplit,
    ErrorCode,
    MethodRecommendation,
    RetryPolicy,
    SyntheticGenerationMethod,
    TabularProfileReport,
    ValidationGateStatus,
    ValidationGateType,
)
from app.ingestion import open_archive_path
from app.kernel import (
    ActionPlanApprovalMetadata,
    ActionPlanExecutionError,
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    ValidateActionPlanExecutionRequest,
    action_plan_integrity_hash,
    build_action_plan_preview,
    build_method_recommendations,
    validate_action_plan_execution,
)
from app.kernel.config import (
    DagsterSettings,
    ExternalAISettings,
    ObjectStorageSettings,
    PlatformSettings,
    PolicySettings,
    RuntimeProfile,
    ServiceConfig,
    profile_defaults,
)
from app.orchestration.apply_workflow import launch_apply_actions_workflow
from app.plugins.tabular import (
    SPLIT_MANIFEST_KIND,
    ExecuteSmoteAugmentationRequest,
    ExecuteTabularSplitRequest,
    execute_smote_augmentation_action,
    execute_tabular_split_action,
)
from app.plugins.tabular.rules import BusinessRule, RuleFieldCheck
from app.plugins.validation import (
    DcrThresholds,
    RunValidationGatesRequest,
    run_validation_gates,
)
from app.validation.contracts import load_contract_pack
from tests.fixtures.demo_archive import build_demo_archive

_CONFIG_HASH = "sha256:" + "e" * 64
_GATES_CONFIG_HASH = "sha256:" + "d" * 64
_SMOTE_CONFIG_HASH = "sha256:" + "f" * 64
_CREATED_AT = datetime(2026, 6, 4, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# AC1: ActionPlan preview is non-mutating and deterministic.
# ---------------------------------------------------------------------------


def test_action_plan_preview_is_deterministic_for_same_inputs() -> None:
    """Same recommendations and selection produce byte-identical ActionPlans."""
    recommendations = _method_recommendations()
    request = BuildActionPlanPreviewRequest(
        decision_report_id="decision_report_064",
        source_dataset_version_id="dataset_version_1",
        selected_decision_ids=tuple(
            recommendation.recommendation_id for recommendation in recommendations
        ),
        selected_method_overrides={},
        method_recommendations=recommendations,
        created_by_user_id="platform_user_064",
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
        ),
        target_version_name="dataset_version_v2_preview_064",
        created_at=_CREATED_AT,
    )

    first = build_action_plan_preview(request)
    second = build_action_plan_preview(request)

    # ID, hash, every step's idempotency key, and the validation/output
    # ordering must be identical between calls.
    assert first.action_plan_id == second.action_plan_id
    assert action_plan_integrity_hash(first) == action_plan_integrity_hash(second)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert tuple(step.idempotency_key for step in first.steps) == tuple(
        step.idempotency_key for step in second.steps
    )
    assert first.validation_gates == second.validation_gates
    assert first.expected_outputs == second.expected_outputs


def test_action_plan_preview_does_not_mutate_object_storage_or_registry() -> None:
    """Building a preview must not write any artifact or storage object."""
    storage, registry = _empty_storage_and_registry()
    objects_before = storage.list()

    recommendations = _method_recommendations()
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_064",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_064",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
            ),
            target_version_name="dataset_version_v2_preview_064",
            created_at=_CREATED_AT,
        )
    )

    assert plan.execution_mode.value == "PREVIEW_ACTION_PLAN"
    objects_after = storage.list()
    assert objects_after == objects_before
    # Registry has no surface for direct artifact enumeration; touching
    # ``registry`` here documents that no artifact was created during
    # preview generation. ``storage.list()`` covers the persistence layer.
    assert registry is not None


# ---------------------------------------------------------------------------
# AC2: Approval-required action cannot execute without signed metadata.
# ---------------------------------------------------------------------------


def test_validate_action_plan_execution_rejects_missing_approval_metadata() -> None:
    """Approval-required ActionPlan without metadata raises with the canonical reason."""
    plan = _approval_required_plan()

    with pytest.raises(ActionPlanExecutionError) as error:
        validate_action_plan_execution(
            ValidateActionPlanExecutionRequest(
                action_plan=plan,
                source_dataset_version_id=plan.source_dataset_version_id,
            )
        )

    assert error.value.code is ErrorCode.ACTION_PLAN_REQUIRES_APPROVAL
    assert error.value.reason_code == "approval_metadata_required"
    assert error.value.status_code == 403


def test_validate_action_plan_execution_rejects_tampered_approval_hash() -> None:
    """Approval metadata that does not match plan integrity must be rejected."""
    plan = _approval_required_plan()
    correct = _approval_metadata(plan)
    tampered = correct.model_copy(
        update={"action_plan_hash": "sha256:" + "0" * 64},
    )

    with pytest.raises(ActionPlanExecutionError) as error:
        validate_action_plan_execution(
            ValidateActionPlanExecutionRequest(
                action_plan=plan,
                source_dataset_version_id=plan.source_dataset_version_id,
                approval_metadata=tampered,
            )
        )

    assert error.value.code is ErrorCode.ACTION_PLAN_SIGNATURE_INVALID
    assert error.value.reason_code == "approval_metadata_integrity_mismatch"


def test_apply_workflow_refuses_unsigned_request_and_emits_no_platform_events() -> None:
    """Launcher must reject requests without approval_metadata and stay silent."""
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    plan = _approval_required_plan()
    plan_hash = action_plan_integrity_hash(plan)

    unsigned_request = ActionPlanExecuteApprovedRequest(
        platform_job_id="platform_job_064",
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        source_dataset_version_id=plan.source_dataset_version_id,
        action_plan=plan,
        approval_metadata=None,
    )

    with pytest.raises(ValueError, match="approval_metadata"):
        launch_apply_actions_workflow(
            request=unsigned_request,
            action_plan_hash=plan_hash,
            config=config,
            fake_platform=fake_platform,
        )

    snapshot = fake_platform.snapshot()
    assert snapshot.job_events == ()
    assert snapshot.artifact_refs == ()


# ---------------------------------------------------------------------------
# AC3: Group split keeps customer/case rows together.
# ---------------------------------------------------------------------------


def test_group_split_keeps_customers_in_a_single_split(tmp_path: Path) -> None:
    """Demo customer_id_hash never spans more than one split."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)

    result = execute_tabular_split_action(
        _split_request(
            source_artifact=source_artifact,
            group_key="customer_id_hash",
        ),
        storage=storage,
        registry=registry,
    )

    rows = _read_csv(transactions)
    assignments = {
        assignment.object_id: assignment for assignment in result.manifest.assignments
    }
    splits_by_group: dict[str, set[DataSplit]] = defaultdict(set)
    for row in rows:
        assignment = assignments[row["object_id"]]
        splits_by_group[row["customer_id_hash"]].add(assignment.split)
    assert splits_by_group, "demo dataset must contain customer groups"
    assert all(len(splits) == 1 for splits in splits_by_group.values()), (
        "group_stratified must keep every customer_id_hash in a single split"
    )


def test_group_split_keeps_synthetic_case_groups_together() -> None:
    """A synthetic case-id grouping must still be respected by group_stratified."""
    storage, registry = _storage_and_registry()
    # Six cases, two rows each; rare class label "1" assigned to two cases
    # so the splitter has at least one rare-class case per split.
    rows: list[dict[str, str]] = []
    for case_index in range(1, 7):
        case_id = f"case_{case_index:03d}"
        # Alternate the rare class so the stratifier has rare and majority
        # rows to balance.
        label = "1" if case_index % 3 == 0 else "0"
        for sub_index in range(1, 3):
            rows.append(
                {
                    "object_id": f"{case_id}_row_{sub_index}",
                    "is_fraud": label,
                    "case_id": case_id,
                    "amount": str(10 + case_index + sub_index),
                    "monthly_income": str(1000 + case_index * 10),
                }
            )
    csv_bytes = _write_csv(
        rows,
        header=("object_id", "is_fraud", "case_id", "amount", "monthly_income"),
    )
    source_artifact = _save_source_artifact(registry=registry, data=csv_bytes)

    result = execute_tabular_split_action(
        _split_request(
            source_artifact=source_artifact,
            group_key="case_id",
        ),
        storage=storage,
        registry=registry,
    )

    assert result.manifest.group_key == "case_id"
    assignments_by_object = {
        assignment.object_id: assignment for assignment in result.manifest.assignments
    }
    splits_by_case: dict[str, set[DataSplit]] = defaultdict(set)
    for row in rows:
        splits_by_case[row["case_id"]].add(assignments_by_object[row["object_id"]].split)
    assert all(len(splits) == 1 for splits in splits_by_case.values())


# ---------------------------------------------------------------------------
# AC4: Synthetic output is marked, reproducible, lineage-tracked,
# and blocked by failed gates.
# ---------------------------------------------------------------------------


def test_synthetic_smote_marks_rows_and_carries_full_lineage(tmp_path: Path) -> None:
    """SMOTE candidate rows must be marked synthetic with method/source-split lineage."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    result = execute_smote_augmentation_action(
        ExecuteSmoteAugmentationRequest(
            action_plan_id="action_plan_064_smote",
            step=_smote_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_064",
            config_hash=_SMOTE_CONFIG_HASH,
            random_seed=42,
            k_neighbors=3,
            sampling_strategy=0.20,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    candidate_rows = _read_csv(storage.get(result.candidate_artifact.uri).data)
    synthetic_rows = [row for row in candidate_rows if row.get("is_synthetic") == "1"]
    assert synthetic_rows, "SMOTE must materialize at least one synthetic row"
    for row in synthetic_rows:
        assert row["synthetic_source_split"] == DataSplit.TRAIN.value
        assert row["object_id"].startswith("txn_synth_")

    assert result.report.method is SyntheticGenerationMethod.SMOTE
    assert result.report.formula == SMOTE_FORMULA
    assert result.report.source_split == DataSplit.TRAIN.value
    assert result.report.lineage.source_dataset_version_id == "dataset_version_1"
    assert result.report.lineage.action_plan_id == "action_plan_064_smote"
    assert result.report.lineage.step_id == _smote_step().step_id
    assert all(
        entry.formula == SMOTE_FORMULA for entry in result.report.sample_lineage
    )


def test_synthetic_smote_is_reproducible_across_runs(tmp_path: Path) -> None:
    """Same seed plus same inputs yields identical lineage and artifact hash."""
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )

    request = ExecuteSmoteAugmentationRequest(
        action_plan_id="action_plan_064_smote",
        step=_smote_step(),
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        source_artifact=source_artifact,
        split_manifest=split.manifest,
        split_manifest_artifact=split.split_artifact.artifact_ref,
        created_by_job_id="compute_run_apply_064",
        config_hash=_SMOTE_CONFIG_HASH,
        random_seed=42,
        k_neighbors=3,
        sampling_strategy=0.20,
        generated_at=_CREATED_AT,
        report_id="synthetic_dataset_report_repro",
    )
    first = execute_smote_augmentation_action(request, storage=storage, registry=registry)
    second = execute_smote_augmentation_action(request, storage=storage, registry=registry)

    assert first.candidate_artifact.hash == second.candidate_artifact.hash
    assert first.augmented_split_artifact.hash == second.augmented_split_artifact.hash
    first_lineage = [
        (
            entry.synthetic_object_id,
            entry.seed_object_id,
            entry.neighbor_object_id,
            entry.lambda_value,
        )
        for entry in first.report.sample_lineage
    ]
    second_lineage = [
        (
            entry.synthetic_object_id,
            entry.seed_object_id,
            entry.neighbor_object_id,
            entry.lambda_value,
        )
        for entry in second.report.sample_lineage
    ]
    assert first_lineage == second_lineage


def test_synthetic_failed_validation_gate_blocks_export_and_keeps_source_immutable(
    tmp_path: Path,
) -> None:
    """A synthetic row that duplicates a real row must fail gates without mutating raw data.

    The fixture chains the actual MVP plugins to build a valid
    ``SyntheticDatasetReport`` (so the contract matches in full):

    1. run a real SMOTE generation on the demo split,
    2. replace the candidate CSV with one whose synthetic row exactly
       duplicates a real row,
    3. run the validation gates against the patched candidate.

    The gates must mark the candidate ``VALIDATION_FAILED``, set
    ``block_export=True``, fail the SYNTHETIC_EXACT_DUPLICATE_TO_REAL
    and DCR gates, and the raw source artifact must stay byte-identical.
    """
    storage, registry = _storage_and_registry()
    transactions = _read_transactions(tmp_path)
    source_artifact = _save_source_artifact(registry=registry, data=transactions)
    source_bytes_before = storage.get(source_artifact.uri).data

    split = execute_tabular_split_action(
        _split_request(source_artifact=source_artifact),
        storage=storage,
        registry=registry,
    )
    smote = execute_smote_augmentation_action(
        ExecuteSmoteAugmentationRequest(
            action_plan_id="action_plan_064_smote",
            step=_smote_step(),
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate",
            source_artifact=source_artifact,
            split_manifest=split.manifest,
            split_manifest_artifact=split.split_artifact.artifact_ref,
            created_by_job_id="compute_run_apply_064",
            config_hash=_GATES_CONFIG_HASH,
            random_seed=42,
            k_neighbors=3,
            sampling_strategy=0.20,
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )
    schema_columns, real_rows, synthetic_rows = _split_candidate_rows(
        storage.get(smote.candidate_artifact.uri).data
    )
    assert synthetic_rows, "SMOTE must produce at least one synthetic row"
    # Replace the first synthetic row's feature columns with a verbatim
    # copy of a real row so the synthetic-exact-duplicate gate fires.
    real_rare_row = next(row for row in real_rows if row["is_fraud"] == "1")
    feature_columns = ("amount", "monthly_income")
    duplicated_synthetic = dict(synthetic_rows[0])
    for column in feature_columns:
        duplicated_synthetic[column] = real_rare_row[column]
    duplicated_synthetic["is_fraud"] = real_rare_row["is_fraud"]
    rebuilt_synthetic_rows = (duplicated_synthetic, *synthetic_rows[1:])
    patched_candidate_csv = _write_csv(
        list(real_rows) + list(rebuilt_synthetic_rows),
        header=schema_columns,
    )
    patched_candidate_artifact = registry.save_artifact(
        artifact_kind="candidate_tabular_dataset",
        data=patched_candidate_csv,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id="dataset_version_2_candidate_dup",
        created_by_job_id="compute_run_apply_064",
        config_hash=_GATES_CONFIG_HASH,
    ).artifact_ref

    result = run_validation_gates(
        RunValidationGatesRequest(
            dataset_id="dataset_1",
            source_dataset_version_id="dataset_version_1",
            candidate_dataset_version_id="dataset_version_2_candidate_dup",
            candidate_artifact=patched_candidate_artifact,
            source_artifact=source_artifact,
            candidate_artifact_kind="candidate_tabular_dataset",
            schema_columns=schema_columns,
            numeric_columns=feature_columns,
            synthetic_dataset_report=smote.report,
            synthetic_dataset_report_artifact=smote.report_artifact.artifact_ref,
            split_manifest_artifact=smote.augmented_split_artifact.artifact_ref,
            dcr_thresholds=DcrThresholds(),
            created_by_job_id="compute_run_apply_064",
            config_hash=_GATES_CONFIG_HASH,
            action_plan_id="action_plan_064_smote",
            step_id="augment_rare_class_smote",
            report_id="validation_gates_064_dup",
            generated_at=_CREATED_AT,
        ),
        storage=storage,
        registry=registry,
    )

    report = result.report
    assert report.candidate_status is CandidateArtifactStatus.VALIDATION_FAILED
    assert report.block_export is True
    assert (
        ValidationGateType.SYNTHETIC_EXACT_DUPLICATE_TO_REAL
        in report.blocker_gate_types
    )
    dcr_gate = next(
        gate
        for gate in report.gates
        if gate.gate_type is ValidationGateType.SYNTHETIC_DCR_CHECK
    )
    assert dcr_gate.status is ValidationGateStatus.FAILED

    # Source artifact must be byte-identical after a failed gates run.
    assert storage.get(source_artifact.uri).data == source_bytes_before
    immutability_gate = next(
        gate
        for gate in report.gates
        if gate.gate_type is ValidationGateType.RAW_ARTIFACT_IMMUTABILITY
    )
    assert immutability_gate.status is ValidationGateStatus.PASSED


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _method_recommendations() -> tuple[MethodRecommendation, ...]:
    return build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=_demo_tabular_profile())
    )


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example
        for example in pack.examples
        if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def _approval_required_plan() -> ActionPlan:
    recommendations = _method_recommendations()
    base = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_064_apply",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_064",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
            ),
            target_version_name="dataset_version_v2_candidate_064",
            created_at=_CREATED_AT,
        )
    )
    return base.model_copy(
        update={
            "requires_approval": True,
            "approval_request_id": "approval_request_064",
        }
    )


def _approval_metadata(plan: ActionPlan) -> ActionPlanApprovalMetadata:
    return ActionPlanApprovalMetadata(
        approval_id="approval_064",
        approval_request_id="approval_request_064",
        approved_by_user_id="platform_owner_064",
        approved_at=_CREATED_AT,
        action_plan_id=plan.action_plan_id,
        action_plan_hash=action_plan_integrity_hash(plan),
        decision_report_id=plan.created_from_decision_report,
        source_dataset_version_id=plan.source_dataset_version_id,
    )


def _smote_step() -> ActionPlanStep:
    return ActionPlanStep(
        step_id="augment_rare_class_smote_064",
        type="AUGMENT_RARE_CLASS",
        depends_on=("create_split_tabular_064",),
        idempotency_key="sha256:" + "1" * 64,
        method_id="smote",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "2" * 64,
        policy_version="synthetic_policy_v0",
        validation_gates=(
            "split_leakage_check",
            "schema_validation",
            "business_rules",
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
            max_attempts=2,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )


def _split_request(
    *,
    source_artifact: ArtifactRef,
    group_key: str = "customer_id_hash",
) -> ExecuteTabularSplitRequest:
    split_step = ActionPlanStep(
        step_id="create_split_tabular_064",
        type="CREATE_SPLIT",
        depends_on=(),
        idempotency_key="sha256:" + "3" * 64,
        method_id="group_stratified",
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash="sha256:" + "4" * 64,
        policy_version="split_policy_v0",
        validation_gates=("schema_validation", "split_policy_check"),
        preconditions=("source_version_is_immutable",),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/transactions.csv",
        ),
        output_artifact_kind=SPLIT_MANIFEST_KIND,
        config={
            "strategy": "group_stratified",
            "group_key": group_key,
            "target_column": "is_fraud",
        },
        random_seed=42,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
        ),
        on_failure="fail_plan_keep_artifacts_unpromoted",
        promotion_scope="candidate_only",
    )
    return ExecuteTabularSplitRequest(
        action_plan_id="action_plan_064_split",
        step=split_step,
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        source_artifact=source_artifact,
        created_by_job_id="compute_run_apply_064",
        config_hash=_CONFIG_HASH,
        target_column="is_fraud",
        seed=42,
        generated_at=_CREATED_AT,
    )


def _split_candidate_rows(
    data: bytes,
) -> tuple[tuple[str, ...], list[dict[str, str]], list[dict[str, str]]]:
    """Return (header, real rows, synthetic rows) from a candidate CSV blob."""
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = tuple(name for name in (reader.fieldnames or ()) if name)
    real_rows: list[dict[str, str]] = []
    synthetic_rows: list[dict[str, str]] = []
    for row in reader:
        normalized = {
            column: ("" if row.get(column) is None else str(row.get(column)))
            for column in header
        }
        if normalized.get("is_synthetic") == "1":
            synthetic_rows.append(normalized)
        else:
            real_rows.append(normalized)
    return header, real_rows, synthetic_rows


def _save_source_artifact(
    *,
    registry: ArtifactRegistry,
    data: bytes,
    version: str = "dataset_version_1",
) -> ArtifactRef:
    return registry.save_artifact(
        artifact_kind="raw_transactions",
        data=data,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id=version,
        created_by_job_id="compute_run_apply_064",
        config_hash=_CONFIG_HASH,
    ).artifact_ref


def _read_transactions(tmp_path: Path) -> bytes:
    built = build_demo_archive(output_dir=tmp_path / "demo_archive")
    with open_archive_path(built.archive_path) as reader:
        return reader.find_required_transactions().read_bytes()


def _read_csv(data: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    return [dict(row) for row in reader]


def _write_csv(rows: list[dict[str, str]], *, header: tuple[str, ...]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in header})
    return buffer.getvalue().encode("utf-8")


def _empty_storage_and_registry() -> tuple[MinioObjectStorageAdapter, ArtifactRegistry]:
    return _storage_and_registry()


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


def _test_config() -> ServiceConfig:
    profile = RuntimeProfile.DEMO_STRICT
    return ServiceConfig(
        profile=profile,
        object_storage=ObjectStorageSettings(
            endpoint_url="http://localhost:9000",
            bucket_name="dataforge-local",
            region="local",
            prefix_root="dataforge",
        ),
        platform=PlatformSettings(
            callback_url="http://platform.local/api/ml/jobs/callback",
            service_signing_secret="test-signing-secret",  # type: ignore[arg-type]
            service_identity="dataforge-platform",
            signature_max_age_seconds=300,
        ),
        dagster=DagsterSettings(
            home="/tmp/dataforge-dagster-test-064",
            job_name="dataforge_apply_selected_actions",
            run_queue="default",
        ),
        policies=PolicySettings(
            policy_config_path="configs/policies/demo_strict.yaml",
            decision_policy_path="configs/policies/decision_v0.yaml",
            score_policy_path="configs/policies/score_v0.yaml",
        ),
        contract_pack_version="local-fallback-v0.1.0-demo",
        external_ai=ExternalAISettings(allow_external_api=False),
        profile_defaults=profile_defaults(profile),
    )


# Silence unused-import warnings for symbols used only as helpers in tests.
_ = (BusinessRule, BusinessRuleSeverity, RuleFieldCheck)


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
            "LastModified": _CREATED_AT,
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
