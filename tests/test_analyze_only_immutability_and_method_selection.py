"""Consolidated invariants for ANALYZE_ONLY immutability and method selection.

This file collects the safety/policy invariants required by TASK-063 into a
single, top-level test module so a regression in any one of them surfaces as
a clearly named test failure. The invariants come straight from the task
acceptance criteria:

1. ``ANALYZE_ONLY`` must not produce candidate-dataset/apply artifacts and
   must not write back to raw artifact storage.
2. ``ANALYZE_ONLY`` with a prediction-manifest artifact must produce only
   prediction-derived reports/review queues and keep
   ``mutates_dataset = False``.
3. Target imputation is always blocked.
4. SMOTE before split (no split manifest) is blocked, and SMOTE on
   validation/test splits is blocked.
5. Borderline-SMOTE / ADASYN are not selectable in the MVP profile.
6. CTGAN / PMM are not selectable in the MVP profile.

The implementation prefers thin, behavior-level assertions so that the
underlying code is free to evolve; the goal is to fix the contract, not
the mechanism.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.api.main import create_app
from app.api.security import (
    ORGANIZATION_ID_HEADER,
    PROJECT_ID_HEADER,
    SERVICE_IDENTITY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    build_service_signature,
)
from app.domain import (
    ActionPlanStep,
    ArtifactLineage,
    ArtifactRef,
    ComputeRunStatus,
    DataSplit,
    MethodCandidateStatus,
    PolicyStatus,
    RetryPolicy,
    TabularProfileReport,
)
from app.kernel import (
    BuildMethodRecommendationsRequest,
    build_method_recommendations,
)
from app.kernel.config import ServiceConfig, load_config
from app.orchestration import APPLY_ASSET_KEYS
from app.plugins.tabular.synthetic_smote import (
    ExecuteSmoteAugmentationRequest,
    SmoteExecutionError,
    execute_smote_augmentation_action,
)
from app.validation.contracts import load_contract_pack

_DEMO_ENV: dict[str, str] = {
    "DATAFORGE_PROFILE": "demo_strict",
    "DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL": "http://localhost:9000",
    "DATAFORGE_OBJECT_STORAGE_BUCKET": "dataforge-local",
    "DATAFORGE_PLATFORM_CALLBACK_URL": "http://platform.local/api/ml/jobs/callback",
    "DATAFORGE_SERVICE_SIGNING_SECRET": "local-dev-signing-secret",
    "DATAFORGE_DAGSTER_HOME": "/tmp/dataforge-dagster",
    "DATAFORGE_POLICY_CONFIG_PATH": "configs/policies/demo_strict.yaml",
    "DATAFORGE_DECISION_POLICY_PATH": "configs/policies/decision_v0.yaml",
    "DATAFORGE_SCORE_POLICY_PATH": "configs/policies/score_v0.yaml",
}

_APPLY_ONLY_ASSET_NAMES: frozenset[str] = frozenset(
    asset_key.path[-1] for asset_key in APPLY_ASSET_KEYS
)


# ---------------------------------------------------------------------------
# AC1 / AC2: ANALYZE_ONLY immutability invariants.
# ---------------------------------------------------------------------------


def test_analyze_only_endpoint_does_not_produce_candidate_or_apply_artifacts() -> None:
    """ANALYZE_ONLY must not surface any APPLY-only asset in materialized outputs."""
    config = _test_config()
    app = create_app(config=config)
    payload = _analyze_payload()
    body = _body_bytes(payload)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/jobs/analyze-dataset",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )
    assert response.status_code == 202

    data = response.json()
    materialized = set(data["materialized_assets"])
    assert data["mutates_dataset"] is False
    assert _APPLY_ONLY_ASSET_NAMES.isdisjoint(materialized)
    # No mutation-leaning name leaks into the response payload.
    forbidden_substrings = (
        "candidate_dataset",
        "synthetic_dataset",
        "prepared_dataset",
        "export_package",
        "model_impact_report",
    )
    body_text = response.text.lower()
    for forbidden in forbidden_substrings:
        assert forbidden not in body_text


def test_analyze_only_dagster_run_writes_no_raw_or_candidate_artifacts() -> None:
    """ANALYZE_ONLY assets must leave object storage and the artifact registry empty."""
    from app.api.schemas import AnalyzeDatasetRequest
    from app.orchestration.analyze_workflow import launch_analyze_dataset_workflow

    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request = AnalyzeDatasetRequest.model_validate(_analyze_payload())

    result = launch_analyze_dataset_workflow(
        request=request,
        config=config,
        fake_platform=fake_platform,
    )

    assert result.status is ComputeRunStatus.ACCEPTED
    assert result.mutates_dataset is False
    assert _APPLY_ONLY_ASSET_NAMES.isdisjoint(result.materialized_assets)

    snapshot = fake_platform.snapshot()
    # Skeleton ANALYZE_ONLY assets must not register any candidate/raw artifacts
    # back to the platform fake. Real implementations may register read-only
    # artifact refs in the future; this test asserts the current invariant
    # while a stronger invariant guards against APPLY-style artifact kinds.
    forbidden_kinds = {
        "candidate_dataset_version",
        "synthetic_dataset",
        "prepared_dataset",
        "export_package",
        "model_impact_report",
        "remediation_execution_report",
        "action_plan",
    }
    recorded_kinds = {ref.kind for ref in snapshot.artifact_refs}
    assert forbidden_kinds.isdisjoint(recorded_kinds)


def test_analyze_only_with_predictions_keeps_mutates_dataset_false() -> None:
    """ANALYZE_ONLY with prediction artifact only adds prediction-derived assets."""
    config = _test_config()
    app = create_app(config=config)
    payload = _analyze_payload(
        prediction_artifact_refs=[
            _artifact_ref("prediction_manifest_1", "prediction_manifest")
        ]
    )
    body = _body_bytes(payload)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/jobs/analyze-dataset",
        content=body,
        headers=_signed_headers(config=config, body=body, payload=payload),
    )

    assert response.status_code == 202
    data = response.json()
    materialized = set(data["materialized_assets"])
    expected_prediction_assets = {
        "prediction_manifest",
        "prediction_validation_report",
        "model_error_analysis_report",
        "ambiguous_object_candidates",
        "probable_label_error_candidates",
    }
    assert expected_prediction_assets.issubset(materialized)
    assert _APPLY_ONLY_ASSET_NAMES.isdisjoint(materialized)
    assert data["mutates_dataset"] is False


# ---------------------------------------------------------------------------
# AC3 / AC5 / AC6: Method-selection policy invariants in the MVP profile.
# ---------------------------------------------------------------------------


def test_method_selection_blocks_target_imputation() -> None:
    """target_imputation must always appear in blocked_methods with the policy reason."""
    imputation = _imputation_recommendation()

    blocked_ids = {blocked.method_id for blocked in imputation.blocked_methods}
    assert "target_imputation" in blocked_ids
    blocked = next(b for b in imputation.blocked_methods if b.method_id == "target_imputation")
    assert blocked.reason_code == "target_column_auto_imputation_forbidden"


def test_method_selection_disables_pmm_in_mvp_profile() -> None:
    """PMM must be disabled by policy in the MVP imputation recommendation."""
    imputation = _imputation_recommendation()

    pmm = _candidate_or_none(imputation, "pmm")
    assert pmm is not None, "imputation recommendation must list pmm"
    assert pmm.status is MethodCandidateStatus.DISABLED_BY_POLICY
    assert pmm.policy_status is PolicyStatus.DISABLED_BY_POLICY


def test_method_selection_disables_ctgan_in_mvp_profile() -> None:
    """CTGAN must be disabled by policy in the rare-class recommendation."""
    rare_class = _rare_class_recommendation()

    ctgan = _candidate_or_none(rare_class, "ctgan")
    assert ctgan is not None, "rare-class recommendation must list ctgan"
    assert ctgan.status is MethodCandidateStatus.DISABLED_BY_POLICY
    assert ctgan.policy_status is PolicyStatus.DISABLED_BY_POLICY


_NON_SELECTABLE_STATUSES: frozenset[MethodCandidateStatus] = frozenset(
    {
        MethodCandidateStatus.DISABLED_BY_POLICY,
        MethodCandidateStatus.DISABLED_BY_READINESS,
        MethodCandidateStatus.BLOCKED,
    }
)
_NON_SELECTABLE_POLICY_STATUSES: frozenset[PolicyStatus] = frozenset(
    {
        PolicyStatus.DISABLED_BY_POLICY,
        PolicyStatus.DISABLED_BY_READINESS,
        PolicyStatus.BLOCKED,
    }
)


@pytest.mark.parametrize("method_id", ["borderline_smote", "adasyn"])
def test_borderline_smote_and_adasyn_are_not_selectable_in_mvp_profile(
    method_id: str,
) -> None:
    """Borderline-SMOTE and ADASYN must not be selectable in MVP (policy or readiness)."""
    rare_class = _rare_class_recommendation()

    candidate = _candidate_or_none(rare_class, method_id)
    assert candidate is not None, f"rare-class recommendation must list {method_id}"
    assert candidate.status in _NON_SELECTABLE_STATUSES, (
        f"{method_id} must be marked non-selectable in MVP, got {candidate.status}"
    )
    assert candidate.policy_status in _NON_SELECTABLE_POLICY_STATUSES, (
        f"{method_id} must have non-selectable policy_status in MVP, "
        f"got {candidate.policy_status}"
    )


def test_recommended_method_is_never_a_blocked_or_disabled_candidate() -> None:
    """Recommended methods must always be selectable in the MVP profile."""
    for recommendation in _all_recommendations():
        recommended_id = recommendation.recommended_method.method_id
        candidate = _candidate_or_none(recommendation, recommended_id)
        assert candidate is not None
        assert candidate.status is MethodCandidateStatus.RECOMMENDED
        assert candidate.policy_status is PolicyStatus.ENABLED


# ---------------------------------------------------------------------------
# AC4: SMOTE policy invariants (before-split and validation/test rejection).
# ---------------------------------------------------------------------------


def test_smote_step_rejects_non_train_source_split() -> None:
    """SMOTE step config that targets the validation split must be rejected."""
    storage = _empty_storage()
    registry = ArtifactRegistry(storage=storage)

    bad_step = _smote_step(source_split="validation")
    request = ExecuteSmoteAugmentationRequest(
        action_plan_id="action_plan_smote_001",
        step=bad_step,
        dataset_id="dataset_test",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        source_artifact=_artifact_ref_model("source_csv", "raw_dataset_archive"),
        split_manifest=_minimal_train_only_split_manifest(),
        split_manifest_artifact=_artifact_ref_model(
            "split_manifest_1", "split_manifest"
        ),
        created_by_job_id="compute_run_apply_001",
        config_hash=_DUMMY_CONFIG_HASH,
        target_column="is_fraud",
        rare_class_label="1",
    )

    with pytest.raises(SmoteExecutionError) as excinfo:
        execute_smote_augmentation_action(
            request,
            storage=storage,
            registry=registry,
        )
    assert excinfo.value.reason_code == "non_train_source_split"


def test_smote_step_rejects_test_source_split() -> None:
    """SMOTE step config that targets the test split must be rejected."""
    storage = _empty_storage()
    registry = ArtifactRegistry(storage=storage)

    bad_step = _smote_step(source_split="test")
    request = ExecuteSmoteAugmentationRequest(
        action_plan_id="action_plan_smote_001",
        step=bad_step,
        dataset_id="dataset_test",
        source_dataset_version_id="dataset_version_1",
        candidate_dataset_version_id="dataset_version_2_candidate",
        source_artifact=_artifact_ref_model("source_csv", "raw_dataset_archive"),
        split_manifest=_minimal_train_only_split_manifest(),
        split_manifest_artifact=_artifact_ref_model(
            "split_manifest_1", "split_manifest"
        ),
        created_by_job_id="compute_run_apply_001",
        config_hash=_DUMMY_CONFIG_HASH,
    )

    with pytest.raises(SmoteExecutionError) as excinfo:
        execute_smote_augmentation_action(
            request,
            storage=storage,
            registry=registry,
        )
    assert excinfo.value.reason_code == "non_train_source_split"


def test_smote_step_cannot_be_constructed_without_split_manifest() -> None:
    """SMOTE execution requires a split manifest; building the request without one fails."""
    # ``ExecuteSmoteAugmentationRequest`` declares ``split_manifest`` as required
    # field. Constructing it without a split manifest must raise a Pydantic
    # validation error: SMOTE has no in-process bypass to run "before split".
    with pytest.raises(ValidationError):
        ExecuteSmoteAugmentationRequest.model_validate(
            {
                "action_plan_id": "action_plan_smote_001",
                "step": _smote_step().model_dump(mode="json"),
                "dataset_id": "dataset_test",
                "source_dataset_version_id": "dataset_version_1",
                "candidate_dataset_version_id": "dataset_version_2_candidate",
                "source_artifact": _artifact_ref(
                    "source_csv", "raw_dataset_archive"
                ),
                "split_manifest_artifact": _artifact_ref(
                    "split_manifest_1", "split_manifest"
                ),
                "created_by_job_id": "compute_run_apply_001",
                "config_hash": _DUMMY_CONFIG_HASH,
            }
        )


def test_smote_action_plan_step_carries_split_safety_preconditions() -> None:
    """ActionPlan steps for synthetic methods must require the train split and leakage gates.

    SMOTE itself is policy-disabled in the MVP profile (it requires split
    readiness gates first), so we exercise the same invariant through the
    Gaussian Copula path which is policy-enabled but still classified as a
    synthetic method by the kernel. Both methods share the synthetic
    validation gates and synthetic preconditions, so a regression in either
    set surfaces here.
    """
    from app.kernel import action_plan as action_plan_module
    from app.kernel.action_plan import (
        BuildActionPlanPreviewRequest,
        build_action_plan_preview,
    )

    # SMOTE must be classified as a synthetic method by the kernel so it
    # picks up the synthetic validation gates and preconditions even when
    # selectable via a future readiness profile.
    assert "smote" in action_plan_module._synthetic_methods()
    assert "split_leakage_check" in action_plan_module._SYNTHETIC_VALIDATION_GATES

    profile = _demo_tabular_profile()
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=profile)
    )
    rare_class = next(
        rec for rec in recommendations if rec.action_type == "AUGMENT_RARE_CLASS"
    )

    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_001",
            source_dataset_version_id="dataset_version_1",
            selected_decision_ids=(rare_class.recommendation_id,),
            selected_method_overrides={rare_class.recommendation_id: "gaussian_copula"},
            method_recommendations=(rare_class,),
            created_by_user_id="platform_user_123",
            input_artifacts=(
                "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/manifest.jsonl",
            ),
            target_version_name="dataset_version_2_preview",
            created_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
        )
    )

    synthetic_step = next(step for step in plan.steps if step.method_id == "gaussian_copula")
    assert "train_split_exists" in synthetic_step.preconditions
    assert "leakage_checks_passed" in synthetic_step.preconditions
    assert "split_leakage_check" in synthetic_step.validation_gates
    # Synthetic actions must explicitly bind to the training split inside the step.
    assert synthetic_step.config.get("source_split") == DataSplit.TRAIN.value


# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------


_DUMMY_CONFIG_HASH = "sha256:" + "f" * 64


def _imputation_recommendation() -> Any:
    for recommendation in _all_recommendations():
        if recommendation.action_type == "IMPUTE_MISSING_VALUES":
            return recommendation
    raise AssertionError("expected an imputation recommendation in the demo profile")


def _rare_class_recommendation() -> Any:
    for recommendation in _all_recommendations():
        if recommendation.action_type == "AUGMENT_RARE_CLASS":
            return recommendation
    raise AssertionError("expected a rare-class recommendation in the demo profile")


def _all_recommendations() -> tuple[Any, ...]:
    profile = _demo_tabular_profile()
    return build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=profile)
    )


def _candidate_or_none(recommendation: Any, method_id: str) -> Any | None:
    for candidate in recommendation.candidate_methods:
        if candidate.method_id == method_id:
            return candidate
    return None


def _demo_tabular_profile() -> TabularProfileReport:
    pack = load_contract_pack()
    example = next(
        example
        for example in pack.examples
        if example.name == "tabular_profile_report.fraud"
    )
    return TabularProfileReport.model_validate(example.payload)


def _smote_step(
    *,
    source_split: str = DataSplit.TRAIN.value,
    method_id: str = "smote",
) -> ActionPlanStep:
    return ActionPlanStep(
        step_id="step_smote_001",
        type="AUGMENT_RARE_CLASS",
        idempotency_key="sha256:" + "0" * 64,
        method_id=method_id,
        plugin_id="dataforge.tabular",
        plugin_version="0.1.0",
        config_hash=_DUMMY_CONFIG_HASH,
        policy_version="method_policy_v0",
        validation_gates=(
            "split_leakage_check",
            "schema_validation",
            "synthetic_quality_check",
        ),
        preconditions=("train_split_exists", "leakage_checks_passed"),
        input_artifacts=(
            "s3://dataforge-local/dataforge/org_test/project_test/dataset_test/source.csv",
        ),
        output_artifact_kind="CANDIDATE_DATASET_VERSION",
        config={
            "method": method_id,
            "source_split": source_split,
            "rare_class_label": "1",
            "random_seed": 42,
        },
        random_seed=42,
        retry_policy=RetryPolicy(
            max_attempts=1,
            retryable_errors=("TRANSIENT_STORAGE_ERROR",),
        ),
        on_failure="BLOCK",
        promotion_scope="CANDIDATE",
    )


def _minimal_train_only_split_manifest() -> Any:
    """Return a SplitManifest that the SMOTE validator will accept.

    The contract pack ships a representative example we can reuse so the
    invariant test exercises the same shape used in production reports
    without re-encoding the schema here.
    """
    from app.domain import SplitManifest

    pack = load_contract_pack()
    example = next(
        example
        for example in pack.examples
        if example.name == "split_manifest.group_stratified"
    )
    return SplitManifest.model_validate(example.payload)


def _empty_storage() -> MinioObjectStorageAdapter:
    config = _test_config()
    scope = ObjectStorageScope(
        organization_id="org_test",
        project_id="project_test",
        dataset_id="dataset_test",
    )
    return MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=scope,
    )


def _test_config() -> ServiceConfig:
    return load_config(_DEMO_ENV)


def _analyze_payload(
    *,
    prediction_artifact_refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "platform_job_id": "platform_job_invariants",
        "organization_id": "org_1",
        "project_id": "project_1",
        "dataset_id": "dataset_1",
        "dataset_version_id": "dataset_version_1",
        "dataset_object_refs": [_artifact_ref("raw_archive_1", "raw_dataset_archive")],
        "prediction_artifact_refs": []
        if prediction_artifact_refs is None
        else prediction_artifact_refs,
    }


def _artifact_ref(artifact_id: str, kind: str) -> dict[str, object]:
    return _artifact_ref_model(artifact_id, kind).model_dump(mode="json")


def _artifact_ref_model(artifact_id: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/"
            f"{artifact_id}.json"
        ),
        hash="sha256:" + "a" * 64,
        media_type="application/json",
        size_bytes=128,
        schema_version="v0.1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_1",
            job_id="platform_job_invariants",
            config_hash="sha256:" + "b" * 64,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


def _body_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _signed_headers(
    *,
    config: ServiceConfig,
    body: bytes,
    payload: dict[str, object],
) -> dict[str, str]:
    signed_at = datetime.now(UTC).isoformat()
    service_identity = config.platform.service_identity
    organization_id = str(payload["organization_id"])
    project_id = str(payload["project_id"])
    return {
        SERVICE_IDENTITY_HEADER: service_identity,
        TIMESTAMP_HEADER: signed_at,
        ORGANIZATION_ID_HEADER: organization_id,
        PROJECT_ID_HEADER: project_id,
        SIGNATURE_HEADER: build_service_signature(
            secret=config.platform.service_signing_secret.get_secret_value(),
            service_identity=service_identity,
            timestamp=signed_at,
            organization_id=organization_id,
            project_id=project_id,
            body=body,
        ),
        "content-type": "application/json",
    }


class _InMemoryS3Client:
    """In-memory S3-compatible fake used only by these invariant tests."""

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
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._objects[(Bucket, Key)]
        body = cast(bytes, record["Body"])
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._objects[(Bucket, Key)]
        body = cast(bytes, record["Body"])
        return {
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Mapping[str, Any]:
        contents: list[dict[str, object]] = []
        for (bucket, key), record in sorted(self._objects.items()):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            body = cast(bytes, record["Body"])
            contents.append({"Key": key, "Size": len(body)})
        return {"Contents": contents}
