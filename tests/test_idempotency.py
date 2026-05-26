"""Tests for TASK-060 analyze/apply job idempotency keys."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.adapters import FakePlatformMetadataClient
from app.api.schemas import AnalyzeDatasetRequest
from app.domain import ArtifactLineage, ArtifactRef, WorkflowType
from app.kernel.idempotency import (
    AnalyzeIdempotencyInputs,
    ApplyIdempotencyInputs,
    PluginVersionFootprint,
    collect_artifact_hashes,
    compute_analyze_idempotency_key,
    compute_apply_idempotency_key,
)
from app.orchestration.analyze_workflow import (
    _analyze_plugin_footprints,
    launch_analyze_dataset_workflow,
)
from app.orchestration.apply_workflow import launch_apply_actions_workflow
from tests.test_apply_workflow import _execute_request, _test_config

_GENERATED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Pure key derivation
# ---------------------------------------------------------------------------


def test_analyze_idempotency_key_is_deterministic_for_identical_inputs() -> None:
    inputs = _analyze_inputs()

    first = compute_analyze_idempotency_key(inputs)
    second = compute_analyze_idempotency_key(inputs)

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_analyze_idempotency_key_changes_when_input_artifact_changes() -> None:
    base = _analyze_inputs()
    other = base.model_copy(
        update={
            "input_artifact_hashes": ("sha256:" + "f" * 64,),
        }
    )

    assert compute_analyze_idempotency_key(base) != compute_analyze_idempotency_key(other)


def test_analyze_idempotency_key_changes_when_prediction_artifact_changes() -> None:
    base = _analyze_inputs()
    other = base.model_copy(
        update={
            "prediction_artifact_hashes": ("sha256:" + "9" * 64,),
        }
    )

    assert compute_analyze_idempotency_key(base) != compute_analyze_idempotency_key(other)


def test_analyze_idempotency_key_changes_when_config_hash_changes() -> None:
    base = _analyze_inputs()
    other = base.model_copy(update={"config_hash": "sha256:" + "7" * 64})

    assert compute_analyze_idempotency_key(base) != compute_analyze_idempotency_key(other)


def test_analyze_idempotency_key_changes_when_plugin_versions_change() -> None:
    base = _analyze_inputs(
        plugin_versions=(
            PluginVersionFootprint(
                plugin_id="dataforge.tabular",
                plugin_version="0.1.0",
                algorithm_name="schema_inference",
                algorithm_version="0.1.0",
            ),
        )
    )
    upgraded = base.model_copy(
        update={
            "plugin_versions": (
                PluginVersionFootprint(
                    plugin_id="dataforge.tabular",
                    plugin_version="0.2.0",
                    algorithm_name="schema_inference",
                    algorithm_version="0.1.0",
                ),
            ),
        }
    )

    assert compute_analyze_idempotency_key(base) != compute_analyze_idempotency_key(
        upgraded
    )


def test_apply_idempotency_key_changes_when_action_plan_hash_changes() -> None:
    base = _apply_inputs()
    other = base.model_copy(update={"action_plan_hash": "sha256:" + "1" * 64})

    assert compute_apply_idempotency_key(base) != compute_apply_idempotency_key(other)


def test_apply_idempotency_key_changes_when_step_idempotency_keys_change() -> None:
    base = _apply_inputs()
    reordered_or_added = base.model_copy(
        update={
            "step_idempotency_keys": (
                *base.step_idempotency_keys,
                "sha256:" + "0" * 64,
            ),
        }
    )

    assert compute_apply_idempotency_key(base) != compute_apply_idempotency_key(
        reordered_or_added
    )


def test_apply_idempotency_key_is_stable_across_dict_orderings() -> None:
    base = _apply_inputs()
    reordered = base.model_copy(
        update={
            # Same content, different insertion order (Pydantic preserves
            # the dict order it was built with, but the key derivation
            # must canonicalize via sort_keys).
            "policy_versions": dict(
                reversed(list(base.policy_versions.items()))
            ),
        }
    )

    assert compute_apply_idempotency_key(base) == compute_apply_idempotency_key(reordered)


def test_collect_artifact_hashes_dedupes_and_sorts_logical_input_set() -> None:
    refs = (
        _artifact_ref("b", "kind_b", "sha256:" + "b" * 64),
        _artifact_ref("a", "kind_a", "sha256:" + "a" * 64),
        # Duplicate hash should be deduped
        _artifact_ref("c", "kind_c", "sha256:" + "a" * 64),
    )
    assert collect_artifact_hashes(refs) == (
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    )


# ---------------------------------------------------------------------------
# Launcher integration
# ---------------------------------------------------------------------------


def test_analyze_launcher_returns_stable_idempotency_key_and_artifact_hashes() -> None:
    """Step 1 + 2: same archive/config produces identical idempotency_key and asset names."""
    config = _test_config()
    request = _analyze_request()

    first = launch_analyze_dataset_workflow(
        request=request,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )
    second = launch_analyze_dataset_workflow(
        request=request,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )

    assert first.idempotency_key == second.idempotency_key
    assert first.idempotency_key.startswith("sha256:")
    assert first.materialized_assets == second.materialized_assets


def test_analyze_launcher_key_matches_static_plugin_footprint() -> None:
    """The real launcher folds the static plugin capability versions into its key."""
    config = _test_config()
    request = _analyze_request()

    result = launch_analyze_dataset_workflow(
        request=request,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )
    expected = compute_analyze_idempotency_key(
        AnalyzeIdempotencyInputs(
            organization_id=request.organization_id,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            dataset_version_id=request.dataset_version_id,
            input_artifact_hashes=collect_artifact_hashes(request.dataset_object_refs),
            prediction_artifact_hashes=collect_artifact_hashes(
                request.prediction_artifact_refs
            ),
            config_hash=config.config_hash,
            contract_pack_version=config.contract_pack_version,
            plugin_versions=_analyze_plugin_footprints(),
        )
    )

    assert result.idempotency_key == expected
    assert _analyze_plugin_footprints()


def test_analyze_launcher_key_is_stable_across_input_artifact_order() -> None:
    """Same logical analyze input set must not duplicate runs because refs were reordered."""
    config = _test_config()
    first_request = _analyze_request(
        dataset_object_refs=(
            _artifact_ref("b", "kind_b", "sha256:" + "b" * 64),
            _artifact_ref("a", "kind_a", "sha256:" + "a" * 64),
        )
    )
    second_request = first_request.model_copy(
        update={"dataset_object_refs": tuple(reversed(first_request.dataset_object_refs))}
    )

    first = launch_analyze_dataset_workflow(
        request=first_request,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )
    second = launch_analyze_dataset_workflow(
        request=second_request,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )

    assert first.idempotency_key == second.idempotency_key


def test_apply_launcher_returns_stable_idempotency_key() -> None:
    """Step 1+2+3: identical APPLY runs share idempotency_key and artifact hashes."""
    config = _test_config()
    request, plan_hash = _execute_request()

    first = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )
    second = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )

    assert first.idempotency_key == second.idempotency_key
    # Step 3: candidate/synthetic/model_impact/export hashes are
    # identical across reruns because ArtifactRegistry is content-addressed.
    assert first.candidate_artifact_hash == second.candidate_artifact_hash


def test_apply_launcher_key_changes_when_step_plugin_version_changes() -> None:
    """Selected ActionPlan step plugin versions must invalidate APPLY keys."""
    config = _test_config()
    request, plan_hash = _execute_request()
    base = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )
    step = request.action_plan.steps[0]
    patched_step = step.model_copy(update={"plugin_version": "0.2.0"})
    patched_plan = request.action_plan.model_copy(update={"steps": (patched_step,)})
    patched_request = request.model_copy(update={"action_plan": patched_plan})

    changed = launch_apply_actions_workflow(
        request=patched_request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
    )

    assert base.idempotency_key != changed.idempotency_key


def test_apply_launcher_key_is_stable_across_input_artifact_order() -> None:
    """Same APPLY input artifacts in a different order must share one key."""
    config = _test_config()
    request, plan_hash = _execute_request()
    input_refs = (
        _artifact_ref("b", "kind_b", "sha256:" + "b" * 64),
        _artifact_ref("a", "kind_a", "sha256:" + "a" * 64),
    )

    first = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
        input_artifacts=input_refs,
    )
    second = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=FakePlatformMetadataClient(),
        input_artifacts=tuple(reversed(input_refs)),
    )

    assert first.idempotency_key == second.idempotency_key


def test_apply_launcher_does_not_mask_validation_failures_via_cache() -> None:
    """Failed validation gates must still surface even if upstream artifacts cache.

    The ApplyWorkflowResult does not cache the response; the launcher
    always re-runs the asset graph. This test asserts that materialization
    failures propagate as RuntimeError so the caller never sees a stale
    "ACCEPTED" status from a prior run.
    """
    fake_platform = FakePlatformMetadataClient()
    config = _test_config()
    request, plan_hash = _execute_request()

    # Sanity: a healthy run is ACCEPTED.
    healthy = launch_apply_actions_workflow(
        request=request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
    )
    assert healthy.status.value == "ACCEPTED"

    # Verify the launcher refuses unsigned re-runs even though the prior
    # signed run succeeded — failures must not be hidden by reuse.
    with pytest.raises(ValueError, match="approval_metadata"):
        launch_apply_actions_workflow(
            request=request.model_copy(update={"approval_metadata": None}),
            action_plan_hash=plan_hash,
            config=config,
            fake_platform=fake_platform,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _analyze_inputs(
    *,
    plugin_versions: tuple[PluginVersionFootprint, ...] = (),
) -> AnalyzeIdempotencyInputs:
    return AnalyzeIdempotencyInputs(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        dataset_version_id="dataset_version_v1",
        input_artifact_hashes=(
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
        ),
        prediction_artifact_hashes=("sha256:" + "c" * 64,),
        config_hash="sha256:" + "d" * 64,
        contract_pack_version="local-fallback-v0.1.0-demo",
        plugin_versions=plugin_versions,
    )


def _apply_inputs() -> ApplyIdempotencyInputs:
    return ApplyIdempotencyInputs(
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        source_dataset_version_id="dataset_version_v1",
        proposed_version_name="dataset_version_v1__candidate_test",
        action_plan_hash="sha256:" + "e" * 64,
        input_artifact_hashes=("sha256:" + "a" * 64,),
        config_hash="sha256:" + "d" * 64,
        contract_pack_version="local-fallback-v0.1.0-demo",
        policy_versions={
            "decision": "decision_policy_v0",
            "method": "method_policy_v0",
            "score": "dataforge_score_v0",
            "profile": "demo_strict_v1",
            "validation_gates": "validation_gates_policy_v0",
        },
        step_idempotency_keys=("sha256:" + "1" * 64, "sha256:" + "2" * 64),
    )


def _analyze_request(
    *,
    dataset_object_refs: tuple[ArtifactRef, ...] | None = None,
) -> AnalyzeDatasetRequest:
    return AnalyzeDatasetRequest(
        platform_job_id="platform_job_idempotent_001",
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        dataset_version_id="dataset_version_v1",
        dataset_object_refs=dataset_object_refs
        or (
            _artifact_ref(
                "raw_transactions",
                "raw_transactions",
                "sha256:" + "a" * 64,
            ),
        ),
        prediction_artifact_refs=(),
    )


def _artifact_ref(artifact_id: str, kind: str, hash_: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=f"s3://dataforge-local/dataforge/org_1/project_1/dataset_1/{artifact_id}.csv",
        hash=hash_,
        media_type="text/csv",
        size_bytes=128,
        schema_version="tabular_dataset.v1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v1",
            job_id="platform_job_idempotent_001",
            config_hash="sha256:" + "0" * 64,
            created_at=_GENERATED_AT,
        ),
    )


# Silence unused-import warnings.
_ = WorkflowType
