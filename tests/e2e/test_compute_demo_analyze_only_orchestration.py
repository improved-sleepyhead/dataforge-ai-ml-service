"""Future ANALYZE_ONLY orchestration E2E smoke.

The product-contract E2E in ``test_compute_demo_analyze_only.py`` exercises
the real builders. This test takes the complementary route: it invokes the
Dagster-backed ``launch_analyze_dataset_workflow`` path directly so runtime
selection, platform status events, prediction-aware asset wiring and
no-mutation guarantees are covered by an E2E-style test.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.adapters import FakePlatformMetadataClient
from app.api.schemas import AnalyzeDatasetRequest
from app.domain import ArtifactLineage, ArtifactRef, ComputeRunStatus
from app.kernel.config import load_config
from app.orchestration.analyze_workflow import (
    expected_analyze_outputs,
    launch_analyze_dataset_workflow,
)
from app.orchestration.apply_assets import APPLY_ASSET_KEYS

_GENERATED_AT = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)


def test_future_analyze_only_orchestration_e2e_with_predictions() -> None:
    """Dagster ANALYZE_ONLY launcher wires prediction assets and no APPLY outputs."""
    config = load_config(
        {
            "DATAFORGE_PROFILE": "demo_strict",
            "DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL": "http://localhost:9000",
            "DATAFORGE_OBJECT_STORAGE_BUCKET": "dataforge-local",
            "DATAFORGE_PLATFORM_CALLBACK_URL": (
                "http://platform.local/api/ml/jobs/callback"
            ),
            "DATAFORGE_SERVICE_SIGNING_SECRET": "local-dev-signing-secret",
            "DATAFORGE_DAGSTER_HOME": "/tmp/dataforge-dagster",
            "DATAFORGE_POLICY_CONFIG_PATH": "configs/policies/demo_strict.yaml",
            "DATAFORGE_DECISION_POLICY_PATH": "configs/policies/decision_v0.yaml",
            "DATAFORGE_SCORE_POLICY_PATH": "configs/policies/score_v0.yaml",
        }
    )
    fake_platform = FakePlatformMetadataClient()
    request = AnalyzeDatasetRequest(
        platform_job_id="platform_job_future_analyze_e2e",
        organization_id="org_1",
        project_id="project_1",
        dataset_id="dataset_1",
        dataset_version_id="dataset_version_v1",
        dataset_object_refs=(
            _artifact_ref("raw_archive_1", "raw_dataset_archive", "a"),
        ),
        prediction_artifact_refs=(
            _artifact_ref("prediction_manifest_1", "prediction_manifest", "b"),
        ),
    )

    result = launch_analyze_dataset_workflow(
        request=request,
        config=config,
        fake_platform=fake_platform,
    )

    expected = expected_analyze_outputs(include_predictions=True)
    apply_asset_names = {key.path[-1] for key in APPLY_ASSET_KEYS}
    assert result.status is ComputeRunStatus.ACCEPTED
    assert result.mutates_dataset is False
    assert result.expected_outputs == expected
    assert set(result.materialized_assets) == set(expected)
    assert result.idempotency_key.startswith("sha256:")
    assert {
        "prediction_manifest",
        "prediction_validation_report",
        "model_error_analysis_report",
        "ambiguous_object_candidates",
        "probable_label_error_candidates",
    }.issubset(result.materialized_assets)
    assert apply_asset_names.isdisjoint(result.materialized_assets)

    snapshot = fake_platform.snapshot()
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED
    assert all(event.details.get("dataset_id") == "dataset_1" for event in snapshot.job_events)
    assert all(
        not event.details.get("artifact_uris")
        for event in snapshot.job_events
    )


def _artifact_ref(artifact_id: str, kind: str, hash_seed: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=(
            "s3://dataforge-local/dataforge/org_1/project_1/dataset_1/"
            f"{artifact_id}.json"
        ),
        hash="sha256:" + hash_seed * 64,
        media_type="application/json",
        size_bytes=128,
        schema_version="v0.1",
        lineage=ArtifactLineage(
            parent_version_id="dataset_version_v1",
            job_id="platform_job_future_analyze_e2e",
            config_hash="sha256:" + "0" * 64,
            created_at=_GENERATED_AT,
        ),
    )
