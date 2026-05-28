"""Future ANALYZE_ONLY orchestration E2E smoke.

The product-contract E2E in ``test_compute_demo_analyze_only.py`` exercises
the real builders. This test takes the complementary route: it invokes the
Dagster-backed ``launch_analyze_dataset_workflow`` path directly so runtime
selection, platform status events, prediction-aware asset wiring and
no-mutation guarantees are covered by an E2E-style test.
"""

from __future__ import annotations

from pathlib import Path

from app.adapters import FakePlatformMetadataClient
from app.api.schemas import AnalyzeDatasetRequest
from app.domain import ComputeRunStatus
from app.orchestration.analyze_workflow import (
    expected_analyze_outputs,
    launch_analyze_dataset_workflow,
)
from app.orchestration.apply_assets import APPLY_ASSET_KEYS
from tests.fixtures.demo_archive import build_demo_archive
from tools.run_mvp_demo import _demo_config, _resources_with_source


def test_analyze_only_orchestration_e2e_with_predictions(tmp_path: Path) -> None:
    """Dagster ANALYZE_ONLY launcher publishes prediction-aware artifact refs."""
    config = _demo_config(tmp_path)
    fake_platform = FakePlatformMetadataClient()
    archive = build_demo_archive(output_dir=tmp_path / "demo_archive")
    resources, source_archive, _source_transactions, prediction_artifact = (
        _resources_with_source(
            config=config,
            fake_platform=fake_platform,
            archive_path=archive.archive_path,
        )
    )
    request = AnalyzeDatasetRequest(
        platform_job_id="platform_job_future_analyze_e2e",
        organization_id="org_demo",
        project_id="project_demo",
        dataset_id="dataset_demo",
        dataset_version_id="dataset_version_v1",
        dataset_object_refs=(source_archive,),
        prediction_artifact_refs=(prediction_artifact,),
    )

    result = launch_analyze_dataset_workflow(
        request=request,
        config=config,
        fake_platform=fake_platform,
        compute_resources=resources,
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
    assert set(result.artifact_uris) == set(expected)
    assert result.artifact_uris["model_error_analysis_report"].startswith(
        "s3://dataforge-local/"
    )

    snapshot = fake_platform.snapshot()
    assert snapshot.job_events[-1].status is ComputeRunStatus.COMPLETED
    assert all(event.details.get("dataset_id") == "dataset_demo" for event in snapshot.job_events)
    assert all(
        event.status is not ComputeRunStatus.COMPLETED
        or len(event.details.get("artifact_uris") or []) > 0
        for event in snapshot.job_events
    )
