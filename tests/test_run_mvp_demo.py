"""Smoke tests for the operator-facing MVP demo runner."""

from __future__ import annotations

from pathlib import Path

from app.adapters import FakePlatformMetadataClient
from app.api.schemas import AnalyzeDatasetRequest
from app.domain import ComputeRunStatus
from app.orchestration.analyze_workflow import (
    expected_analyze_outputs,
    launch_analyze_dataset_workflow,
)
from app.orchestration.apply_workflow import launch_apply_actions_workflow
from tests.fixtures.demo_archive import build_demo_archive
from tools.run_mvp_demo import (
    _action_plan_request,
    _demo_config,
    _parse_args,
    _resources_with_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_args_defaults() -> None:
    args = _parse_args([])
    assert args.workdir is None
    assert args.skip_fastapi is False


def test_parse_args_skip_fastapi_flag() -> None:
    args = _parse_args(["--skip-fastapi"])
    assert args.skip_fastapi is True


def test_parse_args_workdir_override(tmp_path: Path) -> None:
    args = _parse_args(["--workdir", str(tmp_path)])
    assert args.workdir == str(tmp_path)


def test_demo_config_uses_demo_strict_profile_and_no_external_ai(tmp_path: Path) -> None:
    config = _demo_config(tmp_path)
    assert config.profile.value == "demo_strict"
    assert config.contract_pack_version == "local-fallback-v0.1.0-demo"
    assert config.external_ai.allow_external_api is False


def test_full_flow_helpers_drive_real_dagster_materialization(tmp_path: Path) -> None:
    """The demo helpers must drive a successful ANALYZE + APPLY end-to-end."""
    config = _demo_config(tmp_path)
    archive = build_demo_archive(output_dir=tmp_path / "demo_archive")
    fake_platform = FakePlatformMetadataClient()
    resources, source_archive, source_artifact, prediction_artifact = _resources_with_source(
        config=config,
        fake_platform=fake_platform,
        archive_path=archive.archive_path,
    )

    analyze_request = AnalyzeDatasetRequest(
        platform_job_id="platform_job_demo_smoke_analyze",
        organization_id="org_demo",
        project_id="project_demo",
        dataset_id="dataset_demo",
        dataset_version_id="dataset_version_v1",
        dataset_object_refs=(source_archive,),
        prediction_artifact_refs=(prediction_artifact,),
    )
    analyze_result = launch_analyze_dataset_workflow(
        request=analyze_request,
        config=config,
        fake_platform=fake_platform,
        compute_resources=resources,
    )
    assert analyze_result.status is ComputeRunStatus.ACCEPTED
    assert set(analyze_result.materialized_assets) == set(
        expected_analyze_outputs(include_predictions=True)
    )
    assert "model_error_analysis_report" in analyze_result.artifact_uris

    apply_request, plan_hash = _action_plan_request(source_artifact=source_artifact)
    apply_result = launch_apply_actions_workflow(
        request=apply_request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=apply_request.source_artifacts,
        compute_resources=resources,
    )
    assert apply_result.status is ComputeRunStatus.ACCEPTED
    assert apply_result.candidate_artifact_uri is not None
    assert apply_result.export_package_artifact_uri is not None
    assert {span.name for span in resources.tracing.snapshot()}.issuperset(
        {"ingestion", "prediction.validate", "model_error.analyze", "export.build"}
    )


def test_makefile_exposes_run_mvp_demo_target() -> None:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "run-mvp-demo:" in text
    assert "tools.run_mvp_demo" in text
