"""Run the DataForge AI ML service MVP demo end-to-end.

The script wires the public launchers + builders into a real user
scenario on the deterministic demo archive, materializes both Dagster
asset graphs (ANALYZE_ONLY then APPLY_SELECTED_ACTIONS), and prints
per-stage progress plus a final artifact summary. It uses the same
in-memory storage fake and FakePlatformMetadataClient that
``tests/e2e/test_compute_demo_full_flow.py`` uses, so the demo runs
without a real backend, real MinIO, real Vault, or external AI access.

Optionally boots the FastAPI compute API in a background subprocess
and probes ``/api/v1/health`` to prove the wire-level endpoint comes
up against the same configuration.

Usage:

    .venv/bin/python -m tools.run_mvp_demo
    .venv/bin/python -m tools.run_mvp_demo --skip-fastapi
    .venv/bin/python -m tools.run_mvp_demo --workdir /tmp/dataforge-demo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError, S3CompatibleClient
from app.api.schemas import (
    ActionPlanExecuteApprovedRequest,
    AnalyzeDatasetRequest,
)
from app.domain import (
    ArtifactRef,
    ErrorCode,
    ExportPackage,
    ExportPackageStatus,
    ModelImpactReport,
    TabularProfileReport,
)
from app.ingestion import open_archive_path
from app.kernel import (
    BuildActionPlanPreviewRequest,
    BuildMethodRecommendationsRequest,
    action_plan_integrity_hash,
    build_action_plan_preview,
    build_method_recommendations,
)
from app.kernel.action_plan import ActionPlanApprovalMetadata
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
from app.orchestration.analyze_workflow import (
    expected_analyze_outputs,
    launch_analyze_dataset_workflow,
)
from app.orchestration.apply_assets import APPLY_ASSET_KEYS
from app.orchestration.apply_workflow import launch_apply_actions_workflow
from app.orchestration.resources import ComputeResources
from app.validation.contracts import load_contract_pack
from tests.fixtures.demo_archive import build_demo_archive

REPO_ROOT = Path(__file__).resolve().parents[1]

_GENERATED_AT = datetime.now(UTC)
_ORG_ID = "org_demo"
_PROJECT_ID = "project_demo"
_DATASET_ID = "dataset_demo"
_PARENT_VERSION_ID = "dataset_version_v1"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    workdir = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="dataforge-mvp-"))
    )
    workdir.mkdir(parents=True, exist_ok=True)
    _print_banner("DataForge AI MVP demo runner")
    print(f"  workdir            : {workdir}")
    print("  contract pack      : local-fallback-v0.1.0-demo")
    print("  profile            : demo_strict")
    print("  external AI access : disabled")

    config = _demo_config(workdir)

    # 1) Optional FastAPI smoke.
    if not args.skip_fastapi:
        _print_step("FastAPI compute API smoke")
        with _fastapi_app(config=config) as base_url:
            _probe_health(base_url)
            _probe_capabilities(base_url)

    # 2) Build deterministic demo archive + load source artifact into scoped storage.
    _print_step("Build deterministic demo archive")
    archive = build_demo_archive(output_dir=workdir / "demo_archive")
    print(f"  archive            : {archive.archive_path}")
    print(f"  archive_sha256     : {archive.archive_sha256}")
    print(f"  expected counts    : {archive.expected_counts_path}")

    fake_platform = FakePlatformMetadataClient()
    resources, source_archive, source_artifact, prediction_artifact = _resources_with_source(
        config=config,
        fake_platform=fake_platform,
        archive_path=archive.archive_path,
    )
    print(f"  archive artifact   : {source_archive.uri}")
    print(f"  archive hash       : {source_archive.hash}")
    print(f"  tabular artifact   : {source_artifact.uri}")
    print(f"  tabular hash       : {source_artifact.hash}")
    print(f"  prediction artifact: {prediction_artifact.uri}")

    # 3) Real Dagster ANALYZE_ONLY materialization.
    _print_step("ANALYZE_ONLY (real Dagster materialization)")
    analyze_request = AnalyzeDatasetRequest(
        platform_job_id="platform_job_demo_analyze",
        organization_id=_ORG_ID,
        project_id=_PROJECT_ID,
        dataset_id=_DATASET_ID,
        dataset_version_id=_PARENT_VERSION_ID,
        dataset_object_refs=(source_archive,),
        prediction_artifact_refs=(prediction_artifact,),
    )
    analyze_started = time.perf_counter()
    analyze_result = launch_analyze_dataset_workflow(
        request=analyze_request,
        config=config,
        fake_platform=fake_platform,
    )
    analyze_duration = time.perf_counter() - analyze_started
    print(f"  status             : {analyze_result.status.value}")
    print(f"  materialized       : {len(analyze_result.materialized_assets)} assets")
    for name in analyze_result.materialized_assets:
        print(f"    - {name}")
    print(f"  duration           : {analyze_duration:.2f}s")
    print(f"  mutates_dataset    : {analyze_result.mutates_dataset}")
    apply_only_assets = {key.path[-1] for key in APPLY_ASSET_KEYS}
    leaked = apply_only_assets.intersection(analyze_result.materialized_assets)
    if leaked:
        raise SystemExit(
            f"ANALYZE_ONLY materialized apply-only assets {sorted(leaked)} — drift detected"
        )
    expected = expected_analyze_outputs(include_predictions=True)
    missing = set(expected) - set(analyze_result.materialized_assets)
    if missing:
        raise SystemExit(
            f"ANALYZE_ONLY skipped expected assets {sorted(missing)} — drift detected"
        )
    required_artifacts = {
        "prediction_manifest",
        "prediction_validation_report",
        "model_error_analysis_report",
        "ambiguous_object_candidates",
        "probable_label_error_candidates",
        "decision_report",
        "review_queue",
    }
    missing_artifacts = required_artifacts - set(analyze_result.artifact_uris)
    if missing_artifacts:
        raise SystemExit(
            "ANALYZE_ONLY did not publish required artifact refs "
            f"{sorted(missing_artifacts)}"
        )

    # 4) Build approved ActionPlan from kernel builders.
    _print_step("Build approved ActionPlan from method recommendations")
    apply_request, plan_hash = _action_plan_request(source_artifact=source_artifact)
    print(f"  action_plan_id     : {apply_request.action_plan.action_plan_id}")
    print(f"  action_plan_hash   : {plan_hash}")
    print(f"  steps              : {len(apply_request.action_plan.steps)}")
    for step in apply_request.action_plan.steps:
        print(
            f"    - step {step.step_id}: {step.type} via {step.method_id} "
            f"(plugin={step.plugin_id}@{step.plugin_version})"
        )

    # 5) Real Dagster APPLY_SELECTED_ACTIONS materialization.
    _print_step("APPLY_SELECTED_ACTIONS (real Dagster materialization)")
    apply_started = time.perf_counter()
    apply_result = launch_apply_actions_workflow(
        request=apply_request,
        action_plan_hash=plan_hash,
        config=config,
        fake_platform=fake_platform,
        input_artifacts=apply_request.source_artifacts,
        compute_resources=resources,
    )
    apply_duration = time.perf_counter() - apply_started
    print(f"  status             : {apply_result.status.value}")
    print(f"  materialized       : {len(apply_result.materialized_assets)} assets")
    for name in apply_result.materialized_assets:
        print(f"    - {name}")
    print(f"  duration           : {apply_duration:.2f}s")
    print(f"  mutates_dataset    : {apply_result.mutates_dataset}")
    print(f"  candidate_artifact : {apply_result.candidate_artifact_uri}")
    print(f"  candidate_hash     : {apply_result.candidate_artifact_hash}")
    print(f"  synthetic_status   : {apply_result.synthetic_status}")
    print(f"  model_impact       : {apply_result.model_impact_artifact_uri}")
    print(f"  export_package     : {apply_result.export_package_artifact_uri}")

    # 6) Verify raw artifact immutability.
    _print_step("Raw source immutability check")
    raw_after = resources.object_storage.get(source_artifact.uri).data
    raw_before_sha = source_artifact.hash
    if raw_after_hash := _sha256_str(raw_after):
        same = raw_after_hash == raw_before_sha
        status = "ok" if same else "FAIL"
        print(f"  raw bytes hash     : {raw_after_hash}")
        print(f"  matches lineage    : {status}")
        if not same:
            raise SystemExit("APPLY mutated the raw source artifact — fatal regression")

    # 7) Inspect export package + model impact + dataset_card + lineage.
    if apply_result.export_package_artifact_uri is not None:
        _print_step("Export package readiness gates")
        package = _load_export_package(
            storage=resources.object_storage,
            uri=apply_result.export_package_artifact_uri,
        )
        print(f"  status             : {package.status.value}")
        print(f"  blocked_reasons    : {list(package.blocked_reason_codes)}")
        print("  artifact_kinds     :")
        for ref in package.artifacts:
            print(f"    - {ref.kind}: {ref.uri}")
        print(f"  object_counts      : included={package.object_counts.included}, "
              f"blocked={package.object_counts.blocked}, "
              f"excluded={package.object_counts.excluded}")
        if package.status is not ExportPackageStatus.READY:
            raise SystemExit(
                f"Export package not READY: {list(package.blocked_reason_codes)}"
            )

    if apply_result.model_impact_artifact_uri is not None:
        _print_step("Model impact verdict")
        impact = _load_model_impact_report(
            storage=resources.object_storage,
            uri=apply_result.model_impact_artifact_uri,
        )
        print(f"  verdict            : {impact.verdict.value}")
        print(f"  rare_class_recall  : "
              f"{impact.rare_class_recall_before:.3f} → {impact.rare_class_recall_after:.3f}")
        print(f"  macro_f1           : "
              f"{impact.macro_f1_before:.3f} → {impact.macro_f1_after:.3f}")
        print(f"  pr_auc             : {impact.pr_auc_status.value}")

    # 8) Fake platform event tape.
    _print_step("FakePlatform event tape")
    snapshot = fake_platform.snapshot()
    print(f"  total events       : {len(snapshot.job_events)}")
    for event in snapshot.job_events:
        progress_value = event.details.get("progress")
        progress = (
            f"{float(progress_value) * 100:5.1f}%"
            if isinstance(progress_value, (int, float))
            else "  -.-%"
        )
        print(
            f"    [{event.platform_job_id:<28}] {event.stage:<22} "
            f"status={event.status.value:<10} progress={progress}"
        )

    _print_step("Telemetry snapshot")
    metrics_snapshot = resources.metrics.snapshot()
    tracing_snapshot = resources.tracing.snapshot()
    print(f"  metric counters    : {len(metrics_snapshot.counters)}")
    print(f"  metric gauges      : {len(metrics_snapshot.gauges)}")
    print(f"  metric histograms  : {len(metrics_snapshot.histograms)}")
    print(f"  tracing spans      : {len(tracing_snapshot)}")
    for span in tracing_snapshot:
        print(
            f"    - {span.name:<24} status={span.status.value:<9} "
            f"duration={span.duration_ms:.2f}ms"
        )

    # 9) Final summary.
    _print_banner("DataForge AI MVP demo — PASSED")
    print(f"  workdir            : {workdir}")
    print(f"  analyze duration   : {analyze_duration:.2f}s")
    print(f"  apply duration     : {apply_duration:.2f}s")
    print(f"  total events       : {len(snapshot.job_events)}")
    print()

    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DataForge MVP demo.")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to use for the demo archive output (default: a fresh tmp dir).",
    )
    parser.add_argument(
        "--skip-fastapi",
        action="store_true",
        help="Skip the FastAPI /api/v1/health smoke check.",
    )
    return parser.parse_args(argv)


def _print_banner(text: str) -> None:
    bar = "=" * 72
    print()
    print(bar)
    print(text)
    print(bar)


def _print_step(text: str) -> None:
    print()
    print(f"--- {text} ".ljust(72, "-"))


def _demo_config(workdir: Path) -> ServiceConfig:
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
            service_signing_secret="local-dev-signing-secret",  # type: ignore[arg-type]
            service_identity="dataforge-platform",
            signature_max_age_seconds=300,
        ),
        dagster=DagsterSettings(
            home=str(workdir / "dagster"),
            job_name="dataforge_mvp_demo",
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


@contextmanager
def _fastapi_app(*, config: ServiceConfig) -> Iterator[str]:
    """Boot uvicorn in a subprocess against the demo config and probe health."""
    port = _free_port()
    env = os.environ.copy()
    env.update(
        DATAFORGE_PROFILE=config.profile.value,
        DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL=config.object_storage.endpoint_url,
        DATAFORGE_OBJECT_STORAGE_BUCKET=config.object_storage.bucket_name,
        DATAFORGE_PLATFORM_CALLBACK_URL=config.platform.callback_url,
        DATAFORGE_SERVICE_SIGNING_SECRET="local-dev-signing-secret",
        DATAFORGE_DAGSTER_HOME=config.dagster.home,
        DATAFORGE_POLICY_CONFIG_PATH=config.policies.policy_config_path,
        DATAFORGE_DECISION_POLICY_PATH=config.policies.decision_policy_path,
        DATAFORGE_SCORE_POLICY_PATH=config.policies.score_policy_path,
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{base_url}/api/v1/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except (OSError, ConnectionError):
                time.sleep(0.3)
        else:
            raise SystemExit("FastAPI app did not become healthy within 9 seconds")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return int(port)


def _probe_health(base_url: str) -> None:
    with urllib.request.urlopen(f"{base_url}/api/v1/health", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    print(f"  /api/v1/health     : {payload}")


def _probe_capabilities(base_url: str) -> None:
    with urllib.request.urlopen(f"{base_url}/api/v1/capabilities", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    print("  /api/v1/capabilities:")
    for plugin in payload["plugins"]:
        executable = "yes" if plugin["executable"] else "no"
        print(
            f"    - {plugin['plugin_id']:<24} readiness={plugin['readiness']:<14} "
            f"executable={executable}"
        )


def _resources_with_source(
    *,
    config: ServiceConfig,
    fake_platform: FakePlatformMetadataClient,
    archive_path: Path,
) -> tuple[ComputeResources, ArtifactRef, ArtifactRef, ArtifactRef]:
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=ObjectStorageScope(
            organization_id=_ORG_ID,
            project_id=_PROJECT_ID,
            dataset_id=_DATASET_ID,
        ),
    )
    registry = ArtifactRegistry(storage=storage)
    source_archive = registry.save_artifact(
        artifact_kind="raw_dataset_archive",
        data=archive_path.read_bytes(),
        artifact_format="zip",
        media_type="application/zip",
        schema_version="demo_archive.v1",
        dataset_version_id=_PARENT_VERSION_ID,
        created_by_job_id="compute_run_demo_source",
        config_hash="sha256:" + "9" * 64,
    ).artifact_ref
    with open_archive_path(archive_path) as reader:
        transactions = reader.find_required_transactions().read_bytes()
        prediction_payload = _read_predictions(reader.descriptors())
    source_artifact = registry.save_artifact(
        artifact_kind="raw_transactions",
        data=transactions,
        artifact_format="csv",
        media_type="text/csv",
        schema_version="tabular_dataset.v1",
        dataset_version_id=_PARENT_VERSION_ID,
        created_by_job_id="compute_run_demo_source",
        config_hash="sha256:" + "9" * 64,
    ).artifact_ref
    prediction_artifact = registry.save_artifact(
        artifact_kind="raw_predictions",
        data=prediction_payload,
        artifact_format="jsonl",
        media_type="application/jsonl",
        schema_version="prediction_manifest_row.v1",
        dataset_version_id=_PARENT_VERSION_ID,
        created_by_job_id="compute_run_demo_source",
        config_hash="sha256:" + "9" * 64,
    ).artifact_ref
    return (
        ComputeResources(
            service_config=config,
            object_storage=storage,
            artifact_registry=registry,
            fake_platform=fake_platform,
        ),
        source_archive,
        source_artifact,
        prediction_artifact,
    )


def _read_predictions(descriptors: tuple[Any, ...]) -> bytes:
    for descriptor in descriptors:
        if descriptor.kind.value == "predictions":
            with descriptor.open() as handle:
                return cast(bytes, handle.read())
    raise RuntimeError("demo archive must include predictions.jsonl")


def _action_plan_request(
    *,
    source_artifact: ArtifactRef,
) -> tuple[ActionPlanExecuteApprovedRequest, str]:
    pack = load_contract_pack()
    profile_payload = next(
        e.payload for e in pack.examples if e.name == "tabular_profile_report.fraud"
    )
    profile = TabularProfileReport.model_validate(profile_payload)
    recommendations = build_method_recommendations(
        BuildMethodRecommendationsRequest(tabular_profile=profile)
    )
    plan = build_action_plan_preview(
        BuildActionPlanPreviewRequest(
            decision_report_id="decision_report_demo_001",
            source_dataset_version_id=_PARENT_VERSION_ID,
            selected_decision_ids=(recommendations[0].recommendation_id,),
            selected_method_overrides={},
            method_recommendations=(recommendations[0],),
            created_by_user_id="platform_user_demo",
            input_artifacts=(source_artifact.uri,),
            target_version_name="dataset_version_v2_candidate",
            created_at=_GENERATED_AT,
        )
    ).model_copy(
        update={
            "requires_approval": True,
            "approval_request_id": "approval_request_demo_001",
        }
    )
    plan_hash = action_plan_integrity_hash(plan)
    approval = ActionPlanApprovalMetadata(
        approval_id="approval_demo_001",
        approval_request_id="approval_request_demo_001",
        approved_by_user_id="platform_owner_demo",
        approved_at=_GENERATED_AT,
        action_plan_id=plan.action_plan_id,
        action_plan_hash=plan_hash,
        decision_report_id=plan.created_from_decision_report,
        source_dataset_version_id=plan.source_dataset_version_id,
    )
    request = ActionPlanExecuteApprovedRequest(
        platform_job_id="platform_job_demo_apply",
        organization_id=_ORG_ID,
        project_id=_PROJECT_ID,
        dataset_id=_DATASET_ID,
        source_dataset_version_id=plan.source_dataset_version_id,
        action_plan=plan,
        approval_metadata=approval,
        source_artifacts=(source_artifact,),
    )
    return request, plan_hash


def _load_export_package(*, storage: MinioObjectStorageAdapter, uri: str) -> ExportPackage:
    payload = json.loads(storage.get(uri).data.decode("utf-8"))
    return ExportPackage.model_validate(payload)


def _load_model_impact_report(
    *,
    storage: MinioObjectStorageAdapter,
    uri: str,
) -> ModelImpactReport:
    payload = json.loads(storage.get(uri).data.decode("utf-8"))
    return ModelImpactReport.model_validate(payload)


def _sha256_str(data: bytes) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


# Silence unused-import warnings for typing helpers.
_ = (shutil,)


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
        Metadata: Any,
    ) -> Any:
        self._objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata),
            "LastModified": datetime.now(UTC),
        }
        return {"ETag": "fake-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> Any:
        record = self._object(Bucket, Key)
        return {
            "Body": BytesIO(record["Body"]),
            "ContentLength": len(record["Body"]),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Any:
        record = self._object(Bucket, Key)
        return {
            "ContentLength": len(record["Body"]),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
            "LastModified": record["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> Any:
        contents: list[dict[str, object]] = []
        for (bucket, key), record in sorted(self._objects.items()):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            contents.append(
                {
                    "Key": key,
                    "Size": len(record["Body"]),
                    "LastModified": record["LastModified"],
                }
            )
        return {"Contents": contents}

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
