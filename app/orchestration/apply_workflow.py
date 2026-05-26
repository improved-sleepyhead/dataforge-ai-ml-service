"""FastAPI-facing launcher for APPLY_SELECTED_ACTIONS Dagster materializations.

The launcher is the dev/test wiring for the
``POST /api/v1/action-plans/execute-approved`` endpoint. It accepts an
already-validated approved ActionPlan execution request, builds a Dagster
``Definitions`` instance with in-memory adapters, materializes the APPLY
asset graph and returns a safe summary of:

* the platform job id that the request carried;
* the Dagster status URL stub the platform UI can poll;
* the asset names that were materialized;
* the artifact references each apply stage registered (so the platform
  can pin the candidate / synthetic / model_impact / export artifacts to
  the platform job record without learning Dagster internals).

The launcher never overwrites raw artifacts: it always materializes new
immutable JSON placeholder records via :class:`ArtifactRegistry`. Real
algorithm wiring (full SMOTE / Gaussian Copula / model impact) lands in
TASK-058 and onwards; this launcher provides the pipeline shape so those
tasks can plug their builders into the corresponding apply assets.

The launcher is also the single place that builds an
:class:`ApplyRunContext` from an :class:`ActionPlanExecuteApprovedRequest`,
so the API handler can stay thin and free of Dagster knowledge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from dagster import AssetSelection, materialize

from app.adapters import (
    ArtifactRegistry,
    FakePlatformMetadataClient,
    MinioObjectStorageAdapter,
    ObjectStorageScope,
)
from app.adapters.object_storage import ObjectStorageError
from app.api.schemas import ActionPlanExecuteApprovedRequest
from app.domain import (
    ActionPlan,
    ArtifactRef,
    CandidatePolicyVersions,
    ComputeRunStatus,
    DecisionAction,
    ErrorCode,
    WorkflowType,
)
from app.domain.common import Sha256Digest
from app.kernel.config import ServiceConfig
from app.kernel.idempotency import (
    ApplyIdempotencyInputs,
    collect_artifact_hashes,
    compute_apply_idempotency_key,
    plugin_footprints_from_action_plan,
)
from app.orchestration.apply_assets import (
    APPLY_ASSET_KEYS,
    APPLY_ASSETS,
)
from app.orchestration.cancellation import (
    CancellationToken,
    RetryMetadata,
    RunCancelledError,
    RunFailureReason,
    classify_failure,
)
from app.orchestration.definitions import build_definitions
from app.orchestration.resources import ComputeResources
from app.orchestration.run_context import ApplyRunContext, RunContextResource
from app.orchestration.status_bridge import RunContext, RunStatusBridge


@dataclass(frozen=True)
class ApplyWorkflowResult:
    """Safe summary of an accepted APPLY_SELECTED_ACTIONS materialization."""

    job_id: str
    status: ComputeRunStatus
    status_url: str
    expected_outputs: tuple[str, ...]
    materialized_assets: tuple[str, ...]
    action_plan_id: str
    action_plan_hash: Sha256Digest
    idempotency_key: Sha256Digest
    retry_metadata: RetryMetadata
    candidate_artifact_uri: str | None
    candidate_artifact_hash: Sha256Digest | None
    synthetic_artifact_uri: str | None
    synthetic_status: str
    model_impact_artifact_uri: str | None
    export_package_artifact_uri: str | None
    mutates_dataset: bool = True


_SYNTHETIC_ACTION_TYPES: frozenset[str] = frozenset(
    {
        DecisionAction.AUGMENT_RARE_CLASS.value,
        DecisionAction.GENERATE_SYNTHETIC_CANDIDATE.value,
    }
)


def launch_apply_actions_workflow(
    *,
    request: ActionPlanExecuteApprovedRequest,
    action_plan_hash: Sha256Digest,
    config: ServiceConfig,
    fake_platform: FakePlatformMetadataClient,
    proposed_version_name: str | None = None,
    config_hash: Sha256Digest | None = None,
    policy_versions: CandidatePolicyVersions | None = None,
    require_model_impact_eligibility: bool = False,
    input_artifacts: tuple[ArtifactRef, ...] = (),
    cancellation_token: CancellationToken | None = None,
    attempt_number: int = 1,
) -> ApplyWorkflowResult:
    """Materialize the APPLY asset graph for an approved ActionPlan.

    The launcher constructs an in-memory Dagster definitions object with
    the same configuration the analyze launcher uses. It refuses to
    proceed without ``approval_metadata`` because the API layer only
    forwards approved requests.

    When ``cancellation_token`` is provided the launcher checks it
    before and after Dagster materialization. A triggered token raises
    :class:`RunCancelledError` after emitting a CANCELLED stage event,
    so the platform UI shows the run as terminated rather than hanging.
    Cancelled or failed runs MUST NOT publish a candidate artifact: the
    launcher returns no result and the per-asset placeholders, even if
    already registered, are not promoted because there is no
    ``ApplyWorkflowResult`` returned to the caller.

    Failures during Dagster materialization are classified into a stable
    :class:`RunFailureReason`, surfaced as ``recoverable=true|false`` in
    the FAILED stage event, and re-raised as :class:`RuntimeError` so
    callers cannot silently treat a failed run as ACCEPTED.
    """
    if request.approval_metadata is None:
        raise ValueError(
            "launch_apply_actions_workflow requires approval_metadata"
            " from the platform; refusing to materialize APPLY assets"
        )

    apply_context = _build_apply_context(
        request=request,
        proposed_version_name=proposed_version_name,
        config_hash=config_hash,
        policy_versions=policy_versions,
        input_artifacts=input_artifacts,
        require_model_impact_eligibility=require_model_impact_eligibility,
    )
    idempotency_key = compute_apply_idempotency_key(
        ApplyIdempotencyInputs(
            organization_id=request.organization_id,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            source_dataset_version_id=apply_context.source_dataset_version_id,
            proposed_version_name=apply_context.proposed_version_name,
            action_plan_hash=action_plan_hash,
            input_artifact_hashes=collect_artifact_hashes(input_artifacts),
            config_hash=apply_context.config_hash,
            contract_pack_version=config.contract_pack_version,
            policy_versions={
                "profile": apply_context.policy_versions.profile_policy_version,
                "decision": apply_context.policy_versions.decision_policy_version,
                "score": apply_context.policy_versions.score_policy_version,
                "method": apply_context.policy_versions.method_policy_version,
                "validation_gates": (
                    apply_context.policy_versions.validation_gates_policy_version
                    or "not_recorded"
                ),
            },
            step_idempotency_keys=tuple(
                step.idempotency_key for step in request.action_plan.steps
            ),
            plugin_versions=plugin_footprints_from_action_plan(request.action_plan),
        )
    )
    run_context = RunContext(
        compute_run_id=f"compute_{request.platform_job_id}",
        platform_job_id=request.platform_job_id,
        organization_id=request.organization_id,
        project_id=request.project_id,
        dataset_id=request.dataset_id,
        dataset_version_id=apply_context.proposed_version_name,
    )

    bridge = RunStatusBridge(fake_platform=fake_platform)
    if cancellation_token is not None and cancellation_token.is_cancelled:
        bridge.emit_cancelled(run_context=run_context, progress=0.0)
        raise RunCancelledError(
            reason_code=cancellation_token.reason_code,
            message=(
                "APPLY_SELECTED_ACTIONS workflow cancelled before "
                f"materialization: {cancellation_token.reason_code}"
            ),
        )

    definitions = build_definitions(
        compute_resources=_build_in_memory_resources(
            config=config,
            request=request,
            fake_platform=fake_platform,
        ),
        run_context_resource=RunContextResource(
            run_context=run_context,
            workflow_type=WorkflowType.APPLY_SELECTED_ACTIONS,
            apply_context=apply_context,
        ),
    )

    expected_outputs = tuple(key.path[-1] for key in APPLY_ASSET_KEYS)
    selection = AssetSelection.assets(*expected_outputs)
    bridge.emit_started(run_context=run_context)

    try:
        result = materialize(
            APPLY_ASSETS,
            selection=selection,
            resources=definitions.resources,
            raise_on_error=False,
        )
    except Exception as exc:  # noqa: BLE001 - boundary catches Dagster failures
        reason = classify_failure(str(exc))
        bridge.emit_failed(
            run_context=run_context,
            progress=0.0,
            error_code=reason.value,
        )
        raise RuntimeError(
            f"APPLY_SELECTED_ACTIONS workflow materialization raised: {reason.value}"
        ) from exc

    if cancellation_token is not None and cancellation_token.is_cancelled:
        bridge.emit_cancelled(run_context=run_context, progress=0.5)
        raise RunCancelledError(
            reason_code=cancellation_token.reason_code,
            message=(
                "APPLY_SELECTED_ACTIONS workflow cancelled after "
                f"materialization: {cancellation_token.reason_code}"
            ),
        )

    if not result.success:
        reason = RunFailureReason.INTERNAL_ERROR
        bridge.emit_failed(
            run_context=run_context,
            progress=0.0,
            error_code=reason.value,
        )
        raise RuntimeError("APPLY_SELECTED_ACTIONS workflow materialization failed")

    materialized_assets: list[str] = []
    artifact_uris: dict[str, tuple[str, str]] = {}
    synthetic_status = "registered" if apply_context.has_synthetic else "not_applicable"
    for event in result.get_asset_materialization_events():
        asset_name = _asset_name(event.asset_key)
        materialized_assets.append(asset_name)
        metadata = _materialization_metadata(event)
        uri_value = metadata.get("artifact_uri")
        hash_value = metadata.get("artifact_hash")
        if isinstance(uri_value, str) and isinstance(hash_value, str):
            artifact_uris[asset_name] = (uri_value, hash_value)
        if asset_name == "synthetic_dataset":
            status_value = metadata.get("asset_status")
            if isinstance(status_value, str):
                synthetic_status = status_value

    candidate_uri, candidate_hash = artifact_uris.get(
        "prepared_dataset", (None, None)
    )
    synthetic_uri = artifact_uris.get("synthetic_dataset", (None, None))[0]
    model_impact_uri = artifact_uris.get(
        "model_impact_report", (None, None)
    )[0]
    export_uri = artifact_uris.get("export_package", (None, None))[0]

    return ApplyWorkflowResult(
        job_id=request.platform_job_id,
        status=ComputeRunStatus.ACCEPTED,
        status_url=f"/api/v1/jobs/{request.platform_job_id}/status",
        expected_outputs=expected_outputs,
        materialized_assets=tuple(materialized_assets),
        action_plan_id=request.action_plan.action_plan_id,
        action_plan_hash=action_plan_hash,
        idempotency_key=idempotency_key,
        retry_metadata=RetryMetadata(attempt_number=attempt_number),
        candidate_artifact_uri=candidate_uri,
        candidate_artifact_hash=candidate_hash,
        synthetic_artifact_uri=synthetic_uri,
        synthetic_status=synthetic_status,
        model_impact_artifact_uri=model_impact_uri,
        export_package_artifact_uri=export_uri,
        mutates_dataset=True,
    )


def _build_apply_context(
    *,
    request: ActionPlanExecuteApprovedRequest,
    proposed_version_name: str | None,
    config_hash: Sha256Digest | None,
    policy_versions: CandidatePolicyVersions | None,
    input_artifacts: tuple[ArtifactRef, ...],
    require_model_impact_eligibility: bool,
) -> ApplyRunContext:
    plan = request.action_plan
    resolved_version = (
        proposed_version_name
        or _default_proposed_version_name(
            source_dataset_version_id=request.source_dataset_version_id,
            action_plan_id=plan.action_plan_id,
        )
    )
    resolved_config_hash = config_hash or _default_config_hash(plan)
    resolved_policy_versions = policy_versions or _default_policy_versions()
    return ApplyRunContext(
        action_plan_id=plan.action_plan_id,
        decision_report_id=plan.created_from_decision_report,
        source_dataset_version_id=request.source_dataset_version_id,
        proposed_version_name=resolved_version,
        config_hash=resolved_config_hash,
        policy_versions=resolved_policy_versions,
        action_plan=plan,
        input_artifacts=input_artifacts,
        synthetic_step_ids=_collect_synthetic_step_ids(plan),
        require_model_impact_eligibility=require_model_impact_eligibility,
    )


def _collect_synthetic_step_ids(plan: ActionPlan) -> tuple[str, ...]:
    return tuple(
        step.step_id for step in plan.steps if step.type in _SYNTHETIC_ACTION_TYPES
    )


def _default_proposed_version_name(
    *, source_dataset_version_id: str, action_plan_id: str
) -> str:
    suffix = action_plan_id.replace("action_plan_", "").replace("/", "_")
    return f"{source_dataset_version_id}__candidate_{suffix}"[:200]


def _default_config_hash(plan: ActionPlan) -> Sha256Digest:
    # Pick a stable per-plan default; the platform always overrides this
    # with the policy-bound hash but tests can rely on a deterministic value.
    if plan.steps:
        return plan.steps[0].config_hash
    return "sha256:" + "0" * 64


def _default_policy_versions() -> CandidatePolicyVersions:
    return CandidatePolicyVersions(
        profile_policy_version="demo_strict_v1",
        decision_policy_version="decision_policy_v0",
        score_policy_version="dataforge_score_v0",
        method_policy_version="method_policy_v0",
        validation_gates_policy_version="validation_gates_policy_v0",
    )


def _materialization_metadata(event: object) -> Mapping[str, Any]:
    # Dagster places asset materialization metadata at
    # ``event.event_specific_data.materialization.metadata`` and stores
    # values as ``MetadataValue`` instances. We lift the wrapped value
    # back into plain Python so downstream code can index into
    # ``artifact_uri`` and ``artifact_hash`` without learning Dagster
    # internals.
    materialization = getattr(
        getattr(event, "event_specific_data", None), "materialization", None
    )
    raw_metadata = getattr(materialization, "metadata", None)
    if not isinstance(raw_metadata, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw_metadata.items():
        if hasattr(value, "text"):
            out[key] = value.text
        elif hasattr(value, "value"):
            out[key] = value.value
        elif hasattr(value, "data"):
            out[key] = value.data
        else:
            out[key] = value
    return out


def _asset_name(asset_key: object) -> str:
    if not hasattr(asset_key, "path"):
        raise RuntimeError("Dagster materialization event did not include an asset key")
    path = asset_key.path
    if not isinstance(path, list | tuple) or not path:
        raise RuntimeError("Dagster materialization event included an invalid asset key")
    return str(path[-1])


def _build_in_memory_resources(
    *,
    config: ServiceConfig,
    request: ActionPlanExecuteApprovedRequest,
    fake_platform: FakePlatformMetadataClient,
) -> ComputeResources:
    scope = ObjectStorageScope(
        organization_id=request.organization_id,
        project_id=request.project_id,
        dataset_id=request.dataset_id,
    )
    storage = MinioObjectStorageAdapter(
        client=_InMemoryS3Client(),
        bucket_name=config.object_storage.bucket_name,
        prefix_root=config.object_storage.prefix_root,
        scope=scope,
    )
    return ComputeResources(
        service_config=config,
        object_storage=storage,
        artifact_registry=ArtifactRegistry(storage=storage),
        fake_platform=fake_platform,
    )


class _InMemoryS3Client:
    """Tiny S3-compatible fake used only by the apply launcher."""

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
        record = self._object(Bucket, Key)
        body = record["Body"]
        if not isinstance(body, bytes):  # pragma: no cover - defensive
            raise TypeError("InMemoryS3Client body must be bytes")
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "ContentType": record["ContentType"],
            "Metadata": record["Metadata"],
        }

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        record = self._object(Bucket, Key)
        body = record["Body"]
        if not isinstance(body, bytes):  # pragma: no cover - defensive
            raise TypeError("InMemoryS3Client body must be bytes")
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
            body = record["Body"]
            if not isinstance(body, bytes):  # pragma: no cover - defensive
                continue
            contents.append({"Key": key, "Size": len(body)})
        return {"Contents": contents}

    def _object(self, bucket: str, key: str) -> dict[str, Any]:
        try:
            return self._objects[(bucket, key)]
        except KeyError as exc:
            raise ObjectStorageError(
                code=ErrorCode.ARTIFACT_NOT_FOUND,
                message=f"missing object {bucket}/{key}",
            ) from exc


__all__ = [
    "ApplyWorkflowResult",
    "launch_apply_actions_workflow",
]
