"""Per-run compute context exposed as a Dagster resource.

Dagster materializations need access to per-run identifiers (compute_run_id,
platform_job_id, dataset_version_id, workflow_type) without coupling assets
to global state. The control plane passes this context as a signed payload;
the compute plane wraps it in :class:`RunContextResource` and exposes it as
a Dagster resource so assets can read it via ``context.resources.run_context``.

The resource is a tiny dataclass-style holder. It does not perform I/O and
does not log anything. Callers must build it once per run and pass it
through ``ResourceDefinition.hardcoded_resource``.

For ``APPLY_SELECTED_ACTIONS`` runs, ``apply_context`` must be populated with
the approved ActionPlan/decision references plus the lineage envelope the
candidate-version builder will need (parent dataset version, proposed
version name, policy versions, config hash, optional synthetic step
identifiers). ANALYZE_ONLY runs leave it ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain import ActionPlan, ArtifactRef, CandidatePolicyVersions, WorkflowType
from app.domain.common import Sha256Digest
from app.orchestration.status_bridge import RunContext


@dataclass(frozen=True)
class ApplyRunContext:
    """Approved-action references attached to APPLY runs.

    The context is carried into Dagster apply assets through
    ``RunContextResource.apply_context``. Assets read it (read-only) to:

    * record lineage (``source_dataset_version_id`` ->
      ``proposed_version_name``);
    * iterate over approved ``action_plan.steps``;
    * skip the ``synthetic_dataset`` asset when no synthetic step is
      selected;
    * surface the policy versions and config hash on the candidate
      and export artifacts.

    Apply assets must never mutate raw dataset artifacts; this context
    only describes what the run is about to *register* as immutable
    candidate/synthetic/model-impact/export artifacts.
    """

    action_plan_id: str
    decision_report_id: str
    source_dataset_version_id: str = "dataset_version_v1"
    proposed_version_name: str = "dataset_version_v1__candidate_default"
    config_hash: Sha256Digest = "sha256:" + "0" * 64
    policy_versions: CandidatePolicyVersions = field(
        default_factory=lambda: CandidatePolicyVersions(
            profile_policy_version="demo_strict_v1",
            decision_policy_version="decision_policy_v0",
            score_policy_version="dataforge_score_v0",
            method_policy_version="method_policy_v0",
            validation_gates_policy_version="validation_gates_policy_v0",
        )
    )
    action_plan: ActionPlan | None = None
    input_artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    synthetic_step_ids: tuple[str, ...] = field(default_factory=tuple)
    require_model_impact_eligibility: bool = False

    @property
    def has_synthetic(self) -> bool:
        """Return ``True`` when at least one synthetic step is selected."""
        return bool(self.synthetic_step_ids)


@dataclass(frozen=True)
class RunContextResource:
    """Bound per-run context surfaced to Dagster assets."""

    run_context: RunContext
    workflow_type: WorkflowType
    apply_context: ApplyRunContext | None = None


__all__ = ["ApplyRunContext", "RunContextResource"]
