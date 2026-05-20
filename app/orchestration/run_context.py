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
the approved ActionPlan/decision references. ANALYZE_ONLY runs leave it
``None``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain import WorkflowType
from app.orchestration.status_bridge import RunContext


@dataclass(frozen=True)
class ApplyRunContext:
    """Approved-action references attached to APPLY runs."""

    action_plan_id: str
    decision_report_id: str


@dataclass(frozen=True)
class RunContextResource:
    """Bound per-run context surfaced to Dagster assets."""

    run_context: RunContext
    workflow_type: WorkflowType
    apply_context: ApplyRunContext | None = None


__all__ = ["ApplyRunContext", "RunContextResource"]
