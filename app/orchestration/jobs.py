"""Dagster jobs for ANALYZE_ONLY and APPLY_SELECTED_ACTIONS workflows."""

from __future__ import annotations

from dagster import AssetSelection, define_asset_job
from dagster._core.definitions.unresolved_asset_job_definition import (
    UnresolvedAssetJobDefinition,
)

from app.orchestration.apply_assets import APPLY_GROUP
from app.orchestration.assets import ANALYZE_GROUP

ANALYZE_JOB_NAME = "dataforge_analyze_dataset"
APPLY_JOB_NAME = "dataforge_apply_selected_actions"


def build_analyze_job() -> UnresolvedAssetJobDefinition:
    """Job that materializes the ANALYZE_ONLY asset graph.

    The job selects assets by group name only; this prevents it from picking
    up apply-stage assets and accidentally mutating data during analysis.
    """
    return define_asset_job(
        name=ANALYZE_JOB_NAME,
        selection=AssetSelection.groups(ANALYZE_GROUP),
        description="ANALYZE_ONLY: profile, score, evidence, decisions, review queue.",
    )


def build_apply_job() -> UnresolvedAssetJobDefinition:
    """Job that materializes APPLY_SELECTED_ACTIONS assets.

    Run config must provide an ApplyRunContext through the run_context
    resource; the apply assets refuse to run otherwise.
    """
    return define_asset_job(
        name=APPLY_JOB_NAME,
        selection=AssetSelection.groups(APPLY_GROUP),
        description="APPLY_SELECTED_ACTIONS: execute approved ActionPlan steps.",
    )


__all__ = [
    "ANALYZE_JOB_NAME",
    "APPLY_JOB_NAME",
    "build_analyze_job",
    "build_apply_job",
]
