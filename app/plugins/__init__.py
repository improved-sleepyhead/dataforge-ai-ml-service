"""Allowlisted modality and algorithm plugins."""

from app.plugins.predictions import (
    analyze_model_errors,
    build_not_applicable_report,
    compute_object_signals,
)
from app.plugins.registry import build_static_plugin_manager, build_static_plugin_registry
from app.plugins.tabular import (
    BuildProfileResult,
    ProfileBuildRequest,
    build_tabular_profile_report,
    infer_tabular_profile,
)

__all__ = [
    "BuildProfileResult",
    "ProfileBuildRequest",
    "analyze_model_errors",
    "build_not_applicable_report",
    "build_static_plugin_manager",
    "build_static_plugin_registry",
    "build_tabular_profile_report",
    "compute_object_signals",
    "infer_tabular_profile",
]
