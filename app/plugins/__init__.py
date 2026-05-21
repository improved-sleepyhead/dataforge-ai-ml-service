"""Allowlisted modality and algorithm plugins."""

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
    "build_static_plugin_manager",
    "build_static_plugin_registry",
    "build_tabular_profile_report",
    "infer_tabular_profile",
]
