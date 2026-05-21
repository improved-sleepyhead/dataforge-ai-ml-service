"""Tabular plugin: deep MVP modality for the DataForge AI compute plane."""

from app.plugins.tabular.profile import (
    PROFILE_REPORT_FORMAT,
    PROFILE_REPORT_KIND,
    PROFILE_REPORT_MEDIA_TYPE,
    PROFILE_REPORT_SCHEMA_VERSION,
    BuildProfileResult,
    ProfileBuildRequest,
    build_tabular_profile_report,
    infer_tabular_profile,
)

__all__ = [
    "BuildProfileResult",
    "PROFILE_REPORT_FORMAT",
    "PROFILE_REPORT_KIND",
    "PROFILE_REPORT_MEDIA_TYPE",
    "PROFILE_REPORT_SCHEMA_VERSION",
    "ProfileBuildRequest",
    "build_tabular_profile_report",
    "infer_tabular_profile",
]
