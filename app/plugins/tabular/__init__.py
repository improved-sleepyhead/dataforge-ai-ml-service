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
from app.plugins.tabular.rules import (
    BusinessRule,
    BusinessRuleEvaluator,
    RuleFieldCheck,
    compute_rules_config_hash,
    parse_business_rules,
)

__all__ = [
    "BuildProfileResult",
    "BusinessRule",
    "BusinessRuleEvaluator",
    "PROFILE_REPORT_FORMAT",
    "PROFILE_REPORT_KIND",
    "PROFILE_REPORT_MEDIA_TYPE",
    "PROFILE_REPORT_SCHEMA_VERSION",
    "ProfileBuildRequest",
    "RuleFieldCheck",
    "build_tabular_profile_report",
    "compute_rules_config_hash",
    "infer_tabular_profile",
    "parse_business_rules",
]
