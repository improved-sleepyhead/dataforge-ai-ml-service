"""Tabular plugin: deep MVP modality for the DataForge AI compute plane."""

from app.plugins.tabular.imputation import (
    CANDIDATE_TABULAR_DATASET_KIND,
    CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION,
    IMPUTATION_REPORT_KIND,
    IMPUTATION_REPORT_SCHEMA_VERSION,
    ExecuteTabularImputationRequest,
    ExecuteTabularImputationResult,
    ImputationExecutionError,
    execute_tabular_imputation_action,
)
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
    "CANDIDATE_TABULAR_DATASET_KIND",
    "CANDIDATE_TABULAR_DATASET_SCHEMA_VERSION",
    "BusinessRule",
    "BusinessRuleEvaluator",
    "PROFILE_REPORT_FORMAT",
    "PROFILE_REPORT_KIND",
    "PROFILE_REPORT_MEDIA_TYPE",
    "IMPUTATION_REPORT_KIND",
    "IMPUTATION_REPORT_SCHEMA_VERSION",
    "PROFILE_REPORT_SCHEMA_VERSION",
    "ExecuteTabularImputationRequest",
    "ExecuteTabularImputationResult",
    "ImputationExecutionError",
    "ProfileBuildRequest",
    "RuleFieldCheck",
    "build_tabular_profile_report",
    "execute_tabular_imputation_action",
    "compute_rules_config_hash",
    "infer_tabular_profile",
    "parse_business_rules",
]
