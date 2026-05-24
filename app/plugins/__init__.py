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
from app.plugins.text_ocr import (
    BuildTextOcrReportResult,
    TextOcrBuildRequest,
    build_text_ocr_report,
    validate_ocr_records_jsonl,
    validate_support_messages_jsonl,
)
from app.plugins.validation import (
    DcrThresholds,
    RunValidationGatesRequest,
    RunValidationGatesResult,
    ValidationGatesError,
    run_validation_gates,
)

__all__ = [
    "BuildProfileResult",
    "BuildTextOcrReportResult",
    "DcrThresholds",
    "ProfileBuildRequest",
    "RunValidationGatesRequest",
    "RunValidationGatesResult",
    "TextOcrBuildRequest",
    "ValidationGatesError",
    "analyze_model_errors",
    "build_not_applicable_report",
    "build_static_plugin_manager",
    "build_static_plugin_registry",
    "build_tabular_profile_report",
    "build_text_ocr_report",
    "compute_object_signals",
    "infer_tabular_profile",
    "run_validation_gates",
    "validate_ocr_records_jsonl",
    "validate_support_messages_jsonl",
]
