"""Stable error-code → safe remediation hint mapping for API responses.

The hints are intentionally short, audience-agnostic, and do not expose
plugin internals, raw PII, or stack traces. They are returned in
``ErrorResponse.error.remediation_hint`` so the platform UI can render
a deterministic call-to-action without re-deriving it from internal
codes.
"""

from __future__ import annotations

from app.domain import ErrorCode

_REMEDIATION_HINTS: dict[ErrorCode, str] = {
    ErrorCode.INVALID_JOB_PAYLOAD: (
        "Check request fields, types and required identifiers; resubmit a corrected payload."
    ),
    ErrorCode.UNSUPPORTED_MODALITY: (
        "Select a modality that this plugin profile supports or enable the modality plugin."
    ),
    ErrorCode.ARTIFACT_NOT_FOUND: (
        "Verify the artifact id, version and that ingestion completed before retrying."
    ),
    ErrorCode.ARTIFACT_OUT_OF_SCOPE: (
        "Artifact URI must stay inside organization/project/dataset/version scope; do not"
        " reference foreign tenants."
    ),
    ErrorCode.INVALID_ARCHIVE_STRUCTURE: (
        "Re-upload an archive that matches the documented structure and required files."
    ),
    ErrorCode.ARCHIVE_SAFETY_VIOLATION: (
        "Archive failed safety checks; remove unsafe paths/symlinks and resubmit."
    ),
    ErrorCode.POLICY_BLOCKED: (
        "Action blocked by active policy; pick a different method or request a policy change."
    ),
    ErrorCode.PII_RESTRICTED: (
        "PII detected; redact or run a privacy review before exporting or sharing."
    ),
    ErrorCode.LEAKAGE_DETECTED: (
        "Train/test leakage detected; recreate the split or remove leaking columns/groups."
    ),
    ErrorCode.CONTRACT_VALIDATION_FAILED: (
        "Payload does not match the active contract pack; align with the schema and resubmit."
    ),
    ErrorCode.PREDICTION_VALIDATION_FAILED: (
        "Prediction rows did not validate; ensure predicted_proba/confidence/argmax are"
        " consistent."
    ),
    ErrorCode.PLUGIN_NOT_ENABLED: (
        "Required plugin is not enabled in this profile; ask an admin to enable it."
    ),
    ErrorCode.PLUGIN_CONTRACT_FAILED: (
        "Plugin output did not match its contract; report the plugin id and retry."
    ),
    ErrorCode.PLUGIN_EXECUTION_FAILED: (
        "Plugin execution failed without a normalized contract error; retry later or contact"
        " support if persistent."
    ),
    ErrorCode.DAGSTER_RUN_FAILED: (
        "Compute orchestration run failed; retry once before escalating."
    ),
    ErrorCode.ACTION_PLAN_REQUIRES_APPROVAL: (
        "ActionPlan needs platform approval metadata before execution."
    ),
    ErrorCode.ACTION_PLAN_SIGNATURE_INVALID: (
        "ActionPlan approval/signature does not match the plan integrity; re-sign and retry."
    ),
    ErrorCode.ACTION_PLAN_PRECONDITION_FAILED: (
        "ActionPlan step preconditions are not satisfied; check upstream artifacts and retry."
    ),
    ErrorCode.VALIDATION_GATE_FAILED: (
        "Candidate failed a validation gate; review gate report and fix the candidate."
    ),
    ErrorCode.MODEL_IMPACT_NOT_ELIGIBLE: (
        "Model impact run is not eligible; check minimum samples, splits and labels."
    ),
    ErrorCode.EXPORT_BLOCKED: (
        "Export blocked by hard policy; resolve blockers before retrying."
    ),
    ErrorCode.TENANT_SCOPE_VIOLATION: (
        "Request crossed tenant scope; only operate within your organization/project."
    ),
    ErrorCode.EXTERNAL_API_BLOCKED: (
        "External AI/network egress is disabled by policy in this profile."
    ),
    ErrorCode.RESOURCE_LIMIT_EXCEEDED: (
        "Compute resource quota exceeded; reduce input size or request a quota increase."
    ),
}


def remediation_hint_for(code: ErrorCode) -> str:
    """Return the short safe remediation hint for a stable error code.

    Falls back to ``INVALID_JOB_PAYLOAD`` hint if a code is added to the
    enum without a hint entry, so callers always receive a non-empty,
    safe string.
    """
    return _REMEDIATION_HINTS.get(code) or _REMEDIATION_HINTS[ErrorCode.INVALID_JOB_PAYLOAD]


__all__ = ["remediation_hint_for"]
