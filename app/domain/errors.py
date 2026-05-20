"""Stable compute-plane error response contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.common import NonEmptyStr


class ErrorCode(StrEnum):
    """Stable error taxonomy used by API responses, retries, and audit."""

    INVALID_JOB_PAYLOAD = "INVALID_JOB_PAYLOAD"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    ARTIFACT_OUT_OF_SCOPE = "ARTIFACT_OUT_OF_SCOPE"
    INVALID_ARCHIVE_STRUCTURE = "INVALID_ARCHIVE_STRUCTURE"
    ARCHIVE_SAFETY_VIOLATION = "ARCHIVE_SAFETY_VIOLATION"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    PII_RESTRICTED = "PII_RESTRICTED"
    LEAKAGE_DETECTED = "LEAKAGE_DETECTED"
    CONTRACT_VALIDATION_FAILED = "CONTRACT_VALIDATION_FAILED"
    PLUGIN_NOT_ENABLED = "PLUGIN_NOT_ENABLED"
    PLUGIN_CONTRACT_FAILED = "PLUGIN_CONTRACT_FAILED"
    PLUGIN_EXECUTION_FAILED = "PLUGIN_EXECUTION_FAILED"
    DAGSTER_RUN_FAILED = "DAGSTER_RUN_FAILED"
    ACTION_PLAN_REQUIRES_APPROVAL = "ACTION_PLAN_REQUIRES_APPROVAL"
    ACTION_PLAN_SIGNATURE_INVALID = "ACTION_PLAN_SIGNATURE_INVALID"
    ACTION_PLAN_PRECONDITION_FAILED = "ACTION_PLAN_PRECONDITION_FAILED"
    VALIDATION_GATE_FAILED = "VALIDATION_GATE_FAILED"
    MODEL_IMPACT_NOT_ELIGIBLE = "MODEL_IMPACT_NOT_ELIGIBLE"
    EXPORT_BLOCKED = "EXPORT_BLOCKED"
    TENANT_SCOPE_VIOLATION = "TENANT_SCOPE_VIOLATION"
    EXTERNAL_API_BLOCKED = "EXTERNAL_API_BLOCKED"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"


class ErrorBody(BaseModel):
    """Machine-readable error body with safe details only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ErrorCode
    message: NonEmptyStr
    recoverable: bool
    stage: NonEmptyStr
    job_id: str | None = None
    plugin_id: str | None = None
    remediation_hint: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Stable error response wrapper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorBody
