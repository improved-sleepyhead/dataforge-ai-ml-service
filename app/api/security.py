"""Service-to-service request signature validation for protected API routes."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, Header, Request

from app.domain import ErrorCode
from app.kernel.config import ServiceConfig, load_config

SERVICE_IDENTITY_HEADER = "X-DataForge-Service-Identity"
TIMESTAMP_HEADER = "X-DataForge-Timestamp"
SIGNATURE_HEADER = "X-DataForge-Signature"
ORGANIZATION_ID_HEADER = "X-DataForge-Organization-Id"
PROJECT_ID_HEADER = "X-DataForge-Project-Id"
SIGNATURE_VERSION = "v1"


class ServiceSignatureError(ValueError):
    """Raised when a protected request fails service signature validation."""

    def __init__(
        self,
        *,
        reason_code: str,
        code: ErrorCode = ErrorCode.ACTION_PLAN_SIGNATURE_INVALID,
        status_code: int = 401,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class PlatformRequestIdentity:
    """Verified platform service identity and tenant/project scope."""

    service_identity: str
    organization_id: str
    project_id: str
    signed_at: datetime


async def require_platform_signature(
    request: Request,
    service_identity: Annotated[str | None, Header(alias=SERVICE_IDENTITY_HEADER)] = None,
    timestamp: Annotated[str | None, Header(alias=TIMESTAMP_HEADER)] = None,
    signature: Annotated[str | None, Header(alias=SIGNATURE_HEADER)] = None,
    organization_id: Annotated[str | None, Header(alias=ORGANIZATION_ID_HEADER)] = None,
    project_id: Annotated[str | None, Header(alias=PROJECT_ID_HEADER)] = None,
    config: Annotated[ServiceConfig | None, Depends(_get_service_config)] = None,
) -> PlatformRequestIdentity:
    """Verify platform identity, HMAC signature, timestamp freshness, and scope."""
    if config is None:
        raise ServiceSignatureError(reason_code="service_config_unavailable", status_code=500)
    if not service_identity:
        raise ServiceSignatureError(reason_code="missing_service_identity")
    if service_identity != config.platform.service_identity:
        raise ServiceSignatureError(reason_code="invalid_service_identity")
    if not timestamp:
        raise ServiceSignatureError(reason_code="missing_timestamp")
    if not signature:
        raise ServiceSignatureError(reason_code="missing_signature")
    if not organization_id or not project_id:
        raise ServiceSignatureError(reason_code="missing_scope_headers", status_code=403)

    signed_at = _parse_timestamp(timestamp)
    _validate_timestamp_freshness(signed_at, config.platform.signature_max_age_seconds)

    body = await request.body()
    payload = _parse_json_body(body)
    _validate_payload_scope(payload, organization_id=organization_id, project_id=project_id)

    expected = build_service_signature(
        secret=config.platform.service_signing_secret.get_secret_value(),
        service_identity=service_identity,
        timestamp=timestamp,
        organization_id=organization_id,
        project_id=project_id,
        body=body,
    )
    if not hmac.compare_digest(signature, expected):
        raise ServiceSignatureError(reason_code="signature_mismatch")

    return PlatformRequestIdentity(
        service_identity=service_identity,
        organization_id=organization_id,
        project_id=project_id,
        signed_at=signed_at,
    )


PlatformIdentityDep = Annotated[PlatformRequestIdentity, Depends(require_platform_signature)]


def build_service_signature(
    *,
    secret: str,
    service_identity: str,
    timestamp: str,
    organization_id: str,
    project_id: str,
    body: bytes,
) -> str:
    """Build the versioned HMAC signature used by platform-to-ML requests."""
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = "\n".join(
        (
            SIGNATURE_VERSION,
            service_identity,
            timestamp,
            organization_id,
            project_id,
            body_hash,
        )
    )
    digest = hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    return f"{SIGNATURE_VERSION}={digest}"


def _get_service_config(request: Request) -> ServiceConfig:
    config = getattr(request.app.state, "service_config", None)
    if isinstance(config, ServiceConfig):
        return config
    return load_config()


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ServiceSignatureError(reason_code="invalid_timestamp") from exc
    if parsed.tzinfo is None:
        raise ServiceSignatureError(reason_code="timestamp_must_include_timezone")
    return parsed.astimezone(UTC)


def _validate_timestamp_freshness(signed_at: datetime, max_age_seconds: int) -> None:
    age_seconds = abs((datetime.now(UTC) - signed_at).total_seconds())
    if age_seconds > max_age_seconds:
        raise ServiceSignatureError(reason_code="stale_timestamp")


def _parse_json_body(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceSignatureError(
            reason_code="invalid_json_payload",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            status_code=422,
        ) from exc
    if not isinstance(payload, dict):
        raise ServiceSignatureError(
            reason_code="payload_must_be_object",
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            status_code=422,
        )
    return payload


def _validate_payload_scope(
    payload: dict[str, Any],
    *,
    organization_id: str,
    project_id: str,
) -> None:
    if payload.get("organization_id") != organization_id or payload.get("project_id") != project_id:
        raise ServiceSignatureError(reason_code="scope_mismatch", status_code=403)
