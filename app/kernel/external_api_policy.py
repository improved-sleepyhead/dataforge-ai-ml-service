"""External AI/network egress policy gate.

PRD §1.5/§2.7 require that external AI calls be policy-gated and that the
strict banking profile keep external API access disabled by default. This
module provides a small, side-effect-free gate that the Python compute plane
can call before any external connector or network egress.

The gate never logs raw provider URLs, raw secrets, or raw payloads. It only
returns a stable error code (``EXTERNAL_API_BLOCKED``) and a structured reason
code that the caller can surface through the :class:`ErrorResponse` taxonomy.

The gate intentionally does not perform any network I/O. Network adapters
should call :func:`require_external_api_allowed` before constructing a client
or initiating egress.
"""

from __future__ import annotations

from app.domain import ErrorCode
from app.kernel.config import RuntimeProfile, ServiceConfig


class ExternalApiBlockedError(PermissionError):
    """Raised when external AI/network egress is denied by policy."""

    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code: ErrorCode = ErrorCode.EXTERNAL_API_BLOCKED
        self.reason_code = reason_code
        self.provider = provider


def is_external_api_allowed(config: ServiceConfig) -> bool:
    """Return ``True`` only when the runtime profile and policy permit external API."""

    if not config.external_ai.allow_external_api:
        return False
    if config.profile is RuntimeProfile.BANKING_STRICT:
        # Defense-in-depth: even if the flag is mistakenly enabled, the banking
        # strict profile must never allow external API access without an
        # explicit, audited admin override that is not part of this gate.
        return False
    return True


def require_external_api_allowed(
    config: ServiceConfig,
    *,
    provider: str | None = None,
) -> None:
    """Raise :class:`ExternalApiBlockedError` unless external API is allowed."""

    if is_external_api_allowed(config):
        return
    if config.profile is RuntimeProfile.BANKING_STRICT:
        reason_code = "banking_strict_profile_blocks_external_api"
        message = (
            "External AI/network egress is disabled in banking_strict profile."
        )
    else:
        reason_code = "external_api_disabled_by_policy"
        message = "External AI/network egress is disabled by policy."
    raise ExternalApiBlockedError(
        reason_code=reason_code,
        message=message,
        provider=provider,
    )


__all__ = [
    "ExternalApiBlockedError",
    "is_external_api_allowed",
    "require_external_api_allowed",
]
