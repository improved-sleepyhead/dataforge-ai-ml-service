"""Compute-run cancellation, retry and failure classification (TASK-061).

The compute plane needs to:

* surface explicit ``CANCELLED`` and ``FAILED`` lifecycle states so the
  platform can render a stable timeline (PRD §20 + AGENTS.md);
* tag every failure with a stable ``recoverable`` flag so the platform
  retry policy can decide whether to re-run automatically;
* refuse to publish candidate artifacts for failed/cancelled runs
  (PRD §26 + DATASETS.md "failed validation must not promote candidate
  versions").

This module provides the building blocks:

* :class:`RunFailureReason` — stable taxonomy of compute failures with
  a deterministic ``is_recoverable`` classifier;
* :class:`RetryMetadata` — per-run retry envelope (attempt number,
  max attempts, retryable reasons);
* :class:`CancellationToken` — cooperative cancel signal that
  launchers check before/after Dagster materialization;
* :class:`CancellationRegistry` — in-process map of token-by-job-id
  used by the test/dev launcher and the test ``/cancel`` endpoint;
* :class:`RunCancelledError` — terminal error raised by launchers
  when a token has been triggered.

All entry points are pure (besides the registry's controlled in-memory
state) and never log raw PII or signed bodies. Production deployments
will replace ``CancellationRegistry`` with a real Dagster cancellation
signal or platform-control-plane channel; the contracts in this module
are designed to plug into those replacements without changes to the
launchers or assets.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.common import NonEmptyStr


class RunFailureReason(StrEnum):
    """Stable taxonomy of compute failure causes.

    Recoverable reasons (``is_recoverable=True``) typically indicate a
    transient infrastructure failure where a retry is safe. Non-
    recoverable reasons indicate logical/policy/contract failures
    where a retry without external intervention would deterministically
    fail the same way.
    """

    # Recoverable
    TRANSIENT_STORAGE_ERROR = "TRANSIENT_STORAGE_ERROR"
    DAGSTER_WORKER_RESTART = "DAGSTER_WORKER_RESTART"
    PLATFORM_CALLBACK_TIMEOUT = "PLATFORM_CALLBACK_TIMEOUT"
    EXTERNAL_AI_TIMEOUT = "EXTERNAL_AI_TIMEOUT"

    # Non-recoverable (do NOT auto-retry; surface to the user)
    INVALID_INPUT = "INVALID_INPUT"
    CONTRACT_VALIDATION_FAILED = "CONTRACT_VALIDATION_FAILED"
    POLICY_GATE_BLOCKED = "POLICY_GATE_BLOCKED"
    APPROVAL_HASH_MISMATCH = "APPROVAL_HASH_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_RECOVERABLE_REASONS: frozenset[RunFailureReason] = frozenset(
    {
        RunFailureReason.TRANSIENT_STORAGE_ERROR,
        RunFailureReason.DAGSTER_WORKER_RESTART,
        RunFailureReason.PLATFORM_CALLBACK_TIMEOUT,
        RunFailureReason.EXTERNAL_AI_TIMEOUT,
    }
)


def is_recoverable(reason: RunFailureReason) -> bool:
    """Return ``True`` when a retry without external intervention is safe."""
    return reason in _RECOVERABLE_REASONS


def recoverable_reasons() -> tuple[RunFailureReason, ...]:
    """Return the canonical recoverable-reason tuple sorted by enum value."""
    return tuple(
        sorted(_RECOVERABLE_REASONS, key=lambda reason: reason.value)
    )


class RetryMetadata(BaseModel):
    """Retry envelope attached to one compute run.

    The compute plane never picks a re-run on its own. The platform
    decides whether to retry based on the surfaced ``recoverable`` flag
    and retry policy. This metadata is what the launcher returns to the
    platform after a run terminates.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_number: int = Field(ge=1, default=1)
    max_attempts: int = Field(ge=1, default=2)
    recoverable: bool = False
    failure_reason: RunFailureReason | None = None
    last_error_message: str | None = None
    cancellation_reason_code: NonEmptyStr | None = None

    @property
    def is_terminal_failure(self) -> bool:
        return self.failure_reason is not None

    @property
    def can_retry(self) -> bool:
        return (
            self.is_terminal_failure
            and self.recoverable
            and self.attempt_number < self.max_attempts
        )


class RunCancelledError(RuntimeError):
    """Raised by a launcher when a cancellation token has been triggered."""

    def __init__(self, *, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass
class CancellationToken:
    """Cooperative cancel signal checked by launchers between stages.

    The token is intentionally cooperative: launchers must call
    :meth:`raise_if_cancelled` at the safe checkpoints they pick. This
    keeps cancellation deterministic and free of platform-specific
    signal handling. Once a token is cancelled it cannot be reset; the
    launcher must build a fresh token on retry.
    """

    reason_code: str = "platform_user_cancelled"
    _cancelled: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def cancel(self, *, reason_code: str | None = None) -> None:
        with self._lock:
            self._cancelled = True
            if reason_code is not None and reason_code:
                self.reason_code = reason_code

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise RunCancelledError(
                reason_code=self.reason_code,
                message=f"Compute run cancelled: {self.reason_code}",
            )


class CancellationRegistry:
    """In-process registry of cancellation tokens keyed by platform job id.

    The registry is a development/test substitute for a real Dagster
    cancellation signal or platform-driven cancel channel. Launchers
    register a fresh token before materialization and unregister it on
    completion. The test ``/cancel`` endpoint and helpers can then
    trigger the token by job id.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def register(self, *, platform_job_id: str) -> CancellationToken:
        """Return a token for the given job id, reusing an existing one if present.

        Reuse is intentional: the platform may pre-cancel a job before
        the worker has registered it (for example when the user clicks
        "cancel" before the request reaches the launcher). In that case
        ``register`` must return the *same* token that the platform has
        already triggered, so the launcher honors the cancel signal
        instead of overwriting it with a fresh, uncancelled token.
        """
        with self._lock:
            existing = self._tokens.get(platform_job_id)
            if existing is not None:
                return existing
            token = CancellationToken()
            self._tokens[platform_job_id] = token
        return token

    def cancel(self, *, platform_job_id: str, reason_code: str = "platform_user_cancelled") -> bool:
        """Cancel the token for ``platform_job_id`` if registered.

        Returns ``True`` if a token was found and cancelled, ``False``
        otherwise. The ``False`` path is the safe default for the
        platform: cancelling a job that already completed (or never
        started) is a no-op.
        """
        with self._lock:
            token = self._tokens.get(platform_job_id)
        if token is None:
            return False
        token.cancel(reason_code=reason_code)
        return True

    def is_cancelled(self, *, platform_job_id: str) -> bool:
        with self._lock:
            token = self._tokens.get(platform_job_id)
        return token is not None and token.is_cancelled

    def discard(self, *, platform_job_id: str) -> None:
        """Remove a token from the registry once the run terminates."""
        with self._lock:
            self._tokens.pop(platform_job_id, None)

    def known_job_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._tokens))


def classify_failure(
    message: str | None,
    fallback: RunFailureReason | None = None,
) -> RunFailureReason:
    """Best-effort failure classifier from a stage error message.

    The classifier picks a stable :class:`RunFailureReason` based on
    safe substring checks. It never reads raw PII because it only
    looks at exception messages, not at any payload bytes. Callers may
    pass an explicit ``fallback`` to bias the classification when the
    error class is known up front.
    """
    if message is None:
        return fallback or RunFailureReason.INTERNAL_ERROR
    lowered = message.lower()
    if "timeout" in lowered and "platform" in lowered:
        return RunFailureReason.PLATFORM_CALLBACK_TIMEOUT
    if "timeout" in lowered and "external" in lowered:
        return RunFailureReason.EXTERNAL_AI_TIMEOUT
    if "transient" in lowered or "storage" in lowered and "error" in lowered:
        return RunFailureReason.TRANSIENT_STORAGE_ERROR
    if "worker" in lowered and "restart" in lowered:
        return RunFailureReason.DAGSTER_WORKER_RESTART
    if "approval" in lowered and "hash" in lowered:
        return RunFailureReason.APPROVAL_HASH_MISMATCH
    if "policy" in lowered and ("blocked" in lowered or "gate" in lowered):
        return RunFailureReason.POLICY_GATE_BLOCKED
    if "contract" in lowered and ("invalid" in lowered or "fail" in lowered):
        return RunFailureReason.CONTRACT_VALIDATION_FAILED
    if "invalid" in lowered or "validation" in lowered:
        return RunFailureReason.INVALID_INPUT
    return fallback or RunFailureReason.INTERNAL_ERROR


def merge_unique_reasons(
    reasons: Iterable[RunFailureReason],
) -> tuple[RunFailureReason, ...]:
    """Return the unique ``RunFailureReason`` tuple in canonical order."""
    seen: dict[RunFailureReason, None] = {}
    for reason in reasons:
        seen.setdefault(reason, None)
    return tuple(sorted(seen, key=lambda value: value.value))


__all__ = [
    "CancellationRegistry",
    "CancellationToken",
    "RetryMetadata",
    "RunCancelledError",
    "RunFailureReason",
    "classify_failure",
    "is_recoverable",
    "merge_unique_reasons",
    "recoverable_reasons",
]
