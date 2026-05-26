"""Privacy-safe tracing spans for the compute pipeline.

The :class:`TracingRegistry` is a small, stdlib-only span recorder
inspired by OpenTelemetry's span shape but intentionally narrower: it
records technical fields only and refuses to accept raw payloads,
secrets, or user-controlled metadata.

Spans cover the named compute stages required by PRD §15 / TASK-070:

- ``ingestion`` and ``manifest_builder`` for the asset-manifest stage;
- ``prediction.validate`` for PredictionManifest ingestion + coverage;
- ``model_error.analyze`` for ambiguous / probable label-error analysis;
- ``tabular.profile`` for the tabular profile builder;
- ``text_ocr.profile`` for text/OCR validation + PII detection;
- ``decision_core.score`` for DecisionReport scoring;
- ``model_impact.evaluate`` for model-impact evaluation;
- ``export.build`` for the ExportPackage builder.

Each span carries technical attributes only (allow-listed keys, bounded
values) and a stable status (``OK`` / ``ERROR`` / ``CANCELLED``). The
recorder is deterministic and synchronous; it has no exporter wired in,
so it is safe to use in tests and the local demo without touching the
network.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field

from app.domain.errors import ErrorCode

# Allow-listed span attribute keys. Anything else is dropped on entry.
_ALLOWED_ATTRIBUTE_KEYS: frozenset[str] = frozenset(
    {
        "stage",
        "plugin",
        "job_type",
        "row_count",
        "object_count",
        "queue_type",
        "verdict",
        "status",
        "modality",
        "error_code",
        "reason_code",
        "duration_ms",
        "sample_size",
        "ambiguous_object_count",
        "probable_label_error_count",
        "blocked_count",
    }
)
# Bounded string-attribute length so a regression cannot blow up trace
# cardinality with long user-provided strings.
_MAX_ATTRIBUTE_VALUE_LENGTH = 64


class TracingError(ValueError):
    """Raised when a tracing operation is unsafe or invalid."""


class SpanStatus(StrEnum):
    """Stable span outcome used by the local tracing registry."""

    OK = "OK"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class SpanRecord(BaseModel):
    """Recorded span snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    duration_ms: float = Field(ge=0.0)
    status: SpanStatus
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


@dataclass
class _ActiveSpan:
    name: str
    started_at: float
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)
    status: SpanStatus = SpanStatus.OK


@dataclass
class TracingRegistry:
    """Thread-safe in-memory span registry for compute traces."""

    spans: list[SpanRecord] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[_ActiveSpan]:
        """Open a new span context, time it, and append a :class:`SpanRecord`."""
        if not name:
            raise TracingError("span name must be a non-empty string")
        active = _ActiveSpan(
            name=name,
            started_at=time.perf_counter(),
            attributes=_safe_attributes(attributes),
        )
        try:
            yield active
        except BaseException:
            active.status = SpanStatus.ERROR
            raise
        finally:
            duration_ms = (time.perf_counter() - active.started_at) * 1000.0
            attrs = dict(active.attributes)
            attrs.setdefault("duration_ms", round(duration_ms, 3))
            with self._lock:
                self.spans.append(
                    SpanRecord(
                        name=active.name,
                        duration_ms=round(duration_ms, 3),
                        status=active.status,
                        attributes=attrs,
                    )
                )

    def record_span(
        self,
        *,
        name: str,
        duration_ms: float,
        status: SpanStatus = SpanStatus.OK,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        """Record a span without using the context manager."""
        if not name:
            raise TracingError("span name must be a non-empty string")
        if duration_ms < 0:
            raise TracingError("duration_ms must be non-negative")
        attrs = _safe_attributes(attributes)
        attrs.setdefault("duration_ms", round(duration_ms, 3))
        with self._lock:
            self.spans.append(
                SpanRecord(
                    name=name,
                    duration_ms=round(duration_ms, 3),
                    status=status,
                    attributes=attrs,
                )
            )

    def snapshot(self) -> tuple[SpanRecord, ...]:
        """Return spans in registration order."""
        with self._lock:
            return tuple(self.spans)


def _safe_attributes(
    attributes: Mapping[str, object] | None,
) -> dict[str, str | int | float | bool]:
    if attributes is None:
        return {}
    safe: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        if key not in _ALLOWED_ATTRIBUTE_KEYS:
            continue
        coerced = _coerce_attribute_value(value)
        if coerced is None:
            continue
        safe[key] = coerced
    return safe


def _coerce_attribute_value(
    value: object,
) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, ErrorCode):
        return value.value
    if hasattr(value, "value") and isinstance(value.value, str):  # StrEnum
        text = value.value
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return None
    if len(text) > _MAX_ATTRIBUTE_VALUE_LENGTH:
        text = text[:_MAX_ATTRIBUTE_VALUE_LENGTH]
    return text


__all__ = [
    "SpanRecord",
    "SpanStatus",
    "TracingError",
    "TracingRegistry",
]
