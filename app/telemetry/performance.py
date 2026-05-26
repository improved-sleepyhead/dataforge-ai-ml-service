"""Privacy-safe stage timing helpers for compute performance acceptance.

The :class:`StageTimer` collects monotonic ``perf_counter`` durations for
named compute stages (manifest build, prediction validation, tabular
profile, text/OCR validation, model error analysis, review queue, etc.)
and emits a deterministic, contract-shaped report.

The timer never reads or stores raw artifact payloads, raw text, raw
PII, secrets, or any user-controlled metadata. Only stable technical
fields are recorded:

- stage name (string identifier from the caller);
- monotonic duration in milliseconds (float, rounded for stability);
- attempt counter (int);
- sample size hint (int, e.g. row count) when supplied by the caller;
- optional ``unit`` for the sample size (rows, records, objects, etc.).

The timer is designed to be used by the performance acceptance test
suite under :mod:`tests.performance`; it is not an OpenTelemetry export
adapter and is not in the API hot path.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StageTimingError(RuntimeError):
    """Raised when a stage timing operation cannot complete safely."""


class StageTiming(BaseModel):
    """Single recorded compute-stage timing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1)
    duration_ms: float = Field(ge=0.0)
    sample_size: int | None = Field(default=None, ge=0)
    sample_unit: str | None = None
    attempt: int = Field(default=1, ge=1)


class StageThreshold(BaseModel):
    """Maximum allowed duration for one stage on the demo profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1)
    max_duration_ms: float = Field(gt=0.0)
    note: str | None = None


class StageThresholdViolation(BaseModel):
    """A single stage that exceeded its documented MVP threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1)
    duration_ms: float = Field(ge=0.0)
    max_duration_ms: float = Field(gt=0.0)
    overshoot_ms: float = Field(ge=0.0)


class PerformanceReport(BaseModel):
    """Deterministic performance report saved to a test artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str = Field(min_length=1)
    timings: tuple[StageTiming, ...]
    thresholds: tuple[StageThreshold, ...]
    violations: tuple[StageThresholdViolation, ...]
    passed: bool


@dataclass
class StageTimer:
    """Collect monotonic stage durations and produce a contract-shaped report."""

    profile: str = "demo_strict"
    _records: list[StageTiming] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Avoid the mutable-default-argument pitfall while keeping the
        # dataclass shape simple for typing.
        self._records = []

    @contextmanager
    def measure(
        self,
        stage: str,
        *,
        sample_size: int | None = None,
        sample_unit: str | None = None,
    ) -> Iterator[None]:
        """Time a block of code and append a :class:`StageTiming` record."""
        if not stage:
            raise StageTimingError("stage name must be a non-empty string")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._records.append(
                StageTiming(
                    stage=stage,
                    duration_ms=round(elapsed_ms, 3),
                    sample_size=sample_size,
                    sample_unit=sample_unit,
                )
            )

    def record(
        self,
        stage: str,
        *,
        duration_ms: float,
        sample_size: int | None = None,
        sample_unit: str | None = None,
    ) -> None:
        """Record an externally measured stage duration."""
        if duration_ms < 0:
            raise StageTimingError("duration_ms must be non-negative")
        self._records.append(
            StageTiming(
                stage=stage,
                duration_ms=round(duration_ms, 3),
                sample_size=sample_size,
                sample_unit=sample_unit,
            )
        )

    @property
    def timings(self) -> tuple[StageTiming, ...]:
        """Return recorded stage timings in registration order."""
        return tuple(self._records)

    def build_report(
        self,
        *,
        thresholds: tuple[StageThreshold, ...],
    ) -> PerformanceReport:
        """Compare timings against thresholds and return a deterministic report."""
        threshold_by_stage = {threshold.stage: threshold for threshold in thresholds}
        # Group by stage and pick the maximum duration so flaky single-call
        # outliers cannot hide behind a faster sibling timing in the same stage.
        max_by_stage: dict[str, StageTiming] = {}
        for record in self._records:
            current = max_by_stage.get(record.stage)
            if current is None or record.duration_ms > current.duration_ms:
                max_by_stage[record.stage] = record

        violations: list[StageThresholdViolation] = []
        for stage, threshold in threshold_by_stage.items():
            timing = max_by_stage.get(stage)
            if timing is None:
                # A documented stage that did not run is a hard error: the
                # acceptance test must surface incomplete coverage.
                violations.append(
                    StageThresholdViolation(
                        stage=stage,
                        duration_ms=0.0,
                        max_duration_ms=threshold.max_duration_ms,
                        overshoot_ms=0.0,
                    )
                )
                continue
            if timing.duration_ms > threshold.max_duration_ms:
                violations.append(
                    StageThresholdViolation(
                        stage=stage,
                        duration_ms=timing.duration_ms,
                        max_duration_ms=threshold.max_duration_ms,
                        overshoot_ms=round(
                            timing.duration_ms - threshold.max_duration_ms, 3
                        ),
                    )
                )

        return PerformanceReport(
            profile=self.profile,
            timings=self.timings,
            thresholds=thresholds,
            violations=tuple(violations),
            passed=not violations,
        )


def write_performance_report(report: PerformanceReport, path: Path) -> None:
    """Persist a performance report as a deterministic JSON artifact."""
    payload = report.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "PerformanceReport",
    "StageThreshold",
    "StageThresholdViolation",
    "StageTimer",
    "StageTiming",
    "StageTimingError",
    "write_performance_report",
]
