"""Single ML service quality gate command.

Runs every documented quality suite in a stable order, captures per-suite
results, prints a deterministic ``passed/failed`` summary table with
artifact paths, and exits non-zero when any critical suite fails.

Suites:

    1. lint               — `ruff check app tests`
    2. typecheck          — `mypy app tests` (strict)
    3. unit               — `pytest tests`
    4. contract           — `pytest tests/contracts`
    5. plugin             — `pytest tests/plugins`
    6. security_privacy   — `pytest tests/security`
    7. e2e_compute        — `pytest tests/e2e`
    8. performance        — `pytest tests/performance`

The first six are critical: any failure fails the gate. The performance
suite is critical too (TASK-069 requires it to fail when any stage
breaches its threshold), so the default gate runs all eight.

Usage:

    .venv/bin/python -m tools.quality_gate
    .venv/bin/python -m tools.quality_gate --report build/quality_gate.json
    .venv/bin/python -m tools.quality_gate --skip performance unit
    .venv/bin/python -m tools.quality_gate --python .venv/bin/python

The script never reads or stores raw payloads; it only forwards the
existing Make-driven commands and records the resulting exit codes,
durations, and command lines.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Suite:
    """A single quality suite the gate runs."""

    name: str
    description: str
    command: tuple[str, ...]
    optional: bool = False


@dataclass
class SuiteResult:
    """Result of running one quality suite."""

    suite: Suite
    returncode: int
    duration_seconds: float
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def passed(self) -> bool:
        return not self.skipped and self.returncode == 0


@dataclass
class GateReport:
    """Deterministic summary of a quality-gate run."""

    results: list[SuiteResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    artifact_paths: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(
            result.skipped or result.passed
            for result in self.results
            if not result.suite.optional or not result.skipped
        ) and not self.failed_results()

    def failed_results(self) -> list[SuiteResult]:
        return [
            result
            for result in self.results
            if not result.skipped and result.returncode != 0
        ]

    def to_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "artifact_paths": list(self.artifact_paths),
            "results": [
                {
                    "name": r.suite.name,
                    "description": r.suite.description,
                    "command": list(r.suite.command),
                    "returncode": r.returncode,
                    "duration_seconds": round(r.duration_seconds, 3),
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "passed": r.passed,
                    "optional": r.suite.optional,
                }
                for r in self.results
            ],
        }


def build_suites(python: str) -> tuple[Suite, ...]:
    """Build the canonical list of suites pinned to a python interpreter."""
    return (
        Suite(
            name="lint",
            description="ruff check app tools tests",
            command=(python, "-m", "ruff", "check", "app", "tools", "tests"),
        ),
        Suite(
            name="typecheck",
            description="mypy app tools tests (strict)",
            command=(python, "-m", "mypy", "app", "tools", "tests"),
        ),
        Suite(
            name="unit",
            description="pytest tests (full suite)",
            command=(python, "-m", "pytest", "tests"),
        ),
        Suite(
            name="contract",
            description="pytest tests/contracts",
            command=(python, "-m", "pytest", "tests/contracts"),
        ),
        Suite(
            name="plugin",
            description="pytest tests/plugins",
            command=(python, "-m", "pytest", "tests/plugins"),
        ),
        Suite(
            name="security_privacy",
            description="pytest tests/security",
            command=(python, "-m", "pytest", "tests/security"),
        ),
        Suite(
            name="e2e_compute",
            description="pytest tests/e2e",
            command=(python, "-m", "pytest", "tests/e2e"),
        ),
        Suite(
            name="performance",
            description="pytest tests/performance",
            command=(python, "-m", "pytest", "tests/performance"),
        ),
    )


def run_gate(
    *,
    suites: Sequence[Suite],
    cwd: Path = REPO_ROOT,
    skip: frozenset[str] = frozenset(),
    artifact_paths: Sequence[str] = (),
    stream: bool = True,
) -> GateReport:
    """Run every suite in order and return a :class:`GateReport`."""
    report = GateReport(artifact_paths=list(artifact_paths))
    overall_start = time.perf_counter()

    for suite in suites:
        if suite.name in skip:
            report.results.append(
                SuiteResult(
                    suite=suite,
                    returncode=0,
                    duration_seconds=0.0,
                    skipped=True,
                    skip_reason="explicitly skipped via --skip",
                )
            )
            if stream:
                print(f"[skip ] {suite.name}: explicitly skipped via --skip")
            continue

        if shutil.which(suite.command[0]) is None and not Path(suite.command[0]).exists():
            report.results.append(
                SuiteResult(
                    suite=suite,
                    returncode=0,
                    duration_seconds=0.0,
                    skipped=True,
                    skip_reason=f"interpreter {suite.command[0]!r} not found",
                )
            )
            if stream:
                print(
                    f"[skip ] {suite.name}: interpreter "
                    f"{suite.command[0]!r} not found"
                )
            continue

        if stream:
            print(f"[run  ] {suite.name}: {' '.join(suite.command)}")
        suite_start = time.perf_counter()
        completed = subprocess.run(
            list(suite.command),
            cwd=str(cwd),
            check=False,
        )
        duration = time.perf_counter() - suite_start

        report.results.append(
            SuiteResult(
                suite=suite,
                returncode=completed.returncode,
                duration_seconds=duration,
            )
        )
        if stream:
            tag = "ok   " if completed.returncode == 0 else "fail "
            print(
                f"[{tag}] {suite.name}: rc={completed.returncode} "
                f"in {duration:.2f}s"
            )

    report.total_duration_seconds = time.perf_counter() - overall_start
    return report


def render_summary(report: GateReport) -> str:
    """Return the deterministic console summary block."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(
        "DataForge ML quality gate — "
        + ("PASSED" if report.passed else "FAILED")
    )
    lines.append("=" * 72)
    lines.append(
        f"{'suite':<22}{'status':<10}{'duration':<12}{'command'}"
    )
    lines.append("-" * 72)
    for result in report.results:
        if result.skipped:
            status = "skipped"
        elif result.passed:
            status = "passed"
        else:
            status = "failed"
        duration = (
            f"{result.duration_seconds:.2f}s" if not result.skipped else "—"
        )
        lines.append(
            f"{result.suite.name:<22}{status:<10}{duration:<12}"
            f"{' '.join(result.suite.command)}"
        )
    lines.append("-" * 72)
    lines.append(f"total wall time: {report.total_duration_seconds:.2f}s")
    if report.artifact_paths:
        lines.append("artifacts:")
        for path in report.artifact_paths:
            lines.append(f"  - {path}")
    failures = report.failed_results()
    if failures:
        lines.append("failed suites:")
        for failure in failures:
            lines.append(
                f"  - {failure.suite.name} (rc={failure.returncode})"
            )
    lines.append("=" * 72)
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DataForge AI ML service quality gate."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to run the suites with (default: %(default)s).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="SUITE",
        help="Suite names to skip (e.g. performance unit).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write a deterministic JSON summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suites = build_suites(args.python)
    skip_set = frozenset(args.skip)

    artifact_paths: list[str] = []
    if args.report:
        artifact_paths.append(args.report)

    report = run_gate(
        suites=suites,
        skip=skip_set,
        artifact_paths=artifact_paths,
    )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_payload(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    print(render_summary(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
