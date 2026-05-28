"""TASK-074: tests for the unified ML service quality gate command."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from tools.quality_gate import (
    GateReport,
    Suite,
    SuiteResult,
    build_suites,
    main,
    parse_args,
    render_summary,
    run_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — command covers every documented suite.
# ---------------------------------------------------------------------------


def test_build_suites_covers_every_documented_quality_suite() -> None:
    """Build must enumerate every suite required by the PRD/INFRA pipeline."""
    suites = build_suites("python", timeout_seconds=123)
    names = tuple(suite.name for suite in suites)

    assert names == (
        "lint",
        "typecheck",
        "unit",
        "contract",
        "plugin",
        "security_privacy",
        "e2e_compute",
        "performance",
    )
    # Every command starts with the documented Python interpreter so
    # callers can pin the venv.
    for suite in suites:
        assert suite.command[0] == "python"
        assert suite.timeout_seconds == 123

    # Documented commands match the make targets the runbook publishes.
    by_name = {suite.name: suite for suite in suites}
    assert "ruff" in by_name["lint"].command
    assert "tools" in by_name["lint"].command
    assert "mypy" in by_name["typecheck"].command
    assert "tools" in by_name["typecheck"].command
    assert by_name["unit"].command[-1] == "tests"
    assert by_name["contract"].command[-1] == "tests/contracts"
    assert by_name["plugin"].command[-1] == "tests/plugins"
    assert by_name["security_privacy"].command[-1] == "tests/security"
    assert by_name["e2e_compute"].command[-1] == "tests/e2e"
    assert by_name["performance"].command[-1] == "tests/performance"


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — gate fails on any failed critical suite.
# ---------------------------------------------------------------------------


def test_run_gate_fails_when_any_suite_returns_non_zero(tmp_path: Path) -> None:
    """A failing suite must mark the gate as failed."""
    fake_failing = _fake_suite(name="contract", returncode=1)
    fake_passing = _fake_suite(name="lint", returncode=0)

    report = run_gate(suites=[fake_passing, fake_failing], cwd=tmp_path, stream=False)

    assert report.passed is False
    failed = report.failed_results()
    assert len(failed) == 1
    assert failed[0].suite.name == "contract"
    assert failed[0].returncode == 1
    # The successful suite must still appear in the result list with
    # passed=True so the summary stays informative.
    by_name = {result.suite.name: result for result in report.results}
    assert by_name["lint"].passed is True


def test_run_gate_passes_when_all_suites_return_zero(tmp_path: Path) -> None:
    """A clean run reports passed=True and zero failed suites."""
    suites = [_fake_suite(name=name, returncode=0) for name in ("lint", "unit")]
    report = run_gate(suites=suites, cwd=tmp_path, stream=False)
    assert report.passed is True
    assert report.failed_results() == []
    assert all(result.passed for result in report.results)


def test_run_gate_records_per_suite_durations(tmp_path: Path) -> None:
    """Each result must carry a non-negative duration measurement."""
    report = run_gate(
        suites=[_fake_suite(name="lint", returncode=0)],
        cwd=tmp_path,
        stream=False,
    )
    assert len(report.results) == 1
    assert report.results[0].duration_seconds >= 0.0
    assert report.total_duration_seconds >= report.results[0].duration_seconds


def test_run_gate_skips_named_suites(tmp_path: Path) -> None:
    """The ``skip`` argument must mark suites as skipped without running them."""
    report = run_gate(
        suites=[
            _fake_suite(name="lint", returncode=1),  # would fail if it ran
            _fake_suite(name="unit", returncode=0),
        ],
        cwd=tmp_path,
        skip=frozenset({"lint"}),
        stream=False,
    )
    by_name = {result.suite.name: result for result in report.results}
    assert by_name["lint"].skipped is True
    assert by_name["lint"].skip_reason == "explicitly skipped via --skip"
    assert by_name["unit"].passed is True
    # Skipped suites do not flip the passed flag.
    assert report.passed is True


def test_run_gate_records_timeout_as_failed_suite(tmp_path: Path) -> None:
    """A hung suite must become a structured failed timeout result."""
    suite = Suite(
        name="unit",
        description="fake hung unit suite",
        command=("/bin/sleep", "2"),
        timeout_seconds=1,
    )

    report = run_gate(suites=[suite], cwd=tmp_path, stream=False)

    assert report.passed is False
    assert len(report.failed_results()) == 1
    result = report.failed_results()[0]
    assert result.suite.name == "unit"
    assert result.returncode == 124
    assert result.timed_out is True
    assert result.passed is False


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — output summary lists passed/failed suites + paths.
# ---------------------------------------------------------------------------


def test_render_summary_lists_every_suite_with_status() -> None:
    """The console summary must list each suite name plus pass/fail/skip."""
    report = GateReport(
        results=[
            SuiteResult(
                suite=_fake_suite("lint", 0),
                returncode=0,
                duration_seconds=1.5,
            ),
            SuiteResult(
                suite=_fake_suite("contract", 1),
                returncode=1,
                duration_seconds=2.5,
            ),
            SuiteResult(
                suite=_fake_suite("performance", 0),
                returncode=0,
                duration_seconds=0.0,
                skipped=True,
                skip_reason="explicitly skipped via --skip",
            ),
        ],
        total_duration_seconds=4.0,
        artifact_paths=["build/quality_gate.json"],
    )

    text = render_summary(report)
    assert "DataForge ML quality gate — FAILED" in text
    assert "lint" in text and "passed" in text
    assert "contract" in text and "failed" in text
    assert "performance" in text and "skipped" in text
    assert "build/quality_gate.json" in text
    assert "rc=1" in text


def test_main_writes_json_report_when_report_path_supplied(tmp_path: Path) -> None:
    """``--report`` must produce a deterministic JSON artifact."""
    fake_python = _success_python_stub(tmp_path)
    report_path = tmp_path / "build" / "quality_gate.json"

    rc = main(
        [
            "--python",
            str(fake_python),
            "--report",
            str(report_path),
            "--skip",
            *(_DEFAULT_SUITE_NAMES),  # every default suite skipped
        ]
    )
    # When everything is skipped the gate is a no-op pass.
    assert rc == 0
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    names = [r["name"] for r in payload["results"]]
    assert names == list(_DEFAULT_SUITE_NAMES)
    assert all(r["skipped"] for r in payload["results"])
    assert all(r["timed_out"] is False for r in payload["results"])
    assert all("timeout_seconds" in r for r in payload["results"])


def test_main_returns_non_zero_when_any_suite_fails(tmp_path: Path) -> None:
    """End-to-end: exit code must be non-zero when a real suite fails."""
    fake_python = _failing_python_stub(tmp_path)

    rc = main(
        [
            "--python",
            str(fake_python),
            # Skip every documented suite except a single tiny one to
            # keep the test fast.
            "--skip",
            "typecheck",
            "unit",
            "contract",
            "plugin",
            "security_privacy",
            "e2e_compute",
            "performance",
        ]
    )
    assert rc != 0


def test_parse_args_defaults_match_documented_runbook() -> None:
    args = parse_args([])
    assert args.python == sys.executable
    assert args.skip == []
    assert args.report is None
    assert args.suite_timeout_seconds == 600


# ---------------------------------------------------------------------------
# Make-target wiring.
# ---------------------------------------------------------------------------


def test_makefile_exposes_quality_gate_target() -> None:
    """``make quality-gate`` must be wired so the runbook command is real."""
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "quality-gate:" in text
    assert "tools.quality_gate" in text
    assert "build/quality_gate.json" in text


def test_runbook_documents_quality_gate_command() -> None:
    """The runbook must point operators at the unified quality gate command."""
    text = (REPO_ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "make quality-gate" in text or "tools.quality_gate" in text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_DEFAULT_SUITE_NAMES = (
    "lint",
    "typecheck",
    "unit",
    "contract",
    "plugin",
    "security_privacy",
    "e2e_compute",
    "performance",
)


def _fake_suite(name: str, returncode: int) -> Suite:
    # ``true`` / ``false`` are POSIX shell builtins backed by a tiny
    # binary; the gate runs them through subprocess so the suite-level
    # invocation is real.
    binary = "/bin/true" if returncode == 0 else "/bin/false"
    return Suite(
        name=name,
        description=f"fake {name}",
        command=(binary,),
        timeout_seconds=30,
    )


def _success_python_stub(tmp_path: Path) -> Path:
    """Write an executable shell script that exits with code 0."""
    return _python_stub(tmp_path, returncode=0)


def _failing_python_stub(tmp_path: Path) -> Path:
    """Write an executable shell script that exits non-zero."""
    return _python_stub(tmp_path, returncode=1)


def _python_stub(tmp_path: Path, *, returncode: int) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "python"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            exit {returncode}
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


# Silence unused-import warnings for helpers used only by skip paths.
_ = (subprocess, shutil, os)
