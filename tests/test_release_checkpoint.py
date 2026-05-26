"""TASK-075: static checks for the MVP release checkpoint.

These tests lock the documentation contract delivered with the
release: `KNOWN_LIMITATIONS.md` exists at the repo root, lists every
plugin readiness level, names the disabled-by-policy synthesizers, and
is linked from both `README.md` and `RUNBOOK.md`. They also assert
that the unified quality gate command is documented as the mandatory
pre-`status=done` step.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_LIMITATIONS = REPO_ROOT / "KNOWN_LIMITATIONS.md"


def _known_limitations_text() -> str:
    return KNOWN_LIMITATIONS.read_text(encoding="utf-8")


def test_known_limitations_lives_in_repo_root_and_is_committed() -> None:
    """KNOWN_LIMITATIONS.md must be at the repo root, not in gitignored docs/."""
    assert KNOWN_LIMITATIONS.exists()
    assert not (REPO_ROOT / "docs" / "KNOWN_LIMITATIONS.md").exists()


def test_known_limitations_documents_every_readiness_level() -> None:
    """All five PRD readiness levels must appear in the table."""
    text = _known_limitations_text()
    for level in ("implemented", "proof", "contract-ready", "pilot-ready", "bank-strict-ready"):
        assert level in text, f"missing readiness level: {level}"


def test_known_limitations_lists_every_mvp_plugin_with_readiness() -> None:
    """Every shipped plugin manifest must be enumerated with its readiness."""
    text = _known_limitations_text()
    expected_plugins = (
        "dataforge.tabular",
        "dataforge.text_ocr",
        "image_stub",
        "audio_stub",
        "video_stub",
    )
    for plugin in expected_plugins:
        assert plugin in text, f"missing plugin in MVP readiness table: {plugin}"


def test_known_limitations_names_disabled_by_policy_synthesizers() -> None:
    """Synthesizers gated by policy must be explicitly listed so reviewers know."""
    text = _known_limitations_text()
    for synth in ("PMM", "Borderline-SMOTE", "ADASYN", "CTGAN", "TVAE"):
        assert synth in text, f"missing disabled-by-policy synthesizer: {synth}"


def test_known_limitations_calls_out_external_ai_default_off() -> None:
    """External AI must be documented as disabled by default."""
    text = _known_limitations_text()
    assert "DATAFORGE_ALLOW_EXTERNAL_API=false" in text
    assert "banking_strict" in text


def test_known_limitations_documents_excluded_infrastructure() -> None:
    """The runbook must be honest about what is NOT in this repo."""
    text = _known_limitations_text()
    assert "Helm" in text or "helm" in text
    assert "NestJS" in text
    assert "Vault" in text
    assert "OpenLineage" in text or "OpenTelemetry" in text


def test_readme_links_to_known_limitations_and_quality_gate() -> None:
    """README must surface the limitations doc and the unified quality gate."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "KNOWN_LIMITATIONS.md" in text
    assert "make quality-gate" in text


def test_runbook_links_to_known_limitations() -> None:
    """The runbook must point operators at the limitations doc."""
    text = (REPO_ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "KNOWN_LIMITATIONS.md" in text


def test_progress_md_carries_mvp_release_summary() -> None:
    """progress.md must contain an MVP release summary entry."""
    text = (REPO_ROOT / "progress.md").read_text(encoding="utf-8")
    # The release checkpoint summary is appended after TASK-074.
    assert "mvp-release-checkpoint" in text
    assert "TASK-075" in text
