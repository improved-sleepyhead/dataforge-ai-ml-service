"""TASK-073: static checks for the compute demo runbook.

These tests lock the runbook structure so a regression (deleted
troubleshooting table, missing prediction-signal explanation, etc.)
fails CI without needing a human reviewer to remember every section.

They cover the five acceptance criteria from ``tasks.json``:

1. Runbook describes local dependencies, fixtures, analyze-only,
   action plan, apply, and export.
2. Runbook explains optional ``PredictionManifest`` input plus the
   ``ambiguous_object_score`` and ``probable_label_error_score``
   signals from the demo predictions fixture.
3. Runbook explains the ``FakePlatformMetadataClient`` and the
   no-real-backend testing posture.
4. Runbook lists expected blockers / recommendations for the demo
   archive.
5. Runbook contains a troubleshooting table keyed by every stable
   error code in :class:`app.domain.ErrorCode`.
"""

from __future__ import annotations

from pathlib import Path

from app.domain import ErrorCode

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = REPO_ROOT / "RUNBOOK.md"


def _runbook_text() -> str:
    return RUNBOOK_PATH.read_text(encoding="utf-8")


def test_runbook_lives_in_repo_root_and_is_committed() -> None:
    """Runbook must be in the repo root, not in the gitignored docs/ folder."""
    assert RUNBOOK_PATH.exists(), "RUNBOOK.md must be committed at the repo root"
    # docs/ is ``/docs`` in .gitignore; never put the runbook there.
    assert not (REPO_ROOT / "docs" / "RUNBOOK.md").exists(), (
        "RUNBOOK.md must not live inside the gitignored docs/ folder"
    )


def test_runbook_describes_local_dependencies_quality_gates_and_demo_flow() -> None:
    """AC1: local deps → fixtures → analyze → action plan → apply → export."""
    text = _runbook_text()

    assert "make install-dev" in text
    assert "make lint" in text
    assert "make typecheck" in text
    assert "make test" in text
    assert "make test-contracts" in text
    assert "make test-plugins" in text
    assert "make test-security" in text
    assert "make test-e2e-compute-demo" in text
    assert "make test-performance" in text
    # Demo archive walkthrough.
    assert "demo_archive.zip" in text
    assert "transactions.csv" in text
    assert "predictions.jsonl" in text
    assert "support_messages.jsonl" in text
    assert "ocr_records.jsonl" in text
    # Each compute stage is named.
    assert "Asset Manifest" in text
    assert "ActionPlan" in text
    assert "ANALYZE_ONLY" in text
    assert "APPLY" in text
    assert "ExportPackage" in text or "export_package" in text
    assert "model_impact_report" in text


def test_runbook_explains_predictionmanifest_and_ambiguous_vs_label_error() -> None:
    """AC2: PredictionManifest, demo predictions fixture, ambiguous + label error."""
    text = _runbook_text()

    assert "PredictionManifest" in text
    assert "ambiguous_object_score" in text
    assert "probable_label_error_score" in text
    # Reason codes the kernel emits for each signal.
    assert "ambiguous_object" in text
    assert "high_model_uncertainty" in text
    assert "low_prediction_margin" in text
    assert "probable_label_error" in text
    assert "high_confidence_label_conflict" in text
    # Demo example must be unambiguous about which fixture row is which.
    assert "predicted_proba" in text
    assert "fraud" in text
    assert "not_applicable" in text  # behavior when no prediction manifest is supplied


def test_runbook_explains_fake_platform_client_and_no_real_backend() -> None:
    """AC3: FakePlatformMetadataClient + no real backend / real signing key."""
    text = _runbook_text()

    assert "FakePlatformMetadataClient" in text
    assert "no real backend" in text or "no NestJS backend" in text
    assert "fake.snapshot()" in text or "fake_platform.snapshot()" in text
    assert "JobEvent" in text
    # Compute boundary / signing wiring.
    assert "service signature" in text or "service signatures" in text
    # Defense-in-depth: user JWTs must never be treated as service identity.
    normalized = " ".join(text.split())
    assert (
        "User JWTs are never accepted" in normalized
        or "user jwt" in normalized.lower()
    )


def test_runbook_lists_expected_demo_blockers_and_recommendations() -> None:
    """AC4: expected blockers + recommendations for the demo archive."""
    text = _runbook_text()

    expected_blockers = (
        "severe_class_imbalance",
        "segment_dependent_missingness",
        "exact_duplicate_rows",
        "iqr_outlier",
        "target_leakage_candidate",
        "pii_email",
        "pii_phone",
        "ambiguous_object",
        "probable_label_error",
    )
    for blocker in expected_blockers:
        assert blocker in text, f"runbook is missing expected blocker {blocker!r}"

    expected_recommendations = (
        "IMPUTE_MISSING_VALUES",
        "REMOVE_DUPLICATES",
        "SEND_TO_LABEL_REVIEW",
        "BLOCK_LEAKAGE_COLUMN",
        "AUGMENT_RARE_CLASS",
        "group_median",
        "smote",
        "disabled_by_policy",
    )
    for recommendation in expected_recommendations:
        assert recommendation in text, (
            f"runbook is missing expected recommendation {recommendation!r}"
        )


def test_runbook_troubleshooting_covers_every_stable_error_code() -> None:
    """AC5: every ErrorCode value must appear in the troubleshooting section."""
    text = _runbook_text()

    troubleshooting_index = text.find("Troubleshooting by stable error code")
    assert troubleshooting_index != -1, "runbook must have a troubleshooting section"
    # Restrict the lookup so a stray mention earlier in the runbook does
    # not paper over a missing troubleshooting entry.
    troubleshooting_block = text[troubleshooting_index:]

    for code in ErrorCode:
        assert code.value in troubleshooting_block, (
            f"runbook troubleshooting table is missing ErrorCode {code.value!r}"
        )


def test_readme_links_to_the_runbook() -> None:
    """README must point operators at the runbook so it is discoverable."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "RUNBOOK.md" in readme
