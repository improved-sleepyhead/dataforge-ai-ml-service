"""TASK-072: static checks for the dataforgeai-ml-service Jenkinsfile.

These tests lock the CI contract so a regression in stage names or
parameter wiring fails the suite without needing a real Jenkins
controller. Acceptance criteria covered:

1. Jenkinsfile runs install, lint, typecheck, unit tests, contract
   tests, plugin tests, security tests, E2E compute tests.
2. Pipeline builds a Docker image after successful tests.
3. Pipeline does not require a real backend.
4. Pipeline accepts ``CONTRACT_VERSION`` and ``IMAGE_TAG`` parameters.

Real Jenkinsfile linting (``jenkins-cli declarative-linter``) runs in
the live Jenkins controller; here we only enforce structural / textual
invariants.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
JENKINSFILE = REPO_ROOT / "Jenkinsfile"


def _jenkinsfile_text() -> str:
    return JENKINSFILE.read_text(encoding="utf-8")


def test_jenkinsfile_uses_declarative_pipeline_with_kubernetes_agents() -> None:
    """Top-level pipeline block must use a Kubernetes agent and modern syntax."""
    text = _jenkinsfile_text()

    assert text.startswith("// dataforgeai-ml-service CI pipeline."), (
        "Jenkinsfile must start with the documented banner so reviewers "
        "see the contract immediately"
    )
    assert "pipeline {" in text
    assert "agent {" in text
    assert "kubernetes" in text
    # Two agent containers: python for tests/lint, docker for image build.
    assert "name: python" in text
    assert "name: docker" in text


def test_jenkinsfile_runs_full_quality_gate_chain_in_order() -> None:
    """Every stage required by INFRA.md ML pipeline template is present and ordered."""
    text = _jenkinsfile_text()
    expected_stages = [
        "stage('Checkout')",
        "stage('Install')",
        "stage('Lint')",
        "stage('Typecheck')",
        "stage('Unit tests')",
        "stage('Contract tests')",
        "stage('Plugin tests')",
        "stage('Security tests')",
        "stage('E2E compute demo')",
        "stage('Performance acceptance')",
        "stage('FastAPI health smoke')",
        "stage('Dagster definitions load')",
        "stage('Build Docker image')",
        "stage('Scan image')",
        "stage('Push image')",
        "stage('Publish build metadata')",
    ]
    last_index = -1
    for stage in expected_stages:
        index = text.find(stage)
        assert index != -1, f"missing stage: {stage}"
        assert index > last_index, (
            f"stage {stage!r} appears before the previous expected stage; "
            "INFRA.md ML pipeline order is not preserved"
        )
        last_index = index


def test_jenkinsfile_invokes_all_make_quality_gates() -> None:
    """Every Make-driven quality gate is wired through the pipeline."""
    text = _jenkinsfile_text()
    expected_make_targets = (
        "make lint",
        "make typecheck",
        "make test",
        "make test-contracts",
        "make test-plugins",
        "make test-security",
        "make test-e2e-compute-demo",
        "make test-performance",
    )
    for target in expected_make_targets:
        assert target in text, f"Jenkinsfile is missing make target {target!r}"


def test_jenkinsfile_builds_docker_image_after_tests() -> None:
    """The Docker build step runs only after the test stages."""
    text = _jenkinsfile_text()

    build_index = text.find("stage('Build Docker image')")
    e2e_index = text.find("stage('E2E compute demo')")
    perf_index = text.find("stage('Performance acceptance')")

    assert build_index != -1
    assert e2e_index != -1
    assert perf_index != -1
    assert build_index > e2e_index
    assert build_index > perf_index

    # The build invocation itself uses the tag we resolved earlier.
    assert "docker build" in text
    assert "${env.RESOLVED_IMAGE}" in text


def test_jenkinsfile_declares_contract_version_and_image_tag_parameters() -> None:
    """The pipeline must accept CONTRACT_VERSION + IMAGE_TAG (+ IMAGE_NAME)."""
    text = _jenkinsfile_text()

    assert "name: 'CONTRACT_VERSION'" in text
    assert "name: 'IMAGE_TAG'" in text
    assert "name: 'IMAGE_NAME'" in text
    assert "name: 'PUSH_IMAGE'" in text
    # Defaults must point at the local fallback contract pack so the
    # pipeline can run without external contract distribution.
    assert "local-fallback-v0.1.0-demo" in text


def test_jenkinsfile_does_not_require_real_backend_or_real_secrets() -> None:
    """The pipeline must run with fake/local config only."""
    text = _jenkinsfile_text()

    # Defense in depth: the env block uses the documented fake/test
    # values, never references a Vault path or a production secret.
    assert "DATAFORGE_PROFILE" in text
    assert "demo_strict" in text
    assert "DATAFORGE_SERVICE_SIGNING_SECRET" in text
    assert "fake-signing-secret" in text
    # No live HashiCorp Vault integration: the substring may appear in
    # comments only.
    code_only = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("//")
    ).lower()
    assert "vault" not in code_only
    assert "kubectl" not in code_only
    # The pipeline must never declare a deployment step or Helm rollout.
    assert "helm " not in text
    assert "argocd" not in text.lower()


def test_jenkinsfile_runs_fastapi_health_and_dagster_definitions_smoke() -> None:
    """Pipeline must include FastAPI health smoke and Dagster definitions load checks."""
    text = _jenkinsfile_text()

    assert "/api/v1/health" in text
    assert "uvicorn" in text
    assert "build_local_demo_definitions" in text
    assert "dagster definitions loaded" in text


def test_jenkinsfile_has_post_metadata_publish_and_cleanup_steps() -> None:
    """Build metadata is archived and the local image is cleaned up."""
    text = _jenkinsfile_text()

    assert "Publish build metadata" in text
    assert "build/metadata.json" in text
    assert "archiveArtifacts" in text
    assert "docker image rm" in text


def test_jenkinsfile_image_scan_stage_is_present_but_optional() -> None:
    """Scan stage must exist (acceptance test step), but tolerate trivy absence."""
    text = _jenkinsfile_text()

    assert "stage('Scan image')" in text
    assert "trivy image" in text
    masked_scan = (
        "trivy image --exit-code 1 --severity CRITICAL,HIGH "
        "${env.RESOLVED_IMAGE} || true"
    )
    assert masked_scan not in text
    # The when-clause must guard on trivy availability so the
    # pipeline does not break on agents without the scanner installed.
    scan_index = text.index("stage('Scan image')")
    when_index = text.index("when {", scan_index)
    assert when_index > scan_index


def test_jenkinsfile_publishes_non_empty_junit_reports() -> None:
    """Critical pytest stages must write real JUnit XML and publish it."""
    text = _jenkinsfile_text()

    assert "PYTEST_ADDOPTS=\"--junitxml=build/junit/unit.xml\"" in text
    assert "PYTEST_ADDOPTS=\"--junitxml=build/junit/contracts.xml\"" in text
    assert "PYTEST_ADDOPTS=\"--junitxml=build/junit/plugins.xml\"" in text
    assert "PYTEST_ADDOPTS=\"--junitxml=build/junit/security.xml\"" in text
    assert "PYTEST_ADDOPTS=\"--junitxml=build/junit/e2e-compute.xml\"" in text
    assert "PYTEST_ADDOPTS=\"--junitxml=build/junit/performance.xml\"" in text
    assert "junit allowEmptyResults: false" in text
    assert "junit allowEmptyResults: true" not in text


def test_jenkinsfile_uses_timestamps_and_buildlog_options() -> None:
    """Pipeline carries the basic ops conveniences."""
    text = _jenkinsfile_text()

    assert "timestamps()" in text
    assert "buildDiscarder(logRotator" in text
    assert "timeout(time:" in text



# ---------------------------------------------------------------------------
# Optional Groovy syntax check.
# ---------------------------------------------------------------------------
#
# When ``groovy`` (and Java) are available locally, parse the Jenkinsfile
# through Groovy's ``CompilationUnit`` at the conversion phase so a
# Groovy-syntax regression fails the suite. The test skips cleanly on
# agents without a Groovy/Java toolchain so CI Python-only containers
# stay green.


def _has_groovy() -> bool:
    return shutil.which("groovy") is not None


@pytest.mark.skipif(not _has_groovy(), reason="groovy is not installed")
def test_jenkinsfile_parses_through_groovy_compilation_unit() -> None:
    """The Jenkinsfile must parse cleanly through Groovy's CompilationUnit."""
    parser = textwrap.dedent(
        """
        import org.codehaus.groovy.control.CompilationUnit
        import org.codehaus.groovy.control.CompilerConfiguration
        import org.codehaus.groovy.control.Phases

        def src = new File("Jenkinsfile").text
        def cu = new CompilationUnit(new CompilerConfiguration())
        cu.addSource("Jenkinsfile", src)
        cu.compile(Phases.CONVERSION)
        println "OK"
        """
    ).strip()

    result = subprocess.run(
        ["groovy", "-e", parser],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        "Groovy failed to parse Jenkinsfile.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout
