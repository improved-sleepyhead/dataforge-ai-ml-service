"""TASK-071: static checks for the ML service Dockerfile and .dockerignore.

Acceptance criteria covered without launching a real Docker build:

1. Dockerfile builds a FastAPI/Dagster ML service image — verified by
   the presence of the `app.api.main:app` entrypoint, the dagster
   install in the builder stage, and a `python:` base image.
2. The image runs as a non-root user — verified by ``USER dataforge``
   and the explicit ``useradd``/``groupadd`` block.
3. The image does not bake secrets, .env files, or raw demo data —
   verified by `.dockerignore` excluding `tests`, `.env`, `.kiro`,
   `.agents`, `.codex`, secrets/keys, and progress/task workflow files.
4. The image exposes the health endpoint — verified by ``EXPOSE 8000``,
   the ``HEALTHCHECK`` curl probe, and the ``/api/v1/health`` URL.

These are static guarantees: a real Docker daemon is not required to
run the suite, and the build can still be exercised end-to-end via
``make docker-build`` when Docker is available.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_multistage_python_base_and_dagster_install() -> None:
    """The Dockerfile must declare a Python base and install dagster."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:" in text
    assert "AS builder" in text
    assert "AS runtime" in text
    # Editable install + dagster runtime — both are required by the
    # FastAPI compute API and the Dagster orchestration runtime.
    assert "/opt/venv/bin/pip install ." in text
    assert "dagster" in text


def test_dockerfile_runs_uvicorn_against_the_fastapi_entrypoint() -> None:
    """The runtime CMD must launch uvicorn against ``app.api.main:app``."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uvicorn" in text
    assert "app.api.main:app" in text
    assert "EXPOSE 8000" in text


def test_dockerfile_runs_as_non_root_user() -> None:
    """The runtime stage must drop privileges before CMD runs."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "useradd" in text
    assert "groupadd" in text
    assert "USER dataforge" in text
    # Defense in depth: ``USER`` must appear after the non-root user is
    # created (i.e. after the ``useradd`` line).
    user_index = text.index("USER dataforge")
    useradd_index = text.index("useradd")
    assert useradd_index < user_index


def test_dockerfile_declares_healthcheck_against_health_endpoint() -> None:
    """A HEALTHCHECK probing /api/v1/health must be declared."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "HEALTHCHECK" in text
    assert "/api/v1/health" in text
    # The healthcheck should hit localhost so Kubernetes/Docker daemons
    # can run it without DNS access.
    assert "127.0.0.1:8000" in text or "localhost:8000" in text


def test_dockerfile_declares_dagster_home_for_runtime() -> None:
    """The runtime image must point Dagster at a writable directory."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "DATAFORGE_DAGSTER_HOME" in text
    assert "/var/lib/dataforge/dagster" in text


def test_dockerfile_does_not_copy_tests_or_local_workflow_state() -> None:
    """The runtime image must not bake test fixtures or workflow files in."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    # The Dockerfile copies only ``app/``, ``contracts/``, and the
    # bare-minimum project metadata. It must never have a blanket
    # ``COPY . .`` statement that would slurp tests/ secrets/ .env.
    assert "COPY . " not in text
    assert "COPY ./" not in text
    # Defense in depth: we explicitly do not copy tests / progress / tasks.
    assert "COPY tests" not in text
    assert "COPY progress.md" not in text
    assert "COPY tasks.json" not in text


def test_dockerignore_excludes_tests_secrets_and_workflow_state() -> None:
    """The .dockerignore must exclude tests, secrets, and workflow files."""
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    excluded_patterns = {
        ".git",
        ".venv",
        "tests",
        "scripts",
        "progress.md",
        "tasks.json",
        ".kiro",
        ".agents",
        ".codex",
        ".env",
        "*.pem",
        "*.key",
        "secrets",
    }
    for pattern in excluded_patterns:
        assert pattern in text, f".dockerignore is missing pattern {pattern!r}"


def test_dockerignore_excludes_python_build_artifacts() -> None:
    """Python build/cache artifacts must not ship in the image."""
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "__pycache__" in text
    assert "*.egg-info" in text
    assert ".mypy_cache" in text
    assert ".ruff_cache" in text
    assert ".pytest_cache" in text


def test_pyproject_declares_uvicorn_runtime_dependency() -> None:
    """The runtime image runs uvicorn; it must be a declared runtime dep."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "uvicorn" in text
    # Must not be in the dev-only optional-dependencies section.
    runtime_block = text.split("[project.optional-dependencies]", 1)[0]
    assert "uvicorn" in runtime_block


def test_makefile_exposes_docker_build_target() -> None:
    """``make docker-build`` must be wired so deploy pipelines can call it."""
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "docker-build" in text
    assert "$(DOCKER) build" in text
