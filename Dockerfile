# DataForge AI ML compute service image.
#
# - Multi-stage build: a builder layer compiles deps into a virtualenv,
#   the runtime layer copies only the venv + the source tree.
# - Runs uvicorn against ``app.api.main:app`` as a non-root user.
# - Exposes ``/api/v1/health`` for Kubernetes liveness/readiness probes.
# - Refuses to bake any secrets, .env files, or raw demo data into the
#   image (see .dockerignore).
#
# Local build:
#   docker build -t dataforgeai-ml-service:local .
# Local run:
#   docker run --rm -p 8000:8000 \
#     -e DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL=http://minio:9000 \
#     -e DATAFORGE_OBJECT_STORAGE_BUCKET=dataforge-local \
#     -e DATAFORGE_PLATFORM_CALLBACK_URL=http://platform.local/api/ml/jobs/callback \
#     -e DATAFORGE_SERVICE_SIGNING_SECRET=local-dev-signing-secret \
#     -e DATAFORGE_DAGSTER_HOME=/var/lib/dataforge/dagster \
#     -e DATAFORGE_POLICY_CONFIG_PATH=configs/policies/demo_strict.yaml \
#     -e DATAFORGE_DECISION_POLICY_PATH=configs/policies/decision_v0.yaml \
#     -e DATAFORGE_SCORE_POLICY_PATH=configs/policies/score_v0.yaml \
#     dataforgeai-ml-service:local

# ----------------------------------------------------------------------------
# Builder stage — install runtime deps into a self-contained virtualenv.
# ----------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System packages that pyarrow / scikit-learn wheels need at runtime are
# already vendored in the manylinux wheels we install. We still keep
# build-essential out of the runtime layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal sources required for an editable install.
COPY pyproject.toml README.md ./
COPY app ./app

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install . \
    && /opt/venv/bin/pip install "dagster>=1.13,<2.0"

# ----------------------------------------------------------------------------
# Runtime stage — minimal image with non-root user.
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    DATAFORGE_DAGSTER_HOME="/var/lib/dataforge/dagster" \
    DATAFORGE_PERFORMANCE_REPORT_DIR="/var/lib/dataforge/perf"

# curl is used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user/group for runtime.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid "${APP_GID}" dataforge \
    && useradd --system --uid "${APP_UID}" --gid "${APP_GID}" \
        --home-dir /home/dataforge --create-home \
        --shell /usr/sbin/nologin dataforge \
    && mkdir -p /var/lib/dataforge/dagster /var/lib/dataforge/perf \
    && chown -R dataforge:dataforge /var/lib/dataforge

WORKDIR /srv/dataforge-ai-ml-service

# Copy the virtualenv and the application source from the builder.
# Tests, contracts fixtures, scripts, .git and .venv are excluded by
# .dockerignore so they cannot accidentally ship in the image.
COPY --from=builder /opt/venv /opt/venv
COPY --chown=dataforge:dataforge app ./app
COPY --chown=dataforge:dataforge contracts ./contracts
COPY --chown=dataforge:dataforge pyproject.toml README.md ./

USER dataforge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent --show-error \
        "http://127.0.0.1:8000/api/v1/health" \
        || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--no-server-header"]
