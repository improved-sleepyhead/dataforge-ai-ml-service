# dataforgeai-ml-service

Python ML service for the DataForge AI Dataset Intelligence Kernel.

This repository owns the private compute plane for dataset analysis and
preparation. It is responsible for profiling, validation, manifest generation,
plugin execution, evidence normalization, Decision Core recommendations,
ActionPlan preview/execution after platform approval, model-impact evaluation,
reports, export artifacts, telemetry, and safe adapter boundaries.

It is not the DataForge platform backend and it is not the frontend. User
identity, RBAC/ABAC, project ownership, dataset registry lifecycle, approval
workflow, user-facing audit records, and final dataset promotion belong to the
separate NestJS backend/control-plane repository. The Next.js frontend is also
implemented in a separate repository and calls the backend, not this service.

## Repository Scope

The service follows the PRD microkernel layout:

```text
app/
  api/             FastAPI request boundary and safe response handlers
  kernel/          policies, contract validation, Decision Core, action planning
  domain/          contract-compatible domain models
  plugin_sdk/      stable interfaces exposed to allowlisted plugins
  ingestion/       archive reading, manifest building, modality detection, hashes
  plugins/         modality and algorithm implementations
  orchestration/   Dagster assets, jobs, resources, and run status bridge
  adapters/        object storage, platform metadata, vector DB, MLflow, lineage
  validation/      Pydantic/Pandera/contract validation helpers
  telemetry/       privacy-safe logging, metrics, tracing, audit events
  reports/         reports, dataset cards, review queues, export packages
tests/
```

## Compute-Plane Boundaries

- Python receives signed internal requests from the platform backend.
- Python validates request shape, service identity, project scope, policy
  constraints, and object-storage prefixes.
- Python writes derived artifacts, reports, evidence, review queues, and export
  packages to object storage.
- Python does not authenticate end users, approve actions, own RBAC, promote
  final dataset versions, implement Next.js screens, implement NestJS
  controllers, or define Prisma schema/migrations.
- `ANALYZE_ONLY` workflows must not mutate source dataset artifacts. Dataset
  changes are allowed only through approved `ActionPlan` /
  `APPLY_SELECTED_ACTIONS` flows.

## Safety Baseline

- Raw files and generated artifacts live in object storage, not application
  logs or PostgreSQL.
- Raw PII, raw text payloads, raw file contents, secrets, signed request bodies,
  and production credentials must not be logged.
- External AI/network egress is disabled by default and must be policy-gated.
- Plugins produce normalized, contract-compatible evidence; Decision Core
  consumes `EvidenceBundle`-style outputs rather than plugin-private data.

## Local Tooling

Create a local virtual environment and install development dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
make install-dev
```

Run quality gates:

```bash
make lint
make typecheck
make test
make test-contracts
make test-plugins
make test-security
make test-e2e-compute-demo
```

The local setup uses only development dependencies declared in
`pyproject.toml`. It does not require production secrets, object-storage
credentials, platform signing keys, or external AI credentials.
