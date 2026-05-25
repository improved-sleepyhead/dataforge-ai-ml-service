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

## Configuration

The service reads configuration from environment variables through
`app.kernel.config.load_config`. Minimal local demo values:

```bash
export DATAFORGE_PROFILE=demo_strict
export DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL=http://localhost:9000
export DATAFORGE_OBJECT_STORAGE_BUCKET=dataforge-local
export DATAFORGE_PLATFORM_CALLBACK_URL=http://platform.local/api/ml/jobs/callback
export DATAFORGE_SERVICE_SIGNING_SECRET=local-dev-signing-secret
export DATAFORGE_DAGSTER_HOME=/tmp/dataforge-dagster
export DATAFORGE_POLICY_CONFIG_PATH=configs/policies/demo_strict.yaml
export DATAFORGE_DECISION_POLICY_PATH=configs/policies/decision_v0.yaml
export DATAFORGE_SCORE_POLICY_PATH=configs/policies/score_v0.yaml
```

Supported profiles are `demo_strict` and `banking_strict`. External AI access is
disabled by default through `DATAFORGE_ALLOW_EXTERNAL_API=false`.
`DATAFORGE_CONTRACT_PACK_VERSION` defaults to `local-fallback-v0.1.0-demo`.

## Contract Pack

Until the shared `dataforgeai-contracts` repository publishes canonical schemas,
this service carries a temporary fallback contract pack at
`contracts/local_fallback/v0.1.0-demo`. It includes JSON Schemas and examples for
core compute-plane contracts including artifacts, contexts, manifests,
prediction manifests, evidence, decision reports, method recommendations,
action plans, review queues, DataForge reports, export packages, and errors;
`make test-contracts` loads the pack and validates all examples without
requiring a real backend, frontend, or external contract repository.

### Contract Test Suite

`make test-contracts` runs the contract-side compatibility suite under
`tests/contracts/`. Two layers run on every invocation:

- `tests/contracts/test_contract_pack.py` loads `contract_pack.json`, validates
  each schema with `Draft202012Validator`, and validates every shipped example
  against its declared schema.
- `tests/contracts/test_pydantic_model_contract_compatibility.py` parses every
  required example through the corresponding `app.domain` Pydantic model and
  round-trips the model dump back through the JSON Schema. This guards against
  silent drift between Pydantic models and JSON Schemas for `ManifestRow`,
  `PredictionManifest`, `EvidenceBundle`, `DecisionReport`,
  `MethodRecommendation`, `ActionPlan`, `ReviewQueue`, `DataForgeReport`, and
  `ExportPackage`. A canary test additionally injects a forbidden extra field
  into each required example to confirm a corrupted example would actually
  break the suite.

The suite has no runtime dependency on the NestJS backend, the Next.js
frontend, real object storage, signing keys, or external AI providers.

### Updating Contract Fixtures

Whenever a contract surface changes — a new artifact, a new field, a new
example, or a new error code — apply the changes inside this repository in the
following order:

1. Update or add the JSON Schema in
   `contracts/local_fallback/v0.1.0-demo/schemas/<contract>.schema.json`. New
   schemas must use `additionalProperties: false` and JSON Schema draft 2020-12.
2. Add a representative payload in
   `contracts/local_fallback/v0.1.0-demo/examples/<example_name>.json`. Examples
   must be PII-free and deterministic.
3. Register both files in
   `contracts/local_fallback/v0.1.0-demo/contract_pack.json` under the
   appropriate `schemas` and `examples` lists.
4. Update or add the matching Pydantic model in `app/domain/`. Models should
   use `ConfigDict(extra="forbid", frozen=True)` so they reject unknown fields
   and stay aligned with the JSON Schema.
5. If the contract is one of the required contracts in
   `tests/contracts/test_contract_pack.py` or
   `tests/contracts/test_pydantic_model_contract_compatibility.py`, extend the
   expected schema/example sets and the parametrized model-compatibility cases
   so the new contract is exercised on every `make test-contracts` run.
6. Run `make test-contracts` to confirm the suite still passes. The contract
   pack version recorded in `DATAFORGE_CONTRACT_PACK_VERSION` and embedded in
   report metadata stays the same until the canonical
   `dataforgeai-contracts` repository publishes a new version.

Once the upstream `dataforgeai-contracts` repository ships canonical schemas,
the local fallback pack will be replaced or pinned to a published release; the
test suite layout above will continue to apply against the published pack.
