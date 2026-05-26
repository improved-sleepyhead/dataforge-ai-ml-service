# DataForge AI ML Service — Known Limitations

> Honest accounting of what the MVP build of `dataforgeai-ml-service`
> actually ships vs what is contract-ready, proof-level, or out of
> scope. Read this together with [`README.md`](README.md),
> [`RUNBOOK.md`](RUNBOOK.md), and `docs/PRD.md` (§2.5.5 Strong-Team
> Implementation Scope and Claim Discipline).

The PRD readiness scale used throughout this document:

| Level | Meaning |
|---|---|
| **implemented** | End-to-end on the deterministic demo archive; returns contract-valid artifacts |
| **proof** | Works for a constrained input type or controlled demo case |
| **contract-ready** | Plugin manifest, schemas, ActionPlan support, and roadmap exist; production computation is **not** claimed |
| pilot-ready | Handles realistic data scale, failures, permissions, and validation gates (post-MVP) |
| bank-strict-ready | Tenant isolation, security review, audit, retention, resource limits, policy approval (post-MVP) |

The MVP ships **implemented + proof + contract-ready** levels only.

---

## 1. Plugin readiness in the MVP

| Plugin | Readiness | What it actually does on the demo |
|---|---|---|
| `dataforge.tabular` | implemented | Schema inference, missingness + segment-aware patterns, exact + duplicate-key + hash duplicates, IQR/z-score outliers, class imbalance + minority share + imbalance ratio + effective number of samples, business rules, target-leakage candidates, imputation advisor, SMOTE, Gaussian Copula synthesizer, validation gates, Parquet/CSV export writer |
| `dataforge.text_ocr` | proof | JSONL/text/OCR record validation, regex-based PII detection, exact + semantic-where-supported duplicate detection, redaction action, review queue entries for high-risk objects, redacted JSONL export |
| `image_stub` | contract-ready | Manifest validation only; **no** real bbox/polygon analysis, **no** image embeddings, **no** near-duplicate visual detection in the MVP |
| `audio_stub` | contract-ready | Manifest validation only; **no** sample-rate/silence/clipping/ASR analysis |
| `video_stub` | contract-ready | Manifest validation only; **no** keyframe extraction, fps/codec analysis, or temporal annotation checks |

Disabled-by-policy in the MVP `demo_strict` profile: PMM imputation,
Borderline-SMOTE, ADASYN, CTGAN, TVAE synthesizers. They have plugin
manifests and policy entries (`disabled_by_policy`); turning them on
requires explicit profile change + compute-budget review.

---

## 2. Synthetic data generation

Implemented MVP synthesizers:

- **SMOTE** for tabular rare-class augmentation. Split-safe (train
  split only), seeded, k-neighbors recorded in `ActionPlanStep.config`,
  generated rows tagged as `synthetic`, gated by privacy + business
  rules + model-impact validation gates.
- **Gaussian Copula** for distribution-level tabular synthesis. Demo
  profile only (`banking_strict` blocks it). Stdlib + scikit-learn
  numerical kernel; no SDV/CTGAN/TVAE under the hood.

Not implemented in the MVP:

- CTGAN / TVAE training (heavy GPU/CPU budget, contract policy
  `disabled_by_policy`).
- GMM / Bayesian GMM / Bayesian Networks synthesizers.
- TabDDPM and other diffusion-based tabular generators.
- DCR (distance-to-closest-record) and full nearest-neighbor privacy
  scoring for synthetic outputs. The MVP privacy gate is
  exact-duplicate-to-real only.
- TSTR / TRTS for the Gaussian Copula pipeline. Tabular SMOTE-driven
  candidates do compute model-impact metrics, but synthetic-only
  utility scoring is post-MVP.

---

## 3. Decision Core, scoring, and governance

Implemented:

- Hard-policy gates and reason-code registry (`DecisionPolicy v0`).
- DataForge Score v0 with completeness, validity, uniqueness, balance,
  privacy_safety, model_readiness, and lineage_completeness components
  plus penalty tracking; high score never overrides hard blockers.
- Object Value Score with configurable weights and decomposition
  (learning value, rarity, uncertainty, diversity, business
  importance, duplicate / quality / privacy / label-risk penalties).
- Method Recommendation Score covering imputation methods (per PRD
  §11.x weights) and synthetic-method gating.
- ActionPlan preview, integrity hash, signed-approval execution path,
  candidate version builder, validation gates, model-impact runner,
  fallback readiness report, version compare, lineage report, dataset
  card, ExportPackage with readiness gates including
  `blocked_objects_excluded`.
- Privacy-safe structured logs (with deterministic scanner), metrics
  registry, and tracing-span registry — all stdlib-only.

Not implemented:

- Learning-to-rank or RLHF-style decision learning. The Decision Core
  is rule-based + reason-code-driven; no historical-feedback model.
- Differential privacy. The MVP privacy gate is regex/structural; no
  DP mechanism, no privacy budget accounting, no DP synthesis.
- Federated training or cross-customer dataset analytics.
- Active learning loop closure (the MVP surfaces ambiguous-object and
  active-learning queues, but the loop closure is the platform's job).
- Foundation-model fine-tuning. DataForge prepares safe, lineage-
  tracked datasets; downstream training is outside the compute plane.
- Production-grade ASR / OCR engines. Text/OCR plugin is regex- and
  contract-driven; integration with Tesseract / paddleOCR / cloud OCR
  is post-MVP.

---

## 4. Compute boundary, infrastructure, and ops

Implemented:

- FastAPI compute API with stable `ErrorResponse` taxonomy and
  documented `remediation_hint` per error code.
- HMAC-signed service-to-service request validation with timestamp
  freshness, scope check, and explicit user-JWT rejection.
- `MinioObjectStorageAdapter` with `ObjectStorageScope` (organization
  / project / dataset prefix); URIs outside scope rejected with
  `ARTIFACT_OUT_OF_SCOPE`. No bucket admin operations exposed.
- `ArtifactRegistry` with content-addressed paths and idempotent
  re-writes.
- `FakePlatformMetadataClient` for tests, fake callback HTTP server,
  privacy-redacting in-memory event store.
- Dagster definitions: ANALYZE_ONLY skeleton + APPLY_SELECTED_ACTIONS
  graph, run-status bridge, idempotency keys, cancellation tokens,
  retry classification, and recoverable-failure reasons.
- Multi-stage Dockerfile with a non-root `dataforge` user, `tini`
  PID 1, `/api/v1/health` probe; `.dockerignore` blocks tests/secrets.
- Declarative `Jenkinsfile` with Kubernetes agents for the full
  quality gate chain plus image build / scan / push.
- `tools.quality_gate` unified command + JSON report.
- Deterministic performance acceptance suite with documented MVP
  thresholds.

Not implemented (out of scope for this repo):

- Helm charts / Kustomize overlays / production deploy pipeline.
  `dataforgeai-deploy` owns those per `docs/INFRA.md`.
- NestJS backend / Prisma schema. They live in a separate repository.
- Vault / KMS / external-secrets integration. The compute plane reads
  signed env / Kubernetes Secrets at runtime, but the secret
  provisioning side is platform-level.
- Public ingress for the ML service. Per PRD §3.1 the FastAPI app is
  internal-only; only the platform backend may call it.
- Real OpenLineage / MLflow / OpenTelemetry exporters. The
  `MetricsRegistry` and `TracingRegistry` are in-process only;
  shipping them to a backend is pilot work.
- Real Qdrant / Milvus integration. The MVP uses no vector store; the
  text/OCR plugin computes only lightweight similarity locally.
- Production CTGAN / TVAE / TabDDPM compute workers and GPU pools.
- Dagster sensors that trigger long-running cancellation (TASK-061
  recoverable failure classification is the dev-side handler; full
  Dagster RunFailureSensor + auto-retry policy lives in deploy).

---

## 5. Data, modalities, and external systems

| Capability | Status | Note |
|---|---|---|
| Tabular (CSV) | implemented | Up to ~200 rows on the demo archive; performance tested |
| Tabular (Parquet) | implemented | Same code path; the export writer also emits Parquet |
| Text JSONL | proof | Fixed-shape `support_messages.jsonl` records |
| OCR JSONL | proof | Fixed-shape `ocr_records.jsonl` records |
| Image manifests | contract-ready | Manifest schema only, no real image processing |
| Audio manifests | contract-ready | Manifest schema only, no ASR |
| Video manifests | contract-ready | Manifest schema only, no keyframe / temporal analysis |
| Multimodal cases | proof | Linked through `case_id` / `customer_id_hash` / `document_id` metadata in the manifest, but cross-modal ML is post-MVP |
| Label Studio export | contract-ready | Adapter slot exists; production export wiring is post-MVP |
| CVAT export | contract-ready | Adapter slot exists; production export wiring is post-MVP |
| External AI / OpenAI / OpenRouter | disabled by default | `DATAFORGE_ALLOW_EXTERNAL_API=false`; `banking_strict` profile blocks it even when the flag is mistakenly enabled |

---

## 6. Honest scope statement

The MVP build of `dataforgeai-ml-service` proves the **dataset
intelligence kernel** end-to-end on a deterministic tabular fraud
fixture, with a proof-level text/OCR extension and contract-ready
plugin scaffolding for image / audio / video / annotation flows.

It is **not** yet:

- a bank-grade certified anonymization service;
- a full multimodal training-data factory;
- a production Dagster deployment with HA workers and GPU pools;
- a substitute for the NestJS platform backend or the deploy pipeline.

Everything above is by design — see PRD §2.5.5 ("Strong-Team
Implementation Scope and Claim Discipline"): we do not understate the
architecture, but we also do not claim production maturity for any
capability we have not shipped end-to-end on the demo fixture.
