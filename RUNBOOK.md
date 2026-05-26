# DataForge AI ML Service — Compute Demo Runbook

> Operational runbook for the `dataforgeai-ml-service` repository: how
> to run the analyze-only demo, build an `ActionPlan`, apply selected
> actions, evaluate model impact, and produce an export package — all
> on the deterministic demo archive, **without a real backend**.

This runbook is the operator-facing companion to `docs/PRD.md`,
`docs/DOCS.md`, `docs/INFRA.md` and `docs/TESTING.md`. Read those first
if you need product or architecture context.

---

## 1. Local prerequisites

You only need a working Python 3.12 toolchain and (optionally) Docker.
Real MinIO, real backend, real signing keys, and external AI APIs are
**not required** for any of the documented runbook steps.

- Python 3.12 or 3.11.
- Git.
- Docker (only for `make docker-build` / image smoke).

Quick setup:

```bash
python -m venv .venv
PYTHON=.venv/bin/python make install-dev
```

That installs the FastAPI runtime deps (Pydantic, jsonschema, pyarrow,
scikit-learn, uvicorn) plus the dev toolchain (pytest, pytest-cov,
ruff, mypy, httpx, types-jsonschema). Dagster is needed for the
orchestration tests; install it explicitly:

```bash
.venv/bin/python -m pip install "dagster>=1.13,<2.0"
```

Set the demo env so `app.kernel.config.load_config(...)` succeeds:

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

The signing secret is **local-only** and never leaves the machine. The
real platform will inject its own secret through Kubernetes Secrets or
Vault.

---

## 2. Quality gates

Run these locally before opening a PR. CI runs the same chain.

```bash
PYTHON=.venv/bin/python make lint
PYTHON=.venv/bin/python make typecheck
PYTHON=.venv/bin/python make test
PYTHON=.venv/bin/python make test-contracts
PYTHON=.venv/bin/python make test-plugins
PYTHON=.venv/bin/python make test-security
PYTHON=.venv/bin/python make test-e2e-compute-demo
PYTHON=.venv/bin/python make test-performance
```

Targeted runs:

```bash
.venv/bin/python -m pytest tests/test_api_app.py -q
.venv/bin/python -m pytest tests/plugins/test_tabular_profile.py -q
.venv/bin/python -m pytest tests/security/test_service_signature.py -q
```

`tests/performance/test_compute_performance_acceptance.py` writes
`performance_report.json` either to `tmp_path` or, when set, to
`DATAFORGE_PERFORMANCE_REPORT_DIR`.

---

## 3. Deterministic demo archive

`tests/fixtures/demo_archive/builder.py` builds a deterministic
`demo_archive.zip` from a fixed seed (`20260520`) plus an
`expected_counts.json` for drift detection. The archive contains:

| File | What it carries | Why |
|---|---|---|
| `transactions.csv` | 200 rows; `is_fraud`, `customer_id_hash`, `case_id`, `amount`, `monthly_income`, `customer_segment`, `manual_review_flag` | Tabular dataset with rare class, missing income, duplicates, outliers, and a leakage column |
| `predictions.jsonl` | Per-row `predicted_proba`, `predicted_label`, `confidence`, `split`, `model_id`, `model_version`, `inference_timestamp` | Optional `PredictionManifest` — see §4 |
| `support_messages.jsonl` | Synthetic support messages with fake PII tokens | Text/OCR plugin proof input |
| `ocr_records.jsonl` | Synthetic OCR records with fake passport-like tokens | Text/OCR plugin proof input |
| `README.md` | Description for human reviewers | Provenance |

Demo invariants the suite locks in:

- 2% rare `is_fraud=1` class (within the 1–3% PRD target).
- ~55% missing `monthly_income` for `customer_segment=young_customers`.
- 4 exact-duplicate transaction pairs.
- 3 numeric outliers on `amount`.
- `manual_review_flag` is a leakage candidate fully aligned with `is_fraud=1`.
- 5 ambiguous prediction rows + 3 probable label-error candidates.

All values are synthetic. The archive must never be replaced by a
production CSV that contains real customer data.

---

## 4. Optional `PredictionManifest` input

`ANALYZE_ONLY` accepts an optional **predictions** artifact alongside
the dataset. It is **evidence**, never label authority. When supplied,
the analyzer surfaces two distinct signals:

- **`ambiguous_object_score`** — the model is uncertain. Indicators:
  low `confidence = max_k p_k`, low margin (`p_top1 - p_top2`), high
  `normalized_entropy = entropy / log(K)`. Routes to `LABEL_REVIEW` /
  `ACTIVE_LEARNING_QUEUE` with reason codes `ambiguous_object`,
  `high_model_uncertainty`, `low_prediction_margin`.
- **`probable_label_error_score`** — the model is confidently wrong vs
  the dataset label. Indicators: `predicted_label != true_label`,
  high `confidence`, large margin. Routes to `LABEL_REVIEW` (and
  optionally `BLOCK_TRAINING_UNTIL_REVIEW` per policy) with reason
  codes `probable_label_error`, `high_confidence_label_conflict`,
  `neighbor_label_disagreement`.

Tabular fraud illustrations from the demo:

```text
ambiguous:    true_label=fraud, predicted_proba={fraud:0.52, not_fraud:0.48}
label error:  true_label=not_fraud, predicted_proba={fraud:0.97, not_fraud:0.03}
```

If no `PredictionManifest` is supplied, the analyzer still runs base
diagnostics; prediction-derived metrics are surfaced as
`not_applicable` with `reason="prediction_manifest_not_provided"`.

---

## 5. Fake platform client and no-real-backend testing

The repository deliberately contains **no NestJS backend, no Prisma
schema, no production Vault wiring**. Tests prove the compute plane
works against a `FakePlatformMetadataClient`:

- The fake stores `JobEvent`s, `AuditEvent`s, and `ArtifactRef`s in
  memory.
- It redacts obvious PII / secret patterns even though the real status
  bridge already strips them upstream.
- Tests inspect `fake_platform.snapshot()` to confirm event order,
  stable stages (`QUEUED → INGESTING → BUILDING_MANIFEST → … →
  COMPLETED|FAILED|CANCELLED`), and absence of raw payload leaks.

The signed compute boundary (`app/api/security.py`) is exercised with
HMAC fixtures from `tests/security/test_service_signature.py`. User
JWTs are never accepted as authorization for compute requests — only
platform service signatures.

---

## 6. End-to-end compute demo

The end-to-end pipeline is exercised by two test files, each a real
operator runbook:

### 6.1 Analyze-only

`tests/e2e/test_compute_demo_analyze_only.py` runs:

1. Build `Asset Manifest` from the demo archive.
2. Validate the manifest against the contract pack.
3. Validate the optional `PredictionManifest` and produce a
   `prediction_validation_report` with coverage stats.
4. Build the `tabular_profile_report`.
5. Validate text/OCR sources, run PII detection, build the
   `text_ocr_report`.
6. Run `analyze_model_errors` to produce ambiguous and
   probable-label-error candidates.
7. Build `object_analytics_passports` + `evidence_bundles`.
8. Build the `decision_report` and `method_recommendations`.
9. Build the `review_queue` artifact.

`ANALYZE_ONLY` never produces candidate artifacts (no
`candidate_dataset_version`, no `prepared_dataset`, no
`synthetic_dataset`, no `export_package`). The test asserts every one
of those apply-only kinds is **absent** from registered storage.

```bash
PYTHON=.venv/bin/python make test-e2e-compute-demo
```

### 6.2 Full flow: analyze → action plan → apply → impact → export

`tests/e2e/test_compute_demo_full_flow.py` invokes
`launch_analyze_dataset_workflow(...)`, builds an `ActionPlan` from
`build_method_recommendations(...)` + `build_action_plan_preview(...)`,
then invokes `launch_apply_actions_workflow(...)`. The test asserts:

- ANALYZE materializes only analyze assets.
- APPLY materializes the full APPLY asset graph and produces:
  - a real `candidate_artifact_uri` and hash,
  - a `model_impact_artifact_uri`,
  - a `READY` `export_package_artifact_uri`.
- `synthetic_status="not_applicable"` for an imputation-only plan.
- The raw `raw_transactions` artifact bytes are byte-identical
  before and after APPLY (no source mutation).
- The fake platform receives `COMPLETED` lifecycle events for both
  ANALYZE and APPLY, plus the proposed candidate URI in event
  `details.artifact_uris`.
- `ExportPackage.artifacts` includes `DATASET_CARD` + `lineage_report`.
- `model_impact_report.json` validates against the contract pack with
  a non-`REJECTED` verdict.

A blocked-gate variant injects a `customer_email` column and flips
`pii_restricted=True`. The launcher records a `validation_gates_report`
artifact but never publishes a `candidate_artifact_uri` or
`export_package_artifact_uri` — proving "Export package создается
только если gates pass".

---

## 7. Expected demo blockers and recommendations

When the analyze-only flow runs against the deterministic demo archive,
the following findings are expected. They are the operator's primary
confidence signal — if the runbook produces a different shape, drift
has occurred.

| Finding | Where it surfaces | Reason code(s) |
|---|---|---|
| Rare-class imbalance (~2% `is_fraud=1`) | `tabular_profile_report.class_imbalance` | `severe_class_imbalance` |
| Segment-dependent missing `monthly_income` | `tabular_profile_report.missing_values` | `segment_dependent_missingness` |
| Exact duplicate transaction pairs | `tabular_profile_report.duplicates` | `exact_duplicate_rows` |
| Numeric outliers on `amount` | `tabular_profile_report.outliers` | `iqr_outlier`, `zscore_outlier` |
| Leakage candidate `manual_review_flag` | `tabular_profile_report.leakage_candidates` | `target_leakage_candidate` |
| PII tokens in support messages / OCR | `text_ocr_report.findings` | `pii_email`, `pii_phone`, `pii_passport_like` |
| Ambiguous prediction objects | `model_error_analysis_report` + `review_queue` | `ambiguous_object`, `high_model_uncertainty`, `low_prediction_margin` |
| Probable label-error candidates | `model_error_analysis_report` + `review_queue` | `probable_label_error`, `high_confidence_label_conflict` |

Recommended actions the demo run typically surfaces:

- `IMPUTE_MISSING_VALUES` for `monthly_income`: `group_median`
  recommended; `global_median` available; `pmm` disabled by policy;
  `target_imputation` blocked (target column auto-imputation is
  forbidden).
- `REMOVE_DUPLICATES` for exact duplicate rows.
- `SEND_TO_LABEL_REVIEW` for ambiguous and probable-label-error rows
  (separate reason codes).
- `BLOCK_LEAKAGE_COLUMN` for `manual_review_flag` (validation gate
  blocks until reviewer confirms).
- `AUGMENT_RARE_CLASS` with `smote` (split-safe; only on the train
  split; only after split + leakage gates pass). Borderline-SMOTE,
  ADASYN, CTGAN, TVAE remain `disabled_by_policy` in the MVP profile.

---

## 8. Performance acceptance

`tests/performance/test_compute_performance_acceptance.py` times the
six critical stages (`ingestion.manifest_builder`,
`predictions.validate`, `predictions.model_error.analyze`,
`tabular.profile`, `text_ocr.validate`, `review_queue.build`) on the
demo archive and writes `performance_report.json`. Defaults are
generous (5–10 s per stage) and overridable via
`DATAFORGE_PERFORMANCE_MAX_MS_<STAGE>` env so local hardware variance
does not flake the suite.

```bash
DATAFORGE_PERFORMANCE_REPORT_DIR=/tmp/dataforge-perf \
    PYTHON=.venv/bin/python make test-performance
cat /tmp/dataforge-perf/performance_report.json
```

---

## 9. Docker image and Jenkins CI

Build the image locally:

```bash
make docker-build              # builds dataforgeai-ml-service:local
docker run --rm -p 8000:8000 \
  -e DATAFORGE_OBJECT_STORAGE_ENDPOINT_URL=http://minio:9000 \
  -e DATAFORGE_OBJECT_STORAGE_BUCKET=dataforge-local \
  -e DATAFORGE_PLATFORM_CALLBACK_URL=http://platform.local/api/ml/jobs/callback \
  -e DATAFORGE_SERVICE_SIGNING_SECRET=local-dev-signing-secret \
  -e DATAFORGE_DAGSTER_HOME=/var/lib/dataforge/dagster \
  -e DATAFORGE_POLICY_CONFIG_PATH=configs/policies/demo_strict.yaml \
  -e DATAFORGE_DECISION_POLICY_PATH=configs/policies/decision_v0.yaml \
  -e DATAFORGE_SCORE_POLICY_PATH=configs/policies/score_v0.yaml \
  dataforgeai-ml-service:local
curl -s http://localhost:8000/api/v1/health
```

The image runs as the non-root `dataforge` user (uid=10001) and ships
no tests, no `.env`, no secrets. See `Dockerfile` and `.dockerignore`.

The `Jenkinsfile` runs the full quality gate chain plus `Build Docker
image`, optional `Scan image` (Trivy when available), optional `Push
image`, and `Publish build metadata`. Parameters: `CONTRACT_VERSION`,
`IMAGE_NAME`, `IMAGE_TAG`, `PUSH_IMAGE`. The pipeline never touches a
real backend or production secret.

---

## 10. Troubleshooting by stable error code

The compute service maps every recoverable failure to a stable
`ErrorCode` (see `app/domain/errors.py`). Each code has a documented
`remediation_hint` (see `app/api/error_taxonomy.py`).

| Error code | Likely cause on demo | Operator action |
|---|---|---|
| `INVALID_JOB_PAYLOAD` | Missing/typo'd field in the compute request | Re-validate the request shape; check `tests/test_api_app.py` for canonical examples |
| `UNSUPPORTED_MODALITY` | Disabled stub plugin selected (image/audio/video in MVP) | Use a `readiness=implemented`/`proof` plugin or enable the modality in admin profile |
| `ARTIFACT_NOT_FOUND` | Object storage URI does not exist yet | Verify ingestion finished; retry only after manifest build completes |
| `ARTIFACT_OUT_OF_SCOPE` | URI crossed organization/project/dataset prefix | Stay inside the signed scope; never reference foreign tenants |
| `INVALID_ARCHIVE_STRUCTURE` | Demo archive missing `transactions.csv` | Rebuild via `tests/fixtures/demo_archive/builder.py` |
| `ARCHIVE_SAFETY_VIOLATION` | zip slip / forbidden extension / oversized payload | Re-pack archive matching `app/ingestion/archive_safety.py` policy |
| `POLICY_BLOCKED` | Action disabled by the active profile (e.g. CTGAN in `demo_strict`) | Choose a `recommended`/`available` method or escalate policy |
| `PII_RESTRICTED` | PII detected in text/OCR while exporting | Redact, run privacy review, then retry |
| `LEAKAGE_DETECTED` | Train/test split or leakage column is poisoning the candidate | Recreate the split or drop the leaking column |
| `CONTRACT_VALIDATION_FAILED` | Plugin output broke its JSON Schema | Re-run after fixing the plugin output; align with the active contract pack version |
| `PREDICTION_VALIDATION_FAILED` | `predicted_proba` sum != 1, missing `confidence`, etc. | Re-emit predictions matching `prediction_manifest_row` schema |
| `PLUGIN_NOT_ENABLED` | Disabled or `contract-ready` plugin attempted to execute | Enable plugin in admin profile or pick a different capability |
| `PLUGIN_CONTRACT_FAILED` | Plugin claimed a capability it does not implement | Report plugin id/version; rerun |
| `PLUGIN_EXECUTION_FAILED` | Generic plugin runtime error | Retry once; escalate with the platform job id |
| `DAGSTER_RUN_FAILED` | Orchestration error inside a Dagster step | Inspect the `JobEvent` `FAILED` payload; retry once |
| `ACTION_PLAN_REQUIRES_APPROVAL` | APPLY launched without approval metadata | Backend must attach signed approval before the launcher accepts the request |
| `ACTION_PLAN_SIGNATURE_INVALID` | HMAC signature mismatch / stale timestamp / scope mismatch / unsigned protected request | Re-sign with current timestamp; never accept user JWTs as platform identity |
| `ACTION_PLAN_PRECONDITION_FAILED` | Step depends on a missing upstream artifact | Run upstream stages first |
| `VALIDATION_GATE_FAILED` | A gate inside `app/plugins/validation/gates.py` blocked the candidate | Fix the candidate (PII, leakage, schema, business rules) |
| `MODEL_IMPACT_NOT_ELIGIBLE` | Insufficient samples / no labeled split | Provide labeled validation/test or accept the eligibility fallback report |
| `EXPORT_BLOCKED` | Hard-policy gate failed on the export package | Resolve `blocked_reason_codes` listed in the package |
| `TENANT_SCOPE_VIOLATION` | Backend forwarded a request that crosses tenants | Reject at platform level; never forward |
| `EXTERNAL_API_BLOCKED` | External AI / network egress disabled in this profile | Enable through the explicit policy switch — strict banking still blocks regardless |
| `RESOURCE_LIMIT_EXCEEDED` | Compute quota exceeded | Reduce input size or request a quota increase |

`ErrorResponse` always carries a stable `code`, a safe `message`, a
`remediation_hint`, and `details.reason_code` when a plugin/kernel
emitted one. Raw exception text never leaks into the API boundary —
any plugin failure without a documented code falls through to the safe
500 path with `code=PLUGIN_EXECUTION_FAILED` and `recoverable=false`.

---

## 11. Verifying the runbook on a clean checkout

Acceptance test `Шаг 3` from TASK-073 — "convince yourself the runbook
works". On a fresh clone:

```bash
git clone <ml-service-repo> dataforgeai-ml-service
cd dataforgeai-ml-service
python -m venv .venv
PYTHON=.venv/bin/python make install-dev
.venv/bin/python -m pip install "dagster>=1.13,<2.0"

# Set the demo env block from §1, then:
PYTHON=.venv/bin/python make lint
PYTHON=.venv/bin/python make typecheck
PYTHON=.venv/bin/python make test
PYTHON=.venv/bin/python make test-contracts
PYTHON=.venv/bin/python make test-plugins
PYTHON=.venv/bin/python make test-security
PYTHON=.venv/bin/python make test-e2e-compute-demo
PYTHON=.venv/bin/python make test-performance
```

Every gate must pass. The E2E suite both runs the full compute flow
against the in-memory storage fake and validates that no
candidate / export artifact is produced when validation gates fail.

If any of the documented blockers (§7) does not appear, the
deterministic fixture has drifted — regenerate it via the builder and
update `expected_counts.json`.
