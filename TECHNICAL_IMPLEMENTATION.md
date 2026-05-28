# Техническая заметка по реализации dataforgeai-ml-service

Эта заметка фиксирует, как текущий Python ML service реализован технически:
FastAPI boundary, domain contracts, kernel, plugins, adapters, Dagster
orchestration, безопасность, артефакты и тесты.

## 1. Общая роль сервиса

`dataforgeai-ml-service` - это private compute plane DataForge AI. Он получает
внутренние signed requests от platform backend, читает raw artifacts из scoped
object storage, строит производные аналитические артефакты и возвращает ссылки
на результаты. Сервис не владеет пользователями, ролями, approval workflow,
dataset registry lifecycle или final version promotion.

Главная архитектурная идея:

```text
FastAPI = граница запроса и безопасных ответов
Domain = контрактные Pydantic-модели
Kernel = правила, scoring, planning, gates, export readiness
Plugins = modality-specific analysis/action implementations
Adapters = object storage / platform boundary
Dagster = deterministic asset orchestration
Reports = user-facing compute artifacts
Telemetry = privacy-safe technical observability
```

## 2. FastAPI слой

Точка входа: `app.api.main:create_app`.

FastAPI слой делает только boundary work:

- принимает request payloads через Pydantic schemas;
- проверяет service-to-service подпись;
- вызывает kernel/orchestration launchers;
- сохраняет compute-side read state в in-memory `ComputeResultStore`;
- нормализует ошибки в `ErrorResponse`;
- не содержит ML/business logic.

Стабильные endpoint группы:

```text
/api/v1/health
/api/v1/capabilities
/api/v1/jobs/analyze-dataset
/api/v1/action-plans/preview
/api/v1/action-plans/execute-approved
/api/v1/jobs/{job_id}
/api/v1/action-plans/{action_plan_id}
/api/v1/reports/{report_id}
/api/v1/reports/{report_id}/issues
/api/v1/compare/{compare_id}
/api/v1/jobs/{platform_job_id}/cancel
```

Middleware и exception handlers intentionally safe:

- unhandled exceptions превращаются в generic `PLUGIN_EXECUTION_FAILED`;
- validation errors возвращают `INVALID_JOB_PAYLOAD`;
- signature errors возвращают стабильный code/reason_code;
- raw traceback, raw file bytes, raw text payloads и secrets не попадают в
  response.

## 3. Подпись и compute boundary

Проверка находится в `app.api.security`.

Защищенный запрос должен содержать:

```text
X-DataForge-Service-Identity
X-DataForge-Timestamp
X-DataForge-Signature
X-DataForge-Organization-Id
X-DataForge-Project-Id
```

Подпись строится как versioned HMAC-SHA256 от canonical payload:

```text
v1
service_identity
timestamp
organization_id
project_id
sha256(body)
```

Проверяются:

- service identity равен configured platform identity;
- timestamp содержит timezone и укладывается в freshness window;
- HMAC совпадает через constant-time compare;
- organization/project scope в headers совпадает с payload scope;
- для GET endpoint'ов используется no-body variant, но scope headers все равно
  подписаны.

Python не интерпретирует пользовательские JWT как authorization source. Это
принципиально: авторизация пользователя и approval принадлежат backend/control
plane.

## 4. Domain contracts

Контрактные модели живут в `app/domain`.

Они описывают:

- `ArtifactRef`, lineage и общие scalar-типы;
- dataset/platform contexts и `ComputeRun`;
- `ManifestRow`, `PredictionManifest`, `PredictionRow`;
- `ObjectAnalyticalPassport`, `EvidenceBundle`;
- `DecisionReport`, `MethodRecommendation`, `ActionPlan`;
- `ReviewQueue`, `DataForgeReport`, `ExportPackage`;
- split/leakage/imputation/synthetic/model-impact/version-compare artifacts;
- stable `ErrorCode` taxonomy.

Модели используют строгую валидацию: extra fields forbidden, обязательные
organization/project/dataset/version/job fields не исчезают там, где есть доступ
к данным. Missing metrics задаются явно как `null` или `not_applicable`, а не
пропадают молча.

Локальный contract pack находится в:

```text
contracts/local_fallback/v0.1.0-demo
```

`tests/contracts` валидирует JSON Schemas, examples и Pydantic round-trip, чтобы
schema и runtime models не расходились.

## 5. Конфигурация

Config layer: `app.kernel.config`.

Сервис читает env через typed Pydantic settings-like слой и собирает:

- runtime profile: `demo_strict` или `banking_strict`;
- object storage endpoint/bucket/prefix;
- platform callback URL, service identity, signing secret и max signature age;
- Dagster settings;
- policy paths;
- contract pack version;
- external AI policy.

External AI disabled by default. `banking_strict` дополнительно блокирует
небезопасные внешние capability даже при ошибочной env-настройке.

## 6. Object storage и ArtifactRegistry

Object storage boundary находится в `app.adapters.object_storage`.

`MinioObjectStorageAdapter` работает не с конкретным boto3 type, а с минимальным
`S3CompatibleClient` protocol. Это позволяет тестировать код на in-memory fake и
потом подменить production MinIO/S3 client без протаскивания provider classes в
domain/kernel.

Scope задается через:

```text
organization_id
project_id
dataset_id
```

Разрешенный prefix:

```text
<prefix_root>/<organization_id>/<project_id>/<dataset_id>/
```

Все `put/get/head/list` операции проверяют scope. Path traversal, чужой bucket,
чужой prefix и out-of-scope URI отклоняются через стабильный error code.
Сервис не создает и не удаляет buckets.

`ArtifactRegistry` поверх storage пишет immutable artifacts:

```text
versions/<dataset_version_id>/jobs/<job_id>/artifacts/
  <artifact_kind>/<schema_version>/<sha256>.<format>
```

Если payload уже существует с тем же hash/metadata, запись idempotent. Если по
тому же пути найден объект с несовместимой metadata, registry падает с contract
validation error, а не перезаписывает artifact.

## 7. Ingestion

Ingestion слой находится в `app/ingestion`.

Основные компоненты:

- `archive_safety` - zip slip/path traversal/extension/size/count checks;
- `archive_reader` - streaming/lazy чтение archive descriptors;
- `identity` - stable hash и deterministic object_id generation;
- `manifest_builder` - построение `ManifestRow` для tabular/text/OCR/skeleton
  modalities;
- `manifest_validation` - JSON Schema + Pydantic validation;
- `predictions` - `PredictionManifest` ingestion и validation.

Raw archive превращается в versioned manifest artifact. Manifest row несет
`object_id`, `dataset_id`, `version_id`, modality, asset URI/hash, metadata и
lineage. Prediction artifacts не становятся source labels: это evidence для
анализа ошибок модели.

## 8. Plugin SDK и реестр capabilities

Plugin SDK находится в `app/plugin_sdk`.

Он задает:

- `PluginManifest`;
- `CapabilityDescriptor`;
- readiness level;
- resource requirements;
- runtime policy;
- `CapabilityRegistry` и `PluginManager`.

Реестр MVP статический и allowlisted (`app.plugins.registry`):

```text
dataforge.tabular      implemented
dataforge.text_ocr     proof
image_stub             contract-ready
audio_stub             contract-ready
video_stub             contract-ready
```

Runtime policy фиксирует, что plugin не может мутировать source data, не должен
писать raw payloads в logs, не имеет network egress и работает только с scoped
artifact refs / scoped compute outputs.

## 9. Реальные plugin implementations

### Tabular

Код: `app/plugins/tabular`.

Реализовано:

- schema inference;
- missing values и segment-aware missingness;
- duplicates;
- outliers;
- class imbalance;
- target leakage candidates;
- business rules;
- imputation actions;
- duplicate marking/removal candidate action;
- train/test split creation;
- split/leakage checks;
- SMOTE rare-class augmentation only on train split;
- Gaussian Copula synthetic flow, policy-gated;
- validation helper logic for candidate artifacts.

### Text/OCR

Код: `app/plugins/text_ocr`.

Реализовано на proof-level:

- JSONL validation;
- exact duplicate checks;
- regex-based PII detection;
- redaction action;
- privacy review queue inputs;
- redacted JSONL export через export writer.

### Predictions

Код: `app/plugins/predictions`.

Реализовано:

- validation of prediction rows;
- confidence, margin, entropy, normalized entropy;
- `ambiguous_object_score`;
- `probable_label_error_score`;
- reason codes для label review и active learning.

Предикты не меняют human labels и не могут approve export/mutation.

## 10. Object analytics и EvidenceBundle

`app/plugins/object_analytics/passports.py` строит
`ObjectAnalyticalPassport` из manifest rows и normalized diagnostics.

Passport содержит:

- identity и lineage;
- technical quality;
- privacy risk;
- duplicate signals;
- learning value stub/signals;
- prediction-derived ambiguity/label-error block;
- evidence refs;
- recommended action hints.

`app/plugins/object_analytics/evidence.py` строит `EvidenceBundle` только из
валидированных passports/outputs. Это важная граница: `Decision Core` не должен
читать plugin-private structures. Он потребляет только normalized signals и
evidence refs.

## 11. Kernel

Kernel modules в `app/kernel` реализуют decision/business слой без зависимости
от FastAPI, Dagster и provider clients.

Ключевые модули:

- `decision_policy` - hard gates и reason code registry;
- `decision_report` - dataset/object-level decisions;
- `dataforge_score` - score decomposition;
- `method_selection` - method recommendations;
- `action_plan` - preview, deterministic step ids, config hashes,
  idempotency keys и approval validation;
- `idempotency` - deterministic analyze/apply keys;
- `candidate_version` - candidate artifact descriptors без platform DB;
- `model_impact_eligibility` и `model_impact` - eligibility и sklearn baseline
  report;
- `version_compare` - сравнение source/candidate;
- `lineage` - lineage report;
- `export_package` - readiness gates и package summary;
- `external_api_policy` - запрет внешнего egress по умолчанию.

Hard policy gates имеют приоритет над score. Высокий DataForge Score не может
обойти privacy/leakage/validation blockers.

## 12. Dagster orchestration

Dagster слой находится в `app/orchestration`.

`build_definitions(...)` получает уже созданные typed adapters/resources и
оборачивает их в Dagster resources. Это сохраняет deterministic startup и не
заставляет Dagster CLI читать production secrets.

ANALYZE asset graph:

```text
raw_manifest
validated_manifest
prediction_validation_report
tabular_profile_report
text_ocr_report
model_error_report
object_analytics_passports
evidence_bundle
decision_report
recommended_actions
review_queue
dataforge_report
```

APPLY asset graph:

```text
action_plan
remediation_execution_report
prepared_dataset
synthetic_dataset
validation_gates_report
candidate_dataset_version
model_impact_report
version_compare_report
lineage_report
dataset_card
export_package
```

FastAPI launchers:

- `launch_analyze_dataset_workflow`;
- `launch_apply_actions_workflow`.

Они используют `dagster.materialize(...)` in-process, потому что текущий MVP и
tests должны работать без Dagster daemon, real backend и real MinIO. При этом
asset boundaries и materialization metadata остаются настоящими Dagster
концептами.

## 13. Run status bridge, retry и cancellation

`RunStatusBridge` переводит compute lifecycle в platform job events.

События содержат только safe technical fields:

```text
job_id
organization_id
project_id
dataset_id
version_id
stage
status
progress
duration
error_code
artifact refs
```

Raw PII, raw text, raw files и secrets в event payload не пишутся.

Cancellation реализована через `CancellationRegistry` и `CancellationToken`.
Launchers проверяют token до и после materialization. Failure classification
возвращает stable recoverability reason, чтобы platform UI мог отличать
recoverable failure от hard failure.

## 14. Reports и exports

`app/reports` и `app/kernel/export_package.py` собирают user-facing artifacts:

- DataForge report;
- Decision report;
- Review queues;
- dataset card;
- export package;
- lineage report;
- version compare;
- model impact report.

Export package публикуется только если validation/readiness gates не блокируют
candidate. При blocked gates могут быть записаны audit/validation artifacts, но
final candidate/export refs не публикуются как ready result.

## 15. Telemetry

Telemetry слой находится в `app/telemetry`.

Реализовано:

- structured privacy-safe logging;
- deterministic PII/secret scanner для log payloads;
- in-process metrics registry;
- in-process tracing registry;
- performance timing helpers и report writer.

Это не production OpenTelemetry exporter. Production metrics/traces contract и
exporter integration остаются отдельной post-MVP задачей.

## 16. Тестовая стратегия

Проект проверяется несколькими уровнями:

- `make lint` - ruff по `app tools tests`;
- `make typecheck` - strict mypy по `app tools tests`;
- `make test` - полный pytest suite;
- `make test-contracts` - contract pack, examples и Pydantic round-trip;
- `make test-plugins` - plugin compatibility и plugin outputs;
- `make test-security` - signatures, archive safety, scoped storage, privacy;
- `make test-e2e-compute-demo` - analyze-only и analyze/apply/export flows;
- `make test-performance` - deterministic performance acceptance;
- `make quality-gate` - единый canonical gate с JSON report.

Тесты используют deterministic demo archive, in-memory object storage,
`FakePlatformMetadataClient` и contract fixtures. Реальные customer data,
production credentials, real backend, PostgreSQL, Vault и MinIO не нужны.

## 17. MVP ограничения

Текущий сервис доказывает полный loop для tabular fraud demo, proof-level
text/OCR и contract-ready stubs для image/audio/video. Он не является:

- production bank-grade anonymization service;
- полноценной multimodal factory;
- production Dagster deployment;
- OpenLineage/MLflow/OpenTelemetry integrated service;
- backend/control plane replacement;
- deploy repository.

Подробный список ограничений находится в `KNOWN_LIMITATIONS.md`.
