# dataforgeai-ml-service

Python/FastAPI compute plane для DataForge AI Dataset Intelligence Kernel.

Этот репозиторий содержит только ML service: внутренний FastAPI API, Dagster
compute workflow, доменные контракты, kernel, плагины, адаптеры, отчеты,
экспортные артефакты, telemetry и тестовые fake-интеграции. Он не содержит
Next.js frontend, NestJS backend/control plane, Prisma schema, пользовательскую
авторизацию, approval workflow и production deploy manifests.

## Быстрый старт

Требования:

- Python 3.11 или 3.12.
- Git.
- Docker опционально, только для сборки локального образа.

Создать окружение и установить зависимости:

```bash
python -m venv .venv
PYTHON=.venv/bin/python make install-dev
```

Запустить полный quality gate:

```bash
PYTHON=.venv/bin/python make quality-gate
```

Команда последовательно запускает lint, typecheck, unit, contract, plugin,
security/privacy, e2e и performance suites, затем пишет отчет в
`build/quality_gate.json`.

Запустить демо-сценарий MVP:

```bash
PYTHON=.venv/bin/python make run-mvp-demo
```

Демо поднимает локальный smoke для FastAPI, строит synthetic demo archive,
запускает `ANALYZE_ONLY`, собирает `ActionPlan`, выполняет
`APPLY_SELECTED_ACTIONS`, проверяет immutability raw artifacts и печатает
сводку export/model-impact/fake-platform events.

Запустить FastAPI локально:

```bash
.venv/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Полезные endpoint'ы:

```text
GET  /api/v1/health
GET  /api/v1/capabilities
POST /api/v1/jobs/analyze-dataset
POST /api/v1/action-plans/preview
POST /api/v1/action-plans/execute-approved
GET  /api/docs
```

Защищенные endpoint'ы требуют platform service signature. Пользовательские JWT
не являются источником авторизации для ML service.

## Локальные проверки

Отдельные команды:

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

Точечные pytest-запуски:

```bash
.venv/bin/python -m pytest tests/test_api_app.py -q
.venv/bin/python -m pytest tests/plugins/test_tabular_profile.py -q
.venv/bin/python -m pytest tests/security/test_service_signature.py -q
```

Сборка Docker image:

```bash
PYTHON=.venv/bin/python make docker-build
```

## Конфигурация для локального запуска

Минимальные demo env значения:

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

Поддержанные профили: `demo_strict` и `banking_strict`. External AI выключен по
умолчанию через `DATAFORGE_ALLOW_EXTERNAL_API=false`. Локальные тесты не требуют
реальный backend, MinIO, production signing keys, Vault или внешние AI API.

## Что делает сервис

ML service отвечает за compute-часть DataForge AI:

- безопасное чтение archive/object artifacts;
- построение `Asset Manifest`;
- validation contract examples и Pydantic domain models;
- tabular profiling: schema, missingness, duplicates, outliers, imbalance,
  leakage candidates и business rules;
- optional `PredictionManifest` ingestion и model-error analysis;
- text/OCR proof plugin: JSONL validation, duplicate checks, PII detection,
  redaction и privacy review queue;
- `ObjectAnalyticalPassport` и `EvidenceBundle`;
- `Decision Core`, `DataForge Score`, method recommendations и hard policy gates;
- `ActionPlan` preview и выполнение только после platform approval;
- candidate dataset artifacts без изменения raw inputs;
- split/leakage checks, imputation, duplicate actions, SMOTE/Gaussian Copula
  flows в рамках MVP policy;
- validation gates, model-impact report, version compare, lineage, dataset card;
- `ExportPackage` и export writers для tabular/redacted text-OCR;
- privacy-safe logs, in-process metrics/traces и status bridge events.

Основной безопасный workflow:

```text
raw archive refs
  -> ANALYZE_ONLY
  -> Asset Manifest
  -> profile / plugin diagnostics / prediction analysis
  -> Object Analytical Passports
  -> EvidenceBundle
  -> DecisionReport + MethodRecommendation + ReviewQueue
  -> ActionPlan preview
  -> platform approval
  -> APPLY_SELECTED_ACTIONS
  -> candidate / validation / model impact / export artifacts
```

`ANALYZE_ONLY` не мутирует датасет. Изменяющие операции разрешены только через
утвержденный `ActionPlan` и `APPLY_SELECTED_ACTIONS`. Raw artifacts не
перезаписываются.

## Архитектура репозитория

```text
app/
  api/             FastAPI boundary, request schemas, safe ErrorResponse, signing
  domain/          contract-compatible Pydantic models
  kernel/          config, policy gates, Decision Core, ActionPlan, scores, exports
  ingestion/       archive safety, readers, identity, manifests, predictions
  plugin_sdk/      PluginManifest, CapabilityRegistry, runtime policy contracts
  plugins/         tabular, text/OCR, prediction, validation, export modules
  orchestration/   Dagster assets/jobs/resources and status bridge
  adapters/        object storage, artifact registry, fake platform metadata client
  validation/      contract pack loader and JSON Schema validation
  telemetry/       privacy-safe logging, metrics, tracing, performance timing
  reports/         DataForge report, decision report, review queue, dataset card
contracts/         local fallback JSON Schema contract pack and examples
tests/             unit, contract, plugin, security, orchestration, e2e, performance
tools/             quality gate and MVP demo runner
```

Ключевой принцип: API handlers остаются тонкой границей, kernel не зависит от
FastAPI/Dagster/provider clients, плагины нормализуют outputs в contract-shaped
artifacts, adapters не протаскивают provider clients в domain models.

## FastAPI API

FastAPI app создается в `app.api.main:create_app`.

Публичные system endpoint'ы:

- `/api/v1/health` - статус сервиса, service version и contract pack version.
- `/api/v1/capabilities` - allowlisted plugin capabilities и readiness.

Protected compute endpoint'ы:

- `/api/v1/jobs/analyze-dataset` - запускает in-process Dagster
  `ANALYZE_ONLY` materialization.
- `/api/v1/action-plans/preview` - строит безопасный preview без мутации.
- `/api/v1/action-plans/execute-approved` - проверяет approved `ActionPlan` и
  запускает `APPLY_SELECTED_ACTIONS`.
- `/api/v1/jobs/{job_id}`, `/api/v1/action-plans/{action_plan_id}`,
  `/api/v1/reports/{report_id}`, `/api/v1/compare/{compare_id}` - compute-side
  read endpoints для тестового/локального API.

Подпись запроса проверяется в `app.api.security`: service identity, timestamp
freshness, HMAC-SHA256 signature, organization/project scope headers и scope в
payload. Ошибки нормализуются в стабильный `ErrorResponse`.

## Плагины

Реестр плагинов статический и allowlisted:

| Plugin | Уровень | Назначение |
|---|---|---|
| `dataforge.tabular` | implemented | Табличная диагностика, remediation, synthetic/action flows, exports |
| `dataforge.text_ocr` | proof | Text/OCR JSONL validation, PII, redaction, review queues |
| `image_stub` | contract-ready | Контрактная заготовка для image validation |
| `audio_stub` | contract-ready | Контрактная заготовка для audio metadata validation |
| `video_stub` | contract-ready | Контрактная заготовка для video metadata validation |

Plugin runtime policy запрещает network egress, запись за пределы scoped compute
outputs, мутацию source data и raw payloads в logs. Post-MVP усиление sandboxing
остается отдельной задачей.

## Адаптеры и хранилище

`MinioObjectStorageAdapter` работает с S3-compatible client protocol и
ограничивает доступ prefix'ом:

```text
<prefix_root>/<organization_id>/<project_id>/<dataset_id>/
```

URI вне scope отклоняются стабильной ошибкой `ARTIFACT_OUT_OF_SCOPE`. Сервис не
создает, не удаляет и не администрирует buckets.

`ArtifactRegistry` пишет immutable compute artifacts в content-addressed paths и
возвращает `ArtifactRef` с hash, schema version, job id, config hash и lineage.
PostgreSQL в этом репозитории не используется для хранения raw/generated
artifacts.

`FakePlatformMetadataClient` и fake callback server нужны для tests/dev. Они не
являются production adapter'ами.

## Dagster orchestration

Dagster definitions находятся в `app/orchestration`.

- `ANALYZE_ONLY` assets строят manifest, profile reports, prediction analysis,
  object passports, evidence bundles, decision report, method recommendations,
  review queue и DataForge report.
- `APPLY_SELECTED_ACTIONS` assets строят remediation report, prepared/candidate
  artifacts, synthetic artifacts при наличии action, validation gates,
  model-impact report, version compare, lineage, dataset card и export package.
- `RunStatusBridge` отправляет safe stage events в fake/platform client без raw
  PII/secrets.
- `build_local_demo_definitions()` создает local Dagster definitions на
  in-memory object storage и fake platform client.

Локальные FastAPI launchers используют `dagster.materialize(...)` in-process,
чтобы тесты доказывали workflow без реального Dagster daemon, backend и MinIO.

## Контракты

Пока внешний `dataforgeai-contracts` не публикует canonical pack, сервис несет
локальный fallback pack:

```text
contracts/local_fallback/v0.1.0-demo
```

В нем есть JSON Schema и examples для `ArtifactRef`, контекстов, manifest,
`PredictionManifest`, passport, evidence, decision/action/report/export/error
контрактов. `make test-contracts` проверяет JSON Schema, examples и Pydantic
round-trip совместимость.

При изменении публичной формы сначала обновляются schema/example/model/tests, и
только потом код, который эту форму производит.

## Документация

- [TECHNICAL_IMPLEMENTATION.md](TECHNICAL_IMPLEMENTATION.md) - подробная
  техническая заметка по устройству FastAPI, kernel, plugins, adapters и
  Dagster workflow.

## Границы ответственности

ML service не делает:

- frontend screens;
- NestJS controllers;
- Prisma schema/migrations;
- user auth/RBAC/ABAC;
- ownership approval workflow;
- final dataset version promotion;
- production Helm/Kustomize/Jenkins deploy pipeline;
- public ingress;
- bucket/database administration;
- arbitrary user code execution;
- logging raw PII, raw file contents, secrets или full signed request bodies.

Frontend обращается к backend. Backend вызывает ML service внутренним
подписанным запросом. Python compute plane считает, валидирует, пишет
производные артефакты и возвращает безопасные ссылки/отчеты.
