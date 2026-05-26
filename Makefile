PYTHON ?= python
DOCKER ?= docker
IMAGE ?= dataforgeai-ml-service:local

.PHONY: install-dev lint typecheck test test-contracts test-plugins test-security test-e2e-compute-demo test-performance docker-build quality-gate run-mvp-demo

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check app tools tests

typecheck:
	$(PYTHON) -m mypy app tools tests

test:
	$(PYTHON) -m pytest tests

test-contracts:
	$(PYTHON) -m pytest tests/contracts

test-plugins:
	$(PYTHON) -m pytest tests/plugins

test-security:
	$(PYTHON) -m pytest tests/security

test-e2e-compute-demo:
	$(PYTHON) -m pytest tests/e2e

test-performance:
	$(PYTHON) -m pytest tests/performance

docker-build:
	$(DOCKER) build --tag $(IMAGE) .

quality-gate:
	$(PYTHON) -m tools.quality_gate --python $(PYTHON) --report build/quality_gate.json

run-mvp-demo:
	$(PYTHON) -m tools.run_mvp_demo
