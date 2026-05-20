PYTHON ?= python

.PHONY: install-dev lint typecheck test test-contracts test-plugins test-security test-e2e-compute-demo

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check app tests

typecheck:
	$(PYTHON) -m mypy app tests

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
