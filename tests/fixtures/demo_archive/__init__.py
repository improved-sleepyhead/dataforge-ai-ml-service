"""Deterministic DataForge AI demo archive fixtures.

This package builds a fake/synthetic ``demo_archive.zip`` used by ML-service
tests and local Dagster smoke runs. It is deterministic by ``seed`` and
contract-shaped:

* ``transactions.csv`` — tabular fraud-style dataset with rare class,
  segment-dependent missing ``monthly_income``, exact duplicates, numeric
  outliers, and a target-leakage candidate column.
* ``predictions.jsonl`` — model predictions joined to transactions by
  canonical Asset Manifest ``object_id`` with ``true_label``,
  ``predicted_label``, ``predicted_proba`` (sums to 1.0), ``confidence``,
  ``split``, ``model_id``, ``model_version``, ``inference_timestamp``. Includes
  at least one ambiguous object and one probable-label-error candidate.
* ``support_messages.jsonl`` and ``ocr_records.jsonl`` — text/OCR proof data
  with PII-like tokens and exact duplicates. The PII-like tokens use only
  fake/synthetic values so the fixture is safe to commit.

All values are synthetic; no real customer data, real PII, or real secrets
are present.
"""

from tests.fixtures.demo_archive.builder import (
    DEFAULT_SEED,
    DEMO_DATASET_VERSION_ID,
    DemoArchive,
    build_demo_archive,
    load_expected_counts,
)

__all__ = [
    "DEFAULT_SEED",
    "DEMO_DATASET_VERSION_ID",
    "DemoArchive",
    "build_demo_archive",
    "load_expected_counts",
]
