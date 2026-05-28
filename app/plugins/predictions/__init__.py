"""Predictions plugin: model-error analysis on a validated PredictionManifest.

The plugin consumes :class:`PredictionRow` records (already validated by
``app.ingestion.predictions``) plus :class:`ManifestRow` labels and
produces a :class:`ModelErrorReport` separating ambiguous-object signals
from probable-label-error signals. Predictions are evidence, not
authority — the analyzer never mutates labels, never approves export and
never decides actions on its own.
"""

from app.plugins.predictions.analyzer import (
    analyze_model_errors,
    build_not_applicable_report,
    compute_object_signals,
)

__all__ = [
    "analyze_model_errors",
    "build_not_applicable_report",
    "compute_object_signals",
]
