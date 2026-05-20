"""Unit tests for object manifest, prediction, passport, and evidence contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from app.domain import (
    EvidenceBundle,
    ManifestRow,
    ObjectAnalyticalPassport,
    PredictionManifest,
    PredictionRow,
    SignalStatus,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload


def example_payload(name: str) -> dict[str, Any]:
    pack = load_contract_pack()
    for example in pack.examples:
        if example.name == name:
            return example.payload
    raise AssertionError(f"missing example {name}")


def test_manifest_prediction_passport_and_evidence_examples_validate() -> None:
    pack = load_contract_pack()
    cases: list[tuple[str, str, type[BaseModel]]] = [
        ("manifest_row.tabular", "manifest_row", ManifestRow),
        ("prediction_manifest.fraud", "prediction_manifest", PredictionManifest),
        (
            "object_analytical_passport.with_prediction",
            "object_analytical_passport",
            ObjectAnalyticalPassport,
        ),
        ("evidence_bundle.with_prediction", "evidence_bundle", EvidenceBundle),
        ("evidence_bundle.no_prediction", "evidence_bundle", EvidenceBundle),
    ]

    for example_name, schema_name, model in cases:
        payload = example_payload(example_name)
        parsed = model.model_validate(payload)
        validate_contract_payload(pack, schema_name, parsed.model_dump(mode="json"))


def test_evidence_bundle_without_object_id_is_invalid() -> None:
    payload = example_payload("evidence_bundle.with_prediction")
    del payload["object_id"]

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_prediction_row_without_predicted_proba_is_invalid() -> None:
    manifest = example_payload("prediction_manifest.fraud")
    rows = cast(list[dict[str, Any]], manifest["rows"])
    row = dict(rows[0])
    del row["predicted_proba"]

    with pytest.raises(ValidationError):
        PredictionRow.model_validate(row)


def test_prediction_row_with_invalid_probability_map_is_invalid() -> None:
    manifest = example_payload("prediction_manifest.fraud")
    rows = cast(list[dict[str, Any]], manifest["rows"])
    row = dict(rows[0])
    row["predicted_proba"] = {"fraud": 0.4, "not_fraud": 0.4}

    with pytest.raises(ValidationError, match="sum to 1.0"):
        PredictionRow.model_validate(row)


def test_prediction_row_requires_argmax_label_and_confidence() -> None:
    manifest = example_payload("prediction_manifest.fraud")
    rows = cast(list[dict[str, Any]], manifest["rows"])
    row = dict(rows[0])
    row["predicted_label"] = "fraud"

    with pytest.raises(ValidationError, match="argmax"):
        PredictionRow.model_validate(row)


def test_missing_prediction_metrics_are_explicitly_not_applicable() -> None:
    evidence = EvidenceBundle.model_validate(example_payload("evidence_bundle.no_prediction"))

    assert evidence.signals.prediction_confidence.status is SignalStatus.NOT_APPLICABLE
    assert evidence.signals.prediction_confidence.value is None
    assert evidence.signals.prediction_confidence.reason == "prediction_manifest_not_provided"
