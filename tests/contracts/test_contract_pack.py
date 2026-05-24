"""Contract pack loading and validation tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.validation.contracts import (
    DEFAULT_CONTRACT_PACK_ROOT,
    ContractValidationError,
    load_contract_pack,
    validate_contract_examples,
    validate_contract_payload,
)


def test_local_contract_pack_loads_schemas_and_examples() -> None:
    pack = load_contract_pack()

    assert pack.version == "local-fallback-v0.1.0-demo"
    assert pack.source == "temporary_ml_service_fixture_until_dataforgeai_contracts_is_ready"
    assert {
        "artifact_ref",
        "compute_run",
        "dataset_version",
        "prediction_manifest",
        "platform_job",
        "error_response",
        "evidence_bundle",
        "manifest_row",
        "object_analytical_passport",
        "decision_report",
        "method_recommendation",
        "action_plan",
        "review_queue",
        "dataforge_report",
        "export_package",
        "split_manifest",
        "tabular_profile_report",
        "tabular_imputation_report",
        "text_ocr_report",
    } <= set(pack.schemas)
    assert "split_leakage_report" in pack.schemas
    assert "synthetic_dataset_report" in pack.schemas
    assert {example.name for example in pack.examples} == {
        "artifact_ref.basic",
        "compute_run.analyze_only",
        "dataset_version.context",
        "prediction_manifest.fraud",
        "platform_job.context",
        "error_response.invalid_job_payload",
        "manifest_row.tabular",
        "object_analytical_passport.with_prediction",
        "evidence_bundle.with_prediction",
        "evidence_bundle.no_prediction",
        "decision_report.needs_review",
        "method_recommendation.imputation",
        "action_plan.imputation_preview",
        "review_queue.label_review",
        "dataforge_report.analyze_only",
        "export_package.ready",
        "split_manifest.group_stratified",
        "split_leakage_report.demo",
        "synthetic_dataset_report.smote",
        "synthetic_dataset_report.gaussian_copula",
        "tabular_profile_report.fraud",
        "prediction_manifest_row.fraud",
        "tabular_imputation_report.group_median",
        "text_ocr_report.privacy",
    }


def test_all_contract_examples_validate() -> None:
    validated = validate_contract_examples()

    assert validated == [
        "artifact_ref.basic",
        "compute_run.analyze_only",
        "dataset_version.context",
        "prediction_manifest.fraud",
        "platform_job.context",
        "error_response.invalid_job_payload",
        "manifest_row.tabular",
        "object_analytical_passport.with_prediction",
        "evidence_bundle.with_prediction",
        "evidence_bundle.no_prediction",
        "decision_report.needs_review",
        "method_recommendation.imputation",
        "action_plan.imputation_preview",
        "review_queue.label_review",
        "dataforge_report.analyze_only",
        "export_package.ready",
        "split_manifest.group_stratified",
        "split_leakage_report.demo",
        "synthetic_dataset_report.smote",
        "synthetic_dataset_report.gaussian_copula",
        "tabular_profile_report.fraud",
        "prediction_manifest_row.fraud",
        "tabular_imputation_report.group_median",
        "text_ocr_report.privacy",
    ]


def test_prediction_manifest_example_validates_by_schema() -> None:
    pack = load_contract_pack()
    prediction_example = next(
        example for example in pack.examples if example.name == "prediction_manifest.fraud"
    )

    validate_contract_payload(pack, "prediction_manifest", prediction_example.payload)


def test_invalid_prediction_manifest_probability_map_fails() -> None:
    pack = load_contract_pack()
    prediction_example = next(
        example for example in pack.examples if example.name == "prediction_manifest.fraud"
    )
    invalid_payload = dict(prediction_example.payload)
    invalid_rows = [dict(row) for row in prediction_example.payload["rows"]]
    invalid_rows[0] = dict(invalid_rows[0])
    invalid_rows[0]["predicted_proba"] = {}
    invalid_payload["rows"] = invalid_rows

    with pytest.raises(ContractValidationError, match="prediction_manifest validation failed"):
        validate_contract_payload(pack, "prediction_manifest", invalid_payload)


def test_broken_temporary_example_makes_contract_test_fail(tmp_path: Path) -> None:
    broken_pack_root = tmp_path / "contract_pack"
    shutil.copytree(DEFAULT_CONTRACT_PACK_ROOT, broken_pack_root)
    example_path = broken_pack_root / "examples" / "prediction_manifest.fraud.json"
    payload = json.loads(example_path.read_text(encoding="utf-8"))
    del payload["rows"][0]["predicted_proba"]
    example_path.write_text(json.dumps(payload), encoding="utf-8")

    pack = load_contract_pack(broken_pack_root)
    with pytest.raises(ContractValidationError, match="Example prediction_manifest.fraud failed"):
        validate_contract_examples(pack)
