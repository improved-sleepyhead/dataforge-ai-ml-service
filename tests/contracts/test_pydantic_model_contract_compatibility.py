"""Cross-validate Pydantic domain models against contract pack examples.

This file is the contract-side counterpart of ``test_contract_pack``. It ensures
the Pydantic models exposed from ``app.domain`` stay in sync with the JSON
schemas and examples shipped in the local fallback contract pack.

For every contract listed in TASK-062 acceptance criteria the suite checks:

* the contract example parses through its Pydantic model,
* the model's JSON dump round-trips through the same JSON Schema,
* removing a required field from the model dump makes contract validation
  fail (so the suite would catch a future drift between models and schemas),
* a temporarily broken example causes the contract suite to fail (sanity
  check that the suite would catch future contract drift).

The test set is intentionally conservative: it does not assert business
content, only that the public contract surface remains compatible.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.domain import (
    ActionPlan,
    DataForgeReport,
    DecisionReport,
    EvidenceBundle,
    ExportPackage,
    ManifestRow,
    MethodRecommendation,
    PredictionManifest,
    ReviewQueue,
)
from app.validation.contracts import (
    DEFAULT_CONTRACT_PACK_ROOT,
    ContractValidationError,
    load_contract_pack,
    validate_contract_examples,
    validate_contract_payload,
)

# Mapping required by TASK-062 acceptance criteria:
#   ManifestRow, PredictionManifest, EvidenceBundle, DecisionReport,
#   MethodRecommendation, ActionPlan, ReviewQueue, ExportPackage.
#
# Each entry is (example_name, schema_name, pydantic_model).
REQUIRED_CONTRACT_CASES: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("manifest_row.tabular", "manifest_row", ManifestRow),
    ("prediction_manifest.fraud", "prediction_manifest", PredictionManifest),
    ("evidence_bundle.with_prediction", "evidence_bundle", EvidenceBundle),
    ("evidence_bundle.no_prediction", "evidence_bundle", EvidenceBundle),
    ("decision_report.needs_review", "decision_report", DecisionReport),
    ("method_recommendation.imputation", "method_recommendation", MethodRecommendation),
    ("action_plan.imputation_preview", "action_plan", ActionPlan),
    ("review_queue.label_review", "review_queue", ReviewQueue),
    ("dataforge_report.analyze_only", "dataforge_report", DataForgeReport),
    ("export_package.ready", "export_package", ExportPackage),
)


def _example_payload(name: str) -> dict[str, Any]:
    pack = load_contract_pack()
    for example in pack.examples:
        if example.name == name:
            return copy.deepcopy(dict(example.payload))
    raise AssertionError(f"missing contract example {name}")


@pytest.mark.parametrize(
    ("example_name", "schema_name", "model"),
    REQUIRED_CONTRACT_CASES,
    ids=[case[0] for case in REQUIRED_CONTRACT_CASES],
)
def test_contract_example_parses_through_pydantic_model(
    example_name: str,
    schema_name: str,
    model: type[BaseModel],
) -> None:
    """Each required contract example must parse through its Pydantic model."""
    payload = _example_payload(example_name)

    parsed = model.model_validate(payload)

    # The parsed model is a real instance of the expected class.
    assert isinstance(parsed, model)


@pytest.mark.parametrize(
    ("example_name", "schema_name", "model"),
    REQUIRED_CONTRACT_CASES,
    ids=[case[0] for case in REQUIRED_CONTRACT_CASES],
)
def test_pydantic_model_dump_roundtrips_through_contract_schema(
    example_name: str,
    schema_name: str,
    model: type[BaseModel],
) -> None:
    """Pydantic dump of a contract example must validate against the contract schema."""
    pack = load_contract_pack()
    payload = _example_payload(example_name)

    parsed = model.model_validate(payload)
    dump = parsed.model_dump(mode="json")

    # No silent dropping of fields: dump should validate against the same schema.
    validate_contract_payload(pack, schema_name, dump)


def test_contract_pack_has_examples_for_every_required_model() -> None:
    """All TASK-062 contracts must have at least one example shipped in the pack."""
    pack = load_contract_pack()
    available_examples = {example.name for example in pack.examples}
    missing = [
        example_name
        for example_name, _, _ in REQUIRED_CONTRACT_CASES
        if example_name not in available_examples
    ]

    assert not missing, f"contract pack is missing required examples: {missing}"


def test_required_schemas_are_registered_in_contract_pack() -> None:
    """All TASK-062 contract schemas must be registered in the pack manifest."""
    pack = load_contract_pack()
    registered_schemas = set(pack.schemas)
    required_schemas = {schema_name for _, schema_name, _ in REQUIRED_CONTRACT_CASES}

    missing = required_schemas - registered_schemas
    assert not missing, f"contract pack is missing required schemas: {missing}"


def test_evidence_bundle_without_object_id_is_invalid() -> None:
    """Drift guard: EvidenceBundle without object_id must fail Pydantic validation."""
    payload = _example_payload("evidence_bundle.with_prediction")
    payload.pop("object_id", None)

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_action_plan_step_without_idempotency_key_is_invalid_for_pydantic() -> None:
    """Drift guard: ActionPlanStep without idempotency_key must fail Pydantic validation."""
    payload = _example_payload("action_plan.imputation_preview")
    payload["steps"][0].pop("idempotency_key", None)

    with pytest.raises(ValidationError):
        ActionPlan.model_validate(payload)


def test_method_recommendation_without_recommended_method_is_invalid_for_schema() -> None:
    """Drift guard: a MethodRecommendation example missing recommended_method must fail schema."""
    pack = load_contract_pack()
    payload = _example_payload("method_recommendation.imputation")
    payload.pop("recommended_method", None)

    with pytest.raises(ContractValidationError, match="method_recommendation validation failed"):
        validate_contract_payload(pack, "method_recommendation", payload)


def test_export_package_without_status_is_invalid_for_pydantic() -> None:
    """Drift guard: ExportPackage without status must fail Pydantic validation."""
    payload = _example_payload("export_package.ready")
    payload.pop("status", None)

    with pytest.raises(ValidationError):
        ExportPackage.model_validate(payload)



@pytest.mark.parametrize(
    "example_name",
    [
        "manifest_row.tabular",
        "evidence_bundle.with_prediction",
        "decision_report.needs_review",
        "method_recommendation.imputation",
        "action_plan.imputation_preview",
        "review_queue.label_review",
        "export_package.ready",
    ],
)
def test_temporarily_broken_example_makes_contract_suite_fail(
    tmp_path: Path,
    example_name: str,
) -> None:
    """Sanity check: any required contract example, when broken, must fail the suite."""
    broken_root = tmp_path / "contract_pack"
    shutil.copytree(DEFAULT_CONTRACT_PACK_ROOT, broken_root)

    example_path = broken_root / "examples" / f"{example_name}.json"
    payload = json.loads(example_path.read_text(encoding="utf-8"))

    # Inject an invalid extra field at the top level. Domain Pydantic models use
    # ``extra="forbid"`` and JSON Schemas use ``additionalProperties=false``,
    # so this is guaranteed to break validation for any required contract.
    payload["__contract_drift_canary__"] = "should-fail"
    example_path.write_text(json.dumps(payload), encoding="utf-8")

    pack = load_contract_pack(broken_root)
    with pytest.raises(ContractValidationError, match=f"Example {example_name} failed"):
        validate_contract_examples(pack)
