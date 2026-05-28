"""Unit tests for basic compute-plane domain contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain import (
    ArtifactRef,
    ComputeRun,
    DatasetVersionContext,
    ErrorCode,
    ErrorResponse,
    PlatformJobContext,
)
from app.validation.contracts import load_contract_pack, validate_contract_payload


def example_payload(name: str) -> dict[str, object]:
    pack = load_contract_pack()
    for example in pack.examples:
        if example.name == name:
            return example.payload
    raise AssertionError(f"missing example {name}")


def test_artifact_ref_model_matches_contract_example() -> None:
    pack = load_contract_pack()
    artifact = ArtifactRef.model_validate(example_payload("artifact_ref.basic"))

    assert artifact.artifact_id == "artifact_manifest_001"
    validate_contract_payload(pack, "artifact_ref", artifact.model_dump(mode="json"))


def test_dataset_version_context_model_matches_contract_example() -> None:
    pack = load_contract_pack()
    context = DatasetVersionContext.model_validate(example_payload("dataset_version.context"))

    assert context.organization_id == "org_1"
    assert context.project_id == "project_1"
    validate_contract_payload(pack, "dataset_version", context.model_dump(mode="json"))


def test_platform_job_context_model_matches_contract_example() -> None:
    pack = load_contract_pack()
    context = PlatformJobContext.model_validate(example_payload("platform_job.context"))

    assert context.organization_id == "org_1"
    assert context.project_id == "project_1"
    validate_contract_payload(pack, "platform_job", context.model_dump(mode="json"))


def test_compute_run_model_matches_contract_example() -> None:
    pack = load_contract_pack()
    compute_run = ComputeRun.model_validate(example_payload("compute_run.analyze_only"))

    assert compute_run.platform_job_id == "platform_job_001"
    validate_contract_payload(pack, "compute_run", compute_run.model_dump(mode="json"))


def test_error_response_model_matches_contract_example() -> None:
    pack = load_contract_pack()
    response = ErrorResponse.model_validate(example_payload("error_response.invalid_job_payload"))

    assert response.error.code is ErrorCode.INVALID_JOB_PAYLOAD
    validate_contract_payload(pack, "error_response", response.model_dump(mode="json"))


def test_dataset_context_requires_organization_and_project() -> None:
    payload = example_payload("dataset_version.context")
    payload.pop("organization_id")
    payload.pop("project_id")

    with pytest.raises(ValidationError):
        DatasetVersionContext.model_validate(payload)


def test_error_response_without_code_is_invalid() -> None:
    payload = example_payload("error_response.invalid_job_payload")
    error = payload["error"]
    assert isinstance(error, dict)
    del error["code"]

    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(payload)
