"""JSON Schema contract pack loading and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

JsonObject: TypeAlias = dict[str, Any]

DEFAULT_CONTRACT_PACK_ROOT = (
    Path(__file__).resolve().parents[2] / "contracts" / "local_fallback" / "v0.1.0-demo"
)


class ContractPackError(ValueError):
    """Raised when contract pack metadata, schemas, or examples are invalid."""


class ContractValidationError(ValueError):
    """Raised when a payload does not validate against a contract schema."""


@dataclass(frozen=True)
class ContractExample:
    """Contract example payload and its target schema."""

    name: str
    schema_name: str
    path: Path
    payload: JsonObject


@dataclass(frozen=True)
class ContractPack:
    """Loaded contract pack with schemas and examples."""

    root: Path
    version: str
    source: str
    schemas: Mapping[str, JsonObject]
    examples: tuple[ContractExample, ...]


def load_contract_pack(root: Path | None = None) -> ContractPack:
    """Load schemas and examples from the local fallback contract pack."""
    contract_root = DEFAULT_CONTRACT_PACK_ROOT if root is None else root
    manifest_path = contract_root / "contract_pack.json"
    manifest = _load_json(manifest_path)

    schemas: dict[str, JsonObject] = {}
    for schema_ref in _list_items(manifest, "schemas", manifest_path):
        schema_name = _required_str(schema_ref, "name", manifest_path)
        schema_path = contract_root / _required_str(schema_ref, "path", manifest_path)
        schema = _load_json(schema_path)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ContractPackError(f"Invalid JSON Schema {schema_name}: {exc.message}") from exc
        schemas[schema_name] = schema

    examples: list[ContractExample] = []
    for example_ref in _list_items(manifest, "examples", manifest_path):
        example_name = _required_str(example_ref, "name", manifest_path)
        schema_name = _required_str(example_ref, "schema", manifest_path)
        if schema_name not in schemas:
            raise ContractPackError(
                f"Example {example_name} references unknown schema {schema_name!r}"
            )
        example_path = contract_root / _required_str(example_ref, "path", manifest_path)
        examples.append(
            ContractExample(
                name=example_name,
                schema_name=schema_name,
                path=example_path,
                payload=_load_json(example_path),
            )
        )

    return ContractPack(
        root=contract_root,
        version=_required_str(manifest, "version", manifest_path),
        source=_required_str(manifest, "source", manifest_path),
        schemas=schemas,
        examples=tuple(examples),
    )


def validate_contract_payload(pack: ContractPack, schema_name: str, payload: JsonObject) -> None:
    """Validate a JSON object against a named schema from the contract pack."""
    schema = pack.schemas.get(schema_name)
    if schema is None:
        raise ContractPackError(f"Unknown contract schema: {schema_name}")

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if not errors:
        return

    first = errors[0]
    path = _error_path(first)
    raise ContractValidationError(f"{schema_name} validation failed at {path}: {first.message}")


def validate_contract_examples(pack: ContractPack | None = None) -> list[str]:
    """Validate all examples in a contract pack and return validated example names."""
    loaded_pack = load_contract_pack() if pack is None else pack
    validated: list[str] = []
    for example in loaded_pack.examples:
        try:
            validate_contract_payload(loaded_pack, example.schema_name, example.payload)
        except ContractValidationError as exc:
            raise ContractValidationError(f"Example {example.name} failed: {exc}") from exc
        validated.append(example.name)
    return validated


def _load_json(path: Path) -> JsonObject:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except OSError as exc:
        raise ContractPackError(f"Cannot read contract file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractPackError(f"Invalid JSON in contract file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractPackError(f"Contract file {path} must contain a JSON object")
    return payload


def _list_items(manifest: JsonObject, key: str, manifest_path: Path) -> list[JsonObject]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ContractPackError(f"{manifest_path} must contain a list field {key!r}")
    items: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            raise ContractPackError(f"{manifest_path} field {key!r} contains a non-object item")
        items.append(item)
    return items


def _required_str(payload: JsonObject, key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ContractPackError(f"{path} is missing required string field {key!r}")
    return value


def _error_path(error: JsonSchemaValidationError) -> str:
    if not error.path:
        return "$"
    return "$." + ".".join(str(part) for part in error.path)
