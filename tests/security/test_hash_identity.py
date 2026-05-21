"""Hash service and stable object_id generation tests for TASK-020.

Acceptance:

* content hashes are computed for files and object-level records;
* object_id is stable for a fixed (dataset_version_id, row_key, content_hash);
* re-running ingestion on the same archive produces the same object_id;
* hashes and object_ids are usable from downstream ManifestRow construction.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from app.ingestion import (
    compute_content_sha256,
    compute_record_sha256,
    derive_object_id,
    open_archive_path,
)
from tests.fixtures.demo_archive import build_demo_archive


def test_compute_content_sha256_for_bytes_matches_hashlib() -> None:
    payload = b"object_id,is_fraud\nrow_1,0\n"

    digest = compute_content_sha256(payload)

    assert digest == f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_compute_content_sha256_streams_in_chunks() -> None:
    payload = b"a" * (200 * 1024)
    stream = io.BytesIO(payload)

    digest = compute_content_sha256(stream)

    assert digest == f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_compute_record_sha256_is_canonical_across_dict_orderings() -> None:
    record_a = {"object_id": "txn_1", "amount": 10.0, "is_fraud": 0}
    record_b = {"is_fraud": 0, "amount": 10.0, "object_id": "txn_1"}
    record_c = {"object_id": "txn_1", "amount": 10.0, "is_fraud": 1}

    assert compute_record_sha256(record_a) == compute_record_sha256(record_b)
    assert compute_record_sha256(record_a) != compute_record_sha256(record_c)


def test_derive_object_id_is_stable_for_same_inputs() -> None:
    digest = "sha256:" + "a" * 64

    first = derive_object_id(
        dataset_version_id="v1",
        row_key="row_1",
        content_hash=digest,
    )
    second = derive_object_id(
        dataset_version_id="v1",
        row_key="row_1",
        content_hash=digest,
    )

    assert first == second
    assert first.startswith("obj_")


def test_derive_object_id_changes_when_any_input_changes() -> None:
    digest = "sha256:" + "a" * 64
    other_digest = "sha256:" + "b" * 64

    base = derive_object_id(
        dataset_version_id="v1",
        row_key="row_1",
        content_hash=digest,
    )
    different_version = derive_object_id(
        dataset_version_id="v2",
        row_key="row_1",
        content_hash=digest,
    )
    different_row = derive_object_id(
        dataset_version_id="v1",
        row_key="row_2",
        content_hash=digest,
    )
    different_hash = derive_object_id(
        dataset_version_id="v1",
        row_key="row_1",
        content_hash=other_digest,
    )

    assert base != different_version
    assert base != different_row
    assert base != different_hash


def test_derive_object_id_rejects_empty_or_malformed_inputs() -> None:
    digest = "sha256:" + "a" * 64

    with pytest.raises(ValueError):
        derive_object_id(dataset_version_id="", row_key="row", content_hash=digest)
    with pytest.raises(ValueError):
        derive_object_id(dataset_version_id="v1", row_key="", content_hash=digest)
    with pytest.raises(ValueError):
        derive_object_id(
            dataset_version_id="v1",
            row_key="row",
            content_hash="not-a-sha256",
        )


def test_repeated_archive_ingestion_yields_identical_object_ids(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)

    object_ids_first = _extract_transactions_object_ids(built.archive_path)
    object_ids_second = _extract_transactions_object_ids(built.archive_path)

    assert object_ids_first == object_ids_second
    assert all(value.startswith("obj_") for value in object_ids_first)


def test_changing_one_record_changes_only_its_hash_and_object_id(tmp_path: Path) -> None:
    built = build_demo_archive(output_dir=tmp_path)

    descriptors = _read_transactions_records(built.archive_path)
    assert len(descriptors) >= 2

    first_record = descriptors[0]
    mutated_record = {**first_record, "amount": "999999.99"}

    first_hash = compute_record_sha256(first_record)
    mutated_hash = compute_record_sha256(mutated_record)
    second_hash = compute_record_sha256(descriptors[1])

    assert first_hash != mutated_hash
    assert second_hash == compute_record_sha256(descriptors[1])

    first_object_id = derive_object_id(
        dataset_version_id="version_demo",
        row_key=first_record["object_id"],
        content_hash=first_hash,
    )
    mutated_object_id = derive_object_id(
        dataset_version_id="version_demo",
        row_key=first_record["object_id"],
        content_hash=mutated_hash,
    )
    assert first_object_id != mutated_object_id


def _extract_transactions_object_ids(archive_path: Path) -> tuple[str, ...]:
    object_ids: list[str] = []
    for record in _read_transactions_records(archive_path):
        content_hash = compute_record_sha256(record)
        object_ids.append(
            derive_object_id(
                dataset_version_id="version_demo",
                row_key=record["object_id"],
                content_hash=content_hash,
            )
        )
    return tuple(object_ids)


def _read_transactions_records(archive_path: Path) -> list[dict[str, str]]:
    import csv

    with open_archive_path(archive_path) as reader:
        descriptor = reader.find_required_transactions()
        with descriptor.open() as handle:
            text = handle.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))
