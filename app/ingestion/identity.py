"""Hash service and stable object_id generation for ingestion.

This module provides two small, pure helpers used by the manifest builder:

* :func:`compute_content_sha256` — content sha256 for raw bytes or a binary
  stream. Streams are read in fixed-size chunks so very large entries do
  not balloon memory.
* :func:`compute_record_sha256` — content sha256 for a structured record,
  computed from a canonical JSON encoding (sorted keys, no whitespace,
  deterministic separators) so logically equal records always produce the
  same digest regardless of dict iteration order.
* :func:`derive_object_id` — deterministic ``object_id`` derivation from
  ``(dataset_version_id, row_key, content_hash)``. Two runs over the same
  input produce identical ``object_id`` values; changing any one of the
  three inputs changes the ``object_id`` deterministically.

These helpers are pure stdlib and never log payload content.
"""

from __future__ import annotations

import hashlib
import json
from typing import IO, Any

_STREAM_CHUNK_SIZE = 64 * 1024


def compute_content_sha256(data: bytes | IO[bytes]) -> str:
    """Return ``sha256:`` digest of raw bytes or a binary stream.

    For ``IO[bytes]`` the stream is consumed in fixed-size chunks, so the
    caller is expected to position it at the start before calling this
    function. The function never reads the entire stream into memory at once.
    """
    digest = hashlib.sha256()
    if isinstance(data, bytes):
        digest.update(data)
        return f"sha256:{digest.hexdigest()}"

    while True:
        chunk = data.read(_STREAM_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def compute_record_sha256(record: dict[str, Any]) -> str:
    """Return ``sha256:`` digest of a record using a canonical JSON encoding.

    Canonical encoding pins ``sort_keys=True`` and a deterministic separator
    so equivalent records produce equal digests regardless of insertion
    order. Non-string keys are not allowed (json will reject them); values
    must be JSON-serializable.
    """
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def derive_object_id(
    *,
    dataset_version_id: str,
    row_key: str,
    content_hash: str,
) -> str:
    """Derive a stable ``object_id`` from version + row key + content hash.

    The derivation is deterministic: the same triple of inputs always yields
    the same output. The output is short enough for log lines and stable
    across reruns of ingestion on the same archive.

    The function rejects empty inputs and content hashes that are not in
    the documented ``sha256:<hex>`` shape so callers can not produce
    accidentally-collidable identifiers.
    """
    if not dataset_version_id:
        raise ValueError("dataset_version_id must not be empty")
    if not row_key:
        raise ValueError("row_key must not be empty")
    if not content_hash.startswith("sha256:") or len(content_hash) != len("sha256:") + 64:
        raise ValueError("content_hash must use the canonical sha256:<hex> shape")

    canonical = "\n".join(("v1", dataset_version_id, row_key, content_hash))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"obj_{digest[:24]}"


__all__ = [
    "compute_content_sha256",
    "compute_record_sha256",
    "derive_object_id",
]
