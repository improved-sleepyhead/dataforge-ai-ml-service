"""Deterministic idempotency-key derivation for analyze/apply jobs (TASK-060).

A *job idempotency key* is a stable sha256 digest computed from the
inputs that fully determine a compute run's outputs. Two compute runs
with identical idempotency keys must produce identical artifact
hashes, and downstream consumers (the platform, audit, the tests) can
use the key to detect whether a re-run is logically the same job.

Per PRD §20 / DATASETS.md and the TASK-060 acceptance criteria, the
key derivation must include:

* the workflow type (``ANALYZE_ONLY`` vs ``APPLY_SELECTED_ACTIONS``);
* the input artifact hashes (sorted, deduplicated);
* the prediction manifest hashes when present (separate channel);
* the config_hash that captures the policy snapshot;
* the algorithm/plugin version footprint (so a new plugin release
  invalidates the cache even when input bytes are identical).

The keys are *not* security tokens. They are stable identifiers that
the compute plane and the tests can use to assert reused vs new runs.
The platform control plane is the source of truth for whether a job
should be replayed; this module just gives it a deterministic shape
to compare against.

Everything in this module is a pure function: it does not perform I/O
and does not log anything. Callers must build the input record and
ask for the digest. The same input always returns the same digest.

Validation gates — and any other failure signal — are not folded into
the key. This is intentional: failed-gate signals must always reach
the response surface, even when the underlying artifact bytes are
cache hits. Otherwise a previously failed run could be silently
"reused" without re-checking the gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domain import ArtifactRef, WorkflowType
from app.domain.common import NonEmptyStr, Sha256Digest


class JobIdempotencyKeyError(ValueError):
    """Raised when the inputs for an idempotency key are not safe to hash."""

    def __init__(self, *, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class PluginVersionFootprint(BaseModel):
    """One plugin/algorithm version pin folded into the idempotency key.

    The footprint must be stable across runs that intend to share a
    cache. ``plugin_id`` and ``algorithm_name`` together identify the
    capability; ``plugin_version`` and ``algorithm_version`` make the
    capability snapshot reproducible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin_id: NonEmptyStr
    plugin_version: NonEmptyStr
    algorithm_name: NonEmptyStr
    algorithm_version: NonEmptyStr


class AnalyzeIdempotencyInputs(BaseModel):
    """Inputs that fully determine an ANALYZE_ONLY run's outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    dataset_version_id: NonEmptyStr
    input_artifact_hashes: tuple[Sha256Digest, ...]
    prediction_artifact_hashes: tuple[Sha256Digest, ...] = ()
    config_hash: Sha256Digest
    contract_pack_version: NonEmptyStr
    plugin_versions: tuple[PluginVersionFootprint, ...] = ()


class ApplyIdempotencyInputs(BaseModel):
    """Inputs that fully determine an APPLY_SELECTED_ACTIONS run's outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization_id: NonEmptyStr
    project_id: NonEmptyStr
    dataset_id: NonEmptyStr
    source_dataset_version_id: NonEmptyStr
    proposed_version_name: NonEmptyStr
    action_plan_hash: Sha256Digest
    input_artifact_hashes: tuple[Sha256Digest, ...]
    prediction_artifact_hashes: tuple[Sha256Digest, ...] = ()
    config_hash: Sha256Digest
    contract_pack_version: NonEmptyStr
    policy_versions: Mapping[str, str]
    step_idempotency_keys: tuple[Sha256Digest, ...]
    plugin_versions: tuple[PluginVersionFootprint, ...] = ()


def compute_analyze_idempotency_key(inputs: AnalyzeIdempotencyInputs) -> Sha256Digest:
    """Return a stable sha256 idempotency key for an ANALYZE_ONLY run."""
    payload = {
        "workflow": WorkflowType.ANALYZE_ONLY.value,
        "scope": {
            "organization_id": inputs.organization_id,
            "project_id": inputs.project_id,
            "dataset_id": inputs.dataset_id,
            "dataset_version_id": inputs.dataset_version_id,
        },
        "input_artifact_hashes": _ordered_unique(inputs.input_artifact_hashes),
        "prediction_artifact_hashes": _ordered_unique(
            inputs.prediction_artifact_hashes
        ),
        "config_hash": inputs.config_hash,
        "contract_pack_version": inputs.contract_pack_version,
        "plugin_versions": _normalize_plugin_versions(inputs.plugin_versions),
    }
    return _digest(payload)


def compute_apply_idempotency_key(inputs: ApplyIdempotencyInputs) -> Sha256Digest:
    """Return a stable sha256 idempotency key for an APPLY_SELECTED_ACTIONS run."""
    payload = {
        "workflow": WorkflowType.APPLY_SELECTED_ACTIONS.value,
        "scope": {
            "organization_id": inputs.organization_id,
            "project_id": inputs.project_id,
            "dataset_id": inputs.dataset_id,
            "source_dataset_version_id": inputs.source_dataset_version_id,
            "proposed_version_name": inputs.proposed_version_name,
        },
        "action_plan_hash": inputs.action_plan_hash,
        "input_artifact_hashes": _ordered_unique(inputs.input_artifact_hashes),
        "prediction_artifact_hashes": _ordered_unique(
            inputs.prediction_artifact_hashes
        ),
        "config_hash": inputs.config_hash,
        "contract_pack_version": inputs.contract_pack_version,
        "policy_versions": _ordered_dict(inputs.policy_versions),
        "step_idempotency_keys": list(inputs.step_idempotency_keys),
        "plugin_versions": _normalize_plugin_versions(inputs.plugin_versions),
    }
    return _digest(payload)


def collect_artifact_hashes(refs: Iterable[ArtifactRef]) -> tuple[Sha256Digest, ...]:
    """Return artifact hashes for an iterable of refs.

    The result preserves a deterministic order: hashes are deduplicated
    and emitted in their first-seen order. Empty inputs return an
    empty tuple.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for ref in refs:
        digest = ref.hash
        if digest in seen_set:
            continue
        seen_set.add(digest)
        seen.append(digest)
    return tuple(seen)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _digest(payload: object) -> Sha256Digest:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ordered_dict(mapping: Mapping[str, str]) -> dict[str, str]:
    return {key: mapping[key] for key in sorted(mapping)}


def _normalize_plugin_versions(
    versions: Iterable[PluginVersionFootprint],
) -> list[dict[str, Any]]:
    serialized = [
        {
            "plugin_id": footprint.plugin_id,
            "plugin_version": footprint.plugin_version,
            "algorithm_name": footprint.algorithm_name,
            "algorithm_version": footprint.algorithm_version,
        }
        for footprint in versions
    ]
    serialized.sort(
        key=lambda entry: (
            entry["plugin_id"],
            entry["algorithm_name"],
            entry["plugin_version"],
            entry["algorithm_version"],
        )
    )
    return serialized


__all__ = [
    "AnalyzeIdempotencyInputs",
    "ApplyIdempotencyInputs",
    "JobIdempotencyKeyError",
    "PluginVersionFootprint",
    "collect_artifact_hashes",
    "compute_analyze_idempotency_key",
    "compute_apply_idempotency_key",
]
