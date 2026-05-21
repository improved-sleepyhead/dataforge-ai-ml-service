"""Lightweight business-rule validation engine for the tabular profiler.

TASK-026 scope:

* MVP rule DSL with simple row/cross-column conditions;
* deterministic rule evaluation per row, no raw payload logging;
* aggregate per-rule pass rate + bounded sample of violating object_ids;
* critical violations produce a ``BLOCK_RULES_REVIEW`` blocker
  candidate surfaced via :class:`BusinessRulesReport`.

The engine is intentionally stdlib-only and does not execute any user
Python code. Conditions are described declaratively with a small
operator set so the rule config can be hashed and stored in the report
without re-parsing logic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain import (
    BusinessRuleSeverity,
    BusinessRulesReport,
    BusinessRuleSummary,
    BusinessRuleViolation,
)
from app.domain.common import NonEmptyStr, Sha256Digest

_DEFAULT_RULES_VERSION = "tabular_business_rules.v1"
_DEFAULT_SAMPLE_LIMIT = 50

_NumericOp = Literal["gte", "gt", "lte", "lt"]
_EqualityOp = Literal["eq", "ne"]
_SetOp = Literal["in", "not_in"]
_FieldOp = Literal["gte", "gt", "lte", "lt", "eq", "ne", "in", "not_in"]


class RuleFieldCheck(BaseModel):
    """Atomic predicate over one row field.

    The predicate is evaluated on the raw string value (CSV semantics)
    or on the parsed numeric form when the operator is numeric. Missing
    values (``""``) never satisfy a predicate; they are surfaced as
    violations only if the rule's ``allow_missing`` flag is False.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: NonEmptyStr
    op: _FieldOp
    value: float | int | str | bool | tuple[str, ...]

    @field_validator("value")
    @classmethod
    def _normalize_value(
        cls, value: Any
    ) -> float | int | str | bool | tuple[str, ...]:
        if isinstance(value, list):
            return tuple(str(v) for v in value)
        if isinstance(value, (float, int, str, bool, tuple)):
            return value
        raise TypeError(f"unsupported rule value type: {type(value).__name__}")


class BusinessRule(BaseModel):
    """Declarative business rule.

    A rule is satisfied when ALL ``checks`` pass on the row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: NonEmptyStr
    description: NonEmptyStr | None = None
    severity: BusinessRuleSeverity = BusinessRuleSeverity.WARNING
    columns: tuple[NonEmptyStr, ...] = ()
    checks: tuple[RuleFieldCheck, ...] = Field(min_length=1)
    allow_missing: bool = False
    message: NonEmptyStr | None = None


def compute_rules_config_hash(rules: tuple[BusinessRule, ...]) -> Sha256Digest:
    """Return a deterministic ``sha256:<hex>`` digest for a rule list.

    The digest is computed on the canonical JSON representation of the
    rules so a config change always changes the hash. The hash is stored
    in :class:`BusinessRulesReport.rules_config_hash` and included in
    the per-row evidence so reruns are reproducible.
    """
    payload = [rule.model_dump(mode="json") for rule in rules]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass
class _RuleCounters:
    evaluated: int = 0
    violations: int = 0
    sample_object_ids: list[str] = field(default_factory=list)


class BusinessRuleEvaluator:
    """Stateful evaluator that accumulates per-rule counters across rows.

    The evaluator is plugin-internal: tabular profiler instantiates it
    once per build, calls :meth:`observe_row` for every row, and then
    calls :meth:`build_report` to produce the contract block.
    """

    def __init__(
        self,
        rules: tuple[BusinessRule, ...],
        *,
        rules_version: str = _DEFAULT_RULES_VERSION,
        rules_config_hash: Sha256Digest | None = None,
        sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
    ) -> None:
        self._rules = rules
        self._rules_version = rules_version
        self._rules_config_hash = (
            rules_config_hash
            if rules_config_hash is not None
            else compute_rules_config_hash(rules)
        )
        self._sample_limit = sample_limit
        self._counters: dict[str, _RuleCounters] = {
            rule.rule_id: _RuleCounters() for rule in rules
        }
        self._sample_violations: list[BusinessRuleViolation] = []

    @property
    def rules_version(self) -> str:
        return self._rules_version

    @property
    def rules_config_hash(self) -> Sha256Digest:
        return self._rules_config_hash

    def observe_row(self, row: Mapping[str, str], object_id: str | None) -> None:
        for rule in self._rules:
            counters = self._counters[rule.rule_id]
            evaluated, passed, failed_check = _evaluate_rule(rule, row)
            if not evaluated:
                continue
            counters.evaluated += 1
            if passed:
                continue
            counters.violations += 1
            if (
                len(counters.sample_object_ids) < self._sample_limit
                and object_id
            ):
                counters.sample_object_ids.append(object_id)
            if len(self._sample_violations) < self._sample_limit:
                self._sample_violations.append(
                    BusinessRuleViolation(
                        rule_id=rule.rule_id,
                        object_id=object_id or None,
                        column=failed_check.field if failed_check else None,
                        message=rule.message,
                    )
                )

    def build_report(self) -> BusinessRulesReport:
        summaries: list[BusinessRuleSummary] = []
        blockers: list[str] = []
        for rule in self._rules:
            counters = self._counters[rule.rule_id]
            evaluated = counters.evaluated
            violations = counters.violations
            pass_rate = (
                (evaluated - violations) / evaluated if evaluated > 0 else 1.0
            )
            summaries.append(
                BusinessRuleSummary(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    description=rule.description,
                    columns=rule.columns,
                    violation_count=violations,
                    evaluated_count=evaluated,
                    pass_rate=pass_rate,
                )
            )
            if (
                rule.severity is BusinessRuleSeverity.CRITICAL
                and violations > 0
            ):
                blockers.append(rule.rule_id)
        return BusinessRulesReport(
            rules_version=self._rules_version,
            rules_config_hash=self._rules_config_hash,
            rules=tuple(summaries),
            sample_violations=tuple(self._sample_violations),
            blocker_candidate_rule_ids=tuple(blockers),
        )


def _evaluate_rule(
    rule: BusinessRule, row: Mapping[str, str]
) -> tuple[bool, bool, RuleFieldCheck | None]:
    """Return ``(evaluated, passed, first_failed_check)``."""
    for check in rule.checks:
        raw = row.get(check.field, "")
        if raw == "":
            if rule.allow_missing:
                continue
            return True, False, check
        if not _check_value(raw, check):
            return True, False, check
    return True, True, None


def _check_value(raw: str, check: RuleFieldCheck) -> bool:
    op = check.op
    if op in {"gte", "gt", "lte", "lt"}:
        try:
            actual = float(raw)
        except (TypeError, ValueError):
            return False
        expected_value = check.value
        if isinstance(expected_value, tuple) or isinstance(expected_value, str):
            try:
                expected = float(str(expected_value))
            except (TypeError, ValueError):
                return False
        else:
            expected = float(expected_value)
        if op == "gte":
            return actual >= expected
        if op == "gt":
            return actual > expected
        if op == "lte":
            return actual <= expected
        return actual < expected  # op == "lt"
    if op in {"eq", "ne"}:
        actual_str = raw
        expected_str = str(check.value)
        if op == "eq":
            return actual_str == expected_str
        return actual_str != expected_str
    if op in {"in", "not_in"}:
        if not isinstance(check.value, tuple):
            raise ValueError(
                f"rule operator '{op}' requires a list value, got {type(check.value).__name__}"
            )
        if op == "in":
            return raw in check.value
        return raw not in check.value
    raise ValueError(f"unsupported rule operator: {op}")


def parse_business_rules(payload: Iterable[Mapping[str, Any]]) -> tuple[BusinessRule, ...]:
    """Parse a list of rule dicts (e.g. from JSON config) into ``BusinessRule``.

    Helper for tests/CLI; production code can build the rules directly
    from typed configuration.
    """
    return tuple(BusinessRule.model_validate(item) for item in payload)


__all__ = [
    "BusinessRule",
    "BusinessRuleEvaluator",
    "RuleFieldCheck",
    "compute_rules_config_hash",
    "parse_business_rules",
]
