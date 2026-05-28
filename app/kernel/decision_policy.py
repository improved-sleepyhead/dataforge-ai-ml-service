"""DecisionPolicy v0 and stable reason-code registry.

TASK-031 establishes the deterministic policy vocabulary used by later
Decision Core tasks. It intentionally does not build DecisionReport yet; it
loads versioned policy, exposes a config hash, evaluates MVP hard gates from
normalized policy inputs, and maps EvidenceBundle signals to reason codes.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain import EvidenceBundle, MethodCandidateStatus, PolicyStatus, SignalStatus
from app.domain.common import NonEmptyStr, Score, Sha256Digest

DECISION_POLICY_VERSION = "decision_policy_v0"
REASON_CODE_REGISTRY_VERSION = "reason_code_registry_v0"
OBJECT_VALUE_POLICY_VERSION = "object_value_policy_v0"


class ReasonCodeSeverity(StrEnum):
    """Reason-code severity levels."""

    INFO = "info"
    WARNING = "warning"
    REVIEW = "review"
    BLOCKER = "blocker"


class ReasonCodeDefinition(BaseModel):
    """Stable reason code metadata for reports, actions, and review queues."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: NonEmptyStr
    category: NonEmptyStr
    severity: ReasonCodeSeverity
    description: NonEmptyStr


class ReasonCodeRegistry(BaseModel):
    """Versioned registry of stable reason codes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: NonEmptyStr
    codes: dict[NonEmptyStr, ReasonCodeDefinition]

    def require(self, code: str) -> ReasonCodeDefinition:
        """Return a code definition or raise a clear policy error."""
        try:
            return self.codes[code]
        except KeyError as exc:
            raise ValueError(f"unknown reason code: {code}") from exc


class HardGateId(StrEnum):
    """MVP hard gates required by the PRD."""

    PII_UNMASKED = "pii_unmasked"
    TARGET_COLUMN_MISSING = "target_column_missing"
    SPLIT_LEAKAGE = "split_leakage"
    EXTERNAL_API_RAW_PII = "external_api_raw_pii"
    TARGET_LEAKAGE_CANDIDATE = "target_leakage_candidate"
    BUSINESS_RULE_FAILURE = "business_rule_failure"
    PLUGIN_DISABLED = "plugin_disabled"
    METHOD_DISABLED_BY_POLICY = "method_disabled_by_policy"
    METHOD_DISABLED_BY_READINESS = "method_disabled_by_readiness"


class RecommendedActionType(StrEnum):
    """Recommendation actions emitted by policy rules."""

    REPORT_AND_REVIEW = "REPORT_AND_REVIEW"
    SEND_TO_LABEL_REVIEW = "SEND_TO_LABEL_REVIEW"
    SEND_TO_PRIVACY_REVIEW = "SEND_TO_PRIVACY_REVIEW"
    SEND_TO_DUPLICATE_REVIEW = "SEND_TO_DUPLICATE_REVIEW"
    RECOMMEND_IMPUTATION_REVIEW = "RECOMMEND_IMPUTATION_REVIEW"
    RECOMMEND_RARE_CLASS_AUGMENTATION = "RECOMMEND_RARE_CLASS_AUGMENTATION"


class HardGateRule(BaseModel):
    """Policy rule that blocks unsafe operations before scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: HardGateId
    reason_code: NonEmptyStr
    action: NonEmptyStr
    enabled: bool = True


class RecommendationRule(BaseModel):
    """Policy rule that maps normalized evidence to suggested action classes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: NonEmptyStr
    reason_code: NonEmptyStr
    action: RecommendedActionType
    threshold: Score | None = None
    enabled: bool = True


class OutlierPolicy(BaseModel):
    """Outlier handling defaults.

    Capping is intentionally disabled by default in the MVP. Outliers are
    reported and routed to review until a project policy enables capping.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_action: RecommendedActionType = RecommendedActionType.REPORT_AND_REVIEW
    capping_enabled: bool = False
    disabled_reason_code: NonEmptyStr = "outlier_capping_disabled_by_policy"


class ObjectValueWeights(BaseModel):
    """Configurable ObjectValueScore weights from DATASETS.md."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_value: float = 0.25
    rarity: float = 0.20
    uncertainty: float = 0.20
    diversity: float = 0.15
    business_importance: float = 0.10
    duplicate_penalty: float = -0.15
    quality_penalty: float = -0.10
    privacy_risk_penalty: float = -0.25
    label_risk_penalty: float = -0.10


class ObjectValuePolicy(BaseModel):
    """Versioned object-value scoring policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: NonEmptyStr = OBJECT_VALUE_POLICY_VERSION
    weights: ObjectValueWeights = ObjectValueWeights()
    uncertainty_probability_notation: NonEmptyStr = "p_i,k = P(Y = k | x_i)"
    uncertainty_formulas: tuple[NonEmptyStr, ...] = (
        "U(x_i) = 1 - max_k p_i,k",
        "H(x_i) = - sum_k p_i,k * log(p_i,k)",
        "H_norm(x_i) = H(x_i) / log(K)",
    )


class MethodAvailability(BaseModel):
    """Normalized method availability supplied by platform/policy inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: NonEmptyStr
    status: MethodCandidateStatus
    policy_status: PolicyStatus
    reason_code: str | None = None


class PolicyInputEnvelope(BaseModel):
    """Normalized inputs DecisionPolicy can consume.

    The envelope lists the allowed policy inputs explicitly. It rejects
    plugin-private raw outputs so policy evaluation cannot bypass
    EvidenceBundle normalization.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized_dataset_profile: dict[str, Any] = Field(default_factory=dict)
    plugin_readiness: dict[str, str] = Field(default_factory=dict)
    method_availability: tuple[MethodAvailability, ...] = ()
    user_role_project_policy: dict[str, Any] = Field(default_factory=dict)
    risk_profile: dict[str, Any] = Field(default_factory=dict)
    split_leakage_report: dict[str, Any] = Field(default_factory=dict)
    synthetic_validation_report: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_raw_plugin_outputs(self) -> Self:
        forbidden = {"raw_plugin_outputs", "plugin_private_outputs", "raw_rows"}
        for field_name, value in self.model_dump(mode="python").items():
            if isinstance(value, dict) and forbidden & set(value):
                raise ValueError("policy inputs must not include raw plugin outputs")
            if field_name in forbidden:
                raise ValueError("policy inputs must not include raw plugin outputs")
        return self


class HardGateEvaluation(BaseModel):
    """One hard-gate evaluation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: HardGateId
    triggered: bool
    action: NonEmptyStr
    reason_code: NonEmptyStr


class ObjectValueScore(BaseModel):
    """Object-level score with full decomposition and blocker metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    value: Score
    raw_value: float
    policy_version: NonEmptyStr
    weights: dict[NonEmptyStr, float]
    components: dict[NonEmptyStr, float]
    weighted_components: dict[NonEmptyStr, float]
    hard_blocked: bool
    blocker_actions: tuple[NonEmptyStr, ...] = ()
    reason_codes: tuple[NonEmptyStr, ...] = ()
    uncertainty_probability_notation: NonEmptyStr
    uncertainty_formulas: tuple[NonEmptyStr, ...]


class ObjectDecisionEvaluation(BaseModel):
    """DecisionPolicy evaluation for one EvidenceBundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    hard_gates_evaluated_before_score: bool
    hard_gates: tuple[HardGateEvaluation, ...]
    object_value_score: ObjectValueScore
    hard_blocked: bool
    action: NonEmptyStr | None
    reason_codes: tuple[NonEmptyStr, ...]


class EvidenceRecommendation(BaseModel):
    """Policy recommendation reason emitted for one EvidenceBundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: NonEmptyStr
    action: RecommendedActionType
    reason_codes: tuple[NonEmptyStr, ...]


class DecisionPolicy(BaseModel):
    """Versioned MVP decision policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: NonEmptyStr = DECISION_POLICY_VERSION
    reason_code_registry_version: NonEmptyStr = REASON_CODE_REGISTRY_VERSION
    hard_gates: tuple[HardGateRule, ...]
    recommendation_rules: tuple[RecommendationRule, ...]
    outlier_policy: OutlierPolicy
    object_value_policy: ObjectValuePolicy
    policy_input_names: tuple[NonEmptyStr, ...]
    config_hash: Sha256Digest

    @model_validator(mode="after")
    def validate_policy_references(self) -> Self:
        registry = load_reason_code_registry()
        for gate in self.hard_gates:
            registry.require(gate.reason_code)
        for rule in self.recommendation_rules:
            registry.require(rule.reason_code)
        registry.require(self.outlier_policy.disabled_reason_code)
        return self


def load_reason_code_registry() -> ReasonCodeRegistry:
    """Load the built-in stable reason-code registry."""
    return ReasonCodeRegistry(
        version=REASON_CODE_REGISTRY_VERSION,
        codes={
            definition.code: definition
            for definition in (
                ReasonCodeDefinition(
                    code="pii_unmasked",
                    category="privacy",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="Unmasked PII is present in an object or artifact.",
                ),
                ReasonCodeDefinition(
                    code="target_column_missing",
                    category="schema",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="The target column has missing values.",
                ),
                ReasonCodeDefinition(
                    code="split_leakage",
                    category="leakage",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="Leakage detected between train/validation/test splits.",
                ),
                ReasonCodeDefinition(
                    code="external_api_raw_pii",
                    category="privacy",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="External API call would expose raw PII.",
                ),
                ReasonCodeDefinition(
                    code="target_leakage_candidate",
                    category="leakage",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="A feature appears to leak target information.",
                ),
                ReasonCodeDefinition(
                    code="business_rule_failure",
                    category="business_rules",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="A critical business rule failed.",
                ),
                ReasonCodeDefinition(
                    code="plugin_disabled",
                    category="plugin_readiness",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="A required plugin is disabled.",
                ),
                ReasonCodeDefinition(
                    code="method_disabled_by_policy",
                    category="method_policy",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="A candidate method is disabled by project policy.",
                ),
                ReasonCodeDefinition(
                    code="method_disabled_by_readiness",
                    category="method_policy",
                    severity=ReasonCodeSeverity.BLOCKER,
                    description="A candidate method is unavailable at current readiness.",
                ),
                ReasonCodeDefinition(
                    code="missing_numeric",
                    category="missingness",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Numeric missingness should be reviewed for imputation.",
                ),
                ReasonCodeDefinition(
                    code="exact_duplicate",
                    category="duplicates",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Object is part of an exact duplicate group.",
                ),
                ReasonCodeDefinition(
                    code="rare_class_underrepresented",
                    category="class_balance",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Object belongs to an underrepresented class.",
                ),
                ReasonCodeDefinition(
                    code="text_pii_detected",
                    category="privacy",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Text/OCR PII was detected and needs review/redaction.",
                ),
                ReasonCodeDefinition(
                    code="severe_outlier",
                    category="outliers",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Object has severe outlier evidence.",
                ),
                ReasonCodeDefinition(
                    code="outlier_capping_disabled_by_policy",
                    category="outliers",
                    severity=ReasonCodeSeverity.INFO,
                    description="Outlier capping is disabled unless policy enables it.",
                ),
                ReasonCodeDefinition(
                    code="ambiguous_object",
                    category="model_error",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Model is uncertain about this object.",
                ),
                ReasonCodeDefinition(
                    code="high_model_uncertainty",
                    category="model_error",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Prediction confidence is low or entropy is high.",
                ),
                ReasonCodeDefinition(
                    code="low_prediction_margin",
                    category="model_error",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Top prediction classes have a low probability margin.",
                ),
                ReasonCodeDefinition(
                    code="probable_label_error",
                    category="model_error",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Model evidence indicates a likely label issue.",
                ),
                ReasonCodeDefinition(
                    code="high_confidence_label_conflict",
                    category="model_error",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="High-confidence prediction conflicts with the label.",
                ),
                ReasonCodeDefinition(
                    code="neighbor_label_disagreement",
                    category="model_error",
                    severity=ReasonCodeSeverity.REVIEW,
                    description="Neighbor or cluster labels support a different label.",
                ),
            )
        },
    )


def load_decision_policy_v0() -> DecisionPolicy:
    """Load the built-in MVP DecisionPolicy v0."""
    payload: dict[str, Any] = {
        "policy_version": DECISION_POLICY_VERSION,
        "reason_code_registry_version": REASON_CODE_REGISTRY_VERSION,
        "hard_gates": [
            _gate(HardGateId.PII_UNMASKED, "BLOCK_PRIVACY_REVIEW"),
            _gate(HardGateId.TARGET_COLUMN_MISSING, "BLOCK_TRAINING"),
            _gate(HardGateId.SPLIT_LEAKAGE, "BLOCK_MODEL_EVALUATION"),
            _gate(HardGateId.EXTERNAL_API_RAW_PII, "BLOCK_EXTERNAL_API_CALL"),
            _gate(HardGateId.TARGET_LEAKAGE_CANDIDATE, "BLOCK_TRAINING"),
            _gate(HardGateId.BUSINESS_RULE_FAILURE, "BLOCK_RULES_REVIEW"),
            _gate(HardGateId.PLUGIN_DISABLED, "BLOCK_PLUGIN_EXECUTION"),
            _gate(HardGateId.METHOD_DISABLED_BY_POLICY, "BLOCK_METHOD"),
            _gate(HardGateId.METHOD_DISABLED_BY_READINESS, "BLOCK_METHOD"),
        ],
        "recommendation_rules": [
            _rule(
                "missing_numeric_review",
                "missing_numeric",
                RecommendedActionType.RECOMMEND_IMPUTATION_REVIEW,
                threshold=0.1,
            ),
            _rule(
                "exact_duplicate_review",
                "exact_duplicate",
                RecommendedActionType.SEND_TO_DUPLICATE_REVIEW,
                threshold=0.5,
            ),
            _rule(
                "rare_class_augmentation_review",
                "rare_class_underrepresented",
                RecommendedActionType.RECOMMEND_RARE_CLASS_AUGMENTATION,
                threshold=0.7,
            ),
            _rule(
                "text_pii_privacy_review",
                "text_pii_detected",
                RecommendedActionType.SEND_TO_PRIVACY_REVIEW,
                threshold=0.01,
            ),
            _rule(
                "severe_outlier_review",
                "severe_outlier",
                RecommendedActionType.REPORT_AND_REVIEW,
                threshold=0.75,
            ),
            _rule(
                "ambiguous_label_review",
                "ambiguous_object",
                RecommendedActionType.SEND_TO_LABEL_REVIEW,
                threshold=0.5,
            ),
            _rule(
                "probable_label_error_review",
                "probable_label_error",
                RecommendedActionType.SEND_TO_LABEL_REVIEW,
                threshold=0.5,
            ),
        ],
        "outlier_policy": OutlierPolicy().model_dump(mode="json"),
        "object_value_policy": ObjectValuePolicy().model_dump(mode="json"),
        "policy_input_names": [
            "normalized_dataset_profile",
            "plugin_readiness",
            "method_availability",
            "user_role_project_policy",
            "risk_profile",
            "split_leakage_report",
            "synthetic_validation_report",
        ],
    }
    payload["config_hash"] = _config_hash(payload)
    return DecisionPolicy.model_validate(payload)


def evaluate_hard_gates(
    *,
    policy: DecisionPolicy,
    inputs: PolicyInputEnvelope,
) -> tuple[HardGateEvaluation, ...]:
    """Evaluate enabled hard gates from normalized policy inputs."""
    results: list[HardGateEvaluation] = []
    for gate in policy.hard_gates:
        if not gate.enabled:
            continue
        results.append(
            HardGateEvaluation(
                gate_id=gate.gate_id,
                triggered=_gate_triggered(gate.gate_id, inputs),
                action=gate.action,
                reason_code=gate.reason_code,
            )
        )
    return tuple(results)


def recommend_from_evidence(
    *,
    policy: DecisionPolicy,
    evidence: EvidenceBundle,
) -> tuple[EvidenceRecommendation, ...]:
    """Map normalized EvidenceBundle signals to policy recommendation reasons."""
    recommendations: list[EvidenceRecommendation] = []
    for rule in policy.recommendation_rules:
        if not rule.enabled or not _rule_matches(rule, evidence):
            continue
        reason_codes = [rule.reason_code]
        if rule.reason_code == "ambiguous_object":
            reason_codes.extend(("high_model_uncertainty", "low_prediction_margin"))
        if rule.reason_code == "probable_label_error":
            reason_codes.append("high_confidence_label_conflict")
        recommendations.append(
            EvidenceRecommendation(
                object_id=evidence.object_id,
                action=rule.action,
                reason_codes=tuple(dict.fromkeys(reason_codes)),
            )
        )
    return tuple(recommendations)


def disabled_outlier_capping_reason(policy: DecisionPolicy) -> str | None:
    """Return disabled_by_policy reason for outlier capping when disabled."""
    if policy.outlier_policy.capping_enabled:
        return None
    return policy.outlier_policy.disabled_reason_code


def evaluate_object_decision(
    *,
    policy: DecisionPolicy,
    evidence: EvidenceBundle,
    inputs: PolicyInputEnvelope | None = None,
) -> ObjectDecisionEvaluation:
    """Evaluate hard gates first, then compute ObjectValueScore.

    The returned score is explanatory only when a hard blocker fired. A high
    score cannot override ``hard_blocked=True`` or the selected blocker action.
    """
    policy_inputs = inputs or PolicyInputEnvelope()
    hard_gates = _object_hard_gates(policy=policy, evidence=evidence, inputs=policy_inputs)
    triggered = tuple(gate for gate in hard_gates if gate.triggered)
    score = compute_object_value_score(
        policy=policy,
        evidence=evidence,
        hard_gates=hard_gates,
    )
    reason_codes = tuple(dict.fromkeys(gate.reason_code for gate in triggered))
    action = triggered[0].action if triggered else None
    return ObjectDecisionEvaluation(
        object_id=evidence.object_id,
        hard_gates_evaluated_before_score=True,
        hard_gates=hard_gates,
        object_value_score=score,
        hard_blocked=bool(triggered),
        action=action,
        reason_codes=reason_codes,
    )


def compute_object_value_score(
    *,
    policy: DecisionPolicy,
    evidence: EvidenceBundle,
    hard_gates: tuple[HardGateEvaluation, ...] = (),
) -> ObjectValueScore:
    """Compute ObjectValueScore from EvidenceBundle using policy weights."""
    components = _object_value_components(evidence)
    weights = policy.object_value_policy.weights.model_dump(mode="python")
    weighted = {
        component: float(value) * float(weights[component])
        for component, value in components.items()
    }
    raw = sum(weighted.values())
    clamped = min(1.0, max(0.0, raw))
    triggered = tuple(gate for gate in hard_gates if gate.triggered)
    reason_codes = tuple(
        dict.fromkeys(
            [
                *(
                    code
                    for code, value in _component_reason_codes(components).items()
                    if value
                ),
                *(gate.reason_code for gate in triggered),
            ]
        )
    )
    return ObjectValueScore(
        object_id=evidence.object_id,
        value=clamped,
        raw_value=raw,
        policy_version=policy.object_value_policy.policy_version,
        weights={str(key): float(value) for key, value in weights.items()},
        components=components,
        weighted_components=weighted,
        hard_blocked=bool(triggered),
        blocker_actions=tuple(dict.fromkeys(gate.action for gate in triggered)),
        reason_codes=reason_codes,
        uncertainty_probability_notation=(
            policy.object_value_policy.uncertainty_probability_notation
        ),
        uncertainty_formulas=policy.object_value_policy.uncertainty_formulas,
    )


def _gate(gate_id: HardGateId, action: str) -> dict[str, Any]:
    return {
        "gate_id": gate_id.value,
        "reason_code": gate_id.value,
        "action": action,
        "enabled": True,
    }


def _rule(
    rule_id: str,
    reason_code: str,
    action: RecommendedActionType,
    *,
    threshold: float | None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "reason_code": reason_code,
        "action": action.value,
        "threshold": threshold,
        "enabled": True,
    }


def _gate_triggered(gate_id: HardGateId, inputs: PolicyInputEnvelope) -> bool:
    profile = inputs.normalized_dataset_profile
    risk = inputs.risk_profile
    split = inputs.split_leakage_report
    if gate_id is HardGateId.PII_UNMASKED:
        return bool(risk.get("pii_unmasked") or profile.get("pii_unmasked"))
    if gate_id is HardGateId.TARGET_COLUMN_MISSING:
        return bool(profile.get("target_column_missing"))
    if gate_id is HardGateId.SPLIT_LEAKAGE:
        return bool(split.get("split_leakage"))
    if gate_id is HardGateId.EXTERNAL_API_RAW_PII:
        return bool(risk.get("external_api_raw_pii"))
    if gate_id is HardGateId.TARGET_LEAKAGE_CANDIDATE:
        return bool(profile.get("target_leakage_candidate"))
    if gate_id is HardGateId.BUSINESS_RULE_FAILURE:
        return bool(profile.get("business_rule_failure"))
    if gate_id is HardGateId.PLUGIN_DISABLED:
        return any(status == "disabled" for status in inputs.plugin_readiness.values())
    if gate_id is HardGateId.METHOD_DISABLED_BY_POLICY:
        return any(
            method.policy_status is PolicyStatus.DISABLED_BY_POLICY
            or method.status is MethodCandidateStatus.DISABLED_BY_POLICY
            for method in inputs.method_availability
        )
    if gate_id is HardGateId.METHOD_DISABLED_BY_READINESS:
        return any(
            method.policy_status is PolicyStatus.DISABLED_BY_READINESS
            or method.status is MethodCandidateStatus.DISABLED_BY_READINESS
            for method in inputs.method_availability
        )
    return False


def _object_hard_gates(
    *,
    policy: DecisionPolicy,
    evidence: EvidenceBundle,
    inputs: PolicyInputEnvelope,
) -> tuple[HardGateEvaluation, ...]:
    input_gates = {gate.gate_id: gate for gate in evaluate_hard_gates(policy=policy, inputs=inputs)}
    results: list[HardGateEvaluation] = []
    for rule in policy.hard_gates:
        input_gate = input_gates.get(rule.gate_id)
        triggered = input_gate.triggered if input_gate is not None else False
        if rule.gate_id is HardGateId.PII_UNMASKED:
            triggered = triggered or _signal_at_least(evidence.signals.privacy_risk, 0.80)
        results.append(
            HardGateEvaluation(
                gate_id=rule.gate_id,
                triggered=triggered,
                action=rule.action,
                reason_code=rule.reason_code,
            )
        )
    return tuple(results)


def _object_value_components(evidence: EvidenceBundle) -> dict[str, float]:
    signals = evidence.signals
    rare = _signal_value(signals.rare_segment_score)
    uncertainty = _uncertainty_component(evidence)
    label_risk = _signal_value(signals.label_issue_score)
    probable_label_error = _signal_value(signals.probable_label_error_score)
    ambiguous = _signal_value(signals.ambiguous_object_score)
    learning_value = max(rare, uncertainty, ambiguous, probable_label_error, label_risk)
    quality_penalty = 1.0 - _signal_value(signals.technical_quality, default=1.0)
    return {
        "learning_value": learning_value,
        "rarity": rare,
        "uncertainty": uncertainty,
        "diversity": 0.0,
        "business_importance": _signal_value(signals.business_importance),
        "duplicate_penalty": _signal_value(signals.duplicate_score),
        "quality_penalty": quality_penalty,
        "privacy_risk_penalty": _signal_value(signals.privacy_risk),
        "label_risk_penalty": max(label_risk, probable_label_error),
    }


def _uncertainty_component(evidence: EvidenceBundle) -> float:
    signals = evidence.signals
    if signals.model_uncertainty.status is SignalStatus.AVAILABLE:
        return _signal_value(signals.model_uncertainty)
    if signals.prediction_entropy.status is SignalStatus.AVAILABLE:
        return _signal_value(signals.prediction_entropy)
    if signals.ambiguous_object_score.status is SignalStatus.AVAILABLE:
        return _signal_value(signals.ambiguous_object_score)
    if signals.prediction_confidence.status is SignalStatus.AVAILABLE:
        return 1.0 - _signal_value(signals.prediction_confidence)
    return 0.0


def _signal_value(signal: Any, *, default: float = 0.0) -> float:
    if signal.status is not SignalStatus.AVAILABLE or signal.value is None:
        return default
    return float(signal.value)


def _component_reason_codes(components: dict[str, float]) -> dict[str, bool]:
    return {
        "exact_duplicate": components["duplicate_penalty"] >= 0.5,
        "rare_class_underrepresented": components["rarity"] >= 0.7,
        "text_pii_detected": components["privacy_risk_penalty"] > 0.0,
        "severe_outlier": components["quality_penalty"] >= 0.75,
        "ambiguous_object": components["uncertainty"] >= 0.5,
        "probable_label_error": components["label_risk_penalty"] >= 0.5,
    }


def _rule_matches(rule: RecommendationRule, evidence: EvidenceBundle) -> bool:
    threshold = rule.threshold or 0.0
    signals = evidence.signals
    if rule.reason_code == "missing_numeric":
        return False
    if rule.reason_code == "exact_duplicate":
        return _signal_at_least(signals.duplicate_score, threshold)
    if rule.reason_code == "rare_class_underrepresented":
        return _signal_at_least(signals.rare_segment_score, threshold)
    if rule.reason_code == "text_pii_detected":
        return (
            evidence.modality.value in {"text", "document_ocr"}
            and _signal_at_least(signals.privacy_risk, threshold)
        )
    if rule.reason_code == "severe_outlier":
        return (
            signals.technical_quality.status is SignalStatus.AVAILABLE
            and signals.technical_quality.value is not None
            and signals.technical_quality.value <= (1.0 - threshold)
        )
    if rule.reason_code == "ambiguous_object":
        return _signal_at_least(signals.ambiguous_object_score, threshold)
    if rule.reason_code == "probable_label_error":
        return _signal_at_least(signals.probable_label_error_score, threshold)
    return False


def _signal_at_least(signal: Any, threshold: float) -> bool:
    return (
        signal.status is SignalStatus.AVAILABLE
        and signal.value is not None
        and signal.value >= threshold
    )


def _config_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


__all__ = [
    "DECISION_POLICY_VERSION",
    "OBJECT_VALUE_POLICY_VERSION",
    "REASON_CODE_REGISTRY_VERSION",
    "DecisionPolicy",
    "EvidenceRecommendation",
    "HardGateEvaluation",
    "HardGateId",
    "HardGateRule",
    "MethodAvailability",
    "ObjectDecisionEvaluation",
    "ObjectValuePolicy",
    "ObjectValueScore",
    "ObjectValueWeights",
    "OutlierPolicy",
    "PolicyInputEnvelope",
    "ReasonCodeDefinition",
    "ReasonCodeRegistry",
    "ReasonCodeSeverity",
    "RecommendationRule",
    "RecommendedActionType",
    "compute_object_value_score",
    "disabled_outlier_capping_reason",
    "evaluate_object_decision",
    "evaluate_hard_gates",
    "load_decision_policy_v0",
    "load_reason_code_registry",
    "recommend_from_evidence",
]
