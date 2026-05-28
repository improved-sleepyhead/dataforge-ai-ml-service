"""DecisionReport assembly from normalized Decision Core evidence."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from app.domain import (
    BlockedDecisionAction,
    CriticalBlocker,
    DataModality,
    DatasetDecision,
    DatasetReadiness,
    DecisionAction,
    DecisionReport,
    EvidenceBundle,
    EvidenceRef,
    ObjectLevelDecision,
    PolicyVersions,
    ReadinessAssessment,
    RecommendedAction,
    WorkflowType,
)
from app.domain.common import NonEmptyStr, Sha256Digest
from app.kernel.decision_policy import (
    DecisionPolicy,
    EvidenceRecommendation,
    HardGateEvaluation,
    ObjectDecisionEvaluation,
    PolicyInputEnvelope,
    RecommendedActionType,
    evaluate_hard_gates,
    evaluate_object_decision,
    load_decision_policy_v0,
    load_reason_code_registry,
    recommend_from_evidence,
)

DECISION_REPORT_SCHEMA_VERSION = "decision_report.v1"
PRIVACY_POLICY_VERSION = "privacy_v0"
METHOD_POLICY_VERSION = "method_selection_v0"


class BuildDecisionReportRequest(BaseModel):
    """Inputs required to assemble a DecisionReport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyStr
    version_id: NonEmptyStr
    created_by_job_id: NonEmptyStr
    config_hash: Sha256Digest
    decision_report_id: str | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class _ObjectDecisionContext:
    evidence: EvidenceBundle
    evaluation: ObjectDecisionEvaluation
    recommendations: tuple[EvidenceRecommendation, ...]
    decision: ObjectLevelDecision


@dataclass(frozen=True)
class _RecommendationGroup:
    action: DecisionAction
    modality: DataModality
    primary_reason: str
    object_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


def build_decision_report(
    *,
    evidence_bundles: Iterable[EvidenceBundle],
    request: BuildDecisionReportRequest,
    policy: DecisionPolicy | None = None,
    inputs: PolicyInputEnvelope | None = None,
) -> DecisionReport:
    """Build a contract-valid DecisionReport from EvidenceBundle rows."""
    active_policy = policy or load_decision_policy_v0()
    policy_inputs = inputs or PolicyInputEnvelope()
    evidence_tuple = tuple(evidence_bundles)
    contexts = tuple(
        _object_context(
            evidence=evidence,
            policy=active_policy,
            inputs=policy_inputs,
        )
        for evidence in evidence_tuple
    )
    dataset_gates = evaluate_hard_gates(policy=active_policy, inputs=policy_inputs)
    blockers = _critical_blockers(
        dataset_gates=dataset_gates,
        object_contexts=contexts,
    )
    recommendation_groups = _recommendation_groups(contexts)
    recommended_actions = _recommended_actions(recommendation_groups)
    readiness = _readiness(
        blockers=blockers,
        recommended_actions=recommended_actions,
    )
    safe_actions_available = any(
        not action.requires_approval for action in recommended_actions
    )

    return DecisionReport(
        decision_report_id=request.decision_report_id
        or _decision_report_id(request=request, evidence_bundles=evidence_tuple),
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        decision_schema_version=DECISION_REPORT_SCHEMA_VERSION,
        dataset_decision=_dataset_decision(blockers, recommended_actions),
        readiness=readiness,
        critical_blockers=blockers,
        safe_actions_available=safe_actions_available,
        recommended_next_job=_recommended_next_job(
            blockers=blockers,
            recommended_actions=recommended_actions,
        ),
        object_decisions=tuple(context.decision for context in contexts),
        recommended_actions=recommended_actions,
        policy_versions=PolicyVersions(
            decision_policy=active_policy.policy_version,
            score_policy=active_policy.object_value_policy.policy_version,
            privacy_policy=PRIVACY_POLICY_VERSION,
            method_policy=METHOD_POLICY_VERSION,
        ),
        generated_at=request.generated_at or datetime.now(UTC),
    )


def _object_context(
    *,
    evidence: EvidenceBundle,
    policy: DecisionPolicy,
    inputs: PolicyInputEnvelope,
) -> _ObjectDecisionContext:
    evaluation = evaluate_object_decision(
        policy=policy,
        evidence=evidence,
        inputs=inputs,
    )
    recommendations = recommend_from_evidence(policy=policy, evidence=evidence)
    reasons = _object_reason_codes(
        evaluation=evaluation,
        recommendations=recommendations,
    )
    action = _object_action(evaluation=evaluation, recommendations=recommendations)
    decision = ObjectLevelDecision(
        object_id=evidence.object_id,
        modality=evidence.modality,
        action=action,
        reasons=reasons,
        blocked_actions=_blocked_actions(evaluation),
        object_value_score=evaluation.object_value_score.value,
    )
    return _ObjectDecisionContext(
        evidence=evidence,
        evaluation=evaluation,
        recommendations=recommendations,
        decision=decision,
    )


def _object_reason_codes(
    *,
    evaluation: ObjectDecisionEvaluation,
    recommendations: tuple[EvidenceRecommendation, ...],
) -> tuple[str, ...]:
    codes = [
        *(code for recommendation in recommendations for code in recommendation.reason_codes),
        *evaluation.reason_codes,
        *evaluation.object_value_score.reason_codes,
    ]
    if not codes:
        codes.append("export_ready")
    return tuple(dict.fromkeys(codes))


def _object_action(
    *,
    evaluation: ObjectDecisionEvaluation,
    recommendations: tuple[EvidenceRecommendation, ...],
) -> DecisionAction:
    if evaluation.hard_blocked:
        return DecisionAction.BLOCK_EXPORT
    reason_codes = {
        code for recommendation in recommendations for code in recommendation.reason_codes
    }
    action_types = {recommendation.action for recommendation in recommendations}
    if "probable_label_error" in reason_codes:
        return DecisionAction.SEND_TO_LABEL_REVIEW
    if "ambiguous_object" in reason_codes:
        return DecisionAction.SEND_TO_LABEL_REVIEW
    if RecommendedActionType.SEND_TO_PRIVACY_REVIEW in action_types:
        return DecisionAction.SEND_TO_PRIVACY_REVIEW
    if RecommendedActionType.SEND_TO_DUPLICATE_REVIEW in action_types:
        return DecisionAction.REMOVE_DUPLICATE
    if RecommendedActionType.RECOMMEND_IMPUTATION_REVIEW in action_types:
        return DecisionAction.IMPUTE_MISSING_VALUES
    if RecommendedActionType.RECOMMEND_RARE_CLASS_AUGMENTATION in action_types:
        return DecisionAction.AUGMENT_RARE_CLASS
    return DecisionAction.KEEP


def _blocked_actions(
    evaluation: ObjectDecisionEvaluation,
) -> tuple[BlockedDecisionAction, ...]:
    if not evaluation.hard_blocked:
        return ()
    reasons = evaluation.reason_codes or evaluation.object_value_score.reason_codes
    return (
        BlockedDecisionAction(
            action=DecisionAction.EXPORT_READY,
            reason="Hard policy gates must be resolved before export.",
            reason_codes=reasons,
        ),
    )


def _critical_blockers(
    *,
    dataset_gates: tuple[HardGateEvaluation, ...],
    object_contexts: tuple[_ObjectDecisionContext, ...],
) -> tuple[CriticalBlocker, ...]:
    registry = load_reason_code_registry()
    blockers: dict[str, CriticalBlocker] = {}
    for gate in dataset_gates:
        if gate.triggered:
            blockers.setdefault(
                gate.reason_code,
                _critical_blocker(gate=gate, evidence_refs=()),
            )
    for context in object_contexts:
        for gate in context.evaluation.hard_gates:
            if not gate.triggered:
                continue
            definition = registry.require(gate.reason_code)
            existing = blockers.get(gate.reason_code)
            refs = _limited_refs(context.evidence.evidence_refs)
            if existing is None:
                blockers[gate.reason_code] = CriticalBlocker(
                    code=gate.reason_code.upper(),
                    severity="critical",
                    message=definition.description,
                    evidence_refs=refs,
                )
                continue
            blockers[gate.reason_code] = existing.model_copy(
                update={
                    "evidence_refs": tuple(
                        dict.fromkeys([*existing.evidence_refs, *refs])
                    )
                }
            )
    return tuple(blockers.values())


def _critical_blocker(
    *,
    gate: HardGateEvaluation,
    evidence_refs: tuple[EvidenceRef, ...],
) -> CriticalBlocker:
    definition = load_reason_code_registry().require(gate.reason_code)
    return CriticalBlocker(
        code=gate.reason_code.upper(),
        severity="critical",
        message=definition.description,
        evidence_refs=evidence_refs,
    )


def _limited_refs(refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    return refs[:3]


def _recommendation_groups(
    contexts: tuple[_ObjectDecisionContext, ...],
) -> tuple[_RecommendationGroup, ...]:
    grouped: dict[tuple[DecisionAction, DataModality, str], list[_ObjectDecisionContext]]
    grouped = defaultdict(list)
    for context in contexts:
        action = _recommended_action_for_context(context)
        if action is None:
            continue
        primary_reason = _primary_reason(context.decision.reasons)
        grouped[(action, context.decision.modality, primary_reason)].append(context)

    groups: list[_RecommendationGroup] = []
    for (action, modality, primary_reason), members in grouped.items():
        groups.append(
            _RecommendationGroup(
                action=action,
                modality=modality,
                primary_reason=primary_reason,
                object_ids=tuple(member.decision.object_id for member in members),
                reason_codes=tuple(
                    dict.fromkeys(
                        code
                        for member in members
                        for code in member.decision.reasons
                        if code != "export_ready"
                    )
                ),
            )
        )
    return tuple(groups)


def _recommended_action_for_context(
    context: _ObjectDecisionContext,
) -> DecisionAction | None:
    if context.decision.action in {
        DecisionAction.KEEP,
        DecisionAction.EXPORT_READY,
    }:
        return None
    if context.decision.action is DecisionAction.BLOCK_EXPORT:
        if "pii_unmasked" in context.decision.reasons:
            return DecisionAction.SEND_TO_PRIVACY_REVIEW
        return None
    return context.decision.action


def _primary_reason(reason_codes: tuple[str, ...]) -> str:
    for code in (
        "probable_label_error",
        "ambiguous_object",
        "text_pii_detected",
        "pii_unmasked",
        "exact_duplicate",
        "rare_class_underrepresented",
        "missing_numeric",
    ):
        if code in reason_codes:
            return code
    return reason_codes[0] if reason_codes else "review_required"


def _recommended_actions(
    groups: tuple[_RecommendationGroup, ...],
) -> tuple[RecommendedAction, ...]:
    actions: list[RecommendedAction] = []
    for index, group in enumerate(groups, start=1):
        actions.append(
            RecommendedAction(
                recommendation_id=f"rec_{_slug(group.action.value)}_{group.primary_reason}_{index}",
                action=group.action,
                modality=group.modality,
                count=len(group.object_ids),
                reason_codes=group.reason_codes or (group.primary_reason,),
                reason=_recommendation_reason(group),
                segment=f"reason={group.primary_reason}",
                method_recommendation_id=None,
                requires_approval=group.action is DecisionAction.GENERATE_SYNTHETIC_CANDIDATE,
            )
        )
    return tuple(actions)


def _recommendation_reason(group: _RecommendationGroup) -> str:
    if group.primary_reason == "probable_label_error":
        return "Probable label errors require label review before training or export."
    if group.primary_reason == "ambiguous_object":
        return "Ambiguous prediction evidence requires label review."
    if group.primary_reason in {"text_pii_detected", "pii_unmasked"}:
        return "Privacy evidence requires review or redaction before export."
    if group.primary_reason == "exact_duplicate":
        return "Exact duplicate evidence supports duplicate removal."
    if group.primary_reason == "rare_class_underrepresented":
        return "Underrepresented class evidence supports rare-class augmentation."
    if group.primary_reason == "missing_numeric":
        return "Missing numeric evidence supports imputation review."
    return "Decision Core evidence requires review."


def _readiness(
    *,
    blockers: tuple[CriticalBlocker, ...],
    recommended_actions: tuple[RecommendedAction, ...],
) -> ReadinessAssessment:
    reason_codes = tuple(
        dict.fromkeys(
            [
                *(blocker.code.lower() for blocker in blockers),
                *(code for action in recommended_actions for code in action.reason_codes),
            ]
        )
    )
    if blockers:
        return ReadinessAssessment(
            status=DatasetReadiness.BLOCKED,
            score=0.0,
            reason_codes=reason_codes,
        )
    if recommended_actions:
        return ReadinessAssessment(
            status=DatasetReadiness.NOT_READY_FOR_EXPORT,
            score=0.75,
            reason_codes=reason_codes,
        )
    return ReadinessAssessment(
        status=DatasetReadiness.READY_FOR_EXPORT,
        score=1.0,
        reason_codes=(),
    )


def _dataset_decision(
    blockers: tuple[CriticalBlocker, ...],
    recommended_actions: tuple[RecommendedAction, ...],
) -> DatasetDecision:
    if blockers:
        return DatasetDecision.BLOCKED
    if recommended_actions:
        return DatasetDecision.NEEDS_REVIEW
    return DatasetDecision.READY_FOR_TRAINING


def _recommended_next_job(
    *,
    blockers: tuple[CriticalBlocker, ...],
    recommended_actions: tuple[RecommendedAction, ...],
) -> WorkflowType | None:
    if recommended_actions:
        return WorkflowType.PREVIEW_ACTION_PLAN
    if blockers:
        return None
    return WorkflowType.EXPORT


def _decision_report_id(
    *,
    request: BuildDecisionReportRequest,
    evidence_bundles: tuple[EvidenceBundle, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(request.dataset_id.encode("utf-8"))
    digest.update(request.version_id.encode("utf-8"))
    for evidence in evidence_bundles:
        digest.update(evidence.evidence_bundle_id.encode("utf-8"))
        digest.update(evidence.object_id.encode("utf-8"))
    return f"decision_report_{digest.hexdigest()[:16]}"


def _slug(value: str) -> str:
    return value.lower().replace("_", "-")


__all__ = [
    "DECISION_REPORT_SCHEMA_VERSION",
    "METHOD_POLICY_VERSION",
    "PRIVACY_POLICY_VERSION",
    "BuildDecisionReportRequest",
    "build_decision_report",
]
