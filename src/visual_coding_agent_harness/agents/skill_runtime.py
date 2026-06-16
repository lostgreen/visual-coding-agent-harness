"""Runtime skill selection state and lock-policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .question_policy import classify_narration_subroute, classify_question_route
from .skills.specs import EvidencePolicy, SkillRegistry, SkillSpec, builtin_skill_registry, select_skill


@dataclass
class SkillRuntimeState:
    recommended_skill: SkillSpec
    compatible_skill_ids: tuple[str, ...]
    effective_skill: SkillSpec | None = None
    effective_policy: EvidencePolicy | None = None
    locked: bool = False
    selected_round: int | None = None
    unlock_used: bool = False
    override_reason: str = ""
    recommendation_source: str = "route_classifier"


class TransitionDecision(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_WITH_POLICY_UNCHANGED = "policy_unchanged"
    REJECTED_INCOMPATIBLE = "rejected_incompatible"
    REJECTED_THRASHING = "rejected_thrashing"


@dataclass(frozen=True)
class SkillSwitchRecord:
    round_number: int
    from_skill: str
    to_skill: str


@dataclass(frozen=True)
class TransitionVerdict:
    decision: TransitionDecision
    guide: SkillSpec | None = None
    policy: EvidencePolicy | None = None
    reason: str = ""


class TransitionPolicy:
    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self.registry = registry or builtin_skill_registry()

    def evaluate(
        self,
        *,
        current: SkillRuntimeState,
        proposed_guide_name: str,
        task_route: str,
        evidence_state: Any,
        rationale: str,
        recent_switches: tuple[SkillSwitchRecord, ...],
    ) -> TransitionVerdict:
        del task_route, evidence_state, rationale
        proposed_name = _skill_name_from_id(proposed_guide_name)
        try:
            proposed = self.registry.get(proposed_name)
        except KeyError:
            return TransitionVerdict(
                decision=TransitionDecision.REJECTED_INCOMPATIBLE,
                guide=current.effective_skill,
                policy=current.effective_policy,
                reason="unknown_skill",
            )
        current_policy = current.effective_policy or (
            current.effective_skill.policy if current.effective_skill is not None else current.recommended_skill.policy
        )
        recent = recent_switches[-3:]
        if len(recent) >= 2:
            return TransitionVerdict(
                decision=TransitionDecision.REJECTED_THRASHING,
                guide=proposed,
                policy=current_policy,
                reason="recent_switch_limit",
            )
        proposed_modalities = set(proposed.policy.allowed_modalities)
        current_modalities = set(current_policy.allowed_modalities)
        if proposed_modalities and current_modalities and proposed_modalities.isdisjoint(current_modalities):
            return TransitionVerdict(
                decision=TransitionDecision.REJECTED_INCOMPATIBLE,
                guide=current.effective_skill,
                policy=current_policy,
                reason="disjoint_evidence_modalities",
            )
        if proposed_modalities < current_modalities:
            return TransitionVerdict(decision=TransitionDecision.ACCEPTED, guide=proposed, policy=proposed.policy)
        if current_modalities < proposed_modalities:
            return TransitionVerdict(
                decision=TransitionDecision.ACCEPTED_WITH_POLICY_UNCHANGED,
                guide=proposed,
                policy=current_policy,
                reason="policy_would_loosen",
            )
        return TransitionVerdict(decision=TransitionDecision.ACCEPTED, guide=proposed, policy=proposed.policy)


def _planner_selected_skill(action: Mapping[str, Any]) -> tuple[SkillSpec | None, dict[str, str]]:
    requested = str(action.get("skill", "") or "").strip()
    if not requested:
        return None, {"status": "missing", "requested_skill": ""}
    name = requested.split("@", 1)[0].strip()
    registry = builtin_skill_registry()
    try:
        return registry.get(name), {"status": "selected", "requested_skill": requested}
    except KeyError:
        return None, {"status": "invalid", "requested_skill": requested}


def _initial_skill_runtime_state(
    question: str,
    *,
    route: str | None = None,
    recommended_skill_id: str = "",
) -> SkillRuntimeState:
    recommended = _recommended_effective_skill(
        question,
        route=route,
        recommended_skill_id=recommended_skill_id,
    )
    return SkillRuntimeState(
        recommended_skill=recommended,
        compatible_skill_ids=_compatible_skill_ids(question=question, recommended=recommended, route=route),
        effective_skill=recommended,
        effective_policy=recommended.policy,
        locked=True,
        selected_round=1,
        recommendation_source="grounding_plan" if recommended_skill_id else "route_classifier",
    )


def _recommended_effective_skill(
    question: str,
    *,
    route: str | None = None,
    recommended_skill_id: str = "",
) -> SkillSpec:
    registry = builtin_skill_registry()
    skill_name = _skill_name_from_id(recommended_skill_id)
    if skill_name:
        try:
            return registry.get(skill_name)
        except KeyError:
            pass
    resolved_route = route or classify_question_route(question)
    if resolved_route == "temporal_order":
        if classify_narration_subroute(question) == "narration_timeline":
            try:
                return registry.get("narration_timeline_qa")
            except KeyError:
                pass
        try:
            return registry.get("visual_timeline_qa")
        except KeyError:
            pass
    return select_skill(question, route=resolved_route)


def _compatible_skill_ids(
    *,
    question: str,
    recommended: SkillSpec,
    route: str | None = None,
) -> tuple[str, ...]:
    recommended_id = _skill_id(recommended)
    resolved_route = route or classify_question_route(question)
    if resolved_route == "temporal_order":
        if recommended.name == "narration_timeline_qa" or classify_narration_subroute(question) == "narration_timeline":
            return _unique_tuple(
                [
                    recommended_id,
                    "mixed_asr_visual_qa@v1",
                    "visual_timeline_qa@v1",
                ]
            )
        return _unique_tuple([recommended_id, "mixed_asr_visual_qa@v1"])
    return (recommended_id,)


def update_effective_skill_runtime(
    state: SkillRuntimeState,
    *,
    requested_skill: SkillSpec | None,
    requested_skill_text: str,
    round_number: int,
    rationale: str,
    executed_rounds: int,
    supported_binding_no_growth_rounds: int,
    no_evidence_growth_rounds: int,
    write_trace_event: Callable[[str, Mapping[str, Any]], None] | None = None,
    transition_policy: TransitionPolicy | None = None,
    recent_switches: tuple[SkillSwitchRecord, ...] = (),
) -> None:
    if requested_skill is None:
        return
    current_id = _skill_id(state.effective_skill)
    requested_id = _skill_id(requested_skill)
    if requested_skill.name == "timeline_ordering" and current_id in {
        "narration_timeline_qa@v1",
        "visual_timeline_qa@v1",
    }:
        _write_trace_event(
            write_trace_event,
            "legacy_skill_deprecated",
            {
                "round": round_number,
                "requested_skill": requested_skill_text,
                "effective_skill": current_id,
                "message": "timeline_ordering@v1 is retained for replay only; the run-level effective skill stays locked.",
            },
        )
        return
    if requested_id == current_id:
        return
    if requested_id not in state.compatible_skill_ids:
        _write_trace_event(
            write_trace_event,
            "skill_transition_rejected",
            {
                "round": round_number,
                "requested_skill": requested_skill_text,
                "effective_skill": current_id,
                "decision": TransitionDecision.REJECTED_INCOMPATIBLE.value,
                "reason": "incompatible_skill",
            },
        )
        _write_trace_event(
            write_trace_event,
            "effective_skill_change_rejected",
            {
                "round": round_number,
                "requested_skill": requested_skill_text,
                "effective_skill": current_id,
                "reason": "incompatible_skill",
                "compatible_skills": list(state.compatible_skill_ids),
            },
        )
        return
    verdict = (transition_policy or TransitionPolicy()).evaluate(
        current=state,
        proposed_guide_name=requested_id,
        task_route=requested_skill.trigger.route,
        evidence_state=None,
        rationale=rationale,
        recent_switches=recent_switches,
    )
    if verdict.decision in {
        TransitionDecision.ACCEPTED,
        TransitionDecision.ACCEPTED_WITH_POLICY_UNCHANGED,
    }:
        previous_id = current_id
        state.effective_skill = verdict.guide or requested_skill
        if verdict.policy is not None:
            state.effective_policy = verdict.policy
        state.override_reason = rationale
        _write_trace_event(
            write_trace_event,
            "skill_transition_accepted",
            {
                "round": round_number,
                "from": previous_id,
                "to": _skill_id(state.effective_skill),
                "decision": verdict.decision.value,
                "reason": verdict.reason or rationale,
            },
        )
        return
    _write_trace_event(
        write_trace_event,
        "skill_transition_rejected",
        {
            "round": round_number,
            "requested_skill": requested_skill_text,
            "effective_skill": current_id,
            "decision": verdict.decision.value,
            "reason": verdict.reason,
        },
    )
    return


def _write_trace_event(
    write_trace_event: Callable[[str, Mapping[str, Any]], None] | None,
    event_type: str,
    payload: Mapping[str, Any],
) -> None:
    if write_trace_event is not None:
        write_trace_event(event_type, payload)


def _unique_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_unique_preserving_order([str(value) for value in values if str(value)]))


def _skill_id(skill: SkillSpec | None) -> str:
    if skill is None:
        return ""
    return f"{skill.name}@v{skill.version}"


def _skill_name_from_id(skill_id: str) -> str:
    return str(skill_id or "").strip().split("@", 1)[0].strip()


def _skill_id_from_name(skill_name: str) -> str:
    name = _skill_name_from_id(skill_name)
    if not name:
        return ""
    try:
        return _skill_id(builtin_skill_registry().get(name))
    except KeyError:
        return name


def _rationale_mentions_modality_mismatch(rationale: str) -> bool:
    lowered = str(rationale or "").lower()
    if "modality" in lowered and any(marker in lowered for marker in ("mismatch", "wrong", "switch", "instead")):
        return True
    return any(
        marker in lowered
        for marker in (
            "visual evidence is insufficient",
            "need narration",
            "narration is required",
            "asr is required",
            "transcript is required",
            "cannot be verified visually",
        )
    )


def _unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
