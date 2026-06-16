from __future__ import annotations

from visual_coding_agent_harness.agents.skills.specs import builtin_skill_registry, skill_catalog_prompt
from visual_coding_agent_harness.agents.skill_runtime import (
    SkillRuntimeState,
    SkillSwitchRecord,
    TransitionDecision,
    TransitionPolicy,
    TransitionVerdict,
    update_effective_skill_runtime,
    _initial_skill_runtime_state,
    _skill_id,
)
from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.prompt_stack import build_replanning_prompt
from visual_coding_agent_harness.contracts import ClaimModality
from visual_coding_agent_harness.video_index import fixed_window_scene_index


def test_all_skills_have_description_and_when_to_use() -> None:
    for skill in builtin_skill_registry().list():
        assert 0 < len(skill.guide.description) <= 240
        assert 0 < len(skill.guide.when_to_use) <= 240


def test_general_exploration_is_registered() -> None:
    skill = builtin_skill_registry().get("general_exploration")

    assert skill.name == "general_exploration"
    assert skill.version == 1
    assert "Fallback skill" in skill.guide.description


def test_general_exploration_policy_is_conservative() -> None:
    policy = builtin_skill_registry().get("general_exploration").policy

    assert ClaimModality.VISUAL_FACT in policy.allowed_modalities
    assert ClaimModality.NARRATED_FACT in policy.allowed_modalities
    assert "basic_grounding_check" in policy.verifier_checks
    assert "at_least_one_citation" in policy.final_conditions


def test_catalog_renders_guide_not_policy() -> None:
    rendered = skill_catalog_prompt()

    assert "description:" in rendered
    assert "when_to_use:" in rendered
    assert "suggested_actions:" in rendered
    assert "allowed_modalities" not in rendered
    assert "verifier_checks" not in rendered
    assert "forbidden_actions" not in rendered


def test_transition_policy_thrashing() -> None:
    registry = builtin_skill_registry()
    current_skill = registry.get("visual_timeline_qa")
    state = SkillRuntimeState(
        recommended_skill=current_skill,
        compatible_skill_ids=("visual_timeline_qa@v1", "mixed_asr_visual_qa@v1"),
        effective_skill=current_skill,
        effective_policy=current_skill.policy,
    )

    verdict = TransitionPolicy(registry).evaluate(
        current=state,
        proposed_guide_name="mixed_asr_visual_qa",
        task_route="temporal_order",
        evidence_state=None,
        rationale="need another modality",
        recent_switches=(
            SkillSwitchRecord(round_number=1, from_skill="a", to_skill="b"),
            SkillSwitchRecord(round_number=2, from_skill="b", to_skill="a"),
        ),
    )

    assert verdict.decision is TransitionDecision.REJECTED_THRASHING
    assert verdict.policy is current_skill.policy


def test_rejected_transition_does_not_mutate_state() -> None:
    class ThrashingPolicy:
        def __init__(self, guide, policy) -> None:
            self.guide = guide
            self.policy = policy

        def evaluate(self, **kwargs) -> TransitionVerdict:
            del kwargs
            return TransitionVerdict(
                decision=TransitionDecision.REJECTED_THRASHING,
                guide=self.guide,
                policy=self.policy,
                reason="recent_switch_limit",
            )

    registry = builtin_skill_registry()
    current_skill = registry.get("visual_timeline_qa")
    requested_skill = registry.get("mixed_asr_visual_qa")
    state = SkillRuntimeState(
        recommended_skill=current_skill,
        compatible_skill_ids=("visual_timeline_qa@v1", "mixed_asr_visual_qa@v1"),
        effective_skill=current_skill,
        effective_policy=current_skill.policy,
        override_reason="classifier default",
    )
    previous_skill = state.effective_skill
    previous_policy = state.effective_policy
    previous_reason = state.override_reason
    events: list[tuple[str, dict[str, object]]] = []

    update_effective_skill_runtime(
        state,
        requested_skill=requested_skill,
        requested_skill_text="mixed_asr_visual_qa@v1",
        round_number=3,
        rationale="planner wants another modality",
        executed_rounds=2,
        supported_binding_no_growth_rounds=0,
        no_evidence_growth_rounds=0,
        write_trace_event=lambda event_type, payload: events.append((event_type, dict(payload))),
        transition_policy=ThrashingPolicy(requested_skill, requested_skill.policy),
        recent_switches=(
            SkillSwitchRecord(round_number=1, from_skill="visual_timeline_qa@v1", to_skill="mixed_asr_visual_qa@v1"),
            SkillSwitchRecord(round_number=2, from_skill="mixed_asr_visual_qa@v1", to_skill="visual_timeline_qa@v1"),
        ),
    )

    assert state.effective_skill is previous_skill
    assert state.effective_policy is previous_policy
    assert state.override_reason == previous_reason
    assert events == [
        (
            "skill_transition_rejected",
            {
                "round": 3,
                "requested_skill": "mixed_asr_visual_qa@v1",
                "effective_skill": "visual_timeline_qa@v1",
                "decision": "rejected_thrashing",
                "reason": "recent_switch_limit",
            },
        )
    ]


def test_transition_policy_accepts_strictening() -> None:
    registry = builtin_skill_registry()
    current_skill = registry.get("general_exploration")
    state = SkillRuntimeState(
        recommended_skill=current_skill,
        compatible_skill_ids=("general_exploration@v1", "visual_timeline_qa@v1"),
        effective_skill=current_skill,
        effective_policy=current_skill.policy,
    )

    verdict = TransitionPolicy(registry).evaluate(
        current=state,
        proposed_guide_name="visual_timeline_qa",
        task_route="temporal_order",
        evidence_state=None,
        rationale="visual order question",
        recent_switches=(),
    )

    assert verdict.decision is TransitionDecision.ACCEPTED
    assert verdict.policy is registry.get("visual_timeline_qa").policy


def test_transition_policy_rejects_disjoint_modality_switch() -> None:
    registry = builtin_skill_registry()
    current_skill = registry.get("narration_timeline_qa")
    state = SkillRuntimeState(
        recommended_skill=current_skill,
        compatible_skill_ids=("narration_timeline_qa@v1", "visual_timeline_qa@v1"),
        effective_skill=current_skill,
        effective_policy=current_skill.policy,
    )

    verdict = TransitionPolicy(registry).evaluate(
        current=state,
        proposed_guide_name="visual_timeline_qa",
        task_route="temporal_order",
        evidence_state=None,
        rationale="",
        recent_switches=(),
    )

    assert verdict.decision is TransitionDecision.REJECTED_INCOMPATIBLE
    assert verdict.policy is current_skill.policy


def test_transition_policy_keeps_policy_when_loosening() -> None:
    registry = builtin_skill_registry()
    current_skill = registry.get("visual_timeline_qa")
    state = SkillRuntimeState(
        recommended_skill=current_skill,
        compatible_skill_ids=("visual_timeline_qa@v1", "general_exploration@v1"),
        effective_skill=current_skill,
        effective_policy=current_skill.policy,
    )

    verdict = TransitionPolicy(registry).evaluate(
        current=state,
        proposed_guide_name="general_exploration",
        task_route="temporal_order",
        evidence_state=None,
        rationale="fallback",
        recent_switches=(),
    )

    assert verdict.decision is TransitionDecision.ACCEPTED_WITH_POLICY_UNCHANGED
    assert verdict.policy is current_skill.policy


def test_update_effective_skill_runtime_emits_transition_event() -> None:
    registry = builtin_skill_registry()
    current_skill = registry.get("general_exploration")
    requested = registry.get("visual_timeline_qa")
    state = SkillRuntimeState(
        recommended_skill=current_skill,
        compatible_skill_ids=("general_exploration@v1", "visual_timeline_qa@v1"),
        effective_skill=current_skill,
        effective_policy=current_skill.policy,
    )
    events: list[tuple[str, dict[str, object]]] = []

    update_effective_skill_runtime(
        state,
        requested_skill=requested,
        requested_skill_text=_skill_id(requested),
        round_number=2,
        rationale="visual order question",
        executed_rounds=0,
        supported_binding_no_growth_rounds=0,
        no_evidence_growth_rounds=0,
        write_trace_event=lambda event_type, payload: events.append((event_type, dict(payload))),
        transition_policy=TransitionPolicy(registry),
    )

    assert state.effective_skill is requested
    assert state.effective_policy is requested.policy
    assert events[-1][0] == "skill_transition_accepted"
    assert events[-1][1]["decision"] == "accepted"


def test_initial_skill_runtime_state_records_recommendation_source() -> None:
    planned = _initial_skill_runtime_state("What is visible?", recommended_skill_id="general_exploration@v1")
    routed = _initial_skill_runtime_state("What is visible?")

    assert planned.recommendation_source == "grounding_plan"
    assert routed.recommendation_source == "route_classifier"


def test_base_identity_explains_ownership() -> None:
    prompt, _report = build_replanning_prompt(
        question="What is visible?",
        scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0),
        ledger_text="# Compact Evidence Context\n(none)",
        round_number=1,
        budget=AgentBudget(max_rounds=2),
        allocator=default_context_budget_allocator(total_budget_tokens=12000),
        active_skill="visual_timeline_qa@v1",
    )

    assert "Skill selection is your choice" in prompt
    assert "Evidence policy is the harness's responsibility" in prompt
    assert "skill_locked: true" not in prompt
    assert "To switch skill: set" in prompt
    assert "general_exploration" in prompt
