from __future__ import annotations

from visual_coding_agent_harness.agents import iterative_agent
from visual_coding_agent_harness.agents.skill_runtime import (
    _initial_skill_runtime_state,
    _skill_id,
    update_effective_skill_runtime,
)
from visual_coding_agent_harness.agents.runtime.state import FinalizationContext, RoundState, RunState


def test_skill_runtime_symbols_are_not_reexported_from_iterative_agent() -> None:
    assert not hasattr(iterative_agent, "SkillRuntimeState")
    assert not hasattr(iterative_agent, "_initial_skill_runtime_state")
    assert not hasattr(iterative_agent, "_skill_id")


def test_initial_skill_runtime_state_keeps_current_lock_semantics() -> None:
    state = _initial_skill_runtime_state("How was his life journey according to the video?", route="temporal_order")

    assert state.locked is True
    assert state.selected_round == 1
    assert _skill_id(state.effective_skill) in state.compatible_skill_ids


def test_update_effective_skill_runtime_writes_trace_for_locked_change() -> None:
    state = _initial_skill_runtime_state("How was his life journey according to the video?", route="temporal_order")
    requested = state.recommended_skill
    events: list[tuple[str, dict[str, object]]] = []

    update_effective_skill_runtime(
        state,
        requested_skill=requested,
        requested_skill_text=_skill_id(requested),
        round_number=2,
        rationale="same skill",
        executed_rounds=0,
        supported_binding_no_growth_rounds=0,
        no_evidence_growth_rounds=0,
        write_trace_event=lambda event_type, payload: events.append((event_type, dict(payload))),
    )

    assert events == []


def test_runtime_state_dataclasses_are_importable() -> None:
    run_state = RunState(question="q", video_path="/videos/demo.mp4", question_route="needle_local")
    round_state = RoundState(round_number=1)
    finalization = FinalizationContext(question_context="ctx", effective_route="needle_local")

    assert run_state.question == "q"
    assert round_state.recent_observations == []
    assert finalization.planner_final_answer_blocked is False


def test_default_runstate_has_empty_sets() -> None:
    run_state = RunState(question="q", video_path="/videos/demo.mp4")

    assert run_state.raw_question == ""
    assert run_state.vlm_safe_question == ""
    assert run_state.effective_route == ""
    assert run_state.inspected_segment_ids == set()
    assert run_state.seen_tool_semantic_keys == set()
    assert run_state.zero_yield_tool_signatures == set()
    assert run_state.executed_recommended_action_ids == set()
    assert run_state.auto_evidence_promotion_attempted_keys == set()
    assert run_state.route_repair_counts == {}
    assert run_state.exhausted_one_shot_tools == set()
    assert run_state.skill_switch_history == []


def test_roundstate_owns_round_scoped_budget_counter() -> None:
    round_state = RoundState(round_number=2)

    assert round_state.issued_tool_calls == 0
    assert round_state.recent_observations == []
    assert round_state.normalization_notes == []
    assert round_state.hypothesis_snapshot is None
