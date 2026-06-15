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
