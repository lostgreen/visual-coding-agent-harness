"""Run, round, and finalization state containers for iterative runtime refactors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class AnswerSuggestionState:
    option: str = ""
    citations: tuple[str, ...] = ()
    confidence: float = 0.0
    count: int = 0


@dataclass
class RunState:
    question: str
    video_path: str
    question_route: str = ""
    raw_question: str = ""
    vlm_safe_question: str = ""
    effective_route: str = ""
    inspected_segment_ids: set[str] = field(default_factory=set)
    final_round_reserved: bool = False
    planner_skill_snapshot: str = ""
    seen_tool_semantic_keys: set[str] = field(default_factory=set)
    zero_yield_tool_signatures: set[str] = field(default_factory=set)
    program_key_counts: dict[str, int] = field(default_factory=dict)
    banned_program_keys: set[str] = field(default_factory=set)
    executed_recommended_action_ids: set[str] = field(default_factory=set)
    auto_evidence_promotion_attempted_keys: set[Any] = field(default_factory=set)
    route_repair_counts: dict[Any, int] = field(default_factory=dict)
    route_repair_exhausted: Mapping[str, Any] | None = None
    exhausted_one_shot_tools: set[str] = field(default_factory=set)
    skill_switch_history: list[Any] = field(default_factory=list)
    answer_suggestion_state: AnswerSuggestionState = field(default_factory=AnswerSuggestionState)
    answer_feedback: list[str] = field(default_factory=list)
    pending_inferences: list[str] = field(default_factory=list)
    requested_prompt_tool_names: set[str] = field(default_factory=set)
    last_normalization_notes: list[Any] = field(default_factory=list)
    no_progress_warning_emitted: bool = False
    grounding_runtime: Any | None = None
    bootstrap_failure: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RoundState:
    round_number: int
    issued_tool_calls: int = 0
    recent_observations: list[Any] = field(default_factory=list)
    normalization_notes: list[Any] = field(default_factory=list)
    hypothesis_snapshot: str | None = None


@dataclass
class FinalizationContext:
    question_context: Any
    effective_route: str
    citations: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    planner_final_answer_blocked: bool = False
    planner_final_low_confidence_blocked: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
