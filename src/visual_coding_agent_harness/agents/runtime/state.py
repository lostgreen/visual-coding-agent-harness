"""Run, round, and finalization state containers for iterative runtime refactors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class RunState:
    question: str
    video_path: str
    question_route: str
    grounding_runtime: Any | None = None
    bootstrap_failure: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RoundState:
    round_number: int
    recent_observations: list[Any] = field(default_factory=list)
    hypothesis: str | None = None
    reflection_memory: tuple[str, ...] = ()
    normalization_notes: list[Any] = field(default_factory=list)


@dataclass
class FinalizationContext:
    question_context: Any
    effective_route: str
    citations: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    planner_final_answer_blocked: bool = False
    planner_final_low_confidence_blocked: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
