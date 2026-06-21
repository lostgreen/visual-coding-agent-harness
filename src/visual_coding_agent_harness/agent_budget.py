"""Shared agent budget/configuration values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .agent_contracts import VISUAL_EVIDENCE_NFRAMES


@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = 8
    max_tool_calls_per_round: int = 2
    default_nframes: int = VISUAL_EVIDENCE_NFRAMES
    high_fps_nframes: int = 32
    planner_receives_media: bool = False
    reserve_final_round: bool = True
    answer_probe_rounds_before_final: int = 0
    persist_planner_io: bool = True
    planner_io_max_chars: int = 200_000
    context_budget_tokens: int = 12000
    context_budget_ratios: Mapping[str, float] | None = None
    max_repeated_programs: int = 2
    max_repeated_invalid_programs: int = 3
    hard_skill_runtime: bool = False
    planner_owned_grounding: bool = False
    prompt_role_split_enabled: bool = False
    reflection_memory_max_items: int = 5
    rewrite_mcq_for_exploration: bool = False
