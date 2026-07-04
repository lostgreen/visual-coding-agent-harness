"""Shared agent budget/configuration values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping

from .contracts import VISUAL_EVIDENCE_NFRAMES


SlotName = Literal[
    "task",
    "trajectory",
    "hypothesis",
    "evidence",
    "scene_index",
    "feedback",
    "budget",
    "tooling",
]

DEFAULT_SLOT_RATIOS: Dict[SlotName, float] = {
    "task": 0.08,
    "trajectory": 0.07,
    "hypothesis": 0.12,
    "evidence": 0.28,
    "scene_index": 0.22,
    "feedback": 0.10,
    "budget": 0.05,
    "tooling": 0.08,
}


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


def parse_budget_ratios(value: str) -> Dict[SlotName, float]:
    ratios: Dict[SlotName, float] = {}
    allowed = set(DEFAULT_SLOT_RATIOS)
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid budget ratio item: {item}")
        key, raw_ratio = item.split(":", 1)
        slot = key.strip()
        if slot not in allowed:
            raise ValueError(f"Unknown budget slot: {slot}")
        ratios[slot] = float(raw_ratio)
    missing = allowed - set(ratios)
    if missing:
        raise ValueError(f"Missing budget ratios for: {', '.join(sorted(missing))}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Budget ratios must sum to 1.0, got {total:.3f}")
    return ratios
