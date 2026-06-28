"""Read-only evidence query facade over workspace state."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .item import EvidenceItem
from .need import EvidenceNeed

_VERIFIED_POLARITIES = frozenset({"supports", "refutes", "absent", "inconclusive"})
_VERIFIED_MEMORY_KINDS = frozenset(
    {
        "visual_support",
        "answer_support",
        "caption_support",
        "synthesized_support",
        "answer_conflict",
        "answer_conflict_resolved",
        "contradiction",
        "contradicting",
        "conflict",
        "local_negative",
        "verification_uncertain",
    }
)


class EvidenceLedger:
    """Read-only queries over memory entries, sub-goals, and findings."""

    def __init__(self, *, workspace: Any, mutator: Any) -> None:
        self.workspace = workspace
        self.mutator = mutator

    def items(self) -> list[EvidenceItem]:
        return [EvidenceItem.from_memory_entry(entry) for entry in self.workspace.memory_entries()]

    def items_for_option(self, option_id: str) -> list[EvidenceItem]:
        normalized = _option_id(option_id)
        return [item for item in self.items() if _option_id(item.option_id) == normalized]

    def supports_by_option(self) -> dict[str, list[EvidenceItem]]:
        return _items_by_option(item for item in self.items() if item.polarity == "supports")

    def refutes_by_option(self) -> dict[str, list[EvidenceItem]]:
        return _items_by_option(item for item in self.items() if item.polarity == "refutes")

    def needs(self) -> list[EvidenceNeed]:
        return [EvidenceNeed.from_sub_goal(sub_goal) for sub_goal in self.mutator.sub_goals()]

    def open_needs(self) -> list[EvidenceNeed]:
        return [need for need in self.needs() if need.status == "open"]

    def verified_windows_for_option(self, option_id: str) -> set[tuple[str, float, float]]:
        normalized = _option_id(option_id)
        windows: set[tuple[str, float, float]] = set()
        for item in self.items():
            if (
                _option_id(item.option_id) != normalized
                or item.polarity not in _VERIFIED_POLARITIES
                or item.memory_kind not in _VERIFIED_MEMORY_KINDS
            ):
                continue
            if item.segment_id is None or item.time_range is None:
                continue
            windows.add((item.segment_id, float(item.time_range[0]), float(item.time_range[1])))
        return windows

    def coverage_by_segment(self) -> dict[str, float]:
        coverage: dict[str, float] = defaultdict(float)
        seen: set[tuple[str, float, float]] = set()
        for item in self.items():
            if item.polarity not in _VERIFIED_POLARITIES or item.memory_kind not in _VERIFIED_MEMORY_KINDS:
                continue
            if item.segment_id is None or item.time_range is None:
                continue
            key = (item.segment_id, float(item.time_range[0]), float(item.time_range[1]))
            if key in seen:
                continue
            seen.add(key)
            coverage[item.segment_id] += max(0.0, key[2] - key[1])
        return dict(coverage)


def _items_by_option(items) -> dict[str, list[EvidenceItem]]:
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in items:
        option_id = _option_id(item.option_id)
        if option_id:
            grouped[option_id].append(item)
    return dict(grouped)


def _option_id(value: str | None) -> str:
    return str(value or "").strip().upper()[:1]
