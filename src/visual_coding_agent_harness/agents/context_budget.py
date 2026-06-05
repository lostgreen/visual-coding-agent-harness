"""Context budget allocation primitives for prompt slots."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


SlotName = Literal["task", "navigation", "evidence", "feedback"]

DEFAULT_SLOT_RATIOS: Dict[SlotName, float] = {
    "task": 0.10,
    "navigation": 0.15,
    "evidence": 0.50,
    "feedback": 0.25,
}


@dataclass
class ContextSlot:
    name: SlotName
    budget_ratio: float
    compact_strategy: str
    content: str = ""
    tokens_used: int = 0
    tokens_budget: int = 0


@dataclass
class ContextBudgetReport:
    total_budget_tokens: int
    used_tokens_per_slot: Dict[SlotName, int] = field(default_factory=dict)
    compact_events: List[Dict[str, Any]] = field(default_factory=list)
    overflow: bool = False
    turn_index: int = 0


class CompactStrategy(ABC):
    name: str

    @abstractmethod
    def compact(self, content: str, budget: int, ctx: Dict[str, Any]) -> str:
        """Return compacted content for a slot budget."""


class ContextBudgetAllocator:
    def __init__(
        self,
        total_budget_tokens: int = 12000,
        slot_ratios: Optional[Dict[SlotName, float]] = None,
        token_counter: Callable[[str], int] = lambda s: len(s) // 4,
    ):
        self.total = total_budget_tokens
        self.ratios = slot_ratios or DEFAULT_SLOT_RATIOS
        self.count_tokens = token_counter
        self._strategies: Dict[SlotName, CompactStrategy] = {}
        self._turn = 0

    def register_strategy(self, slot: SlotName, strategy: CompactStrategy) -> None:
        self._strategies[slot] = strategy

    def allocate(
        self, slots: Dict[SlotName, str]
    ) -> Tuple[Dict[SlotName, str], ContextBudgetReport]:
        report = ContextBudgetReport(
            total_budget_tokens=self.total,
            used_tokens_per_slot={},
            compact_events=[],
            overflow=False,
            turn_index=self._turn,
        )
        final: Dict[SlotName, str] = {}

        for name, content in slots.items():
            budget = int(self.total * self.ratios[name])
            tokens = self.count_tokens(content)

            if tokens > budget:
                strategy = self._strategies.get(name)
                if strategy:
                    before = tokens
                    content = strategy.compact(
                        content, budget, ctx={"turn": self._turn}
                    )
                    tokens = self.count_tokens(content)
                    report.compact_events.append(
                        {
                            "slot": name,
                            "before_tokens": before,
                            "after_tokens": tokens,
                            "strategy": strategy.name,
                        }
                    )
                if tokens > budget:
                    report.overflow = True

            final[name] = content
            report.used_tokens_per_slot[name] = tokens

        self._turn += 1
        return final, report
