"""Context budget allocation primitives for prompt slots."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
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
        self, slots: Dict[SlotName, str], *, ctx: Optional[Dict[str, Any]] = None
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
                        content, budget, ctx={"turn": self._turn, **(ctx or {})}
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


class TaskSlotCompact(CompactStrategy):
    name = "task_no_compact"

    def compact(self, content: str, budget: int, ctx: Dict[str, Any]) -> str:
        return content


class NavLatestWinsCompact(CompactStrategy):
    name = "nav_latest_wins"

    def compact(self, content: str, budget: int, ctx: Dict[str, Any]) -> str:
        blocks = _split_blocks(content)
        kept: list[str] = []
        remaining = max(0, budget)
        for block in reversed(blocks):
            cost = _rough_tokens(block)
            if cost <= remaining or not kept:
                kept.append(block)
                remaining -= min(cost, remaining)
            if remaining <= 0:
                break
        return "\n\n".join(reversed(kept))


class EvidenceTieredCompact(CompactStrategy):
    name = "evidence_tiered"

    def compact(self, content: str, budget: int, ctx: Dict[str, Any]) -> str:
        rows = _ledger_rows(content)
        if not rows:
            return _truncate_to_budget(content, budget)
        active_query = str(ctx.get("active_followup_target_query", ""))
        scored = [
            (_relevance(row, active_query) + index / max(len(rows), 1), row)
            for index, row in enumerate(rows, start=1)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[str] = []
        remaining = max(0, budget)
        for _, row in scored:
            full = row.strip()
            short = _short_ledger_row(full)
            chosen = full if _rough_tokens(full) + 20 <= remaining else short
            cost = _rough_tokens(chosen)
            if cost > remaining and out:
                continue
            out.append(chosen)
            remaining -= min(cost, remaining)
            if remaining <= 0:
                break
        return "\n".join(out)


class FeedbackLatestOnlyCompact(CompactStrategy):
    name = "feedback_latest_only"

    def compact(self, content: str, budget: int, ctx: Dict[str, Any]) -> str:
        blocks = _split_blocks(content)
        if not blocks:
            return ""
        latest = blocks[-1]
        if len(blocks) == 1:
            return _truncate_to_budget(latest, budget)
        history = []
        for index, block in enumerate(blocks[:-1], start=1):
            missing = _compact_missing_items(block)
            history.append(f"attempt {index}: missing {missing}")
        compact = "\n".join(history + [latest])
        return compact if _rough_tokens(compact) <= budget else _truncate_to_budget(latest, budget)


def default_context_budget_allocator(
    *,
    total_budget_tokens: int = 12000,
    slot_ratios: Optional[Dict[SlotName, float]] = None,
) -> ContextBudgetAllocator:
    allocator = ContextBudgetAllocator(total_budget_tokens=total_budget_tokens, slot_ratios=slot_ratios)
    allocator.register_strategy("task", TaskSlotCompact())
    allocator.register_strategy("navigation", NavLatestWinsCompact())
    allocator.register_strategy("evidence", EvidenceTieredCompact())
    allocator.register_strategy("feedback", FeedbackLatestOnlyCompact())
    return allocator


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


def _split_blocks(content: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content or "") if block.strip()]
    return blocks or ([content.strip()] if content.strip() else [])


def _ledger_rows(content: str) -> list[str]:
    return [
        line.strip()
        for line in (content or "").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _relevance(row: str, query: str) -> float:
    if not query:
        return 0.0
    row_terms = set(re.findall(r"[a-z0-9]+", row.lower()))
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not row_terms or not query_terms:
        return 0.0
    return len(row_terms & query_terms) / len(query_terms)


def _short_ledger_row(row: str) -> str:
    parts = [part.strip() for part in row.split("|") if part.strip()]
    if len(parts) >= 2:
        return " | ".join(parts[:2])
    return row[:160]


def _compact_missing_items(block: str) -> str:
    text = re.sub(r"\s+", " ", block or "").strip()
    if not text:
        return "unknown"
    return text[:96] + ("..." if len(text) > 96 else "")


def _rough_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _truncate_to_budget(content: str, budget: int) -> str:
    if budget <= 0:
        return ""
    char_limit = max(1, budget * 4)
    text = content or ""
    return text[:char_limit]
