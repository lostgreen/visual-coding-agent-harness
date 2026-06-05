from visual_coding_agent_harness.agents.context_budget import (
    CompactStrategy,
    ContextBudgetAllocator,
)


class TruncateStrategy(CompactStrategy):
    name = "truncate"

    def __init__(self):
        self.calls = []

    def compact(self, content, budget, ctx):
        self.calls.append({"content": content, "budget": budget, "ctx": ctx})
        return content[:budget]


class NoopStrategy(CompactStrategy):
    name = "noop"

    def compact(self, content, budget, ctx):
        return content


def test_no_compact_when_under_budget():
    strategy = TruncateStrategy()
    allocator = ContextBudgetAllocator(
        total_budget_tokens=100,
        slot_ratios={"task": 0.5},
        token_counter=len,
    )
    allocator.register_strategy("task", strategy)

    allocated, report = allocator.allocate({"task": "short"})

    assert allocated == {"task": "short"}
    assert strategy.calls == []
    assert report.total_budget_tokens == 100
    assert report.used_tokens_per_slot == {"task": 5}
    assert report.compact_events == []
    assert report.overflow is False
    assert report.turn_index == 0


def test_compact_called_when_over():
    strategy = TruncateStrategy()
    allocator = ContextBudgetAllocator(
        total_budget_tokens=10,
        slot_ratios={"evidence": 0.5},
        token_counter=len,
    )
    allocator.register_strategy("evidence", strategy)

    allocated, report = allocator.allocate({"evidence": "abcdefghi"})

    assert allocated == {"evidence": "abcde"}
    assert strategy.calls == [
        {"content": "abcdefghi", "budget": 5, "ctx": {"turn": 0}}
    ]
    assert report.used_tokens_per_slot == {"evidence": 5}
    assert report.compact_events == [
        {
            "slot": "evidence",
            "before_tokens": 9,
            "after_tokens": 5,
            "strategy": "truncate",
        }
    ]
    assert report.overflow is False


def test_overflow_flag_when_strategy_insufficient():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=10,
        slot_ratios={"feedback": 0.5},
        token_counter=len,
    )
    allocator.register_strategy("feedback", NoopStrategy())

    allocated, report = allocator.allocate({"feedback": "still too long"})

    assert allocated == {"feedback": "still too long"}
    assert report.used_tokens_per_slot == {"feedback": 14}
    assert report.compact_events == [
        {
            "slot": "feedback",
            "before_tokens": 14,
            "after_tokens": 14,
            "strategy": "noop",
        }
    ]
    assert report.overflow is True
