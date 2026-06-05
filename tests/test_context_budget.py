from visual_coding_agent_harness.agents.context_budget import (
    CompactStrategy,
    ContextBudgetAllocator,
    EvidenceTieredCompact,
    FeedbackLatestOnlyCompact,
    NavLatestWinsCompact,
    parse_budget_ratios,
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


def test_nav_latest_wins_keeps_newest_block():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=1,
        slot_ratios={"navigation": 1.0},
        token_counter=len,
    )
    allocator.register_strategy("navigation", NavLatestWinsCompact())

    allocated, report = allocator.allocate({"navigation": "old block\n\nnew block"})

    assert allocated["navigation"] == "new block"
    assert report.compact_events[0]["strategy"] == "nav_latest_wins"


def test_evidence_tiered_keeps_relevant_row():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=60,
        slot_ratios={"evidence": 1.0},
        token_counter=len,
    )
    allocator.register_strategy("evidence", EvidenceTieredCompact())
    content = "\n".join(
        [
            "| obs_0001 | blue cup | old unrelated detail |",
            "| obs_0002 | red car appears | relevant visual detail |",
        ]
    )

    allocated, report = allocator.allocate(
        {"evidence": content},
        ctx={"active_followup_target_query": "red car"},
    )

    assert "red car" in allocated["evidence"]
    assert report.compact_events[0]["strategy"] == "evidence_tiered"


def test_feedback_latest_only_compacts_history():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=48,
        slot_ratios={"feedback": 1.0},
        token_counter=len,
    )
    allocator.register_strategy("feedback", FeedbackLatestOnlyCompact())

    allocated, _report = allocator.allocate(
        {"feedback": "first missing evidence paragraph\n\nlatest precise gap"},
    )

    assert "attempt 1: missing" in allocated["feedback"]
    assert "latest precise gap" in allocated["feedback"]


def test_parse_budget_ratios_validates_full_distribution():
    ratios = parse_budget_ratios("task:0.1,navigation:0.15,evidence:0.5,feedback:0.25")

    assert ratios == {"task": 0.1, "navigation": 0.15, "evidence": 0.5, "feedback": 0.25}


def test_parse_budget_ratios_rejects_bad_sum():
    try:
        parse_budget_ratios("task:0.1,navigation:0.1,evidence:0.1,feedback:0.1")
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("expected ValueError")
