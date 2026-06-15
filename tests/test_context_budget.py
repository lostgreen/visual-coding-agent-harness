import pytest

from visual_coding_agent_harness.agents.context_budget import (
    BudgetExceededError,
    CompactStrategy,
    ContextBudgetAllocator,
    EvidenceTieredCompact,
    FeedbackLatestOnlyCompact,
    NavLatestWinsCompact,
    parse_budget_ratios,
)
from visual_coding_agent_harness.workspace import EvidenceWorkspace


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


def test_total_budget_enforced():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=10,
        slot_ratios={"task": 1.0},
        token_counter=len,
        raise_on_overflow=True,
    )
    allocator.register_strategy("task", NoopStrategy())

    with pytest.raises(BudgetExceededError, match="Context budget exceeded"):
        allocator.allocate({"task": "too long for budget"})


def test_nav_latest_wins_keeps_newest_block():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=1,
        slot_ratios={"trajectory": 1.0},
        token_counter=len,
    )
    allocator.register_strategy("trajectory", NavLatestWinsCompact())

    allocated, report = allocator.allocate({"trajectory": "old block\n\nnew block"})

    assert allocated["trajectory"] == "new block"
    assert report.compact_events[0]["strategy"] == "nav_latest_wins"


def test_parse_budget_ratios_rejects_removed_navigation_slot():
    with pytest.raises(ValueError, match="Unknown budget slot: navigation"):
        parse_budget_ratios("navigation:1.0")


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


def test_evidence_tiered_keeps_multiline_ledger_entries_intact():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=200,
        slot_ratios={"evidence": 1.0},
        token_counter=len,
    )
    allocator.register_strategy("evidence", EvidenceTieredCompact())
    content = "\n".join(
        [
            "- obs_0001 | claim: older",
            "  raw_output:",
            "  {",
            '    "target": "red car",',
            '    "status": "supported"',
            "  }",
            "- obs_0002 | claim: newer unrelated",
            "  raw_output:",
            "  {",
            '    "target": "blue cup"',
            "  }",
        ]
    )

    allocated, _report = allocator.allocate(
        {"evidence": content},
        ctx={"active_followup_target_query": "red car"},
    )

    red_block_start = allocated["evidence"].index("- obs_0001")
    red_block_end = allocated["evidence"].find("- obs_0002")
    red_block = allocated["evidence"][red_block_start:] if red_block_end < 0 else allocated["evidence"][red_block_start:red_block_end]
    assert "{\n" in red_block
    assert '"target": "red car"' in red_block
    assert "}\n" in red_block or red_block.rstrip().endswith("}")


def test_evidence_tiered_keeps_relevant_ledger_block_intact():
    allocator = ContextBudgetAllocator(
        total_budget_tokens=80,
        slot_ratios={"evidence": 1.0},
        token_counter=len,
    )
    allocator.register_strategy("evidence", EvidenceTieredCompact())
    content = "\n\n".join(
        [
            "\n".join(
                [
                    "- `obs_0001` | tool: `vision_read` | claim: A blue cup is on the table.",
                    "  raw_output: visual_caption=blue cup; detected_objects=[cup]",
                ]
            ),
            "\n".join(
                [
                    "- `obs_0002` | tool: `vision_read` | claim: A red car passes the camera.",
                    "  raw_output: visual_caption=red car; detected_objects=[red car]",
                ]
            ),
        ]
    )

    allocated, _report = allocator.allocate(
        {"evidence": content},
        ctx={"active_followup_target_query": "red car"},
    )

    assert "`obs_0002`" in allocated["evidence"]
    assert "raw_output: visual_caption=red car" in allocated["evidence"]
    assert "`obs_0001`" not in allocated["evidence"]


def test_recent_tool_outputs_compacts_and_deduplicates_raw_output_for_planner(tmp_path):
    workspace = EvidenceWorkspace.create(tmp_path, "planner_raw_output_compact")
    repeated = "same repeated caption " * 40
    workspace.write_observation(
        tool_name="vision_read",
        claim="A red car passes the camera.",
        confidence=0.82,
        raw_output={
            "visual_caption": repeated,
            "summary": repeated,
            "candidates": [
                {"segment_id": "seg_0001", "score": 0.9, "reason": repeated},
                {"segment_id": "seg_0001", "score": 0.9, "reason": repeated},
                {"segment_id": "seg_0002", "score": 0.3, "reason": "blue cup"},
            ],
        },
    )

    output = workspace.recent_tool_outputs(limit=1)[0]["raw_output"]

    assert output["visual_caption"].endswith("chars]")
    assert output["summary"] == "[duplicate of raw_output.visual_caption]"
    assert output["candidates"][0]["reason"] == "[duplicate of raw_output.visual_caption]"
    assert output["candidates"] == [
        output["candidates"][0],
        {"segment_id": "seg_0002", "score": 0.3, "reason": "blue cup"},
    ]


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
    ratios = parse_budget_ratios(
        "task:0.08,trajectory:0.07,hypothesis:0.12,evidence:0.28,scene_index:0.22,feedback:0.10,budget:0.05,tooling:0.08"
    )

    assert ratios == {
        "task": 0.08,
        "trajectory": 0.07,
        "hypothesis": 0.12,
        "evidence": 0.28,
        "scene_index": 0.22,
        "feedback": 0.10,
        "budget": 0.05,
        "tooling": 0.08,
    }


def test_parse_budget_ratios_accepts_scene_index_and_late_tooling_slots():
    ratios = parse_budget_ratios(
        "task:0.08,trajectory:0.07,hypothesis:0.12,evidence:0.28,scene_index:0.22,feedback:0.10,budget:0.05,tooling:0.08"
    )

    assert ratios == {
        "task": 0.08,
        "trajectory": 0.07,
        "hypothesis": 0.12,
        "evidence": 0.28,
        "scene_index": 0.22,
        "feedback": 0.10,
        "budget": 0.05,
        "tooling": 0.08,
    }


def test_parse_budget_ratios_rejects_bad_sum():
    try:
        parse_budget_ratios(
            "task:0.1,trajectory:0.1,hypothesis:0.1,evidence:0.1,scene_index:0.1,feedback:0.1,budget:0.1,tooling:0.1"
        )
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("expected ValueError")
