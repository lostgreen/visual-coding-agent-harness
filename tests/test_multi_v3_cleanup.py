from __future__ import annotations

import importlib.util

import pytest


def test_videomme_runner_is_multi_v3_only() -> None:
    from runs import eval_runner

    assert eval_runner.parse_strategies(None) == ("multi_v3",)
    assert eval_runner.parse_strategies(("multi_v3",)) == ("multi_v3",)
    for legacy_strategy in ("workspace_v2", "multi_agent_v0"):
        with pytest.raises(ValueError, match=f"Unknown strategy: {legacy_strategy}"):
            eval_runner.parse_strategies((legacy_strategy,))


def test_legacy_agent_tool_and_ledger_modules_are_removed() -> None:
    removed_modules = (
        "visual_coding_agent_harness.agents.workspace_agent",
        "visual_coding_agent_harness.agents.multi",
        "visual_coding_agent_harness.tools.workspace_v2",
        "visual_coding_agent_harness.tools.navigation",
        "visual_coding_agent_harness.tools.inspector",
        "visual_coding_agent_harness.tools.enrichment",
        "visual_coding_agent_harness.tools.timeline",
        "visual_coding_agent_harness.tools.global_view",
        "visual_coding_agent_harness.tools.exploration",
        "visual_coding_agent_harness.tools.verification",
        "visual_coding_agent_harness.workspace.search_ledger",
        "visual_coding_agent_harness.workspace.views",
        "visual_coding_agent_harness.evidence.need",
        "visual_coding_agent_harness.evidence.ledger",
        "visual_coding_agent_harness.evidence.order_extraction",
        "visual_coding_agent_harness.evidence.order_hypotheses",
        "visual_coding_agent_harness.evidence.predicates",
    )

    assert {name: importlib.util.find_spec(name) for name in removed_modules} == {
        name: None for name in removed_modules
    }
