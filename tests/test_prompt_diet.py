from __future__ import annotations

from unittest.mock import patch

from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.agents.prompt_stack import build_replanning_prompt
from visual_coding_agent_harness.video_index import fixed_window_scene_index


def test_catalog_compresses_when_active() -> None:
    prompt, _report = build_replanning_prompt(
        question="What is visible?",
        scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0),
        ledger_text="# Compact Evidence Context\n(none)",
        round_number=1,
        budget=AgentBudget(max_rounds=2),
        allocator=default_context_budget_allocator(total_budget_tokens=12000),
        active_skill="visual_timeline_qa@v1",
    )

    catalog = prompt.split("# Skill Catalog", 1)[1].split("## Trajectory", 1)[0]
    skill_lines = [line for line in catalog.splitlines() if line.startswith("- ") and "@v" in line]

    assert len(catalog.splitlines()) <= 25
    assert "- visual_timeline_qa@v1" in catalog
    assert "  when_to_use: Use when the question asks about before" in catalog
    assert any(line.startswith("- main_idea@v1: ") for line in skill_lines)
    assert "main_idea@v1\n  description:" not in catalog


def test_evidence_pointer_replaces_full_recent_tool_text() -> None:
    long_payload = {"transcript": "alpha " * 300, "segment_id": "seg_0001"}
    with patch.dict("os.environ", {"HARNESS_LEGACY_PROMPT_EVIDENCE_SNAPSHOT": "1"}):
        prompt, _report = build_replanning_prompt(
            question="What is visible?",
            scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=10.0, window_sec=5.0),
            ledger_text="# Compact Evidence Context\n- `obs_0001` | ev: `ev_1` | claim: alpha",
            round_number=2,
            budget=AgentBudget(max_rounds=3),
            allocator=default_context_budget_allocator(total_budget_tokens=12000),
            recent_tool_outputs=[
                {
                    "observation_id": "obs_0001",
                    "tool": "read_segment_detail",
                    "claim": "alpha",
                    "raw_output": long_payload,
                    "in_evidence_table": True,
                    "evidence_id": "ev_1",
                    "segment_id": "seg_0001",
                    "modality": "narrated_fact",
                    "verdict": "supported",
                }
            ],
        )

    assert "[obs:obs_0001] segment=seg_0001 modality=narrated_fact verdict=supported -> see workspace/evidence_table.jsonl" in prompt
    assert "alpha alpha alpha alpha alpha alpha alpha alpha" not in prompt
