from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.agents.prompt_stack import build_replanning_prompt
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.registry import ToolRegistry
from visual_coding_agent_harness.video_index import fixed_window_scene_index
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_final_blocked_when_slot_empty(tmp_path: Path):
    class FinalBackend(VisionLanguageBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            return BackendResponse(
                text='{"status": "final", "answer": "The door opens.", "citations": ["obs_0001"], "confidence": 0.8}'
            )

    workspace = EvidenceWorkspace.create(tmp_path, "hypothesis_empty")
    workspace.write_observation(
        tool_name="vision_read",
        claim="The door opens.",
        confidence=0.88,
        raw_output={"grounding_quality": "visually_confirmed"},
    )
    workspace.write_hypothesis(
        {
            "slot_door": {
                "status": "empty",
                "evidence_obs_id": "",
            }
        }
    )
    agent = IterativeVisualAgent(
        backend=FinalBackend(),
        registry=ToolRegistry(),
        workspace=workspace,
        scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=10.0),
        budget=AgentBudget(max_rounds=1),
    )

    result = agent.run(question="What happens in the video?", video_path="/videos/demo.mp4")

    assert result.status == "final"
    assert result.final_decision_owner == "model"
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "hypothesis_slots_unsatisfied" in trace


def test_hypothesis_slot_in_replanning_prompt():
    prompt, _report = build_replanning_prompt(
        question="What happens in the video?",
        scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=10.0),
        ledger_text="obs_0001: The door opens.",
        round_number=2,
        budget=AgentBudget(max_rounds=3),
        allocator=default_context_budget_allocator(),
        hypothesis_text="# Hypothesis\n\n- slot_door | status: partial | evidence_obs_id: obs_0001\n",
    )

    assert "## Hypothesis" in prompt
    assert "slot_door" in prompt
    assert "status: partial" in prompt
