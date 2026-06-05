import json
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class StaticBackend:
    def __init__(self, text: str):
        self.text = text

    def generate(self, request: BackendRequest) -> BackendResponse:
        return BackendResponse(text=self.text)


def _scene_index() -> SceneIndex:
    return SceneIndex(
        video_path="/videos/demo.mp4",
        duration_sec=12.0,
        segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
    )


def _inspect_registry(counter: dict[str, int]) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="inspect_segment", description="Inspect a segment.")
    def inspect_segment(segment_id: str, question: str = "", **kwargs):
        counter["inspect_segment"] = counter.get("inspect_segment", 0) + 1
        return {
            "claim": f"inspected {segment_id}: {question}",
            "confidence": 0.9,
            "regions": [{"segment_id": segment_id, "start_sec": 0.0, "end_sec": 12.0}],
        }

    registry.register(inspect_segment)
    return registry


def test_gist_qa_blocks_inspect_segment(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = StaticBackend(
        json.dumps(
            {
                "status": "continue",
                "program": [
                    {"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "main idea"}}
                ],
            }
        )
    )
    workspace = EvidenceWorkspace.create(tmp_path, "route_block")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=_inspect_registry(counter),
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(question="What is the video mainly about?", video_path="/videos/demo.mp4")

    assert counter.get("inspect_segment", 0) == 0
    assert "route_violation" in (workspace.root / "trace.jsonl").read_text(encoding="utf-8")


def test_free_explore_allows_all(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = StaticBackend(
        json.dumps(
            {
                "status": "continue",
                "program": [
                    {"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "main idea"}}
                ],
            }
        )
    )
    workspace = EvidenceWorkspace.create(tmp_path, "route_free")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=_inspect_registry(counter),
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget.free_explore(max_rounds=1, max_tool_calls_per_round=1),
    )

    agent.run(question="What is the video mainly about?", video_path="/videos/demo.mp4")

    assert counter.get("inspect_segment", 0) == 1
