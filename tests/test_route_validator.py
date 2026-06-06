import json
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.agents.skills.specs import builtin_skill_registry
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class StaticBackend:
    def __init__(self, text: str):
        self.text = text

    def generate(self, request: BackendRequest) -> BackendResponse:
        return BackendResponse(text=self.text)


class SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if not self.responses:
            return BackendResponse(text='{"answer": "need_more_evidence", "citations": []}')
        return BackendResponse(text=self.responses.pop(0))


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


def test_main_idea_repairs_only_first_planner_vision_read_to_global_gist(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = SequenceBackend(
        [
            '{"answer": "need_more_evidence", "missing_evidence": ["global evidence"], "citations": []}',
            json.dumps(
                {
                    "status": "continue",
                    "program": [
                        {
                            "tool": "vision_read",
                            "args": {"segment_id": "seg_0001", "ask_for": "main idea"},
                        }
                    ],
                }
            ),
            '{"answer": "need_more_evidence", "missing_evidence": ["more evidence"], "citations": ["obs_0002"]}',
        ]
    )
    registry = ToolRegistry()

    @tool(name="global_gist", description="Read sparse global evidence.")
    def global_gist(video_path: str, question: str, duration_sec: float, seed: int = 0):
        counter["global_gist"] = counter.get("global_gist", 0) + 1
        return {
            "claim": "Global evidence is ambiguous.",
            "confidence": 0.55,
            "regions": [{"start_sec": 0.0, "end_sec": duration_sec, "seed": seed}],
            "grounding_quality": "global_sparse",
        }

    @tool(name="vision_read", description="Wrong tool for main idea.")
    def vision_read(**kwargs):
        counter["vision_read"] = counter.get("vision_read", 0) + 1
        return {"claim": "should not run", "confidence": 0.1}

    registry.register(global_gist)
    registry.register(vision_read)
    workspace = EvidenceWorkspace.create(tmp_path, "main_idea_repair")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(question="What is the video mainly about?", video_path="/videos/demo.mp4")

    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert counter.get("global_gist", 0) == 1
    assert counter.get("vision_read", 0) == 1
    assert "global_gist_topic_seeded" in trace
    assert "route_violation" not in trace


def test_main_idea_allows_local_read_after_global_floor(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="global_gist", description="Read sparse global evidence.")
    def global_gist(video_path: str, question: str, duration_sec: float):
        return {"claim": "global", "confidence": 0.7}

    @tool(name="vision_read", description="Read localized visual evidence.")
    def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, **kwargs):
        return {"claim": "local", "confidence": 0.8}

    registry.register(global_gist)
    registry.register(vision_read)
    workspace = EvidenceWorkspace.create(tmp_path, "main_idea_local_after_floor")
    for index in range(2):
        workspace.write_observation(
            tool_name="global_gist",
            claim=f"global floor {index}",
            confidence=0.7,
            regions=[{"start_sec": 0.0, "end_sec": 12.0}],
        )
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=2, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "main idea"}}],
        question="What is the video mainly about?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        tool_class_counts={"cheap": 0, "expensive": 0, "verifier": 0},
        final_round_reserved=False,
    )

    assert normalized[0]["tool"] == "vision_read"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"


def test_mutex_fact_repairs_planner_inspect_segment_to_vision_read(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = SequenceBackend(
        [
            json.dumps(
                {
                    "status": "continue",
                    "program": [
                        {
                            "tool": "inspect_segment",
                            "args": {
                                "segment_id": "seg_0001",
                                "question": "Did Francisco Goya have humble origins?",
                            },
                        }
                    ],
                }
            ),
            '{"answer": "need_more_evidence", "missing_evidence": ["more evidence"], "citations": ["obs_0001"]}',
        ]
    )
    registry = ToolRegistry()

    @tool(name="inspect_segment", description="Wrong tool for mutex fact QA.")
    def inspect_segment(**kwargs):
        counter["inspect_segment"] = counter.get("inspect_segment", 0) + 1
        return {"claim": "should not run", "confidence": 0.1}

    @tool(name="vision_read", description="Read localized mutex fact.")
    def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, **kwargs):
        counter["vision_read"] = counter.get("vision_read", 0) + 1
        return {
            "claim": f"Visual read for {ask_for}",
            "confidence": 0.6,
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            "grounding_quality": "visually_confirmed",
        }

    registry.register(inspect_segment)
    registry.register(vision_read)
    workspace = EvidenceWorkspace.create(tmp_path, "mutex_repair")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(
        question=(
            "What was Francisco Goya's background and social status at birth?\n"
            "Options:\n"
            "A. noble family\n"
            "B. humble origins\n"
            "C. neither\n"
            "D. upper class"
        ),
        video_path="/videos/demo.mp4",
    )

    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert counter.get("inspect_segment", 0) == 0
    assert counter.get("vision_read", 0) == 1
    assert "route_tool_repaired" in trace
    assert "route_violation" not in trace


def test_timeline_ordering_allows_verify_ledger_answer(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = StaticBackend(
        json.dumps(
            {
                "status": "continue",
                "program": [
                    {
                        "tool": "verify_ledger_answer",
                        "args": {
                            "answer": "A",
                            "ledger_text": "partial timeline",
                            "required_citations": [],
                        },
                    }
                ],
            }
        )
    )
    registry = ToolRegistry()

    @tool(name="verify_ledger_answer", description="Verifier tool.")
    def verify_ledger_answer(answer: str, ledger_text: str = "", required_citations=None, **kwargs):
        counter["verify_ledger_answer"] = counter.get("verify_ledger_answer", 0) + 1
        return {"claim": f"verified {answer}", "confidence": 0.5}

    registry.register(verify_ledger_answer)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_verify_allowed")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(
        question="Which order happens first, the door opening before the light turning on?",
        video_path="/videos/demo.mp4",
    )

    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert counter.get("verify_ledger_answer", 0) == 1
    assert "route_violation" not in trace


def test_planner_selected_skill_overrides_fallback_route_for_tool_policy(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = StaticBackend(
        json.dumps(
            {
                "status": "continue",
                "skill": "timeline_ordering",
                "program": [
                    {
                        "tool": "caption_segment",
                        "args": {"segment_id": "seg_0001", "question": "Find life-stage evidence."},
                    }
                ],
            }
        )
    )
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption one segment.")
    def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", **kwargs):
        counter["caption_segment"] = counter.get("caption_segment", 0) + 1
        return {
            "claim": f"{segment_id} caption for {question}",
            "confidence": 0.75,
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
        }

    registry.register(caption_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "planner_skill_override")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(question="How was his life journey according to the video?", video_path="/videos/demo.mp4")

    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert counter.get("caption_segment", 0) == 1
    assert "planner_skill_selection" in trace
    assert "route_violation" not in trace


def test_timeline_skill_repairs_batch_caption_segments_to_single_caption_segment(tmp_path: Path):
    backend = StaticBackend("{}")
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption one segment.")
    def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", **kwargs):
        return {
            "claim": f"{segment_id} caption for {question}",
            "confidence": 0.75,
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
        }

    registry.register(caption_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_batch_caption_repair")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=24.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0),
                VideoSegment(segment_id="seg_0002", start_sec=12.0, end_sec=24.0),
            ],
        ),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [
            {
                "tool": "caption_segments",
                "args": {
                    "segment_ids": ["seg_0001", "seg_0002"],
                    "question": "Find ordered entities.",
                },
            }
        ],
        question="Which order is shown?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        tool_class_counts={"cheap": 0, "expensive": 0, "verifier": 0},
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "caption_segment"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"
    assert normalized[0]["args"]["question"] == "Find ordered entities."


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
