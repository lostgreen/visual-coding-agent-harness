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


class FakeTargetRegistry:
    def __init__(self, known_keys: set[str]):
        self.known_keys = set(known_keys)

    def has(self, key: str) -> bool:
        return key in self.known_keys


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
                "skill": "main_idea",
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
                    "skill": "main_idea",
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
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("main_idea"),
    )

    assert normalized[0]["tool"] == "vision_read"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"


def test_main_idea_allows_video_map_exploration(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="target_coverage", description="Build target coverage matrix.")
    def target_coverage(targets: list, top_k: int = 3):
        return {"claim": f"covered {targets}", "confidence": 0.4}

    registry.register(target_coverage)
    workspace = EvidenceWorkspace.create(tmp_path, "main_idea_video_ls_allowed")
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [{"tool": "target_coverage", "args": {"targets": ["main topic"], "top_k": 2}}],
        question="What is the video mainly about?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("main_idea"),
    )

    assert normalized[0]["tool"] == "target_coverage"
    assert "route_violation" not in (workspace.root / "trace.jsonl").read_text(encoding="utf-8")


def test_timeline_repairs_read_segment_to_detail_with_targets(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="read_segment_detail", description="Read detailed indexed segment packet.")
    def read_segment_detail(segment_id: str, targets: list | None = None):
        return {"claim": f"detail {segment_id}: {targets}", "confidence": 1.0}

    registry.register(read_segment_detail)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_read_segment_to_detail")
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )
    agent._exploration_target_entities = ("David", "Apollo and Daphne")

    normalized = agent._normalize_program(
        [{"tool": "read_segment", "args": {"segment_id": "seg_0001"}}],
        question="Which artwork appears first?\nA. David\nB. Apollo and Daphne",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "read_segment_detail"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"
    assert normalized[0]["args"]["targets"] == ["David", "Apollo and Daphne"]
    assert "tool_not_in_allowed_actions" not in (workspace.root / "trace.jsonl").read_text(encoding="utf-8")


def test_normalization_strips_unsupported_read_timeline_sorted_args(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="read_timeline_sorted", description="Read sorted timeline.")
    def read_timeline_sorted():
        return {"claim": "timeline", "confidence": 1.0}

    registry.register(read_timeline_sorted)
    workspace = EvidenceWorkspace.create(tmp_path, "strip_timeline_args")
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [{"tool": "read_timeline_sorted", "args": {"segment_id": "seg_0006"}}],
        question="Which event appears first?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized == [{"tool": "read_timeline_sorted", "args": {}}]
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "strip_unsupported_tool_args" in trace


def test_known_legacy_target_id_rewrites_to_target_refs(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="locate_targets_in_segment", description="Locate targets in one segment.")
    def locate_targets_in_segment(
        segment_id: str,
        targets: list | None = None,
        target_refs: list | None = None,
        normalized_target_keys: list | None = None,
    ):
        return {
            "claim": f"located {segment_id}: {targets} {target_refs} {normalized_target_keys}",
            "confidence": 0.4,
        }

    registry.register(locate_targets_in_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "target_refs_legacy_rewrite")
    workspace.target_registry = FakeTargetRegistry({"T1"})
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )
    notes = []

    normalized = agent._normalize_program(
        [{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "targets": ["T1"]}}],
        question="Where is T1?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        notes_out=notes,
    )

    assert normalized == [
        {
            "tool": "locate_targets_in_segment",
            "args": {"segment_id": "seg_0001", "target_refs": ["T1"], "normalized_target_keys": ["T1"]},
        }
    ]
    assert any(note.reason == "rewrite_legacy_targets_to_target_refs" for note in notes)


def test_unknown_target_ref_rejects_tool_call(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="locate_targets_in_segment", description="Locate targets in one segment.")
    def locate_targets_in_segment(segment_id: str, target_refs: list | None = None):
        return {"claim": f"located {segment_id}: {target_refs}", "confidence": 0.4}

    registry.register(locate_targets_in_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "target_refs_unknown_reject")
    workspace.target_registry = FakeTargetRegistry({"T1"})
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )
    notes = []

    normalized = agent._normalize_program(
        [{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "target_refs": ["T9"]}}],
        question="Where is T9?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        notes_out=notes,
    )

    assert normalized == []
    assert any(note.reason == "unknown_target_ref" for note in notes)
    assert "unknown_target_ref" in (workspace.root / "trace.jsonl").read_text(encoding="utf-8")


def test_unknown_legacy_target_id_rejects_entire_tool_call(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="locate_targets_in_segment", description="Locate targets in one segment.")
    def locate_targets_in_segment(segment_id: str, targets: list | None = None):
        return {"claim": f"located {segment_id}: {targets}", "confidence": 0.4}

    registry.register(locate_targets_in_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "target_refs_unknown_legacy_reject")
    workspace.target_registry = FakeTargetRegistry({"T1"})
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )
    notes = []

    normalized = agent._normalize_program(
        [{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "targets": ["T9"]}}],
        question="Where is T9?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        notes_out=notes,
    )

    assert normalized == []
    assert any(note.reason == "unknown_legacy_target_ref" for note in notes)
    assert "unknown_legacy_target_ref" in (workspace.root / "trace.jsonl").read_text(encoding="utf-8")


def test_free_text_target_ref_rejects_tool_call(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="locate_targets_in_segment", description="Locate targets in one segment.")
    def locate_targets_in_segment(segment_id: str, target_refs: list | None = None):
        return {"claim": f"located {segment_id}: {target_refs}", "confidence": 0.4}

    registry.register(locate_targets_in_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "target_refs_free_text_reject")
    workspace.target_registry = FakeTargetRegistry({"T1"})
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [
            {
                "tool": "locate_targets_in_segment",
                "args": {"segment_id": "seg_0001", "target_refs": ["Apollo and Daphne"]},
            }
        ],
        question="Where is Apollo and Daphne?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
    )

    assert normalized == []


def test_natural_language_targets_are_preserved(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="locate_targets_in_segment", description="Locate targets in one segment.")
    def locate_targets_in_segment(segment_id: str, targets: list | None = None):
        return {"claim": f"located {segment_id}: {targets}", "confidence": 0.4}

    registry.register(locate_targets_in_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "target_refs_free_text_targets")
    workspace.target_registry = FakeTargetRegistry({"T1"})
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [
            {
                "tool": "locate_targets_in_segment",
                "args": {"segment_id": "seg_0001", "targets": ["Apollo and Daphne"]},
            }
        ],
        question="Where is Apollo and Daphne?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
    )

    assert normalized[0]["args"] == {"segment_id": "seg_0001", "targets": ["Apollo and Daphne"]}


def test_mutex_fact_repairs_planner_inspect_segment_to_vision_read(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = SequenceBackend(
        [
            json.dumps(
                {
                    "status": "continue",
                    "skill": "mutex_fact_qa",
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


def test_timeline_ordering_rewrites_window_expansion_to_locator(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="locate_targets_in_segment", description="Locate targets in one segment.")
    def locate_targets_in_segment(segment_id: str, targets: list | None = None):
        return {
            "claim": f"located {segment_id}: {targets}",
            "confidence": 0.4,
            "regions": [{"segment_id": segment_id, "targets": targets or []}],
        }

    registry.register(locate_targets_in_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_expand_to_locate")
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )
    agent._exploration_target_entities = ("David", "Apollo and Daphne")

    normalized = agent._normalize_program(
        [{"tool": "expand_window", "args": {"segment_id": "seg_0001", "before_sec": 2.0, "after_sec": 3.0}}],
        question="Which order is shown?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "locate_targets_in_segment"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"
    assert normalized[0]["args"]["targets"] == ["David", "Apollo and Daphne"]
    assert "route_violation" not in (workspace.root / "trace.jsonl").read_text(encoding="utf-8")


def test_timeline_ordering_repairs_repeated_locator_to_verify_anchors(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="locate_targets_in_segment", description="Locate targets in one segment.")
    def locate_targets_in_segment(segment_id: str, targets: list | None = None):
        return {"claim": f"located {segment_id}: {targets}", "confidence": 0.4}

    @tool(name="verify_segment_anchors", description="Verify anchors.")
    def verify_segment_anchors(segment_id: str, anchors: list, targets: list | None = None, question: str = ""):
        return {
            "claim": f"verified {segment_id}",
            "confidence": 0.8,
            "regions": [{"segment_id": segment_id, "anchors": anchors}],
        }

    registry.register(locate_targets_in_segment)
    registry.register(verify_segment_anchors)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_repeated_locate_to_verify")
    verify_args = {
        "segment_id": "seg_0001",
        "anchors": [
            {
                "anchor_id": "anchor_0001",
                "segment_id": "seg_0001",
                "start_sec": 2.0,
                "end_sec": 8.0,
                "targets": ["David", "Apollo and Daphne"],
            }
        ],
        "targets": ["David", "Apollo and Daphne"],
    }
    workspace.write_observation(
        tool_name="locate_targets_in_segment",
        claim="locate_targets_in_segment(seg_0001) found anchors.",
        confidence=1.0,
        raw_output={"segment_id": "seg_0001", "verify_call_args": verify_args},
    )
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "targets": ["David"]}}],
        question="Which order is shown?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "verify_segment_anchors"
    assert normalized[0]["args"]["segment_id"] == verify_args["segment_id"]
    assert normalized[0]["args"]["anchors"] == verify_args["anchors"]
    assert normalized[0]["args"]["targets"] == verify_args["targets"]
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "repair_repeated_locator_to_verify_segment_anchors" in trace
    assert "route_violation" not in trace


def test_verify_segment_anchors_does_not_avoid_repeated_anchor_segment(tmp_path: Path):
    registry = ToolRegistry()

    @tool(name="verify_segment_anchors", description="Verify anchors.")
    def verify_segment_anchors(
        video_path: str,
        segment_id: str,
        anchors: list,
        targets: list | None = None,
        question: str = "",
        start_sec: float = 0.0,
        end_sec: float = 0.0,
    ):
        return {
            "claim": f"verified {segment_id}",
            "confidence": 0.8,
            "regions": [{"segment_id": segment_id, "anchors": anchors}],
        }

    registry.register(verify_segment_anchors)
    workspace = EvidenceWorkspace.create(tmp_path, "verify_anchor_no_segment_swap")
    scene_index = SceneIndex(
        video_path="/videos/demo.mp4",
        duration_sec=1500.0,
        segments=[
            VideoSegment(segment_id="seg_0002", start_sec=300.0, end_sec=600.0),
            VideoSegment(segment_id="seg_0005", start_sec=1200.0, end_sec=1500.0),
        ],
    )
    agent = IterativeVisualAgent(
        backend=StaticBackend("{}"),
        registry=registry,
        workspace=workspace,
        scene_index=scene_index,
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )
    anchors = [
        {
            "anchor_id": "anchor_0001",
            "segment_id": "seg_0002",
            "start_sec": 492.0,
            "end_sec": 544.0,
            "targets": ["The rape of Persephone", "Apollo and Daphne"],
        }
    ]

    normalized = agent._normalize_program(
        [{"tool": "verify_segment_anchors", "args": {"segment_id": "seg_0002", "anchors": anchors}}],
        question="Which order is shown?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids={"seg_0002"},
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "verify_segment_anchors"
    assert normalized[0]["args"]["segment_id"] == "seg_0002"
    assert normalized[0]["args"]["anchors"] == anchors
    assert normalized[0]["args"]["video_path"] == "/videos/demo.mp4"
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "avoid_repeated_segment" not in trace
    assert "seg_0005" not in trace


def test_timeline_skill_upgrades_empty_read_segment_before_deny_list(tmp_path: Path):
    backend = StaticBackend("{}")
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption one segment.")
    def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", **kwargs):
        return {"claim": f"caption {segment_id}", "confidence": 0.75}

    registry.register(caption_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_read_upgrade_before_deny")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=24.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
        ),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [{"tool": "read_segment", "args": {"segment_id": "seg_0001"}}],
        question="Which order is shown?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "caption_segment"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"
    assert "tool_not_in_allowed_actions" not in (workspace.root / "trace.jsonl").read_text(encoding="utf-8")


def test_skill_aware_empty_program_fallback_uses_allowed_visual_tool(tmp_path: Path):
    backend = StaticBackend("{}")
    registry = ToolRegistry()

    @tool(name="inspect_segment", description="Registered but not allowed for timeline skill.")
    def inspect_segment(**kwargs):
        return {"claim": "should not be selected", "confidence": 0.1}

    @tool(name="caption_segment", description="Allowed timeline fallback.")
    def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", **kwargs):
        return {"claim": f"caption {segment_id}", "confidence": 0.75}

    registry.register(inspect_segment)
    registry.register(caption_segment)
    workspace = EvidenceWorkspace.create(tmp_path, "skill_aware_fallback")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [],
        question="Which order is shown?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "caption_segment"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"


def test_same_round_duplicate_global_gist_is_dropped(tmp_path: Path):
    backend = StaticBackend("{}")
    registry = ToolRegistry()

    @tool(name="global_gist", description="Read sparse global evidence.")
    def global_gist(video_path: str, question: str, duration_sec: float):
        return {"claim": "global", "confidence": 0.7}

    registry.register(global_gist)
    workspace = EvidenceWorkspace.create(tmp_path, "same_round_one_shot")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=2, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [
            {"tool": "global_gist", "args": {"question": "main idea", "duration_sec": 12.0}},
            {"tool": "global_gist", "args": {"question": "main idea again", "duration_sec": 12.0}},
        ],
        question="What is the video mainly about?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("main_idea"),
    )

    assert [step["tool"] for step in normalized] == ["global_gist"]


def test_skill_name_tool_step_is_dropped_without_route_violation(tmp_path: Path):
    backend = StaticBackend("{}")
    workspace = EvidenceWorkspace.create(tmp_path, "skill_name_tool_alias")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=ToolRegistry(),
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    normalized = agent._normalize_program(
        [{"tool": "main_idea@v1", "args": {}}],
        question="What is the video mainly about?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("main_idea"),
    )

    assert normalized == []
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "skill_name_as_tool" in trace
    assert "route_violation" not in trace


def test_skill_name_tool_step_is_dropped(tmp_path: Path):
    backend = StaticBackend("{}")
    workspace = EvidenceWorkspace.create(tmp_path, "skill_name_tool")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=ToolRegistry(),
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False),
    )

    normalized = agent._normalize_program(
        [{"tool": "main_idea@v1", "args": {}}],
        question="What is the video mainly about?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("main_idea"),
    )

    assert normalized == []
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "skill_name_as_tool" in trace
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


def test_missing_planner_skill_does_not_enable_route_policy(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = StaticBackend(
        json.dumps(
            {
                "status": "continue",
                "program": [
                    {
                        "tool": "inspect_segment",
                        "args": {"segment_id": "seg_0001", "question": "Inspect visible order."},
                    }
                ],
            }
        )
    )
    registry = _inspect_registry(counter)
    workspace = EvidenceWorkspace.create(tmp_path, "missing_planner_skill_policy")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(question="Which event happened first?\nA. red then blue\nB. blue then red", video_path="/videos/demo.mp4")

    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert counter.get("inspect_segment", 0) == 1
    assert "planner_skill_missing" in trace
    assert "route_violation" not in trace


def test_invalid_planner_skill_does_not_fallback_to_route_classifier(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = StaticBackend(
        json.dumps(
            {
                "status": "continue",
                "skill": "not_a_real_skill",
                "program": [
                    {
                        "tool": "inspect_segment",
                        "args": {"segment_id": "seg_0001", "question": "Inspect visible order."},
                    }
                ],
            }
        )
    )
    registry = _inspect_registry(counter)
    workspace = EvidenceWorkspace.create(tmp_path, "invalid_planner_skill_policy")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=_scene_index(),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(question="Which event happened first?\nA. red then blue\nB. blue then red", video_path="/videos/demo.mp4")

    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert counter.get("inspect_segment", 0) == 1
    assert "planner_skill_invalid" in trace
    assert "planner_skill_selection" not in trace
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
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "caption_segment"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"
    assert normalized[0]["args"]["question"] == "Find ordered entities? Do not choose an option. timeline_ordering"
    assert "segment_ids" not in normalized[0]["args"]


def test_timeline_skill_repairs_caption_segment_with_segment_ids_argument(tmp_path: Path):
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
    workspace = EvidenceWorkspace.create(tmp_path, "timeline_caption_segment_ids_repair")
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
                "tool": "caption_segment",
                "args": {
                    "segment_ids": ["seg_0001", "seg_0002"],
                    "question": "Find ordered entities.",
                },
            }
        ],
        question="Which order is shown?",
        video_path="/videos/demo.mp4",
        inspected_segment_ids=set(),
        final_round_reserved=False,
        planner_skill=builtin_skill_registry().get("timeline_ordering"),
    )

    assert normalized[0]["tool"] == "caption_segment"
    assert normalized[0]["args"]["segment_id"] == "seg_0001"
    assert "segment_ids" not in normalized[0]["args"]


def test_skill_allowed_actions_still_apply_under_round_only_budget(tmp_path: Path):
    counter: dict[str, int] = {}
    backend = StaticBackend(
        json.dumps(
            {
                "status": "continue",
                "skill": "main_idea@v1",
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
        budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=1, reserve_final_round=False),
    )

    agent.run(question="What is the video mainly about?", video_path="/videos/demo.mp4")

    assert counter.get("inspect_segment", 0) == 0
