import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.iterative_smoke import run_iterative_smoke
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.tools.exploration import build_video_exploration_registry
from visual_coding_agent_harness.tools.segments import build_segment_vlm_registry
from visual_coding_agent_harness.video_index import VideoSegment, SceneIndex, fixed_window_scene_index
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class ScriptedPlannerBackend(VisionLanguageBackend):
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task != "replan":
            return BackendResponse(text="unexpected tool backend call")
        if not self.responses:
            return BackendResponse(text='{"status": "final", "answer": "No more scripted responses.", "citations": []}')
        return BackendResponse(text=self.responses.pop(0))


def build_segment_test_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption one indexed video segment.")
    def caption_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        nframes: int = 8,
    ):
        return {
            "claim": f"{segment_id} from {start_sec:.1f}s to {end_sec:.1f}s shows aircraft history.",
            "confidence": 0.72,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "nframes": nframes}],
        }

    registry.register(caption_segment)

    @tool(name="qa_segment", description="Answer a question about one indexed video segment.")
    def qa_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        nframes: int = 8,
    ):
        return {
            "claim": f"{segment_id} from {start_sec:.1f}s to {end_sec:.1f}s answers: aircraft history.",
            "confidence": 0.78,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "question": question, "nframes": nframes}],
        }

    registry.register(qa_segment)

    @tool(name="inspect_segment", description="Inspect one indexed video segment through a subagent boundary.")
    def inspect_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        candidate_options=None,
        nframes: int = 16,
    ):
        return {
            "claim": f"{segment_id} inspector answers: aircraft history.",
            "confidence": 0.8,
            "input_artifacts": [video_path],
            "regions": [
                {
                    "segment_id": segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "question": question,
                    "candidate_options": list(candidate_options or []),
                    "nframes": nframes,
                }
            ],
        }

    registry.register(inspect_segment)
    return registry


def build_global_route_test_registry() -> ToolRegistry:
    registry = build_segment_test_registry()

    @tool(name="global_gist", description="Inspect a sparse whole-video view.")
    def global_gist(
        video_path: str,
        question: str,
        duration_sec: float,
        nframes: int = 64,
        max_pixels: int = 151200,
    ):
        return {
            "claim": "Supported option: D. The sparse whole-video view shows an aviation documentary.",
            "confidence": 0.76,
            "input_artifacts": [video_path],
            "regions": [
                {
                    "start_sec": 0.0,
                    "end_sec": duration_sec,
                    "nframes": nframes,
                    "max_pixels": max_pixels,
                }
            ],
            "limitations": "Sparse full-video sampling.",
            "supported_option": "D",
            "grounding_quality": "global_sparse",
        }

    registry.register(global_gist)
    return registry


class IterativeAgentTest(unittest.TestCase):
    def test_agent_budget_defaults_to_answer_capable_loop(self):
        budget = AgentBudget()

        self.assertEqual(budget.max_rounds, 8)
        self.assertEqual(budget.max_tool_calls_per_round, 2)
        self.assertTrue(budget.reserve_final_round)
        self.assertGreaterEqual(budget.cheap_tool_budget, budget.max_rounds)
        self.assertGreaterEqual(budget.expensive_tool_budget, 4)
        self.assertGreaterEqual(budget.verifier_tool_budget, 1)
        self.assertEqual(budget.answer_probe_rounds_before_final, 0)

    def test_free_exploration_budget_disables_policy_budgets_but_keeps_safety_caps(self):
        budget = AgentBudget.free_explore(max_rounds=24, max_tool_calls_per_round=4)

        self.assertTrue(budget.free_exploration)
        self.assertEqual(budget.max_rounds, 24)
        self.assertEqual(budget.max_tool_calls_per_round, 4)
        self.assertFalse(budget.reserve_final_round)
        self.assertEqual(budget.cheap_tool_budget, 0)
        self.assertEqual(budget.expensive_tool_budget, 0)
        self.assertEqual(budget.verifier_tool_budget, 0)

    def test_fixed_window_scene_index_creates_addressable_segments(self):
        index = fixed_window_scene_index(video_path="demo.mp4", duration_sec=65.0, window_sec=30.0)

        self.assertEqual([segment.segment_id for segment in index.segments], ["seg_0001", "seg_0002", "seg_0003"])
        self.assertEqual(index.segments[-1].start_sec, 60.0)
        self.assertEqual(index.segments[-1].end_sec, 65.0)

    def test_scene_index_summarizes_fixed_segments(self):
        index = SceneIndex(
            video_path="demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="middle"),
            ],
        )

        summary = index.summary()

        self.assertIn("seg_0001 [0.0-30.0s] opening", summary)
        self.assertIn("seg_0002 [30.0-60.0s] middle", summary)

    def test_iterative_agent_replans_after_reading_ledger(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "Need inspect likely segment.", '
                    '"program": [{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "What is discussed?"}, "assign": "middle"}]}'
                ),
                (
                    '{"status": "final", "answer": "The video discusses aircraft history.", '
                    '"citations": ["obs_0001"], "confidence": 0.78}'
                ),
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=120.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0),
                VideoSegment(segment_id="seg_0002", start_sec=60.0, end_sec=120.0, low_fps_caption="aircraft museum"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="iterative")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="What is the video mainly about?", video_path="/videos/demo.mp4")

            self.assertEqual(result.answer, "The video discusses aircraft history.")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual([round_result.status for round_result in result.rounds], ["continue", "final"])
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001"])
            self.assertEqual(len(backend.requests), 2)
            self.assertIn("Scene index", backend.requests[0].prompt)
            self.assertIn("seg_0002 [60.0-120.0s] aircraft museum", backend.requests[0].prompt)
            self.assertIn("Evidence ledger", backend.requests[1].prompt)
            self.assertIn("aircraft history", backend.requests[1].prompt)
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("seg_0002", ledger)
            self.assertIn("aircraft history", ledger)

    def test_iterative_agent_planner_is_text_only_by_default(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=20.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=20.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="text_only")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            planner_requests = [request for request in backend.requests if request.task == "replan"]
            self.assertTrue(planner_requests)
            self.assertTrue(all(request.media_path is None for request in planner_requests))
            self.assertTrue(all(request.media_type is None for request in planner_requests))
            self.assertIn("Planner input mode: text-only", planner_requests[0].prompt)

    def test_iterative_agent_prompt_exposes_navigation_tools(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="navigation_prompt")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            prompt = backend.requests[0].prompt
            self.assertIn("video_ls(query", prompt)
            self.assertIn("search_segments(query", prompt)
            self.assertIn("read_segment(segment_id", prompt)
            self.assertIn("expand_window(segment_id", prompt)
            self.assertIn("zoom(segment_id", prompt)
            self.assertIn("inspect_segment(video_path", prompt)
            self.assertIn("caption_segments(segment_ids", prompt)
            self.assertIn("ingest_segment_metadata(segment_id", prompt)
            self.assertIn("verify_ledger_answer(answer", prompt)
            self.assertIn("summarize_ledger_evidence", prompt)
            self.assertIn("vision_read(video_path", prompt)
            self.assertIn("max_pixels", prompt)
            self.assertIn("fps", prompt)
            self.assertIn("delegate localized visual reading to vision_read or inspect_segment", prompt)
            self.assertIn("Do not spend every round on navigation-only tools", prompt)
            self.assertIn("Multiple-choice answers must use vision_read or inspect_segment", prompt)
            self.assertIn("non-navigation visual observation", prompt)
            self.assertIn("caption_segments is offline VideoMap cache building", prompt)

    def test_iterative_agent_prompt_includes_task_type_playbook(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_playbook")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(
                question=(
                    "Which option is visible?\n"
                    "A. aircraft museum\n"
                    "B. submarine\n"
                    "C. mountain road"
                ),
                video_path="/videos/demo.mp4",
            )

            prompt = backend.requests[0].prompt
            self.assertIn("Task playbook: multiple_choice", prompt)
            self.assertIn("candidate_options", prompt)
            self.assertIn("verify option consistency", prompt)

    def test_gist_global_mcq_routes_through_global_gist_before_planning(self):
        backend = ScriptedPlannerBackend([])
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1896.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="global_route")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_global_route_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "A. cooking\n"
                    "B. a local airport scene\n"
                    "C. a city walk\n"
                    "D. an aviation documentary"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D. an aviation documentary")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual(backend.requests, [])
            self.assertEqual(result.rounds[0].program[0]["tool"], "global_gist")

    def test_iterative_agent_prompt_tells_planner_to_use_option_letters_in_json(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="option_letter_prompt")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(
                question='Which artwork is first?\nA. "David"\nB. "Apollo and Daphne"',
                video_path="/videos/demo.mp4",
            )

            prompt = backend.requests[0].prompt
            self.assertIn(
                'candidate_options in JSON should be option letters only, for example ["A", "B", "C", "D"]',
                prompt,
            )
            self.assertIn("Do not copy quoted option text into JSON string values", prompt)

    def test_iterative_agent_limits_tool_calls_per_round(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect 1"}, "assign": "s1"},'
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "Inspect 2"}, "assign": "s2"},'
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0003", "question": "Inspect 3"}, "assign": "s3"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=90.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="limit_tools")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_tool_calls_per_round=1),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(len(result.rounds[0].program), 1)
            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0001")
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001"])

    def test_iterative_agent_gates_expensive_tools_by_budget(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect 1"}, "assign": "s1"},'
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0002", "question": "Inspect 2"}, "assign": "s2"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="expensive_budget")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_tool_calls_per_round=2, expensive_tool_budget=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual([step["tool"] for step in result.rounds[0].program], ["inspect_segment"])
            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0001")
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("tool_budget_exhausted", trace)

    def test_iterative_agent_avoids_repeated_segments_with_fallback(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect again"}, "assign": "repeat"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001", "obs_0002"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="dedupe")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0001")
            self.assertEqual(result.rounds[1].program[0]["args"]["segment_id"], "seg_0002")
            self.assertIn("Already inspected segments: seg_0001", backend.requests[1].prompt)

    def test_iterative_agent_can_video_ls_then_refine_candidate_segment(self):
        class NavigationFirstBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "rationale": "Need a map before inspecting pixels.", '
                                '"program": [{"tool": "video_ls", "args": {"query": "overall description aircraft", "max_segments": 3}, "assign": "map"}]}'
                            )
                        )
                    if self.replan_calls == 2:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "rationale": "The map points to seg_0002.", '
                                '"program": [{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "Describe the key event."}, "assign": "detail"}]}'
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "final", "answer": "The video is mainly about aircraft history.", '
                            '"citations": ["obs_0001", "obs_0002"], "confidence": 0.8}'
                        )
                    )
                if request.task == "caption_segment":
                    return BackendResponse(text="The segment shows aircraft history exhibits.")
                return BackendResponse(text="unexpected")

        backend = NavigationFirstBackend()
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening greetings"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="aircraft museum history"),
                VideoSegment(segment_id="seg_0003", start_sec=60.0, end_sec=90.0, low_fps_caption="closing credits"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="video_ls_loop")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3),
            )

            result = agent.run(question="Describe the video.", video_path="/videos/demo.mp4")

            self.assertEqual([round_result.program[0]["tool"] for round_result in result.rounds[:2]], ["video_ls", "caption_segment"])
            self.assertEqual(result.answer, "The video is mainly about aircraft history.")
            self.assertEqual([request.task for request in backend.requests], ["replan", "replan", "caption_segment", "replan"])
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("Candidate segments", ledger)
            self.assertIn("seg_0002", ledger)
            self.assertIn("aircraft history exhibits", ledger)

    def test_iterative_agent_does_not_inject_media_args_into_navigation_segment_tools(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "read_segment", "args": {"segment_id": "seg_0002"}, "assign": "metadata"}'
                    "]}"
                ),
                '{"status": "final", "answer": "metadata read", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="aircraft museum"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="read_segment_args")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Describe the video.", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[0].program[0]["tool"], "read_segment")
            self.assertEqual(result.rounds[0].program[0]["args"], {"segment_id": "seg_0002"})
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("aircraft museum", ledger)

    def test_iterative_agent_upgrades_empty_read_segment_to_caption_segment(self):
        class ReadUpgradeBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "read_segment", "args": {"segment_id": "seg_0002"}, "assign": "metadata"}'
                                "]}"
                            )
                        )
                    return BackendResponse(text='{"status": "final", "answer": "captioned", "citations": ["obs_0001"]}')
                if request.task == "caption_segment":
                    return BackendResponse(text="The segment visually shows aircraft exhibits.")
                return BackendResponse(text="unexpected")

        backend = ReadUpgradeBackend()
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="upgrade_empty_read")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Describe the video.", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[0].program[0]["tool"], "caption_segment")
            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0002")
            self.assertEqual(backend.requests[1].task, "caption_segment")
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("aircraft exhibits", ledger)

    def test_iterative_agent_resolves_segment_id_into_tool_arguments(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=40.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=5.0, end_sec=17.5)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="resolve")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Inspect the clip", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["video_path"], "/videos/demo.mp4")
            self.assertEqual(tool_args["start_sec"], 5.0)
            self.assertEqual(tool_args["end_sec"], 17.5)
            self.assertEqual(tool_args["nframes"], 128)

    def test_iterative_agent_persists_planner_prompt_and_response_artifacts(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "inspect once", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=40.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=5.0, end_sec=17.5)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_io")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="Inspect the clip", video_path="/videos/demo.mp4")

            prompt_path = workspace.root / "artifacts" / "planner_io" / "round_0001_prompt.txt"
            response_path = workspace.root / "artifacts" / "planner_io" / "round_0001_response.txt"
            self.assertTrue(prompt_path.exists())
            self.assertTrue(response_path.exists())
            self.assertIn("Question: Inspect the clip", prompt_path.read_text(encoding="utf-8"))
            self.assertIn('"rationale": "inspect once"', response_path.read_text(encoding="utf-8"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"planner_io"', trace)
            self.assertIn("round_0001_prompt.txt", trace)
            self.assertIn("round_0001_response.txt", trace)

    def test_iterative_agent_allows_zoom_child_segments_with_explicit_window(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "qa_segment", "args": {"segment_id": "seg_0002_z01", "start_sec": 30.0, "end_sec": 45.0, "question": "Which option is supported?"}, "assign": "zoomed_qa"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="dynamic_zoom_child")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Which option is supported?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["segment_id"], "seg_0002_z01")
            self.assertEqual(tool_args["start_sec"], 30.0)
            self.assertEqual(tool_args["end_sec"], 45.0)
            self.assertEqual(result.rounds[0].program[0]["tool"], "qa_segment")

    def test_iterative_agent_resolves_zoom_child_id_from_prior_zoom_observation(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "zoom", "args": {"segment_id": "seg_0001", "target_granularity_sec": 10.0}, "assign": "zoomed"}'
                    "]}"
                ),
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001_z02", "question": "Inspect child"}, "assign": "child"}'
                    "]}"
                ),
            ]
        )
        store = VideoMapStore(
            VideoMap(
                video_path="/videos/demo.mp4",
                duration_sec=30.0,
                segments=[VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
            )
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="zoom_child_from_trace")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=store, backend=backend, workspace=workspace),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False),
            )

            result = agent.run(question="Which child is visible?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[1].program[0]["args"]
            self.assertEqual(tool_args["segment_id"], "seg_0001_z02")
            self.assertEqual(tool_args["start_sec"], 10.0)
            self.assertEqual(tool_args["end_sec"], 20.0)

    def test_iterative_agent_stops_repeated_identical_programs_as_no_progress(self):
        repeated_program = (
            '{"status": "continue", "rationale": "same map", "program": ['
            '{"tool": "video_ls", "args": {"query": "same segment"}, "assign": "map"}'
            "]}"
        )
        backend = ScriptedPlannerBackend([repeated_program] * 6)
        registry = ToolRegistry()

        @tool(name="video_ls", description="Return the same map.")
        def video_ls(query: str = ""):
            return {"claim": f"same candidate for {query}", "confidence": 1.0}

        registry.register(video_ls)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_progress")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=6,
                    reserve_final_round=False,
                    max_repeated_programs=2,
                ),
            )

            result = agent.run(question="Find the same thing", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(len(result.rounds), 2)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_no_progress_guard", trace)
            self.assertIn("repeated_program", trace)

    def test_iterative_agent_repairs_media_tool_missing_segment_id_from_time_window(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {'
                    '"video_path": "artifacts/clips/seg_0002.mp4", '
                    '"start_sec": 30.0, "end_sec": 45.0, "question": "Inspect this window"}'
                    ', "assign": "detail"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="repair_missing_segment")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertTrue(tool_args["segment_id"].startswith("window_"))
            self.assertEqual(tool_args["video_path"], "/videos/demo.mp4")
            self.assertEqual(tool_args["start_sec"], 30.0)
            self.assertEqual(tool_args["end_sec"], 45.0)
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001"])

    def test_iterative_agent_clamps_dynamic_tail_window_to_video_duration(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {'
                    '"start_sec": 1800.0, "end_sec": 1805.0, "question": "Inspect the tail"}'
                    ', "assign": "tail"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1804.96, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="tail_window")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens at the end?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["start_sec"], 1800.0)
            self.assertEqual(tool_args["end_sec"], 1804.96)
            self.assertTrue(tool_args["segment_id"].startswith("window_"))

    def test_iterative_agent_resolves_reused_dynamic_window_segment_id(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "window_000300000_000600000", '
                    '"question": "Reinspect this dynamic window"}'
                    ', "assign": "dyn"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=900.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="reused_dynamic_window")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["segment_id"], "window_000300000_000600000")
            self.assertEqual(tool_args["start_sec"], 300.0)
            self.assertEqual(tool_args["end_sec"], 600.0)

    def test_iterative_agent_normalizes_dynamic_window_milliseconds(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"start_sec": 1800000.0, "end_sec": 1804957.0, '
                    '"question": "Inspect this millisecond window"}'
                    ', "assign": "dyn"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1804.957, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="dynamic_window_ms")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["start_sec"], 1800.0)
            self.assertEqual(tool_args["end_sec"], 1804.957)
            self.assertEqual(tool_args["segment_id"], "window_001800000_001804957")

    def test_iterative_agent_injects_mcq_options_into_inspector_args(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Which option is visible?"}, "assign": "inspection"}'
                    "]}"
                ),
                '{"status": "final", "answer": "A", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="inject_mcq_options")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question="Which option is visible?\nA. aircraft museum\nB. submarine",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["candidate_options"], ["A. aircraft museum", "B. submarine"])

    def test_iterative_agent_replaces_letter_only_candidate_options_with_full_options(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Which option is visible?", '
                    '"candidate_options": ["A", "B"]}, "assign": "inspection"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_full_options")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which option is visible?\nA. aircraft museum\nB. submarine",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["candidate_options"], ["A. aircraft museum", "B. submarine"])

    def test_iterative_agent_recovers_when_planner_copies_unescaped_option_quotes(self):
        malformed_planner_json = (
            '{"status": "continue", "program": [{"tool": "inspect_segment", '
            '"args": {"segment_id": "seg_0001", "question": "Which artwork appears after "David"?", '
            '"candidate_options": ["A. "David" then Apollo", "B. plain option"]}, "assign": "bad_json"}]}'
        )
        backend = ScriptedPlannerBackend(
            [
                malformed_planner_json,
                '{"status": "final", "answer": "A. The cited inspection supports A.", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_bad_json_recovery")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question='Which artwork appears after David?\nA. "David" then Apollo\nB. plain option',
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.rounds[0].program[0]["tool"], "inspect_segment")
            self.assertEqual(
                result.rounds[0].program[0]["args"]["candidate_options"],
                ['A. "David" then Apollo', "B. plain option"],
            )
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_json_parse_error", trace)

    def test_iterative_agent_appends_mcq_options_to_caption_question(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Describe the sequence."}, '
                    '"assign": "caption"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="caption_options")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which order is shown?\nA. red then blue\nB. blue then red",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertIn("Options:", tool_args["question"])
            self.assertIn("A. red then blue", tool_args["question"])
            self.assertIn("B. blue then red", tool_args["question"])

    def test_iterative_agent_blocks_mcq_final_until_inspector_with_options(self):
        backend = ScriptedPlannerBackend(
            [
                '{"status": "final", "answer": "A", "citations": [], "confidence": 0.9}',
                '{"status": "final", "answer": "A", "citations": ["obs_0001"], "confidence": 0.9}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="block_unsupported_mcq_final")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question="Which option is visible?\nA. aircraft museum\nB. submarine",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual(result.rounds[0].status, "continue")
            self.assertEqual(result.rounds[0].program[0]["tool"], "inspect_segment")
            self.assertEqual(result.rounds[0].program[0]["args"]["candidate_options"], ["A. aircraft museum", "B. submarine"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_final_blocked", trace)

    def test_iterative_agent_prompt_includes_broad_long_video_index(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/long.mp4", duration_sec=600.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="long_index")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens near the end?", video_path="/videos/long.mp4")

            self.assertIn("seg_0020 [570.0-600.0s]", backend.requests[0].prompt)

    def test_iterative_agent_stops_at_round_budget_with_partial_evidence(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                )
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="budget")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertIn("Partial evidence summary", result.answer)
            self.assertIn("aircraft history", result.answer)

    def test_iterative_agent_reserves_final_round_from_new_visual_tools(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                )
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="reserved_final")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=True),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(result.citations, [])
            self.assertEqual(result.rounds[0].program, [])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("reserve_final_round", trace)

    def test_iterative_agent_uses_answer_agent_when_reserved_final_round_would_continue(self):
        class ReservedFinalAnswerBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                                "]}"
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0002", "question": "Inspect more"}, "assign": "s2"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    self.answer_prompt = request.prompt
                    return BackendResponse(
                        text='{"answer": "B", "rationale": "obs_0001 supports B.", "citations": ["obs_0001"], "missing_evidence": [], "confidence": 0.82}'
                    )
                return BackendResponse(text="unexpected")

        backend = ReservedFinalAnswerBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="reserved_final_answer")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=True),
            )

            result = agent.run(
                question="Which option is visible?\nA. submarine\nB. aircraft museum",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "B")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertIn("obs_0001", backend.answer_prompt)
            self.assertEqual([request.task for request in backend.requests], ["replan", "replan", "answer_from_evidence"])

    def test_iterative_agent_answer_agent_arbitrates_option_grouped_evidence(self):
        class ArbitrationBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    segment_id = f"seg_{self.replan_calls:04d}"
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "'
                            + segment_id
                            + '", "question": "Inspect option support", '
                            '"candidate_options": ["A. first", "D. fourth"]}, "assign": "s"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text='{"answer": "A. first", "rationale": "recency fallback", "citations": ["obs_0002"], "missing_evidence": [], "confidence": 0.95}'
                    )
                return BackendResponse(text="unexpected")

        registry = ToolRegistry()

        @tool(name="inspect_segment", description="Inspect one segment with option support.")
        def inspect_segment(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            question: str,
            candidate_options=None,
            nframes: int = 16,
        ):
            if segment_id == "seg_0001":
                return {
                    "claim": "Visual evidence supports option D.",
                    "confidence": 0.72,
                    "input_artifacts": [f"{video_path}#t={start_sec:.1f},{end_sec:.1f}"],
                    "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                    "limitations": "Directly visible in the sampled segment.",
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [
                        {"option": "D", "relation": "support", "strength": 0.72, "assigned_by": "answer_agent"}
                    ],
                }
            return {
                "claim": "Caption-like evidence guesses option A.",
                "confidence": 0.95,
                "input_artifacts": [f"{video_path}#t={start_sec:.1f},{end_sec:.1f}"],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "limitations": "Inferred from context; lacks explicit visual confirmation.",
                "candidate_option_relations": [
                    {"option": "A", "relation": "support", "strength": 0.95, "assigned_by": "answer_agent"}
                ],
            }

        registry.register(inspect_segment)
        backend = ArbitrationBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=90.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="answer_arbitration")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, reserve_final_round=True),
            )

            result = agent.run(
                question="Which option is correct?\nA. first\nD. fourth",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D. fourth")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual([request.task for request in backend.requests], ["replan", "replan", "replan"])

    def test_iterative_agent_feeds_prefinal_answer_gaps_into_next_prompt(self):
        class PrefinalProbeBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0
                self.answer_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect first option", '
                                '"candidate_options": ["A. first", "B. second", "C. third", "D. fourth"]}, "assign": "s1"}'
                                "]}"
                            )
                        )
                    if self.replan_calls == 2:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0002", "question": "Inspect second option", '
                                '"candidate_options": ["A. first", "B. second", "C. third", "D. fourth"]}, "assign": "s2"}'
                                "]}"
                            )
                        )
                    if self.replan_calls == 3:
                        self.round3_prompt = request.prompt
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0003", "question": "Resolve the missing ordering evidence", '
                                '"candidate_options": ["A. first", "B. second", "C. third", "D. fourth"]}, "assign": "s3"}'
                                "]}"
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0004", "question": "Should be reserved"}, "assign": "s4"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    self.answer_calls += 1
                    if self.answer_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"answer": "need_more_evidence", "rationale": "Need order evidence.", '
                                '"citations": [], "missing_evidence": ["explicit order of the four options"], "confidence": 0.0}'
                            )
                        )
                    return BackendResponse(
                        text='{"answer": "D", "rationale": "obs_0003 resolves the order.", "citations": ["obs_0003"], "missing_evidence": [], "confidence": 0.86}'
                    )
                return BackendResponse(text="unexpected")

        backend = PrefinalProbeBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="prefinal_answer_probe")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=4, reserve_final_round=True, answer_probe_rounds_before_final=2),
            )

            result = agent.run(
                question="Which option is correct?\nA. first\nB. second\nC. third\nD. fourth",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D")
            self.assertIn("Answer Agent says these evidence gaps", backend.round3_prompt)
            self.assertIn("explicit order of the four options", backend.round3_prompt)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("prefinal_probe", trace)
            self.assertIn("reserved_final", trace)

    def test_segment_vlm_tools_share_backend_and_pass_temporal_metadata(self):
        class SegmentToolBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                return BackendResponse(text="The segment shows a runway and aircraft.")

        backend = SegmentToolBackend()
        registry = build_segment_vlm_registry(backend)

        result = registry.execute(
            "caption_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0002",
                "start_sec": 10.0,
                "end_sec": 20.0,
                "question": "What is visible?",
                "nframes": 12,
            },
        )

        self.assertEqual(result["claim"], "The segment shows a runway and aircraft.")
        self.assertEqual(backend.requests[0].task, "caption_segment")
        self.assertEqual(backend.requests[0].media_path, "/videos/demo.mp4")
        self.assertEqual(backend.requests[0].metadata["segment_id"], "seg_0002")
        self.assertEqual(backend.requests[0].metadata["start_sec"], 10.0)
        self.assertEqual(backend.requests[0].metadata["end_sec"], 20.0)
        self.assertEqual(backend.requests[0].metadata["nframes"], 64)
        self.assertEqual(backend.requests[0].metadata["max_pixels"], 151200)
        self.assertEqual(backend.requests[0].metadata["question"], "What is visible?")

    def test_segment_vlm_tools_can_extract_physical_clip_before_backend_call(self):
        class SegmentToolBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                return BackendResponse(text="The extracted clip shows a museum exhibit.")

        extracted = []

        def fake_clip_extractor(video_path, output_path, start_sec, end_sec):
            extracted.append((video_path, output_path, start_sec, end_sec))
            Path(output_path).write_text("fake clip", encoding="utf-8")
            return output_path

        backend = SegmentToolBackend()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="clip_tools")
            registry = build_segment_vlm_registry(
                backend,
                workspace=workspace,
                extract_clips=True,
                clip_extractor=fake_clip_extractor,
            )

            result = registry.execute(
                "caption_segment",
                {
                    "video_path": "/videos/demo.mp4",
                    "segment_id": "seg_0002",
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                    "question": "What is visible?",
                    "nframes": 12,
                },
            )

            self.assertEqual(len(extracted), 1)
            self.assertEqual(extracted[0][0], "/videos/demo.mp4")
            self.assertIn("seg_0002_10000_20000.mp4", extracted[0][1])
            self.assertEqual(backend.requests[0].media_path, extracted[0][1])
            self.assertEqual(result["input_artifacts"], [extracted[0][1]])
            self.assertEqual(result["regions"][0]["source_video_path"], "/videos/demo.mp4")
            self.assertEqual(result["regions"][0]["clip_path"], extracted[0][1])

    def test_iterative_smoke_runner_uses_shared_backend_and_fixed_window_index(self):
        class SharedBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "Inspect"}, "assign": "s2"}'
                                "]}"
                            )
                        )
                    return BackendResponse(text='{"status": "final", "answer": "Segment two is relevant.", "citations": ["obs_0001"]}')
                if request.task == "caption_segment":
                    return BackendResponse(text="Segment two contains the relevant evidence.")
                return BackendResponse(text="")

        backend = SharedBackend()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_iterative_smoke(
                base_dir=Path(tmp),
                backend=backend,
                media_path="/videos/demo.mp4",
                question="What matters?",
                duration_sec=60.0,
                window_sec=30.0,
                run_id="iterative_smoke",
            )

            self.assertEqual(result.answer, "Segment two is relevant.")
            self.assertEqual([request.task for request in backend.requests], ["replan", "caption_segment", "replan"])
            self.assertEqual(backend.requests[1].metadata["start_sec"], 30.0)


if __name__ == "__main__":
    unittest.main()
