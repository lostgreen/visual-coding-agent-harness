import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import (
    AgentBudget,
    IterativeVisualAgent,
    _sanitize_option_blind_feedback,
)
from visual_coding_agent_harness.agents.question_policy import extract_candidate_options
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
        sample_offset_sec: float = 0.0,
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
                    "sample_offset_sec": sample_offset_sec,
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
            self.assertIn("Compact scene index", backend.requests[0].prompt)
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

    def test_iterative_agent_prompt_puts_scene_evidence_before_tooling(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    low_fps_caption="opening clue",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="late_tooling_prompt")
            workspace.write_observation(tool_name="vision_read", claim="prior visual fact", confidence=0.8)
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            prompt = backend.requests[0].prompt
            self.assertLess(prompt.index("Evidence ledger"), prompt.index("Available tools"))
            self.assertLess(prompt.index("Compact scene index"), prompt.index("Available tools"))
            self.assertLess(prompt.index("Current budgets"), prompt.index("Available tools"))
            self.assertGreater(prompt.rindex("Return only JSON"), prompt.index("Available tools"))

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

    def test_iterative_agent_prompt_includes_workspace_evidence_status_summary(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "B", "citations": ["obs_0001"], "confidence": 0.7}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="evidence_status_prompt")
            workspace.write_observation(
                tool_name="vision_read",
                claim="The clip shows a red aircraft.",
                confidence=0.82,
                input_artifacts=["/videos/demo.mp4"],
                regions=[{"segment_id": "seg_0001", "start_sec": 1.0, "end_sec": 2.0}],
                raw_output={
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [{"option": "B", "relation": "support", "strength": 0.82}],
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(
                question="Which option is visible?\nA. blue car\nB. red aircraft",
                video_path="/videos/demo.mp4",
            )

            prompt = backend.requests[0].prompt
            self.assertIn("Evidence status summary:", prompt)
            self.assertIn("option_coverage: 1/2", prompt)
            self.assertIn("B: strong=1 weak=0 visual=yes", prompt)

    def test_gist_global_route_seeds_one_topic_hint_without_finalizing(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "No more evidence.", "citations": [], "confidence": 0.0}']
        )
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

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(workspace.observation_count(tool_name="global_gist"), 1)
            self.assertIn("replan", [request.task for request in backend.requests])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("global_gist_topic_seeded", trace)

    def test_budget_can_disable_global_gist_shortcut_for_planner_trace_debugging(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "planner handled it", "citations": [], "confidence": 0.4}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1896.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="disable_global_route")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_global_route_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(disable_global_gist_route=True, reserve_final_round=False),
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

            self.assertGreaterEqual(len(backend.requests), 1)
            self.assertEqual(backend.requests[0].task, "replan")
            self.assertEqual(workspace.observation_count(tool_name="global_gist"), 0)

    def test_main_idea_vision_read_allowed_after_one_global_gist(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "inspect local coverage", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001", '
                    '"ask_for": "Describe this segment facts."}, "assign": "local"}]}'
                )
            ]
        )
        registry = build_global_route_test_registry()

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            return {
                "claim": f"{segment_id} discusses empire chronology.",
                "confidence": 0.82,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
                "event_label": event_label or ask_for,
            }

        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=60.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="allow_local_after_global")
            workspace.write_observation(
                tool_name="global_gist",
                claim="Sparse whole-video topic hint.",
                confidence=0.76,
                regions=[{"start_sec": 0.0, "end_sec": 120.0}],
                raw_output={"grounding_quality": "global_sparse", "candidate_option_hint": "D"},
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    max_tool_calls_per_round=1,
                    reserve_final_round=False,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nB. empire division\nD. empire rise and fall",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.rounds[0].program[0]["tool"], "vision_read")
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("repair_main_idea_vision_read_to_global_gist", trace)

    def test_main_idea_repeated_global_gist_repaired_to_local_read(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "planner repeated sparse topic hint", '
                    '"skill": "main_idea", '
                    '"program": [{"tool": "global_gist", "args": {"question": "main idea", '
                    '"duration_sec": 120.0}, "assign": "repeat_global"}]}'
                )
            ]
        )
        registry = build_global_route_test_registry()

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            return {
                "claim": f"{segment_id} local coverage fact.",
                "confidence": 0.82,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=60.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="repeat_global_to_local")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=1, reserve_final_round=False),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nB. empire division\nD. empire rise and fall",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(workspace.observation_count(tool_name="global_gist"), 1)
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 1)
            self.assertEqual(result.rounds[0].program[0]["tool"], "vision_read")
            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0001")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("repair_repeated_main_idea_global_gist_to_vision_read", trace)

    def test_normalizes_placeholder_video_path_for_global_tools(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "global_gist", "args": {"video_path": "video_path", '
                    '"question": "main idea", "duration_sec": 60.0}, "assign": "g1"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="placeholder_video_path")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_global_route_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=False,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nD. aviation documentary",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["video_path"], "/videos/demo.mp4")
            self.assertEqual(workspace.get_observation("obs_0001").input_artifacts, ["/videos/demo.mp4"])

    def test_repairs_verify_alias_to_registered_ledger_verifier(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "verify", "args": {"answer": "B", "ledger_text": "B is supported"}, "assign": "v1"}'
                    "]}"
                )
            ]
        )
        registry = ToolRegistry()

        @tool(name="verify_ledger_answer", description="Verify answer support.")
        def verify_ledger_answer(answer: str, ledger_text: str = ""):
            return {"claim": f"{answer} is checked against {ledger_text}", "confidence": 0.8}

        registry.register(verify_ledger_answer)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_alias")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=False,
                    hard_skill_runtime=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nB. empire division",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.rounds[0].program[0]["tool"], "verify_ledger_answer")
            self.assertEqual(workspace.observation_count(tool_name="verify_ledger_answer"), 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"type": "route_violation"', trace)

    def test_normalization_failure_surfaces_in_next_prompt(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "inspect first", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "What is visible?"}, "assign": "first"}]}'
                ),
                (
                    '{"status": "continue", "rationale": "repeat first", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "What is visible?"}, "assign": "repeat"}]}'
                ),
                '{"status": "final", "answer": "The video shows aircraft history.", "citations": ["obs_0001"], "confidence": 0.7}',
            ]
        )
        registry = ToolRegistry()

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            return {
                "claim": f"{segment_id} shows aircraft history.",
                "confidence": 0.75,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
                "event_label": event_label or ask_for,
            }

        registry.register(vision_read)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=120.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0),
                VideoSegment(segment_id="seg_0002", start_sec=60.0, end_sec=120.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="normalization_feedback")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, reserve_final_round=False),
            )

            agent.run(question="What is visible?", video_path="/videos/demo.mp4")

            self.assertEqual(len(backend.requests), 3)
            self.assertIn("Last Round Adjustments", backend.requests[2].prompt)
            self.assertIn("avoid_repeated_segment", backend.requests[2].prompt)
            self.assertIn("seg_0001", backend.requests[2].prompt)
            self.assertIn("seg_0002", backend.requests[2].prompt)

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
        class BadJsonRecoveryBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    self.requests.append(request)
                    return BackendResponse(
                        text='{"answer": "A. The cited inspection supports A.", "citations": ["obs_0001"], '
                        '"confidence": 0.8, "candidate_option_relations": ['
                        '{"option": "A", "relation": "support", "strength": 0.8, "observation_id": "obs_0001"}]}'
                    )
                return super().generate(request)

        backend = BadJsonRecoveryBackend(
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

    def test_iterative_agent_rewrites_mcq_caption_question_without_options(self):
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
            self.assertIn("Describe the sequence", tool_args["question"])
            self.assertIn("Do not choose an option", tool_args["question"])
            self.assertNotIn("Options:", tool_args["question"])
            self.assertNotIn("A. red then blue", tool_args["question"])
            self.assertNotIn("B. blue then red", tool_args["question"])

    def test_iterative_agent_blocks_mcq_final_until_inspector_with_options(self):
        class McqFinalBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    self.requests.append(request)
                    return BackendResponse(
                        text='{"answer": "A. aircraft museum", "citations": ["obs_0001"], "confidence": 0.9, '
                        '"candidate_option_relations": [{"option": "A", "relation": "support", '
                        '"strength": 0.9, "observation_id": "obs_0001"}]}'
                    )
                return super().generate(request)

        backend = McqFinalBackend(
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

    def test_planner_final_mcq_is_always_replaced_by_answer_agent_final(self):
        class FinalTakeoverBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text='{"status": "final", "answer": "A", "citations": ["obs_0001"], "confidence": 0.99}'
                    )
                if request.task == "answer_from_evidence":
                    self.assertIn("A. wrong option", request.prompt)
                    self.assertIn("B. correct option from evidence", request.prompt)
                    return BackendResponse(
                        text=(
                            '{"answer": "B. correct option from evidence", "citations": ["obs_0001"], '
                            '"confidence": 0.86, "candidate_option_relations": ['
                            '{"option": "B", "relation": "support", "strength": 0.86, '
                            '"observation_id": "obs_0001", "mapped_from_facts": ["obs_0001"]}]}'
                        )
                    )
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_final_takeover")
            workspace.write_observation(tool_name="vision_read", claim="Evidence supports B.", confidence=0.9)
            agent = IterativeVisualAgent(
                backend=FinalTakeoverBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question="Which option is supported?\nA. wrong option\nB. correct option from evidence",
                video_path="/videos/demo.mp4",
            )

            self.assertTrue(result.answer.startswith("B"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_final_answer_agent_takeover", trace)
            self.assertIn('"source": "planner_final_takeover"', trace)

    def test_iterative_agent_blocks_main_idea_planner_final_without_structured_support(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "vision_read", "args": {"segment_id": "seg_0001", '
                    '"ask_for": "Describe the opening collapse evidence."}, "assign": "v1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "C. the empire declining and collapsing", '
                '"citations": ["obs_0001"], "confidence": 0.91}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)
        registry = build_segment_test_registry()

        @tool(name="vision_read", description="Read localized visual evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 16,
        ):
            return {
                "claim": "The sampled segment shows only an ending collapse scene, not a full-video arc.",
                "confidence": 0.82,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "nframes": nframes}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="block_main_idea_planner_final")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "A. the empire's allies\n"
                    "B. why the empire was divided\n"
                    "C. the empire declining and collapsing\n"
                    "D. the empire's rise, stability, decline, and collapse"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertNotEqual(result.status, "final")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_final_blocked", trace)
            self.assertIn("planner_final_answer_agent_takeover", trace)
            self.assertIn("planner_final_requires_answer_agent", trace)

    def test_iterative_agent_indexes_scene_coverage_for_main_idea_mcq(self):
        class SceneCoverageBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "use indexed coverage rows", '
                            '"citations": [], "missing_evidence": ["more coverage"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text='{"status": "continue", "program": []}')

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    low_fps_caption="ASR/subtitle excerpt: the empire is created and rises with economic growth.",
                ),
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=30.0,
                    end_sec=60.0,
                    low_fps_caption="ASR/subtitle excerpt: internal stability, governance, and prosperity are discussed.",
                ),
                VideoSegment(
                    segment_id="seg_0003",
                    start_sec=60.0,
                    end_sec=90.0,
                    low_fps_caption="ASR/subtitle excerpt: war pressure leads to decline, collapse, and the fall of the empire.",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="scene_coverage_main_idea")
            agent = IterativeVisualAgent(
                backend=SceneCoverageBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "B. Why the Austro-Hungarian Empire was divided\n"
                    "D. How the Austro-Hungarian Empire rises and falls"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D. How the Austro-Hungarian Empire rises and falls")
            self.assertEqual(set(result.citations), {"scene_coverage_seg_0001", "scene_coverage_seg_0003"})
            table = workspace.evidence_table_v2(
                question="What is the video mainly about?",
                options=[
                    "B. Why the Austro-Hungarian Empire was divided",
                    "D. How the Austro-Hungarian Empire rose and fell",
                ],
            )
            self.assertGreaterEqual(len(table["groups"]["D"]), 2)

    def test_iterative_agent_indexes_scene_order_for_videomme_masterpiece_sequence(self):
        class SceneOrderBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "use indexed order rows", '
                            '"citations": [], "missing_evidence": ["ordered artwork evidence"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text='{"status": "continue", "program": []}')

        scene_index = SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=1200.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=300.0,
                    low_fps_caption='The first Borghese masterpiece shown is "Aeneas, Anchises, and Ascanius fleeing Troy".',
                ),
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=300.0,
                    end_sec=600.0,
                    low_fps_caption='The next sculpture presented is "David" by Bernini.',
                ),
                VideoSegment(
                    segment_id="seg_0003",
                    start_sec=600.0,
                    end_sec=900.0,
                    low_fps_caption='The scene then discusses "The rape of Persephone".',
                ),
                VideoSegment(
                    segment_id="seg_0004",
                    start_sec=900.0,
                    end_sec=1200.0,
                    low_fps_caption='The final sculpture in the sequence is "Apollo and Daphne".',
                ),
            ],
        )

        question = (
            "VideoMME multiple-choice question. Answer with exactly one option letter (A/B/C/D) first.\n"
            "Question: As depicted in the video, in what order does the author present Bernini's four "
            "masterpieces created for Borghese in a single scene?\n"
            "Options:\n"
            'A. "The rape of Persephone", "Apollo and Daphne", "David" and "Aeneas, Anchises, and Ascanius fleeing Troy".\n'
            'B. "David", "Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne" and "The rape of Persephone".\n'
            'C. "Apollo and Daphne", "Aeneas, Anchises, and Ascanius fleeing Troy", "David" and "The rape of Persephone".\n'
            'D. "Aeneas, Anchises, and Ascanius fleeing Troy", "David", "The rape of Persephone" and "Apollo and Daphne".'
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="scene_order_masterpieces")
            agent = IterativeVisualAgent(
                backend=SceneOrderBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(question=question, video_path="/videos/bernini.mp4")

            self.assertEqual(result.status, "final")
            self.assertEqual(
                result.answer,
                'D. "Aeneas, Anchises, and Ascanius fleeing Troy", "David", '
                '"The rape of Persephone" and "Apollo and Daphne".',
            )
            self.assertEqual(set(result.citations), {"scene_order_seg_0001", "scene_order_seg_0004"})
            table = workspace.evidence_table_v2(question=question, options=extract_candidate_options(question))
            self.assertGreaterEqual(len(table["groups"]["D"]), 4)

    def test_iterative_agent_indexes_scene_order_for_life_journey_when_planner_selects_timeline_skill(self):
        class LifeJourneyBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": []}')
                return BackendResponse(
                    text='{"status": "continue", "skill": "timeline_ordering", "program": []}'
                )

        scene_index = SceneIndex(
            video_path="/videos/goya.mp4",
            duration_sec=900.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=300.0,
                    low_fps_caption="The narration says he was born with a humble background.",
                ),
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=300.0,
                    end_sec=600.0,
                    low_fps_caption="Later he entered the upper class as his work gained noble patrons.",
                ),
                VideoSegment(
                    segment_id="seg_0003",
                    start_sec=600.0,
                    end_sec=900.0,
                    low_fps_caption="In the final part he lived in seclusion in a farmhouse.",
                ),
            ],
        )
        question = (
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: How was his life journey according to the video?\n"
            "Options:\n"
            "A. Borned with humble background and lived in seclusion in a farmhouse.\n"
            "B. Borned with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
            "C. Borned with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
            "D. Borned in the upper class and lived in seclusion in a farmhouse."
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="scene_order_life_journey")
            agent = IterativeVisualAgent(
                backend=LifeJourneyBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=True, disable_global_gist_route=True),
            )

            result = agent.run(question=question, video_path="/videos/goya.mp4")

            self.assertEqual(result.status, "final")
            self.assertEqual(
                result.answer,
                "B. Borned with a humble background, entered the upper class and then lived in seclusion in a farmhouse.",
            )
            self.assertEqual(set(result.citations), {"scene_order_seg_0001", "scene_order_seg_0003"})
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_skill_selection", trace)

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

    def test_iterative_agent_returns_low_confidence_when_budget_exhausts_with_partial_support(self):
        class LowConfidenceBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", '
                            '"question": "Inspect option evidence", '
                            '"candidate_options": ["A. submarine", "B. aircraft museum"]}, "assign": "s1"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "partial support only", '
                            '"citations": [], '
                            '"candidate_option_relations": ['
                            '{"option": "B", "relation": "support", "strength": 0.8, '
                            '"observation_id": "obs_0001", "grounding_quality": "visually_confirmed"}'
                            '], "missing_evidence": ["need one more local view"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text="unexpected")

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="low_conf_budget")
            agent = IterativeVisualAgent(
                backend=LowConfidenceBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which option is visible?\nA. submarine\nB. aircraft museum",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(result.answer, "B")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertAlmostEqual(result.confidence, 0.56)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("budget_exhausted", trace)
            self.assertIn("low_confidence_final", trace)

    def test_iterative_agent_blocks_low_confidence_without_visual_citation(self):
        class NoVisualLowConfidenceBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "video_ls", "args": {"query": "aircraft"}, "assign": "map"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "partial map-only support", '
                            '"citations": [], '
                            '"candidate_option_relations": ['
                            '{"option": "B", "relation": "support", "strength": 0.8, '
                            '"observation_id": "obs_0001", "grounding_quality": "visually_confirmed"}'
                            '], "missing_evidence": ["need visual confirmation"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text="unexpected")

        registry = ToolRegistry()

        @tool(name="video_ls", description="Return a navigation-only map row.")
        def video_ls(query: str = ""):
            return {"claim": f"navigation candidate for {query}", "confidence": 1.0}

        registry.register(video_ls)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="low_conf_blocked")
            agent = IterativeVisualAgent(
                backend=NoVisualLowConfidenceBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which option is visible?\nA. submarine\nB. aircraft museum",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "max_rounds_reached")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("low_confidence_final_blocked", trace)

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

    def test_temporal_ordering_uses_timeline_for_unique_option(self):
        registry = ToolRegistry()

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            if start_sec < 20.0:
                claim = "The light turns on in this early segment."
            elif start_sec >= 40.0:
                claim = "The door opens in this late segment."
            else:
                claim = "No target event appears here."
            return {
                "claim": claim,
                "confidence": 0.82,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read temporal event timestamp.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
        ):
            observed = 40.5 if "door" in ask_for.lower() else 10.5
            return {
                "claim": f"{event_label} is observed at {observed:.1f} seconds.",
                "confidence": 0.94,
                "event_label": event_label,
                "observed_at_sec": observed,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "grounding_quality": "visually_confirmed",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class NeedMoreBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "timeline should decide before this", '
                            '"citations": [], "missing_evidence": ["more evidence"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text='{"status": "final", "answer": "fallback", "citations": []}')

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_temporal")
            agent = IterativeVisualAgent(
                backend=NeedMoreBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=4, hard_skill_runtime=True),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "B")
            self.assertEqual(result.citations, ["obs_0007", "obs_0008"])

    def test_timeline_ordering_uses_caption_pass_before_focused_reads(self):
        registry = ToolRegistry()
        calls = []

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            calls.append(("caption_segment", segment_id, question))
            claim_by_segment = {
                "seg_0001": "The light turns on in this early segment.",
                "seg_0002": "The door opens in this later segment.",
                "seg_0003": "A closing title appears.",
            }
            return {
                "claim": claim_by_segment[segment_id],
                "confidence": 0.82,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read precise first timestamp.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
        ):
            calls.append(("vision_read", segment_id, ask_for))
            observed = 11.0 if "light" in ask_for.lower() else 35.0
            return {
                "claim": f"{event_label} is first visible at {observed:.1f} seconds.",
                "confidence": 0.94,
                "event_label": event_label,
                "observed_at_sec": observed,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class NeedMoreBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "missing_evidence": ["timeline should decide"], "citations": []}')
                return BackendResponse(text='{"status": "final", "answer": "fallback", "citations": []}')

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=45.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=15.0),
                VideoSegment(segment_id="seg_0002", start_sec=15.0, end_sec=30.0),
                VideoSegment(segment_id="seg_0003", start_sec=30.0, end_sec=45.0),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_caption_pass")
            agent = IterativeVisualAgent(
                backend=NeedMoreBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=8, hard_skill_runtime=True),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "B")
            self.assertEqual(
                [step["tool"] for step in result.rounds[0].program],
                ["caption_segment", "caption_segment", "caption_segment", "vision_read", "vision_read"],
            )
            self.assertNotIn("ground_question", [call[0] for call in calls])
            self.assertEqual([call[1] for call in calls if call[0] == "vision_read"], ["seg_0001", "seg_0002"])

    def test_timeline_ordering_missing_entity_returns_need_more_evidence(self):
        registry = ToolRegistry()

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            return {
                "claim": "Only the light turns on here.",
                "confidence": 0.8,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read precise first timestamp.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = ""):
            return {
                "claim": f"{event_label} is first visible at 10.0 seconds.",
                "confidence": 0.9,
                "event_label": event_label,
                "observed_at_sec": 10.0,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class AbstainBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                return BackendResponse(text='{"answer": "need_more_evidence", "missing_evidence": ["door opens"], "citations": []}')

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_missing")
            agent = IterativeVisualAgent(
                backend=AbstainBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=8, hard_skill_runtime=True, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "need_more_evidence")
            self.assertIn("door opens", result.answer)

    def test_timeline_ordering_ignores_negative_caption_echoes(self):
        registry = ToolRegistry()
        calls = []

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            calls.append(("caption_segment", segment_id))
            return {
                "claim": "The video does not contain any visible evidence of door opens or light turns on.",
                "confidence": 0.8,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read precise first timestamp.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = ""):
            calls.append(("vision_read", segment_id))
            return {
                "claim": "No direct evidence is visible.",
                "confidence": 0.2,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "confidence_signal": "unsupported",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class AbstainBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                return BackendResponse(text='{"answer": "need_more_evidence", "missing_evidence": ["missing timestamps"], "citations": []}')

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_negative_echo")
            agent = IterativeVisualAgent(
                backend=AbstainBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=8, hard_skill_runtime=True, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "need_more_evidence")
            self.assertEqual([call[0] for call in calls].count("vision_read"), 0)
            self.assertIn("door opens", result.answer)
            self.assertIn("timeline_ordering_missing_entity", (workspace.root / "trace.jsonl").read_text(encoding="utf-8"))

    def test_no_evidence_growth_forces_low_confidence(self):
        planner_responses = [
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "first pass"}, "assign": "map1"}]}',
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "second pass"}, "assign": "map2"}]}',
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "third pass"}, "assign": "map3"}]}',
        ]

        class PartialSupportBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "citations": ["obs_0001"], '
                            '"candidate_option_relations": ['
                            '{"option": "A", "relation": "support", "strength": 0.6, "observation_id": "obs_0001"}'
                            '], "missing_evidence": ["need a confirming read"], "confidence": 0.2}'
                        )
                    )
                return super().generate(request)

        registry = ToolRegistry()

        @tool(name="video_ls", description="Cheap navigation that adds no answer evidence.")
        def video_ls(query: str = ""):
            return {"claim": f"navigation only: {query}", "confidence": 1.0}

        registry.register(video_ls)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_evidence_growth")
            observation = workspace.write_observation(
                tool_name="vision_read",
                claim="A red object is visible.",
                confidence=0.8,
                input_artifacts=["/videos/demo.mp4"],
                regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 5.0}],
                raw_output={"grounding_quality": "visually_confirmed"},
            )
            workspace.write_ledger_entry(observation)
            agent = IterativeVisualAgent(
                backend=PartialSupportBackend(planner_responses),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=5,
                    reserve_final_round=False,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(question="Which option is visible?\nA. red object\nB. blue object", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(result.answer, "A")
            self.assertEqual(len(result.rounds), 2)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("evidence_table_no_growth", trace)

    def test_no_evidence_growth_replaces_navigation_only_plan_with_visual_read(self):
        planner_responses = [
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "first pass"}, "assign": "map1"}]}',
            '{"status": "continue", "program": [{"tool": "search_segments", "args": {"query": "second pass"}, "assign": "map2"}]}',
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "third pass"}, "assign": "map3"}]}',
        ]

        class AbstainBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["visual read needed"]}'
                    )
                return super().generate(request)

        registry = ToolRegistry()
        calls = []

        @tool(name="video_ls", description="Cheap navigation that adds no answer evidence.")
        def video_ls(query: str = ""):
            calls.append(("video_ls", query))
            return {"claim": f"navigation only: {query}", "confidence": 1.0}

        @tool(name="search_segments", description="Cheap search that adds no answer evidence.")
        def search_segments(query: str, top_k: int = 5):
            calls.append(("search_segments", query))
            return {"claim": f"search only: {query}", "confidence": 1.0}

        @tool(name="vision_read", description="Visual read that creates answer evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append(("vision_read", segment_id, ask_for))
            return {
                "claim": f"{segment_id} visual evidence for {ask_for}",
                "confidence": 0.74,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(video_ls)
        registry.register(search_segments)
        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_growth_visual_fallback")
            agent = IterativeVisualAgent(
                backend=AbstainBackend(planner_responses),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=3,
                    reserve_final_round=False,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(question="Describe what is visible.", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[2].program[0]["tool"], "vision_read")
            self.assertEqual(calls[-1][0], "vision_read")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("force_visual_after_no_evidence_growth", trace)

    def test_navigation_only_mcq_round_forces_uninspected_visual_when_no_option_support(self):
        planner_responses = [
            '{"status": "continue", "program": [{"tool": "search_segments", "args": {"query": "first localization"}, "assign": "map1"}]}',
            '{"status": "continue", "program": [{"tool": "search_segments", "args": {"query": "repeat localization"}, "assign": "map2"}]}',
        ]
        registry = ToolRegistry()
        calls = []

        @tool(name="search_segments", description="Cheap search that adds no answer evidence.")
        def search_segments(query: str, top_k: int = 5):
            calls.append(("search_segments", query))
            return {"claim": f"search only: {query}", "confidence": 1.0}

        @tool(name="vision_read", description="Visual read that creates answer evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append(("vision_read", segment_id, start_sec, end_sec))
            return {
                "claim": f"{segment_id} visual evidence",
                "confidence": 0.74,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(search_segments)
        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_sweep")
            agent = IterativeVisualAgent(
                backend=ScriptedPlannerBackend(planner_responses),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=2,
                    reserve_final_round=False,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(
                question="Which option is visible?\nA. red object\nB. blue object",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.rounds[0].program[0]["tool"], "search_segments")
            self.assertEqual(result.rounds[1].program[0]["tool"], "vision_read")
            self.assertEqual(calls[-1], ("vision_read", "seg_0001", 0.0, 20.0))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("force_uninspected_visual_without_option_support", trace)

    def test_mcq_full_segment_sweep_hands_off_to_answer_agent_before_budget_end(self):
        class FullSweepAnswerBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "search_segments", "args": {"query": "still searching"}, "assign": "map"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "A. red object", "rationale": "obs_0002 has the local visual read.", '
                            '"citations": ["obs_0002"], "missing_evidence": [], "confidence": 0.84}'
                        )
                    )
                raise AssertionError(request.task)

        registry = ToolRegistry()
        calls = []

        @tool(name="search_segments", description="Cheap search that adds no answer evidence.")
        def search_segments(query: str, top_k: int = 5):
            calls.append(("search_segments", query))
            return {"claim": f"search only: {query}", "confidence": 1.0}

        @tool(name="vision_read", description="Visual read that creates answer evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append(("vision_read", segment_id))
            return {
                "claim": "A red object is visible in the only segment.",
                "confidence": 0.84,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(search_segments)
        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=20.0)
        backend = FullSweepAnswerBackend()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_full_sweep_answer")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=5,
                    reserve_final_round=True,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(
                question="Which option is visible?\nA. red object\nB. blue object",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("A"))
            self.assertEqual(result.citations, ["obs_0002"])
            self.assertEqual([call[0] for call in calls], ["search_segments", "vision_read"])
            self.assertEqual([request.task for request in backend.requests], ["replan", "replan", "replan", "answer_from_evidence"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"source": "all_segments_inspected"', trace)

    def test_model_rewritten_mcq_is_used_for_planner_and_tools_only(self):
        raw_question = (
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: What's the main idea of the video?\n"
            "Options:\n"
            "A. The fall of Rome\n"
            "B. Why the Austro-Hungarian Empire was divided\n"
            "C. A battle timeline\n"
            "D. How the Austro-Hungarian Empire rises and falls.\n"
            "Select option A, B, C, or D."
        )
        raw_option_texts = (
            "The fall of Rome",
            "Why the Austro-Hungarian Empire was divided",
            "A battle timeline",
            "How the Austro-Hungarian Empire rises and falls.",
        )
        rewritten = (
            "Describe the overall topic and narrative arc of the video. Identify how the Austro-Hungarian Empire "
            "is covered, including time span, "
            "major stages, and whether it covers origin, growth, stability, decline, collapse, causes, or consequences."
        )
        testcase = self

        class RewriteFlowBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "rewrite_exploration_question":
                    testcase.assertIn("A. The fall of Rome", request.prompt)
                    return BackendResponse(
                        text=(
                            '{"exploration_question":"'
                            + rewritten
                            + '","focus_points":["narrative arc"],"target_entities":["Austro-Hungarian Empire"]}'
                        )
                    )
                if request.task == "replan":
                    testcase.assertIn(rewritten, request.prompt)
                    testcase.assertNotIn("Options:", request.prompt)
                    for label in ("A.", "B.", "C.", "D."):
                        testcase.assertNotIn(label, request.prompt)
                    for option_text in raw_option_texts:
                        testcase.assertNotIn(option_text, request.prompt)
                    testcase.assertNotIn("candidate_options", request.prompt)
                    return BackendResponse(
                        text=(
                            '{"status":"continue","program":[{"tool":"vision_read","args":{"segment_id":"seg_0001",'
                            '"ask_for":"Inspect option B. Why the Austro-Hungarian Empire was divided"},'
                            '"assign":"v1"}]}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    testcase.assertIn("A. The fall of Rome", request.prompt)
                    testcase.assertIn("D. How the Austro-Hungarian Empire rises and falls.", request.prompt)
                    return BackendResponse(
                        text='{"answer":"D. How the Austro-Hungarian Empire rises and falls.","citations":["obs_0001"],"confidence":0.86}'
                    )
                raise AssertionError(request.task)

        registry = ToolRegistry()
        calls = []

        @tool(name="vision_read", description="Read visible facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append({"ask_for": ask_for, "event_label": event_label})
            return {
                "claim": "The segment describes the Austro-Hungarian Empire across formation, stability, decline, and collapse.",
                "confidence": 0.86,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)
        backend = RewriteFlowBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="rewrite_mcq_flow")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=2,
                    reserve_final_round=True,
                    rewrite_mcq_for_exploration=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(question=raw_question, video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            self.assertEqual(result.question, raw_question)
            self.assertEqual(rewritten, calls[0]["ask_for"])
            self.assertNotIn("option B", calls[0]["ask_for"])
            self.assertNotIn("The fall of Rome", calls[0]["ask_for"])
            self.assertNotIn("Why the Austro-Hungarian Empire was divided", calls[0]["ask_for"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("mcq_exploration_question_rewrite", trace)

    def test_option_blind_answer_feedback_strips_option_labels_and_full_text(self):
        raw_question = (
            "Question: What's the main idea?\n"
            "Options:\n"
            "A. The fall of Rome\n"
            "B. Why the Austro-Hungarian Empire was divided\n"
            "C. A battle timeline\n"
            "D. How the Austro-Hungarian Empire rises and falls.\n"
        )

        feedback = _sanitize_option_blind_feedback(
            [
                "Need evidence for option B. Why the Austro-Hungarian Empire was divided, not A. The fall of Rome.",
                "need timestamp evidence",
            ],
            raw_question=raw_question,
        )

        self.assertEqual(feedback[0], "Resolve the remaining evidence gap with factual observations.")
        self.assertEqual(feedback[1], "need timestamp evidence")
        joined = "\n".join(feedback)
        self.assertNotIn("option B", joined)
        self.assertNotIn("A. The fall of Rome", joined)
        self.assertNotIn("Why the Austro-Hungarian Empire was divided", joined)

    def test_repeated_empty_program_finalizes_from_structured_evidence_before_stopping(self):
        class ShouldNotAnswerBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    raise AssertionError("structured table should arbitrate without backend")
                return super().generate(request)

        responses = [
            '{"status": "continue", "program": []}',
            '{"status": "continue", "program": []}',
            '{"status": "continue", "program": []}',
        ]
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="repeated_empty_finalizes")
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_scene_coverage_1",
                    "obs_id": "scene_coverage_seg_0001",
                    "tool": "timeline_asr_summary",
                    "supported_option": "D",
                    "claim": "Early transcript covers the rise and formation of the empire.",
                    "confidence": 0.84,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.84}],
                    "time_range": [0.0, 20.0],
                }
            )
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_scene_coverage_2",
                    "obs_id": "scene_coverage_seg_0003",
                    "tool": "timeline_asr_summary",
                    "supported_option": "D",
                    "claim": "Late transcript covers decline, collapse, and the fall of the empire.",
                    "confidence": 0.84,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.84}],
                    "time_range": [40.0, 60.0],
                }
            )
            agent = IterativeVisualAgent(
                backend=ShouldNotAnswerBackend(responses),
                registry=ToolRegistry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=6, reserve_final_round=True, max_repeated_programs=1),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "B. Why the empire was divided\n"
                    "D. How the empire rose and fell"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D. How the empire rose and fell")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("source\": \"repeated_program_guard", trace)

    def test_navigation_only_no_growth_forces_visual_read_on_requested_segment(self):
        class NavThenAnswerBackend(VisionLanguageBackend):
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
                                '{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "initial read"}, '
                                '"assign": "v1"}]}'
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "expand_window", "args": {"segment_id": "seg_0001", "before_sec": 30, "after_sec": 30}, '
                            '"assign": "nav"}]}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["more visual facts"]}')
                return BackendResponse(text="unexpected")

        registry = ToolRegistry()
        calls = []

        @tool(name="vision_read", description="Read a segment.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, **kwargs):
            calls.append(("vision_read", segment_id, ask_for))
            return {
                "claim": f"visual read {len(calls)}",
                "confidence": 0.8,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="expand_window", description="Expand a segment.")
        def expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0):
            calls.append(("expand_window", segment_id, ""))
            return {"claim": f"expanded {segment_id}", "confidence": 1.0}

        registry.register(vision_read)
        registry.register(expand_window)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="nav_to_visual")
            agent = IterativeVisualAgent(
                backend=NavThenAnswerBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=4, reserve_final_round=False, max_repeated_programs=0),
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertIn(("vision_read", "seg_0001", "What happens?"), calls)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("force_visual_from_navigation_no_growth", trace)

    def test_answer_agent_final_trace_includes_scene_index_citation_provenance(self):
        class ProvenanceBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "D. Rome rose and fell", "citations": ["scene_order_seg_0001"], '
                            '"confidence": 0.86, "candidate_option_relations": ['
                            '{"option": "D", "relation": "support", "strength": 0.86, '
                            '"observation_id": "scene_order_seg_0001"}]}'
                        )
                    )
                return super().generate(request)

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    source_segment_id="dual_seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    asr_summary="Rome rose and later fell.",
                    visual_caption="Historical maps and narration.",
                    visual_caption_source="caption_scene_segment:vl-mini",
                    raw_asr_ref="cue-1,cue-2",
                    citation_provenance={"asr": "subtitle", "visual": "video"},
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="final_citation_provenance")
            workspace.write_evidence_row(
                {
                    "obs_id": "scene_order_seg_0001",
                    "tool": "timeline_asr_summary",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 30.0],
                    "supported_option": "D",
                    "claim": "Indexed segment supports the rise and fall sequence.",
                    "confidence": 0.86,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.86}],
                }
            )
            agent = IterativeVisualAgent(
                backend=ProvenanceBackend(
                    ['{"status": "final", "answer": "D. Rome rose and fell", "citations": ["scene_order_seg_0001"]}']
                ),
                registry=ToolRegistry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "What sequence is described?\n"
                    "Options:\n"
                    "A. Rome only rises\n"
                    "D. Rome rose and fell\n"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"citation_provenance"', trace)
            self.assertIn('"source_segment_id": "dual_seg_0001"', trace)
            self.assertIn('"raw_asr_ref": "cue-1,cue-2"', trace)
            self.assertIn('"visual_caption_source": "caption_scene_segment:vl-mini"', trace)

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
        self.assertEqual(backend.requests[0].metadata["question"], "What is visible? Do not choose an option.")

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
