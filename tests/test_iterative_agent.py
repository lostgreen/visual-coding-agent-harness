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
from visual_coding_agent_harness.video_map import VideoMap
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
    return registry


class IterativeAgentTest(unittest.TestCase):
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
            self.assertIn("Do not spend every round on navigation-only tools", prompt)

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
            self.assertEqual(tool_args["nframes"], 8)

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
                budget=AgentBudget(max_rounds=1),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertIn("Partial evidence summary", result.answer)
            self.assertIn("aircraft history", result.answer)

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
        self.assertEqual(
            backend.requests[0].metadata,
            {"segment_id": "seg_0002", "start_sec": 10.0, "end_sec": 20.0, "nframes": 12},
        )

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
