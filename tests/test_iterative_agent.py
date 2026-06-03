import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.iterative_smoke import run_iterative_smoke
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.tools.segments import build_segment_vlm_registry
from visual_coding_agent_harness.video_index import VideoSegment, SceneIndex, fixed_window_scene_index
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
            self.assertIn("Stopped after 1 exploration round", result.answer)

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
