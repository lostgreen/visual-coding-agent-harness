import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.comparison import DescriptionComparisonConfig, run_description_comparison, run_qwen_description_comparison
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment


class DescriptionComparisonBackend(VisionLanguageBackend):
    def __init__(self):
        self.requests = []
        self.replan_calls = 0

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "direct_description":
            return BackendResponse(text="Direct baseline describes only the opening.")
        if request.task == "replan":
            self.replan_calls += 1
            if self.replan_calls == 1:
                return BackendResponse(
                    text=(
                        '{"status": "continue", "program": ['
                        '{"tool": "video_ls", "args": {"query": "describe aircraft", "max_segments": 3}, "assign": "map"}'
                        "]}"
                    )
                )
            if self.replan_calls == 2:
                return BackendResponse(
                    text=(
                        '{"status": "continue", "program": ['
                        '{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "Describe this segment."}, "assign": "detail"}'
                        "]}"
                    )
                )
            return BackendResponse(
                text=(
                    '{"status": "final", "answer": "Exploration describes the aircraft museum section.", '
                    '"citations": ["obs_0001", "obs_0002"], "confidence": 0.82}'
                )
            )
        if request.task == "caption_segment":
            return BackendResponse(text="The segment contains aircraft museum evidence.")
        return BackendResponse(text="")


class DescriptionComparisonTest(unittest.TestCase):
    def test_comparison_runner_records_direct_and_map_first_exploration_outputs(self):
        backend = DescriptionComparisonBackend()
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="aircraft museum"),
                VideoSegment(segment_id="seg_0003", start_sec=60.0, end_sec=90.0, low_fps_caption="ending"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_description_comparison(
                base_dir=Path(tmp),
                backend=backend,
                media_path="/videos/demo.mp4",
                duration_sec=90.0,
                scene_index=scene_index,
                run_id="compare",
                budget=AgentBudget(max_rounds=3),
            )

            self.assertEqual(result.direct_answer, "Direct baseline describes only the opening.")
            self.assertEqual(result.exploration_result.answer, "Exploration describes the aircraft museum section.")
            self.assertEqual([request.task for request in backend.requests], ["direct_description", "replan", "replan", "caption_segment", "replan"])
            self.assertEqual(backend.requests[0].metadata["nframes"], 64)
            self.assertEqual(backend.requests[0].metadata["max_pixels"], 151200)
            report_path = Path(tmp) / "runs" / "compare" / "comparison.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["strategies"][0]["name"], "direct_full_video")
            self.assertEqual(report["strategies"][1]["name"], "map_first_explore")

    def test_qwen_comparison_wrapper_loads_one_shared_backend(self):
        backend = DescriptionComparisonBackend()
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="aircraft museum"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("visual_coding_agent_harness.backends.qwen_vl.QwenVLBackend.from_pretrained", return_value=backend) as loader:
                result = run_qwen_description_comparison(
                    DescriptionComparisonConfig(
                        model_path="/models/qwen",
                        media_path="/videos/demo.mp4",
                        question="Describe the video.",
                        duration_sec=60.0,
                        run_id="qwen_compare",
                        max_rounds=3,
                    ),
                    base_dir=Path(tmp),
                    scene_index=scene_index,
                )

            loader.assert_called_once_with("/models/qwen")
            self.assertEqual(result.direct_answer, "Direct baseline describes only the opening.")
            self.assertEqual(result.exploration_result.status, "final")


if __name__ == "__main__":
    unittest.main()
