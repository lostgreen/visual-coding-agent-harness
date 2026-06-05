import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class RecordingBackend(VisionLanguageBackend):
    def __init__(self, text="D. The global sparse view supports the aviation-documentary synopsis."):
        self.text = text
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.text)


class GlobalViewToolTest(unittest.TestCase):
    def test_global_gist_samples_whole_video_and_records_global_sparse_support(self):
        from visual_coding_agent_harness.tools.global_view import build_global_view_registry

        backend = RecordingBackend()
        registry = build_global_view_registry(backend)

        result = registry.execute(
            "global_gist",
            {
                "video_path": "/videos/long.mp4",
                "question": "What is the video mainly about?\nA. one\nB. two\nC. three\nD. four",
                "duration_sec": 1896.0,
            },
        )

        self.assertEqual(backend.requests[0].task, "global_gist")
        self.assertEqual(backend.requests[0].media_path, "/videos/long.mp4")
        self.assertEqual(backend.requests[0].metadata["nframes"], 128)
        self.assertEqual(result["raw_output"]["supported_option"], "D")
        self.assertEqual(result["raw_output"]["grounding_quality"], "global_sparse")
        self.assertEqual(result["regions"][0]["start_sec"], 0.0)
        self.assertEqual(result["regions"][0]["end_sec"], 1896.0)

    def test_global_gist_is_first_class_answer_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="global_table")
            observation = workspace.write_observation(
                tool_name="global_gist",
                claim="Supported option: D. The whole video is an aviation documentary.",
                confidence=0.76,
                regions=[{"start_sec": 0.0, "end_sec": 1896.0}],
                limitations="Sparse full-video sampling.",
                raw_output={
                    "supported_option": "D",
                    "grounding_quality": "global_sparse",
                    "time_range": [0.0, 1896.0],
                },
            )
            workspace.write_ledger_entry(observation)

            table = workspace.evidence_table(
                question="What is the video mainly about?",
                options=["A. one", "B. two", "C. three", "D. aviation documentary"],
            )

            d_rows = table["groups"]["D"]
            self.assertEqual(len(d_rows), 1)
            self.assertEqual(d_rows[0]["tool"], "global_gist")
            self.assertEqual(d_rows[0]["grounding_quality"], "global_sparse")


if __name__ == "__main__":
    unittest.main()
