import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.distill import distill
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.workspace import EvidenceWorkspace
from tests.test_open_questions import MCQ_QUESTION, assert_no_mcq_leak


class RecordingBackend(VisionLanguageBackend):
    def __init__(self, text="D. The global sparse view suggests an aviation-documentary synopsis."):
        self.text = text
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.text)


class GlobalViewToolTest(unittest.TestCase):
    def test_global_gist_samples_whole_video_without_option_vote(self):
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
        self.assertNotIn("Start multiple-choice answers", backend.requests[0].prompt)
        self.assertNotIn("supported_option", result["raw_output"])
        self.assertEqual(result["raw_output"]["grounding_quality"], "global_sparse")
        self.assertEqual(result["raw_output"]["candidate_option_hint"], "D")
        self.assertEqual(result["raw_output"]["candidate_option_relations"], [])
        self.assertEqual(result["regions"][0]["start_sec"], 0.0)
        self.assertEqual(result["regions"][0]["end_sec"], 1896.0)

    def test_global_gist_sanitizes_full_mcq_before_backend_generate(self):
        from visual_coding_agent_harness.tools.global_view import build_global_view_registry

        backend = RecordingBackend()
        registry = build_global_view_registry(backend)

        registry.execute(
            "global_gist",
            {"video_path": "/videos/long.mp4", "question": MCQ_QUESTION, "duration_sec": 1896.0},
        )

        request = backend.requests[0]
        self.assertEqual(request.metadata["original_question"], MCQ_QUESTION)
        self.assertIn("What is the video mainly about?", request.prompt)
        self.assertIn("Do not choose an option.", request.prompt)
        assert_no_mcq_leak(self, request.prompt)

    def test_global_gist_is_unassigned_sparse_answer_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="global_table")
            observation = workspace.write_observation(
                tool_name="global_gist",
                claim="D. The whole video appears to be an aviation documentary.",
                confidence=0.76,
                regions=[{"start_sec": 0.0, "end_sec": 1896.0}],
                limitations="Sparse full-video sampling.",
                raw_output={
                    "candidate_option_hint": "D",
                    "grounding_quality": "global_sparse",
                    "time_range": [0.0, 1896.0],
                },
            )
            distilled_records = distill(observation, workspace)
            for record in distilled_records:
                workspace.write_evidence(record)
            workspace.write_ledger_entry(observation, parent_records=distilled_records)

            table = workspace.evidence_table(
                question="What is the video mainly about?",
                options=["A. one", "B. two", "C. three", "D. aviation documentary"],
            )
            chains = workspace.evidence_chain_summaries()

            self.assertEqual(table["groups"]["D"], [])
            unassigned_rows = table["groups"]["unassigned"]
            self.assertEqual(unassigned_rows, [])
            self.assertEqual(table["rows"], [])
            self.assertEqual(chains, [])

    def test_global_gist_claim_is_hidden_from_planner_working_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="global_context")
            observation = workspace.write_observation(
                tool_name="global_gist",
                claim=(
                    "B. Why the Austro-Hungarian Empire was divided. The video shows "
                    "ethnic group distribution followed by division."
                ),
                confidence=0.76,
                regions=[{"start_sec": 0.0, "end_sec": 1896.0}],
                limitations="Sparse full-video sampling.",
                raw_output={
                    "candidate_option_hint": "B",
                    "grounding_quality": "global_sparse",
                    "time_range": [0.0, 1896.0],
                },
            )
            distilled_records = distill(observation, workspace)
            for record in distilled_records:
                workspace.write_evidence(record)
            workspace.write_ledger_entry(observation, parent_records=distilled_records)

            context = workspace.compact_ledger_text()

            self.assertIn("Context-Only Visual Hints", context)
            self.assertIn("global_gist", context)
            self.assertIn("sparse topic hint", context)
            self.assertNotIn("B. Why the Austro-Hungarian Empire was divided", context)
            self.assertNotIn("ethnic group distribution followed by division", context)


if __name__ == "__main__":
    unittest.main()
