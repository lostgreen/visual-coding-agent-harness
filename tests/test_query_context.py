import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.distill import distill
from visual_coding_agent_harness.evidence_predicates import grounding_quality_floor
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.tools.exploration import build_video_exploration_registry
from visual_coding_agent_harness.tools.query_context import build_query_context_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceRecord, EvidenceWorkspace


MCQ_OPTIONS = [
    "The fall of Rome",
    "Why the Austro-Hungarian Empire was divided",
    "A battle timeline",
    "How the Austro-Hungarian Empire rose and fell",
]
MCQ_QUESTION = (
    "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
    "Question: What is the video mainly about?\n"
    "Options:\n"
    "A. The fall of Rome\n"
    "B. Why the Austro-Hungarian Empire was divided\n"
    "C. A battle timeline\n"
    "D. How the Austro-Hungarian Empire rose and fell\n"
    "Select option A, B, C, or D."
)


def assert_no_mcq_leak(testcase: unittest.TestCase, prompt: str, option_texts=MCQ_OPTIONS) -> None:
    text = str(prompt)
    testcase.assertNotIn("Options:", text)
    testcase.assertNotIn("Candidate options:", text)
    for label in ("A.", "B.", "C.", "D."):
        testcase.assertNotIn(label, text)
    testcase.assertNotRegex(text, r"\boption\s+[A-D]\b")
    for option in option_texts:
        testcase.assertNotIn(option, text)


class QueryContextBackend(VisionLanguageBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text="The aircraft appears in museum-like scenes.")


def _video_map() -> VideoMap:
    return VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=120.0,
        segments=[
            VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=40.0, low_fps_caption="opening"),
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=40.0,
                end_sec=80.0,
                low_fps_caption="A blue aircraft is shown in a museum.",
                entities=["aircraft", "museum"],
            ),
        ],
    )


class QueryContextTest(unittest.TestCase):
    def test_query_context_default_128_and_no_option_vote(self):
        backend = QueryContextBackend()
        registry = build_query_context_registry(video_map=_video_map(), backend=backend)

        result = registry.execute(
            "query_context",
            {"video_path": "/videos/demo.mp4", "query": "Which scene has aircraft?", "duration_sec": 120.0},
        )

        self.assertEqual(backend.requests[0].metadata["nframes"], 128)
        self.assertEqual(result["grounding_quality"], "query_global_context")
        self.assertNotIn("supported_option", result)
        self.assertIn("not sole support", result["limitations"])

    def test_query_context_sanitizes_full_mcq_before_backend_generate(self):
        backend = QueryContextBackend()
        registry = build_query_context_registry(video_map=_video_map(), backend=backend)

        registry.execute(
            "query_context",
            {"video_path": "/videos/demo.mp4", "query": MCQ_QUESTION, "duration_sec": 120.0},
        )

        request = backend.requests[0]
        self.assertEqual(request.metadata["original_query"], MCQ_QUESTION)
        self.assertIn("What is the video mainly about?", request.prompt)
        self.assertIn("Do not choose an option.", request.prompt)
        assert_no_mcq_leak(self, request.prompt)

    def test_query_context_cannot_be_sole_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="query_context_floor")
            observation = workspace.write_observation(
                tool_name="query_context",
                input_artifacts=["/videos/demo.mp4"],
                claim="The aircraft appears in museum-like scenes.",
                confidence=0.62,
                regions=[{"start_sec": 0.0, "end_sec": 120.0}],
                limitations="Global query context capsule.",
                raw_output={"grounding_quality": "query_global_context"},
            )
            distilled = distill(observation, workspace)
            for record in distilled:
                workspace.write_evidence(record)
            mapped = EvidenceRecord(
                evidence_id=workspace.next_evidence_id("mapped"),
                stage="mapped",
                parent_id=distilled[0].evidence_id,
                tool="query_context",
                observation_id=observation.observation_id,
                frame_set_id="fs_query_context_floor_00001",
                content={"option": "B", "relation": "support"},
                grounding_quality="query_global_context",
                confidence=0.62,
                created_at=1.0,
            )
            workspace.write_evidence(mapped)

            reason = grounding_quality_floor([mapped], workspace=workspace, require_visual=True)

            self.assertIn("no visually_confirmed", reason or "")

    def test_query_context_rows_do_not_enter_answer_support_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="query_context_table")
            observation = workspace.write_observation(
                tool_name="query_context",
                input_artifacts=["/videos/demo.mp4"],
                claim="A. The aircraft appears in museum-like scenes.",
                confidence=0.62,
                regions=[{"start_sec": 0.0, "end_sec": 120.0}],
                limitations="Global query context capsule.",
                raw_output={"grounding_quality": "query_global_context", "supported_option": "A"},
            )
            workspace.write_ledger_entry(observation)

            table = workspace.evidence_table_v2(
                question="Which option is shown?",
                options=["A. aircraft", "B. cooking"],
            )
            ledger_text = workspace.compact_ledger_text()

            self.assertNotIn(observation.observation_id, [row["obs_id"] for row in table["rows"]])
            self.assertFalse(workspace.has_non_navigation_visual_citation([observation.observation_id]))
            self.assertIn("Context-Only Visual Hints (Not Answer Support)", ledger_text)
            self.assertIn("context hint only; not answer support", ledger_text)
            self.assertNotIn("Long-Term Visual Evidence\n- `obs_0001`", ledger_text)
            self.assertNotIn("Short-Term Working Buffer\n- `obs_0001`", ledger_text)

    def test_route_relevant_query_context_creates_segment_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="query_context_proposal")
            backend = QueryContextBackend()
            registry = build_video_exploration_registry(video_map=_video_map(), backend=backend, workspace=workspace)
            interpreter = ProgramInterpreter(registry=registry, workspace=workspace)

            interpreter.run(
                [
                    {
                        "tool": "query_context",
                        "args": {
                            "video_path": "/videos/demo.mp4",
                            "query": "aircraft museum",
                            "duration_sec": 120.0,
                            "scope": "route_relevant",
                        },
                    }
                ]
            )
            proposals = workspace.load_pending_proposals()

            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].target_segment_id, "seg_0002")
            self.assertEqual(proposals[0].source_frame_set_id, "fs_query_context_proposal_00001")


if __name__ == "__main__":
    unittest.main()
