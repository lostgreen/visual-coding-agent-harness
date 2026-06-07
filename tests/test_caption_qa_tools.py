import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.tools.enrichment import build_video_enrichment_registry
from visual_coding_agent_harness.tools.inspector import build_segment_inspector_registry
from visual_coding_agent_harness.tools.segments import build_segment_vlm_registry
from visual_coding_agent_harness.tools.vlm import build_vlm_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.workspace import EvidenceWorkspace
from tests.test_open_questions import MCQ_OPTIONS, MCQ_QUESTION, assert_no_mcq_leak


class CaptionQARecordingBackend(VisionLanguageBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=f"{request.task} answer", raw={"task": request.task})


class CaptionQAToolsTest(unittest.TestCase):
    def test_caption_and_qa_prompts_are_structured_by_task(self):
        backend = CaptionQARecordingBackend()
        registry = build_vlm_registry(backend)

        caption = registry.execute("caption_video", {"video_path": "/videos/demo.mp4", "question": "Describe the scene.", "nframes": 12})
        qa = registry.execute("qa_video", {"video_path": "/videos/demo.mp4", "question": "What vehicle appears?", "nframes": 12})

        self.assertIn("visible evidence", backend.requests[0].prompt)
        self.assertIn("Do not invent details", backend.requests[0].prompt)
        self.assertIn("Caption task", backend.requests[0].prompt)
        self.assertIn("QA task", backend.requests[1].prompt)
        self.assertIn("What vehicle appears?", backend.requests[1].prompt)
        self.assertEqual(caption["regions"][0]["task"], "caption_video")
        self.assertEqual(qa["regions"][0]["task"], "qa_video")
        self.assertEqual(qa["regions"][0]["nframes"], 12)

    def test_region_caption_and_qa_crop_artifacts_before_vlm_call(self):
        backend = CaptionQARecordingBackend()
        crops = []

        def fake_cropper(image_path, bbox, output_path):
            crops.append((image_path, list(bbox), output_path))
            Path(output_path).write_text("fake crop", encoding="utf-8")
            return {
                "claim": "crop created",
                "confidence": 1.0,
                "input_artifacts": [image_path],
                "regions": [{"bbox": list(bbox), "output_path": output_path}],
            }

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="region_tools")
            registry = build_vlm_registry(backend, workspace=workspace, cropper=fake_cropper)

            result = registry.execute(
                "qa_region",
                {
                    "image_path": "/images/frame.jpg",
                    "bbox": [100, 200, 400, 500],
                    "question": "What text is visible?",
                },
            )

            self.assertEqual(len(crops), 1)
            self.assertIn("region_100_200_400_500.png", crops[0][2])
            self.assertEqual(backend.requests[0].task, "qa_region")
            self.assertEqual(backend.requests[0].media_path, crops[0][2])
            self.assertEqual(backend.requests[0].media_type, "image")
            self.assertIn("focus only on the cropped region", backend.requests[0].prompt)
            self.assertEqual(result["input_artifacts"], [crops[0][2]])
            self.assertEqual(result["regions"][0]["bbox"], [100, 200, 400, 500])
            self.assertEqual(result["regions"][0]["source_image_path"], "/images/frame.jpg")

    def test_region_tools_require_workspace_for_crop_artifacts(self):
        registry = build_vlm_registry(CaptionQARecordingBackend())

        with self.assertRaises(ValueError) as context:
            registry.execute(
                "caption_region",
                {
                    "image_path": "/images/frame.jpg",
                    "bbox": [0, 0, 100, 100],
                },
            )

        self.assertIn("workspace", str(context.exception))

    def test_segment_qa_accepts_video_sampling_controls(self):
        backend = CaptionQARecordingBackend()
        registry = build_segment_vlm_registry(backend)

        result = registry.execute(
            "qa_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0004",
                "start_sec": 30.0,
                "end_sec": 42.0,
                "question": "What happens?",
                "nframes": 16,
                "max_pixels": 200000,
                "fps": 2.0,
            },
        )

        request = backend.requests[0]
        self.assertEqual(request.metadata["nframes"], 64)
        self.assertEqual(request.metadata["max_pixels"], 200000)
        self.assertEqual(request.metadata["fps"], 2.0)
        self.assertIn("QA task", request.prompt)
        self.assertEqual(result["regions"][0]["max_pixels"], 200000)

    def test_caption_and_qa_segment_sanitize_full_mcq_before_backend_generate(self):
        backend = CaptionQARecordingBackend()
        registry = build_segment_vlm_registry(backend)

        registry.execute(
            "caption_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0001",
                "start_sec": 0.0,
                "end_sec": 30.0,
                "question": MCQ_QUESTION,
            },
        )
        registry.execute(
            "qa_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0002",
                "start_sec": 30.0,
                "end_sec": 60.0,
                "question": MCQ_QUESTION,
            },
        )

        for request in backend.requests:
            self.assertEqual(request.metadata["original_question"], MCQ_QUESTION)
            self.assertIn("What is the video mainly about?", request.prompt)
            self.assertIn("Do not choose an option.", request.prompt)
            assert_no_mcq_leak(self, request.prompt)

    def test_caption_segments_sanitizes_full_mcq_before_backend_generate(self):
        backend = CaptionQARecordingBackend()
        store = VideoMapStore(
            VideoMap(
                video_path="/videos/demo.mp4",
                duration_sec=40.0,
                segments=[VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=40.0)],
            )
        )
        registry = build_video_enrichment_registry(video_map_store=store, backend=backend)

        registry.execute("caption_segments", {"segment_ids": ["seg_0001"], "question": MCQ_QUESTION})

        request = backend.requests[0]
        self.assertEqual(request.metadata["original_question"], MCQ_QUESTION)
        self.assertIn("What is the video mainly about?", request.prompt)
        self.assertIn("Do not choose an option.", request.prompt)
        assert_no_mcq_leak(self, request.prompt)

    def test_segment_inspector_returns_one_distilled_observation(self):
        backend = CaptionQARecordingBackend()
        registry = build_segment_inspector_registry(backend)

        result = registry.execute(
            "inspect_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0004",
                "start_sec": 30.0,
                "end_sec": 42.0,
                "question": "Which option is supported?",
                "candidate_options": ["A. aircraft", "B. submarine"],
                "nframes": 20,
            },
        )

        request = backend.requests[0]
        self.assertEqual(request.task, "inspect_segment")
        self.assertEqual(request.media_path, "/videos/demo.mp4")
        self.assertEqual(request.metadata["candidate_options"], ["A. aircraft", "B. submarine"])
        self.assertIn("You are a Segment Inspector subagent", request.prompt)
        self.assertIn("Return one distilled local observation", request.prompt)
        self.assertIn("Do not choose an option", request.prompt)
        self.assertIn("Do not emit supported_option", request.prompt)
        self.assertNotIn("supported option if any", request.prompt)
        assert_no_mcq_leak(self, request.prompt, option_texts=["aircraft", "submarine"])
        self.assertEqual(result["claim"], "inspect_segment answer")
        self.assertNotIn("supported_option", result)
        self.assertNotIn("answer_option", result)
        self.assertEqual(result["regions"][0]["tool_role"], "segment_inspector")
        self.assertEqual(result["regions"][0]["segment_id"], "seg_0004")

    def test_inspect_segment_sanitizes_full_mcq_question_into_fact_request(self):
        backend = CaptionQARecordingBackend()
        registry = build_segment_inspector_registry(backend)
        full_mcq = (
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: As depicted in the video, in what order do these sculptures appear?\n"
            "Options:\n"
            'A. "The rape of Persephone", "Apollo and Daphne", "David".\n'
            'B. "David", "Aeneas", "Apollo and Daphne".\n'
            'C. "Apollo and Daphne", "Aeneas", "David".\n'
            'D. "Aeneas", "David", "The rape of Persephone".'
        )

        result = registry.execute(
            "inspect_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0003",
                "start_sec": 600.0,
                "end_sec": 900.0,
                "question": full_mcq,
                "candidate_options": [
                    'A. "The rape of Persephone", "Apollo and Daphne", "David".',
                    'B. "David", "Aeneas", "Apollo and Daphne".',
                    'C. "Apollo and Daphne", "Aeneas", "David".',
                    'D. "Aeneas", "David", "The rape of Persephone".',
                ],
                "nframes": 20,
            },
        )

        request = backend.requests[0]
        self.assertIn("As depicted in the video, in what order do these sculptures appear?", request.prompt)
        self.assertIn("Do not choose an option.", request.prompt)
        assert_no_mcq_leak(
            self,
            request.prompt,
            option_texts=[
                '"The rape of Persephone", "Apollo and Daphne", "David"',
                '"David", "Aeneas", "Apollo and Daphne"',
                '"Apollo and Daphne", "Aeneas", "David"',
                '"Aeneas", "David", "The rape of Persephone"',
            ],
        )
        self.assertNotIn("supported_option", result)

    def test_vision_read_emits_typed_fact_without_option_vote(self):
        backend = CaptionQARecordingBackend()
        registry = build_segment_inspector_registry(backend)

        result = registry.execute(
            "vision_read",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0004",
                "start_sec": 30.0,
                "end_sec": 42.0,
                "ask_for": "presence and timestamp of: red object",
                "event_label": "red object",
                "nframes": 20,
            },
        )

        request = backend.requests[0]
        self.assertEqual(request.task, "vision_read")
        self.assertIn("Return typed visual facts", request.prompt)
        self.assertNotIn("supported option if any", request.prompt)
        self.assertNotIn("supported_option", result)
        self.assertNotIn("answer_option", result)
        self.assertEqual(result["event_label"], "red object")
        self.assertEqual(result["time_range"], [30.0, 42.0])
        self.assertEqual(result["grounding_quality"], "visually_confirmed")
        self.assertEqual(result["facts"][0]["event_label"], "red object")
        self.assertEqual(result["facts"][0]["time_range"], [30.0, 42.0])

    def test_vision_read_sanitizes_full_mcq_into_fact_request(self):
        backend = CaptionQARecordingBackend()
        registry = build_segment_inspector_registry(backend)
        full_mcq = (
            "What is the video mainly about?\n"
            "A. The fall of Rome\n"
            "B. Why the Austro-Hungarian Empire was divided\n"
            "C. A battle timeline\n"
            "D. How the Austro-Hungarian Empire rose and fell"
        )

        result = registry.execute(
            "vision_read",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0007",
                "start_sec": 360.0,
                "end_sec": 420.0,
                "ask_for": full_mcq,
                "nframes": 20,
            },
        )

        request = backend.requests[0]
        self.assertIn("What is the video mainly about?", request.prompt)
        self.assertIn("Do not choose an option.", request.prompt)
        assert_no_mcq_leak(
            self,
            request.prompt,
            option_texts=[
                "The fall of Rome",
                "Why the Austro-Hungarian Empire was divided",
                "A battle timeline",
                "How the Austro-Hungarian Empire rose and fell",
            ],
        )
        self.assertNotIn("supported_option", result)


if __name__ == "__main__":
    unittest.main()
