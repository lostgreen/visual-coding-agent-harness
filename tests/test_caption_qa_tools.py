import tempfile
import json
import unittest
from pathlib import Path
import pytest

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.contracts import TargetRegistry, TargetSpec
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.tools.enrichment import build_video_enrichment_registry
from visual_coding_agent_harness.tools.inspector import build_segment_inspector_registry
from visual_coding_agent_harness.tools.segments import build_segment_vlm_registry
from visual_coding_agent_harness.tools.vlm import build_vlm_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.workspace import EvidenceWorkspace


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


class CaptionQARecordingBackend(VisionLanguageBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=f"{request.task} answer", raw={"task": request.task})


class FixedTextBackend(VisionLanguageBackend):
    def __init__(self, text: str):
        self.text = text
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.text, raw={"task": request.task})


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
            self.assertNotIn("original_question", request.metadata)
            self.assertIn("What is the video mainly about?", request.prompt)
            self.assertIn("Do not choose an option.", request.prompt)
            assert_no_mcq_leak(self, request.prompt)

    def test_general_qa_video_sanitizes_full_mcq_before_backend_generate(self):
        backend = CaptionQARecordingBackend()
        registry = build_vlm_registry(backend)

        registry.execute("qa_video", {"video_path": "/videos/demo.mp4", "question": MCQ_QUESTION, "nframes": 12})

        request = backend.requests[0]
        self.assertIn("What is the video mainly about?", request.prompt)
        self.assertIn("Do not choose an option.", request.prompt)
        self.assertNotIn("original_question", request.metadata)
        assert_no_mcq_leak(self, request.prompt)
        assert_no_mcq_leak(self, request.metadata["question"])

    def test_segment_tool_marks_explicit_no_evidence_as_unsupported(self):
        backend = FixedTextBackend("The video does not depict Bernini's four masterpieces in this segment.")
        registry = build_segment_vlm_registry(backend)

        result = registry.execute(
            "caption_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0001",
                "start_sec": 0.0,
                "end_sec": 30.0,
            },
        )

        self.assertEqual(result["confidence_signal"], "unsupported")
        self.assertEqual(result["grounding_quality"], "inferred")

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
        self.assertNotIn("original_question", request.metadata)
        self.assertIn("What is the video mainly about?", request.prompt)
        self.assertIn("Do not choose an option.", request.prompt)
        assert_no_mcq_leak(self, request.prompt)

    def test_caption_segments_extracts_physical_clips_when_enabled(self):
        backend = CaptionQARecordingBackend()
        store = VideoMapStore(
            VideoMap(
                video_path="/videos/demo.mp4",
                duration_sec=80.0,
                segments=[
                    VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=40.0),
                    VideoMapSegment(segment_id="seg_0002", start_sec=40.0, end_sec=80.0),
                ],
            )
        )
        extracted = []

        def fake_clip_extractor(video_path, output_path, start_sec, end_sec):
            extracted.append((video_path, output_path, start_sec, end_sec))
            Path(output_path).write_text("fake clip", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="caption_segments_clip")
            registry = build_video_enrichment_registry(
                video_map_store=store,
                backend=backend,
                workspace=workspace,
                extract_clips=True,
                clip_extractor=fake_clip_extractor,
            )

            result = registry.execute("caption_segments", {"segment_ids": ["seg_0002"], "question": MCQ_QUESTION})

        request = backend.requests[0]
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0][0], "/videos/demo.mp4")
        self.assertEqual((extracted[0][2], extracted[0][3]), (40.0, 80.0))
        self.assertNotEqual(request.media_path, "/videos/demo.mp4")
        self.assertEqual(request.media_path, extracted[0][1])
        self.assertEqual(request.metadata["source_video_path"], "/videos/demo.mp4")
        self.assertEqual(request.metadata["clip_path"], extracted[0][1])
        self.assertEqual(result["input_artifacts"], [extracted[0][1]])
        self.assertEqual(result["regions"][0]["clip_path"], extracted[0][1])
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
        self.assertEqual(request.metadata["candidate_options"], [])
        self.assertNotIn("original_candidate_options", request.metadata)
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
        self.assertIn("ORDERED_VISIBLE", request.prompt)
        self.assertNotIn("supported option if any", request.prompt)
        self.assertNotIn("supported_option", result)
        self.assertNotIn("answer_option", result)
        self.assertEqual(result["event_label"], "red object")
        self.assertEqual(result["time_range"], [30.0, 42.0])
        self.assertEqual(result["grounding_quality"], "visually_confirmed")
        self.assertEqual(result["facts"][0]["event_label"], "red object")

    def test_vision_read_parses_ordered_visible_output(self):
        backend = FixedTextBackend("Visible targets are c, then a, then b.\nORDERED_VISIBLE: c -> a -> b")
        registry = build_segment_inspector_registry(backend)

        result = registry.execute(
            "vision_read",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0004",
                "start_sec": 30.0,
                "end_sec": 42.0,
                "ask_for": "visible order of c, a, b",
                "nframes": 20,
            },
        )

        self.assertIn("ORDERED_VISIBLE", backend.requests[0].prompt)
        self.assertEqual(result["ordered_visible_in_window"], ["c", "a", "b"])

    def test_verify_segment_anchors_parses_confirmations_into_evidence_and_timeline(self):
        backend = FixedTextBackend(
            json.dumps(
                {
                    "confirmations": [
                        {
                            "target": "David",
                            "relative_sec": 2.0,
                            "observed_at_sec": 432.0,
                            "evidence": "A sculpture identified as David is shown.",
                        }
                    ],
                    "rejections": [
                        {
                            "target": "Apollo and Daphne",
                            "reason": "Not visible in this anchor.",
                        }
                    ],
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_anchors")
            registry = build_segment_inspector_registry(backend, workspace=workspace)
            result = ProgramInterpreter(registry=registry, workspace=workspace).run(
                [
                    {
                        "tool": "verify_segment_anchors",
                        "args": {
                            "video_path": "/videos/bernini.mp4",
                            "segment_id": "seg_0002",
                            "start_sec": 300.0,
                            "end_sec": 600.0,
                            "question": "Determine artwork order.",
                            "anchors": [
                                {
                                    "anchor_id": "anchor_0001",
                                    "start_sec": 430.0,
                                    "end_sec": 448.0,
                                    "targets": ["David", "Apollo and Daphne"],
                                    "reason": "ASR lists targets here.",
                                }
                            ],
                        },
                    }
                ]
            )
            observation = workspace.read_observations(tool_name="verify_segment_anchors")[0]
            evidence_row_count = workspace.evidence_table_row_count()
            timeline_text = (workspace.root / "timeline.md").read_text(encoding="utf-8")

        request = backend.requests[0]
        self.assertEqual(result.observation_ids, ["obs_0001"])
        self.assertEqual(request.task, "verify_segment_anchors")
        self.assertEqual(request.metadata["nframes"], 8)
        self.assertEqual(request.metadata["segment_id"], "seg_0002")
        self.assertIn("ASR lists targets here", request.prompt)
        self.assertIn("relative seconds", request.prompt)
        self.assertIn("David", request.prompt)
        self.assertEqual(observation.raw_output["confirmations"][0]["target"], "David")
        self.assertEqual(observation.raw_output["rejections"][0]["target"], "Apollo and Daphne")
        self.assertEqual(observation.raw_output["timeline_rows"][0]["entity"], "David")
        self.assertEqual(evidence_row_count, 1)
        self.assertIn("David", timeline_text)

    def test_verify_segment_anchors_accepts_target_refs_from_registry(self):
        backend = FixedTextBackend("{}")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_target_refs")
            workspace.target_registry = TargetRegistry.from_specs(
                targets=[TargetSpec("T1", "humble background", subject="Goya")]
            )
            registry = build_segment_inspector_registry(backend, workspace=workspace)

            result = registry.execute(
                "verify_segment_anchors",
                {
                    "video_path": "/videos/goya.mp4",
                    "segment_id": "seg_0001",
                    "anchors": [],
                    "target_refs": ["T1"],
                },
            )

        self.assertEqual(result["targets"], ["humble background"])

    def test_verify_segment_anchors_parses_ordered_visible_into_timeline_order(self):
        backend = FixedTextBackend(
            json.dumps(
                {
                    "confirmations": [
                        {"target": "c", "evidence": "c is visible."},
                        {"target": "a", "evidence": "a is visible."},
                        {"target": "b", "evidence": "b is visible."},
                    ],
                    "rejections": [],
                }
            )
            + "\nORDERED_VISIBLE: c -> a -> b"
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_ordered_visible")
            registry = build_segment_inspector_registry(backend, workspace=workspace)
            ProgramInterpreter(registry=registry, workspace=workspace).run(
                [
                    {
                        "tool": "verify_segment_anchors",
                        "args": {
                            "video_path": "/videos/demo.mp4",
                            "segment_id": "seg_0001",
                            "start_sec": 10.0,
                            "end_sec": 25.0,
                            "anchors": [
                                {
                                    "anchor_id": "anchor_0001",
                                    "segment_id": "seg_0001",
                                    "start_sec": 10.0,
                                    "end_sec": 25.0,
                                    "targets": ["a", "b", "c"],
                                }
                            ],
                        },
                    }
                ]
            )
            observation = workspace.read_observations(tool_name="verify_segment_anchors")[0]
            timeline = workspace.read_timeline_sorted()

        self.assertIn("ORDERED_VISIBLE", backend.requests[0].prompt)
        self.assertEqual(observation.raw_output["ordered_visible_in_window"], ["c", "a", "b"])
        self.assertEqual([row["entity"] for row in observation.raw_output["timeline_rows"]], ["c", "a", "b"])
        self.assertEqual([row["entity"] for row in timeline], ["c", "a", "b"])
        self.assertLess(timeline[0]["observed_at_sec"], timeline[1]["observed_at_sec"])
        self.assertLess(timeline[1]["observed_at_sec"], timeline[2]["observed_at_sec"])

    def test_verify_segment_anchors_splits_long_anchor_unions(self):
        backend = FixedTextBackend(
            json.dumps(
                {
                    "confirmations": [
                        {
                            "target": "David",
                            "relative_sec": 1.0,
                            "evidence": "A target artwork is visible.",
                        }
                    ],
                    "rejections": [],
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_anchor_split")
            registry = build_segment_inspector_registry(backend, workspace=workspace)
            ProgramInterpreter(registry=registry, workspace=workspace).run(
                [
                    {
                        "tool": "verify_segment_anchors",
                        "args": {
                            "video_path": "/videos/bernini.mp4",
                            "segment_id": "seg_0002",
                            "start_sec": 300.0,
                            "end_sec": 600.0,
                            "anchors": [
                                {
                                    "anchor_id": "anchor_0001",
                                    "start_sec": 315.0,
                                    "end_sec": 325.0,
                                    "targets": ["Aeneas"],
                                    "reason": "Early ASR mention.",
                                },
                                {
                                    "anchor_id": "anchor_0002",
                                    "start_sec": 475.0,
                                    "end_sec": 485.0,
                                    "targets": ["Apollo and Daphne"],
                                    "reason": "Late ASR mention.",
                                },
                            ],
                        },
                    }
                ]
            )
            observation = workspace.read_observations(tool_name="verify_segment_anchors")[0]

        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(
            [(request.metadata["start_sec"], request.metadata["end_sec"]) for request in backend.requests],
            [(315.0, 325.0), (475.0, 485.0)],
        )
        self.assertEqual(len(observation.raw_output["verify_windows"]), 2)
        self.assertIn("split", observation.raw_output["limitations"])

    def test_verify_segment_anchors_rejects_cross_segment_anchors(self):
        backend = FixedTextBackend('{"confirmations": [], "rejections": []}')
        registry = build_segment_inspector_registry(backend)

        with pytest.raises(ValueError, match="anchor segment_id"):
            registry.execute(
                "verify_segment_anchors",
                {
                    "video_path": "/videos/bernini.mp4",
                    "segment_id": "seg_0005",
                    "start_sec": 1200.0,
                    "end_sec": 1500.0,
                    "anchors": [
                        {
                            "anchor_id": "anchor_0001",
                            "segment_id": "seg_0002",
                            "start_sec": 492.0,
                            "end_sec": 544.0,
                            "targets": ["The rape of Persephone", "Apollo and Daphne"],
                        }
                    ],
                },
            )

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
