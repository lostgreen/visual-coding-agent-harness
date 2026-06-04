import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.tools.inspector import build_segment_inspector_registry
from visual_coding_agent_harness.tools.segments import build_segment_vlm_registry
from visual_coding_agent_harness.tools.vlm import build_vlm_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


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
        self.assertEqual(request.metadata["nframes"], 16)
        self.assertEqual(request.metadata["max_pixels"], 200000)
        self.assertEqual(request.metadata["fps"], 2.0)
        self.assertIn("QA task", request.prompt)
        self.assertEqual(result["regions"][0]["max_pixels"], 200000)

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
        self.assertIn("Return one distilled observation", request.prompt)
        self.assertIn("A. aircraft", request.prompt)
        self.assertEqual(result["claim"], "inspect_segment answer")
        self.assertEqual(result["regions"][0]["tool_role"], "segment_inspector")
        self.assertEqual(result["regions"][0]["segment_id"], "seg_0004")


if __name__ == "__main__":
    unittest.main()
