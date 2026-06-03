import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.vlm_agent import VisualAgent
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.vlm_smoke import run_vlm_smoke
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class RecordingBackend(VisionLanguageBackend):
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "plan":
            return BackendResponse(
                text=(
                    '{"answer": "The video shows a person opening a door.", '
                    '"program": ['
                    '{"tool": "caption_video", "args": {"video_path": "input/demo.mp4", "question": "What happens?"}, "assign": "caption"}, '
                    '{"tool": "qa_video", "args": {"video_path": "input/demo.mp4", "question": "What happens?"}, "assign": "qa"}'
                    " ]}"
                ),
                raw={"role": "agent"},
            )
        if request.task == "caption_video":
            return BackendResponse(
                text="A person opens a door and walks into a room.",
                raw={"role": "tool"},
            )
        if request.task == "qa_video":
            return BackendResponse(
                text="The person opens a door.",
                raw={"role": "tool"},
            )
        return BackendResponse(text="fallback")


class VisualAgentTest(unittest.TestCase):
    def test_agent_and_tools_share_one_vlm_backend(self):
        backend = RecordingBackend()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="shared_backend")
            agent = VisualAgent.with_vlm_tools(backend=backend, workspace=workspace)

            result = agent.run(
                question="What happens?",
                media_path="input/demo.mp4",
                media_type="video",
            )

            self.assertEqual(result.answer, "The video shows a person opening a door.")
            self.assertEqual(result.program_result.observation_ids, ["obs_0001", "obs_0002"])
            self.assertEqual([request.task for request in backend.requests], ["plan", "caption_video", "qa_video"])
            self.assertTrue(all(request.media_path == "input/demo.mp4" for request in backend.requests))
            self.assertEqual(backend.requests[1].metadata["nframes"], 8)
            self.assertEqual(backend.requests[2].metadata["nframes"], 8)
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("A person opens a door", ledger)
            self.assertIn("The person opens a door", ledger)

    def test_planner_prompt_contains_tool_use_contract(self):
        backend = RecordingBackend()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="tool_prompt")
            agent = VisualAgent.with_vlm_tools(backend=backend, workspace=workspace)

            agent.run(
                question="What happens?",
                media_path="input/demo.mp4",
                media_type="video",
            )

            prompt = backend.requests[0].prompt
            self.assertIn("Available tools", prompt)
            self.assertIn("caption_video(video_path", prompt)
            self.assertIn("qa_video(video_path", prompt)
            self.assertIn('"answer"', prompt)
            self.assertIn('"program"', prompt)
            self.assertIn('"tool"', prompt)
            self.assertIn('"args"', prompt)
            self.assertIn("Use only these tools", prompt)

    def test_run_result_exposes_structured_input_output_payload(self):
        backend = RecordingBackend()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="io_payload")
            agent = VisualAgent.with_vlm_tools(backend=backend, workspace=workspace)

            result = agent.run(
                question="What happens?",
                media_path="input/demo.mp4",
                media_type="video",
            )

            payload = result.to_dict()
            self.assertEqual(
                payload["input"],
                {
                    "question": "What happens?",
                    "media_path": "input/demo.mp4",
                    "media_type": "video",
                    "tool_policy": "required",
                },
            )
            self.assertEqual(payload["output"]["answer"], "The video shows a person opening a door.")
            self.assertEqual(payload["output"]["observation_ids"], ["obs_0001", "obs_0002"])
            self.assertEqual(payload["output"]["assignments"], {"caption": "obs_0001", "qa": "obs_0002"})
            self.assertEqual(payload["output"]["program"][0]["tool"], "caption_video")

    def test_agent_normalizes_generic_media_path_arguments(self):
        class GenericMediaBackend(RecordingBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "plan":
                    return BackendResponse(
                        text=(
                            '{"answer": "planned", "program": ['
                            '{"tool": "caption_video", "args": {"media_path": "input/demo.mp4", "question": "What happens?"}, "assign": "caption"}'
                            "]}"
                        )
                    )
                return BackendResponse(text="caption ok")

        backend = GenericMediaBackend()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="generic_media")
            agent = VisualAgent.with_vlm_tools(backend=backend, workspace=workspace)

            result = agent.run(
                question="What happens?",
                media_path="input/demo.mp4",
                media_type="video",
            )

            self.assertEqual(result.program[0]["args"], {"video_path": "input/demo.mp4", "question": "What happens?"})
            self.assertEqual(result.program_result.observation_ids, ["obs_0001"])

    def test_agent_overrides_placeholder_media_paths_from_planner(self):
        class PlaceholderPathBackend(RecordingBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "plan":
                    return BackendResponse(
                        text=(
                            '{"answer": "planned", "program": ['
                            '{"tool": "caption_video", "args": {"video_path": "video", "question": "What happens?"}, "assign": "caption"}'
                            "]}"
                        )
                    )
                return BackendResponse(text=f"caption path: {request.media_path}")

        backend = PlaceholderPathBackend()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="placeholder_media")
            agent = VisualAgent.with_vlm_tools(backend=backend, workspace=workspace)

            result = agent.run(
                question="What happens?",
                media_path="/real/path/demo.mp4",
                media_type="video",
            )

            self.assertEqual(result.program[0]["args"]["video_path"], "/real/path/demo.mp4")
            self.assertEqual(backend.requests[1].media_path, "/real/path/demo.mp4")

    def test_smoke_runner_uses_backend_without_pythonpath_coupling(self):
        backend = RecordingBackend()

        with tempfile.TemporaryDirectory() as tmp:
            result = run_vlm_smoke(
                base_dir=Path(tmp),
                backend=backend,
                media_path="input/demo.mp4",
                question="What happens?",
                run_id="smoke",
            )

            self.assertEqual(result.answer, "The video shows a person opening a door.")
            self.assertEqual(result.program_result.observation_ids, ["obs_0001", "obs_0002"])
            self.assertEqual(len(backend.requests), 3)


if __name__ == "__main__":
    unittest.main()
