import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.tools.dummy import build_dummy_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class HarnessTest(unittest.TestCase):
    def test_registry_executes_registered_tool_with_validated_arguments(self):
        registry = ToolRegistry()

        @tool(name="echo_claim", description="Return a visual claim for testing.")
        def echo_claim(text: str, confidence: float = 1.0):
            return {
                "claim": text,
                "confidence": confidence,
            }

        registry.register(echo_claim)

        result = registry.execute(
            "echo_claim",
            {"text": "The sign reads EXIT.", "confidence": 0.9},
        )

        self.assertEqual(result["claim"], "The sign reads EXIT.")
        self.assertEqual(result["confidence"], 0.9)

    def test_workspace_persists_observations_trace_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="case_001")

            observation = workspace.write_observation(
                tool_name="ocr_region",
                input_artifacts=["artifacts/crops/frame_001_box_1.png"],
                claim="The sign reads EXIT.",
                confidence=0.91,
                regions=[{"frame": "frame_001.jpg", "bbox": [10, 20, 80, 60]}],
                limitations="slight blur",
            )
            workspace.write_trace_event(
                "tool_result",
                {"observation_id": observation.observation_id},
            )
            workspace.write_ledger_entry(observation)

            self.assertEqual(observation.observation_id, "obs_0001")
            self.assertEqual((workspace.root / "observations.jsonl").read_text().count("obs_0001"), 1)
            self.assertEqual((workspace.root / "trace.jsonl").read_text().count("tool_result"), 1)
            self.assertIn("The sign reads EXIT.", (workspace.root / "ledger.md").read_text())

    def test_workspace_compacts_ledger_for_planner_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="compact")
            ledger_path = workspace.root / "ledger.md"
            ledger_path.write_text(
                "# Evidence Ledger\n\n"
                "- `obs_0001` | tool: `video_ls` | confidence: 1.00 | artifacts: demo.mp4 | claim: Candidate segments include seg_0002 and seg_0003 with a very long navigation explanation. | limitations: -\n"
                "- `obs_0002` | tool: `search_segments` | confidence: 0.85 | artifacts: demo.mp4 | claim: Search returned seg_0002 for aircraft. | limitations: lexical\n"
                "- `obs_0003` | tool: `inspect_segment` | confidence: 0.78 | artifacts: demo.mp4#t=30,42 | claim: The localized segment shows aircraft history. | limitations: slight blur\n"
                "- `obs_0004` | tool: `qa_segment` | confidence: 0.66 | artifacts: demo.mp4#t=42,50 | claim: A narrator discusses aviation exhibits. | limitations: low resolution\n",
                encoding="utf-8",
            )

            compact = workspace.compact_ledger_text(max_working_observations=1)

            self.assertIn("Long-Term Visual Evidence", compact)
            self.assertIn("obs_0003", compact)
            self.assertIn("The localized segment shows aircraft history", compact)
            self.assertIn("Short-Term Working Buffer", compact)
            self.assertIn("obs_0004", compact)
            self.assertIn("Navigation Summary", compact)
            self.assertIn("obs_0001: video_ls", compact)
            self.assertNotIn("very long navigation explanation", compact)

    def test_interpreter_runs_visual_program_and_returns_observation_ids(self):
        registry = ToolRegistry()

        @tool(name="caption_image", description="Caption a test image.")
        def caption_image(image_path: str):
            return {
                "claim": f"{image_path} contains a red cup.",
                "confidence": 0.8,
                "input_artifacts": [image_path],
            }

        registry.register(caption_image)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="case_002")
            interpreter = ProgramInterpreter(registry=registry, workspace=workspace)

            result = interpreter.run(
                [
                    {
                        "tool": "caption_image",
                        "args": {"image_path": "input/frame_001.jpg"},
                        "assign": "caption",
                    }
                ]
            )

            self.assertEqual(result.assignments["caption"], "obs_0001")
            self.assertEqual(result.observation_ids, ["obs_0001"])
            self.assertIn("contains a red cup", (workspace.root / "ledger.md").read_text())

    def test_dummy_registry_provides_seed_visual_tools(self):
        registry = build_dummy_registry()

        caption = registry.execute("caption_image", {"image_path": "input/frame_001.jpg"})
        ocr = registry.execute("ocr_region", {"image_path": "input/sign_crop.jpg"})
        verification = registry.execute(
            "verify_answer",
            {
                "answer": "The sign reads EXIT.",
                "ledger_text": "- obs_0001 claim: The sign reads EXIT.",
            },
        )

        self.assertIn("input/frame_001.jpg", caption["input_artifacts"])
        self.assertEqual(ocr["claim"], "The visible text reads EXIT.")
        self.assertTrue(verification["confidence"] >= 0.5)

    def test_interpreter_rejects_step_without_tool_name(self):
        registry = ToolRegistry()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="case_003")
            interpreter = ProgramInterpreter(registry=registry, workspace=workspace)

            with self.assertRaises(ValueError) as context:
                interpreter.run([{"args": {}}])

            self.assertIn("missing required 'tool'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
