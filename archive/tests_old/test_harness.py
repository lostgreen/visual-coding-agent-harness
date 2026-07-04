import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.legacy.interpreter import ProgramInterpreter
from visual_coding_agent_harness.legacy.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.legacy.tools.dummy import build_dummy_registry
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


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

    def test_compact_ledger_preserves_multiline_visual_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="multiline_claim")
            observation = workspace.write_observation(
                tool_name="vision_read",
                input_artifacts=["demo.mp4#t=0,120"],
                claim="B. Why the empire was divided.\nThe video shows maps and narration about its collapse.",
                confidence=0.76,
                regions=[{"start_sec": 0.0, "end_sec": 120.0}],
                raw_output={"grounding_quality": "visually_confirmed"},
            )
            workspace.write_ledger_entry(observation)

            compact = workspace.compact_ledger_text()

            self.assertIn("claim: B. Why the empire was divided. The video shows maps", compact)
            self.assertNotIn("claim:  | limitations", compact)

    def test_workspace_builds_option_grouped_evidence_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="evidence_table")
            navigation = workspace.write_observation(
                tool_name="video_ls",
                input_artifacts=["demo.mp4"],
                claim="Candidate windows include seg_0001 and seg_0002.",
                confidence=1.0,
                limitations="-",
                raw_output={"supported_option": "A"},
            )
            d_observation = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["demo.mp4#t=300,600"],
                claim="The localized visual evidence matches the fourth sequence.",
                confidence=0.82,
                regions=[{"segment_id": "seg_0002", "start_sec": 300.0, "end_sec": 600.0}],
                limitations="Directly visible in the sampled segment.",
                raw_output={"supported_option": "D", "grounding_quality": "visually_confirmed"},
            )
            a_observation = workspace.write_observation(
                tool_name="caption_segment",
                input_artifacts=["demo.mp4#t=1500,1800"],
                claim="The caption guesses the first sequence.",
                confidence=0.91,
                regions=[{"segment_id": "seg_0005", "start_sec": 1500.0, "end_sec": 1800.0}],
                limitations="Inferred from context; lacks explicit visual confirmation.",
                raw_output={"supported_option": "A"},
            )
            for observation in [navigation, d_observation, a_observation]:
                workspace.write_ledger_entry(observation)

            table = workspace.evidence_table(
                question="Which sequence is correct?",
                options=["A. first sequence", "B. second sequence", "C. third sequence", "D. fourth sequence"],
                include_legacy_worker_votes=True,
            )

            self.assertEqual(table["question"], "Which sequence is correct?")
            self.assertNotIn("obs_0001", [row["obs_id"] for row in table["rows"]])
            self.assertEqual(table["groups"]["D"][0]["obs_id"], "obs_0002")
            self.assertEqual(table["groups"]["D"][0]["grounding_quality"], "visually_confirmed")
            self.assertEqual(table["groups"]["D"][0]["time_range"], [300.0, 600.0])
            self.assertEqual(table["groups"]["A"][0]["obs_id"], "obs_0003")
            self.assertEqual(table["groups"]["A"][0]["grounding_quality"], "inferred")
            self.assertEqual(table["groups"]["A"][0]["artifact"], "demo.mp4#t=1500,1800")

    def test_evidence_table_ignores_legacy_worker_vote_claim_text_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="claim_supported_option")
            observation = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["demo.mp4#t=0,300"],
                claim=(
                    "Claim: The visible artwork order matches the fourth sequence.\n"
                    "Supported option: D.\n"
                    "Confidence: High."
                ),
                confidence=0.74,
                limitations="Inspector used a physical segment clip.",
                raw_output={},
            )
            workspace.write_ledger_entry(observation)

            table = workspace.evidence_table(
                question="Which sequence is correct?",
                options=["A. first sequence", "B. second sequence", "C. third sequence", "D. fourth sequence"],
            )

            self.assertEqual(table["groups"]["D"], [])
            self.assertEqual(table["groups"]["unassigned"][0]["obs_id"], "obs_0001")
            self.assertTrue(table["groups"]["unassigned"][0]["legacy_worker_vote"])

    def test_local_worker_bare_option_answer_is_quarantined_from_ledger_and_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="bare_option_vote")
            observation = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["demo.mp4#t=0,90"],
                claim="A. Because the object appears to be from the Ming Dynasty.",
                confidence=0.74,
                regions=[
                    {
                        "segment_id": "seg_0001",
                        "start_sec": 0.0,
                        "end_sec": 90.0,
                        "candidate_options": [
                            "A. Ming Dynasty significance.",
                            "B. New railway nearby.",
                            "C. Treasure inside.",
                            "D. Highway realignment.",
                        ],
                    }
                ],
                limitations="Inspector used a physical segment clip.",
                raw_output={},
            )
            workspace.write_ledger_entry(observation)

            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("legacy local-worker option vote quarantined", ledger)
            self.assertNotIn("claim: A. Because", ledger)
            self.assertIn("fact_text: Because the object appears to be from the Ming Dynasty.", ledger)

            table = workspace.evidence_table_v2(
                question="Which reason motivated the archaeologist?",
                options=[
                    "A. Ming Dynasty significance.",
                    "B. New railway nearby.",
                    "C. Treasure inside.",
                    "D. Highway realignment.",
                ],
            )

            self.assertEqual(table["groups"]["A"], [])
            self.assertEqual(table["legacy_worker_vote_rows"], 1)
            self.assertEqual(table["groups"]["unassigned"][0]["obs_id"], "obs_0001")
            self.assertTrue(table["groups"]["unassigned"][0]["legacy_worker_vote"])
            self.assertEqual(table["groups"]["unassigned"][0]["claim"], "Because the object appears to be from the Ming Dynasty.")

    def test_evidence_table_can_replay_legacy_worker_votes_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="legacy_claim_supported_option")
            observation = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["demo.mp4#t=0,300"],
                claim=(
                    "Claim: The visible artwork order matches the fourth sequence.\n"
                    "Supported option: D.\n"
                    "Confidence: High."
                ),
                confidence=0.74,
                limitations="Inspector used a physical segment clip.",
                raw_output={},
            )
            workspace.write_ledger_entry(observation)

            table = workspace.evidence_table(
                question="Which sequence is correct?",
                options=["A. first sequence", "B. second sequence", "C. third sequence", "D. fourth sequence"],
                include_legacy_worker_votes=True,
            )

            self.assertEqual(table["groups"]["D"][0]["obs_id"], "obs_0001")
            self.assertEqual(table["groups"]["D"][0]["supported_option"], "D")
            self.assertTrue(table["groups"]["D"][0]["legacy_worker_vote"])

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
