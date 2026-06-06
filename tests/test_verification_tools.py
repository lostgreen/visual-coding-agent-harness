import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.tools.verification import build_verification_registry
from visual_coding_agent_harness.workspace import EvidenceRecord, EvidenceWorkspace


class VerificationToolsTest(unittest.TestCase):
    def test_verify_ledger_answer_reads_workspace_ledger_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify")
            (workspace.root / "ledger.md").write_text(
                "# Evidence Ledger\n\n"
                "- `obs_0001` | tool: `caption_segment` | confidence: 0.66 | artifacts: clip.mp4 | claim: The segment shows a red aircraft landing on a runway. | limitations: -\n",
                encoding="utf-8",
            )
            registry = build_verification_registry(workspace=workspace)

            result = registry.execute("verify_ledger_answer", {"answer": "A red aircraft lands on a runway."})

            self.assertIn("supported", result["claim"])
            self.assertGreaterEqual(result["confidence"], 0.5)
            self.assertEqual(result["regions"][0]["cited_observations"], ["obs_0001"])
            self.assertIn("aircraft", result["regions"][0]["supported_terms"])

    def test_verify_ledger_answer_reports_missing_terms(self):
        registry = build_verification_registry()

        result = registry.execute(
            "verify_ledger_answer",
            {
                "answer": "A red aircraft lands beside a submarine.",
                "ledger_text": "- `obs_0001` claim: The segment shows a red aircraft landing.",
            },
        )

        self.assertIn("insufficient", result["claim"])
        self.assertIn("submarine", result["regions"][0]["missing_terms"])

    def test_verify_ledger_answer_rejects_navigation_only_support(self):
        registry = build_verification_registry()

        result = registry.execute(
            "verify_ledger_answer",
            {
                "answer": "The video shows an aircraft museum.",
                "ledger_text": (
                    "- `obs_0001` | tool: `video_ls` | confidence: 1.00 | artifacts: demo.mp4 | "
                    "claim: Candidate segments include an aircraft museum. | limitations: -\n"
                ),
            },
        )

        self.assertIn("insufficient", result["claim"])
        self.assertEqual(result["regions"][0]["evidence_gate"]["visual_observation_ids"], [])
        self.assertIn("no non-navigation visual evidence", result["regions"][0]["evidence_gate"]["reasons"])

    def test_verify_ledger_answer_counts_vision_read_as_visual_evidence(self):
        registry = build_verification_registry()

        result = registry.execute(
            "verify_ledger_answer",
            {
                "answer": "The segment describes the empire collapse.",
                "ledger_text": (
                    "- `obs_0001` | tool: `vision_read` | confidence: 0.82 | artifacts: clip.mp4 | "
                    "claim: The segment describes the empire collapse. | limitations: -\n"
                ),
            },
        )

        gate = result["regions"][0]["evidence_gate"]
        self.assertIn("supported", result["claim"])
        self.assertEqual(gate["visual_observation_ids"], ["obs_0001"])
        self.assertNotIn("no non-navigation visual evidence", gate["reasons"])

    def test_verify_ledger_answer_checks_required_citations(self):
        registry = build_verification_registry()

        result = registry.execute(
            "verify_ledger_answer",
            {
                "answer": "The segment shows aircraft history.",
                "ledger_text": (
                    "- `obs_0001` | tool: `qa_segment` | confidence: 0.80 | artifacts: clip.mp4 | "
                    "claim: The segment shows aircraft history. | limitations: -\n"
                ),
                "required_citations": ["obs_0001", "obs_0002"],
            },
        )

        self.assertIn("insufficient", result["claim"])
        self.assertEqual(result["regions"][0]["evidence_gate"]["missing_citations"], ["obs_0002"])

    def test_verify_ledger_answer_rejects_mapped_support_below_grounding_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_grounding_floor")
            observation = workspace.write_observation(
                tool_name="caption_segment",
                input_artifacts=["clip_a.mp4"],
                claim="Visual row text supports option A.",
                confidence=0.92,
                raw_output={
                    "candidate_option_relations": [
                        {"option": "A", "relation": "support", "strength": 0.92, "assigned_by": "answer_agent"}
                    ],
                    "grounding_quality": "visually_confirmed",
                },
            )
            distilled = EvidenceRecord(
                evidence_id=workspace.next_evidence_id("distilled"),
                stage="distilled",
                parent_id=None,
                tool="caption_segment",
                observation_id=observation.observation_id,
                frame_set_id=None,
                content={"claim": observation.claim},
                grounding_quality="inferred",
                confidence=0.92,
                created_at=1.0,
            )
            workspace.write_evidence(distilled)
            workspace.write_ledger_entry(observation, parent_records=[distilled])
            registry = build_verification_registry(workspace=workspace)

            result = registry.execute(
                "verify_ledger_answer",
                {
                    "answer": "A. option A",
                    "candidate_options": ["A. option A", "B. option B"],
                    "required_citations": [observation.observation_id],
                    "min_score": 0.0,
                },
            )

            gate = result["regions"][0]["evidence_gate"]
            self.assertIn("insufficient", result["claim"])
            self.assertTrue(any("no visually_confirmed" in reason for reason in gate["reasons"]))

    def test_verify_ledger_answer_rejects_uncited_strong_conflicting_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_conflict")
            d_observation = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["clip_d.mp4"],
                claim="Visual evidence supports option D.",
                confidence=0.82,
                regions=[{"start_sec": 300.0, "end_sec": 600.0}],
                limitations="Directly visible in the sampled segment.",
                raw_output={
                    "candidate_option_relations": [
                        {"option": "D", "relation": "support", "strength": 0.82, "assigned_by": "answer_agent"}
                    ],
                    "grounding_quality": "visually_confirmed",
                },
            )
            a_observation = workspace.write_observation(
                tool_name="caption_segment",
                input_artifacts=["clip_a.mp4"],
                claim="Caption-like evidence guesses option A.",
                confidence=0.91,
                regions=[{"start_sec": 1500.0, "end_sec": 1800.0}],
                limitations="Inferred from context; lacks explicit visual confirmation.",
                raw_output={
                    "candidate_option_relations": [
                        {"option": "A", "relation": "support", "strength": 0.91, "assigned_by": "answer_agent"}
                    ]
                },
            )
            workspace.write_ledger_entry(d_observation)
            workspace.write_ledger_entry(a_observation)
            registry = build_verification_registry(workspace=workspace)

            result = registry.execute(
                "verify_ledger_answer",
                {
                    "answer": "A. first option",
                    "required_citations": ["obs_0002"],
                    "min_score": 0.0,
                },
            )

            gate = result["regions"][0]["evidence_gate"]
            self.assertIn("insufficient", result["claim"])
            self.assertIn("uncited stronger conflicting option support", gate["reasons"])
            self.assertEqual(gate["top_conflicting_observation"], "obs_0001")
            self.assertEqual(gate["top_conflict_relation"], "Contradict")

    def test_verify_ledger_answer_labels_option_relations(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_relations")
            selected = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["clip_a.mp4"],
                claim="Visual evidence supports option A.",
                confidence=0.8,
                limitations="Directly visible.",
                raw_output={
                    "candidate_option_relations": [
                        {"option": "A", "relation": "support", "strength": 0.8, "assigned_by": "answer_agent"}
                    ],
                    "grounding_quality": "visually_confirmed",
                },
            )
            conflicting = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["clip_b.mp4"],
                claim="Visual evidence supports option B.",
                confidence=0.79,
                limitations="Directly visible.",
                raw_output={
                    "candidate_option_relations": [
                        {"option": "B", "relation": "support", "strength": 0.79, "assigned_by": "answer_agent"}
                    ],
                    "grounding_quality": "visually_confirmed",
                },
            )
            neutral = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["clip_unknown.mp4"],
                claim="Visual evidence shows a sculpture but no option mapping.",
                confidence=0.7,
                limitations="Directly visible.",
            )
            for observation in [selected, conflicting, neutral]:
                workspace.write_ledger_entry(observation)
            registry = build_verification_registry(workspace=workspace)

            result = registry.execute(
                "verify_ledger_answer",
                {"answer": "A. first option", "required_citations": ["obs_0001"], "min_score": 0.0},
            )

            relations = result["regions"][0]["evidence_gate"]["option_relations"]
            self.assertEqual(relations["obs_0001"], "Support")
            self.assertEqual(relations["obs_0002"], "Contradict")
            self.assertEqual(relations["obs_0003"], "Neutral")

    def test_verify_ledger_answer_rejects_temporal_order_contradiction(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_temporal_order")
            blue = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["clip_blue.mp4"],
                claim="The blue object appears first.",
                confidence=0.88,
                regions=[{"start_sec": 10.0, "end_sec": 12.0}],
                limitations="Directly visible.",
                raw_output={"event_label": "blue object", "grounding_quality": "visually_confirmed"},
            )
            red = workspace.write_observation(
                tool_name="inspect_segment",
                input_artifacts=["clip_red.mp4"],
                claim="The red object appears later.",
                confidence=0.89,
                regions=[{"start_sec": 20.0, "end_sec": 22.0}],
                limitations="Directly visible.",
                raw_output={"event_label": "red object", "grounding_quality": "visually_confirmed"},
            )
            for observation in [blue, red]:
                workspace.write_ledger_entry(observation)
            registry = build_verification_registry(workspace=workspace)

            result = registry.execute(
                "verify_ledger_answer",
                {
                    "answer": "A. red object then blue object",
                    "question": "Which order is shown?",
                    "candidate_options": [
                        "A. red object then blue object",
                        "B. blue object then red object",
                    ],
                    "required_citations": ["obs_0001", "obs_0002"],
                    "min_score": 0.0,
                },
            )

            gate = result["regions"][0]["evidence_gate"]
            self.assertIn("insufficient", result["claim"])
            self.assertIn("temporal order contradicts evidence", gate["reasons"])
            self.assertEqual(gate["temporal_order_verdict"], "Contradict")

    def test_summarize_ledger_evidence_extracts_compact_claims(self):
        registry = build_verification_registry()

        result = registry.execute(
            "summarize_ledger_evidence",
            {
                "ledger_text": (
                    "- `obs_0001` | claim: first claim. | limitations: -\n"
                    "- `obs_0002` | claim: second claim. | limitations: blur\n"
                ),
                "max_claims": 1,
            },
        )

        self.assertEqual(result["regions"][0]["claims"], [{"observation_id": "obs_0001", "claim": "first claim."}])
        self.assertIn("1 ledger claim", result["claim"])


if __name__ == "__main__":
    unittest.main()
