import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.tools.verification import build_verification_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


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
