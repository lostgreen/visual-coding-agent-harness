import unittest

from visual_coding_agent_harness.agents.answer_agent import AnswerAgent, arbitrate_evidence_table
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend


class StaticBackend(VisionLanguageBackend):
    def __init__(self, text: str):
        self.text = text
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.text)


class AnswerAgentArbitrationTest(unittest.TestCase):
    def test_answer_agent_parses_candidate_option_relations(self):
        backend = StaticBackend(
            '{"answer": "B. red car", "rationale": "obs_0002 supports B", '
            '"citations": ["obs_0002"], '
            '"candidate_option_relations": [{"option": "b", "relation": "support", '
            '"strength": 0.82, "observation_id": "obs_0002"}], '
            '"missing_evidence": [], "confidence": 0.82}'
        )
        agent = AnswerAgent(backend)

        result = agent.run(question="Which object?\nA. blue car\nB. red car", evidence_text="- obs_0002 red car")

        self.assertEqual(result.status, "final")
        self.assertEqual(result.candidate_option_relations[0]["option"], "B")
        self.assertEqual(result.candidate_option_relations[0]["observation_id"], "obs_0002")
        self.assertIn("candidate_option_relations", backend.requests[0].prompt)

    def test_arbitration_prefers_visually_grounded_support_over_weak_caption(self):
        table = {
            "options": ["A. first order", "D. fourth order"],
            "groups": {
                "A": [
                    {
                        "obs_id": "obs_0010",
                        "claim": "Caption guesses option A.",
                        "confidence": 0.95,
                        "grounding_quality": "inferred",
                    }
                ],
                "D": [
                    {
                        "obs_id": "obs_0002",
                        "claim": "Visual evidence supports option D.",
                        "confidence": 0.72,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
            },
        }

        result = arbitrate_evidence_table(table)

        self.assertEqual(result.status, "final")
        self.assertEqual(result.answer, "D. fourth order")
        self.assertEqual(result.citations, ["obs_0002"])
        self.assertEqual(result.conflict["options"], ["A", "D"])
        self.assertIn("D", result.rationale)

    def test_arbitration_is_invariant_to_observation_order(self):
        first = {
            "options": ["A. first order", "D. fourth order"],
            "groups": {
                "D": [
                    {
                        "obs_id": "obs_0002",
                        "claim": "Visual evidence supports option D.",
                        "confidence": 0.8,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
                "A": [
                    {
                        "obs_id": "obs_0010",
                        "claim": "Caption guesses option A.",
                        "confidence": 0.9,
                        "grounding_quality": "weak",
                    }
                ],
            },
        }
        shuffled = {
            "options": ["A. first order", "D. fourth order"],
            "groups": {
                "A": list(reversed(first["groups"]["A"])),
                "D": list(reversed(first["groups"]["D"])),
            },
        }

        self.assertEqual(arbitrate_evidence_table(first).answer, arbitrate_evidence_table(shuffled).answer)

    def test_arbitration_abstains_when_margin_is_too_small(self):
        table = {
            "options": ["A. first order", "D. fourth order"],
            "groups": {
                "A": [
                    {
                        "obs_id": "obs_0001",
                        "claim": "One window weakly supports option A.",
                        "confidence": 0.7,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
                "D": [
                    {
                        "obs_id": "obs_0002",
                        "claim": "Another window weakly supports option D.",
                        "confidence": 0.66,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
            },
        }

        result = arbitrate_evidence_table(table, min_margin=0.1)

        self.assertEqual(result.status, "need_more_evidence")
        self.assertEqual(result.answer, "need_more_evidence")
        self.assertTrue(result.missing_evidence)
        self.assertIn("targeted", result.missing_evidence[0])

    def test_arbitration_holds_global_gist_floor_for_gist_questions(self):
        table = {
            "options": ["B. a local scene guess", "D. whole-video synopsis"],
            "groups": {
                "B": [
                    {
                        "obs_id": "obs_0002",
                        "tool": "inspect_segment",
                        "claim": "One local window appears to support option B.",
                        "confidence": 0.8,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
                "D": [
                    {
                        "obs_id": "obs_0001",
                        "tool": "global_gist",
                        "claim": "Supported option: D. The global sparse view captures the whole-video synopsis.",
                        "confidence": 0.74,
                        "grounding_quality": "global_sparse",
                    }
                ],
            },
        }

        result = arbitrate_evidence_table(table)

        self.assertEqual(result.status, "final")
        self.assertEqual(result.answer, "D. whole-video synopsis")
        self.assertEqual(result.citations, ["obs_0001"])


if __name__ == "__main__":
    unittest.main()
