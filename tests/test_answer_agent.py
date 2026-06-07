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

    def test_answer_agent_parses_single_quoted_json_like_response(self):
        backend = StaticBackend(
            "Here is the final JSON:\n"
            "{'answer': 'D. full rise and fall', 'rationale': 'obs_0003 supports D', "
            "'citations': ['obs_0003'], 'missing_evidence': [], 'confidence': 0.81}"
        )

        result = AnswerAgent(backend).run(question="Which option?", evidence_text="- obs_0003 supports D")

        self.assertEqual(result.status, "final")
        self.assertEqual(result.answer, "D. full rise and fall")
        self.assertEqual(result.citations, ["obs_0003"])

    def test_answer_agent_falls_back_to_main_idea_unassigned_evidence_after_parse_failure(self):
        backend = StaticBackend("I think D because the whole video covers the empire.")
        table = {
            "options": [
                "A. The fall of Rome",
                "B. Why the Austro-Hungarian Empire was divided",
                "C. A battle timeline",
                "D. How the Austro-Hungarian Empire rises and falls",
            ],
            "rows": [
                {
                    "obs_id": "obs_0002",
                    "tool": "vision_read",
                    "claim": "A map of Austria and Hungary appears with historical political regions.",
                    "confidence": 0.72,
                    "grounding_quality": "visually_confirmed",
                },
                {
                    "obs_id": "obs_0003",
                    "tool": "vision_read",
                    "claim": "A cartoon character stands in front of a map of Austria-Hungary.",
                    "confidence": 0.78,
                    "grounding_quality": "visually_confirmed",
                },
                {
                    "obs_id": "obs_0007",
                    "tool": "vision_read",
                    "claim": "A map of the Austro-Hungarian Empire shows ethnic groups in 1910.",
                    "confidence": 0.74,
                    "grounding_quality": "weak",
                },
            ],
        }

        result = AnswerAgent(backend).run(
            question="What is the main idea?",
            evidence_text="- evidence",
            evidence_table=table,
        )

        self.assertEqual(result.status, "final")
        self.assertEqual(result.answer, "D. How the Austro-Hungarian Empire rises and falls")
        self.assertTrue(result.citations)
        self.assertEqual(result.conflict["source"], "fallback_main_idea_unassigned_evidence")

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

    def test_arbitration_does_not_final_from_global_sparse_floor(self):
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
                        "claim": "D. The global sparse view hints at the whole-video synopsis.",
                        "confidence": 0.74,
                        "grounding_quality": "global_sparse",
                    }
                ],
            },
        }

        result = arbitrate_evidence_table(table)

        self.assertEqual(result.status, "final")
        self.assertEqual(result.answer, "B. a local scene guess")
        self.assertEqual(result.citations, ["obs_0002"])

    def test_main_idea_coverage_prefers_full_rise_and_fall_over_partial_division(self):
        table = {
            "options": [
                "B. Why the Austro-Hungarian Empire was divided",
                "D. How the Austro-Hungarian Empire rose and fell",
            ],
            "groups": {
                "B": [
                    {
                        "obs_id": "obs_0007",
                        "tool": "vision_read",
                        "claim": "The ending explains why the empire was divided after the war.",
                        "confidence": 0.86,
                        "grounding_quality": "visually_confirmed",
                        "candidate_option_relations": [
                            {"option": "B", "relation": "support", "strength": 0.86}
                        ],
                    }
                ],
                "D": [
                    {
                        "obs_id": "obs_0003",
                        "tool": "timeline_asr_summary",
                        "claim": (
                            "Middle segments discuss the creation of the Austro-Hungarian Empire, "
                            "economic growth, prosperity, population, industrial ability, and stability."
                        ),
                        "confidence": 0.84,
                        "grounding_quality": "indexed_transcript",
                        "candidate_option_relations": [
                            {"option": "D", "relation": "support", "strength": 0.84}
                        ],
                    },
                    {
                        "obs_id": "obs_0006",
                        "tool": "timeline_asr_summary",
                        "claim": (
                            "Late segments discuss war pressure, independence movements, collapse, "
                            "and why the empire fell apart."
                        ),
                        "confidence": 0.84,
                        "grounding_quality": "indexed_transcript",
                        "candidate_option_relations": [
                            {"option": "D", "relation": "support", "strength": 0.84}
                        ],
                    },
                ],
                "unassigned": [
                    {
                        "obs_id": "obs_0001",
                        "tool": "global_gist",
                        "claim": "Sparse topic hint: empire formation, internal tensions, and collapse.",
                        "confidence": 0.76,
                        "grounding_quality": "global_sparse",
                    }
                ],
            },
        }

        result = arbitrate_evidence_table(table)

        self.assertEqual(result.status, "final")
        self.assertEqual(result.answer, "D. How the Austro-Hungarian Empire rose and fell")
        self.assertEqual(set(result.citations), {"obs_0003", "obs_0006"})
        self.assertEqual(result.conflict["winner"], "D")
        self.assertIn("coverage", result.rationale)

    def test_mutex_conflict_blocks_final(self):
        table = {
            "options": ["B. humble background", "D. upper class background"],
            "groups": {
                "B": [
                    {
                        "obs_id": "obs_0002",
                        "claim": "The person is presented as humble.",
                        "confidence": 0.82,
                        "grounding_quality": "visually_confirmed",
                        "supported_option": "B",
                        "mutex_group_id": "life_background",
                    }
                ],
                "D": [
                    {
                        "obs_id": "obs_0008",
                        "claim": "The person is presented as upper class.",
                        "confidence": 0.84,
                        "grounding_quality": "visually_confirmed",
                        "supported_option": "D",
                        "mutex_group_id": "life_background",
                    }
                ],
            },
        }

        result = arbitrate_evidence_table(table)

        self.assertEqual(result.status, "need_more_evidence")
        self.assertIn("mutex_conflict", result.missing_evidence[0])

    def test_overlapping_options_require_distinguishing_fact(self):
        table = {
            "options": [
                "B. Why the Austro-Hungarian Empire was divided",
                "D. How the Austro-Hungarian Empire rises and falls",
            ],
            "groups": {
                "B": [
                    {
                        "obs_id": "obs_0004",
                        "tool": "inspect_segment",
                        "claim": "The video discusses the Austro-Hungarian Empire.",
                        "confidence": 0.54,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
                "D": [
                    {
                        "obs_id": "obs_0001",
                        "tool": "inspect_segment",
                        "claim": "The sparse view discusses the Austro-Hungarian Empire.",
                        "confidence": 0.82,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
            },
        }

        result = arbitrate_evidence_table(table)

        self.assertEqual(result.status, "need_more_evidence")
        self.assertEqual(result.missing_evidence, ["disambiguate_overlapping_options"])

    def test_low_conf_picks_most_supported_option(self):
        backend = StaticBackend(
            '{"answer": "need_more_evidence", "rationale": "partial", "citations": [], '
            '"candidate_option_relations": ['
            '{"option": "A", "relation": "support", "strength": 0.5, "observation_id": "obs_0001", "grounding_quality": "visually_confirmed"},'
            '{"option": "B", "relation": "support", "strength": 0.8, "observation_id": "obs_0002", "grounding_quality": "visually_confirmed"},'
            '{"option": "B", "relation": "support", "strength": 0.7, "observation_id": "obs_0003", "grounding_quality": "visually_confirmed"}'
            '], "missing_evidence": ["need one more window"], "confidence": 0.0}'
        )
        result = AnswerAgent(backend).run(question="Which option?", evidence_text="- partial evidence")

        low_conf = result.as_low_confidence_final()

        self.assertTrue(result.has_partial_support())
        self.assertEqual(low_conf.status, "low_confidence_final")
        self.assertEqual(low_conf.answer, "B")
        self.assertEqual(low_conf.citations, ["obs_0002", "obs_0003"])
        self.assertAlmostEqual(low_conf.confidence, 0.525)

    def test_low_conf_requires_at_least_one_visual(self):
        backend = StaticBackend(
            '{"answer": "need_more_evidence", "rationale": "partial", "citations": [], '
            '"candidate_option_relations": [{"option": "A", "relation": "support", '
            '"strength": 0.7, "observation_id": "obs_0001", "grounding_quality": "inferred"}], '
            '"missing_evidence": ["visual confirmation"], "confidence": 0.0}'
        )
        result = AnswerAgent(backend).run(question="Which option?", evidence_text="- weak evidence")

        self.assertFalse(result.has_partial_support())
        self.assertEqual(result.as_low_confidence_final().status, "need_more_evidence")


if __name__ == "__main__":
    unittest.main()
