import unittest

from visual_coding_agent_harness.agents.question_policy import (
    classify_question_route,
    classify_narration_subroute,
    extract_candidate_options,
    extract_option_sequence_specs,
    extract_option_target_atom_map,
    extract_option_target_atoms,
    select_question_playbook,
)


class QuestionPolicyTest(unittest.TestCase):
    def test_selects_multiple_choice_playbook_from_options(self):
        playbook = select_question_playbook(
            "Which option is visible?\nA. aircraft museum\nB. submarine\nC. mountain road"
        )

        self.assertEqual(playbook.name, "multiple_choice")
        self.assertIn("discriminative search atoms", " ".join(playbook.instructions))
        self.assertIn("verify option consistency", " ".join(playbook.sufficiency_rules))

    def test_selects_temporal_ordering_playbook_from_question(self):
        playbook = select_question_playbook("What happens before the person opens the door?")

        self.assertEqual(playbook.name, "timeline_ordering")
        self.assertIn("timestamped", " ".join(playbook.sufficiency_rules))

    def test_question_playbook_renders_option_blind_variants(self):
        cases = [
            ("What is the video mainly about?", "global_gist to get a sparse whole-video topic"),
            ("What happens before the person opens the door?", "Local workers should report facts and presentation order only."),
            ("Which option is visible?\nA. red car\nB. blue cup", "Local workers should report facts only."),
        ]

        for question, expected in cases:
            with self.subTest(question=question):
                playbook = select_question_playbook(question)

                self.assertEqual(playbook.to_prompt(), playbook.to_prompt(option_blind=False))
                self.assertIn(expected, playbook.to_prompt(option_blind=True))

    def test_classifies_synopsis_mcq_as_global_gist_route(self):
        route = classify_question_route(
            "What is the video mainly about?\n"
            "A. a cooking tutorial\n"
            "B. a city tour\n"
            "C. a product demo\n"
            "D. an aviation documentary"
        )

        self.assertEqual(route, "gist_global")

    def test_classifies_videomme_main_idea_wrapper_as_global_gist_route(self):
        route = classify_question_route(
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: What's the main idea of the video?\n"
            "Options:\n"
            "A. What did the French gain from World War One.\n"
            "B. Why the Austro-Hungarian Empire was divided.\n"
            "C. The process of World War One.\n"
            "D. How the Austro-Hungarian Empire rises and falls."
        )

        self.assertEqual(route, "gist_global")

    def test_extracts_videomme_options_from_single_line_wrapper(self):
        options = extract_candidate_options(
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: What's the main idea of the video?\n"
            "Options: A. What did the French gain from World War One. "
            "B. Why the Austro-Hungarian Empire was divided. "
            "C. The process of World War One. "
            "D. How the Austro-Hungarian Empire rises and falls."
        )

        self.assertEqual(
            list(options),
            [
                "A. What did the French gain from World War One.",
                "B. Why the Austro-Hungarian Empire was divided.",
                "C. The process of World War One.",
                "D. How the Austro-Hungarian Empire rises and falls.",
            ],
        )

    def test_classifies_temporal_mcq_as_temporal_order_route(self):
        route = classify_question_route(
            "What happens right after the aircraft takes off?\n"
            "A. one event\n"
            "B. another event\n"
            "C. third event\n"
            "D. fourth event"
        )

        self.assertEqual(route, "temporal_order")

    def test_option_life_journey_markers_make_route_temporal(self):
        question = (
            "How was his life journey according to the video?\n"
            "A. Born with humble background and lived in seclusion in a farmhouse.\n"
            "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
            "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
            "D. Born in the upper class and lived in seclusion in a farmhouse."
        )

        self.assertEqual(classify_question_route(question), "temporal_order")

    def test_option_atoms_preserve_surface_clauses_without_semantic_synonyms(self):
        question = (
            "How was his life journey according to the video?\n"
            "A. Born with humble background and lived in seclusion in a farmhouse.\n"
            "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse."
        )

        atoms = extract_option_target_atoms(question, include_synonyms=True)
        self.assertIn("Born with humble background and lived in seclusion in a farmhouse", atoms)
        self.assertIn("entered the upper class", atoms)
        self.assertNotIn("isolation", atoms)
        self.assertNotIn("upper echelons", atoms)
        self.assertNotIn("withdrew from public life", atoms)

    def test_option_atom_map_preserves_option_order(self):
        question = (
            "How was his life journey according to the video?\n"
            "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
            "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class."
        )

        mapping = extract_option_target_atom_map(question, include_synonyms=False)

        self.assertLess(mapping["B"].index("entered the upper class"), mapping["B"].index("lived in seclusion in a farmhouse"))
        self.assertLess(mapping["C"].index("lived in seclusion in a farmhouse"), mapping["C"].index("entered the upper class"))

    def test_life_journey_option_sequences_are_not_semantically_canonicalized(self):
        question = (
            "How was his life journey according to the video?\n"
            "A. Born with humble background and lived in seclusion in a farmhouse.\n"
            "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
            "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
            "D. Born in the upper class and lived in seclusion in a farmhouse."
        )

        sequences = extract_option_sequence_specs(question)

        self.assertEqual(sequences, {})

    def test_real_612_borned_option_is_left_for_grounding_planner(self):
        question = (
            "How was his life journey according to the video?\n"
            "A. Born with humble background and lived in seclusion in a farmhouse.\n"
            "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
            "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
            "D. Borned in the upper class and lived in seclusion in a farmhouse."
        )

        sequences = extract_option_sequence_specs(question)

        self.assertEqual(sequences, {})

    def test_temporal_route_ignores_formatting_instruction_first(self):
        route = classify_question_route(
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: Which event happens after the aircraft takes off?\n"
            "Options:\n"
            "A. one event\n"
            "B. another event\n"
            "C. third event\n"
            "D. fourth event"
        )

        self.assertEqual(route, "temporal_order")

    def test_classifies_narration_subroute_from_explicit_or_biographical_asr_markers(self):
        positives = [
            "How was his life journey according to the video?",
            "According to the narrator, why did he leave?",
            "What does the video tell us about her early life?",
        ]

        for question in positives:
            with self.subTest(question=question):
                self.assertEqual(classify_narration_subroute(question), "narration_timeline")

    def test_narration_subroute_rejects_visual_action_and_spatial_hard_negatives(self):
        negatives = [
            "How did the man open the door?",
            "According to the video, how did the man open the door?",
            "How was the painting positioned?",
            "According to the video, how was the painting positioned?",
            "How did the ball move after impact?",
            "What does she pick up next?",
        ]

        for question in negatives:
            with self.subTest(question=question):
                self.assertEqual(classify_narration_subroute(question), "visual_timeline")


if __name__ == "__main__":
    unittest.main()
