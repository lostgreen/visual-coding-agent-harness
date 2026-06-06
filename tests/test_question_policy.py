import unittest

from visual_coding_agent_harness.agents.question_policy import classify_question_route, extract_candidate_options, select_question_playbook


class QuestionPolicyTest(unittest.TestCase):
    def test_selects_multiple_choice_playbook_from_options(self):
        playbook = select_question_playbook(
            "Which option is visible?\nA. aircraft museum\nB. submarine\nC. mountain road"
        )

        self.assertEqual(playbook.name, "multiple_choice")
        self.assertIn("inspect_segment", " ".join(playbook.instructions))
        self.assertIn("verify option consistency", " ".join(playbook.sufficiency_rules))

    def test_selects_temporal_ordering_playbook_from_question(self):
        playbook = select_question_playbook("What happens before the person opens the door?")

        self.assertEqual(playbook.name, "timeline_ordering")
        self.assertIn("timestamped", " ".join(playbook.sufficiency_rules))

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


if __name__ == "__main__":
    unittest.main()
