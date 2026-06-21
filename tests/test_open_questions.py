import unittest

from visual_coding_agent_harness.workspace.open_questions import build_question_context, exploration_question, rewrite_exploration_question_with_model
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend


MCQ_QUESTION = (
    "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
    "Question: What is the video mainly about?\n"
    "Options:\n"
    "A. The fall of Rome\n"
    "B. Why the Austro-Hungarian Empire was divided\n"
    "C. A battle timeline\n"
    "D. How the Austro-Hungarian Empire rose and fell\n"
    "Select option A, B, C, or D."
)
MCQ_OPTIONS = [
    "The fall of Rome",
    "Why the Austro-Hungarian Empire was divided",
    "A battle timeline",
    "How the Austro-Hungarian Empire rose and fell",
]
TEMPORAL_MCQ_QUESTION = (
    "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
    "Question: In what order does the video present the four sculptures?\n"
    "Options:\n"
    'A. "The Rape of Persephone", "Apollo and Daphne", "David", "Aeneas fleeing Troy".\n'
    'B. "David", "Aeneas fleeing Troy", "Apollo and Daphne", "The Rape of Persephone".\n'
    'C. "Apollo and Daphne", "The Rape of Persephone", "Aeneas fleeing Troy", "David".\n'
    'D. "Aeneas fleeing Troy", "David", "The Rape of Persephone", "Apollo and Daphne".\n'
    "Select option A, B, C, or D."
)
TEMPORAL_OPTION_TEXTS = [
    '"The Rape of Persephone", "Apollo and Daphne", "David", "Aeneas fleeing Troy".',
    '"David", "Aeneas fleeing Troy", "Apollo and Daphne", "The Rape of Persephone".',
    '"Apollo and Daphne", "The Rape of Persephone", "Aeneas fleeing Troy", "David".',
    '"Aeneas fleeing Troy", "David", "The Rape of Persephone", "Apollo and Daphne".',
]


def assert_no_mcq_leak(testcase: unittest.TestCase, prompt: str, option_texts=MCQ_OPTIONS) -> None:
    text = str(prompt)
    testcase.assertNotIn("Options:", text)
    testcase.assertNotIn("Candidate options:", text)
    for label in ("A.", "B.", "C.", "D."):
        testcase.assertNotIn(label, text)
    testcase.assertNotRegex(text, r"\boption\s+[A-D]\b")
    for option in option_texts:
        testcase.assertNotIn(option, text)


class OpenQuestionsTest(unittest.TestCase):
    def test_question_context_keeps_raw_mcq_for_planner_and_answer(self):
        context = build_question_context(MCQ_QUESTION)

        self.assertEqual(context.planner_question, MCQ_QUESTION)
        self.assertEqual(context.answer_question, MCQ_QUESTION)
        self.assertEqual(context.navigation_question, MCQ_QUESTION)
        self.assertIn("A. The fall of Rome", context.planner_question)
        assert_no_mcq_leak(self, context.vlm_safe_question)

    def test_exploration_question_strips_mcq_labels_options_and_answer_letter_instruction(self):
        question = exploration_question(MCQ_QUESTION)

        self.assertIn("What is the video mainly about?", question)
        self.assertIn("Do not choose an option.", question)
        assert_no_mcq_leak(self, question)
        self.assertNotIn("Answer with exactly one option letter", question)
        self.assertNotIn("Select", question)

    def test_exploration_question_preserves_route_hint_without_option_vote(self):
        question = exploration_question(MCQ_QUESTION, route_hint="Inspect the opening montage.")

        self.assertIn("What is the video mainly about?", question)
        self.assertIn("Inspect the opening montage.", question)
        self.assertIn("Do not choose an option.", question)
        assert_no_mcq_leak(self, question)

    def test_model_rewrite_returns_option_blind_open_exploration_question(self):
        class RewriteBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                return BackendResponse(
                    text=(
                        '{"exploration_question":"Describe the overall topic and narrative arc of the video. '
                        'Identify how the Austro-Hungarian Empire is covered, including time span, major stages, '
                        'and whether it covers origin, growth, stability, decline, collapse, causes, or consequences.",'
                        '"focus_points":["overall topic","narrative arc"],'
                        '"target_entities":["Austro-Hungarian Empire"]}'
                    )
                )

        backend = RewriteBackend()
        rewrite = rewrite_exploration_question_with_model(backend, question=MCQ_QUESTION, route_hint="gist_global")

        self.assertTrue(rewrite.used_model)
        self.assertEqual(backend.requests[0].task, "rewrite_exploration_question")
        self.assertIn("Austro-Hungarian Empire", rewrite.exploration_question)
        assert_no_mcq_leak(self, rewrite.exploration_question)

    def test_model_rewrite_falls_back_when_output_copies_option_surface(self):
        class LeakyBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                return BackendResponse(
                    text='{"exploration_question":"Compare option B. Why the Austro-Hungarian Empire was divided."}'
                )

        rewrite = rewrite_exploration_question_with_model(LeakyBackend(), question=MCQ_QUESTION)

        self.assertFalse(rewrite.used_model)
        self.assertEqual(rewrite.fallback_reason, "rewrite_option_leak")
        assert_no_mcq_leak(self, rewrite.exploration_question)

    def test_model_rewrite_falls_back_when_output_lists_option_values_case_insensitively(self):
        question = (
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: Which is the best title of the video?\n"
            "Options:\n"
            "A. Wild animals\n"
            "B. Ocean animals\n"
            "C. Diverse fishes\n"
            "D. Protect the sea"
        )

        class CandidateValueBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                return BackendResponse(
                    text=(
                        '{"exploration_question":"Determine whether the focus is on wild animals, ocean animals, '
                        'diverse fishes, or a call to protect the sea.","target_entities":["wild animals",'
                        '"ocean animals","diverse fishes","protect the sea"]}'
                    )
                )

        rewrite = rewrite_exploration_question_with_model(CandidateValueBackend(), question=question)

        self.assertFalse(rewrite.used_model)
        self.assertEqual(rewrite.fallback_reason, "rewrite_option_leak")
        lowered = rewrite.exploration_question.lower()
        self.assertNotIn("wild animals", lowered)
        self.assertNotIn("ocean animals", lowered)
        self.assertNotIn("diverse fishes", lowered)
        self.assertNotIn("protect the sea", lowered)

    def test_temporal_rewrite_keeps_targets_as_metadata_not_local_tool_instruction(self):
        class CandidateOrderBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                return BackendResponse(
                    text=(
                        '{"exploration_question":"Determine the order in which the video presents these sculptures: '
                        'The Rape of Persephone, Apollo and Daphne, David, and Aeneas fleeing Troy.",'
                        '"focus_points":["presentation order"],'
                        '"target_entities":["The Rape of Persephone","Apollo and Daphne","David","Aeneas fleeing Troy"]}'
                    )
                )

        rewrite = rewrite_exploration_question_with_model(
            CandidateOrderBackend(),
            question=TEMPORAL_MCQ_QUESTION,
            route_hint="temporal_order",
        )

        self.assertTrue(rewrite.used_model)
        self.assertIn("Describe the video segment by segment", rewrite.exploration_question)
        self.assertIn("artworks", rewrite.exploration_question)
        self.assertNotIn("target items", rewrite.exploration_question)
        self.assertNotIn("unordered list", rewrite.exploration_question)
        self.assertNotIn("present or absent", rewrite.exploration_question)
        for target in ("Aeneas fleeing Troy", "Apollo and Daphne", "David", "The Rape of Persephone"):
            self.assertNotIn(target, rewrite.exploration_question)
        assert_no_mcq_leak(self, rewrite.exploration_question, TEMPORAL_OPTION_TEXTS)
        self.assertEqual(
            rewrite.target_entities,
            ("Aeneas fleeing Troy", "Apollo and Daphne", "David", "The Rape of Persephone"),
        )


if __name__ == "__main__":
    unittest.main()
