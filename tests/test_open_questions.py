import unittest

from visual_coding_agent_harness.agents.open_questions import exploration_question, rewrite_exploration_question_with_model
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


if __name__ == "__main__":
    unittest.main()
