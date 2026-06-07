import unittest

from visual_coding_agent_harness.agents.open_questions import exploration_question


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


if __name__ == "__main__":
    unittest.main()
