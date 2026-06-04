import unittest

from visual_coding_agent_harness.agents.question_policy import select_question_playbook


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

        self.assertEqual(playbook.name, "temporal_ordering")
        self.assertIn("timestamped", " ".join(playbook.sufficiency_rules))


if __name__ == "__main__":
    unittest.main()
