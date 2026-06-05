import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.skills.predicates import (
    direct_floor_holds,
    every_event_has_confirmed_timestamp,
    no_decisive_weak_grounding,
    selected_option_has_structured_support,
    temporal_order_consistent,
)
from visual_coding_agent_harness.agents.skills.specs import (
    builtin_skill_registry,
    compile_skill_program,
    select_skill,
)
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class V4FoundationTest(unittest.TestCase):
    def test_builtin_skill_registry_selects_and_compiles_only_selected_skill(self):
        registry = builtin_skill_registry()
        question = "What's the main idea of the video?\nA. cooking\nD. aviation documentary"

        skill = select_skill(question, registry=registry)
        program = compile_skill_program(
            skill,
            slots={
                "video_id": "demo.mp4",
                "question": question,
                "options": ["A. cooking", "D. aviation documentary"],
                "duration_sec": 30.0,
            },
        )
        prompt_context = skill.prompt_context()

        self.assertEqual(skill.name, "gist_qa")
        self.assertEqual(program[0]["tool"], "global_gist")
        self.assertEqual(program[0]["args"]["video_path"], "demo.mp4")
        self.assertEqual(program[0]["args"]["duration_sec"], 30.0)
        self.assertIn("global_gist", prompt_context)
        self.assertNotIn("temporal_ordering", prompt_context)

    def test_evidence_table_v2_includes_vision_read_rows_without_worker_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="v4_table")
            observation = workspace.write_observation(
                tool_name="vision_read",
                input_artifacts=["clip_red.mp4"],
                claim="The red object is visible.",
                confidence=0.91,
                regions=[{"start_sec": 20.0, "end_sec": 22.0}],
                limitations="Directly visible.",
                raw_output={
                    "event_label": "red object",
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [
                        {"option": "B", "relation": "support", "strength": 0.91, "assigned_by": "answer_agent"}
                    ],
                },
            )
            workspace.write_ledger_entry(observation)

            table = workspace.evidence_table_v2(
                question="Which option is shown?",
                options=["A. blue object", "B. red object"],
            )

            self.assertEqual(table["schema_version"], "EvidenceTableV2")
            self.assertEqual(table["groups"]["B"][0]["obs_id"], "obs_0001")
            self.assertEqual(table["groups"]["B"][0]["event_label"], "red object")
            self.assertEqual(table["groups"]["B"][0]["grounding_quality"], "visually_confirmed")
            self.assertNotIn("supported_option_letter", table["groups"]["B"][0])
            self.assertFalse(table["groups"]["B"][0]["legacy_worker_vote"])

    def test_answer_agent_relations_can_annotate_existing_visual_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="answer_relation_annotation")
            observation = workspace.write_observation(
                tool_name="vision_read",
                input_artifacts=["clip_red.mp4"],
                claim="The localized window shows a red object.",
                confidence=0.87,
                regions=[{"start_sec": 20.0, "end_sec": 22.0}],
                raw_output={
                    "event_label": "red object",
                    "grounding_quality": "visually_confirmed",
                },
            )

            changed = workspace.annotate_candidate_option_relations(
                observation_ids=[observation.observation_id],
                relations=[
                    {
                        "option": "B",
                        "relation": "support",
                        "strength": 0.87,
                        "observation_id": observation.observation_id,
                    }
                ],
            )
            table = workspace.evidence_table_v2(
                question="Which option is shown?",
                options=["A. blue object", "B. red object"],
            )

            self.assertEqual(changed, 1)
            self.assertEqual(table["groups"]["B"][0]["obs_id"], observation.observation_id)
            self.assertEqual(table["groups"]["B"][0]["candidate_option_relations"][0]["assigned_by"], "answer_agent")

    def test_predicates_check_structured_temporal_support(self):
        table = {
            "options": ["A. red object then blue object", "B. blue object then red object"],
            "groups": {
                "A": [
                    {
                        "obs_id": "obs_red",
                        "tool": "vision_read",
                        "event_label": "red object",
                        "time_range": [10.0, 12.0],
                        "grounding_quality": "visually_confirmed",
                        "candidate_option_relations": [
                            {"option": "A", "relation": "support", "strength": 0.9}
                        ],
                        "confidence": 0.9,
                    }
                ],
                "B": [],
                "unassigned": [
                    {
                        "obs_id": "obs_blue",
                        "tool": "vision_read",
                        "event_label": "blue object",
                        "time_range": [20.0, 22.0],
                        "grounding_quality": "visually_confirmed",
                        "candidate_option_relations": [],
                        "confidence": 0.88,
                    }
                ],
            },
            "rows": [],
        }
        table["rows"] = table["groups"]["A"] + table["groups"]["unassigned"]

        timestamp_result = every_event_has_confirmed_timestamp(
            table,
            expected_events=["red object", "blue object"],
        )
        support_result = selected_option_has_structured_support(table, selected_option="A")
        order_result = temporal_order_consistent(
            table,
            selected_option="A",
            expected_events=["red object", "blue object"],
        )

        self.assertTrue(timestamp_result.passed)
        self.assertTrue(support_result.passed)
        self.assertTrue(order_result.passed)

    def test_predicates_block_weak_winner_and_floor_regression(self):
        table = {
            "options": ["A. local weak guess", "D. whole-video synopsis"],
            "groups": {
                "A": [
                    {
                        "obs_id": "obs_local",
                        "tool": "caption_segment",
                        "grounding_quality": "inferred",
                        "candidate_option_relations": [{"option": "A", "relation": "support", "strength": 0.95}],
                        "confidence": 0.95,
                    }
                ],
                "D": [
                    {
                        "obs_id": "obs_global",
                        "tool": "global_gist",
                        "grounding_quality": "global_sparse",
                        "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.76}],
                        "confidence": 0.76,
                    }
                ],
            },
            "rows": [],
        }
        table["rows"] = table["groups"]["A"] + table["groups"]["D"]

        self.assertFalse(no_decisive_weak_grounding(table, selected_option="A").passed)
        self.assertFalse(direct_floor_holds(table, selected_option="A").passed)
        self.assertTrue(direct_floor_holds(table, selected_option="D").passed)

    def test_interpreter_foreach_fills_slots_and_collects_assignments(self):
        registry = ToolRegistry()

        @tool(name="ground_question", description="Ground an event query.")
        def ground_question(query: str):
            return {
                "claim": f"candidate for {query}",
                "confidence": 0.8,
                "regions": [{"query": query}],
            }

        registry.register(ground_question)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="foreach")
            interpreter = ProgramInterpreter(registry=registry, workspace=workspace)

            result = interpreter.run(
                [
                    {
                        "tool": "ground_question",
                        "foreach": "events",
                        "args": {"query": "{event}"},
                        "assign": "cand[{event}]",
                    }
                ],
                slots={"events": ["red object", "blue object"]},
            )

            self.assertEqual(result.observation_ids, ["obs_0001", "obs_0002"])
            self.assertEqual(result.assignments["cand[red object]"], "obs_0001")
            self.assertEqual(result.assignments["cand[blue object]"], "obs_0002")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"query": "red object"', trace)
            self.assertIn('"query": "blue object"', trace)

    def test_interpreter_can_stop_when_sufficiency_is_met(self):
        registry = ToolRegistry()

        @tool(name="read_once", description="Read one fact.")
        def read_once(label: str):
            return {"claim": label, "confidence": 1.0}

        registry.register(read_once)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="sufficiency")
            interpreter = ProgramInterpreter(registry=registry, workspace=workspace)

            result = interpreter.run(
                [
                    {"tool": "read_once", "args": {"label": "enough"}, "assign": "first"},
                    {"tool": "read_once", "args": {"label": "should not run"}, "assign": "second"},
                ],
                sufficiency_predicate=lambda workspace, assignments: "first" in assignments,
            )

            self.assertEqual(result.observation_ids, ["obs_0001"])
            self.assertEqual(result.assignments, {"first": "obs_0001"})
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("sufficiency_stop", trace)
            self.assertNotIn("should not run", trace)


if __name__ == "__main__":
    unittest.main()
