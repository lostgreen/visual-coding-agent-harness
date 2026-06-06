import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.skills.predicates import (
    direct_floor_holds,
    every_event_has_confirmed_timestamp,
    grounding_quality_floor,
    no_decisive_weak_grounding,
    selected_option_has_structured_support,
    temporal_order_consistent,
)
from visual_coding_agent_harness.agents.skills.specs import (
    builtin_skill_registry,
    compile_skill_program,
    select_skill,
)
from visual_coding_agent_harness.agents.distill import distill
from visual_coding_agent_harness.agents.iterative_agent import _hard_skill_gate_reason
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.workspace import EvidenceRecord, EvidenceWorkspace


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

        self.assertEqual(skill.name, "main_idea")
        self.assertEqual(program[0]["tool"], "global_gist")
        self.assertEqual(program[0]["args"]["video_path"], "demo.mp4")
        self.assertEqual(program[0]["args"]["duration_sec"], 30.0)
        self.assertIn("global_gist", prompt_context)
        self.assertNotIn("timeline_ordering", prompt_context)

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
            distilled_records = distill(observation, workspace)
            for record in distilled_records:
                workspace.write_evidence(record)
            workspace.write_ledger_entry(observation, parent_records=distilled_records)

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
        self.assertTrue(direct_floor_holds(table, selected_option="A").passed)
        self.assertTrue(direct_floor_holds(table, selected_option="D").passed)

    def test_grounded_factual_requires_visual(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="grounding_floor")
            mapped = _write_evidence_chain(
                workspace,
                option="A",
                grounding_quality="inferred",
                tool="caption_segment",
            )

            reason = grounding_quality_floor([mapped], workspace=workspace, require_visual=True)

            self.assertIsNotNone(reason)
            self.assertIn("no visually_confirmed", reason or "")

    def test_gist_qa_allows_global_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="grounding_floor")
            mapped = _write_evidence_chain(
                workspace,
                option="A",
                grounding_quality="inferred",
                tool="caption_segment",
            )

            self.assertIsNone(grounding_quality_floor([mapped], workspace=workspace, require_visual=False))

    def test_visual_plus_global_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="grounding_floor")
            weak_mapped = _write_evidence_chain(
                workspace,
                option="A",
                grounding_quality="inferred",
                tool="caption_segment",
            )
            visual_mapped = _write_evidence_chain(
                workspace,
                option="A",
                grounding_quality="visually_confirmed",
                tool="vision_read",
            )

            self.assertIsNone(
                grounding_quality_floor([weak_mapped, visual_mapped], workspace=workspace, require_visual=True)
            )

    def test_hard_skill_gate_uses_mapped_grounding_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="grounding_floor")
            _write_evidence_chain(
                workspace,
                option="A",
                grounding_quality="inferred",
                tool="caption_segment",
            )
            table = workspace.evidence_table_v2(
                question="Which option is shown?",
                options=["A. global synopsis", "B. local fact"],
            )

            reason = _hard_skill_gate_reason(
                skill_name="grounded_factual",
                question="Which option is shown?\nA. global synopsis\nB. local fact",
                table=table,
                selected_option="A",
                citations=["obs_0001"],
                workspace=workspace,
            )

            self.assertEqual(reason, "selected_option_has_structured_support")

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

    def test_interpreter_feeds_grounding_candidates_into_next_foreach(self):
        registry = ToolRegistry()

        @tool(name="ground_question", description="Ground an event query.")
        def ground_question(query: str, top_k: int = 2):
            return {
                "claim": f"found windows for {query}",
                "confidence": 0.8,
                "candidates": [
                    {"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 5.0},
                    {"segment_id": "seg_0002", "start_sec": 5.0, "end_sec": 10.0},
                ][:top_k],
            }

        @tool(name="vision_read", description="Read one grounded candidate.")
        def vision_read(window: dict, ask_for: str):
            return {
                "claim": f"{window['segment_id']} confirms {ask_for}",
                "confidence": 0.86,
                "regions": [window],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(ground_question)
        registry.register(vision_read)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="dynamic_candidates")
            interpreter = ProgramInterpreter(registry=registry, workspace=workspace)

            result = interpreter.run(
                [
                    {"tool": "ground_question", "args": {"query": "red aircraft", "top_k": 2}, "assign": "cand"},
                    {
                        "tool": "vision_read",
                        "foreach": "candidates",
                        "args": {"window": "{candidate}", "ask_for": "red aircraft"},
                        "assign": "fact[{candidate}]",
                    },
                ]
            )

            self.assertEqual(result.observation_ids, ["obs_0001", "obs_0002", "obs_0003"])
            self.assertEqual(result.assignments["cand"], "obs_0001")
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("seg_0001 confirms red aircraft", ledger)
            self.assertIn("seg_0002 confirms red aircraft", ledger)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("foreach_slot_update", trace)

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


def _write_evidence_chain(
    workspace: EvidenceWorkspace,
    *,
    option: str,
    grounding_quality: str,
    tool: str,
) -> EvidenceRecord:
    observation = workspace.write_observation(
        tool_name=tool,
        claim=f"{option} is supported by {grounding_quality}.",
        confidence=0.86,
        raw_output={
            "grounding_quality": grounding_quality,
            "candidate_option_relations": [{"option": option, "relation": "support", "strength": 0.86}],
        },
    )
    distilled = EvidenceRecord(
        evidence_id=workspace.next_evidence_id("distilled"),
        stage="distilled",
        parent_id=None,
        tool=tool,
        observation_id=observation.observation_id,
        frame_set_id=None,
        content={"claim": observation.claim},
        grounding_quality=grounding_quality,  # type: ignore[arg-type]
        confidence=0.86,
        created_at=1.0,
    )
    workspace.write_evidence(distilled)
    ledger = workspace.write_ledger_entry(observation, parent_records=[distilled])[0]
    changed = workspace.annotate_candidate_option_relations(
        observation_ids=[observation.observation_id],
        relations=[
            {
                "option": option,
                "relation": "support",
                "strength": 0.86,
                "observation_id": observation.observation_id,
                "parent_evidence_id": ledger.evidence_id,
            }
        ],
    )
    assert changed == 1
    mapped = workspace.mapped_evidence_records(
        observation_ids=[observation.observation_id],
        selected_option=option,
    )
    assert mapped
    return mapped[-1]


if __name__ == "__main__":
    unittest.main()
