import tempfile
import json
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import (
    AgentBudget,
    IterativeVisualAgent,
    _semantic_question_text,
    _skill_target_facts,
)
from visual_coding_agent_harness.agents.prompt_stack import (
    _final_gate_block,
    _normalization_notes_body,
    _tool_schema_block,
    build_replanning_prompt,
    compose_replanning_prompt_blocks,
    render_prompt_blocks,
)
from visual_coding_agent_harness.agents.context_budget import default_context_budget_allocator
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment, fixed_window_scene_index
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class RecordingBackend(VisionLanguageBackend):
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task != "replan":
            return BackendResponse(text="tool backend response")
        if self.responses:
            return BackendResponse(text=self.responses.pop(0))
        return BackendResponse(text='{"status": "final", "answer": "done", "citations": []}')


class PromptStackAndSkillRuntimeTest(unittest.TestCase):
    def test_replanning_prompt_is_composed_from_named_contract_blocks(self):
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        blocks = compose_replanning_prompt_blocks(
            question="Which event happened first?\nA. red then blue\nB. blue then red",
            scene_index=scene_index,
            ledger_text="# Compact Evidence Context\n(none)",
            round_number=2,
            budget=AgentBudget(max_rounds=4),
            inspected_segment_ids=["seg_0001"],
            tool_class_counts={"cheap": 1, "expensive": 0, "verifier": 0},
            answer_feedback=["confirm timestamps for both events"],
            reflection_memory=["temporal_order: confirm every event timestamp before comparing options"],
        )
        prompt = render_prompt_blocks(blocks)

        self.assertEqual(
            [block.name for block in blocks],
            [
                "base_identity",
                "route_playbook",
                "skill_catalog",
                "trajectory_snapshot",
                "evidence_snapshot",
                "hypothesis",
                "scene_index_snapshot",
                "answer_feedback",
                "reflection_memory",
                "budget_snapshot",
                "tool_schema",
                "final_gate",
                "response_contract",
            ],
        )
        self.assertLess(prompt.index("# Base Identity"), prompt.index("# Route Playbook"))
        self.assertLess(prompt.index("# Route Playbook"), prompt.index("# Skill Catalog"))
        self.assertLess(prompt.index("# Evidence Snapshot"), prompt.index("# Tool Schema"))
        self.assertLess(prompt.index("# Compact Scene Index"), prompt.index("# Tool Schema"))
        self.assertLess(prompt.index("# Evidence Snapshot"), prompt.index("# Hypothesis"))
        self.assertIn("Allowed ReAct actions: ground_question, vision_read, answer_agent, verify", prompt)
        self.assertIn("Available skills:", prompt)
        self.assertIn('"skill": string', prompt)
        self.assertIn("Select the skill that best matches this case", prompt)
        self.assertNotIn("Recommended fallback skill", prompt)
        self.assertIn("timeline_ordering@v1", prompt)
        self.assertIn("confirm every event timestamp", prompt)
        self.assertIn("Final answers require at least one evidence-grade visual observation", prompt)

    def test_slot_prompt_contains_all_four_sections_and_budget_report(self):
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)
        allocator = default_context_budget_allocator(total_budget_tokens=400)

        prompt, report = build_replanning_prompt(
            question="What is visible?",
            scene_index=scene_index,
            ledger_text="# Compact Evidence Context\nobs_0001 | red car",
            round_number=1,
            budget=AgentBudget(max_rounds=3),
            allocator=allocator,
            answer_feedback=["confirm red car"],
        )

        self.assertLess(prompt.index("## Task"), prompt.index("## Trajectory"))
        self.assertLess(prompt.index("## Evidence"), prompt.index("## Compact scene index"))
        self.assertLess(prompt.index("## Compact scene index"), prompt.index("## Current budgets"))
        self.assertLess(prompt.index("## Current budgets"), prompt.index("## Tooling"))
        self.assertIn("obs_0001 | red car", prompt)
        self.assertEqual(report.turn_index, 0)
        self.assertIn("task", report.used_tokens_per_slot)
        self.assertIn("tooling", report.used_tokens_per_slot)
        self.assertIn("target_coverage(targets", prompt)
        self.assertIn("read_segment_detail(segment_id", prompt)
        self.assertNotIn("video_ls(query", prompt)
        self.assertNotIn("caption_segments(segment_ids", prompt)
        self.assertIn("verify_ledger_answer(answer: str, question: str", prompt)
        self.assertNotIn("verify_ledger_answer(answer: str, ledger_text", prompt)

    def test_tool_schema_filters_to_active_skill_allowed_actions(self):
        rendered = _tool_schema_block(
            option_blind=True,
            active_skill="timeline_ordering@v1",
            exhausted=frozenset(),
        )

        self.assertIn("caption_segment(", rendered)
        self.assertIn("read_segment_detail(", rendered)
        self.assertIn("locate_targets_in_segment(", rendered)
        self.assertIn("verify_segment_anchors(", rendered)
        self.assertIn("read_timeline_sorted(", rendered)
        self.assertNotIn("commit_map_proposals(", rendered)
        self.assertNotIn("update_hypothesis_slot(", rendered)
        self.assertNotIn("grep_evidence(", rendered)
        self.assertNotIn("zoom(", rendered)
        self.assertNotIn("expand_window(", rendered)
        self.assertNotIn("read_segment(", rendered)

    def test_tool_schema_marks_exhausted_tools_inline(self):
        rendered = _tool_schema_block(
            option_blind=True,
            active_skill="main_idea@v1",
            exhausted=frozenset({"global_gist"}),
        )

        self.assertIn("global_gist(", rendered)
        self.assertIn("=exhausted", rendered)

    def test_final_gate_guides_temporal_route_to_locate_then_verify(self):
        body = _final_gate_block(
            final_round_reserved=False,
            option_blind=True,
            route="temporal_order",
        )

        self.assertIn("locate_targets_in_segment", body)
        self.assertIn("verify_segment_anchors", body)
        self.assertNotIn("Main-idea answers", body)
        self.assertNotIn("scan_segment", body)
        self.assertNotIn("Use zoom", body)
        self.assertIn("Do not repeat already inspected segments", body)

    def test_final_gate_keeps_main_idea_rule_for_gist_route(self):
        body = _final_gate_block(
            final_round_reserved=False,
            option_blind=True,
            route="gist_global",
        )

        self.assertIn("Main-idea answers", body)

    def test_slot_prompt_includes_structured_evidence_status_summary(self):
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)
        allocator = default_context_budget_allocator(total_budget_tokens=800)

        prompt, _report = build_replanning_prompt(
            question="Which option is visible?\nA. blue car\nB. red aircraft",
            scene_index=scene_index,
            ledger_text="# Compact Evidence Context\nobs_0001 | red aircraft",
            round_number=2,
            budget=AgentBudget(max_rounds=4),
            allocator=allocator,
            evidence_status_summary={
                "option_coverage": "1/2",
                "coverage_pct": 0.5,
                "duplicate_observations": 1,
                "total_evidence_rows": 2,
                "option_status": {
                    "A": {"strong_evidence_count": 0, "weak_evidence_count": 0, "has_visual_citation": False},
                    "B": {"strong_evidence_count": 2, "weak_evidence_count": 0, "has_visual_citation": True},
                },
                "hypothesis_gaps": ["entered upper class"],
            },
        )

        self.assertIn("Evidence status summary:", prompt)
        self.assertIn("option_coverage: 1/2", prompt)
        self.assertIn("duplicate_observations: 1", prompt)
        self.assertIn("B: strong=2 weak=0 visual=yes", prompt)
        self.assertIn("hypothesis_gaps: entered upper class", prompt)

    def test_prompt_omits_exhausted_global_gist_from_skill_catalog(self):
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)
        allocator = default_context_budget_allocator(total_budget_tokens=12000)

        prompt, _report = build_replanning_prompt(
            question="What is the main idea of the video?",
            scene_index=scene_index,
            ledger_text="# Compact Evidence Context\n(none)",
            round_number=2,
            budget=AgentBudget(max_rounds=3),
            allocator=allocator,
            exhausted_tools=frozenset({"global_gist"}),
        )

        catalog_block = prompt.split("# Skill Catalog", 1)[1].split("# ", 1)[0]
        main_idea_line = next(line for line in catalog_block.splitlines() if line.startswith("- main_idea@"))
        allowed_actions = main_idea_line.split("allowed_actions=", 1)[1].split(";", 1)[0]
        self.assertNotIn("global_gist", allowed_actions.split("(", 1)[0])
        self.assertIn("(global_gist=exhausted)", main_idea_line)

    def test_normalization_next_action_is_rendered_as_do_next(self):
        rendered = _normalization_notes_body(
            [
                {
                    "tool": "global_gist",
                    "reason": "repair_repeated_main_idea_global_gist_to_vision_read",
                    "original": {"tool": "global_gist"},
                    "resolved": {"tool": "vision_read", "segment_id": "seg_0002"},
                    "next_action": "Stop requesting global_gist. Read obs_0001.",
                }
            ]
        )

        self.assertIn("DO NEXT: Stop requesting global_gist", rendered)

    def test_ground_question_returns_candidate_windows_without_option_vote(self):
        video_map = VideoMap(
            video_path="/videos/demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
                VideoMapSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="red car appears"),
                VideoMapSegment(segment_id="seg_0003", start_sec=60.0, end_sec=90.0, low_fps_caption="blue car appears"),
            ],
        )
        registry = build_video_navigation_registry(video_map)

        result = registry.execute("ground_question", {"query": "red car", "top_k": 2})

        self.assertIn("candidates", result)
        self.assertEqual(result["candidates"][0]["segment_id"], "seg_0002")
        self.assertEqual(result["candidates"][0]["start_sec"], 30.0)
        self.assertEqual(result["candidates"][0]["end_sec"], 60.0)
        self.assertIn("reason", result["candidates"][0])
        self.assertIn("modality", result["candidates"][0])
        self.assertIn("confidence", result["candidates"][0])
        self.assertNotIn("supported_option", result)
        self.assertNotIn("answer_option", result)
        self.assertNotIn("final_answer", result)

    def test_ground_question_normalizes_stopword_heavy_queries_before_search(self):
        video_map = VideoMap(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="the and of"),
                VideoMapSegment(
                    segment_id="seg_0002",
                    start_sec=30.0,
                    end_sec=60.0,
                    low_fps_caption="The Rape of Persephone appears in the sculpture discussion.",
                ),
            ],
        )
        registry = build_video_navigation_registry(video_map)

        result = registry.execute("ground_question", {"query": '"The rape of Persephone" and the of', "top_k": 1})

        self.assertEqual(result["normalized_query"], "rape Persephone")
        self.assertEqual(result["candidates"][0]["segment_id"], "seg_0002")

    def test_skill_target_facts_strip_wrappers_and_compile_option_facts(self):
        wrapped = (
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: How was his life journey according to the video?\n"
            "Options:\n"
            "A. Borned with humble background and lived in seclusion in a farmhouse.\n"
            "B. Borned with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
        )
        temporal = (
            "Question: What is the correct order?\n"
            "Options:\n"
            'A. "The rape of Persephone", "Apollo and Daphne", "David" and '
            '"Aeneas, Anchises, and Ascanius fleeing Troy".\n'
        )

        self.assertEqual(_semantic_question_text(wrapped), "How was his life journey according to the video?")
        targets = _skill_target_facts(question=wrapped, skill_name="grounded_factual_qa")
        self.assertIn("Borned with humble background", targets)
        self.assertIn("entered the upper class", targets)
        self.assertIn("lived in seclusion in a farmhouse", targets)
        self.assertTrue(all("VideoMME" not in target for target in targets))
        self.assertEqual(
            _skill_target_facts(question=temporal, skill_name="timeline_ordering"),
            [
                "The rape of Persephone",
                "Apollo and Daphne",
                "David",
                "Aeneas, Anchises, and Ascanius fleeing Troy",
            ],
        )

    def test_hard_grounded_skill_runtime_runs_ground_then_vision_then_answer(self):
        backend = RecordingBackend()
        registry = ToolRegistry()

        @tool(name="ground_question", description="Ground a query to candidate windows.")
        def ground_question(query: str, top_k: int = 3):
            return {
                "claim": f"grounded {query}",
                "confidence": 0.9,
                "candidates": [
                    {
                        "segment_id": "seg_0001",
                        "start_sec": 0.0,
                        "end_sec": 12.0,
                        "reason": "caption match",
                        "modality": "caption",
                        "confidence": 0.9,
                    }
                ],
                "regions": [
                    {
                        "segment_id": "seg_0001",
                        "start_sec": 0.0,
                        "end_sec": 12.0,
                        "reason": "caption match",
                        "modality": "caption",
                        "confidence": 0.9,
                    }
                ],
            }

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = ""):
            return {
                "claim": "The localized window shows a red car.",
                "confidence": 0.92,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
                "candidate_option_relations": [
                    {"option": "B", "relation": "support", "strength": 0.92, "assigned_by": "answer_agent"}
                ],
                "event_label": event_label or ask_for,
                "time_range": [start_sec, end_sec],
            }

        registry.register(ground_question)
        registry.register(vision_read)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=12.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="hard_grounded_skill")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(hard_skill_runtime=True, reserve_final_round=False),
            )

            result = agent.run(
                question="Which object is visible?\nA. blue car\nB. red car",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "B. red car")
            self.assertEqual(
                [step["tool"] for step in result.rounds[0].program],
                ["ground_question", "vision_read", "ground_question", "vision_read"],
            )
            self.assertEqual([request.task for request in backend.requests], [])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("hard_skill_runtime", trace)
            self.assertIn("followup_attempt", trace)

    def test_hard_skill_runtime_continues_followup_chunks_before_abstaining(self):
        backend = RecordingBackend()
        registry = ToolRegistry()

        @tool(name="ground_question", description="Ground a query to candidate windows.")
        def ground_question(query: str, top_k: int = 3):
            return {
                "claim": f"grounded {query}",
                "confidence": 0.9,
                "candidates": [
                    {
                        "segment_id": "seg_0001",
                        "start_sec": 0.0,
                        "end_sec": 12.0,
                        "reason": "caption match",
                        "modality": "caption",
                        "confidence": 0.9,
                    }
                ],
                "regions": [{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
            }

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = ""):
            return {
                "claim": f"The localized window mentions {ask_for}.",
                "confidence": 0.7,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
                "event_label": event_label or ask_for,
            }

        registry.register(ground_question)
        registry.register(vision_read)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=12.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="hard_skill_followup_chunks")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, max_tool_calls_per_round=2, hard_skill_runtime=True, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "Question: Which artwork appears in the video?\n"
                    "Options:\n"
                    'A. "The rape of Persephone"\n'
                    'B. "Apollo and Daphne"\n'
                    'C. "David"\n'
                    'D. "Aeneas, Anchises, and Ascanius fleeing Troy".\n'
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "need_more_evidence")
            self.assertEqual(len(result.rounds), 2)
            self.assertEqual(
                [step["tool"] for round_item in result.rounds for step in round_item.program],
                [
                    "ground_question",
                    "vision_read",
                    "ground_question",
                    "vision_read",
                    "ground_question",
                    "vision_read",
                    "ground_question",
                    "vision_read",
                ],
            )

    def test_hard_skill_need_more_hands_off_to_main_replanning_loop(self):
        class HandoffBackend(RecordingBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    self.requests.append(request)
                    return BackendResponse(
                        text='{"answer": "D. enough after follow-up", "citations": ["obs_0002"], '
                        '"confidence": 0.7, "candidate_option_relations": ['
                        '{"option": "D", "relation": "support", "strength": 0.7, "observation_id": "obs_0002"}]}'
                    )
                return super().generate(request)

        backend = HandoffBackend(
            [
                '{"status": "final", "answer": "D. enough after follow-up", "citations": ["obs_0002"], "confidence": 0.7}'
            ]
        )
        registry = ToolRegistry()

        @tool(name="ground_question", description="Ground a query to candidate windows.")
        def ground_question(query: str, top_k: int = 3):
            return {
                "claim": f"grounded {query}",
                "confidence": 0.9,
                "candidates": [
                    {
                        "segment_id": "seg_0001",
                        "start_sec": 0.0,
                        "end_sec": 12.0,
                        "reason": "caption match",
                        "modality": "caption",
                        "confidence": 0.9,
                    }
                ],
                "regions": [{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
            }

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = ""):
            return {
                "claim": f"The localized window mentions {ask_for}.",
                "confidence": 0.7,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
                "event_label": event_label or ask_for,
            }

        registry.register(ground_question)
        registry.register(vision_read)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=12.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=12.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="hard_skill_handoff")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, max_tool_calls_per_round=2, hard_skill_runtime=True, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "Question: Which artwork appears in the video?\n"
                    "Options:\n"
                    'A. "The rape of Persephone"\n'
                    'B. "Apollo and Daphne"\n'
                    'C. "David"\n'
                    'D. "Aeneas, Anchises, and Ascanius fleeing Troy".\n'
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D. enough after follow-up")
            self.assertEqual(result.rounds[-1].status, "final")
            self.assertEqual([request.task for request in backend.requests if request.task == "replan"], [])
            self.assertEqual(
                [request.task for request in backend.requests if request.task == "answer_from_evidence"],
                ["answer_from_evidence"],
            )
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"source": "hard_skill_runtime"', trace)

    def test_planner_final_with_navigation_only_citation_is_blocked(self):
        backend = RecordingBackend(
            [
                '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "red car"}, "assign": "map"}]}',
                '{"status": "final", "answer": "A. red car", "citations": ["obs_0001"], "confidence": 0.8}',
            ]
        )
        registry = ToolRegistry()

        @tool(name="video_ls", description="Navigation only.")
        def video_ls(query: str = ""):
            return {"claim": f"map for {query}", "confidence": 1.0}

        registry.register(video_ls)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="navigation_final_gate")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False),
            )

            result = agent.run(question="What is visible?", video_path="/videos/demo.mp4")

            self.assertNotEqual(result.status, "final")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("final_requires_non_navigation_visual_evidence", trace)

    def test_failure_reflection_memory_is_injected_after_parse_error(self):
        backend = RecordingBackend(
            [
                "not json",
                '{"status": "final", "answer": "A. enough", "citations": ["obs_0001"], "confidence": 0.8}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="reflection_after_parse_error")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=_caption_only_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False),
            )

            agent.run(question="What is visible?", video_path="/videos/demo.mp4")

            self.assertEqual(len(backend.requests), 2)
            self.assertIn("# Reflection Memory", backend.requests[1].prompt)
            self.assertIn("planner_json_parse_error", backend.requests[1].prompt)
            self.assertIn("return valid JSON", backend.requests[1].prompt)

    def test_context_budget_report_is_persisted_in_trace(self):
        backend = RecordingBackend(
            ['{"status": "final", "answer": "A. enough", "citations": ["obs_0001"], "confidence": 0.8}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="context_budget_trace")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=_caption_only_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False, context_budget_tokens=400),
            )

            agent.run(question="What is visible?", video_path="/videos/demo.mp4")

            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("context_budget_report", trace)
            self.assertIn('"turn_index": 0', trace)

    def test_exports_longvideoagent_ready_trajectory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="trajectory_export")
            workspace.write_trace_event(
                "tool_use",
                {"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}},
            )
            observation = workspace.write_observation(
                tool_name="vision_read",
                claim="The localized window shows a red car.",
                confidence=0.91,
                regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
                raw_output={
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [{"option": "B", "relation": "support", "strength": 0.91}],
                },
            )
            workspace.write_ledger_entry(observation)
            workspace.write_trace_event(
                "tool_result",
                {"step": 1, "tool": "vision_read", "observation_id": observation.observation_id},
            )

            payload = workspace.export_longvideoagent_trajectory(
                question="Which object is visible?\nA. blue car\nB. red car",
                video_path="/videos/demo.mp4",
                final={"answer": "B. red car", "citations": [observation.observation_id], "confidence": 0.91},
                verifier_result={"status": "supported"},
                reward_tags=["structured_support", "non_navigation_visual_citation"],
            )

            trajectory_path = workspace.root / "artifacts" / "trajectories" / "longvideoagent_trajectory.json"
            self.assertTrue(trajectory_path.exists())
            disk_payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "LongVideoAgentTrajectoryV1")
            self.assertEqual(disk_payload["state"]["question"], "Which object is visible?\nA. blue car\nB. red car")
            self.assertEqual(disk_payload["actions"][0]["tool"], "vision_read")
            self.assertEqual(disk_payload["actions"][0]["observation_id"], observation.observation_id)
            self.assertEqual(disk_payload["final"]["answer"], "B. red car")
            self.assertEqual(disk_payload["verifier_result"]["status"], "supported")
            self.assertIn("structured_support", disk_payload["reward_tags"])


def _caption_only_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption one segment.")
    def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 8):
        return {
            "claim": f"{segment_id} shows enough visual evidence.",
            "confidence": 0.8,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
        }

    registry.register(caption_segment)
    return registry


if __name__ == "__main__":
    unittest.main()
