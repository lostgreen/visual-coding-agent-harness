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
    compose_replanning_prompt_blocks,
    render_prompt_blocks,
)
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
                "active_skill",
                "tool_schema",
                "evidence_snapshot",
                "answer_feedback",
                "reflection_memory",
                "final_gate",
                "response_contract",
            ],
        )
        self.assertLess(prompt.index("# Base Identity"), prompt.index("# Route Playbook"))
        self.assertLess(prompt.index("# Route Playbook"), prompt.index("# Active Skill"))
        self.assertLess(prompt.index("# Active Skill"), prompt.index("# Tool Schema"))
        self.assertIn("Allowed ReAct actions: ground_question, vision_read, answer_agent, verify", prompt)
        self.assertIn("Skill: temporal_ordering@v1", prompt)
        self.assertIn("confirm every event timestamp", prompt)
        self.assertIn("Final answers require at least one non-navigation visual observation", prompt)

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
            _skill_target_facts(question=temporal, skill_name="temporal_ordering"),
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
