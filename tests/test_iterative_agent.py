import json
import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import (
    AgentBudget,
    IterativeVisualAgent,
    _blocked_final_reason,
    _blocked_planner_final_reason,
    _exhausted_one_shot_tools,
    _latest_asr_binding_candidates,
    _local_fact_question,
    _planner_final_answer_with_option,
    _program_signature,
    _projection_evidence_from_table,
    _route_repair_recovery_program,
    _sanitize_option_blind_feedback,
    _supported_binding_no_growth_feedback,
)
from visual_coding_agent_harness.agents.answer_agent import AnswerAgentResult
from visual_coding_agent_harness.agents.question_policy import extract_candidate_options
from visual_coding_agent_harness.agents.runtime.state import RunState
from visual_coding_agent_harness.agents.skills.specs import (
    EvidenceFollowupKind,
    ExplorationProfile,
    FinalGateProfile,
    OptionEvaluationKind,
    PrefinalRepairKind,
    RouteRepairPolicyKind,
    SchedulerKind,
    SkillBehaviors,
)
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.contracts import ClaimRelation, ClaimModality, OptionSpec, TargetRegistry, TargetSpec
from visual_coding_agent_harness.iterative_smoke import run_iterative_smoke
from visual_coding_agent_harness.registry import ToolRegistry, tool
from visual_coding_agent_harness.tools.exploration import build_video_exploration_registry
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.tools.segments import build_segment_vlm_registry
from visual_coding_agent_harness.video_index import VideoSegment, SceneIndex, fixed_window_scene_index
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class ScriptedPlannerBackend(VisionLanguageBackend):
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task != "replan":
            return BackendResponse(text="unexpected tool backend call")
        if not self.responses:
            return BackendResponse(text='{"status": "final", "answer": "No more scripted responses.", "citations": []}')
        return BackendResponse(text=self.responses.pop(0))


class StaticTaskBackend(VisionLanguageBackend):
    def __init__(self, responses):
        self.responses = dict(responses)
        self.requests = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.responses.get(request.task, "{}"))


def _trace_events(workspace: EvidenceWorkspace):
    trace_path = workspace.root / "trace.jsonl"
    if not trace_path.exists():
        return []
    return [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]


def test_supported_binding_no_growth_feedback_recommends_asr_binding_without_route_change():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_growth_feedback")
        workspace.write_observation(
            tool_name="target_coverage",
            claim="coverage",
            confidence=1.0,
            raw_output={
                "coverage": [
                    {
                        "target_id": "T1",
                        "target_ref": "T1",
                        "target": "Austria Hungary rises and falls",
                        "candidates": [
                            {
                                "segment_id": "seg_0007",
                                "score": 0.8,
                                "source": "asr_text",
                                "snippet": "Austria Hungary rose and fell.",
                            }
                        ],
                    }
                ]
            },
        )

        candidates = _latest_asr_binding_candidates(workspace=workspace, target_refs=(), limit=2)
        feedback = _supported_binding_no_growth_feedback(candidates=candidates, skill_locked=True)

    joined = "\n".join(feedback)
    assert "bind_asr_claim(segment_id='seg_0007', target_refs=['T1'])" in joined
    assert "change route" not in joined.lower()


def test_read_segment_detail_does_not_auto_inject_option_targets():
    backend = ScriptedPlannerBackend(
        [
            (
                '{"status": "continue", "program": ['
                '{"tool": "read_segment_detail", "args": {"segment_id": "seg_0001"}, "assign": "detail"}'
                "]}"
            ),
            '{"status": "final", "answer": "B", "citations": ["obs_0001"], "confidence": 0.7}',
        ]
    )
    video_map = VideoMap(
        video_path="/videos/demo.mp4",
        duration_sec=30.0,
        segments=[VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, asr_text="Relevant narration.")],
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_option_targets")
        agent = IterativeVisualAgent(
            backend=backend,
            registry=build_video_navigation_registry(video_map, workspace=workspace),
            workspace=workspace,
            scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0),
            budget=AgentBudget(max_rounds=1, reserve_final_round=False),
        )

        result = agent.run(
            question="Which option is supported?\nA. wrong full option\nB. right full option",
            video_path="/videos/demo.mp4",
        )

    assert "option_targets" not in result.rounds[0].program[0]["args"]


def test_mcq_budget_exhaustion_never_returns_empty_answer_after_answer_parse_failure():
    class ParseFailingAnswerBackend(VisionLanguageBackend):
        def __init__(self):
            self.requests = []

        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            if request.task == "replan":
                return BackendResponse(
                    text=(
                        '{"status": "continue", "program": ['
                        '{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "Inspect"}, "assign": "obs"}'
                        "]}"
                    )
                )
            if request.task == "answer_from_evidence":
                return BackendResponse(text="not json")
            return BackendResponse(text="unexpected")

    scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_parse_failure_fallback")
        agent = IterativeVisualAgent(
            backend=ParseFailingAnswerBackend(),
            registry=ToolRegistry(),
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=1, reserve_final_round=False),
        )
        calls = {"vision": 0}

        @tool(name="vision_read", description="Read a focused visual fact.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float = 0.0,
            end_sec: float = 0.0,
            ask_for: str = "",
            event_label: str = "",
            nframes: int = 0,
        ):
            calls["vision"] += 1
            return {
                "claim": "The second option has direct support.",
                "confidence": 0.9,
                "grounding_quality": "visually_confirmed",
                "answer_evidence_rows": [
                    {
                        "tool": "vision_read",
                        "segment_id": segment_id,
                        "claim": "The second option has direct support.",
                        "confidence": 0.9,
                        "grounding_quality": "visually_confirmed",
                        "supported_option": "B",
                    }
                ],
            }

        agent.registry.register(vision_read)

        result = agent.run(
            question="Which option is supported?\nA. first\nB. second\nC. third\nD. fourth",
            video_path="/videos/demo.mp4",
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert result.status in {"final", "low_confidence_final"}
    assert result.answer.startswith("B")
    assert result.answer
    assert calls["vision"] == 1
    assert "iterative_answer_agent" in trace


def test_mcq_terminal_fallback_prefers_latest_hypothesis_option_before_fixed_option():
    class AbstainingBackend(VisionLanguageBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            if request.task == "replan":
                return BackendResponse(text='{"status": "continue", "program": []}')
            if request.task == "answer_from_evidence":
                return BackendResponse(
                    text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["more"], "confidence": 0.0}'
                )
            return BackendResponse(text="unexpected")

    scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_hypothesis_fallback")
        (workspace.root / "hypothesis.md").write_text("# Hypothesis\n\nlatest hypothesis option: C\n", encoding="utf-8")
        agent = IterativeVisualAgent(
            backend=AbstainingBackend(),
            registry=ToolRegistry(),
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=1, reserve_final_round=False),
        )

        result = agent.run(
            question="Which option is supported?\nA. first\nB. second\nC. third",
            video_path="/videos/demo.mp4",
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert result.status == "low_confidence_final"
    assert result.answer == "C"
    assert '"fallback_source": "latest_hypothesis"' in trace


def test_mcq_terminal_fallback_prefers_complete_projection_over_low_confidence_answer():
    class LowConfidenceAnswerBackend(VisionLanguageBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            if request.task == "replan":
                return BackendResponse(
                    text=(
                        '{"status": "continue", "program": ['
                        '{"tool": "vision_read", "args": {"segment_id": "seg_0001", "target_ref": "T1"}, "assign": "obs1"},'
                        '{"tool": "vision_read", "args": {"segment_id": "seg_0004", "target_ref": "T2"}, "assign": "obs2"}'
                        "]}"
                    )
                )
            if request.task == "answer_from_evidence":
                return BackendResponse(
                    text=(
                        '{"answer": "need_more_evidence", "citations": [], '
                        '"candidate_option_relations": ['
                        '{"option": "A", "relation": "support", "strength": 0.8, '
                        '"observation_id": "obs_9999", "grounding_quality": "visually_confirmed"}], '
                        '"missing_evidence": ["more evidence"], "confidence": 0.0}'
                    )
                )
            return BackendResponse(text="unexpected")

    scene_index = SceneIndex(
        video_path="/videos/demo.mp4",
        duration_sec=80.0,
        segments=[
            VideoSegment("seg_0001", 0.0, 10.0),
            VideoSegment("seg_0004", 30.0, 40.0),
        ],
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_projection_fallback")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[
                TargetSpec("T1", "theme marker one"),
                TargetSpec("T2", "theme marker two"),
            ],
            options=[
                OptionSpec("A", target_sequence=("T1",), raw_option_text="theme marker one"),
                OptionSpec("B", target_sequence=("T2",), raw_option_text="theme marker two"),
                OptionSpec("C", target_sequence=(), raw_option_text="unrelated marker"),
                OptionSpec(
                    "D",
                    target_sequence=("T1", "T2"),
                    raw_option_text="theme marker one theme marker two",
                    option_kind="theme",
                ),
            ],
        )
        agent = IterativeVisualAgent(
            backend=LowConfidenceAnswerBackend(),
            registry=ToolRegistry(),
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=1, reserve_final_round=False),
        )

        @tool(name="vision_read", description="Read a focused visual fact.")
        def vision_read(
            video_path: str,
            segment_id: str,
            target_ref: str = "",
            start_sec: float = 0.0,
            end_sec: float = 0.0,
            ask_for: str = "",
            event_label: str = "",
            nframes: int = 0,
        ):
            return {
                "claim": f"Observed {target_ref}.",
                "confidence": 0.9,
                "target_ref": target_ref,
                "grounding_quality": "visually_confirmed",
            }

        agent.registry.register(vision_read)

        result = agent.run(
            question=(
                "What is the video mainly about?\n"
                "A. theme marker one\n"
                "B. theme marker two\n"
                "C. unrelated marker\n"
                "D. theme marker one theme marker two"
            ),
            video_path="/videos/demo.mp4",
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert result.status == "low_confidence_final"
    assert result.answer == "D"
    assert '"fallback_source": "answer_evidence_projection"' in trace


def test_projection_evidence_ignores_contradictory_relation_target_refs():
    evidence = _projection_evidence_from_table(
        {
            "rows": [
                {
                    "obs_id": "obs_0001",
                    "tool": "vision_read",
                    "claim": "The row contradicts a target candidate.",
                    "confidence": 0.9,
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [
                        {
                            "option": "B",
                            "relation": "contradict",
                            "target_ref": "T2",
                            "strength": 0.9,
                        }
                    ],
                }
            ]
        }
    )

    assert evidence == ()


def test_mcq_terminal_fallback_ignores_ambiguous_projection_before_hypothesis():
    question = (
        "What is the video mainly about?\n"
        "A. first broad theme\n"
        "B. second broad theme\n"
        "C. unresolved fallback"
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_ambiguous_projection_fallback")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[
                TargetSpec("T1", "first marker"),
                TargetSpec("T2", "second marker"),
                TargetSpec("T3", "third marker"),
                TargetSpec("T4", "fourth marker"),
            ],
            options=[
                OptionSpec("A", target_sequence=("T1", "T2"), raw_option_text="first broad theme", option_kind="theme"),
                OptionSpec("B", target_sequence=("T3", "T4"), raw_option_text="second broad theme", option_kind="theme"),
                OptionSpec("C", target_sequence=(), raw_option_text="unresolved fallback"),
            ],
        )
        (workspace.root / "hypothesis.md").write_text("# Hypothesis\n\nlatest hypothesis option: C\n", encoding="utf-8")
        agent = IterativeVisualAgent(
            backend=StaticTaskBackend({}),
            registry=ToolRegistry(),
            workspace=workspace,
            scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0),
            budget=AgentBudget(max_rounds=1, reserve_final_round=False),
        )
        agent._answer_evidence_table = lambda _question: {
            "options": extract_candidate_options(question),
            "groups": {},
            "rows": [
                {
                    "obs_id": obs_id,
                    "tool": "vision_read",
                    "segment_id": segment_id,
                    "target_ref": target_ref,
                    "claim": f"Observed {target_ref}.",
                    "grounding_quality": "visually_confirmed",
                    "confidence": 0.9,
                }
                for obs_id, target_ref, segment_id in [
                    ("obs_0001", "T1", "seg_0001"),
                    ("obs_0002", "T2", "seg_0002"),
                    ("obs_0003", "T3", "seg_0003"),
                    ("obs_0004", "T4", "seg_0003"),
                ]
            ],
        }

        result = agent._forced_mcq_fallback_result(
            question=question,
            video_path="/videos/demo.mp4",
            rounds=[],
            round_number=1,
            citations=[],
            source="unit_test",
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert result is not None
    assert result.answer == "C"
    assert '"fallback_source": "latest_hypothesis"' in trace


def test_repeated_zero_yield_asr_binding_call_is_skipped():
    class RepeatedBindingBackend(VisionLanguageBackend):
        def __init__(self):
            self.requests = []

        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            if request.task == "replan":
                return BackendResponse(
                    text=(
                        '{"status": "continue", "program": ['
                        '{"tool": "bind_asr_claim", "args": {"segment_id": "seg_0001", "target_refs": ["T1"]}}'
                        "]}"
                    )
                )
            return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["more"]}')

    registry = ToolRegistry()
    calls = {"bind": 0}

    @tool(name="bind_asr_claim", description="Bind ASR claims.")
    def bind_asr_claim(segment_id: str, target_refs: list | None = None):
        calls["bind"] += 1
        return {
            "claim": "No cue supported the requested target.",
            "confidence": 0.0,
            "segment_id": segment_id,
            "target_refs": list(target_refs or []),
            "answer_evidence_rows": [],
            "evidence_bindings": [],
            "limitations": "missing binding for T1",
        }

    registry.register(bind_asr_claim)
    scene_index = fixed_window_scene_index(video_path="/videos/asr.mp4", duration_sec=30.0, window_sec=30.0)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="zero_yield_dedupe")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[TargetSpec("T1", "narrated target", modality_hint=ClaimModality.NARRATED_FACT)],
            options=[OptionSpec("A", target_sequence=("T1",))],
        )
        agent = IterativeVisualAgent(
            backend=RepeatedBindingBackend(),
            registry=registry,
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=3, reserve_final_round=False, max_repeated_programs=0),
        )

        result = agent.run(
            question="Which option is supported?\nA. narrated target\nB. other",
            video_path="/videos/asr.mp4",
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert result.status == "low_confidence_final"
    assert result.answer
    assert calls["bind"] == 1
    assert "zero_yield_tool_call_skipped" in trace


def test_repeated_clean_zero_yield_asr_binding_call_is_skipped_without_limitations():
    class RepeatedBindingBackend(VisionLanguageBackend):
        def generate(self, request: BackendRequest) -> BackendResponse:
            if request.task == "replan":
                return BackendResponse(
                    text=(
                        '{"status": "continue", "program": ['
                        '{"tool": "bind_asr_claim", "args": {"segment_id": "seg_0001", "target_refs": ["T1"]}}'
                        "]}"
                    )
                )
            return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["more"]}')

    registry = ToolRegistry()
    calls = {"bind": 0}

    @tool(name="bind_asr_claim", description="Bind ASR claims.")
    def bind_asr_claim(segment_id: str, target_refs: list | None = None):
        calls["bind"] += 1
        return {
            "claim": "No cue supported the requested target.",
            "confidence": 0.0,
            "segment_id": segment_id,
            "target_refs": list(target_refs or []),
            "answer_evidence_rows": [],
            "evidence_bindings": [],
        }

    registry.register(bind_asr_claim)
    scene_index = fixed_window_scene_index(video_path="/videos/asr.mp4", duration_sec=30.0, window_sec=30.0)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="zero_yield_dedupe_clean")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[TargetSpec("T1", "narrated target", modality_hint=ClaimModality.NARRATED_FACT)],
            options=[OptionSpec("A", target_sequence=("T1",))],
        )
        agent = IterativeVisualAgent(
            backend=RepeatedBindingBackend(),
            registry=registry,
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=3, reserve_final_round=False, max_repeated_programs=0),
        )

        result = agent.run(
            question="Which option is supported?\nA. narrated target\nB. other",
            video_path="/videos/asr.mp4",
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert result.status == "low_confidence_final"
    assert result.answer
    assert calls["bind"] == 1
    assert "zero_yield_tool_call_recorded" in trace
    assert "zero_yield_tool_call_skipped" in trace


def test_stable_answer_agent_suggestion_can_finalize_before_reserved_round():
    class StableSuggestionBackend(VisionLanguageBackend):
        def __init__(self):
            self.requests = []

        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            if request.task == "replan":
                return BackendResponse(
                    text=(
                        '{"status": "continue", "program": ['
                        '{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "Inspect"}, "assign": "vis"}'
                        "]}"
                    )
                )
            if request.task == "answer_from_evidence":
                return BackendResponse(
                    text=json.dumps(
                        {
                            "answer": "B",
                            "rationale": "Stable supported answer.",
                            "citations": ["obs_0001"],
                            "candidate_option_relations": [
                                {
                                    "option": "B",
                                    "relation": "support",
                                    "strength": 0.98,
                                    "observation_id": "obs_0001",
                                    "rationale": "caption supports B",
                                    "answer_grade": True,
                                }
                            ],
                            "missing_evidence": [],
                            "confidence": 0.98,
                        }
                    )
                )
            return BackendResponse(text="unexpected")

    registry = ToolRegistry()

    @tool(name="vision_read", description="Read a focused visual fact.")
    def vision_read(
        segment_id: str,
        ask_for: str = "",
        video_path: str = "",
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        nframes: int = 0,
        event_label: str = "",
    ):
        return {
            "claim": "Segment seg_0001 contains the answer.",
            "confidence": 0.95,
            "grounding_quality": "visually_confirmed",
            "segment_id": segment_id,
            "answer_evidence_rows": [
                {
                    "tool": "vision_read",
                    "segment_id": segment_id,
                    "claim": "Segment seg_0001 contains the answer.",
                    "confidence": 0.95,
                    "grounding_quality": "visually_confirmed",
                    "supported_option": "B",
                }
            ],
        }

    registry.register(vision_read)
    scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="stable_answer_suggestion")
        agent = IterativeVisualAgent(
            backend=StableSuggestionBackend(),
            registry=registry,
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=6, reserve_final_round=True, max_repeated_programs=0, answer_probe_rounds_before_final=4),
        )

        result = agent.run(
            question="Which option is supported?\nA. wrong\nB. Segment seg_0001 contains the answer",
            video_path="/videos/demo.mp4",
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    assert result.status == "final"
    assert result.answer.startswith("B")
    assert len(result.rounds) < 5
    assert "stable_answer_suggestion_finalized" in trace


def test_low_confidence_near_exhaustion_auto_promotes_asr_binding_before_final():
    video_map = VideoMap(
        video_path="/videos/asr.mp4",
        duration_sec=30.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=30.0,
                asr_sentences=[
                    {
                        "cue_id": "cue_0001",
                        "start_sec": 3.0,
                        "end_sec": 7.0,
                        "text": "The video explains how Austria Hungary rose and fell.",
                    }
                ],
            )
        ],
    )
    backend = StaticTaskBackend(
        {
            "asr_claim_binding": '{"T1": {"verdict": "supports", "cue_ids": ["cue_0001"], "quote": "Austria Hungary rose and fell"}}'
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="auto_promote")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[
                TargetSpec(
                    "T1",
                    "Austria Hungary rises and falls",
                    modality_hint=ClaimModality.NARRATED_FACT,
                )
            ],
            options=[OptionSpec("D", target_sequence=("T1",))],
        )
        workspace.write_observation(
            tool_name="target_coverage",
            claim="coverage",
            confidence=1.0,
            raw_output={
                "coverage": [
                    {
                        "target_id": "T1",
                        "target_ref": "T1",
                        "target": "Austria Hungary rises and falls",
                        "candidates": [
                            {
                                "segment_id": "seg_0001",
                                "score": 0.8,
                                "source": "asr_sentence",
                                "snippet": "Austria Hungary rose and fell.",
                            }
                        ],
                    }
                ]
            },
        )
        registry = build_video_exploration_registry(video_map=video_map, backend=backend, workspace=workspace)
        agent = IterativeVisualAgent(
            backend=backend,
            registry=registry,
            workspace=workspace,
            scene_index=fixed_window_scene_index(video_path="/videos/asr.mp4", duration_sec=30.0),
        )
        answer = AnswerAgentResult(
            status="need_more_evidence",
            candidate_option_relations=[
                {
                    "option": "D",
                    "relation": "support",
                    "strength": 0.8,
                    "grounding_quality": "indexed_transcript",
                    "observation_id": "obs_0001",
                }
            ],
            raw_text="partial",
        )

        result = agent._try_low_confidence_final(
            answer_result=answer,
            question="Question?\nD. How Austria Hungary rises and falls.",
            video_path="/videos/asr.mp4",
            rounds=[],
            round_number=8,
            source="evidence_table_no_growth",
            remaining_rounds=2,
            supported_binding_no_growth_rounds=5,
        )
        rows = workspace.read_evidence_table_v3(question="Question?", options=[]).get("rows", [])
        trace = _trace_events(workspace)

    assert result is None
    assert backend.requests[0].task == "asr_claim_binding"
    assert any(row.get("evidence_binding", {}).get("status") == "supported" for row in rows)
    assert any(
        event["type"] == "auto_evidence_promotion_attempted" and event["payload"].get("succeeded") is True
        for event in trace
    )


def test_low_confidence_near_exhaustion_allows_final_after_auto_promotion_failure():
    answer = AnswerAgentResult(
        status="need_more_evidence",
        candidate_option_relations=[
                {
                    "option": "D",
                    "relation": "support",
                    "strength": 0.8,
                    "grounding_quality": "indexed_transcript",
                    "observation_id": "obs_0001",
            }
        ],
        raw_text="partial",
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="auto_promote_fail")
        backend = StaticTaskBackend({})
        registry = build_video_exploration_registry(video_map=VideoMap("/videos/asr.mp4", 30.0, []), backend=backend, workspace=workspace)
        agent = IterativeVisualAgent(
            backend=backend,
            registry=registry,
            workspace=workspace,
            scene_index=fixed_window_scene_index(video_path="/videos/asr.mp4", duration_sec=30.0),
        )

        result = agent._try_low_confidence_final(
            answer_result=answer,
            question="Question?\nD. How Austria Hungary rises and falls.",
            video_path="/videos/asr.mp4",
            rounds=[],
            round_number=8,
            source="evidence_table_no_growth",
            remaining_rounds=2,
            supported_binding_no_growth_rounds=5,
        )

    assert result is not None
    assert result.status == "low_confidence_final"
    assert result.answer == "D"


def test_planner_final_keeps_temporal_free_text_for_verifier():
    question = (
        "VideoMME multiple-choice question. Answer with exactly one option letter (A/B/C/D) first.\n"
        "Question: As depicted in the video, in what order does the author present Bernini's four masterpieces?\n"
        "Options:\n"
        'A. "The rape of Persephone", "Apollo and Daphne", "David" and "Aeneas, Anchises, and Ascanius fleeing Troy".\n'
        'B. "David", "Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne" and "The rape of Persephone".\n'
        'C. "Apollo and Daphne", "Aeneas, Anchises, and Ascanius fleeing Troy", "David" and "The rape of Persephone".\n'
        'D. "Aeneas, Anchises, and Ascanius fleeing Troy", "David", "The rape of Persephone" and "Apollo and Daphne".'
    )
    answer = (
        "The author lists the works as Aeneas, Anchises, and Ascanius fleeing Troy, "
        "then David, then The rape of Persephone, and finally Apollo and Daphne."
    )

    assert _planner_final_answer_with_option(question=question, answer=answer) == answer


def test_planner_final_keeps_partial_temporal_free_text_unchanged():
    question = (
        "Question: order?\n"
        "Options:\n"
        "A. red, blue and green.\n"
        "B. blue, red and green."
    )

    answer = "The clip mentions red and blue."

    assert _planner_final_answer_with_option(question=question, answer=answer) == answer


def test_local_fact_question_uses_behavior_profiles_with_custom_skill_names():
    timeline_skill = type(
        "Skill",
        (),
        {
            "name": "custom_timeline",
            "behaviors": SkillBehaviors(exploration_profile=ExplorationProfile.TIMELINE_FAMILY),
        },
    )()
    timeline_prompt = _local_fact_question(
        question="Which artwork came first?",
        planner_skill=timeline_skill,
        target_entities=("Apollo and Daphne",),
    )

    assert "presentation order" in timeline_prompt
    assert "Apollo and Daphne" in timeline_prompt

    mutex_skill = type(
        "Skill",
        (),
        {
            "name": "custom_mutex",
            "behaviors": SkillBehaviors(scheduler=SchedulerKind.FOLLOWUP_QUEUE),
        },
    )()
    mutex_prompt = _local_fact_question(
        question="A. farmer\nB. noble",
        planner_skill=mutex_skill,
    )

    assert "class/status" in mutex_prompt
    assert "Do not choose an option." in mutex_prompt

    main_idea_skill = type(
        "Skill",
        (),
        {
            "name": "custom_main_idea",
            "behaviors": SkillBehaviors(exploration_profile=ExplorationProfile.MAIN_IDEA),
        },
    )()
    main_idea_prompt = _local_fact_question(
        question="What is this video mainly about?",
        planner_skill=main_idea_skill,
    )

    assert "localized main-idea evidence" in main_idea_prompt


def test_fallback_visual_tool_preferences_use_exploration_profiles_with_custom_skill_names():
    registry = build_segment_test_registry()

    @tool(name="vision_read", description="Read focused visual evidence.")
    def vision_read(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        ask_for: str,
        event_label: str = "",
        nframes: int = 8,
    ):
        return {
            "claim": f"{segment_id} answers {ask_for}",
            "confidence": 0.8,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "nframes": nframes}],
        }

    registry.register(vision_read)

    allowed_actions = frozenset({"caption_segment", "vision_read", "qa_segment", "inspect_segment"})

    def custom_skill(name: str, profile: ExplorationProfile):
        return type(
            "Skill",
            (),
            {
                "name": name,
                "allowed_actions": allowed_actions,
                "behaviors": SkillBehaviors(exploration_profile=profile),
            },
        )()

    with tempfile.TemporaryDirectory() as tmp:
        agent = IterativeVisualAgent(
            backend=StaticTaskBackend({}),
            registry=registry,
            workspace=EvidenceWorkspace.create(Path(tmp), run_id="profile_fallback"),
            scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0),
        )

        assert agent._fallback_visual_tool_name_for_skill(
            custom_skill("custom_timeline", ExplorationProfile.TIMELINE_FAMILY)
        ) == "caption_segment"
        assert agent._fallback_visual_tool_name_for_skill(
            custom_skill("custom_grounded", ExplorationProfile.GROUNDED_FACTUAL)
        ) == "vision_read"
        assert agent._fallback_visual_tool_name_for_skill(
            custom_skill("custom_main_idea", ExplorationProfile.MAIN_IDEA)
        ) == "vision_read"


def test_hard_skill_route_admits_custom_option_evaluation_behavior(monkeypatch):
    from visual_coding_agent_harness.agents import iterative_agent as iterative_agent_module

    skill = type(
        "Skill",
        (),
        {
            "name": "custom_option_eval",
            "version": 1,
            "behaviors": SkillBehaviors(option_evaluation=OptionEvaluationKind.MUTEX_OR_GROUNDED),
        },
    )()
    monkeypatch.setattr(iterative_agent_module, "_recommended_effective_skill", lambda *args, **kwargs: skill)

    with tempfile.TemporaryDirectory() as tmp:
        agent = IterativeVisualAgent(
            backend=StaticTaskBackend({}),
            registry=ToolRegistry(),
            workspace=EvidenceWorkspace.create(Path(tmp), run_id="custom_option_eval"),
            scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0),
        )
        checked_tools: list[str] = []

        def has_tool(tool_name: str) -> bool:
            checked_tools.append(tool_name)
            return False

        monkeypatch.setattr(agent, "_has_tool", has_tool)

        result = agent._try_hard_skill_route(
            question="Which visible fact is true?\nA. farmer\nB. noble",
            exploration_question="Which visible fact is true?",
            video_path="/videos/demo.mp4",
        )

    assert result is None
    assert checked_tools == ["ground_question"]


def test_generic_forced_visual_skip_reason_uses_evidence_followup_behavior():
    skill = type(
        "Skill",
        (),
        {
            "name": "custom_transcript_followup",
            "behaviors": SkillBehaviors(evidence_followup=EvidenceFollowupKind.SEGMENT_DETAIL_AND_ASR),
        },
    )()

    with tempfile.TemporaryDirectory() as tmp:
        agent = IterativeVisualAgent(
            backend=StaticTaskBackend({}),
            registry=ToolRegistry(),
            workspace=EvidenceWorkspace.create(Path(tmp), run_id="custom_forced_skip"),
            scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0),
        )

        assert (
            agent._generic_forced_visual_skip_reason(
                question="How was his life journey according to the video?",
                planner_skill=skill,
                run_state=RunState(question="", video_path=""),
            )
            == "narration_transcript_route"
        )


def test_skill_route_repair_uses_behavior_profiles_with_custom_skill_names(monkeypatch):
    registry = ToolRegistry()

    @tool(name="global_gist", description="Read sparse whole-video gist.")
    def global_gist(video_path: str, question: str, duration_sec: float, seed: int = 0):
        return {"claim": "gist", "confidence": 0.8}

    @tool(name="vision_read", description="Read focused visual evidence.")
    def vision_read(segment_id: str = "", ask_for: str = "", **kwargs):
        return {"claim": f"vision {segment_id} {ask_for}", "confidence": 0.8}

    @tool(name="locate_targets_in_segment", description="Locate target entities.")
    def locate_targets_in_segment(segment_id: str = "", targets: list | None = None, **kwargs):
        return {"claim": f"located {segment_id}", "confidence": 0.6}

    @tool(name="read_segment_detail", description="Read transcript detail.")
    def read_segment_detail(segment_id: str = "", **kwargs):
        return {"claim": f"detail {segment_id}", "confidence": 0.7}

    @tool(name="verify_segment_anchors", description="Verify located anchors.")
    def verify_segment_anchors(segment_id: str, anchors: list, targets: list | None = None, **kwargs):
        return {"claim": f"verified {segment_id}: {targets}", "confidence": 0.8}

    registry.register(global_gist)
    registry.register(vision_read)
    registry.register(locate_targets_in_segment)
    registry.register(read_segment_detail)
    registry.register(verify_segment_anchors)

    def custom_skill(name: str, behaviors: SkillBehaviors):
        return type("Skill", (), {"name": name, "behaviors": behaviors})()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="custom_route_repair")
        workspace.write_observation(
            tool_name="locate_targets_in_segment",
            claim="ordered-list candidate found",
            confidence=1.0,
            raw_output={
                "segment_id": "seg_0001",
                "recommended_next_actions": [
                    {
                        "candidate_id": "ordered_list_custom_1",
                        "route_kind": "focused_ordered_list_vision",
                        "tool": "vision_read",
                        "args": {
                            "segment_id": "seg_0001",
                            "ask_for": "Describe the visible artworks in timestamp order.",
                        },
                    }
                ],
            },
        )
        agent = IterativeVisualAgent(
            backend=StaticTaskBackend({}),
            registry=registry,
            workspace=workspace,
            scene_index=fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0),
        )

        gist_repair = agent._repair_skill_route_tool(
            tool_name="vision_read",
            args={"segment_id": "seg_0001"},
            active_skill=custom_skill(
                "custom_gist_repair",
                SkillBehaviors(route_repair=RouteRepairPolicyKind.GIST_FAMILY),
            ),
            question="What is the main idea?",
            video_path="/videos/demo.mp4",
        )
        assert gist_repair is not None
        assert gist_repair[0] == "global_gist"
        assert gist_repair[2] == "repair_main_idea_vision_read_to_global_gist"

        workspace.write_observation(
            tool_name="global_gist",
            claim="seeded whole-video gist",
            confidence=0.8,
            raw_output={},
        )
        repeated_gist_repair = agent._repair_skill_route_tool(
            tool_name="global_gist",
            args={},
            active_skill=custom_skill(
                "custom_gist_repair",
                SkillBehaviors(route_repair=RouteRepairPolicyKind.GIST_FAMILY),
            ),
            question="What is the main idea?",
            video_path="/videos/demo.mp4",
        )
        assert repeated_gist_repair is not None
        assert repeated_gist_repair[0] == "vision_read"
        assert repeated_gist_repair[2] == "repair_repeated_main_idea_global_gist_to_vision_read"

        focused_vision_repair = agent._repair_skill_route_tool(
            tool_name="locate_targets_in_segment",
            args={"segment_id": "seg_0001"},
            active_skill=custom_skill(
                "custom_visual_timeline",
                SkillBehaviors(
                    exploration_profile=ExplorationProfile.TIMELINE_FAMILY,
                    final_gate=FinalGateProfile.TIMELINE_FAMILY_HINTS,
                ),
            ),
            question="Which artwork appears first?",
            video_path="/videos/demo.mp4",
        )
        assert focused_vision_repair is not None
        assert focused_vision_repair[0] == "vision_read"
        assert focused_vision_repair[2] == "repair_ordered_list_locator_to_focused_ordered_list_vision"

        workspace.write_observation(
            tool_name="locate_targets_in_segment",
            claim="locator produced anchors for verification",
            confidence=1.0,
            raw_output={
                "segment_id": "seg_0001",
                "verify_call_args": {
                    "segment_id": "seg_0001",
                    "anchors": [{"anchor_id": "a1", "segment_id": "seg_0001"}],
                    "targets": ["Apollo and Daphne"],
                },
            },
        )
        verify_repair = agent._repair_skill_route_tool(
            tool_name="locate_targets_in_segment",
            args={"segment_id": "seg_0001"},
            active_skill=custom_skill(
                "custom_plain_timeline",
                SkillBehaviors(exploration_profile=ExplorationProfile.TIMELINE_FAMILY),
            ),
            question="Which artwork appears first?",
            video_path="/videos/demo.mp4",
        )
        assert verify_repair is not None
        assert verify_repair[0] == "verify_segment_anchors"
        assert verify_repair[2] == "repair_repeated_locator_to_verify_segment_anchors"

        monkeypatch.setattr(
            agent,
            "_narration_transcript_promotion_args",
            lambda *, segment_id, original_args: {
                "segment_id": segment_id,
                "target_refs": list(original_args.get("target_refs") or []),
                "promote_answer_evidence": True,
            },
        )
        narration_repair = agent._repair_skill_route_tool(
            tool_name="locate_targets_in_segment",
            args={"segment_id": "seg_0001", "target_refs": ["T1"]},
            active_skill=custom_skill(
                "custom_narration_timeline",
                SkillBehaviors(
                    exploration_profile=ExplorationProfile.TIMELINE_FAMILY,
                    evidence_followup=EvidenceFollowupKind.SEGMENT_DETAIL_AND_ASR,
                ),
            ),
            question="How was his life journey according to the video?",
            video_path="/videos/demo.mp4",
        )
        assert narration_repair is not None
        assert narration_repair[0] == "read_segment_detail"
        assert narration_repair[2] == "repair_narration_locator_to_transcript_promotion"

        mutex_repair = agent._repair_skill_route_tool(
            tool_name="inspect_segment",
            args={"segment_id": "seg_0001", "question": "A or B?", "candidate_options": ["A", "B"]},
            active_skill=custom_skill(
                "custom_mutex",
                SkillBehaviors(
                    exploration_profile=ExplorationProfile.GROUNDED_FACTUAL,
                    option_evaluation=OptionEvaluationKind.MUTEX_OR_GROUNDED,
                    scheduler=SchedulerKind.FOLLOWUP_QUEUE,
                ),
            ),
            question="A or B?",
            video_path="/videos/demo.mp4",
        )
        assert mutex_repair is not None
        assert mutex_repair[0] == "vision_read"
        assert mutex_repair[1]["ask_for"] == "A or B?"
        assert "candidate_options" not in mutex_repair[1]
        assert "question" not in mutex_repair[1]
        assert mutex_repair[2] == "repair_mutex_inspect_segment_to_vision_read"


def test_route_repair_recovery_program_uses_behavior_route_hints_with_custom_skill_names():
    def custom_skill(name: str, behaviors: SkillBehaviors):
        return type("Skill", (), {"name": name, "behaviors": behaviors})()

    timeline_program = _route_repair_recovery_program(
        reason="repeated_route_repair",
        original_args={"segment_id": "seg_0001", "targets": ["David"]},
        repaired_tool_name="verify_segment_anchors",
        repaired_args={"segment_id": "seg_0001"},
        active_skill=custom_skill(
            "custom_timeline_scheduler",
            SkillBehaviors(scheduler=SchedulerKind.SUBEVENT_TIMELINE),
        ),
    )
    assert timeline_program[0]["args"]["question_route"] == "timeline_ordering"

    narration_program = _route_repair_recovery_program(
        reason="repeated_route_repair",
        original_args={"segment_id": "seg_0001", "target_refs": ["T1"]},
        repaired_tool_name="read_segment_detail",
        repaired_args={"segment_id": "seg_0001"},
        active_skill=custom_skill(
            "custom_narration_followup",
            SkillBehaviors(evidence_followup=EvidenceFollowupKind.SEGMENT_DETAIL_AND_ASR),
        ),
    )
    assert narration_program[0]["args"]["question_route"] == "narration_timeline_qa"


def test_option_b_requires_complete_relation_chain():
    question = (
        "How was his life journey according to the video?\n"
        "A. Born with humble background and lived in seclusion in a farmhouse.\n"
        "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
        "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
        "D. Borned in the upper class and lived in seclusion in a farmhouse."
    )
    skill = type(
        "Skill",
        (),
        {"behaviors": SkillBehaviors(prefinal_repair=PrefinalRepairKind.NARRATION_TIMELINE)},
    )()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="missing_relation_gate")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[
                TargetSpec("T1", "humble background", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T2", "entered upper class", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T3", "seclusion/farmhouse", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T4", "born in upper class", modality_hint=ClaimModality.NARRATED_FACT),
            ],
            options=[
                OptionSpec("B", target_sequence=("T1", "T2", "T3"), required_relations=("R1", "R2")),
                OptionSpec("C", target_sequence=("T1", "T3", "T2"), required_relations=("R3", "R4")),
                OptionSpec("D", target_sequence=("T4", "T3"), required_relations=("R5",)),
            ],
            relations=[
                ClaimRelation("R1", "before", "T1", "T2"),
                ClaimRelation("R2", "before", "T2", "T3"),
                ClaimRelation("R3", "before", "T1", "T3"),
                ClaimRelation("R4", "before", "T3", "T2"),
                ClaimRelation("R5", "before", "T4", "T3"),
            ],
        )
        for target_id in ("T1", "T2", "T3"):
            workspace.write_evidence_row(
                {
                    "evidence_id": f"ev_bind_seg_0001_{target_id}",
                    "obs_id": f"obs_{target_id}",
                    "tool": "transcript_evidence_binder",
                    "segment_id": "seg_0001",
                    "claim": "Transcript supports the target but only T1 before T2.",
                    "confidence": 0.9,
                    "grounding_quality": "indexed_transcript",
                    "supported_option": "B",
                    "evidence_binding": {
                        "evidence_id": f"ev_bind_seg_0001_{target_id}",
                        "status": "supported",
                        "claim_modality": "narrated_fact",
                        "target_id": target_id,
                        "segment_id": "seg_0001",
                        "relation_bindings": [
                            {"relation_id": "R1", "status": "supported", "binding_id": "rel_bind_R1"}
                        ],
                    },
                }
            )

        reason = _blocked_planner_final_reason(
            question=question,
            has_inspect_with_candidate_options=False,
            workspace=workspace,
            answer="B",
            citations=["obs_T1", "obs_T2", "obs_T3"],
            evidence_ids=["ev_bind_seg_0001_T1", "ev_bind_seg_0001_T2", "ev_bind_seg_0001_T3"],
            planner_skill=skill,
        )

    assert reason == "final_gate:missing_relation_binding"


def test_612_complete_chain_maps_to_b_gate():
    question = (
        "How was his life journey according to the video?\n"
        "A. Born with humble background and lived in seclusion in a farmhouse.\n"
        "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
        "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
        "D. Borned in the upper class and lived in seclusion in a farmhouse."
    )
    skill = type(
        "Skill",
        (),
        {"behaviors": SkillBehaviors(prefinal_repair=PrefinalRepairKind.NARRATION_TIMELINE)},
    )()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="complete_relation_gate")
        workspace.target_registry = TargetRegistry.from_specs(
            targets=[
                TargetSpec("T1", "humble background", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T2", "entered upper class", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T3", "seclusion/farmhouse", modality_hint=ClaimModality.NARRATED_FACT),
                TargetSpec("T4", "born in upper class", modality_hint=ClaimModality.NARRATED_FACT),
            ],
            options=[OptionSpec("B", target_sequence=("T1", "T2", "T3"), required_relations=("R1", "R2"))],
            relations=[
                ClaimRelation("R1", "before", "T1", "T2"),
                ClaimRelation("R2", "before", "T2", "T3"),
            ],
        )
        for target_id in ("T1", "T2", "T3"):
            workspace.write_evidence_row(
                {
                    "evidence_id": f"ev_bind_seg_0001_{target_id}",
                    "obs_id": f"obs_{target_id}",
                    "tool": "transcript_evidence_binder",
                    "segment_id": "seg_0001",
                    "claim": "Transcript supports T1 before T2 and T2 before T3.",
                    "confidence": 0.9,
                    "grounding_quality": "indexed_transcript",
                    "supported_option": "B",
                    "evidence_binding": {
                        "evidence_id": f"ev_bind_seg_0001_{target_id}",
                        "status": "supported",
                        "claim_modality": "narrated_fact",
                        "target_id": target_id,
                        "segment_id": "seg_0001",
                        "relation_bindings": [
                            {"relation_id": "R1", "status": "supported", "binding_id": "rel_bind_R1"},
                            {"relation_id": "R2", "status": "supported", "binding_id": "rel_bind_R2"},
                        ],
                    },
                }
            )

        reason = _blocked_planner_final_reason(
            question=question,
            has_inspect_with_candidate_options=False,
            workspace=workspace,
            answer="B",
            citations=["obs_T1", "obs_T2", "obs_T3"],
            evidence_ids=["ev_bind_seg_0001_T1", "ev_bind_seg_0001_T2", "ev_bind_seg_0001_T3"],
            planner_skill=skill,
        )

    assert reason == ""


def test_answer_verifier_blocks_conflicting_planner_sequence_final():
    backend = ScriptedPlannerBackend(
        [
            '{"status": "final", "skill": "visual_timeline_qa@v1", "answer": "C", "citations": ["obs_0001"], "confidence": 0.92}',
        ]
    )
    question = BERNINI_ORDER_QUESTION
    scene_index = fixed_window_scene_index(video_path="/videos/bernini.mp4", duration_sec=60.0, window_sec=60.0)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="deterministic_sequence_conflict")
        workspace.write_evidence_row(
            {
                "evidence_id": "seq_obs_0001",
                "obs_id": "obs_0001",
                "tool": "ordered_transcript_sequence",
                "segment_id": "seg_0001",
                "claim": "Transcript sequence maps exactly to D.",
                "confidence": 0.94,
                "grounding_quality": "indexed_transcript",
                "supported_option": "D",
                "evidence_binding": {
                    "evidence_id": "seq_obs_0001",
                    "status": "supported",
                    "claim_modality": "narrated_fact",
                    "target_id": "ordered_sequence",
                    "segment_id": "seg_0001",
                    "ordered_target_refs": ["T1", "T2", "T3", "T4"],
                },
            }
        )
        agent = IterativeVisualAgent(
            backend=backend,
            registry=ToolRegistry(),
            workspace=workspace,
            scene_index=scene_index,
            budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
        )

        result = agent.run(question=question, video_path="/videos/bernini.mp4")

        assert result.status == "low_confidence_final"
        assert result.answer == "D"
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
        assert "deterministic_evidence_mapping" not in trace
        assert "answer_conflict_detected" not in trace
        assert "planner_final_verifier_disagrees" in trace
        assert '"planner_answer": "C"' in trace
        assert "mcq_forced_fallback" in trace


BERNINI_ORDER_QUESTION = (
    "VideoMME multiple-choice question. Answer with exactly one option letter (A/B/C/D) first.\n"
    "Question: As depicted in the video, in what order does the author present Bernini's four masterpieces created for Borghese in a single scene?\n"
    "Options:\n"
    'A. "The rape of  Persephone", "Apollo and Daphne", "David" and "Aeneas, Anchises, and Ascanius fleeing Troy".\n'
    'B. "David", "Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne" and "The rape of  Persephone".\n'
    'C. "Apollo and Daphne", "Aeneas, Anchises, and Ascanius fleeing Troy", "David" and "The rape of  Persephone".\n'
    'D. "Aeneas, Anchises, and Ascanius fleeing Troy", "David", "The rape of  Persephone" and "Apollo and Daphne".'
)


def test_ordered_option_sequence_registry_is_not_seeded_before_grounding_plan(tmp_path: Path):
    backend = ScriptedPlannerBackend(
        [
            '{"status": "continue", "skill": "visual_timeline_qa", "rationale": "inspect", "program": []}',
        ]
    )
    workspace = EvidenceWorkspace.create(tmp_path, "ordered_registry_prompt")
    video_map = VideoMap(
        video_path="/videos/bernini.mp4",
        duration_sec=600.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=300.0,
                end_sec=600.0,
                asr_text=(
                    '"Aeneas, Anchises, and Ascanius fleeing Troy", "David", '
                    '"The rape of Persephone" and "Apollo and Daphne".'
                ),
            )
        ],
    )
    registry = build_video_navigation_registry(video_map, workspace=workspace)
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=600.0,
            segments=[VideoSegment(segment_id="seg_0002", start_sec=300.0, end_sec=600.0)],
        ),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(question=BERNINI_ORDER_QUESTION, video_path="/videos/bernini.mp4")

    prompt = backend.requests[0].prompt
    assert "No target_refs are registered for this run" in prompt
    assert "Registered target_refs:" not in prompt
    assert workspace.observation_count(tool_name="target_coverage") == 0


def test_life_journey_registry_is_not_semantically_canonicalized_before_grounding_plan(tmp_path: Path):
    question = (
        "How was his life journey according to the video?\n"
        "A. Born with humble background and lived in seclusion in a farmhouse.\n"
        "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
        "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
        "D. Born in the upper class and lived in seclusion in a farmhouse."
    )
    backend = ScriptedPlannerBackend(
        [
            '{"status": "continue", "skill": "narration_timeline_qa", "rationale": "inspect", "program": []}',
        ]
    )
    workspace = EvidenceWorkspace.create(tmp_path, "life_journey_registry")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=ToolRegistry(),
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/goya.mp4",
            duration_sec=600.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0)],
        ),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    agent.run(question=question, video_path="/videos/goya.mp4")

    assert getattr(workspace, "target_registry", None) is None
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "target_registry_compiled" not in trace


def test_banned_additional_targets_are_rejected_with_protocol_note(tmp_path: Path):
    backend = ScriptedPlannerBackend(
        [
            json.dumps(
                {
                    "status": "continue",
                    "rationale": "Try detail.",
                    "program": [
                        {
                            "tool": "read_segment_detail",
                            "args": {
                                "segment_id": "seg_0001",
                                "additional_targets": ["visible medal"],
                            },
                        }
                    ],
                }
            ),
        ]
    )
    workspace = EvidenceWorkspace.create(tmp_path, "banned_additional_targets")
    registry = build_video_navigation_registry(
        VideoMap(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        ),
        workspace=workspace,
    )
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        ),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    result = agent.run(question="What is visible?", video_path="/videos/demo.mp4")

    assert result.rounds[0].program == []
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "additional_targets_not_allowed" in trace
    assert "read_segment_detail" in trace


def test_target_refs_reject_coverage_query_id_even_when_legacy_target_matches(tmp_path: Path):
    backend = ScriptedPlannerBackend(
        [
            json.dumps(
                {
                    "status": "continue",
                    "rationale": "Try detail.",
                    "program": [
                        {
                            "tool": "read_segment_detail",
                            "args": {
                                "segment_id": "seg_0001",
                                "target_refs": ["Q1"],
                                "targets": ["humble background"],
                            },
                        }
                    ],
                }
            ),
        ]
    )
    workspace = EvidenceWorkspace.create(tmp_path, "query_id_rejected")
    workspace.write_observation(
        tool_name="target_coverage",
        claim="coverage",
        confidence=1.0,
        raw_output={"coverage": [{"query_id": "Q1", "target": "humble background"}]},
    )
    registry = build_video_navigation_registry(
        VideoMap(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        ),
        workspace=workspace,
    )
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        ),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    result = agent.run(question="What is mentioned?", video_path="/videos/demo.mp4")

    assert result.rounds[0].program == []
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert "coverage_query_id_not_callable" in trace
    assert "coverage_query_id_stripped" not in trace


def test_target_refs_take_precedence_over_legacy_targets_in_bound_tool_prompts(tmp_path: Path):
    backend = ScriptedPlannerBackend(
        [
            json.dumps(
                {
                    "status": "continue",
                    "rationale": "Focused read.",
                    "program": [
                        {
                            "tool": "read_segment_detail",
                            "args": {
                                "segment_id": "seg_0001",
                                "target_refs": ["T1"],
                                "targets": ["unrelated free text"],
                            },
                        }
                    ],
                }
            ),
        ]
    )
    workspace = EvidenceWorkspace.create(tmp_path, "target_refs_precedence")
    workspace.target_registry = TargetRegistry.from_specs(
        targets=[TargetSpec("T1", "humble background", aliases=("poor origins",))]
    )
    registry = build_video_navigation_registry(
        VideoMap(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    asr_text="The narration mentions humble background.",
                )
            ],
        ),
        workspace=workspace,
    )
    agent = IterativeVisualAgent(
        backend=backend,
        registry=registry,
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        ),
        budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
    )

    result = agent.run(question="What is mentioned?", video_path="/videos/demo.mp4")

    args = result.rounds[0].program[0]["args"]
    assert args["target_refs"] == ["T1"]
    assert "targets" not in args
    detail = workspace.read_observations(tool_name="read_segment_detail")[0].raw_output
    assert detail["target_hits"][0]["target"] == "humble background"
    assert "unrelated free text" not in detail["unmatched_targets"]


def test_planner_owned_grounding_controls_runtime_route_skill_and_target_hints(tmp_path: Path):
    class GroundingThenPlanBackend(VisionLanguageBackend):
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request: BackendRequest) -> BackendResponse:
            self.requests.append(request)
            if request.task == "ground_question":
                return BackendResponse(
                    text=json.dumps(
                        {
                            "route": "temporal_order",
                            "recommended_skill": "narration_timeline_qa",
                            "central_subjects": ["Planner Subject"],
                            "subjects": [
                                {
                                    "subject_key": "subject_main",
                                    "canonical_name": "Planner Subject",
                                    "aliases": [],
                                }
                            ],
                            "targets": [
                                {
                                    "target_key": "event_alpha",
                                    "canonical_claim": "Planner canonical alpha event",
                                    "subject_key": "subject_main",
                                    "claim_kind": "narrated_fact",
                                    "claim_modality": "asr",
                                    "aliases": ["planner alpha", "Planner Subject"],
                                    "search_queries": ["planner alpha transcript"],
                                    "polarity": "affirmed",
                                },
                                {
                                    "target_key": "event_beta",
                                    "canonical_claim": "Planner canonical beta event",
                                    "subject_key": "subject_main",
                                    "claim_kind": "narrated_fact",
                                    "claim_modality": "asr",
                                    "aliases": ["planner beta"],
                                    "search_queries": ["planner beta transcript"],
                                    "polarity": "affirmed",
                                },
                            ],
                            "relations": [
                                {
                                    "relation_key": "alpha_before_beta",
                                    "kind": "before",
                                    "source_target_key": "event_alpha",
                                    "destination_target_key": "event_beta",
                                }
                            ],
                            "options": [
                                {
                                    "option_id": "A",
                                    "required_target_keys": [],
                                    "ordered_target_keys": ["event_alpha", "event_beta"],
                                    "required_relation_keys": ["alpha_before_beta"],
                                    "raw_option_text": "legacy option chunk alpha",
                                    "option_kind": "sequence",
                                },
                                {
                                    "option_id": "B",
                                    "required_target_keys": [],
                                    "ordered_target_keys": ["event_beta", "event_alpha"],
                                    "required_relation_keys": [],
                                    "raw_option_text": "legacy option chunk beta",
                                    "option_kind": "sequence",
                                },
                            ],
                            "acceptable_evidence_sources": ["asr"],
                            "confidence": 0.8,
                            "unresolved_ambiguities": [],
                        }
                    )
                )
            if request.task == "replan":
                return BackendResponse(text='{"status": "continue", "program": []}')
            return BackendResponse(text="unexpected")

    backend = GroundingThenPlanBackend()
    workspace = EvidenceWorkspace.create(tmp_path, "grounding_runtime_policy")
    agent = IterativeVisualAgent(
        backend=backend,
        registry=ToolRegistry(),
        workspace=workspace,
        scene_index=SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=60.0,
                    asr_summary="The transcript mentions planner alpha before planner beta.",
                )
            ],
        ),
        budget=AgentBudget(
            max_rounds=1,
            reserve_final_round=False,
            hard_skill_runtime=True,
            planner_owned_grounding=True,
        ),
    )

    agent.run(
        question="Which object is visible?\nA. legacy option chunk alpha\nB. legacy option chunk beta",
        video_path="/videos/demo.mp4",
    )

    assert agent._exploration_target_entities == (
        "Planner canonical alpha event",
        "Planner canonical beta event",
    )
    prompt = next(request.prompt for request in backend.requests if request.task == "replan")
    assert "Question route: temporal_order" in prompt
    assert "current_skill: narration_timeline_qa@v1" in prompt
    assert "To switch skill: set" in prompt
    trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert '"route": "temporal_order"' in trace
    assert '"recommended_skill": "narration_timeline_qa@v1"' in trace


def test_program_signature_ignores_assign_names_and_trace_ids():
    first = [
        {
            "tool": "read_segment_detail",
            "args": {"segment_id": "seg_0001", "targets": ["humble background"]},
            "assign": "obs_0017",
            "trace_id": "trace-a",
            "observation_id": "obs_0017",
        }
    ]
    second = [
        {
            "tool": "read_segment_detail",
            "args": {"targets": ["humble background"], "segment_id": "seg_0001"},
            "assign": "obs_0019",
            "trace_id": "trace-b",
            "observation_id": "obs_0019",
        }
    ]

    assert _program_signature(first) == _program_signature(second)


def build_segment_test_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption one indexed video segment.")
    def caption_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        nframes: int = 8,
    ):
        return {
            "claim": f"{segment_id} from {start_sec:.1f}s to {end_sec:.1f}s shows aircraft history.",
            "confidence": 0.72,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "nframes": nframes}],
        }

    registry.register(caption_segment)

    @tool(name="qa_segment", description="Answer a question about one indexed video segment.")
    def qa_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        nframes: int = 8,
    ):
        return {
            "claim": f"{segment_id} from {start_sec:.1f}s to {end_sec:.1f}s answers: aircraft history.",
            "confidence": 0.78,
            "input_artifacts": [video_path],
            "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "question": question, "nframes": nframes}],
        }

    registry.register(qa_segment)

    @tool(name="inspect_segment", description="Inspect one indexed video segment through a subagent boundary.")
    def inspect_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        candidate_options=None,
        nframes: int = 16,
    ):
        return {
            "claim": f"{segment_id} inspector answers: aircraft history.",
            "confidence": 0.8,
            "input_artifacts": [video_path],
            "regions": [
                {
                    "segment_id": segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "question": question,
                    "candidate_options": list(candidate_options or []),
                    "nframes": nframes,
                }
            ],
        }

    registry.register(inspect_segment)
    return registry


def build_global_route_test_registry() -> ToolRegistry:
    registry = build_segment_test_registry()

    @tool(name="global_gist", description="Inspect a sparse whole-video view.")
    def global_gist(
        video_path: str,
        question: str,
        duration_sec: float,
        nframes: int = 64,
        max_pixels: int = 151200,
        sample_offset_sec: float = 0.0,
    ):
        return {
            "claim": "Supported option: D. The sparse whole-video view shows an aviation documentary.",
            "confidence": 0.76,
            "input_artifacts": [video_path],
            "regions": [
                {
                    "start_sec": 0.0,
                    "end_sec": duration_sec,
                    "nframes": nframes,
                    "max_pixels": max_pixels,
                    "sample_offset_sec": sample_offset_sec,
                }
            ],
            "limitations": "Sparse full-video sampling.",
            "supported_option": "D",
            "grounding_quality": "global_sparse",
        }

    registry.register(global_gist)
    return registry


class IterativeAgentTest(unittest.TestCase):
    def test_global_gist_marked_exhausted_after_first_observation(self):
        class StubWorkspace:
            def observation_count(self, *, tool_name: str) -> int:
                return 1 if tool_name == "global_gist" else 0

        self.assertIn("global_gist", _exhausted_one_shot_tools(StubWorkspace()))

    def test_global_gist_not_exhausted_when_unused(self):
        class StubWorkspace:
            def observation_count(self, *, tool_name: str) -> int:
                return 0

        self.assertEqual(_exhausted_one_shot_tools(StubWorkspace()), frozenset())

    def test_agent_budget_defaults_to_answer_capable_loop(self):
        budget = AgentBudget()

        self.assertEqual(budget.max_rounds, 8)
        self.assertEqual(budget.max_tool_calls_per_round, 2)
        self.assertTrue(budget.reserve_final_round)
        self.assertEqual(budget.answer_probe_rounds_before_final, 0)

    def test_fixed_window_scene_index_creates_addressable_segments(self):
        index = fixed_window_scene_index(video_path="demo.mp4", duration_sec=65.0, window_sec=30.0)

        self.assertEqual([segment.segment_id for segment in index.segments], ["seg_0001", "seg_0002", "seg_0003"])
        self.assertEqual(index.segments[-1].start_sec, 60.0)
        self.assertEqual(index.segments[-1].end_sec, 65.0)

    def test_scene_index_summarizes_fixed_segments(self):
        index = SceneIndex(
            video_path="demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="middle"),
            ],
        )

        summary = index.summary()

        self.assertIn("seg_0001 [0.0-30.0s] opening", summary)
        self.assertIn("seg_0002 [30.0-60.0s] middle", summary)

    def test_iterative_agent_replans_after_reading_ledger(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "Need inspect likely segment.", '
                    '"program": [{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "What is discussed?"}, "assign": "middle"}]}'
                ),
                (
                    '{"status": "final", "answer": "The video discusses aircraft history.", '
                    '"citations": ["obs_0001"], "confidence": 0.78}'
                ),
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=120.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0),
                VideoSegment(segment_id="seg_0002", start_sec=60.0, end_sec=120.0, low_fps_caption="aircraft museum"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="iterative")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="What is the video mainly about?", video_path="/videos/demo.mp4")

            self.assertEqual(result.answer, "The video discusses aircraft history.")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual([round_result.status for round_result in result.rounds], ["continue", "final"])
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001"])
            self.assertEqual(len(backend.requests), 2)
            self.assertIn("Compact scene index", backend.requests[0].prompt)
            self.assertIn("seg_0002 [60.0-120.0s] aircraft museum", backend.requests[0].prompt)
            self.assertIn("Evidence ledger", backend.requests[1].prompt)
            self.assertIn("aircraft history", backend.requests[1].prompt)
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("seg_0002", ledger)
            self.assertIn("aircraft history", ledger)

    def test_compact_scene_index_uses_map_summary_not_full_dual_source_detail(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=300.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=300.0,
                    visual_caption="A long visual caption with detailed descriptions of sketches and paintings.",
                    asr_summary="A long ASR summary about Bernini biography and narration.",
                    map_summary="Bernini introduction with sketches, paintings, and biography context.",
                    entities=("Bernini", "classical painting"),
                    topic_tags=("biography",),
                    stage_tags=("intro",),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="compact_map")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What is the video about?", video_path="/videos/bernini.mp4")

        prompt = backend.requests[0].prompt

        self.assertIn("seg_0001 [0.0-300.0s] Bernini introduction with sketches", prompt)
        self.assertNotIn("Visual:", prompt)
        self.assertNotIn("ASR:", prompt)
        self.assertNotIn("Entities:", prompt)
        self.assertNotIn("A long visual caption", prompt)
        self.assertNotIn("A long ASR summary", prompt)

    def test_iterative_agent_planner_is_text_only_by_default(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=20.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=20.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="text_only")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            planner_requests = [request for request in backend.requests if request.task == "replan"]
            self.assertTrue(planner_requests)
            self.assertTrue(all(request.media_path is None for request in planner_requests))
            self.assertTrue(all(request.media_type is None for request in planner_requests))
            self.assertIn("Planner input mode: text-only", planner_requests[0].prompt)

    def test_iterative_agent_prompt_exposes_navigation_tools(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="navigation_prompt")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            prompt = backend.requests[0].prompt
            self.assertNotIn("video_ls(query", prompt)
            self.assertIn("target_coverage(targets", prompt)
            self.assertIn("search_segments(query", prompt)
            self.assertIn("read_segment(segment_id", prompt)
            self.assertIn("read_segment_detail(segment_id", prompt)
            self.assertIn("promote_answer_evidence", prompt)
            self.assertIn("locate_targets_in_segment(segment_id", prompt)
            self.assertNotIn("target_refs: list", prompt)
            self.assertIn("No target_refs are registered for this run", prompt)
            self.assertIn("Coverage-local Q<n> labels are not callable", prompt)
            self.assertIn("verify_segment_anchors(segment_id", prompt)
            self.assertNotIn("expand_window(segment_id", prompt)
            self.assertNotIn("zoom(segment_id", prompt)
            self.assertIn("inspect_segment(video_path", prompt)
            self.assertNotIn("caption_segments(segment_ids", prompt)
            self.assertNotIn("ingest_segment_metadata(segment_id", prompt)
            self.assertIn("verify_ledger_answer(answer", prompt)
            self.assertNotIn("verify_ledger_answer(answer: str, ledger_text", prompt)
            self.assertIn("summarize_ledger_evidence", prompt)
            self.assertIn("vision_read(video_path", prompt)
            self.assertIn("max_pixels", prompt)
            self.assertIn("fps", prompt)
            self.assertIn("delegate localized visual reading to one focused evidence tool", prompt)
            self.assertIn("Do not spend every round on navigation-only tools", prompt)
            self.assertIn("Local VLM tools must receive neutral factual prompts", prompt)
            self.assertIn("navigation-only evidence and locate candidates are insufficient", prompt)
            self.assertIn("answer-grade citation", prompt)

    def test_option_blind_mcq_seeds_target_coverage_before_first_planner_round(self):
        class RewriteThenPlanBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "rewrite_exploration_question":
                    return BackendResponse(
                        text=(
                            '{"exploration_question":"Describe the video segment by segment.",'
                            '"target_entities":["Aeneas, Anchises, and Ascanius fleeing Troy",'
                            '"David","The rape of Persephone","Apollo and Daphne"]}'
                        )
                    )
                if request.task == "replan":
                    return BackendResponse(text='{"status": "final", "answer": "not enough evidence yet", "citations": []}')
                return BackendResponse(text="unexpected")

        backend = RewriteThenPlanBackend([])
        scene_index = SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=1200.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0004",
                    start_sec=900.0,
                    end_sec=1200.0,
                    map_summary="David and Borghese sculpture comparison.",
                    visual_caption="Bernini's David sculpture is shown.",
                    asr_summary="The narration discusses David.",
                ),
                VideoSegment(
                    segment_id="seg_0005",
                    start_sec=1200.0,
                    end_sec=1500.0,
                    map_summary="Apollo and Daphne details.",
                    visual_caption="Apollo and Daphne is shown.",
                    asr_summary="The narration discusses Apollo and Daphne.",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="auto_target_coverage")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, rewrite_mcq_for_exploration=True),
            )

            agent.run(
                question=(
                    "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
                    "Question: In what order are the four works shown?\n"
                    "Options:\n"
                    "A. David, Apollo and Daphne, The rape of Persephone, Aeneas.\n"
                    "B. Aeneas, David, The rape of Persephone, Apollo and Daphne."
                ),
                video_path="/videos/bernini.mp4",
            )

            prompt = next(request.prompt for request in backend.requests if request.task == "replan")
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

            self.assertNotIn("Target coverage matrix", prompt)
            self.assertNotIn("target_coverage", ledger)
            self.assertNotIn("target_coverage_seeded", trace)
            self.assertEqual(workspace.observation_count(tool_name="target_coverage"), 0)

    def test_read_segment_detail_is_preserved_when_navigation_needs_visual_followup(self):
        class DetailThenFinalBackend(ScriptedPlannerBackend):
            def __init__(self):
                super().__init__([])
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "rewrite_exploration_question":
                    return BackendResponse(
                        text='{"exploration_question":"Describe the video segment by segment.","target_entities":["David"]}'
                    )
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status":"continue","program":['
                                '{"tool":"vision_read","args":{"segment_id":"seg_0001","ask_for":"open visual pass"},"assign":"v1"},'
                                '{"tool":"vision_read","args":{"segment_id":"seg_0002","ask_for":"open visual pass"},"assign":"v2"}'
                                "]}"
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status":"continue","program":['
                            '{"tool":"read_segment_detail","args":{"segment_id":"seg_0002"},"assign":"detail"}'
                            "]}"
                        )
                    )
                if request.task == "vision_read":
                    return BackendResponse(text="A visual follow-up observes David sculpture details.")
                return BackendResponse(text="unexpected")

        backend = DetailThenFinalBackend()
        scene_index = SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    visual_caption="Intro title card.",
                    asr_summary="Intro narration.",
                ),
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=30.0,
                    end_sec=60.0,
                    visual_caption="Bernini's David sculpture is shown.",
                    asr_summary="The narration discusses David.",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="preserve_detail")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend, workspace=workspace),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=2,
                    reserve_final_round=False,
                    max_tool_calls_per_round=2,
                    rewrite_mcq_for_exploration=True,
                ),
            )

            result = agent.run(
                question=(
                    "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
                    "Question: Which work appears?\n"
                    "Options:\n"
                    "A. David.\n"
                    "B. Apollo and Daphne."
                ),
                video_path="/videos/bernini.mp4",
            )

            second_program = result.rounds[1].program
            self.assertEqual([step["tool"] for step in second_program], ["read_segment_detail"])
            self.assertEqual(workspace.observation_count(tool_name="read_segment_detail"), 1)
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 2)

    def test_option_blind_timeline_vision_read_uses_unordered_target_anchors(self):
        class TargetAwareBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "rewrite_exploration_question":
                    return BackendResponse(
                        text=(
                            '{"exploration_question":"Describe the video segment by segment.",'
                            '"target_entities":["David","Apollo and Daphne"]}'
                        )
                    )
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status":"continue","skill":"timeline_ordering","program":['
                            '{"tool":"vision_read","args":{"segment_id":"seg_0001","ask_for":"Inspect the artwork ordering."},"assign":"v1"}'
                            "]}"
                        )
                    )
                if request.task == "vision_read":
                    return BackendResponse(text="The segment shows a sculpture.")
                return BackendResponse(text="unexpected")

        backend = TargetAwareBackend([])
        scene_index = SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="target_aware_prompt")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend, workspace=workspace),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False, rewrite_mcq_for_exploration=True),
            )

            agent.run(
                question=(
                    "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
                    "Question: Which artwork appears first?\n"
                    "Options:\n"
                    "A. David then Apollo and Daphne.\n"
                    "B. Apollo and Daphne then David."
                ),
                video_path="/videos/bernini.mp4",
            )

            vision_request = next(request for request in backend.requests if request.task == "vision_read")
            self.assertIn("Pay special attention to these unordered target", vision_request.prompt)
            self.assertIn("David", vision_request.prompt)
            self.assertIn("Apollo and Daphne", vision_request.prompt)
            self.assertNotIn("A. David then Apollo", vision_request.prompt)
            self.assertNotIn("B. Apollo and Daphne", vision_request.prompt)
            self.assertNotIn("option letter", vision_request.prompt)

    def test_iterative_agent_prompt_puts_scene_evidence_before_tooling(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    low_fps_caption="opening clue",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="late_tooling_prompt")
            workspace.write_observation(tool_name="vision_read", claim="prior visual fact", confidence=0.8)
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            prompt = backend.requests[0].prompt
            self.assertLess(prompt.index("Evidence ledger"), prompt.index("Available tools"))
            self.assertLess(prompt.index("Compact scene index"), prompt.index("Available tools"))
            self.assertLess(prompt.index("Current budgets"), prompt.index("Available tools"))
            self.assertGreater(prompt.rindex("Return only JSON"), prompt.index("Available tools"))

    def test_iterative_agent_prompt_includes_task_type_playbook(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_playbook")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(
                question=(
                    "Which option is visible?\n"
                    "A. aircraft museum\n"
                    "B. submarine\n"
                    "C. mountain road"
                ),
                video_path="/videos/demo.mp4",
            )

            prompt = backend.requests[0].prompt
            self.assertIn("Task playbook: multiple_choice", prompt)
            self.assertIn("candidate_options", prompt)
            self.assertIn("verify option consistency", prompt)

    def test_iterative_agent_prompt_includes_workspace_evidence_status_summary(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "B", "citations": ["obs_0001"], "confidence": 0.7}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="evidence_status_prompt")
            workspace.write_observation(
                tool_name="vision_read",
                claim="The clip shows a red aircraft.",
                confidence=0.82,
                input_artifacts=["/videos/demo.mp4"],
                regions=[{"segment_id": "seg_0001", "start_sec": 1.0, "end_sec": 2.0}],
                raw_output={
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [{"option": "B", "relation": "support", "strength": 0.82}],
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(
                question="Which option is visible?\nA. blue car\nB. red aircraft",
                video_path="/videos/demo.mp4",
            )

            prompt = backend.requests[0].prompt
            self.assertIn("Evidence status summary:", prompt)
            self.assertIn("option_coverage: 1/2", prompt)
            self.assertIn("B: strong=1 weak=0 visual=yes", prompt)

    def test_gist_global_route_seeds_one_topic_hint_without_finalizing(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "No more evidence.", "citations": [], "confidence": 0.0}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1896.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="global_route")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_global_route_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "A. cooking\n"
                    "B. a local airport scene\n"
                    "C. a city walk\n"
                    "D. an aviation documentary"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(workspace.observation_count(tool_name="global_gist"), 1)
            self.assertIn("replan", [request.task for request in backend.requests])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("global_gist_topic_seeded", trace)
            self.assertIn("iterative_final_rejected", trace)
            self.assertIn("mcq_forced_fallback", trace)

    def test_budget_can_disable_global_gist_shortcut_for_planner_trace_debugging(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "planner handled it", "citations": [], "confidence": 0.4}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1896.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="disable_global_route")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_global_route_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(disable_global_gist_route=True, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "A. cooking\n"
                    "B. a local airport scene\n"
                    "C. a city walk\n"
                    "D. an aviation documentary"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertGreaterEqual(len(backend.requests), 1)
            self.assertEqual(backend.requests[0].task, "replan")
            self.assertEqual(workspace.observation_count(tool_name="global_gist"), 0)

    def test_main_idea_vision_read_allowed_after_one_global_gist(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "inspect local coverage", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001", '
                    '"ask_for": "Describe this segment facts."}, "assign": "local"}]}'
                )
            ]
        )
        registry = build_global_route_test_registry()

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            return {
                "claim": f"{segment_id} discusses empire chronology.",
                "confidence": 0.82,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
                "event_label": event_label or ask_for,
            }

        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=60.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="allow_local_after_global")
            workspace.write_observation(
                tool_name="global_gist",
                claim="Sparse whole-video topic hint.",
                confidence=0.76,
                regions=[{"start_sec": 0.0, "end_sec": 120.0}],
                raw_output={"grounding_quality": "global_sparse", "candidate_option_hint": "D"},
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    max_tool_calls_per_round=1,
                    reserve_final_round=False,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nB. empire division\nD. empire rise and fall",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.rounds[0].program[0]["tool"], "vision_read")
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("repair_main_idea_vision_read_to_global_gist", trace)

    def test_main_idea_repeated_global_gist_is_dropped_with_reflection_memory(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "planner repeated sparse topic hint", '
                    '"skill": "main_idea", '
                    '"program": [{"tool": "global_gist", "args": {"question": "main idea", '
                    '"duration_sec": 120.0}, "assign": "repeat_global"}]}'
                )
            ]
        )
        registry = build_global_route_test_registry()

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            return {
                "claim": f"{segment_id} local coverage fact.",
                "confidence": 0.82,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=60.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="repeat_global_to_local")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=1, reserve_final_round=False),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nB. empire division\nD. empire rise and fall",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(workspace.observation_count(tool_name="global_gist"), 1)
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 0)
            self.assertEqual(result.rounds[0].program, [])
            self.assertTrue(any("global_gist" in entry and "already executed" in entry for entry in workspace.reflection_memory(max_items=10)))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("global_gist_one_shot_exhausted", trace)
            self.assertNotIn("repair_repeated_main_idea_global_gist_to_vision_read", trace)

    def test_playbook_non_suggested_skill_tool_runs_with_advisory_note(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "wrong tool", '
                    '"skill": "main_idea", '
                    '"program": [{"tool": "ground_question", "args": {"query": "empire"}, "assign": "g"}]}'
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=60.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="deny_disallowed_tool")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=False,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(question="What is the video mainly about?", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[0].program[0]["tool"], "ground_question")
            self.assertFalse(any("ground_question" in entry and "denied" in entry for entry in workspace.reflection_memory(max_items=10)))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("skill_action_advisory", trace)
            self.assertIn("non_suggested_action", trace)
            self.assertNotIn("tool_not_in_allowed_actions", trace)

    def test_segment_pool_exhaustion_is_reported_without_fallback(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "inspect all coarse segments", '
                    '"skill": "timeline_ordering", '
                    '"program": ['
                    '{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "first event"}, "assign": "v1"},'
                    '{"tool": "vision_read", "args": {"segment_id": "seg_0002", "ask_for": "second event"}, "assign": "v2"}'
                    "]}"
                ),
                (
                    '{"status": "continue", "rationale": "repeat inspected segment", '
                    '"skill": "timeline_ordering", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "order facts"}, "assign": "v"}]}'
                )
            ]
        )
        registry = ToolRegistry()

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            return {
                "claim": f"{segment_id} order fact.",
                "confidence": 0.82,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=120.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0),
                VideoSegment(segment_id="seg_0002", start_sec=60.0, end_sec=120.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="pool_exhausted")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False),
            )

            result = agent.run(question="Which event happened first?", video_path="/videos/demo.mp4")

            self.assertEqual([step["args"]["segment_id"] for step in result.rounds[0].program], ["seg_0001", "seg_0002"])
            self.assertEqual(result.rounds[1].program, [])
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 2)
            self.assertTrue(any("verify_ledger_answer" in entry or "zoom" in entry for entry in workspace.reflection_memory(max_items=10)))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("segment_pool_exhausted", trace)

    def test_coarse_caption_does_not_block_focused_subwindow(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "coarse map", '
                    '"skill": "visual_timeline_qa", '
                    '"program": [{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "map scene"}, "assign": "c"}]}'
                ),
                (
                    '{"status": "continue", "rationale": "focused order window", '
                    '"skill": "visual_timeline_qa", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0002", "start_sec": 70.0, "end_sec": 80.0, "ask_for": "describe order"}, "assign": "v"}]}'
                ),
            ]
        )
        registry = ToolRegistry()

        @tool(name="caption_segment", description="Caption one segment.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, **kwargs):
            return {
                "claim": f"coarse {segment_id}",
                "confidence": 0.7,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read a focused visual window.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, **kwargs):
            return {
                "claim": f"focused {segment_id} {start_sec}-{end_sec}",
                "confidence": 0.8,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(caption_segment)
        registry.register(vision_read)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=120.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0),
                VideoSegment(segment_id="seg_0002", start_sec=60.0, end_sec=120.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="coarse_then_focused")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False, hard_skill_runtime=True),
            )

            result = agent.run(question="Which order is shown first?", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[1].program[0]["tool"], "vision_read")
            self.assertEqual(result.rounds[1].program[0]["args"]["segment_id"], "seg_0002")
            self.assertEqual(result.rounds[1].program[0]["args"]["start_sec"], 70.0)
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("segment_pool_exhausted", trace)

    def test_normalizes_placeholder_video_path_for_global_tools(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "global_gist", "args": {"video_path": "video_path", '
                    '"question": "main idea", "duration_sec": 60.0}, "assign": "g1"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="placeholder_video_path")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_global_route_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=False,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nD. aviation documentary",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["video_path"], "/videos/demo.mp4")
            self.assertEqual(workspace.get_observation("obs_0001").input_artifacts, ["/videos/demo.mp4"])

    def test_repairs_verify_alias_to_registered_ledger_verifier(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "verify", "args": {"answer": "B", "ledger_text": "B is supported"}, "assign": "v1"}'
                    "]}"
                )
            ]
        )
        registry = ToolRegistry()
        received = {}

        @tool(name="verify_ledger_answer", description="Verify answer support.")
        def verify_ledger_answer(answer: str, ledger_text: str = ""):
            received["ledger_text"] = ledger_text
            return {"claim": f"{answer} is checked against {ledger_text}", "confidence": 0.8}

        registry.register(verify_ledger_answer)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="verify_alias")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=False,
                    hard_skill_runtime=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="What is the video mainly about?\nA. cooking\nB. empire division",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.rounds[0].program[0]["tool"], "verify_ledger_answer")
            self.assertNotIn("ledger_text", result.rounds[0].program[0]["args"])
            self.assertEqual(received["ledger_text"], "")
            self.assertEqual(workspace.observation_count(tool_name="verify_ledger_answer"), 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"type": "route_violation"', trace)

    def test_normalization_failure_surfaces_in_next_prompt(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "inspect first", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "What is visible?"}, "assign": "first"}]}'
                ),
                (
                    '{"status": "continue", "rationale": "repeat first", '
                    '"program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "What is visible?"}, "assign": "repeat"}]}'
                ),
                '{"status": "final", "answer": "The video shows aircraft history.", "citations": ["obs_0001"], "confidence": 0.7}',
            ]
        )
        registry = ToolRegistry()

        @tool(name="vision_read", description="Read localized facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            return {
                "claim": f"{segment_id} shows aircraft history.",
                "confidence": 0.75,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
                "event_label": event_label or ask_for,
            }

        registry.register(vision_read)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=120.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0),
                VideoSegment(segment_id="seg_0002", start_sec=60.0, end_sec=120.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="normalization_feedback")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, reserve_final_round=False),
            )

            agent.run(question="What is visible?", video_path="/videos/demo.mp4")

            self.assertEqual(len(backend.requests), 3)
            self.assertIn("Last Round Adjustments", backend.requests[2].prompt)
            self.assertIn("avoid_repeated_segment", backend.requests[2].prompt)
            self.assertIn("seg_0001", backend.requests[2].prompt)
            self.assertIn("seg_0002", backend.requests[2].prompt)

    def test_iterative_agent_prompt_tells_planner_to_use_option_letters_in_json(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="option_letter_prompt")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(
                question='Which artwork is first?\nA. "David"\nB. "Apollo and Daphne"',
                video_path="/videos/demo.mp4",
            )

            prompt = backend.requests[0].prompt
            self.assertNotIn("candidate_options in JSON should be option letters", prompt)
            self.assertIn("Local VLM tools must receive neutral factual prompts", prompt)
            self.assertIn("Do not copy quoted option text into JSON string values", prompt)

    def test_iterative_agent_limits_tool_calls_per_round(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect 1"}, "assign": "s1"},'
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "Inspect 2"}, "assign": "s2"},'
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0003", "question": "Inspect 3"}, "assign": "s3"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=90.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="limit_tools")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_tool_calls_per_round=1),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(len(result.rounds[0].program), 1)
            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0001")
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("tool_call_budget_exceeded", trace)
            self.assertIn("caption_segment", trace)
            self.assertIn("# Last Round Adjustments", backend.requests[1].prompt)
            self.assertIn("max_tool_calls_per_round=1", backend.requests[1].prompt)
            self.assertIn("prioritize the single highest-value evidence action", backend.requests[1].prompt)

    def test_iterative_agent_allows_multiple_visual_tools_within_round_cap(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect 1"}, "assign": "s1"},'
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0002", "question": "Inspect 2"}, "assign": "s2"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="expensive_budget")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_tool_calls_per_round=2, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual([step["tool"] for step in result.rounds[0].program], ["inspect_segment", "inspect_segment"])
            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0001")
            self.assertEqual(result.rounds[0].program[1]["args"]["segment_id"], "seg_0002")
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001", "obs_0002"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("tool_budget_exhausted", trace)

    def test_iterative_agent_avoids_repeated_segments_with_fallback(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect again"}, "assign": "repeat"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001", "obs_0002"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="dedupe")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0001")
            self.assertEqual(result.rounds[1].program[0]["args"]["segment_id"], "seg_0002")
            self.assertIn("Already inspected segments: seg_0001", backend.requests[1].prompt)

    def test_iterative_agent_can_video_ls_then_refine_candidate_segment(self):
        class NavigationFirstBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "rationale": "Need a map before inspecting pixels.", '
                                '"program": [{"tool": "video_ls", "args": {"query": "overall description aircraft", "max_segments": 3}, "assign": "map"}]}'
                            )
                        )
                    if self.replan_calls == 2:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "rationale": "The map points to seg_0002.", '
                                '"program": [{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "Describe the key event."}, "assign": "detail"}]}'
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "final", "answer": "The video is mainly about aircraft history.", '
                            '"citations": ["obs_0001", "obs_0002"], "confidence": 0.8}'
                        )
                    )
                if request.task == "caption_segment":
                    return BackendResponse(text="The segment shows aircraft history exhibits.")
                return BackendResponse(text="unexpected")

        backend = NavigationFirstBackend()
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening greetings"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="aircraft museum history"),
                VideoSegment(segment_id="seg_0003", start_sec=60.0, end_sec=90.0, low_fps_caption="closing credits"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="video_ls_loop")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3),
            )

            result = agent.run(question="Describe the video.", video_path="/videos/demo.mp4")

            self.assertEqual([round_result.program[0]["tool"] for round_result in result.rounds[:2]], ["video_ls", "caption_segment"])
            self.assertEqual(result.answer, "The video is mainly about aircraft history.")
            self.assertEqual([request.task for request in backend.requests], ["replan", "replan", "caption_segment", "replan"])
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("Candidate segments", ledger)
            self.assertIn("seg_0002", ledger)
            self.assertIn("aircraft history exhibits", ledger)

    def test_iterative_agent_does_not_inject_media_args_into_navigation_segment_tools(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "read_segment", "args": {"segment_id": "seg_0002"}, "assign": "metadata"}'
                    "]}"
                ),
                '{"status": "final", "answer": "metadata read", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, low_fps_caption="opening"),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0, low_fps_caption="aircraft museum"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="read_segment_args")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Describe the video.", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[0].program[0]["tool"], "read_segment")
            self.assertEqual(result.rounds[0].program[0]["args"], {"segment_id": "seg_0002"})
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("aircraft museum", ledger)

    def test_iterative_agent_upgrades_empty_read_segment_to_caption_segment(self):
        class ReadUpgradeBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "read_segment", "args": {"segment_id": "seg_0002"}, "assign": "metadata"}'
                                "]}"
                            )
                        )
                    return BackendResponse(text='{"status": "final", "answer": "captioned", "citations": ["obs_0001"]}')
                if request.task == "caption_segment":
                    return BackendResponse(text="The segment visually shows aircraft exhibits.")
                return BackendResponse(text="unexpected")

        backend = ReadUpgradeBackend()
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="upgrade_empty_read")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=VideoMap.from_scene_index(scene_index), backend=backend),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Describe the video.", video_path="/videos/demo.mp4")

            self.assertEqual(result.rounds[0].program[0]["tool"], "caption_segment")
            self.assertEqual(result.rounds[0].program[0]["args"]["segment_id"], "seg_0002")
            self.assertEqual(backend.requests[1].task, "caption_segment")
            ledger = (workspace.root / "ledger.md").read_text(encoding="utf-8")
            self.assertIn("aircraft exhibits", ledger)

    def test_iterative_agent_resolves_segment_id_into_tool_arguments(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=40.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=5.0, end_sec=17.5)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="resolve")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Inspect the clip", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["video_path"], "/videos/demo.mp4")
            self.assertEqual(tool_args["start_sec"], 5.0)
            self.assertEqual(tool_args["end_sec"], 17.5)
            self.assertEqual(tool_args["nframes"], 128)

    def test_iterative_agent_persists_planner_prompt_and_response_artifacts(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "rationale": "inspect once", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=40.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=5.0, end_sec=17.5)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_io")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="Inspect the clip", video_path="/videos/demo.mp4")

            prompt_path = workspace.root / "artifacts" / "planner_io" / "round_0001_prompt.txt"
            response_path = workspace.root / "artifacts" / "planner_io" / "round_0001_response.txt"
            self.assertTrue(prompt_path.exists())
            self.assertTrue(response_path.exists())
            self.assertIn("Question: Inspect the clip", prompt_path.read_text(encoding="utf-8"))
            self.assertIn('"rationale": "inspect once"', response_path.read_text(encoding="utf-8"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"planner_io"', trace)
            self.assertIn("round_0001_prompt.txt", trace)
            self.assertIn("round_0001_response.txt", trace)

    def test_iterative_agent_allows_zoom_child_segments_with_explicit_window(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "qa_segment", "args": {"segment_id": "seg_0002_z01", "start_sec": 30.0, "end_sec": 45.0, "question": "Which option is supported?"}, "assign": "zoomed_qa"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0),
                VideoSegment(segment_id="seg_0002", start_sec=30.0, end_sec=60.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="dynamic_zoom_child")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="Which option is supported?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["segment_id"], "seg_0002_z01")
            self.assertEqual(tool_args["start_sec"], 30.0)
            self.assertEqual(tool_args["end_sec"], 45.0)
            self.assertEqual(result.rounds[0].program[0]["tool"], "qa_segment")

    def test_iterative_agent_resolves_zoom_child_id_from_prior_zoom_observation(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "zoom", "args": {"segment_id": "seg_0001", "target_granularity_sec": 10.0}, "assign": "zoomed"}'
                    "]}"
                ),
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001_z02", "question": "Inspect child"}, "assign": "child"}'
                    "]}"
                ),
            ]
        )
        store = VideoMapStore(
            VideoMap(
                video_path="/videos/demo.mp4",
                duration_sec=30.0,
                segments=[VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
            )
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="zoom_child_from_trace")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_exploration_registry(video_map=store, backend=backend, workspace=workspace),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False),
            )

            result = agent.run(question="Which child is visible?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[1].program[0]["args"]
            self.assertEqual(tool_args["segment_id"], "seg_0001_z02")
            self.assertEqual(tool_args["start_sec"], 10.0)
            self.assertEqual(tool_args["end_sec"], 20.0)

    def test_iterative_agent_stops_repeated_identical_programs_as_no_progress(self):
        repeated_program = (
            '{"status": "continue", "rationale": "same map", "program": ['
            '{"tool": "video_ls", "args": {"query": "same segment"}, "assign": "map"}'
            "]}"
        )
        backend = ScriptedPlannerBackend([repeated_program] * 6)
        registry = ToolRegistry()
        calls = {"video_ls": 0}

        @tool(name="video_ls", description="Return the same map.")
        def video_ls(query: str = ""):
            calls["video_ls"] += 1
            return {"claim": f"same candidate for {query}", "confidence": 1.0}

        registry.register(video_ls)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_progress")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=6,
                    reserve_final_round=False,
                    max_repeated_programs=2,
                ),
            )

            result = agent.run(question="Find the same thing", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(len(result.rounds), 2)
            self.assertEqual(calls["video_ls"], 2)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_no_progress_guard", trace)
            self.assertIn("repeated_program", trace)
            self.assertIn("repeated_program_blocked", trace)

    def test_route_repair_second_repeat_proposes_recovery_without_executing(self):
        repeated_locator = (
            '{"status": "continue", "skill": "timeline_ordering", "program": ['
            '{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "targets": ["David"]}}'
            "]}"
        )
        backend = ScriptedPlannerBackend([repeated_locator, repeated_locator])
        registry = ToolRegistry()
        verify_calls = {"count": 0}

        @tool(name="locate_targets_in_segment", description="Locate targets in a segment.")
        def locate_targets_in_segment(segment_id: str, targets: list | None = None):
            return {"claim": f"located {segment_id}: {targets}", "confidence": 0.4}

        @tool(name="verify_segment_anchors", description="Verify located anchors.")
        def verify_segment_anchors(
            segment_id: str,
            anchors: list,
            targets: list | None = None,
            question: str = "",
            start_sec: float = 0.0,
            end_sec: float = 0.0,
        ):
            verify_calls["count"] += 1
            return {"claim": f"verified {segment_id}", "confidence": 0.8}

        registry.register(locate_targets_in_segment)
        registry.register(verify_segment_anchors)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="route_repair_recovery_proposal")
            workspace.write_observation(
                tool_name="locate_targets_in_segment",
                claim="locator produced anchors",
                confidence=0.8,
                raw_output={
                    "segment_id": "seg_0001",
                    "verify_call_args": {
                        "segment_id": "seg_0001",
                        "anchors": [{"anchor_id": "a1", "segment_id": "seg_0001"}],
                        "targets": ["David"],
                    },
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(question="Which order is shown?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(verify_calls["count"], 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("route_repair_applied", trace)
            self.assertIn("route_repair_recovery_proposed", trace)
            self.assertIn("read_segment_detail", trace)

    def test_ordered_list_recovery_uses_focused_vision_before_anchor_verify(self):
        repeated_locator = (
            '{"status": "continue", "skill": "visual_timeline_qa", "program": ['
            '{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0002", "target_refs": ["T1", "T2", "T3", "T4"]}}'
            "]}"
        )
        backend = ScriptedPlannerBackend([repeated_locator])
        registry = ToolRegistry()
        calls = {"vision": 0, "verify": 0}

        @tool(name="locate_targets_in_segment", description="Locate targets in a segment.")
        def locate_targets_in_segment(segment_id: str, target_refs: list | None = None):
            return {"claim": f"located {segment_id}: {target_refs}", "confidence": 0.4}

        @tool(name="vision_read", description="Read a focused visual window.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float = 0.0,
            end_sec: float = 0.0,
            ask_for: str = "",
            event_label: str = "",
            nframes: int = 0,
        ):
            calls["vision"] += 1
            return {
                "claim": f"focused {segment_id} {start_sec}-{end_sec}",
                "confidence": 0.8,
                "grounding_quality": "visually_confirmed",
                "requested_nframes": nframes,
            }

        @tool(name="verify_segment_anchors", description="Verify located anchors.")
        def verify_segment_anchors(segment_id: str, anchors: list, targets: list | None = None):
            calls["verify"] += 1
            return {"claim": f"verified {segment_id}", "confidence": 0.8}

        registry.register(locate_targets_in_segment)
        registry.register(vision_read)
        registry.register(verify_segment_anchors)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=600.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="ordered_route_repair_focused_vision")
            workspace.target_registry = TargetRegistry.from_specs(
                targets=[
                    TargetSpec("T1", "Aeneas, Anchises, and Ascanius fleeing Troy"),
                    TargetSpec("T2", "David"),
                    TargetSpec("T3", "The rape of Persephone"),
                    TargetSpec("T4", "Apollo and Daphne"),
                ],
                options=[OptionSpec("D", target_sequence=("T1", "T2", "T3", "T4"))],
            )
            workspace.write_observation(
                tool_name="locate_targets_in_segment",
                claim="ordered-list candidate found",
                confidence=1.0,
                raw_output={
                    "segment_id": "seg_0002",
                    "recommended_next_actions": [
                        {
                            "candidate_id": "ordered_list_obs_0001_0",
                            "candidate_type": "ordered_list",
                            "route_kind": "focused_ordered_list_vision",
                            "tool": "vision_read",
                            "target_refs": ["T1", "T2", "T3", "T4"],
                            "args": {
                                "segment_id": "seg_0002",
                                "start_sec": 536.227,
                                "end_sec": 569.488,
                                "ask_for": "Describe the visible artworks in timestamp order.",
                                "event_label": "focused_ordered_list_candidate_4_items",
                                "nframes": 128,
                            },
                        }
                    ],
                    "verify_call_args": {
                        "segment_id": "seg_0002",
                        "anchors": [{"anchor_id": "a1", "segment_id": "seg_0002"}],
                        "targets": ["Aeneas", "David"],
                    },
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(question=BERNINI_ORDER_QUESTION, video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(calls["vision"], 1)
            self.assertEqual(calls["verify"], 0)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("route_recovery_selected", trace)
            self.assertIn("focused_ordered_list_vision_executed", trace)
            self.assertIn("mcq_forced_fallback", trace)
            self.assertNotIn("repair_repeated_locator_to_verify_segment_anchors", trace)

    def test_parse_error_executes_unique_safe_pending_action(self):
        backend = ScriptedPlannerBackend(["not json"])
        registry = ToolRegistry()
        calls = {"vision": 0}

        @tool(name="vision_read", description="Read a focused visual window.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 0,
        ):
            calls["vision"] += 1
            return {
                "claim": f"focused {segment_id} {start_sec}-{end_sec}",
                "confidence": 0.8,
                "grounding_quality": "visually_confirmed",
                "requested_nframes": nframes,
            }

        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=600.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="parse_error_safe_pending_action")
            workspace.write_observation(
                tool_name="locate_targets_in_segment",
                claim="ordered-list candidate found",
                confidence=1.0,
                raw_output={
                    "segment_id": "seg_0002",
                    "recommended_next_actions": [
                        {
                            "candidate_id": "cand_0005",
                            "candidate_type": "ordered_list",
                            "route_kind": "focused_ordered_list_vision",
                            "tool": "vision_read",
                            "target_refs": ["T1", "T2", "T3", "T4"],
                            "args": {
                                "segment_id": "seg_0002",
                                "start_sec": 536.227,
                                "end_sec": 569.488,
                                "ask_for": "Describe the visible artworks in timestamp order.",
                                "event_label": "focused_ordered_list_candidate_4_items",
                                "nframes": 128,
                            },
                        }
                    ],
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(question=BERNINI_ORDER_QUESTION, video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(calls["vision"], 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_json_parse_error", trace)
            self.assertIn("planner_parse_error_recovery_selected", trace)
            self.assertIn("focused_ordered_list_vision_executed", trace)
            self.assertIn("mcq_forced_fallback", trace)

    def test_narration_recovery_uses_transcript_detail_before_anchor_verify(self):
        repeated_locator = (
            '{"status": "continue", "skill": "narration_timeline_qa", "program": ['
            '{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "target_refs": ["T1", "T2", "T3"]}}'
            "]}"
        )
        backend = ScriptedPlannerBackend([repeated_locator])
        registry = ToolRegistry()
        calls = {"detail": 0, "verify": 0}

        @tool(name="locate_targets_in_segment", description="Locate targets in a segment.")
        def locate_targets_in_segment(segment_id: str, target_refs: list | None = None):
            return {"claim": f"located {segment_id}: {target_refs}", "confidence": 0.4}

        @tool(name="read_segment_detail", description="Read transcript detail.")
        def read_segment_detail(segment_id: str, target_refs: list | None = None, promote_answer_evidence: bool = False):
            calls["detail"] += 1
            return {
                "claim": f"detail {segment_id}: {target_refs}",
                "confidence": 0.8,
                "answer_evidence_rows": [],
                "promote_answer_evidence": promote_answer_evidence,
            }

        @tool(name="verify_segment_anchors", description="Verify located anchors.")
        def verify_segment_anchors(segment_id: str, anchors: list, targets: list | None = None):
            calls["verify"] += 1
            return {"claim": f"verified {segment_id}", "confidence": 0.8}

        registry.register(locate_targets_in_segment)
        registry.register(read_segment_detail)
        registry.register(verify_segment_anchors)
        scene_index = fixed_window_scene_index(video_path="/videos/goya.mp4", duration_sec=600.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="narration_route_repair_transcript")
            workspace.target_registry = TargetRegistry.from_specs(
                targets=[
                    TargetSpec("T1", "humble background", modality_hint=ClaimModality.NARRATED_FACT),
                    TargetSpec("T2", "entered upper class", modality_hint=ClaimModality.NARRATED_FACT),
                    TargetSpec("T3", "seclusion/farmhouse", modality_hint=ClaimModality.NARRATED_FACT),
                ],
                options=[OptionSpec("B", target_sequence=("T1", "T2", "T3"))],
            )
            workspace.write_observation(
                tool_name="locate_targets_in_segment",
                claim="locator produced generic anchors",
                confidence=1.0,
                raw_output={
                    "segment_id": "seg_0001",
                    "verify_call_args": {
                        "segment_id": "seg_0001",
                        "anchors": [{"anchor_id": "a1", "segment_id": "seg_0001"}],
                        "targets": ["humble background"],
                    },
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(question="How was his life journey according to the video?", video_path="/videos/goya.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(calls["detail"], 1)
            self.assertEqual(calls["verify"], 0)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("route_recovery_selected", trace)
            self.assertIn("narration_transcript_promotion_executed", trace)
            self.assertNotIn("repair_repeated_locator_to_verify_segment_anchors", trace)

    def test_route_repair_third_repeat_hard_stops(self):
        repeated_locator = (
            '{"status": "continue", "skill": "timeline_ordering", "program": ['
            '{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "targets": ["David"]}}'
            "]}"
        )
        backend = ScriptedPlannerBackend([repeated_locator, repeated_locator, repeated_locator])
        registry = ToolRegistry()
        verify_calls = {"count": 0}

        @tool(name="locate_targets_in_segment", description="Locate targets in a segment.")
        def locate_targets_in_segment(segment_id: str, targets: list | None = None):
            return {"claim": f"located {segment_id}: {targets}", "confidence": 0.4}

        @tool(name="verify_segment_anchors", description="Verify located anchors.")
        def verify_segment_anchors(
            segment_id: str,
            anchors: list,
            targets: list | None = None,
            question: str = "",
            start_sec: float = 0.0,
            end_sec: float = 0.0,
        ):
            verify_calls["count"] += 1
            return {"claim": f"verified {segment_id}", "confidence": 0.8}

        registry.register(locate_targets_in_segment)
        registry.register(verify_segment_anchors)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="route_repair_exhausted")
            workspace.write_observation(
                tool_name="locate_targets_in_segment",
                claim="locator produced anchors",
                confidence=0.8,
                raw_output={
                    "segment_id": "seg_0001",
                    "verify_call_args": {
                        "segment_id": "seg_0001",
                        "anchors": [{"anchor_id": "a1", "segment_id": "seg_0001"}],
                        "targets": ["David"],
                    },
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(question="Which order is shown?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "route_repair_exhausted")
            self.assertEqual(verify_calls["count"], 1)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("route_repair_exhausted", trace)
            self.assertIn("repair_repeated_locator_to_verify_segment_anchors", trace)

    def test_route_repair_count_is_target_key_specific(self):
        locator_for_david = (
            '{"status": "continue", "skill": "timeline_ordering", "program": ['
            '{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "targets": ["David"]}}'
            "]}"
        )
        locator_for_apollo = (
            '{"status": "continue", "skill": "timeline_ordering", "program": ['
            '{"tool": "locate_targets_in_segment", "args": {"segment_id": "seg_0001", "targets": ["Apollo"]}}'
            "]}"
        )
        backend = ScriptedPlannerBackend([locator_for_david, locator_for_apollo, locator_for_david])
        registry = ToolRegistry()
        verify_calls = {"count": 0}

        @tool(name="locate_targets_in_segment", description="Locate targets in a segment.")
        def locate_targets_in_segment(segment_id: str, targets: list | None = None):
            return {"claim": f"located {segment_id}: {targets}", "confidence": 0.4}

        @tool(name="verify_segment_anchors", description="Verify located anchors.")
        def verify_segment_anchors(
            segment_id: str,
            anchors: list,
            targets: list | None = None,
            question: str = "",
            start_sec: float = 0.0,
            end_sec: float = 0.0,
        ):
            verify_calls["count"] += 1
            return {"claim": f"verified {segment_id}: {targets}", "confidence": 0.8}

        registry.register(locate_targets_in_segment)
        registry.register(verify_segment_anchors)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="route_repair_key_specific")
            workspace.write_observation(
                tool_name="locate_targets_in_segment",
                claim="locator produced anchors",
                confidence=0.8,
                raw_output={
                    "segment_id": "seg_0001",
                    "verify_call_args": {
                        "segment_id": "seg_0001",
                        "anchors": [{"anchor_id": "a1", "segment_id": "seg_0001"}],
                        "targets": ["David"],
                    },
                },
            )
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(question="Which order is shown?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(verify_calls["count"], 2)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("route_repair_recovery_proposed", trace)

    def test_no_progress_warning_after_three_rounds_without_supported_binding(self):
        backend = ScriptedPlannerBackend(
            [
                '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "first"}}]}',
                '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "second"}}]}',
                '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "third"}}]}',
            ]
        )
        registry = ToolRegistry()

        @tool(name="video_ls", description="Return navigation-only hints.")
        def video_ls(query: str = ""):
            return {"claim": f"navigation hint for {query}", "confidence": 0.4}

        registry.register(video_ls)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="supported_binding_no_progress")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(question="Find the narrated evidence", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_no_progress_warning", trace)
            self.assertIn("supported_evidence_binding_no_growth", trace)

    def test_iterative_agent_repairs_media_tool_missing_segment_id_from_time_window(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {'
                    '"video_path": "artifacts/clips/seg_0002.mp4", '
                    '"start_sec": 30.0, "end_sec": 45.0, "question": "Inspect this window"}'
                    ', "assign": "detail"}'
                    "]}"
                ),
                '{"status": "final", "answer": "done", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="repair_missing_segment")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertTrue(tool_args["segment_id"].startswith("window_"))
            self.assertEqual(tool_args["video_path"], "/videos/demo.mp4")
            self.assertEqual(tool_args["start_sec"], 30.0)
            self.assertEqual(tool_args["end_sec"], 45.0)
            self.assertEqual(result.rounds[0].observation_ids, ["obs_0001"])

    def test_iterative_agent_clamps_dynamic_tail_window_to_video_duration(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {'
                    '"start_sec": 1800.0, "end_sec": 1805.0, "question": "Inspect the tail"}'
                    ', "assign": "tail"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1804.96, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="tail_window")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens at the end?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["start_sec"], 1800.0)
            self.assertEqual(tool_args["end_sec"], 1804.96)
            self.assertTrue(tool_args["segment_id"].startswith("window_"))

    def test_iterative_agent_resolves_reused_dynamic_window_segment_id(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "window_000300000_000600000", '
                    '"question": "Reinspect this dynamic window"}'
                    ', "assign": "dyn"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=900.0, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="reused_dynamic_window")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["segment_id"], "window_000300000_000600000")
            self.assertEqual(tool_args["start_sec"], 300.0)
            self.assertEqual(tool_args["end_sec"], 600.0)

    def test_iterative_agent_normalizes_dynamic_window_milliseconds(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"start_sec": 1800000.0, "end_sec": 1804957.0, '
                    '"question": "Inspect this millisecond window"}'
                    ', "assign": "dyn"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=1804.957, window_sec=300.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="dynamic_window_ms")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(tool_args["start_sec"], 1800.0)
            self.assertEqual(tool_args["end_sec"], 1804.957)
            self.assertEqual(tool_args["segment_id"], "window_001800000_001804957")

    def test_iterative_agent_keeps_mcq_options_out_of_inspector_args(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Which option is visible?"}, "assign": "inspection"}'
                    "]}"
                ),
                '{"status": "final", "answer": "A", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_inspector_mcq_options")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question="Which option is visible?\nA. aircraft museum\nB. submarine",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertNotIn("candidate_options", tool_args)
            self.assertIn("Which option is visible?", tool_args["question"])

    def test_iterative_agent_strips_planner_candidate_options_from_inspector_args(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Which option is visible?", '
                    '"candidate_options": ["A", "B"]}, "assign": "inspection"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="strip_planner_mcq_options")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which option is visible?\nA. aircraft museum\nB. submarine",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertNotIn("candidate_options", tool_args)

    def test_iterative_agent_recovers_when_planner_copies_unescaped_option_quotes(self):
        malformed_planner_json = (
            '{"status": "continue", "program": [{"tool": "inspect_segment", '
            '"args": {"segment_id": "seg_0001", "question": "Which artwork appears after "David"?", '
            '"candidate_options": ["A. "David" then Apollo", "B. plain option"]}, "assign": "bad_json"}]}'
        )
        class BadJsonRecoveryBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    self.requests.append(request)
                    return BackendResponse(
                        text='{"answer": "A. The cited inspection supports A.", "citations": ["obs_0001"], '
                        '"confidence": 0.8, "candidate_option_relations": ['
                        '{"option": "A", "relation": "support", "strength": 0.8, "observation_id": "obs_0001"}]}'
                    )
                return super().generate(request)

        backend = BadJsonRecoveryBackend(
            [
                malformed_planner_json,
                '{"status": "final", "answer": "A. The cited inspection supports A.", "citations": ["obs_0001"]}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_bad_json_recovery")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question='Which artwork appears after David?\nA. "David" then Apollo\nB. plain option',
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.rounds[0].program[0]["tool"], "inspect_segment")
            self.assertNotIn("candidate_options", result.rounds[0].program[0]["args"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_json_parse_error", trace)

    def test_iterative_agent_rewrites_mcq_caption_question_without_options(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Describe the sequence."}, '
                    '"assign": "caption"}'
                    "]}"
                )
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="caption_options")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which order is shown?\nA. red then blue\nB. blue then red",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertIn("Describe the sequence", tool_args["question"])
            self.assertIn("Do not choose an option", tool_args["question"])
            self.assertNotIn("Options:", tool_args["question"])
            self.assertNotIn("A. red then blue", tool_args["question"])
            self.assertNotIn("B. blue then red", tool_args["question"])

    def test_option_blind_inspect_segment_does_not_receive_candidate_options(self):
        class RewriteInspectBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "rewrite_exploration_question":
                    return BackendResponse(
                        text=(
                            '{"exploration_question":"Describe the actual visible objects and narrated facts.",'
                            '"focus_points":["visible facts"],"target_entities":["aircraft","submarine"]}'
                        )
                    )
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status":"continue","program":[{"tool":"inspect_segment","args":{'
                            '"segment_id":"seg_0001","question":"Inspect option A. aircraft museum"},'
                            '"assign":"inspection"}]}'
                        )
                    )
                return BackendResponse(text='{"answer":"need_more_evidence","citations":[],"missing_evidence":[]}')

        backend = RewriteInspectBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="option_blind_inspect")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=False,
                    rewrite_mcq_for_exploration=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="Which option is visible?\nA. aircraft museum\nB. submarine",
                video_path="/videos/demo.mp4",
            )

            tool_args = result.rounds[0].program[0]["args"]
            self.assertEqual(result.rounds[0].program[0]["tool"], "inspect_segment")
            self.assertNotIn("candidate_options", tool_args)
            self.assertNotIn("option A", tool_args["question"])
            self.assertNotIn("aircraft museum", tool_args["question"])
            self.assertEqual(tool_args["question"], "Describe the actual visible objects and narrated facts.")

    def test_iterative_agent_blocks_mcq_final_until_answer_grade_evidence(self):
        class McqFinalBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    self.requests.append(request)
                    return BackendResponse(
                        text='{"answer": "A. aircraft museum", "citations": ["obs_0001"], "confidence": 0.9, '
                        '"candidate_option_relations": [{"option": "A", "relation": "support", '
                        '"strength": 0.9, "observation_id": "obs_0001"}]}'
                    )
                return super().generate(request)

        backend = McqFinalBackend(
            [
                '{"status": "final", "answer": "A", "citations": [], "confidence": 0.9}',
                '{"status": "final", "answer": "A", "citations": ["obs_0001"], "confidence": 0.9}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="block_unsupported_mcq_final")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            result = agent.run(
                question="Which option is visible?\nA. aircraft museum\nB. submarine",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual(result.rounds[0].status, "continue")
            self.assertEqual(result.rounds[0].program[0]["tool"], "inspect_segment")
            self.assertNotIn("candidate_options", result.rounds[0].program[0]["args"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_final_blocked", trace)

    def test_planner_final_mcq_is_not_replaced_by_answer_agent_final(self):
        class FinalVerifierBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.answer_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text='{"status": "final", "answer": "A", "citations": ["obs_0001"], "confidence": 0.99}'
                    )
                if request.task == "answer_from_evidence":
                    assert "A. wrong option" in request.prompt
                    assert "B. correct option from evidence" in request.prompt
                    self.answer_calls += 1
                    if self.answer_calls > 1:
                        return BackendResponse(
                            text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["planner verifier already disagreed"]}'
                        )
                    return BackendResponse(
                        text=(
                            '{"answer": "B. correct option from evidence", "citations": ["obs_0001"], '
                            '"confidence": 0.86, "candidate_option_relations": ['
                            '{"option": "B", "relation": "support", "strength": 0.86, '
                            '"observation_id": "obs_0001", "mapped_from_facts": ["obs_0001"]}]}'
                        )
                    )
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_final_takeover")
            workspace.write_observation(tool_name="vision_read", claim="Evidence supports B.", confidence=0.9)
            agent = IterativeVisualAgent(
                backend=FinalVerifierBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which option is supported?\nA. wrong option\nB. correct option from evidence",
                video_path="/videos/demo.mp4",
            )

            self.assertNotEqual(result.status, "final")
            self.assertFalse(result.answer.startswith("B"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_final_answer_verifier", trace)
            self.assertIn("planner_final_verifier_disagrees", trace)

    def test_iterative_agent_blocks_main_idea_planner_final_without_structured_support(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "vision_read", "args": {"segment_id": "seg_0001", '
                    '"ask_for": "Describe the opening collapse evidence."}, "assign": "v1"}'
                    "]}"
                ),
                '{"status": "final", "answer": "C. the empire declining and collapsing", '
                '"citations": ["obs_0001"], "confidence": 0.91}',
            ]
        )
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)
        registry = build_segment_test_registry()

        @tool(name="vision_read", description="Read localized visual evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 16,
        ):
            return {
                "claim": "The sampled segment shows only an ending collapse scene, not a full-video arc.",
                "confidence": 0.82,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "nframes": nframes}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="block_main_idea_planner_final")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "A. the empire's allies\n"
                    "B. why the empire was divided\n"
                    "C. the empire declining and collapsing\n"
                    "D. the empire's rise, stability, decline, and collapse"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertNotEqual(result.status, "final")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_final_blocked", trace)
            self.assertIn("planner_final_answer_verifier", trace)
            self.assertIn("selected_option_has_structured_support", trace)

    def test_planner_final_is_used_when_answer_agent_abstains_but_visual_citations_are_valid(self):
        class AbstainingAnswerBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "final", "skill": "main_idea@v1", '
                            '"answer": "D. the empire rise and fall", '
                            '"citations": ["obs_0001", "obs_0002"], "confidence": 0.71}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "missing_evidence": ["more explicit theme"], '
                            '"citations": [], "confidence": 0.0}'
                        )
                    )
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_final_after_answer_abstain")
            workspace.write_observation(
                tool_name="vision_read",
                claim="The video shows the empire's rise and fall as the main arc.",
                confidence=0.82,
                regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
                raw_output={"grounding_quality": "visually_confirmed", "supported_option": "D"},
            )
            workspace.write_observation(
                tool_name="vision_read",
                claim="The segment connects the empire's collapse to a broad historical narrative.",
                confidence=0.8,
                regions=[{"segment_id": "seg_0001", "start_sec": 12.0, "end_sec": 24.0}],
                raw_output={"grounding_quality": "visually_confirmed", "supported_option": "D"},
            )
            agent = IterativeVisualAgent(
                backend=AbstainingAnswerBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "A. a local battle\n"
                    "B. a narrow map\n"
                    "C. a single treaty\n"
                    "D. the empire rise and fall"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_final_answer_verifier", trace)
            self.assertIn("planner_final_after_answer_agent_verifier", trace)

    def test_planner_final_after_abstain_can_use_single_answer_grade_citation(self):
        class OneObservationCitationBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "final", "skill": "main_idea@v1", '
                            '"answer": "D. the empire rise and fall", '
                            '"citations": ["obs_0001", "ev_same_obs"], "confidence": 0.71}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "confidence": 0.0}')
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="planner_final_one_answer_grade_citation")
            workspace.write_observation(
                tool_name="vision_read",
                claim="The video shows the empire's rise and fall as the main arc.",
                confidence=0.82,
                regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
                raw_output={"grounding_quality": "visually_confirmed", "supported_option": "D"},
            )
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_same_obs",
                    "obs_id": "obs_0001",
                    "tool": "vision_read",
                    "supported_option": "D",
                    "claim": "The same observation supports option D.",
                    "confidence": 0.82,
                    "grounding_quality": "visually_confirmed",
                }
            )
            agent = IterativeVisualAgent(
                backend=OneObservationCitationBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "A. a local battle\n"
                    "B. a narrow map\n"
                    "C. a single treaty\n"
                    "D. the empire rise and fall"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_final_answer_verifier", trace)
            self.assertIn("planner_final_after_answer_agent_verifier", trace)

    def test_unique_supported_binding_can_fill_narration_evidence_id(self):
        class NarrationFinalWithoutEvidenceIdBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "final", "skill": "narration_timeline_qa@v1", '
                            '"answer": "B", "citations": ["obs_0001"], "confidence": 0.83}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "confidence": 0.0}')
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="narration_final_requires_evidence_id")
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_bind_seg_0001_T1",
                    "obs_id": "obs_0001",
                    "tool": "timeline_asr_summary",
                    "segment_id": "seg_0001",
                    "claim": "The narration says he became a painter.",
                    "confidence": 0.91,
                    "grounding_quality": "indexed_transcript",
                    "supported_option": "B",
                    "candidate_option_relations": [
                        {"option": "B", "relation": "support", "strength": 0.91, "observation_id": "obs_0001"}
                    ],
                    "evidence_binding": {
                        "status": "supported",
                        "claim_modality": "narrated_fact",
                        "target_id": "T1",
                        "segment_id": "seg_0001",
                    },
                }
            )
            agent = IterativeVisualAgent(
                backend=NarrationFinalWithoutEvidenceIdBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="According to the narration, what happened?\nA. He became a sculptor\nB. He became a painter",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.evidence_ids, ["ev_bind_seg_0001_T1"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_final_evidence_id_bridge", trace)

    def test_multiple_supported_bindings_are_not_guessed_for_narration_final(self):
        class NarrationFinalWithoutEvidenceIdBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "final", "skill": "narration_timeline_qa@v1", '
                            '"answer": "B", "citations": ["obs_0001"], "confidence": 0.83}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "confidence": 0.0}')
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="narration_final_multiple_bindings")
            for target_id in ("T1", "T2"):
                workspace.write_evidence_row(
                    {
                        "evidence_id": f"ev_bind_seg_0001_{target_id}",
                        "obs_id": "obs_0001",
                        "tool": "timeline_asr_summary",
                        "segment_id": "seg_0001",
                        "claim": f"The narration supports {target_id}.",
                        "confidence": 0.91,
                        "grounding_quality": "indexed_transcript",
                        "supported_option": "B",
                        "candidate_option_relations": [
                            {"option": "B", "relation": "support", "strength": 0.91, "observation_id": "obs_0001"}
                        ],
                        "evidence_binding": {
                            "status": "supported",
                            "claim_modality": "narrated_fact",
                            "target_id": target_id,
                            "segment_id": "seg_0001",
                        },
                    }
                )
            agent = IterativeVisualAgent(
                backend=NarrationFinalWithoutEvidenceIdBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="According to the narration, what happened?\nA. He became a sculptor\nB. He became a painter",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "evidence_repair_exhausted")
            self.assertEqual(result.evidence_ids, [])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_final_evidence_id_bridge", trace)
            self.assertIn("candidate_evidence_ids", trace)
            self.assertIn("evidence_repair_exhausted", trace)

    def test_narration_timeline_planner_final_accepts_explicit_supported_evidence_id(self):
        class NarrationFinalWithEvidenceIdBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "final", "skill": "narration_timeline_qa@v1", '
                            '"answer": "B", "citations": ["obs_0001"], '
                            '"evidence_ids": ["ev_bind_seg_0001_T1"], "confidence": 0.83}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "confidence": 0.0}')
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="narration_final_accepts_evidence_id")
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_bind_seg_0001_T1",
                    "obs_id": "obs_0001",
                    "tool": "timeline_asr_summary",
                    "segment_id": "seg_0001",
                    "claim": "The narration says he became a painter.",
                    "confidence": 0.91,
                    "grounding_quality": "indexed_transcript",
                    "supported_option": "B",
                    "candidate_option_relations": [
                        {"option": "B", "relation": "support", "strength": 0.91, "observation_id": "obs_0001"}
                    ],
                    "evidence_binding": {
                        "status": "supported",
                        "claim_modality": "narrated_fact",
                        "target_id": "T1",
                        "segment_id": "seg_0001",
                    },
                }
            )
            agent = IterativeVisualAgent(
                backend=NarrationFinalWithEvidenceIdBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="According to the narration, what happened?\nA. He became a sculptor\nB. He became a painter",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual(result.evidence_ids, ["ev_bind_seg_0001_T1"])
            self.assertEqual(result.to_dict()["output"]["evidence_ids"], ["ev_bind_seg_0001_T1"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_final_after_answer_agent_verifier", trace)

    def test_narration_prefinal_repair_promotes_real_transcript_evidence_id(self):
        class NarrationFinalNeedsRepairBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "final", "skill": "visual_timeline_qa@v1", '
                            '"answer": "B", "citations": ["obs_0001"], "confidence": 0.83}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "confidence": 0.0}')
                raise AssertionError(request.task)

        question = (
            "How was his life journey according to the video?\n"
            "A. He was born wealthy and stayed in court.\n"
            "B. Born with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
            "C. Born with a humble background, lived in seclusion in a farmhouse and then entered the upper class."
        )
        scene_index = SceneIndex(
            video_path="/videos/goya.mp4",
            duration_sec=60.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=60.0,
                    asr_summary=(
                        "Goya was a man from a humble background who rose through the ranks to reach the upper "
                        "class, then withdrew into a farmhouse."
                    ),
                )
            ],
        )
        video_map = VideoMap.from_scene_index(scene_index)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="narration_prefinal_repair")
            workspace.target_registry = TargetRegistry.from_specs(
                targets=[
                    TargetSpec("T1", "humble background", subject="Goya", modality_hint=ClaimModality.NARRATED_FACT),
                    TargetSpec("T2", "upper class", aliases=("upper",), subject="Goya", modality_hint=ClaimModality.NARRATED_FACT),
                    TargetSpec("T3", "farmhouse", subject="Goya", modality_hint=ClaimModality.NARRATED_FACT),
                ],
                options=[
                    OptionSpec("B", target_sequence=("T1", "T2", "T3"), required_relations=("R1", "R2")),
                    OptionSpec("C", target_sequence=("T1", "T3", "T2"), required_relations=("R3", "R4")),
                ],
                relations=[
                    ClaimRelation("R1", "before", "T1", "T2"),
                    ClaimRelation("R2", "before", "T2", "T3"),
                    ClaimRelation("R3", "before", "T1", "T3"),
                    ClaimRelation("R4", "before", "T3", "T2"),
                ],
            )
            agent = IterativeVisualAgent(
                backend=NarrationFinalNeedsRepairBackend(),
                registry=build_video_navigation_registry(video_map, workspace=workspace),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False, hard_skill_runtime=True),
            )

            result = agent.run(question=question, video_path="/videos/goya.mp4")

            self.assertEqual(result.status, "evidence_repair_exhausted")
            self.assertIn("candidate option B", result.answer)
            self.assertEqual(result.evidence_ids, [])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("skill_recommended", trace)
            self.assertIn("skill_transition_rejected", trace)
            self.assertIn("prefinal_evidence_repair_requested", trace)
            self.assertNotIn("planner_final_after_prefinal_evidence_repair", trace)

    def test_repair_failure_has_terminal_evidence_repair_exhausted_status(self):
        class NarrationFinalNoRepairCandidateBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "final", "skill": "narration_timeline_qa@v1", '
                            '"answer": "B", "citations": ["obs_0001"], "confidence": 0.83}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "confidence": 0.0}')
                raise AssertionError(request.task)

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="repair_failure_terminal")
            agent = IterativeVisualAgent(
                backend=NarrationFinalNoRepairCandidateBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="According to the narration, what happened?\nA. He became a sculptor\nB. He became a painter",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "evidence_repair_exhausted")
            self.assertNotEqual(result.status, "max_rounds_reached")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("prefinal_evidence_repair_failed", trace)
            self.assertIn("evidence_repair_exhausted", trace)
            self.assertIn('"selected_option": ""', trace)
            self.assertIn('"candidate_option": "B"', trace)

    def test_post_repair_allows_only_one_action_round(self):
        class PostRepairActionBackend(VisionLanguageBackend):
            def __init__(self):
                self.replan_count = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    self.replan_count += 1
                    if self.replan_count == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "final", "skill": "narration_timeline_qa@v1", '
                                '"answer": "B", "citations": ["obs_0001"], "confidence": 0.83}'
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "skill": "narration_timeline_qa@v1", '
                            '"program": [{"tool": "read_segment_detail", "args": {"segment_id": "seg_0001", '
                            '"targets": ["painter"]}}]}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "confidence": 0.0}')
                raise AssertionError(request.task)

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=30.0, asr_summary="He painted.")],
        )
        video_map = VideoMap.from_scene_index(scene_index)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="post_repair_one_action")
            backend = PostRepairActionBackend()
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_video_navigation_registry(video_map, workspace=workspace),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=4, reserve_final_round=False),
            )

            result = agent.run(
                question="According to the narration, what happened?\nA. He became a sculptor\nB. He became a painter",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "evidence_repair_exhausted")
            self.assertEqual(backend.replan_count, 2)
            self.assertLessEqual(len(result.rounds), 2)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("prefinal_evidence_repair_failed", trace)
            self.assertIn("evidence_repair_exhausted", trace)

    def test_iterative_agent_indexes_scene_coverage_for_main_idea_mcq(self):
        class SceneCoverageBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "use indexed coverage rows", '
                            '"citations": [], "missing_evidence": ["more coverage"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text='{"status": "continue", "program": []}')

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=90.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    low_fps_caption="ASR/subtitle excerpt: the empire is created and rises with economic growth.",
                ),
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=30.0,
                    end_sec=60.0,
                    low_fps_caption="ASR/subtitle excerpt: internal stability, governance, and prosperity are discussed.",
                ),
                VideoSegment(
                    segment_id="seg_0003",
                    start_sec=60.0,
                    end_sec=90.0,
                    low_fps_caption="ASR/subtitle excerpt: war pressure leads to decline, collapse, and the fall of the empire.",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="scene_coverage_main_idea")
            agent = IterativeVisualAgent(
                backend=SceneCoverageBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "B. Why the Austro-Hungarian Empire was divided\n"
                    "D. How the Austro-Hungarian Empire rises and falls"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            table = workspace.evidence_table_v2(
                question="What is the video mainly about?",
                options=[
                    "B. Why the Austro-Hungarian Empire was divided",
                    "D. How the Austro-Hungarian Empire rose and fell",
                ],
            )
            self.assertEqual(table["groups"]["D"], [])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("mcq_forced_fallback", trace)

    def test_iterative_agent_indexes_scene_order_for_videomme_masterpiece_sequence(self):
        class SceneOrderBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "use indexed order rows", '
                            '"citations": [], "missing_evidence": ["ordered artwork evidence"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text='{"status": "continue", "program": []}')

        scene_index = SceneIndex(
            video_path="/videos/bernini.mp4",
            duration_sec=1200.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=300.0,
                    low_fps_caption='The first Borghese masterpiece shown is "Aeneas, Anchises, and Ascanius fleeing Troy".',
                ),
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=300.0,
                    end_sec=600.0,
                    low_fps_caption='The next sculpture presented is "David" by Bernini.',
                ),
                VideoSegment(
                    segment_id="seg_0003",
                    start_sec=600.0,
                    end_sec=900.0,
                    low_fps_caption='The scene then discusses "The rape of Persephone".',
                ),
                VideoSegment(
                    segment_id="seg_0004",
                    start_sec=900.0,
                    end_sec=1200.0,
                    low_fps_caption='The final sculpture in the sequence is "Apollo and Daphne".',
                ),
            ],
        )

        question = (
            "VideoMME multiple-choice question. Answer with exactly one option letter (A/B/C/D) first.\n"
            "Question: As depicted in the video, in what order does the author present Bernini's four "
            "masterpieces created for Borghese in a single scene?\n"
            "Options:\n"
            'A. "The rape of Persephone", "Apollo and Daphne", "David" and "Aeneas, Anchises, and Ascanius fleeing Troy".\n'
            'B. "David", "Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne" and "The rape of Persephone".\n'
            'C. "Apollo and Daphne", "Aeneas, Anchises, and Ascanius fleeing Troy", "David" and "The rape of Persephone".\n'
            'D. "Aeneas, Anchises, and Ascanius fleeing Troy", "David", "The rape of Persephone" and "Apollo and Daphne".'
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="scene_order_masterpieces")
            agent = IterativeVisualAgent(
                backend=SceneOrderBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    reserve_final_round=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(question=question, video_path="/videos/bernini.mp4")

            self.assertNotEqual(result.status, "final")
            self.assertNotIn("timeline_temporal_order", (workspace.root / "trace.jsonl").read_text(encoding="utf-8"))

    def test_iterative_agent_indexes_scene_order_for_life_journey_when_planner_selects_timeline_skill(self):
        class LifeJourneyBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": []}')
                return BackendResponse(
                    text='{"status": "continue", "skill": "timeline_ordering", "program": []}'
                )

        scene_index = SceneIndex(
            video_path="/videos/goya.mp4",
            duration_sec=900.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=300.0,
                    low_fps_caption="The narration says he was born with a humble background.",
                ),
                VideoSegment(
                    segment_id="seg_0002",
                    start_sec=300.0,
                    end_sec=600.0,
                    low_fps_caption="Later he entered the upper class as his work gained noble patrons.",
                ),
                VideoSegment(
                    segment_id="seg_0003",
                    start_sec=600.0,
                    end_sec=900.0,
                    low_fps_caption="In the final part he lived in seclusion in a farmhouse.",
                ),
            ],
        )
        question = (
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: How was his life journey according to the video?\n"
            "Options:\n"
            "A. Borned with humble background and lived in seclusion in a farmhouse.\n"
            "B. Borned with a humble background, entered the upper class and then lived in seclusion in a farmhouse.\n"
            "C. Borned with a humble background, lived in seclusion in a farmhouse and then entered the upper class.\n"
            "D. Borned in the upper class and lived in seclusion in a farmhouse."
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="scene_order_life_journey")
            agent = IterativeVisualAgent(
                backend=LifeJourneyBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=True, disable_global_gist_route=True),
            )

            result = agent.run(question=question, video_path="/videos/goya.mp4")

            self.assertEqual(result.status, "low_confidence_final")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("planner_skill_selection", trace)
            self.assertNotIn("timeline_temporal_order", trace)
            self.assertIn("mcq_forced_fallback", trace)

    def test_iterative_agent_prompt_includes_broad_long_video_index(self):
        backend = ScriptedPlannerBackend(
            ['{"status": "final", "answer": "not enough evidence yet", "citations": []}']
        )
        scene_index = fixed_window_scene_index(video_path="/videos/long.mp4", duration_sec=600.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="long_index")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
            )

            agent.run(question="What happens near the end?", video_path="/videos/long.mp4")

            self.assertIn("seg_0020 [570.0-600.0s]", backend.requests[0].prompt)

    def test_iterative_agent_stops_at_round_budget_with_partial_evidence(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "caption_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                )
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="budget")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertIn("Partial evidence summary", result.answer)
            self.assertIn("aircraft history", result.answer)

    def test_iterative_agent_returns_low_confidence_when_budget_exhausts_with_partial_support(self):
        class LowConfidenceBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", '
                            '"question": "Inspect option evidence", '
                            '"candidate_options": ["A. submarine", "B. aircraft museum"]}, "assign": "s1"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "partial support only", '
                            '"citations": [], '
                            '"candidate_option_relations": ['
                            '{"option": "B", "relation": "support", "strength": 0.8, '
                            '"observation_id": "obs_0001", "grounding_quality": "visually_confirmed"}'
                            '], "missing_evidence": ["need one more local view"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text="unexpected")

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="low_conf_budget")
            agent = IterativeVisualAgent(
                backend=LowConfidenceBackend(),
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which option is visible?\nA. submarine\nB. aircraft museum",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(result.answer, "B")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertAlmostEqual(result.confidence, 0.5)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("budget_exhausted", trace)
            self.assertIn("low_confidence_final", trace)

    def test_iterative_agent_low_confidence_from_arbitration_partial_support_at_budget(self):
        class CloseSupportBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect first order"}, "assign": "a"},'
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0002", "question": "Inspect fourth order"}, "assign": "d"}'
                            "]}"
                        )
                    )
                return BackendResponse(text="unexpected")

        registry = ToolRegistry()

        @tool(name="inspect_segment", description="Inspect one segment with close option support.")
        def inspect_segment(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            question: str,
            candidate_options=None,
            nframes: int = 16,
        ):
            option = "A" if segment_id == "seg_0001" else "D"
            confidence = 0.70 if option == "A" else 0.66
            return {
                "claim": f"{segment_id} visually supports option {option}.",
                "confidence": confidence,
                "candidate_option_relations": [
                    {
                        "option": option,
                        "relation": "support",
                        "strength": confidence,
                        "grounding_quality": "visually_confirmed",
                    }
                ],
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec, "nframes": nframes}],
            }

        registry.register(inspect_segment)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=20.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0),
                VideoSegment(segment_id="seg_0002", start_sec=10.0, end_sec=20.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="low_conf_arbitration_budget")
            agent = IterativeVisualAgent(
                backend=CloseSupportBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=1,
                    max_tool_calls_per_round=2,
                    reserve_final_round=False,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(
                question="Which order is shown?\nA. first order\nD. fourth order",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(result.answer, "A")
            self.assertEqual(result.citations, ["obs_0001"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("resolve close support", trace)
            self.assertIn("low_confidence_final", trace)

    def test_iterative_agent_blocks_low_confidence_without_visual_citation(self):
        class NoVisualLowConfidenceBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "video_ls", "args": {"query": "aircraft"}, "assign": "map"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "partial map-only support", '
                            '"citations": [], '
                            '"candidate_option_relations": ['
                            '{"option": "B", "relation": "support", "strength": 0.8, '
                            '"observation_id": "obs_0001", "grounding_quality": "visually_confirmed"}'
                            '], "missing_evidence": ["need visual confirmation"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text="unexpected")

        registry = ToolRegistry()

        @tool(name="video_ls", description="Return a navigation-only map row.")
        def video_ls(query: str = ""):
            return {"claim": f"navigation candidate for {query}", "confidence": 1.0}

        registry.register(video_ls)
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="low_conf_blocked")
            agent = IterativeVisualAgent(
                backend=NoVisualLowConfidenceBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question="Which option is visible?\nA. submarine\nB. aircraft museum",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(result.answer, "B")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("low_confidence_final_blocked", trace)
            self.assertIn("mcq_forced_fallback", trace)

    def test_iterative_agent_reserves_final_round_from_new_visual_tools(self):
        backend = ScriptedPlannerBackend(
            [
                (
                    '{"status": "continue", "program": ['
                    '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                    "]}"
                )
            ]
        )
        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=10.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=10.0)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="reserved_final")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=True),
            )

            result = agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "max_rounds_reached")
            self.assertEqual(result.citations, [])
            self.assertEqual(result.rounds[0].program, [])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("reserve_final_round", trace)

    def test_iterative_agent_uses_answer_agent_when_reserved_final_round_would_continue(self):
        class ReservedFinalAnswerBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect"}, "assign": "s1"}'
                                "]}"
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0002", "question": "Inspect more"}, "assign": "s2"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    self.answer_prompt = request.prompt
                    return BackendResponse(
                        text='{"answer": "B", "rationale": "obs_0001 supports B.", "citations": ["obs_0001"], "missing_evidence": [], "confidence": 0.82}'
                    )
                return BackendResponse(text="unexpected")

        backend = ReservedFinalAnswerBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="reserved_final_answer")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=2, reserve_final_round=True),
            )

            result = agent.run(
                question="Which option is visible?\nA. submarine\nB. aircraft museum",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "B")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertIn("obs_0001", backend.answer_prompt)
            self.assertEqual([request.task for request in backend.requests], ["replan", "replan", "answer_from_evidence"])

    def test_iterative_agent_answer_agent_arbitrates_option_grouped_evidence(self):
        class ArbitrationBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    segment_id = f"seg_{self.replan_calls:04d}"
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "'
                            + segment_id
                            + '", "question": "Inspect option support", '
                            '"candidate_options": ["A. first", "D. fourth"]}, "assign": "s"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text='{"answer": "A. first", "rationale": "recency fallback", "citations": ["obs_0002"], "missing_evidence": [], "confidence": 0.95}'
                    )
                return BackendResponse(text="unexpected")

        registry = ToolRegistry()

        @tool(name="inspect_segment", description="Inspect one segment with option support.")
        def inspect_segment(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            question: str,
            candidate_options=None,
            nframes: int = 16,
        ):
            if segment_id == "seg_0001":
                return {
                    "claim": "Visual evidence supports option D.",
                    "confidence": 0.72,
                    "input_artifacts": [f"{video_path}#t={start_sec:.1f},{end_sec:.1f}"],
                    "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                    "limitations": "Directly visible in the sampled segment.",
                    "grounding_quality": "visually_confirmed",
                    "candidate_option_relations": [
                        {"option": "D", "relation": "support", "strength": 0.72, "assigned_by": "answer_agent"}
                    ],
                }
            return {
                "claim": "Caption-like evidence guesses option A.",
                "confidence": 0.95,
                "input_artifacts": [f"{video_path}#t={start_sec:.1f},{end_sec:.1f}"],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "limitations": "Inferred from context; lacks explicit visual confirmation.",
                "candidate_option_relations": [
                    {"option": "A", "relation": "support", "strength": 0.95, "assigned_by": "answer_agent"}
                ],
            }

        registry.register(inspect_segment)
        backend = ArbitrationBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=90.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="answer_arbitration")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=3, reserve_final_round=True),
            )

            result = agent.run(
                question="Which option is correct?\nA. first\nD. fourth",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D. fourth")
            self.assertEqual(result.citations, ["obs_0001"])
            self.assertEqual([request.task for request in backend.requests], ["replan", "replan", "replan"])

    def test_iterative_agent_feeds_prefinal_answer_gaps_into_next_prompt(self):
        class PrefinalProbeBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0
                self.answer_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0001", "question": "Inspect first option", '
                                '"candidate_options": ["A. first", "B. second", "C. third", "D. fourth"]}, "assign": "s1"}'
                                "]}"
                            )
                        )
                    if self.replan_calls == 2:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0002", "question": "Inspect second option", '
                                '"candidate_options": ["A. first", "B. second", "C. third", "D. fourth"]}, "assign": "s2"}'
                                "]}"
                            )
                        )
                    if self.replan_calls == 3:
                        self.round3_prompt = request.prompt
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "inspect_segment", "args": {"segment_id": "seg_0003", "question": "Resolve the missing ordering evidence", '
                                '"candidate_options": ["A. first", "B. second", "C. third", "D. fourth"]}, "assign": "s3"}'
                                "]}"
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "inspect_segment", "args": {"segment_id": "seg_0004", "question": "Should be reserved"}, "assign": "s4"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    self.answer_calls += 1
                    if self.answer_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"answer": "need_more_evidence", "rationale": "Need order evidence.", '
                                '"citations": [], "missing_evidence": ["explicit order of the four options"], "confidence": 0.0}'
                            )
                        )
                    return BackendResponse(
                        text='{"answer": "D", "rationale": "obs_0003 resolves the order.", "citations": ["obs_0003"], "missing_evidence": [], "confidence": 0.86}'
                    )
                return BackendResponse(text="unexpected")

        backend = PrefinalProbeBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="prefinal_answer_probe")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=4, reserve_final_round=True, answer_probe_rounds_before_final=2),
            )

            result = agent.run(
                question="Which option is correct?\nA. first\nB. second\nC. third\nD. fourth",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D")
            self.assertIn("Answer Agent says these evidence gaps", backend.round3_prompt)
            self.assertIn("explicit order of the four options", backend.round3_prompt)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("prefinal_probe", trace)
            self.assertIn("reserved_final", trace)

    def test_prefinal_answer_final_is_planner_visible_suggestion(self):
        class PrefinalFinalBackend(VisionLanguageBackend):
            def __init__(self):
                self.replan_calls = 0
                self.answer_calls = 0
                self.round3_prompt = ""

            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 3:
                        self.round3_prompt = request.prompt
                    segment_id = f"seg_000{self.replan_calls}"
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            f'{{"tool": "inspect_segment", "args": {{"segment_id": "{segment_id}", '
                            '"question": "Inspect candidate evidence", '
                            '"candidate_options": ["A. first", "B. second", "C. third", "D. fourth"]}}, '
                            f'"assign": "s{self.replan_calls}"}}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    self.answer_calls += 1
                    return BackendResponse(
                        text='{"answer": "D", "rationale": "observations support D", '
                        '"citations": ["obs_0001"], "missing_evidence": [], "confidence": 0.86}'
                    )
                return BackendResponse(text="unexpected")

        backend = PrefinalFinalBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=120.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="prefinal_final_suggestion")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=build_segment_test_registry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=4, reserve_final_round=True, answer_probe_rounds_before_final=2),
            )

            result = agent.run(
                question="Which option is correct?\nA. first\nB. second\nC. third\nD. fourth",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertEqual(result.answer, "D")
            self.assertIn("AnswerAgent suggestion from prefinal_probe", backend.round3_prompt)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_answer_suggestion", trace)
            self.assertIn('"source": "prefinal_probe"', trace)
            self.assertIn('"source": "reserved_final"', trace)

    def test_temporal_ordering_uses_timeline_for_unique_option(self):
        registry = ToolRegistry()

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            if start_sec < 20.0:
                claim = "The light turns on in this early segment."
            elif start_sec >= 40.0:
                claim = "The door opens in this late segment."
            else:
                claim = "No target event appears here."
            return {
                "claim": claim,
                "confidence": 0.82,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read temporal event timestamp.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
        ):
            observed = 40.5 if "door" in ask_for.lower() else 10.5
            return {
                "claim": f"{event_label} is observed at {observed:.1f} seconds.",
                "confidence": 0.94,
                "event_label": event_label,
                "observed_at_sec": observed,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "grounding_quality": "visually_confirmed",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class NeedMoreBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "rationale": "timeline should decide before this", '
                            '"citations": [], "missing_evidence": ["more evidence"], "confidence": 0.0}'
                        )
                    )
                return BackendResponse(text='{"status": "final", "answer": "fallback", "citations": []}')

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_temporal")
            agent = IterativeVisualAgent(
                backend=NeedMoreBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=4, hard_skill_runtime=True),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertLessEqual(workspace.observation_count(tool_name="caption_segment"), 1)
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 0)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("skill_recommended", trace)
            self.assertIn("visual_timeline_qa@v1", trace)
            self.assertNotIn("timeline_caption_", trace)
            self.assertNotIn("iterative_timeline_temporal_decision", trace)
            self.assertIn("iterative_final_rejected", trace)
            self.assertIn("mcq_forced_fallback", trace)

    def test_interactive_timeline_locator_rows_do_not_auto_final(self):
        video_map = VideoMap(
            video_path="/videos/bernini.mp4",
            duration_sec=600.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0002",
                    start_sec=300.0,
                    end_sec=600.0,
                    asr_sentences=[
                        {
                            "start_sec": 497.12,
                            "end_sec": 539.097,
                            "text": (
                                "The detail of the corpulent Cardinal's button is a classic Bernini touch. "
                                'It is the same attention to detail that we will see with "Apollo and Daphne". '
                                'Then the narration lists radical and colossal marble statues "Aeneas, Anchises, and Ascanius fleeing Troy", '
                                '"David", "The rape of Persephone",'
                            ),
                        },
                        {
                            "start_sec": 539.3,
                            "end_sec": 546.0,
                            "text": 'and "Apollo and Daphne".',
                        },
                    ],
                )
            ],
        )
        question = (
            "As depicted in the video, in what order does the author present Bernini's four masterpieces created for Borghese in a single scene?\n"
            'A. "Apollo and Daphne", "David", "The rape of Persephone", "Aeneas, Anchises, and Ascanius fleeing Troy"\n'
            'B. "David", "Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne", "The rape of Persephone"\n'
            'C. "The rape of Persephone", "Apollo and Daphne", "David", "Aeneas, Anchises, and Ascanius fleeing Troy"\n'
            'D. "Aeneas, Anchises, and Ascanius fleeing Troy", "David", "The rape of Persephone", "Apollo and Daphne"'
        )

        class OrderedLocatorBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status":"continue","skill":"timeline_ordering","rationale":"read ASR list",'
                            '"program":[{"tool":"locate_targets_in_segment","args":{"segment_id":"seg_0002",'
                            '"targets":["Aeneas, Anchises, and Ascanius fleeing Troy","David",'
                            '"The rape of Persephone","Apollo and Daphne"]}}]}'
                        )
                    )
                return BackendResponse(
                    text='{"answer":"need_more_evidence","citations":[],"missing_evidence":["should decide from timeline"]}'
                )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="interactive_ordered_asr")
            registry = build_video_navigation_registry(video_map, workspace=workspace)
            agent = IterativeVisualAgent(
                backend=OrderedLocatorBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=SceneIndex(
                    video_path="/videos/bernini.mp4",
                    duration_sec=600.0,
                    segments=[VideoSegment(segment_id="seg_0002", start_sec=300.0, end_sec=600.0)],
                ),
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=2, reserve_final_round=False),
            )

            result = agent.run(question=question, video_path="/videos/bernini.mp4")

            self.assertEqual(result.status, "low_confidence_final")
            self.assertNotEqual(result.answer, "D")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("iterative_timeline_temporal_decision", trace)
            self.assertIn("mcq_forced_fallback", trace)

    def test_case_611_2_read_segment_detail_ordered_asr_replay_final_selects_d(self):
        video_map = VideoMap(
            video_path="/videos/bernini.mp4",
            duration_sec=60.0,
            segments=[
                VideoMapSegment(segment_id="seg_0001", start_sec=0.0, end_sec=20.0),
                VideoMapSegment(
                    segment_id="seg_0002",
                    start_sec=20.0,
                    end_sec=40.0,
                    asr_text=(
                        'The narration presents the Bernini works in this order: "Aeneas", '
                        '"David", "The rape of Persephone", and "Apollo and Daphne".'
                    ),
                ),
            ],
        )
        question = (
            "Which order are the artworks presented in?\n"
            'A. "The rape of Persephone", "Apollo and Daphne", "David", "Aeneas"\n'
            'B. "David", "Aeneas", "Apollo and Daphne", "The rape of Persephone"\n'
            'C. "Apollo and Daphne", "The rape of Persephone", "Aeneas", "David"\n'
            'D. "Aeneas", "David", "The rape of Persephone", "Apollo and Daphne"'
        )

        class ReadThenFinalBackend(VisionLanguageBackend):
            def __init__(self):
                self.replan_count = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    self.replan_count += 1
                    if self.replan_count == 1:
                        return BackendResponse(
                            text=(
                                '{"status":"continue","skill":"visual_timeline_qa@v1",'
                                '"program":[{"tool":"read_segment_detail","args":{"segment_id":"seg_0002",'
                                '"target_refs":["T1","T2","T3","T4"],"promote_answer_evidence":true}}]}'
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status":"final","skill":"visual_timeline_qa@v1","answer":"D",'
                            '"evidence_ids":["seq_seg_0002"],"citations":[],"confidence":0.94}'
                        )
                    )
                return BackendResponse(text='{"answer":"D","citations":["ev_answer_obs_0001_01"],"confidence":0.94}')

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="case_611_2_ordered_asr_replay")
            workspace.target_registry = TargetRegistry.from_specs(
                targets=[
                    TargetSpec("T1", "Aeneas"),
                    TargetSpec("T2", "David"),
                    TargetSpec("T3", "The rape of Persephone"),
                    TargetSpec("T4", "Apollo and Daphne"),
                ],
                options=[
                    OptionSpec("A", target_sequence=("T3", "T4", "T2", "T1"), option_kind="sequence"),
                    OptionSpec("B", target_sequence=("T2", "T1", "T4", "T3"), option_kind="sequence"),
                    OptionSpec("C", target_sequence=("T4", "T3", "T1", "T2"), option_kind="sequence"),
                    OptionSpec("D", target_sequence=("T1", "T2", "T3", "T4"), option_kind="sequence"),
                ],
            )
            registry = build_video_navigation_registry(video_map, workspace=workspace)
            agent = IterativeVisualAgent(
                backend=ReadThenFinalBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=SceneIndex(
                    video_path="/videos/bernini.mp4",
                    duration_sec=60.0,
                    segments=[
                        VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=20.0),
                        VideoSegment(segment_id="seg_0002", start_sec=20.0, end_sec=40.0),
                    ],
                ),
                budget=AgentBudget(max_rounds=2, max_tool_calls_per_round=1, reserve_final_round=False),
            )

            result = agent.run(question=question, video_path="/videos/bernini.mp4")

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"gate_status": "accepted"', trace)
            self.assertIn("structured_final_gate", trace)
            self.assertIn("seq_seg_0002", trace)

    def test_case_605_1_two_segment_main_idea_replay_final_selects_d(self):
        video_map = VideoMap(
            video_path="/videos/empire.mp4",
            duration_sec=90.0,
            segments=[
                VideoMapSegment(
                    segment_id="seg_0001",
                    start_sec=0.0,
                    end_sec=45.0,
                    asr_text="The narration introduces the rise of an ancient empire through rulers and monuments.",
                ),
                VideoMapSegment(
                    segment_id="seg_0002",
                    start_sec=45.0,
                    end_sec=90.0,
                    asr_text="The segment describes battles, succession crises, and the fall of the empire into ruins.",
                ),
            ],
        )
        question = (
            "What is the video mainly about?\n"
            "A. A cooking contest\n"
            "B. A city traffic update\n"
            "C. A sports highlight reel\n"
            "D. The rise and fall of an ancient empire"
        )

        class ReadBothThenFinalBackend(VisionLanguageBackend):
            def __init__(self):
                self.replan_count = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    self.replan_count += 1
                    if self.replan_count == 1:
                        return BackendResponse(
                            text=(
                                '{"status":"continue","skill":"main_idea@v1","program":['
                                '{"tool":"read_segment_detail","args":{"segment_id":"seg_0001",'
                                '"target_refs":["T1"],"promote_answer_evidence":true}},'
                                '{"tool":"read_segment_detail","args":{"segment_id":"seg_0002",'
                                '"target_refs":["T2"],"promote_answer_evidence":true}}]}'
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status":"final","skill":"main_idea@v1","answer":"D",'
                            '"evidence_ids":["ev_bind_seg_0001_T1","ev_bind_seg_0002_T2"],'
                            '"citations":[],"confidence":0.92}'
                        )
                    )
                return BackendResponse(text='{"answer":"D","citations":["obs_0001","obs_0002"],"confidence":0.92}')

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="case_605_1_main_idea_replay")
            workspace.target_registry = TargetRegistry.from_specs(
                targets=[
                    TargetSpec("T1", "rise of an ancient empire"),
                    TargetSpec("T2", "fall of the empire"),
                ],
                options=[OptionSpec("D", target_sequence=("T1", "T2"), option_kind="topic_arc")],
            )
            registry = build_video_navigation_registry(video_map, workspace=workspace)
            agent = IterativeVisualAgent(
                backend=ReadBothThenFinalBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=SceneIndex(
                    video_path="/videos/empire.mp4",
                    duration_sec=90.0,
                    segments=[
                        VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=45.0),
                        VideoSegment(segment_id="seg_0002", start_sec=45.0, end_sec=90.0),
                    ],
                ),
                budget=AgentBudget(max_rounds=2, max_tool_calls_per_round=2, reserve_final_round=False),
            )

            result = agent.run(question=question, video_path="/videos/empire.mp4")

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"gate_status": "accepted"', trace)
            self.assertIn("structured_final_gate", trace)

    def test_confirmed_timeline_inference_is_prompt_hint_not_auto_final(self):
        registry = ToolRegistry()

        @tool(name="vision_read", description="Read visible temporal anchors.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str,
            nframes: int = 128,
            max_pixels: int = 151200,
            fps: float = 0.0,
        ):
            observed_at = 5.0 if event_label == "light appears" else 20.0
            return {
                "claim": f"{event_label} is visually confirmed.",
                "confidence": 0.92,
                "event_label": event_label,
                "observed_at_sec": observed_at,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)
        question = (
            "In what order are the events shown in the video?\n"
            "A. door opens, light appears\n"
            "B. light appears, door opens"
        )

        class PendingInferenceBackend(VisionLanguageBackend):
            def __init__(self):
                self.replan_prompts = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "replan":
                    self.replan_prompts.append(request.prompt)
                    if len(self.replan_prompts) == 1:
                        return BackendResponse(
                            text='{"status":"continue","skill":"timeline_ordering","rationale":"verify anchors",'
                            '"program":['
                            '{"tool":"vision_read","args":{"video_path":"/videos/demo.mp4","segment_id":"seg_0001",'
                            '"start_sec":0,"end_sec":10,"ask_for":"first event","event_label":"light appears"}},'
                            '{"tool":"vision_read","args":{"video_path":"/videos/demo.mp4","segment_id":"seg_0002",'
                            '"start_sec":15,"end_sec":25,"ask_for":"second event","event_label":"door opens"}}]}'
                        )
                    self.assert_pending_prompt = request.prompt
                    return BackendResponse(
                        text='{"status":"final","skill":"timeline_ordering","answer":"B",'
                        '"citations":["obs_0001"],"confidence":0.8}'
                    )
                return BackendResponse(text='{"answer":"need_more_evidence","citations":[]}')

        backend = PendingInferenceBackend()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="pending_timeline_inference")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=SceneIndex(
                    video_path="/videos/demo.mp4",
                    duration_sec=30.0,
                    segments=[
                        VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=15.0),
                        VideoSegment(segment_id="seg_0002", start_sec=15.0, end_sec=30.0),
                    ],
                ),
                budget=AgentBudget(max_rounds=2, max_tool_calls_per_round=2, reserve_final_round=False),
            )

            result = agent.run(question=question, video_path="/videos/demo.mp4")

            self.assertEqual(len(backend.replan_prompts), 2)
            self.assertNotEqual(result.status, "final")
            self.assertIn("# Pending Inference", backend.replan_prompts[1])
            self.assertIn("option B", backend.replan_prompts[1])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("iterative_timeline_temporal_inference", trace)
            self.assertNotIn("iterative_timeline_temporal_decision", trace)

    def test_single_scene_final_requires_short_window_covering_targets(self):
        question = (
            "As depicted in the video, in what order are these works presented in a single scene?\n"
            'A. "alpha", "beta", "gamma"\n'
            'B. "gamma", "beta", "alpha"'
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="single_scene_gate")
            long_obs = workspace.write_observation(
                tool_name="verify_segment_anchors",
                claim="alpha, beta, gamma are visible but over a broad window.",
                confidence=0.9,
                raw_output={
                    "grounding_quality": "visually_confirmed",
                    "confirmations": [{"target": "alpha"}, {"target": "beta"}, {"target": "gamma"}],
                    "ordered_visible_in_window": ["alpha", "beta", "gamma"],
                    "verify_windows": [{"start_sec": 10.0, "end_sec": 90.0}],
                },
            )

            blocked = _blocked_final_reason(
                question=question,
                has_inspect_with_candidate_options=True,
                workspace=workspace,
                answer="A",
                citations=[long_obs.observation_id],
            )

            self.assertEqual(blocked, "single_scene_subwindow_vision_read_missing")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="single_scene_gate_ok")
            short_obs = workspace.write_observation(
                tool_name="verify_segment_anchors",
                claim="alpha, beta, gamma are visible in one short window.",
                confidence=0.9,
                raw_output={
                    "grounding_quality": "visually_confirmed",
                    "confirmations": [{"target": "alpha"}, {"target": "beta"}, {"target": "gamma"}],
                    "ordered_visible_in_window": ["alpha", "beta", "gamma"],
                    "verify_windows": [{"start_sec": 10.0, "end_sec": 40.0}],
                },
            )

            blocked = _blocked_final_reason(
                question=question,
                has_inspect_with_candidate_options=True,
                workspace=workspace,
                answer="A",
                citations=[short_obs.observation_id],
            )

            self.assertEqual(blocked, "")

    def test_timeline_ordering_uses_caption_pass_before_focused_reads(self):
        registry = ToolRegistry()
        calls = []

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            calls.append(("caption_segment", segment_id, question))
            claim_by_segment = {
                "seg_0001": "The light turns on in this early segment.",
                "seg_0002": "The door opens in this later segment.",
                "seg_0003": "A closing title appears.",
            }
            return {
                "claim": claim_by_segment[segment_id],
                "confidence": 0.82,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read precise first timestamp.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
        ):
            calls.append(("vision_read", segment_id, ask_for))
            observed = 11.0 if "light" in ask_for.lower() else 35.0
            return {
                "claim": f"{event_label} is first visible at {observed:.1f} seconds.",
                "confidence": 0.94,
                "event_label": event_label,
                "observed_at_sec": observed,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class NeedMoreBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "missing_evidence": ["timeline should decide"], "citations": []}')
                return BackendResponse(text='{"status": "final", "answer": "fallback", "citations": []}')

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=45.0,
            segments=[
                VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=15.0),
                VideoSegment(segment_id="seg_0002", start_sec=15.0, end_sec=30.0),
                VideoSegment(segment_id="seg_0003", start_sec=30.0, end_sec=45.0),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_caption_pass")
            agent = IterativeVisualAgent(
                backend=NeedMoreBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=8, hard_skill_runtime=True),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(result.rounds[0].program, [])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("skill_recommended", trace)
            self.assertIn("visual_timeline_qa@v1", trace)
            self.assertNotIn("iterative_timeline_temporal_decision", trace)
            self.assertEqual(calls, [])
            self.assertIn("mcq_forced_fallback", trace)

    def test_timeline_ordering_missing_entity_returns_need_more_evidence(self):
        registry = ToolRegistry()

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            return {
                "claim": "Only the light turns on here.",
                "confidence": 0.8,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read precise first timestamp.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = ""):
            return {
                "claim": f"{event_label} is first visible at 10.0 seconds.",
                "confidence": 0.9,
                "event_label": event_label,
                "observed_at_sec": 10.0,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class AbstainBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                return BackendResponse(text='{"answer": "need_more_evidence", "missing_evidence": ["door opens"], "citations": []}')

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_missing")
            agent = IterativeVisualAgent(
                backend=AbstainBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=8, hard_skill_runtime=True, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertLessEqual(workspace.observation_count(tool_name="caption_segment"), 1)
            self.assertEqual(workspace.observation_count(tool_name="vision_read"), 0)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("mcq_forced_fallback", trace)

    def test_timeline_ordering_ignores_negative_caption_echoes(self):
        registry = ToolRegistry()
        calls = []

        @tool(name="caption_segment", description="Caption coarse temporal windows.")
        def caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str = "", nframes: int = 8):
            calls.append(("caption_segment", segment_id))
            return {
                "claim": "The video does not contain any visible evidence of door opens or light turns on.",
                "confidence": 0.8,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="vision_read", description="Read precise first timestamp.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = ""):
            calls.append(("vision_read", segment_id))
            return {
                "claim": "No direct evidence is visible.",
                "confidence": 0.2,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "confidence_signal": "unsupported",
            }

        registry.register(caption_segment)
        registry.register(vision_read)

        class AbstainBackend(VisionLanguageBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                return BackendResponse(text='{"answer": "need_more_evidence", "missing_evidence": ["missing timestamps"], "citations": []}')

        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="timeline_negative_echo")
            agent = IterativeVisualAgent(
                backend=AbstainBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, max_tool_calls_per_round=8, hard_skill_runtime=True, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "Which order is shown?\n"
                    "A. door opens then light turns on\n"
                    "B. light turns on then door opens"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual([call[0] for call in calls].count("caption_segment"), 1)
            self.assertEqual([call[0] for call in calls].count("vision_read"), 0)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("skill_recommended", trace)
            self.assertIn("visual_timeline_qa@v1", trace)
            self.assertNotIn("timeline_ordering_missing_entity", trace)
            self.assertIn("mcq_forced_fallback", trace)

    def test_no_evidence_growth_defers_low_confidence_until_budget(self):
        planner_responses = [
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "first pass"}, "assign": "map1"}]}',
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "second pass"}, "assign": "map2"}]}',
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "third pass"}, "assign": "map3"}]}',
        ]

        class PartialSupportBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "need_more_evidence", "citations": ["obs_0001"], '
                            '"candidate_option_relations": ['
                            '{"option": "A", "relation": "support", "strength": 0.6, "observation_id": "obs_0001"}'
                            '], "missing_evidence": ["need a confirming read"], "confidence": 0.2}'
                        )
                    )
                return super().generate(request)

        registry = ToolRegistry()

        @tool(name="video_ls", description="Cheap navigation that adds no answer evidence.")
        def video_ls(query: str = ""):
            return {"claim": f"navigation only: {query}", "confidence": 1.0}

        registry.register(video_ls)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_evidence_growth")
            observation = workspace.write_observation(
                tool_name="vision_read",
                claim="A red object is visible.",
                confidence=0.8,
                input_artifacts=["/videos/demo.mp4"],
                regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 5.0}],
                raw_output={"grounding_quality": "visually_confirmed"},
            )
            workspace.write_ledger_entry(observation)
            agent = IterativeVisualAgent(
                backend=PartialSupportBackend(planner_responses),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=5,
                    reserve_final_round=False,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(question="Which option is visible?\nA. red object\nB. blue object", video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(result.answer, "A")
            self.assertGreaterEqual(len(result.rounds), 5)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("evidence_table_no_growth", trace)
            self.assertIn("iterative_answer_suggestion", trace)

    def test_no_evidence_growth_appends_visual_read_to_navigation_only_plan(self):
        planner_responses = [
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "first pass"}, "assign": "map1"}]}',
            '{"status": "continue", "program": [{"tool": "search_segments", "args": {"query": "second pass"}, "assign": "map2"}]}',
            '{"status": "continue", "program": [{"tool": "video_ls", "args": {"query": "third pass"}, "assign": "map3"}]}',
        ]

        class AbstainBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["visual read needed"]}'
                    )
                return super().generate(request)

        registry = ToolRegistry()
        calls = []

        @tool(name="video_ls", description="Cheap navigation that adds no answer evidence.")
        def video_ls(query: str = ""):
            calls.append(("video_ls", query))
            return {"claim": f"navigation only: {query}", "confidence": 1.0}

        @tool(name="search_segments", description="Cheap search that adds no answer evidence.")
        def search_segments(query: str, top_k: int = 5):
            calls.append(("search_segments", query))
            return {"claim": f"search only: {query}", "confidence": 1.0}

        @tool(name="vision_read", description="Visual read that creates answer evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append(("vision_read", segment_id, ask_for))
            return {
                "claim": f"{segment_id} visual evidence for {ask_for}",
                "confidence": 0.74,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(video_ls)
        registry.register(search_segments)
        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_growth_visual_fallback")
            agent = IterativeVisualAgent(
                backend=AbstainBackend(planner_responses),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=3,
                    reserve_final_round=False,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(question="Describe what is visible.", video_path="/videos/demo.mp4")

            self.assertEqual([step["tool"] for step in result.rounds[2].program], ["video_ls"])
            self.assertTrue(calls)
            self.assertNotEqual(calls[-1][0], "vision_read")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("force_visual_after_no_evidence_growth", trace)
            self.assertIn("silent_forced_visual_disabled", trace)
            self.assertNotIn("append_visual_followup", trace)

    def test_navigation_only_mcq_round_forces_uninspected_visual_when_no_option_support(self):
        planner_responses = [
            '{"status": "continue", "program": [{"tool": "search_segments", "args": {"query": "first localization"}, "assign": "map1"}]}',
            '{"status": "continue", "program": [{"tool": "search_segments", "args": {"query": "repeat localization"}, "assign": "map2"}]}',
        ]
        registry = ToolRegistry()
        calls = []

        @tool(name="search_segments", description="Cheap search that adds no answer evidence.")
        def search_segments(query: str, top_k: int = 5):
            calls.append(("search_segments", query))
            return {"claim": f"search only: {query}", "confidence": 1.0}

        @tool(name="vision_read", description="Visual read that creates answer evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append(("vision_read", segment_id, start_sec, end_sec))
            return {
                "claim": f"{segment_id} visual evidence",
                "confidence": 0.74,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(search_segments)
        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_sweep")
            agent = IterativeVisualAgent(
                backend=ScriptedPlannerBackend(planner_responses),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=2,
                    reserve_final_round=False,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(
                question="Which option is visible?\nA. red object\nB. blue object",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.rounds[0].program[0]["tool"], "search_segments")
            self.assertEqual([step["tool"] for step in result.rounds[1].program], ["search_segments"])
            self.assertEqual(calls[-1], ("search_segments", "repeat localization"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("force_uninspected_visual_without_option_support", trace)
            self.assertIn("silent_forced_visual_disabled", trace)
            self.assertNotIn("append_visual_followup", trace)

    def test_mcq_full_segment_sweep_hands_off_to_answer_agent_before_budget_end(self):
        class FullSweepAnswerBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "search_segments", "args": {"query": "still searching"}, "assign": "map"}'
                            "]}"
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "A. red object", "rationale": "obs_0003 has the local visual read.", '
                            '"citations": ["obs_0003"], "missing_evidence": [], "confidence": 0.84}'
                        )
                    )
                raise AssertionError(request.task)

        registry = ToolRegistry()
        calls = []

        @tool(name="search_segments", description="Cheap search that adds no answer evidence.")
        def search_segments(query: str, top_k: int = 5):
            calls.append(("search_segments", query))
            return {"claim": f"search only: {query}", "confidence": 1.0}

        @tool(name="vision_read", description="Visual read that creates answer evidence.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append(("vision_read", segment_id))
            return {
                "claim": "A red object is visible in the only segment.",
                "confidence": 0.84,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(search_segments)
        registry.register(vision_read)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=20.0, window_sec=20.0)
        backend = FullSweepAnswerBackend()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="mcq_full_sweep_answer")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=5,
                    reserve_final_round=True,
                    max_repeated_programs=0,
                    answer_probe_rounds_before_final=0,
                ),
            )

            result = agent.run(
                question="Which option is visible?\nA. red object\nB. blue object",
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "low_confidence_final")
            self.assertEqual(
                [call[0] for call in calls],
                ["search_segments", "search_segments", "search_segments", "search_segments"],
            )
            self.assertEqual(
                [request.task for request in backend.requests],
                [
                    "replan",
                    "replan",
                    "answer_from_evidence",
                    "replan",
                    "answer_from_evidence",
                    "replan",
                    "answer_from_evidence",
                    "replan",
                    "answer_from_evidence",
                    "answer_from_evidence",
                    "answer_from_evidence",
                ],
            )
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"source": "all_segments_inspected"', trace)
            self.assertNotIn("iterative_finalization_ready", trace)
            self.assertIn('"source": "evidence_table_no_growth"', trace)
            self.assertIn("silent_forced_visual_disabled", trace)

    def test_model_rewritten_mcq_is_used_for_planner_and_tools_only(self):
        raw_question = (
            "VideoMME multiple-choice question. Answer with exactly one option letter first.\n"
            "Question: What's the main idea of the video?\n"
            "Options:\n"
            "A. The fall of Rome\n"
            "B. Why the Austro-Hungarian Empire was divided\n"
            "C. A battle timeline\n"
            "D. How the Austro-Hungarian Empire rises and falls.\n"
            "Select option A, B, C, or D."
        )
        raw_option_texts = (
            "The fall of Rome",
            "Why the Austro-Hungarian Empire was divided",
            "A battle timeline",
            "How the Austro-Hungarian Empire rises and falls.",
        )
        rewritten = (
            "Describe the overall topic and narrative arc of the video. Identify how the Austro-Hungarian Empire "
            "is covered, including time span, "
            "major stages, and whether it covers origin, growth, stability, decline, collapse, causes, or consequences."
        )
        testcase = self

        class RewriteFlowBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "rewrite_exploration_question":
                    testcase.assertIn("A. The fall of Rome", request.prompt)
                    return BackendResponse(
                        text=(
                            '{"exploration_question":"'
                            + rewritten
                            + '","focus_points":["narrative arc"],"target_entities":["Austro-Hungarian Empire"]}'
                        )
                    )
                if request.task == "replan":
                    testcase.assertIn(rewritten, request.prompt)
                    testcase.assertNotIn("Options:", request.prompt)
                    for label in ("A.", "B.", "C.", "D."):
                        testcase.assertNotIn(label, request.prompt)
                    for option_text in raw_option_texts:
                        testcase.assertNotIn(option_text, request.prompt)
                    testcase.assertNotIn("candidate_options", request.prompt)
                    return BackendResponse(
                        text=(
                            '{"status":"continue","program":[{"tool":"vision_read","args":{"segment_id":"seg_0001",'
                            '"ask_for":"Inspect option B. Why the Austro-Hungarian Empire was divided"},'
                            '"assign":"v1"}]}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    testcase.assertIn("A. The fall of Rome", request.prompt)
                    testcase.assertIn("D. How the Austro-Hungarian Empire rises and falls.", request.prompt)
                    return BackendResponse(
                        text='{"answer":"D. How the Austro-Hungarian Empire rises and falls.","citations":["obs_0001"],"confidence":0.86}'
                    )
                raise AssertionError(request.task)

        registry = ToolRegistry()
        calls = []

        @tool(name="vision_read", description="Read visible facts.")
        def vision_read(
            video_path: str,
            segment_id: str,
            start_sec: float,
            end_sec: float,
            ask_for: str,
            event_label: str = "",
            nframes: int = 8,
        ):
            calls.append({"ask_for": ask_for, "event_label": event_label})
            return {
                "claim": "The segment describes the Austro-Hungarian Empire across formation, stability, decline, and collapse.",
                "confidence": 0.86,
                "input_artifacts": [video_path],
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
                "grounding_quality": "visually_confirmed",
            }

        registry.register(vision_read)
        backend = RewriteFlowBackend()
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="rewrite_mcq_flow")
            agent = IterativeVisualAgent(
                backend=backend,
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(
                    max_rounds=2,
                    reserve_final_round=True,
                    rewrite_mcq_for_exploration=True,
                    disable_global_gist_route=True,
                ),
            )

            result = agent.run(question=raw_question, video_path="/videos/demo.mp4")

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            self.assertEqual(result.question, raw_question)
            self.assertEqual(rewritten, calls[0]["ask_for"])
            self.assertNotIn("option B", calls[0]["ask_for"])
            self.assertNotIn("The fall of Rome", calls[0]["ask_for"])
            self.assertNotIn("Why the Austro-Hungarian Empire was divided", calls[0]["ask_for"])
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("mcq_exploration_question_rewrite", trace)

    def test_option_blind_answer_feedback_strips_option_labels_and_full_text(self):
        raw_question = (
            "Question: What's the main idea?\n"
            "Options:\n"
            "A. The fall of Rome\n"
            "B. Why the Austro-Hungarian Empire was divided\n"
            "C. A battle timeline\n"
            "D. How the Austro-Hungarian Empire rises and falls.\n"
        )

        feedback = _sanitize_option_blind_feedback(
            [
                "Need evidence for option B. Why the Austro-Hungarian Empire was divided, not A. The fall of Rome.",
                "need timestamp evidence",
            ],
            raw_question=raw_question,
        )

        self.assertEqual(feedback[0], "Resolve the remaining evidence gap with factual observations.")
        self.assertEqual(feedback[1], "need timestamp evidence")
        joined = "\n".join(feedback)
        self.assertNotIn("option B", joined)
        self.assertNotIn("A. The fall of Rome", joined)
        self.assertNotIn("Why the Austro-Hungarian Empire was divided", joined)

    def test_repeated_empty_program_finalizes_from_answer_verifier_when_gate_passes(self):
        class ShouldNotAnswerBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    raise AssertionError("structured table should arbitrate without backend")
                return super().generate(request)

        responses = [
            '{"status": "continue", "program": []}',
            '{"status": "continue", "program": []}',
            '{"status": "continue", "program": []}',
        ]
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="repeated_empty_finalizes")
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_scene_coverage_1",
                    "obs_id": "scene_coverage_seg_0001",
                    "tool": "timeline_asr_summary",
                    "supported_option": "D",
                    "claim": "Early transcript covers the rise and formation of the empire.",
                    "confidence": 0.84,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.84}],
                    "time_range": [0.0, 20.0],
                }
            )
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_scene_coverage_2",
                    "obs_id": "scene_coverage_seg_0003",
                    "tool": "timeline_asr_summary",
                    "supported_option": "D",
                    "claim": "Late transcript covers decline, collapse, and the fall of the empire.",
                    "confidence": 0.84,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.84}],
                    "time_range": [40.0, 60.0],
                }
            )
            agent = IterativeVisualAgent(
                backend=ShouldNotAnswerBackend(responses),
                registry=ToolRegistry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=6, reserve_final_round=True, max_repeated_programs=1),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "B. Why the empire was divided\n"
                    "D. How the empire rose and fell"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("source\": \"evidence_table_no_growth", trace)
            self.assertIn("iterative_finalization_ready", trace)

    def test_no_growth_guard_finalizes_from_answer_verifier_before_budget(self):
        class ShouldNotAnswerBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                if request.task == "answer_from_evidence":
                    raise AssertionError("structured table should arbitrate without backend")
                return super().generate(request)

        responses = [
            '{"status": "continue", "program": []}',
            '{"status": "continue", "program": []}',
            '{"status": "continue", "program": []}',
            '{"status": "continue", "program": []}',
        ]
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=60.0, window_sec=20.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="no_growth_verifier_finalizes")
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_scene_coverage_1",
                    "obs_id": "scene_coverage_seg_0001",
                    "tool": "timeline_asr_summary",
                    "supported_option": "D",
                    "claim": "Early transcript covers the rise and formation of the empire.",
                    "confidence": 0.84,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.84}],
                    "time_range": [0.0, 20.0],
                }
            )
            workspace.write_evidence_row(
                {
                    "evidence_id": "ev_scene_coverage_2",
                    "obs_id": "scene_coverage_seg_0003",
                    "tool": "timeline_asr_summary",
                    "supported_option": "D",
                    "claim": "Late transcript covers decline, collapse, and the fall of the empire.",
                    "confidence": 0.84,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.84}],
                    "time_range": [40.0, 60.0],
                }
            )
            agent = IterativeVisualAgent(
                backend=ShouldNotAnswerBackend(responses),
                registry=ToolRegistry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=6, reserve_final_round=False, max_repeated_programs=0),
            )

            result = agent.run(
                question=(
                    "What is the video mainly about?\n"
                    "B. Why the empire was divided\n"
                    "D. How the empire rose and fell"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            self.assertTrue(result.answer.startswith("D"))
            self.assertLess(len(result.rounds), 6)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("source\": \"evidence_table_no_growth", trace)
            self.assertIn("iterative_finalization_ready", trace)

    def test_navigation_only_no_growth_forces_visual_read_on_requested_segment(self):
        class NavThenAnswerBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "vision_read", "args": {"segment_id": "seg_0001", "ask_for": "initial read"}, '
                                '"assign": "v1"}]}'
                            )
                        )
                    return BackendResponse(
                        text=(
                            '{"status": "continue", "program": ['
                            '{"tool": "expand_window", "args": {"segment_id": "seg_0001", "before_sec": 30, "after_sec": 30}, '
                            '"assign": "nav"}]}'
                        )
                    )
                if request.task == "answer_from_evidence":
                    return BackendResponse(text='{"answer": "need_more_evidence", "citations": [], "missing_evidence": ["more visual facts"]}')
                return BackendResponse(text="unexpected")

        registry = ToolRegistry()
        calls = []

        @tool(name="vision_read", description="Read a segment.")
        def vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, **kwargs):
            calls.append(("vision_read", segment_id, ask_for))
            return {
                "claim": f"visual read {len(calls)}",
                "confidence": 0.8,
                "regions": [{"segment_id": segment_id, "start_sec": start_sec, "end_sec": end_sec}],
            }

        @tool(name="expand_window", description="Expand a segment.")
        def expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0):
            calls.append(("expand_window", segment_id, ""))
            return {"claim": f"expanded {segment_id}", "confidence": 1.0}

        registry.register(vision_read)
        registry.register(expand_window)
        scene_index = fixed_window_scene_index(video_path="/videos/demo.mp4", duration_sec=30.0, window_sec=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="nav_to_visual")
            agent = IterativeVisualAgent(
                backend=NavThenAnswerBackend(),
                registry=registry,
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=4, reserve_final_round=False, max_repeated_programs=0),
            )

            agent.run(question="What happens?", video_path="/videos/demo.mp4")

            self.assertNotIn(("vision_read", "seg_0001", "What happens?"), calls)
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("force_visual_from_navigation_no_growth", trace)
            self.assertIn("silent_forced_visual_disabled", trace)

    def test_answer_agent_final_trace_includes_scene_index_citation_provenance(self):
        class ProvenanceBackend(ScriptedPlannerBackend):
            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "answer_from_evidence":
                    return BackendResponse(
                        text=(
                            '{"answer": "D. Rome rose and fell", "citations": ["scene_order_seg_0001"], '
                            '"confidence": 0.86, "candidate_option_relations": ['
                            '{"option": "D", "relation": "support", "strength": 0.86, '
                            '"observation_id": "scene_order_seg_0001"}]}'
                        )
                    )
                return super().generate(request)

        scene_index = SceneIndex(
            video_path="/videos/demo.mp4",
            duration_sec=30.0,
            segments=[
                VideoSegment(
                    segment_id="seg_0001",
                    source_segment_id="dual_seg_0001",
                    start_sec=0.0,
                    end_sec=30.0,
                    asr_summary="Rome rose and later fell.",
                    visual_caption="Historical maps and narration.",
                    visual_caption_source="caption_scene_segment:vl-mini",
                    raw_asr_ref="cue-1,cue-2",
                    citation_provenance={"asr": "subtitle", "visual": "video"},
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="final_citation_provenance")
            workspace.write_evidence_row(
                {
                    "obs_id": "scene_order_seg_0001",
                    "tool": "timeline_asr_summary",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 30.0],
                    "supported_option": "D",
                    "claim": "Indexed segment supports the rise and fall sequence.",
                    "confidence": 0.86,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [{"option": "D", "relation": "support", "strength": 0.86}],
                }
            )
            agent = IterativeVisualAgent(
                backend=ProvenanceBackend(
                    ['{"status": "final", "answer": "D. Rome rose and fell", "citations": ["scene_order_seg_0001"]}']
                ),
                registry=ToolRegistry(),
                workspace=workspace,
                scene_index=scene_index,
                budget=AgentBudget(max_rounds=1, reserve_final_round=False),
            )

            result = agent.run(
                question=(
                    "What sequence is described?\n"
                    "Options:\n"
                    "A. Rome only rises\n"
                    "D. Rome rose and fell\n"
                ),
                video_path="/videos/demo.mp4",
            )

            self.assertEqual(result.status, "final")
            trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"citation_provenance"', trace)
            self.assertIn('"source_segment_id": "dual_seg_0001"', trace)
            self.assertIn('"raw_asr_ref": "cue-1,cue-2"', trace)
            self.assertIn('"visual_caption_source": "caption_scene_segment:vl-mini"', trace)

    def test_segment_vlm_tools_share_backend_and_pass_temporal_metadata(self):
        class SegmentToolBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                return BackendResponse(text="The segment shows a runway and aircraft.")

        backend = SegmentToolBackend()
        registry = build_segment_vlm_registry(backend)

        result = registry.execute(
            "caption_segment",
            {
                "video_path": "/videos/demo.mp4",
                "segment_id": "seg_0002",
                "start_sec": 10.0,
                "end_sec": 20.0,
                "question": "What is visible?",
                "nframes": 12,
            },
        )

        self.assertEqual(result["claim"], "The segment shows a runway and aircraft.")
        self.assertEqual(backend.requests[0].task, "caption_segment")
        self.assertEqual(backend.requests[0].media_path, "/videos/demo.mp4")
        self.assertEqual(backend.requests[0].metadata["segment_id"], "seg_0002")
        self.assertEqual(backend.requests[0].metadata["start_sec"], 10.0)
        self.assertEqual(backend.requests[0].metadata["end_sec"], 20.0)
        self.assertEqual(backend.requests[0].metadata["nframes"], 64)
        self.assertEqual(backend.requests[0].metadata["max_pixels"], 151200)
        self.assertEqual(backend.requests[0].metadata["question"], "What is visible? Do not choose an option.")

    def test_segment_vlm_tools_can_extract_physical_clip_before_backend_call(self):
        class SegmentToolBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                return BackendResponse(text="The extracted clip shows a museum exhibit.")

        extracted = []

        def fake_clip_extractor(video_path, output_path, start_sec, end_sec):
            extracted.append((video_path, output_path, start_sec, end_sec))
            Path(output_path).write_text("fake clip", encoding="utf-8")
            return output_path

        backend = SegmentToolBackend()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="clip_tools")
            registry = build_segment_vlm_registry(
                backend,
                workspace=workspace,
                extract_clips=True,
                clip_extractor=fake_clip_extractor,
            )

            result = registry.execute(
                "caption_segment",
                {
                    "video_path": "/videos/demo.mp4",
                    "segment_id": "seg_0002",
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                    "question": "What is visible?",
                    "nframes": 12,
                },
            )

            self.assertEqual(len(extracted), 1)
            self.assertEqual(extracted[0][0], "/videos/demo.mp4")
            self.assertIn("seg_0002_10000_20000.mp4", extracted[0][1])
            self.assertEqual(backend.requests[0].media_path, extracted[0][1])
            self.assertEqual(result["input_artifacts"], [extracted[0][1]])
            self.assertEqual(result["regions"][0]["source_video_path"], "/videos/demo.mp4")
            self.assertEqual(result["regions"][0]["clip_path"], extracted[0][1])

    def test_iterative_smoke_runner_uses_shared_backend_and_fixed_window_index(self):
        class SharedBackend(VisionLanguageBackend):
            def __init__(self):
                self.requests = []
                self.replan_calls = 0

            def generate(self, request: BackendRequest) -> BackendResponse:
                self.requests.append(request)
                if request.task == "replan":
                    self.replan_calls += 1
                    if self.replan_calls == 1:
                        return BackendResponse(
                            text=(
                                '{"status": "continue", "program": ['
                                '{"tool": "caption_segment", "args": {"segment_id": "seg_0002", "question": "Inspect"}, "assign": "s2"}'
                                "]}"
                            )
                        )
                    return BackendResponse(text='{"status": "final", "answer": "Segment two is relevant.", "citations": ["obs_0001"]}')
                if request.task == "caption_segment":
                    return BackendResponse(text="Segment two contains the relevant evidence.")
                return BackendResponse(text="")

        backend = SharedBackend()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_iterative_smoke(
                base_dir=Path(tmp),
                backend=backend,
                media_path="/videos/demo.mp4",
                question="What matters?",
                duration_sec=60.0,
                window_sec=30.0,
                run_id="iterative_smoke",
            )

            self.assertEqual(result.answer, "Segment two is relevant.")
            self.assertEqual([request.task for request in backend.requests], ["replan", "caption_segment", "replan"])
            self.assertEqual(backend.requests[1].metadata["start_sec"], 30.0)


if __name__ == "__main__":
    unittest.main()
