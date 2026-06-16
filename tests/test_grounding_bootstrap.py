from __future__ import annotations

import json
from pathlib import Path

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget, IterativeVisualAgent
from visual_coding_agent_harness.agents.grounding.compiler import compile_fallback_plan, compile_grounding_plan
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.backends.routed import RoutedBackend
from visual_coding_agent_harness.registry import ToolRegistry
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class RecordingBackend(VisionLanguageBackend):
    def __init__(self, responses: dict[str, list[str]] | None = None) -> None:
        self.responses = {task: list(values) for task, values in (responses or {}).items()}
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        queue = self.responses.get(request.task, [])
        if queue:
            return BackendResponse(text=queue.pop(0))
        return BackendResponse(text='{"status": "continue", "program": []}')


def _valid_grounding_plan_json() -> str:
    return json.dumps(
        {
            "route": "temporal_order",
            "recommended_skill": "narration_timeline_qa",
            "central_subjects": ["Subject Alpha"],
            "subjects": [
                {
                    "subject_key": "subject_main",
                    "canonical_name": "Subject Alpha",
                    "aliases": [],
                }
            ],
            "targets": [
                {
                    "target_key": "event_one",
                    "canonical_claim": "Subject Alpha first event",
                    "subject_key": "subject_main",
                    "claim_kind": "narrated_fact",
                    "claim_modality": "asr",
                    "aliases": ["first event", "Subject Alpha"],
                    "search_queries": ["first event transcript"],
                    "polarity": "affirmed",
                },
                {
                    "target_key": "event_two",
                    "canonical_claim": "Subject Alpha second event",
                    "subject_key": "subject_main",
                    "claim_kind": "narrated_fact",
                    "claim_modality": "asr",
                    "aliases": ["second event"],
                    "search_queries": ["second event transcript"],
                    "polarity": "affirmed",
                },
            ],
            "relations": [
                {
                    "relation_key": "one_before_two",
                    "kind": "before",
                    "source_target_key": "event_one",
                    "destination_target_key": "event_two",
                }
            ],
            "options": [
                {
                    "option_id": "A",
                    "required_target_keys": [],
                    "ordered_target_keys": ["event_one", "event_two"],
                    "required_relation_keys": ["one_before_two"],
                    "raw_option_text": "first then second",
                    "option_kind": "sequence",
                },
                {
                    "option_id": "B",
                    "required_target_keys": [],
                    "ordered_target_keys": ["event_two", "event_one"],
                    "required_relation_keys": [],
                    "raw_option_text": "second then first",
                    "option_kind": "sequence",
                },
            ],
            "acceptable_evidence_sources": ["asr"],
            "confidence": 0.8,
            "unresolved_ambiguities": [],
        }
    )


def _agent(tmp_path: Path, backend: VisionLanguageBackend, run_id: str) -> IterativeVisualAgent:
    return IterativeVisualAgent(
        backend=backend,
        registry=ToolRegistry(),
        workspace=EvidenceWorkspace.create(tmp_path, run_id),
        scene_index=SceneIndex(
            video_path="/videos/generic.mp4",
            duration_sec=60.0,
            segments=[VideoSegment(segment_id="seg_0001", start_sec=0.0, end_sec=60.0)],
        ),
        budget=AgentBudget(
            max_rounds=1,
            reserve_final_round=False,
            hard_skill_runtime=True,
            planner_owned_grounding=True,
        ),
    )


def test_routed_backend_sends_ground_question_to_text_backend_when_available() -> None:
    text_backend = RecordingBackend({"ground_question": ['{"route": "temporal_order"}']})
    vl_backend = RecordingBackend()
    backend = RoutedBackend(text_backend=text_backend, vl_backend=vl_backend)

    response = backend.generate(BackendRequest(task="ground_question", prompt="ground this"))

    assert response.raw["route_backend"] == "text"
    assert [request.task for request in text_backend.requests] == ["ground_question"]
    assert vl_backend.requests == []


def test_fallback_grounding_plan_validates_for_mcq_options() -> None:
    question = "Which sequence is described?\nA. first then second\nB. second then first"

    plan = compile_fallback_plan(question, ("A. first then second", "B. second then first"), "temporal_order")
    compiled = compile_grounding_plan(
        plan,
        raw_options={"A": "first then second", "B": "second then first"},
        skill_ids=("visual_timeline_qa", "narration_timeline_qa", "main_idea", "grounded_factual_qa"),
    )

    assert compiled.route == "temporal_order"
    assert compiled.recommended_skill_id in {"visual_timeline_qa", "narration_timeline_qa"}
    assert set(compiled.registry.options_by_id) == {"A", "B"}
    assert compiled.registry.option_for("A").target_sequence


def test_fallback_grounding_plan_builds_item_targets_for_quoted_order_options() -> None:
    question = (
        "In what order does the narration list the artworks?\n"
        'A. "Aeneas", "David", "Persephone", "Apollo"\n'
        'B. "David", "Aeneas", "Persephone", "Apollo"\n'
        'C. "Aeneas", "Persephone", "David", "Apollo"\n'
        'D. "Aeneas", "David", "Apollo", "Persephone"'
    )
    options = (
        'A. "Aeneas", "David", "Persephone", "Apollo"',
        'B. "David", "Aeneas", "Persephone", "Apollo"',
        'C. "Aeneas", "Persephone", "David", "Apollo"',
        'D. "Aeneas", "David", "Apollo", "Persephone"',
    )

    plan = compile_fallback_plan(question, options, "temporal_order")
    compiled = compile_grounding_plan(
        plan,
        raw_options={
            "A": '"Aeneas", "David", "Persephone", "Apollo"',
            "B": '"David", "Aeneas", "Persephone", "Apollo"',
            "C": '"Aeneas", "Persephone", "David", "Apollo"',
            "D": '"Aeneas", "David", "Apollo", "Persephone"',
        },
        skill_ids=("visual_timeline_qa", "narration_timeline_qa", "main_idea", "grounded_factual_qa"),
    )

    target_texts = [target.canonical_text for target in compiled.registry.targets_by_id.values()]
    assert target_texts == ["Aeneas", "David", "Persephone", "Apollo"]
    assert compiled.registry.option_for("A").target_sequence == ("T1", "T2", "T3", "T4")
    assert compiled.registry.option_for("B").target_sequence == ("T2", "T1", "T3", "T4")


def test_fallback_grounding_plan_uses_narration_skill_for_life_journey() -> None:
    question = "How was his life journey according to the video?"
    options = (
        "A. Born with humble background and lived in seclusion.",
        "B. Born with humble background, entered upper class, then lived in seclusion.",
    )

    plan = compile_fallback_plan(question, options, "temporal_order")
    compiled = compile_grounding_plan(
        plan,
        raw_options={
            "A": "Born with humble background and lived in seclusion.",
            "B": "Born with humble background, entered upper class, then lived in seclusion.",
        },
        skill_ids=("visual_timeline_qa", "narration_timeline_qa", "main_idea", "grounded_factual_qa"),
    )

    assert compiled.recommended_skill_id == "narration_timeline_qa"


def test_bootstrap_invalid_grounding_falls_back_and_enters_planner_loop(tmp_path: Path) -> None:
    backend = RecordingBackend(
        {
            "ground_question": ["not json", "nope", "still not json " + "x" * 800],
            "replan": ['{"status": "continue", "program": []}'],
        }
    )
    agent = _agent(tmp_path, backend, "bootstrap_fallback")

    result = agent.run(
        question="Which sequence is described?\nA. first then second\nB. second then first",
        video_path="/videos/generic.mp4",
    )

    assert result.status != "grounding_bootstrap_failed"
    assert [request.task for request in backend.requests if request.task == "ground_question"] == [
        "ground_question",
        "ground_question",
        "ground_question",
    ]
    assert any(request.task == "replan" for request in backend.requests)
    trace = (agent.workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert '"type": "grounding_fallback_compiled"' in trace
    assert '"reason": "grounding_parse_failed"' in trace
    assert '"raw_text_chars": 815' in trace
    assert "still not json" in trace
    assert "x" * 700 not in trace


def test_bootstrap_fallback_compile_failure_returns_failure_without_instance_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_compile(*args, **kwargs):
        raise RuntimeError("fallback unavailable")

    monkeypatch.setattr(
        "visual_coding_agent_harness.agents.iterative_agent.compile_fallback_plan",
        fail_compile,
    )
    backend = RecordingBackend(
        {
            "ground_question": ["not json", "nope", "still not json"],
            "replan": ['{"status": "continue", "program": []}'],
        }
    )
    agent = _agent(tmp_path, backend, "bootstrap_failure")

    result = agent.run(
        question="Which sequence is described?\nA. first then second\nB. second then first",
        video_path="/videos/generic.mp4",
    )

    assert result.status == "grounding_bootstrap_failed"
    assert result.answer == "grounding_bootstrap_failed"
    assert not hasattr(agent, "_grounding_bootstrap_failure")
    assert [request.task for request in backend.requests if request.task == "ground_question"] == [
        "ground_question",
        "ground_question",
        "ground_question",
    ]
    assert not any(request.task == "replan" for request in backend.requests)
    trace = (agent.workspace.root / "trace.jsonl").read_text(encoding="utf-8")
    assert '"type": "grounding_bootstrap_failed"' in trace
    assert '"type": "grounding_bootstrap_diagnostics"' in trace
    assert "fallback unavailable" in trace


def test_bootstrap_success_prompt_exposes_target_refs_without_empty_registry_line(tmp_path: Path) -> None:
    backend = RecordingBackend(
        {
            "ground_question": [_valid_grounding_plan_json()],
            "replan": ['{"status": "continue", "program": []}'],
        }
    )
    agent = _agent(tmp_path, backend, "bootstrap_success")

    agent.run(
        question="Which sequence is described?\nA. first then second\nB. second then first",
        video_path="/videos/generic.mp4",
    )

    prompt = next(request.prompt for request in backend.requests if request.task == "replan")
    assert "Registered target_refs:" in prompt
    assert "target_coverage(targets: list = [], target_refs: list = []" in prompt
    assert "read_segment_detail(segment_id: str, targets: list = [], target_refs: list = []" in prompt
    assert "No target_refs are registered for this run" not in prompt
    assert "No target_refs are registered in this run" not in prompt
