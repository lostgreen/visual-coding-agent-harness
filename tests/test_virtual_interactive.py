from __future__ import annotations

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from PIL import Image

from vcah.investigator import _choose_window_from_segment_packet
from vcah.multiround import InvestigationTask
from vcah.types import EvidenceRecord, Frame
from vcah import virtual_video
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)


def _load_tool_module(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_viewer = _load_tool_module("virtual_trace_viewer", "build_virtual_trace_viewer.py")
_interactive = _load_tool_module("virtual_interactive_runner", "run_virtual_videomme_interactive.py")
AssetBundler = _viewer.AssetBundler
_render_case = _viewer._render_case
GeminiInvestigator = _interactive.GeminiInvestigator
GeminiReasoner = _interactive.GeminiReasoner
OpenAICompatibleVisionClient = _interactive.OpenAICompatibleVisionClient
_select_window_with_model = _interactive._select_window_with_model
_select_detail_window = _interactive._select_detail_window
_load_case_group = _interactive._load_case_group
_event_evidence_prompt = _interactive._event_evidence_prompt
_load_existing_case_summary = _interactive._load_existing_case_summary
_run_case_batch = _interactive._run_case_batch
_should_audit_answer = _interactive._should_audit_answer
_requires_independent_claim_verification = _interactive._requires_independent_claim_verification
_matching_claim_assessment = _interactive._matching_claim_assessment
_followup_prompt = _interactive._followup_prompt
_compile_option_claim_contract = _interactive._compile_option_claim_contract
_forced_option_support_dashboard = _interactive._forced_option_support_dashboard
_calibrate_forced_decision = _interactive._calibrate_forced_decision
load_role_clients = _interactive.load_role_clients
run_case = _interactive.run_case


def test_reasoner_prompt_requires_evidence_dynamics_sampling_plan() -> None:
    prompt = _interactive._investigate_prompt(
        {
            "question": "What happens?",
            "options": {"A": "One event", "B": "Another event"},
            "query_contract": {},
            "query_requirements": {},
            "workspace_duration_sec": 120.0,
            "temporal_navigation": {},
            "workspace_overview": {"segment_overviews": []},
        }
    )

    assert "sampling_floor_fps" in prompt
    assert "temporal_resolution_rationale" in prompt
    assert "Choose from evidence dynamics, not the question category" in prompt


def test_forced_count_choice_penalizes_strong_assertions_without_positive_evidence() -> None:
    kwargs = {
        "question": "How many green cylinders are visible?",
        "options": {"A": "One green cylinder", "E": "Four green cylinders"},
        "query_contract": {"quantifier": "distinct_count"},
    }
    wrong_window = ({
        "observation_polarity": "positive",
        "target_presence": {"status": "absent"},
        "summary": "Unrelated objects are visible in this wrong window.",
        "entities": [],
    },)
    dashboard = _forced_option_support_dashboard(kwargs, wrong_window)
    calibrated = _calibrate_forced_decision(
        kwargs,
        wrong_window,
        _interactive.ReasonerDecision(action="answer", answer="E. Four green cylinders"),
    )

    assert dashboard["policy"] == "canonical_option_verdict_table"
    assert dashboard["recommended_option"] == "A"
    assert dashboard["authoritative"] is False
    assert calibrated.answer == "E. Four green cylinders"


def test_forced_count_choice_uses_countable_witnesses() -> None:
    kwargs = {
        "question": "How many green cylinders are visible?",
        "options": {"A": "One", "B": "Two", "E": "Four"},
        "query_contract": {"quantifier": "distinct_count"},
        "completion_status": {
            "option_verdict_table": {
                "best_option": "B",
                "audit_status": "complete",
                "option_verdicts": {
                    "A": {"status": "contradicted", "reason": "canonical entity count is 2"},
                    "B": {"status": "supported", "reason": "canonical entity count is 2"},
                    "E": {"status": "contradicted", "reason": "canonical entity count is 2"},
                },
            },
        },
    }
    evidence = ({
        "observation_polarity": "positive",
        "entities": [
            {"entity_observation_id": "obs:1", "countable": True, "supports_question_relation": True},
            {"entity_observation_id": "obs:2", "countable": True, "supports_question_relation": True},
        ],
    },)

    dashboard = _forced_option_support_dashboard(kwargs, evidence)

    assert dashboard["policy"] == "canonical_option_verdict_table"
    assert dashboard["recommended_option"] == "B"


def test_forced_total_count_choice_uses_canonical_event_candidates() -> None:
    kwargs = {
        "question": "How many times was the rider overtaken?",
        "options": {"A": "One time", "B": "Two times", "C": "Three times"},
        "query_contract": {"quantifier": "total_count"},
        "completion_status": {
            "canonical_fact_snapshot": {"confirmed_events": [{"candidate_id": "event_1"}, {"candidate_id": "event_2"}]},
            "option_verdict_table": {
                "best_option": "B",
                "audit_status": "complete",
                "option_verdicts": {
                    "A": {"status": "contradicted", "reason": "canonical count is 2"},
                    "B": {"status": "supported", "reason": "canonical count is 2"},
                    "C": {"status": "contradicted", "reason": "canonical count is 2"},
                },
            },
        },
    }
    evidence = tuple(
        {
            "evidence_kind": "event_candidate",
            "events": [{"candidate_id": f"event_{index}", "supports_question_event": True}],
        }
        for index in range(2)
    )

    dashboard = _forced_option_support_dashboard(kwargs, evidence)

    assert dashboard["policy"] == "canonical_option_verdict_table"
    assert dashboard["recommended_option"] == "B"
    assert dashboard["authoritative"] is True
    assert dashboard["fact_source"] == "completion_status.canonical_fact_snapshot"


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{start_sec:.3f}.jpg"
    Image.new("RGB", (32, 18), color=(30, 90, 180)).save(path)
    return (Frame(frame_id=path.stem, time_sec=float(start_sec), path=str(path)),)


def test_load_case_group_preserves_order_and_default_construction(tmp_path: Path) -> None:
    path = tmp_path / "group.json"
    path.write_text(
        json.dumps(
            {
                "group_id": "diverse-v1",
                "construction": "source_only",
                "cases": [
                    {"case_id": "606-3", "capability": "entity_count"},
                    {"case_id": "769-1", "capability": "ocr"},
                ],
            }
        ),
        encoding="utf-8",
    )

    group = _load_case_group(path)

    assert group["group_id"] == "diverse-v1"
    assert group["construction"] == "source_only"
    assert group["case_ids"] == ("606-3", "769-1")


def test_reasoner_task_normalization_keeps_enumeration_goals() -> None:
    tasks = _interactive._normalize_reasoner_tasks(
        (
            {
                "goal": "Enumerate every audition shown in this segment.",
                "segment_id": "seg_0020",
                "modality_hint": ["visual", "asr"],
                "inspection_mode": "enumerate_events",
                "sampling_floor_fps": 1.0,
                "temporal_resolution_rationale": "Each appearance should remain visible for roughly two seconds.",
            },
        ),
        round_id=2,
    )

    assert len(tasks) == 1
    assert tasks[0]["segment_id"] == "seg_0020"
    assert tasks[0]["goal"].startswith("Enumerate")
    assert tasks[0]["inspection_mode"] == "enumerate_events"
    assert tasks[0]["sampling_floor_fps"] == 1.0
    assert tasks[0]["temporal_resolution_rationale"].startswith("Each appearance")


def test_investigator_prompt_defines_object_relative_spatial_reference_frame() -> None:
    prompt = _interactive._resolution_prompt(
        SimpleNamespace(conditions=()),
        question="Which direction is red facing in relation to green?",
    )

    assert "reference_frame=object_egocentric" in prompt
    assert "relation_type=relative_facing" in prompt
    assert "Do not substitute the subject's relative position" in prompt
    assert "Viewer-relative or subject-egocentric facts are auxiliary only" in prompt
    assert "compare every answer-option score pair" in prompt


def test_run_case_batch_executes_cases_concurrently_and_preserves_order() -> None:
    barrier = threading.Barrier(3)

    def run_one(case_id: str) -> Mapping[str, Any]:
        barrier.wait(timeout=2.0)
        return {"case_id": case_id}

    rows = _run_case_batch(("case-c", "case-a", "case-b"), run_one, workers=3)

    assert [row["case_id"] for row in rows] == ["case-c", "case-a", "case-b"]


def test_load_existing_case_summary_marks_resumed_case(tmp_path: Path) -> None:
    workspace = tmp_path / "case-1"
    workspace.mkdir()
    (workspace / "run_summary.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "answer": "B",
                "citations": ["ev_1"],
                "correct": True,
                "verified": False,
                "verification_reason": "answer_audit_insufficient",
                "rounds": 4,
                "accepted_investigations": 8,
            }
        ),
        encoding="utf-8",
    )

    row = _load_existing_case_summary(workspace)

    assert row is not None
    assert row["case_id"] == "case-1"
    assert row["skipped_completed"] is True
    assert row["workspace"] == str(workspace)


def test_load_role_clients_uses_distinct_configs(tmp_path: Path) -> None:
    reasoner_config = tmp_path / "reasoner.yaml"
    investigator_config = tmp_path / "investigator.yaml"
    reasoner_config.write_text(
        "planner_api:\n  base: https://reasoner.invalid/v1\n  model: gpt-5.5\n  api_key: reasoner-key\n",
        encoding="utf-8",
    )
    investigator_config.write_text(
        "planner_api:\n  base: https://investigator.invalid/v1\n  model: gemini-2.5-pro\n  api_key: investigator-key\n",
        encoding="utf-8",
    )

    reasoner, investigator = load_role_clients(
        shared_config=None,
        reasoner_config=reasoner_config,
        investigator_config=investigator_config,
    )

    assert reasoner.model == "gpt-5.5"
    assert investigator.model == "gemini-2.5-pro"
    assert reasoner.base != investigator.base


def test_gateway_client_uses_kuaishou_role_headers() -> None:
    client = OpenAICompatibleVisionClient(
        {
            "base": "https://gateway.invalid/v1",
            "model": "gpt-5.5",
            "api_key": "api-secret",
            "type": "gemini_gateway",
            "user_key": "user-secret",
            "biz_scene": "video-agent",
        }
    )

    headers = client._headers()

    assert headers["x-api-key"] == "api-secret"
    assert headers["x-ks-user-key"] == "user-secret"
    assert headers["x-ks-llm-model"] == "gpt-5.5"
    assert headers["x-ks-biz-scene"] == "video-agent"
    assert "Authorization" not in headers


def test_gpt5_client_uses_completion_token_parameter(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200
        text = ""
        headers: dict[str, str] = {}

        @staticmethod
        def json() -> Mapping[str, Any]:
            return {"choices": [{"message": {"content": "{}"}}]}

    def post(*args: Any, **kwargs: Any) -> Response:
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(_interactive.requests, "post", post)
    client = OpenAICompatibleVisionClient(
        {"base": "https://gateway.invalid/v1", "model": "gpt-5.5", "api_key": "secret"}
    )

    client.chat("plan", max_tokens=1400)

    assert captured["json"]["max_completion_tokens"] == 1400
    assert "max_tokens" not in captured["json"]
    assert "temperature" not in captured["json"]


def test_gpt5_client_records_reasoning_token_exhaustion(monkeypatch) -> None:
    class Response:
        status_code = 200
        text = ""
        headers: dict[str, str] = {}

        @staticmethod
        def json() -> Mapping[str, Any]:
            return {
                "choices": [{"finish_reason": "length", "message": {"content": None}}],
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 600,
                    "completion_tokens_details": {"reasoning_tokens": 600},
                },
            }

    monkeypatch.setattr(_interactive.requests, "post", lambda *args, **kwargs: Response())
    client = OpenAICompatibleVisionClient(
        {"base": "https://gateway.invalid/v1", "model": "gpt-5.5", "api_key": "secret"}
    )

    assert client.chat("Return JSON.", max_tokens=600) == ""
    assert client.last_response_metadata == {
        "finish_reason": "length",
        "prompt_tokens": 1200,
        "completion_tokens": 600,
        "reasoning_tokens": 600,
        "content_chars": 0,
        "requested_completion_tokens": 600,
    }


def test_run_case_assigns_text_only_reasoner_and_multimodal_investigator(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    reasoner_api = SimpleNamespace(model="gpt-5.5")
    investigator_api = SimpleNamespace(model="gemini-2.5-pro")
    captured: dict[str, Any] = {}
    sentinel = object()

    class FakeReasoner:
        def __init__(self, api: Any, *, trace_path: Path, allow_visual_input: bool) -> None:
            captured["reasoner"] = (api, trace_path, allow_visual_input)

    class FakeInvestigator:
        def __init__(self, source_workspace: Any, *, api: Any, trace_path: Path) -> None:
            captured["investigator"] = (source_workspace, api, trace_path)

    class FakeDriver:
        def __init__(self, **kwargs: Any) -> None:
            captured["driver"] = kwargs

        def run(self, source_workspace: Any) -> object:
            (source_workspace.root_dir / "run_summary.json").write_text("{}", encoding="utf-8")
            return sentinel

    monkeypatch.setattr(_interactive, "ReasonerAgent", FakeReasoner)
    monkeypatch.setattr(_interactive, "GeminiInvestigator", FakeInvestigator)
    monkeypatch.setattr(_interactive, "VirtualVideoMultiRoundDriver", FakeDriver)

    result = run_case(
        workspace,
        reasoner_api=reasoner_api,
        investigator_api=investigator_api,
        max_rounds=4,
        max_investigations=20,
    )

    assert result is sentinel
    assert captured["reasoner"][0] is reasoner_api
    assert captured["reasoner"][2] is False
    assert captured["investigator"][1] is investigator_api
    summary = json.loads((workspace.root_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["models"] == {"reasoner": "gpt-5.5", "investigator": "gemini-2.5-pro"}


def test_vision_client_retries_transient_http_errors_with_exponential_backoff(monkeypatch) -> None:
    class Response:
        def __init__(self, status_code: int, content: str = "") -> None:
            self.status_code = status_code
            self.text = content
            self.headers: dict[str, str] = {}

        def json(self) -> Mapping[str, Any]:
            return {"choices": [{"message": {"content": self.text}}]}

    responses = iter((Response(429, "rate limited"), Response(503, "busy"), Response(200, "ok")))
    calls = []

    def post(*args: Any, **kwargs: Any) -> Response:
        calls.append((args, kwargs))
        return next(responses)

    sleeps: list[float] = []
    monkeypatch.setattr(_interactive.requests, "post", post)
    monkeypatch.setattr(_interactive.time, "sleep", sleeps.append)
    client = OpenAICompatibleVisionClient(
        {
            "base": "https://example.invalid/v1",
            "model": "test-model",
            "api_key": "secret",
            "timeout": 10,
            "max_retries": 3,
            "retry_base_sec": 0.25,
            "retry_max_sec": 2.0,
            "retry_jitter": 0.0,
        }
    )

    assert client.chat("hello") == "ok"
    assert len(calls) == 3
    assert sleeps == [0.25, 0.5]


def test_event_detail_prompt_accepts_prior_adjacent_events(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    prompt = _event_evidence_prompt(
        workspace,
        InvestigationTask(query_id="q", goal="Count title cards."),
        {"segment_id": "seg_0001", "virtual_time_range": [0.0, 120.0], "beats": []},
        {
            "virtual_time_range": [60.0, 120.0],
            "sampling": {"fps": 2.0},
            "asr_cues": [],
            "source_lineage": [],
        },
        preview={"summary": "A title card may continue."},
        prior_events=(
            {
                "event_key": "opening title card",
                "description": "The title card begins in the prior beat.",
                "end_sec": 60.0,
                "continues_to_next": True,
            },
        ),
    )

    assert "opening title card" in prompt
    assert "Prior adjacent-window ending events" in prompt
    assert "introduction, performance, judging, and result" in prompt
    assert "one exact group-based event_key" in prompt


def test_general_investigator_prompt_exposes_options_only_as_observation_targets(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    prompt = _interactive._preview_prompt(
        workspace,
        InvestigationTask(query_id="q", goal="Inspect the board.", segment_id="seg_0001"),
        {"segment_id": "seg_0001", "virtual_time_range": [0.0, 180.0], "beats": []},
        {
            "virtual_time_range": [40.0, 60.0],
            "sampling": {"fps": 0.5},
            "asr_cues": [],
            "source_lineage": [],
        },
    )

    assert "Candidate option predicates are observation targets" in prompt
    assert '"A": "7"' in prompt
    assert '"B": "9"' in prompt
    assert "not an invitation to choose an answer" in prompt


def test_summary_observation_semantics_separates_narration_from_visual_example() -> None:
    instruction = _interactive._summary_observation_semantics("What title best summarizes this video?")

    assert "Narrative thesis" in instruction
    assert "Segment role" in instruction
    assert "Visual example" in instruction
    assert "another language" in instruction
    assert _interactive._summary_observation_semantics("What number is on the board?") == ""


def test_gemini_reasoner_can_request_global_lexical_asr_navigation(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {
                "action": "investigate",
                "primary_gap": {
                    "gap_id": "gap_cause",
                    "description": "The earlier event that caused the identified injury.",
                    "success_conditions": ["locate a timestamped causal event"],
                },
                "tasks": [
                    {
                        "query_id": "search_cause",
                        "goal": "Locate dialogue around the earlier cause.",
                        "segment_id": "",
                        "inspection_mode": "search_asr",
                        "search_terms": ["dog", "father", "rent"],
                        "modality_hint": ["asr"],
                        "expected_evidence": "literal timestamp hits for later visual inspection",
                    }
                ],
            },
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl")

    decision = reasoner.decide(
        question="How did the identified man sustain his injury?",
        options={"A": "A firework", "D": "A dog dragged his arm"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "multi_window", "aggregation": "compare"},
        query_requirements={"requires_identity_link": True},
        completion_status={"ready_for_answer": False},
        temporal_navigation={},
        remaining_budget=4,
        evidence_digest=(),
    )

    assert decision.action == "investigate"
    assert decision.primary_gap is not None
    assert decision.primary_gap.gap_id == "gap_cause"
    assert decision.tasks[0].inspection_mode == "search_asr"
    assert decision.tasks[0].search_terms == ("dog", "father", "rent")
    assert "search_asr" in api.calls[0]["prompt"]
    assert "navigation only" in api.calls[0]["prompt"]
    assert "competing option" in api.calls[0]["prompt"]
    assert "one contrastive search_asr task" in api.calls[0]["prompt"]
    assert "Do not repeat" in api.calls[0]["prompt"]
    assert "after any navigation result" in api.calls[0]["prompt"]


def test_gemini_reasoner_repairs_truncated_investigation_json(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            '{"action":"investigate","primary_gap":{"gap_id":"gap_clock","description":"read the visible clock"',
            {
                "action": "investigate",
                "primary_gap": {
                    "gap_id": "gap_clock",
                    "description": "The displayed transition clock.",
                    "success_conditions": ["read the clock at the transition"],
                },
                "tasks": [
                    {
                        "query_id": "r1_clock",
                        "goal": "Inspect the scoreboard transition.",
                        "segment_id": "seg_0001",
                        "modality_hint": ["visual", "ocr"],
                    }
                ],
            },
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl")

    decision = reasoner.decide(
        question="When does the visible score change?",
        options={"A": "8:13-5:58", "B": "5:58-2:57"},
        workspace_overview={"segment_overviews": [{"segment_id": "seg_0001"}]},
        query_contract={},
        query_requirements={},
        completion_status={"ready_for_answer": False},
        temporal_navigation={},
        remaining_budget=4,
        evidence_digest=(),
    )

    assert len(api.calls) == 2
    assert decision.primary_gap is not None
    assert decision.primary_gap.gap_id == "gap_clock"
    assert decision.tasks[0].query_id == "r1_clock"


def test_gemini_reasoner_does_not_spend_an_extra_call_guessing_invalid_option(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "The video is unclear.", "citations": ["ev_1"]},
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl")

    decision = reasoner.decide(
        question="What happens next?",
        options={"A": "Walks away", "B": "Leaves on a stretcher"},
        workspace_overview={"segment_overviews": []},
        query_contract={},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=1,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "He is carried away on a stretcher.",
                "virtual_time_range": [10.0, 20.0],
                "modality": "visual",
                "source_lineage": [],
            },
        ),
    )

    assert len(api.calls) == 1
    assert decision.answer == ""
    assert decision.support_status == "insufficient"


def test_gemini_reasoner_uses_explicit_forced_choice_only_at_finalization(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "The video is unclear.", "citations": ["ev_1"]},
            {"answer": "B. Leaves on a stretcher.", "citations": ["ev_1"]},
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl")

    decision = reasoner.decide(
        question="What happens next?",
        options={"A": "Walks away", "B": "Leaves on a stretcher"},
        workspace_overview={"segment_overviews": []},
        query_contract={},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=0,
        force_finalize=True,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "He is carried away on a stretcher.",
                "virtual_time_range": [10.0, 20.0],
                "modality": "visual",
                "source_lineage": [],
            },
        ),
    )

    assert len(api.calls) == 2
    assert "investigation budget is exhausted" in api.calls[1]["prompt"]
    assert decision.answer.startswith("B.")
    assert decision.support_status == "insufficient"


def test_text_only_reasoner_never_sends_overview_or_evidence_images(tmp_path: Path) -> None:
    overview = tmp_path / "overview.jpg"
    Image.new("RGB", (64, 36), color=(20, 30, 40)).save(overview)
    api = ScriptedVisionClient((
        {
            "action": "investigate",
            "primary_gap": {"gap_id": "gap_1", "description": "locate the event", "success_conditions": []},
            "tasks": [{"query_id": "q1", "goal": "Open the segment.", "segment_id": "seg_1"}],
        },
    ))
    api.model = "gpt-5.5"
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl", allow_visual_input=False)

    reasoner.decide(
        question="What happens?",
        options={"A": "One", "B": "Two"},
        workspace_overview={
            "segment_overviews": [{"segment_id": "seg_1", "overview_thumbnail_grid_path": str(overview)}]
        },
        query_contract={},
        query_requirements={},
        temporal_navigation={},
        remaining_budget=4,
        evidence_digest=(),
    )

    assert api.calls[0]["image_paths"] == ()
    assert "You are text-only" in api.calls[0]["prompt"]
    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert trace["agent_role"] == "reasoner"
    assert trace["model"] == "gpt-5.5"
    assert trace["visual_input_enabled"] is False


def test_model_investigator_returns_asr_navigation_hints_without_vlm(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(())
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    task = InvestigationTask(
        query_id="search_number",
        goal="Locate the literal discussion of the number.",
        inspection_mode="search_asr",
        search_terms=("number", "board"),
        modality_hint=("asr",),
    )

    report = investigator.run_batch((task,))[0]

    assert api.calls == []
    assert report.status == "satisfied"
    assert report.cost["vlm_calls"] == 0
    assert report.cost["tool_trace"] == ("search_asr",)
    assert report.evidence
    assert all(record.modality == "asr" for record in report.evidence)
    assert all(record.evidence_kind == "navigation_hint" for record in report.evidence)
    assert all(not record.frame_refs for record in report.evidence)
    assert all(record.operation_metadata["search_terms"] == ["number", "board"] for record in report.evidence)


def test_asr_search_only_resolves_explicit_lexical_navigation_conditions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = GeminiInvestigator(workspace, api=ScriptedVisionClient(()), trace_path=workspace.root_dir / "trace.jsonl")
    task = InvestigationTask(
        query_id="search_number", goal="Find and read the number on the board.", inspection_mode="search_asr",
        search_terms=("number", "board"), gap_id="gap_number",
        conditions=(
            _interactive.GapCondition("gap_number_nav", "locate a literal transcript match", condition_type="lexical_navigation"),
            _interactive.GapCondition("gap_number_read", "read the visible number on the board", condition_type="measurement"),
        ),
    )
    report = investigator.run_batch((task,))[0]
    assert [result.status for result in report.condition_results] == ["satisfied", "unknown"]
    assert report.resolution == "partial"
    assert report.resolved_conditions == ("locate a literal transcript match",)
    assert report.unresolved_conditions == ("read the visible number on the board",)


def test_model_investigator_records_empty_asr_search_as_negative_navigation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(())
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    task = InvestigationTask(
        query_id="search_missing",
        goal="Check whether this literal clue appears in the transcript.",
        inspection_mode="search_asr",
        search_terms=("definitely-not-present",),
        modality_hint=("asr",),
    )

    report = investigator.run_batch((task,))[0]

    assert api.calls == []
    assert report.status == "satisfied"
    assert len(report.evidence) == 1
    record = report.evidence[0]
    assert record.evidence_kind == "navigation_hint"
    assert record.observation_polarity == "negative"
    assert record.start_sec is None
    assert record.end_sec is None
    assert record.operation_metadata["hit_count"] == 0
    assert "No literal ASR matches" in record.verbatim


def test_gemini_reasoner_dispatches_model_audit_repair_tasks(tmp_path: Path) -> None:
    frame_path = tmp_path / "audit_evidence.jpg"
    Image.new("RGB", (64, 36), color=(70, 80, 90)).save(frame_path)
    evidence_record = EvidenceRecord(
        evidence_id="ev_1",
        beat_id="",
        start_sec=50.0,
        end_sec=55.0,
        modality="visual",
        pointer="virtual://case/observations/audit",
        verbatim="The money could help many people later.",
        frame_refs=(str(frame_path),),
        attestation_model="test-model",
        temporal_scope="window",
        evidence_kind="visual_observation",
        observation_polarity="positive",
        sampling_coverage="sparse",
    )
    api = ScriptedVisionClient(
        (
            {
                "action": "answer",
                "answer": "D. A downstream benefit.",
                "citations": ["ev_1"],
                "entity_clusters": [],
            },
            {
                "verdict": "insufficient",
                "reason": "The citation states a consequence, not the motive asked by the question.",
                "evidence_relation": "consequence_only",
                "unresolved_alternatives": ["B"],
                "tasks": [
                    {
                        "query_id": "audit_r2_t1",
                        "goal": "Inspect the preceding conversation for the broader motive.",
                        "segment_id": "seg_0001",
                        "time_range": [40.0, 60.0],
                        "modality_hint": ["visual", "asr"],
                        "expected_evidence": "direct dialogue establishing the motive",
                    }
                ],
            },
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="Why does the person perform the action?",
        options={"B": "A broader motive", "D": "A downstream benefit"},
        workspace_overview={"segment_overviews": []},
        query_contract={},
        query_requirements={"requires_narrative_inference": True},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=4,
        evidence=(evidence_record,),
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "The money could help many people later.",
                "virtual_time_range": [50.0, 55.0],
                "modality": "visual",
                "source_lineage": [{"segment_id": "seg_0001"}],
            },
        ),
    )

    assert decision.action == "investigate"
    assert decision.tasks[0].query_id == "audit_r2_t1"
    assert len(api.calls) == 2
    assert "Do not reward citation relevance alone" in api.calls[1]["prompt"]
    assert "strongest_alternative" in api.calls[1]["prompt"]
    assert '"option_assessments"' not in api.calls[1]["prompt"]
    assert api.calls[1]["max_tokens"] >= 1400
    assert api.calls[0]["image_paths"] == (str(frame_path),)
    assert api.calls[1]["image_paths"] == (str(frame_path),)


def test_gemini_reasoner_downgrades_verdict_from_truncated_answer_audit(tmp_path: Path) -> None:
    class TruncatedAuditClient:
        def __init__(self) -> None:
            self.responses = [
                json.dumps({"action": "answer", "answer": "D. A downstream benefit.", "citations": ["ev_1"]}),
                '```json\n{"verdict":"contradicted","reason":"The cited fact is only a consequence",',
            ]

        def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
            del prompt, image_paths, max_tokens
            return self.responses.pop(0)

    reasoner = GeminiReasoner(TruncatedAuditClient(), trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="Why did the person perform the action?",
        options={"B": "A broader motive", "D": "A downstream benefit"},
        workspace_overview={"segment_overviews": []},
        workspace_duration_sec=300.0,
        query_contract={"required_scope": "window", "aggregation": "none", "observation_target": "relation"},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=3,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "The action has a later benefit.",
                "virtual_time_range": [120.0, 130.0],
                "modality": "visual",
                "source_lineage": [{"segment_id": "seg_0002"}],
            },
        ),
    )

    assert decision.action == "answer"
    assert decision.answer.startswith("D.")
    assert decision.support_status == "unknown"
    assert "truncated" in decision.support_reason.casefold()


def test_gemini_reasoner_preserves_candidate_when_forced_finalization_still_investigates(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {
                "action": "answer",
                "answer": "A. Five.",
                "citations": ["ev_1"],
            },
            {
                "verdict": "insufficient",
                "reason": "Another interviewee may be missing.",
                "tasks": [
                    {
                        "query_id": "audit_more",
                        "goal": "Inspect another interview window.",
                        "segment_id": "seg_0002",
                    }
                ],
            },
            {
                "action": "investigate",
                "tasks": [{"query_id": "too_late", "goal": "Keep looking.", "segment_id": "seg_0003"}],
            },
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")
    kwargs = {
        "question": "How many interviewees appear in total?",
        "options": {"A": "Five", "D": "Six"},
        "workspace_overview": {"segment_overviews": []},
        "query_contract": {
            "required_scope": "full_video",
            "aggregation": "deduplicate",
            "quantifier": "distinct_count",
        },
        "query_requirements": {},
        "completion_status": {"ready_for_answer": True},
        "temporal_navigation": {},
        "evidence_digest": (
            {
                "evidence_id": "ev_1",
                "summary": "Five interviewees are currently identified.",
                "virtual_time_range": [0.0, 60.0],
                "modality": "visual",
                "source_lineage": [{"segment_id": "seg_0001"}],
            },
        ),
    }

    first = reasoner.decide(**kwargs, remaining_budget=2)
    final = reasoner.decide(**kwargs, remaining_budget=0, force_finalize=True)

    assert first.action == "investigate"
    assert final.action == "answer"
    assert final.answer == "A. Five."
    assert final.support_status == "insufficient"
    assert len(api.calls) == 3


def test_gemini_reasoner_keeps_last_valid_choice_when_final_response_is_explanatory(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "C", "citations": ["ev_1"]},
            {
                "action": "answer",
                "answer": "The evidence implies a value larger than the listed estimates, so no option is exact.",
                "citations": ["ev_1"],
            },
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")
    kwargs = {
        "question": "Which estimate does the video state?",
        "options": {"A": "100 trillion", "B": "75 trillion", "C": "over 25 trillion", "D": "50 trillion"},
        "workspace_overview": {"segment_overviews": []},
        "query_contract": {"required_scope": "window", "aggregation": "none"},
        "query_requirements": {},
        "completion_status": {"ready_for_answer": True},
        "temporal_navigation": {},
        "evidence_digest": (
            {
                "evidence_id": "ev_1",
                "summary": "The video states a lower bound above 25 trillion.",
                "virtual_time_range": [10.0, 20.0],
                "modality": "visual",
                "source_lineage": [],
            },
        ),
    }

    first = reasoner.decide(**kwargs, remaining_budget=1)
    final = reasoner.decide(**kwargs, remaining_budget=0, force_finalize=True)

    assert first.answer == "C"
    assert final.answer == "C"
    assert final.citations == ("ev_1",)
    assert final.support_status == "insufficient"
    assert "valid option" in final.support_reason


def test_forced_finalization_answer_is_calibrated_by_unique_canonical_verdict(tmp_path: Path) -> None:
    api = ScriptedVisionClient(({"action": "answer", "answer": "F. unsupported", "citations": ["ev_1"]},))
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")
    options = {"F": "unsupported", "H": "supported narrative"}

    final = reasoner.decide(
        question="Which narrative bridge is supported?",
        options=options,
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "multi_window", "aggregation": "none"},
        query_requirements={},
        completion_status={
            "ready_for_answer": False,
            "canonical_fact_snapshot": {"inferred_facts": [{"fact_id": "bridge_1"}]},
            "option_verdict_table": {
                "best_option": "H",
                "audit_status": "partial",
                "option_verdicts": {
                    "F": {"status": "contradicted", "reason": "Conflicts with the observed outcome."},
                    "H": {"status": "supported", "reason": "Matches the complete narrative bridge."},
                },
            },
        },
        temporal_navigation={},
        evidence_digest=({
            "evidence_id": "ev_1",
            "summary": "The complete setup and outcome support H.",
            "virtual_time_range": [0.0, 20.0],
            "modality": "visual",
            "source_lineage": [],
        },),
        remaining_budget=0,
        force_finalize=True,
    )

    assert final.answer == "H. supported narrative"
    assert "canonical_option_verdict_table" in final.support_reason


def test_gemini_reasoner_normalizes_structured_answer_payload_and_nested_citations(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {
                "action": "answer",
                "answer": {"option": "B", "text": "9", "citations": ["ev_1"]},
            },
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="What number appears on the board?",
        options={"A": "7", "B": "9"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=2,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "The board shows 9.",
                "virtual_time_range": [10.0, 20.0],
                "modality": "visual",
                "source_lineage": [{"segment_id": "seg_0001"}],
            },
        ),
    )

    assert decision.answer == "B. 9"
    assert decision.citations == ("ev_1",)


def test_gemini_reasoner_replays_investigator_frames_with_visual_manifest(tmp_path: Path) -> None:
    overview_path = tmp_path / "overview.jpg"
    Image.new("RGB", (64, 36), color=(20, 40, 60)).save(overview_path)
    frame_paths = []
    for index in range(4):
        path = tmp_path / f"evidence_{index}.jpg"
        Image.new("RGB", (64, 36), color=(60 + index, 80, 100)).save(path)
        frame_paths.append(str(path))
    record = EvidenceRecord(
        evidence_id="ev_visual_1",
        beat_id="",
        start_sec=40.0,
        end_sec=60.0,
        modality="visual",
        pointer="virtual://case/observations/1",
        verbatim="The board visibly shows 9.",
        frame_refs=tuple(frame_paths),
        attestation_model="test-model",
        temporal_scope="window",
        evidence_kind="visual_observation",
        observation_polarity="positive",
        sampling_coverage="sparse",
    )
    api = ScriptedVisionClient(({"action": "answer", "answer": "B. 9", "citations": ["ev_visual_1"]},))
    trace_path = tmp_path / "interactions.jsonl"
    reasoner = GeminiReasoner(api, trace_path=trace_path)

    decision = reasoner.decide(
        question="What number appears on the board?",
        options={"A": "7", "B": "9"},
        workspace_overview={
            "segment_overviews": [{"segment_id": "seg_1", "overview_thumbnail_grid_path": str(overview_path)}]
        },
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=2,
        evidence=(record,),
        evidence_digest=(
            {
                "evidence_id": "ev_visual_1",
                "summary": record.verbatim,
                "virtual_time_range": [40.0, 60.0],
                "modality": "visual",
                "source_lineage": [],
            },
        ),
    )

    assert decision.answer == "B. 9"
    assert api.calls[0]["image_paths"] == (str(overview_path), *frame_paths)
    assert "Visual input manifest" in api.calls[0]["prompt"]
    assert "ev_visual_1" in api.calls[0]["prompt"]
    assert "Re-check replayed evidence images" in api.calls[0]["prompt"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["image_paths"] == [str(overview_path), *frame_paths]


def test_gemini_reasoner_caps_overview_and_evidence_images_at_40(tmp_path: Path) -> None:
    overview_rows = []
    for index in range(12):
        path = tmp_path / f"overview_{index:02d}.jpg"
        Image.new("RGB", (32, 18), color=(index, 20, 30)).save(path)
        overview_rows.append({"segment_id": f"seg_{index:02d}", "overview_thumbnail_grid_path": str(path)})
    records = []
    digest = []
    for record_index in range(20):
        frame_paths = []
        for frame_index in range(4):
            path = tmp_path / f"ev_{record_index:02d}_{frame_index}.jpg"
            Image.new("RGB", (32, 18), color=(record_index, frame_index, 40)).save(path)
            frame_paths.append(str(path))
        evidence_id = f"ev_{record_index:02d}"
        records.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                beat_id="",
                start_sec=float(record_index * 10),
                end_sec=float(record_index * 10 + 5),
                modality="visual",
                pointer=f"virtual://case/{evidence_id}",
                verbatim=f"Observation {record_index}",
                frame_refs=tuple(frame_paths),
                attestation_model="test-model",
                temporal_scope="window",
                evidence_kind="visual_observation",
                observation_polarity="positive",
                sampling_coverage="sparse",
            )
        )
        digest.append(
            {
                "evidence_id": evidence_id,
                "summary": f"Observation {record_index}",
                "virtual_time_range": [record_index * 10, record_index * 10 + 5],
                "modality": "visual",
                "source_lineage": [],
            }
        )
    api = ScriptedVisionClient(({"action": "answer", "answer": "B. 9", "citations": ["ev_00"]},))
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl")

    reasoner.decide(
        question="What number appears?",
        options={"A": "7", "B": "9"},
        workspace_overview={"segment_overviews": overview_rows},
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=2,
        evidence=tuple(records),
        evidence_digest=tuple(digest),
    )

    assert len(api.calls[0]["image_paths"]) == 40
    assert "Visual input manifest" in api.calls[0]["prompt"]


def test_gemini_reasoner_uses_event_ledger_instead_of_sparse_replay_for_atomic_counts(tmp_path: Path) -> None:
    overview_path = tmp_path / "overview.jpg"
    evidence_path = tmp_path / "event.jpg"
    Image.new("RGB", (64, 36), color=(20, 30, 40)).save(overview_path)
    Image.new("RGB", (64, 36), color=(80, 90, 100)).save(evidence_path)
    record = EvidenceRecord(
        evidence_id="ev_event_1",
        beat_id="",
        start_sec=60.0,
        end_sec=120.0,
        modality="visual",
        pointer="virtual://case/events/1",
        verbatim="One verified news segment occurrence.",
        frame_refs=(str(evidence_path),),
        attestation_model="test-model",
        temporal_scope="window",
        evidence_kind="event_observation",
        observation_polarity="positive",
        sampling_coverage="sparse",
    )
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "C. 6", "citations": ["ev_event_1"]},
            {"verdict": "supported", "reason": "The complete event ledger contains six occurrences.", "tasks": []},
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl")

    reasoner.decide(
        question="How many news segments appear in total?",
        options={"A": "4", "B": "5", "C": "6", "D": "8"},
        workspace_overview={
            "segment_overviews": [{"segment_id": "seg_1", "overview_thumbnail_grid_path": str(overview_path)}]
        },
        query_contract={"required_scope": "full_video", "aggregation": "count"},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=0,
        evidence=(record,),
        evidence_digest=(
            {
                "evidence_id": "ev_event_1",
                "summary": record.verbatim,
                "virtual_time_range": [60.0, 120.0],
                "modality": "visual",
                "events": [{"event_key": "news_1", "start_sec": 60.0, "end_sec": 120.0}],
                "source_lineage": [],
            },
        ),
    )

    assert api.calls[0]["image_paths"] == (str(overview_path),)
    assert '"kind": "evidence"' not in api.calls[0]["prompt"]
    assert api.calls[1]["image_paths"] == (str(overview_path),)


def test_gemini_reasoner_forces_best_effort_answer_when_no_candidate_exists(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "investigate", "tasks": [{"query_id": "too_late", "segment_id": "seg_1"}]},
            {"answer": "D. The dog dragged his arm.", "citations": ["ev_1"], "entity_clusters": []},
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="How did the identified man sustain his injury?",
        options={"A": "A firework", "D": "A dog dragged his arm"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "multi_window", "aggregation": "compare"},
        query_requirements={"requires_identity_link": True},
        completion_status={"ready_for_answer": False},
        temporal_navigation={},
        remaining_budget=0,
        force_finalize=True,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "A dog drags a man's arm during an earlier confrontation.",
                "virtual_time_range": [100.0, 110.0],
                "modality": "visual",
                "source_lineage": [{"segment_id": "seg_1"}],
            },
        ),
    )

    assert decision.action == "answer"
    assert decision.answer.startswith("D.")
    assert decision.support_status == "insufficient"
    assert len(api.calls) == 2
    assert "must choose one best option" in api.calls[1]["prompt"]


def test_gemini_reasoner_retries_empty_forced_answer_with_compact_evidence(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "investigate", "tasks": [{"query_id": "late", "goal": "Inspect more.", "segment_id": "seg_1"}]},
            "",
            {"answer": "B. Three", "citations": ["ev_1"], "entity_clusters": []},
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="How many scholars appear?",
        options={"A": "Two", "B": "Three"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "full_video", "aggregation": "deduplicate"},
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=0,
        force_finalize=True,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "Three witnessed scholar identities remain after deduplication.",
                "entities": [],
                "virtual_time_range": [0.0, 60.0],
                "modality": "visual",
            },
        ),
    )

    assert decision.answer == "B. Three"
    assert len(api.calls) == 3
    assert "compact verified evidence" in api.calls[2]["prompt"]
    assert api.calls[1]["max_tokens"] == 4096
    assert api.calls[2]["max_tokens"] == 4096


def test_gpt5_followup_and_answer_audit_use_reasoning_safe_completion_budget(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "B. A direct motive.", "citations": ["ev_1"]},
            {"verdict": "supported", "reason": "The cited dialogue directly states the motive."},
        )
    )
    api.model = "gpt-5.5"
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="Why did the person perform the action?",
        options={"A": "A later consequence", "B": "A direct motive"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={"requires_narrative_inference": True},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=2,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "The person directly states the motive.",
                "virtual_time_range": [10.0, 20.0],
                "modality": "visual",
                "source_lineage": [],
            },
        ),
    )

    assert decision.support_status == "supported"
    assert [call["max_tokens"] for call in api.calls] == [4096, 4096]


def test_gpt5_initial_investigation_uses_reasoning_safe_completion_budget(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {
                "action": "investigate",
                "tasks": [{"query_id": "initial_q1", "goal": "Inspect the event.", "segment_id": "seg_1"}],
            },
        )
    )
    api.model = "gpt-5.5"
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="What happens to the computer?",
        options={"A": "It restarts", "B": "It breaks"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={},
        completion_status={"ready_for_answer": False},
        temporal_navigation={},
        remaining_budget=2,
        evidence_digest=(),
    )

    assert decision.action == "investigate"
    assert [call["max_tokens"] for call in api.calls] == [4096]


def test_gpt5_json_repair_uses_reasoning_safe_completion_budget(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            '{"action":"investigate","tasks":[',
            {
                "action": "investigate",
                "tasks": [{"query_id": "repair_q1", "goal": "Inspect the event.", "segment_id": "seg_1"}],
            },
        )
    )
    api.model = "gpt-5.5"
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="What happens to the computer?",
        options={"A": "It restarts", "B": "It breaks"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={},
        completion_status={"ready_for_answer": False},
        temporal_navigation={},
        remaining_budget=2,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "A relevant scene was located, but its outcome remains unclear.",
                "virtual_time_range": [10.0, 20.0],
                "modality": "visual",
                "source_lineage": [],
            },
        ),
    )

    assert decision.action == "investigate"
    assert [call["max_tokens"] for call in api.calls] == [4096, 4096]


def test_reasoner_trace_keeps_original_response_metadata_when_json_repair_runs(tmp_path: Path) -> None:
    class MetadataClient:
        model = "gpt-5.5"

        def __init__(self) -> None:
            self.calls = 0
            self.last_response_metadata: dict[str, Any] = {}

        def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
            del prompt, image_paths
            self.calls += 1
            if self.calls == 1:
                self.last_response_metadata = {
                    "finish_reason": "length",
                    "completion_tokens": max_tokens,
                    "reasoning_tokens": max_tokens,
                    "content_chars": 0,
                    "requested_completion_tokens": max_tokens,
                }
                return ""
            payload = {
                "action": "investigate",
                "tasks": [{"query_id": "repair_q1", "goal": "Inspect the event.", "segment_id": "seg_1"}],
            }
            raw = json.dumps(payload)
            self.last_response_metadata = {
                "finish_reason": "stop",
                "completion_tokens": 180,
                "reasoning_tokens": 64,
                "content_chars": len(raw),
                "requested_completion_tokens": max_tokens,
            }
            return raw

    trace_path = tmp_path / "interactions.jsonl"
    reasoner = GeminiReasoner(MetadataClient(), trace_path=trace_path)

    decision = reasoner.decide(
        question="What happens to the computer?",
        options={"A": "It restarts", "B": "It breaks"},
        workspace_overview={"segment_overviews": []},
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={},
        completion_status={"ready_for_answer": False},
        temporal_navigation={},
        remaining_budget=2,
        evidence_digest=(),
    )
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    rows_by_type = {row["type"]: row for row in rows}

    assert decision.action == "investigate"
    assert rows_by_type["reasoner_investigate"]["raw"] == ""
    assert rows_by_type["reasoner_investigate"]["api_response"]["finish_reason"] == "length"
    assert rows_by_type["reasoner_investigate"]["api_response"]["reasoning_tokens"] == 4096
    assert rows_by_type["reasoner_json_repair"]["api_response"]["finish_reason"] == "stop"


def test_gemini_reasoner_records_audit_revision_as_unadopted_challenge(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "A. Five.", "citations": ["ev_1"]},
            {
                "verdict": "contradicted",
                "reason": "Six distinct interviewees are directly supported.",
                "revised_answer": "D. Six.",
                "revised_citations": ["ev_1", "ev_2"],
                "revised_support_status": "supported",
                "revised_entity_clusters": [],
                "tasks": [],
            },
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="How many interviewees appear in total?",
        options={"A": "Five", "D": "Six"},
        workspace_overview={"segment_overviews": []},
        query_contract={
            "required_scope": "full_video",
            "aggregation": "deduplicate",
            "quantifier": "distinct_count",
        },
        query_requirements={},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=2,
        evidence_digest=(
            {"evidence_id": "ev_1", "summary": "Three interviewees.", "source_lineage": []},
            {"evidence_id": "ev_2", "summary": "Three additional interviewees.", "source_lineage": []},
        ),
    )

    assert decision.action == "answer"
    assert decision.answer == "A. Five."
    assert decision.citations == ("ev_1",)
    assert decision.support_status == "insufficient"
    rows = [json.loads(line) for line in (tmp_path / "interactions.jsonl").read_text().splitlines()]
    challenge = next(row for row in rows if row["type"] == "reasoner_answer_challenge")
    assert challenge["parsed"]["proposed_answer"] == "D. Six."
    assert challenge["parsed"]["adopted"] is False


def test_gemini_reasoner_dispatches_independent_claim_verification_for_relation_answers(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "D. A downstream benefit.", "citations": ["ev_1"]},
            {"verdict": "supported", "reason": "The cited dialogue appears relevant.", "tasks": []},
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="Why did the person give away the money?",
        options={"B": "A broader motive", "D": "A downstream benefit"},
        workspace_overview={"segment_overviews": []},
        workspace_duration_sec=300.0,
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={"requires_narrative_inference": True},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=3,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "The money could help many people later.",
                "virtual_time_range": [120.0, 130.0],
                "modality": "visual",
                "source_lineage": [{"segment_id": "seg_0002"}],
            },
        ),
    )

    assert decision.action == "investigate"
    assert decision.tasks[0].inspection_mode == "verify_claim"
    assert decision.tasks[0].claim_to_verify.startswith("D.")
    assert decision.tasks[0].claim_relation == "decision_motive"
    assert decision.tasks[0].segment_id == "seg_0002"
    assert decision.tasks[0].time_range == (0.0, 300.0)


def test_comparison_and_sequence_contracts_require_independent_claim_verification() -> None:
    comparison = _requires_independent_claim_verification(
        {
            "question": "What color change occurred?",
            "query_contract": {"aggregation": "compare", "quantifier": "comparison"},
            "query_requirements": {},
        }
    )
    sequence = _requires_independent_claim_verification(
        {
            "question": "Which sequence is correct?",
            "query_contract": {"aggregation": "order", "quantifier": "order"},
            "query_requirements": {"requires_temporal_sequence": True},
        }
    )

    assert comparison is True
    assert sequence is True


def test_matching_claim_assessment_treats_option_label_as_candidate_identity() -> None:
    assessment = {
        "candidate_answer": "D",
        "verdict": "supports",
        "reason": "The exact option was checked.",
    }

    matched = _matching_claim_assessment(({"claim_assessment": assessment},), "D. Full option text")

    assert matched == assessment


def test_forced_finalization_keeps_existing_claim_assessment_active(tmp_path: Path) -> None:
    api = ScriptedVisionClient(
        (
            {"action": "answer", "answer": "D. A downstream use.", "citations": ["ev_1"]},
            {"verdict": "supported", "reason": "The cited sentence mentions the proposed use.", "tasks": []},
        )
    )
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "interactions.jsonl")

    decision = reasoner.decide(
        question="Why did the person perform the action?",
        options={"B": "A decision motive", "D": "A downstream use"},
        workspace_overview={"segment_overviews": []},
        workspace_duration_sec=300.0,
        query_contract={"required_scope": "window", "aggregation": "none"},
        query_requirements={"requires_narrative_inference": True},
        completion_status={"ready_for_answer": True},
        temporal_navigation={},
        remaining_budget=0,
        force_finalize=True,
        evidence_digest=(
            {
                "evidence_id": "ev_1",
                "summary": "The action has a stated downstream use.",
                "virtual_time_range": [120.0, 130.0],
                "modality": "visual",
                "source_lineage": [{"segment_id": "seg_0002"}],
                "claim_assessment": {
                    "candidate_answer": "D",
                    "claim_relation": "decision_motive",
                    "candidate_role": "stated_use",
                    "verdict": "insufficient",
                    "reason": "The observed use does not establish the decision motive.",
                },
            },
        ),
    )

    assert decision.action == "answer"
    assert decision.answer.startswith("D.")
    assert decision.support_status == "insufficient"
    assert "does not establish" in decision.support_reason


def test_model_investigator_records_independent_claim_assessment(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {
                "summary": "The dialogue states a downstream benefit but not the broader motive.",
                "confidence": 0.9,
                "claim_verdict": "refutes",
                "relation_type": "consequence_only",
                "candidate_role": "downstream_consequence",
                "strongest_alternative": "B. A broader motive.",
                "reason": "The cited scene does not establish the candidate as the motive.",
                "need_detail": False,
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="verify_r2_candidate",
        goal="Independently verify the proposed causal answer.",
        segment_id="seg_0001",
        time_range=(0.0, 60.0),
        modality_hint=("visual", "asr"),
        expected_evidence="direct evidence for the causal relation",
        inspection_mode="verify_claim",
        claim_to_verify="D. A downstream benefit.",
        alternative_answers=("B. A broader motive.",),
    )

    report = investigator.run_batch((task,))[0]

    assert "independent claim verifier" in api.calls[0]["prompt"]
    assert "decision motive" in api.calls[0]["prompt"]
    assert "identify the strongest alternative" in api.calls[0]["prompt"]
    assessment = report.evidence[0].operation_metadata["claim_assessment"]
    assert assessment["verdict"] == "refutes"
    assert assessment["candidate_answer"].startswith("D.")
    assert assessment["candidate_role"] == "downstream_consequence"
    assert assessment["strongest_alternative"].startswith("B.")


def test_claim_verifier_downgrades_relation_role_mismatch_without_erasing_answer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {
                "summary": "The money has a stated downstream use.",
                "confidence": 0.95,
                "claim_verdict": "supports",
                "relation_type": "direct",
                "candidate_role": "stated_use",
                "strongest_alternative": "B. A broader motive.",
                "reason": "The dialogue directly states how the money can be used.",
                "need_detail": False,
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="verify_r2_candidate",
        goal="Verify the proposed reason.",
        segment_id="seg_0001",
        time_range=(0.0, 60.0),
        modality_hint=("visual", "asr"),
        inspection_mode="verify_claim",
        claim_to_verify="D. A downstream benefit.",
        alternative_answers=("B. A broader motive.",),
        claim_relation="decision_motive",
    )

    report = investigator.run_batch((task,))[0]
    assessment = report.evidence[0].operation_metadata["claim_assessment"]

    assert assessment["candidate_answer"].startswith("D.")
    assert assessment["claim_relation"] == "decision_motive"
    assert assessment["candidate_role"] == "stated_use"
    assert assessment["verdict"] == "insufficient"
    assert "does not satisfy" in assessment["reason"]


def test_followup_prompt_requires_reasoner_to_act_on_claim_role_mismatch() -> None:
    prompt = _followup_prompt(
        {
            "question": "Why did the person perform the action?",
            "options": {"B": "A motive", "D": "A downstream use"},
            "workspace_overview": {"segment_overviews": []},
            "query_contract": {},
            "query_requirements": {},
            "completion_status": {"ready_for_answer": True},
            "temporal_navigation": {},
        },
        (
            {
                "evidence_id": "ev_claim",
                "claim_assessment": {
                    "claim_relation": "decision_motive",
                    "candidate_role": "stated_use",
                    "verdict": "insufficient",
                },
            },
        ),
    )

    assert "candidate_role does not satisfy claim_relation" in prompt
    assert "investigate the missing relation" in prompt


def test_followup_prompt_requires_explicit_scalar_measurement_derivation() -> None:
    prompt = _followup_prompt(
        {
            "question": "How many calories had he consumed by the time he met his teammate?",
            "options": {"A": "500 calories", "B": "700 calories"},
            "workspace_overview": {"segment_overviews": []},
            "query_contract": {
                "quantifier": "scalar_quantity",
                "aggregation": "accumulate",
                "measurement_unit": "calorie",
                "boundary_hint": "by the time he met his teammate",
            },
            "query_requirements": {},
            "completion_status": {"ready_for_answer": True},
            "temporal_navigation": {},
        },
        (),
    )

    assert "scalar_quantity" in prompt
    assert "delta or cumulative" in prompt
    assert "boundary" in prompt


def test_answer_audit_targets_relation_risk_without_rechecking_simple_quantities() -> None:
    assert _should_audit_answer(
        {
            "question": "Why did the person give away the money?",
            "query_contract": {
                "required_scope": "window", "aggregation": "none", "observation_target": "relation",
            },
            "query_requirements": {},
        }
    )
    assert _should_audit_answer(
        {
            "question": "How many people were interviewed in total?",
            "query_contract": {
                "required_scope": "full_video", "aggregation": "deduplicate", "quantifier": "distinct_count",
            },
            "query_requirements": {},
        }
    )
    for question in (
        "Which factor was not the cause of the split?",
        "In which episode do they get married?",
        "At what time does the target appear?",
        "What title best summarizes this video?",
        "Who protects the mermaid?",
        "How many tasks has the other team completed?",
    ):
        assert _should_audit_answer(
            {
                "question": question,
                "query_contract": {"required_scope": "multi_window", "aggregation": "compare"},
                "query_requirements": {},
            }
        )
    assert _should_audit_answer(
        {
            "question": "Which way is red facing relative to green?",
            "query_contract": {"required_scope": "window", "aggregation": "compare"},
            "query_requirements": {"requires_spatial_relation": True},
        }
    )
    assert not _should_audit_answer(
        {
            "question": "What number is written on the scoreboard?",
            "query_contract": {"required_scope": "window", "aggregation": "none"},
            "query_requirements": {},
        }
    )


    assert not _should_audit_answer(
        {
            "question": "How many meters did they complete in 25 minutes?",
            "query_contract": {"required_scope": "multi_window", "aggregation": "deduplicate"},
            "query_requirements": {},
        }
    )


def test_option_claim_contract_keeps_selected_mechanism_and_excludes_question_background() -> None:
    contract = _compile_option_claim_contract(
        "How does the man get off the airplane while it is flying over an island?",
        {"A": "He applies lipstick, creates fake spots, and is removed on a stretcher.", "B": "He jumps onto the island."},
        "A. He applies lipstick, creates fake spots, and is removed on a stretcher.",
    )
    assert contract["option_id"] == "A"
    assert any("lipstick" in atom for atom in contract["atoms"])
    assert any("spots" in atom for atom in contract["atoms"])
    assert any("stretcher" in atom for atom in contract["atoms"])
    assert all("island" not in atom for atom in contract["atoms"])
    assert contract["compiler_source"] == "selected_option_text"


def test_reasoner_deduplicates_identical_answer_audit(tmp_path: Path) -> None:
    api = ScriptedVisionClient((
        {"action": "answer", "answer": "A. By a stretcher.", "citations": ["ev_1"]},
        {"verdict": "insufficient", "reason": "The mechanism is incomplete.", "tasks": []},
        {"action": "answer", "answer": "A. By a stretcher.", "citations": ["ev_1"]},
    ))
    reasoner = GeminiReasoner(api, trace_path=tmp_path / "trace.jsonl")
    kwargs = {
        "question": "How does he leave the airplane?",
        "options": {"A": "By a stretcher", "B": "By jumping"},
        "workspace_overview": {"segment_overviews": []},
        "query_contract": {"observation_target": "relation"}, "query_requirements": {},
        "completion_status": {"ready_for_answer": True}, "temporal_navigation": {}, "remaining_budget": 2,
        "evidence_digest": ({
            "evidence_id": "ev_1", "summary": "He is carried on a stretcher.",
            "virtual_time_range": [10.0, 20.0], "modality": "visual", "source_lineage": [],
        },),
    }
    first = reasoner.decide(**kwargs)
    second = reasoner.decide(**kwargs)
    assert first.support_status == "insufficient"
    assert second.support_status == "insufficient"
    assert len(api.calls) == 3


class ScriptedVisionClient:
    def __init__(self, responses: Sequence[Any]) -> None:
        self.responses = [dict(item) if isinstance(item, Mapping) else str(item) for item in responses]
        self.calls: list[dict[str, Any]] = []

    def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
        self.calls.append({"prompt": prompt, "image_paths": tuple(image_paths), "max_tokens": max_tokens})
        response = self.responses.pop(0)
        return json.dumps(response) if isinstance(response, Mapping) else str(response)


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="interactive",
        segments=(
            VirtualVideoSegment(
                segment_id="seg_0001",
                source_video_id="source",
                source_path="source.mp4",
                source_start_sec=0.0,
                source_end_sec=180.0,
                virtual_start_sec=0.0,
                virtual_end_sec=180.0,
                role="content",
            ),
        ),
    )
    case = VirtualVideoCase(
        case_id="interactive",
        question="What number appears on the board?",
        options={"A": "7", "B": "9"},
        gold="B",
        target_segment_id="seg_0001",
        target_virtual_interval=(40.0, 60.0),
    )
    workspace = VirtualVideoWorkspace.create(tmp_path / "workspace", manifest=manifest, case=case)
    workspace.write_asr_virtual_cues(
        (
            {"start_sec": 5.0, "end_sec": 7.0, "text": "an unrelated introduction", "segment_id": "seg_0001"},
            {"start_sec": 48.0, "end_sec": 52.0, "text": "the number on the board is discussed", "segment_id": "seg_0001"},
        )
    )
    thumbnail = workspace.root_dir / "beat.jpg"
    Image.new("RGB", (64, 36), color=(80, 80, 80)).save(thumbnail)
    (workspace.root_dir / "beat_index.json").write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "beat_id": "bt0001",
                        "virtual_time_range": [0.0, 180.0],
                        "thumbnail_grid_path": str(thumbnail),
                        "thumbnail_grid_paths": [str(thumbnail)],
                        "asr_cues": workspace.read_asr_virtual_cues(),
                        "source_lineage": [
                            {
                                "segment_id": "seg_0001",
                                "source_video_id": "source",
                                "source_time_range": [0.0, 180.0],
                                "virtual_time_range": [0.0, 180.0],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return workspace


def test_uniform_item_selection_spans_the_complete_input() -> None:
    selected = virtual_video.select_uniform_items(tuple(range(64)), 16)

    assert len(selected) == 16
    assert selected[0] == 0
    assert selected[-1] == 63
    assert selected != tuple(range(16))


def test_window_choice_clusters_all_asr_hits_before_selecting() -> None:
    task = InvestigationTask(
        query_id="q1",
        goal="Find scholars who comment on Napoleon.",
        expected_evidence="scholar comments about Napoleon",
    )
    packet = {
        "virtual_time_range": [0.0, 600.0],
        "asr_cues": [
            {"start_sec": 10.0, "end_sec": 12.0, "text": "Napoleon"},
            {"start_sec": 300.0, "end_sec": 304.0, "text": "a scholar comments on Napoleon"},
            {"start_sec": 309.0, "end_sec": 313.0, "text": "another scholar discusses Napoleon"},
        ],
        "beats": [],
    }

    start, end = _choose_window_from_segment_packet(task, packet)

    assert 290.0 <= start < 305.0
    assert 310.0 < end <= 325.0


def test_invalid_model_window_uses_clustered_fallback(tmp_path: Path) -> None:
    task = InvestigationTask(
        query_id="q1",
        goal="Find scholars who comment on Napoleon.",
        expected_evidence="scholar comments about Napoleon",
    )
    packet = {
        "virtual_time_range": [0.0, 600.0],
        "asr_cues": [
            {"start_sec": 10.0, "end_sec": 12.0, "text": "Napoleon"},
            {"start_sec": 300.0, "end_sec": 304.0, "text": "a scholar comments on Napoleon"},
            {"start_sec": 309.0, "end_sec": 313.0, "text": "another scholar discusses Napoleon"},
        ],
        "beats": [],
    }
    api = ScriptedVisionClient(({},))
    trace_path = tmp_path / "trace.jsonl"

    start, end = _select_window_with_model(api, task, packet, trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert 290.0 <= start < 305.0
    assert 310.0 < end <= 325.0
    assert trace["fallback_used"] is True


def test_model_investigator_uses_preview_then_narrow_uniform_detail(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"start_sec": 0.0, "end_sec": 120.0, "reason": "candidate region"},
            {
                "summary": "The board is present but the number is too small.",
                "confidence": 0.4,
                "need_detail": True,
                "detail_start_sec": 40.0,
                "detail_end_sec": 60.0,
                "requested_fps": 1.0,
                "requested_max_frames": 32,
                "reason": "read the board",
            },
            {
                "summary": "The board shows the number nine beside one presenter.",
                "confidence": 0.95,
                "supports_identity_anchor": False,
                "supports_answer_event": True,
                "entities": [
                    {
                        "local_id": "person_1",
                        "description": "presenter in a dark jacket",
                        "role": "presenter",
                        "question_relation": "stands beside the numbered board",
                        "supports_question_relation": True,
                    }
                ],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Read the number on the board.",
        segment_id="seg_0001",
        modality_hint=("visual", "ocr"),
        expected_evidence="number written on the board",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 3
    preview_call = api.calls[1]
    assert "Preview window metadata" in preview_call["prompt"]
    assert "number on the board is discussed" in preview_call["prompt"]
    assert len(preview_call["image_paths"]) == 16
    assert "frame_0.000.jpg" in preview_call["image_paths"][0]
    assert "frame_120.000.jpg" in preview_call["image_paths"][-1]
    detail_call = api.calls[2]
    assert len(detail_call["image_paths"]) == 20
    assert isinstance(report.evidence[0], EvidenceRecord)
    assert (report.evidence[0].start_sec, report.evidence[0].end_sec) == (40.0, 60.0)
    assert report.evidence[0].sampling_fps == 1.0
    assert report.evidence[0].attestation_model
    assert report.evidence[0].source_lineage[0]["source_video_id"] == "source"
    assert report.evidence[0].evidence_kind == "event_observation"
    assert report.evidence[0].operation_metadata["supports_answer_event"] is True
    assert report.evidence[0].operation_metadata["entities"][0]["local_id"] == "person_1"
    assert len(report.evidence[0].frame_refs) == 20
    assert "model_frames" in report.evidence[0].frame_refs[0]
    assert report.evidence[0].frame_refs[0].endswith("frame_0001.jpg")
    assert report.evidence[0].frame_refs[-1].endswith("frame_0020.jpg")


def test_detail_sampling_request_is_model_selected_and_bounded() -> None:
    assert _interactive._detail_sampling_request(
        {"requested_fps": 1.0, "requested_max_frames": 240}
    ) == (1.0, 240)
    assert _interactive._detail_sampling_request(
        {"requested_fps": 9.0, "requested_max_frames": 900}
    ) == (2.0, 512)
    assert _interactive._detail_sampling_request(
        {"requested_fps": "bad", "requested_max_frames": "bad"},
        default_fps=0.5,
        default_max_frames=64,
    ) == (0.5, 64)


def test_model_frames_are_resized_without_reducing_frame_count(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    sources = []
    for index in range(3):
        path = tmp_path / f"source_{index}.jpg"
        Image.new("RGB", (1280, 720), color=(20 + index, 40, 60)).save(path, quality=95)
        sources.append({"path": str(path), "virtual_time_sec": float(index)})

    rows = _interactive._materialize_model_frames(workspace, "obs_resize", sources)

    assert len(rows) == len(sources)
    assert all(Path(row["path"]).exists() for row in rows)
    assert all(Path(row["path"]) != Path(row["parent_path"]) for row in rows)
    for row in rows:
        with Image.open(row["path"]) as image:
            assert max(image.size) <= 512


def test_detail_window_preserves_temporal_context_for_causal_action() -> None:
    task = InvestigationTask(
        query_id="navigation_repair_r2_002",
        goal="Visually verify the unresolved clue.",
        segment_id="seg_1",
        time_range=(187.429, 226.382),
        modality_hint=("visual",),
        expected_evidence="complete temporal context for the causal action around the dog growl",
    )
    preview = {
        "detail_start_sec": 213.912,
        "detail_end_sec": 217.984,
    }
    segment_packet = {"virtual_time_range": [0.0, 300.0], "asr_cues": [], "beats": []}

    start, end = _select_detail_window(preview, task.time_range, task, segment_packet)

    assert start >= task.time_range[0]
    assert end <= task.time_range[1]
    assert end - start >= 30.0
    assert start < 213.912 < 217.984 < end


def test_investigator_prompt_and_entity_schema_are_question_generic(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Check whether the presenter points to the diagram.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="presenter pointing to the diagram",
    )
    segment_packet = {
        "segment_id": "seg_0001",
        "virtual_time_range": [0.0, 180.0],
        "asr_timeline_summary": "The presenter explains a diagram.",
        "beat_count": 3,
        "thumbnail_grid_paths": [],
    }
    window = {
        "virtual_time_range": [30.0, 60.0],
        "sampling": {"fps": 0.5, "frame_count": 16},
        "asr_cues": [],
        "source_lineage": [],
    }

    preview_prompt = _interactive._preview_prompt(workspace, task, segment_packet, window)
    detail_prompt = _interactive._evidence_prompt(workspace, task, segment_packet, window)
    entities = _interactive._normalize_entities(
        [
            {
                "local_id": "person_1",
                "description": "presenter in a green jacket",
                "role": "presenter",
                "question_relation": "points to the diagram",
                "supports_question_relation": True,
            }
        ]
    )

    assert "scholar" not in preview_prompt.casefold()
    assert "comments_on_topic" not in preview_prompt
    assert "scholar" not in detail_prompt.casefold()
    assert "comments_on_topic" not in detail_prompt
    assert "supports_question_relation" in preview_prompt
    assert '"events"' in preview_prompt
    assert '"events"' in detail_prompt
    assert '"frame_indices"' in preview_prompt
    assert "candidate discovery" in preview_prompt
    assert entities[0]["question_relation"] == "points to the diagram"
    assert entities[0]["supports_question_relation"] is True


def test_entity_normalizer_binds_countable_observation_to_supplied_frames() -> None:
    entities = _interactive._normalize_entities(
        [
            {
                "local_id": "person_1",
                "description": "an older woman in a blue top",
                "visual_signature": "short white hair, blue top, brick wall background",
                "frame_indices": [1, 3, 99],
                "role": "scholar",
                "question_relation": "visibly speaks about Napoleon",
                "supports_question_relation": True,
            }
        ],
        frame_paths=("f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"),
        frame_times=(10.0, 20.0, 30.0, 40.0),
        observation_id="r1_t1_c01",
        window_duration_sec=60.0,
    )

    assert entities[0]["entity_observation_id"] == "r1_t1_c01:person_1"
    assert entities[0]["frame_indices"] == [1, 3]
    assert entities[0]["witness_frame_refs"] == ["f1.jpg", "f3.jpg"]
    assert entities[0]["witness_virtual_times_sec"] == [20.0, 40.0]
    assert entities[0]["countable"] is True
    assert entities[0]["candidate_only"] is False


def test_entity_normalizer_marks_broad_or_unwitnessed_observations_as_candidates() -> None:
    payload = [
        {
            "local_id": "person_1",
            "description": "a man in a dark shirt",
            "visual_signature": "short dark hair and dark shirt",
            "frame_indices": [0],
            "role": "scholar",
            "question_relation": "comments on the topic",
            "supports_question_relation": True,
        }
    ]

    broad = _interactive._normalize_entities(
        payload,
        frame_paths=("frame.jpg",),
        observation_id="broad",
        window_duration_sec=900.0,
    )
    unwitnessed = _interactive._normalize_entities(
        [{**payload[0], "frame_indices": []}],
        frame_paths=("frame.jpg",),
        observation_id="missing",
        window_duration_sec=30.0,
    )

    assert broad[0]["countable"] is False
    assert broad[0]["candidate_reason"] == "coarse_window_candidate"
    assert unwitnessed[0]["countable"] is False
    assert unwitnessed[0]["candidate_reason"] == "missing_frame_witness"


def test_reasoner_payload_normalizes_synonym_actions_from_structured_models() -> None:
    investigate = _interactive._normalize_reasoner_payload(
        {
            "action": "inspect_video_segments",
            "gap": {"gap_id": "gap_entities", "description": "missing entity witnesses"},
            "tasks": [{"query_id": "q1", "goal": "Inspect a narrower window.", "segment_id": "seg_1"}],
        }
    )
    answer = _interactive._normalize_reasoner_payload(
        {"action": "finish", "answer": "B. Three", "citations": ["ev_1"]}
    )

    assert investigate["action"] == "investigate"
    assert investigate["primary_gap"] == investigate["gap"]
    assert answer["action"] == "answer"


def test_reasoner_payload_expands_segment_tasks_and_drops_meta_tasks() -> None:
    payload = _interactive._normalize_reasoner_payload(
        {
            "action": "count_scholars",
            "tasks": [
                {"segments": ["seg_1", "seg_2"], "task": "Inspect each segment for scholar witnesses."},
                {"task": "Deduplicate all scholars and return the answer."},
            ],
        },
        round_id=3,
    )

    assert payload["action"] == "investigate"
    assert [task["segment_id"] for task in payload["tasks"]] == ["seg_1", "seg_2"]
    assert all(task["query_id"].startswith("auto_r3_") for task in payload["tasks"])


def test_empty_answer_audit_is_unknown_not_semantic_insufficiency() -> None:
    empty = _interactive._parse_answer_audit("")
    truncated = _interactive._parse_answer_audit('{"verdict":"insufficient","reason":"missing')

    assert empty["verdict"] == "unknown"
    assert empty["_parseable"] is False
    assert "downgraded" in empty["reason"]
    assert truncated["verdict"] == "insufficient"
    assert truncated["_parseable"] is False


def test_event_normalizer_keeps_only_supported_occurrences_inside_window() -> None:
    events = _interactive._normalize_events(
        [
            {
                "local_id": "event_1",
                "description": "The presenter points to the diagram.",
                "start_sec": 35.0,
                "end_sec": 36.0,
                "supports_question_event": True,
            },
            {
                "local_id": "event_2",
                "description": "An unrelated event outside the inspected window.",
                "start_sec": 100.0,
                "end_sec": 101.0,
                "supports_question_event": True,
            },
            {
                "local_id": "event_3",
                "description": "A visible but question-irrelevant action.",
                "start_sec": 40.0,
                "end_sec": 41.0,
                "supports_question_event": False,
            },
        ],
        (30.0, 60.0),
    )

    assert events == (
        {
            "local_id": "event_1",
            "event_key": "",
            "event_class": "",
                "counting_unit": "",
                "participant_ids": [],
                "participants": [],
                "subject_id": "",
            "object_id": "",
            "state_before": "",
            "transition": "",
                "state_after": "",
                "preconditions_met": None,
                "qualification_status": "",
                "qualification": {},
            "phase": "unknown",
            "description": "The presenter points to the diagram.",
            "start_sec": 35.0,
            "end_sec": 36.0,
            "supports_question_event": True,
            "continues_from_previous": False,
            "continues_to_next": False,
        },
    )


def test_source_only_construction_chunks_the_complete_question_video(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(_interactive, "_duration", lambda dataset_root, video_id: 650.0)

    segments = _interactive._build_source_only_segments(
        tmp_path,
        {"videoID": "source-video"},
        chunk_sec=300.0,
    )

    assert len(segments) == 3
    assert [(segment.source_start_sec, segment.source_end_sec) for segment in segments] == [
        (0.0, 300.0),
        (300.0, 600.0),
        (600.0, 650.0),
    ]
    assert [(segment.virtual_start_sec, segment.virtual_end_sec) for segment in segments] == [
        (0.0, 300.0),
        (300.0, 600.0),
        (600.0, 650.0),
    ]
    assert {segment.source_video_id for segment in segments} == {"source-video"}
    assert {segment.role for segment in segments} == {"target"}


def test_window_selector_samples_beat_thumbnails_across_the_full_segment(tmp_path: Path) -> None:
    task = InvestigationTask(
        query_id="q_full_segment",
        goal="Locate repeated title-card events.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="timestamped title-card occurrences",
    )
    packet = {
        "segment_id": "seg_0001",
        "virtual_time_range": [0.0, 2400.0],
        "asr_timeline_summary": "",
        "beats": [
            {
                "beat_id": f"beat_{index:02d}",
                "virtual_time_range": [float(index * 100), float((index + 1) * 100)],
                "thumbnail_grid_paths": [str(tmp_path / f"beat_{index:02d}.jpg")],
            }
            for index in range(24)
        ],
    }
    api = ScriptedVisionClient(({"start_sec": 1000.0, "end_sec": 1100.0, "reason": "candidate"},))

    _select_window_with_model(api, task, packet, tmp_path / "trace.jsonl")

    image_paths = api.calls[0]["image_paths"]
    assert len(image_paths) == 12
    assert image_paths[0].endswith("beat_00.jpg")
    assert image_paths[-1].endswith("beat_23.jpg")


def test_model_investigator_enumerates_each_beat_within_one_event_count_task(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    thumbnail = workspace.root_dir / "beat.jpg"
    (workspace.root_dir / "beat_index.json").write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "beat_id": f"bt{index:04d}",
                        "virtual_time_range": [float(index * 60), float((index + 1) * 60)],
                        "thumbnail_grid_path": str(thumbnail),
                        "thumbnail_grid_paths": [str(thumbnail)],
                        "asr_cues": [],
                        "source_lineage": [
                            {
                                "segment_id": "seg_0001",
                                "source_video_id": "source",
                                "source_time_range": [float(index * 60), float((index + 1) * 60)],
                                "virtual_time_range": [float(index * 60), float((index + 1) * 60)],
                            }
                        ],
                    }
                    for index in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )
    api = ScriptedVisionClient(
        (
            {
                "summary": "One title card appears in the complete segment.",
                "confidence": 0.95,
                "events": [
                    {
                        "local_id": "event_1",
                        "event_key": "opening title card",
                        "description": "A title card appears.",
                        "start_sec": 82.0,
                        "end_sec": 120.0,
                        "supports_question_event": True,
                        "continues_from_previous": False,
                        "continues_to_next": True,
                    }
                ],
                "supports_answer_event": True,
                "need_detail": False,
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_title_cards",
        goal="Enumerate title-card appearances.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="timestamped title-card occurrences",
        inspection_mode="enumerate_events",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 1
    assert '"entities"' not in api.calls[0]["prompt"]
    assert api.calls[0]["max_tokens"] >= 1000
    assert len(report.evidence) == 1
    assert [(record.start_sec, record.end_sec) for record in report.evidence] == [(0.0, 180.0)]
    assert sum(len(record.operation_metadata["events"]) for record in report.evidence) == 1
    assert report.cost["beat_windows"] == 1

    reused = investigator.run_batch((replace(task, query_id="q_title_cards_repeat"),))[0]

    assert len(api.calls) == 1
    assert reused.cost["reused"] is True
    assert reused.cost["vlm_calls"] == 0


def test_model_investigator_stops_after_sufficient_preview(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"start_sec": 0.0, "end_sec": 120.0, "reason": "candidate region"},
            {"summary": "The visible board clearly shows nine.", "confidence": 0.95, "need_detail": False},
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Check what is visible on the board.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="visible board content",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 2
    assert report.evidence[0].sampling_fps == 0.5
    assert report.cost["tool_trace"] == ("open_segment", "inspect_window:0.5")


def test_narrative_bridge_uses_specialized_schema_only_in_detail_stage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"summary": "The preview locates the story transition.", "confidence": 0.7, "need_detail": True},
            {
                "summary": "Joe changes his plan and stays.",
                "confidence": 0.85,
                "narrative_facts": [{
                    "fact_id": "bridge_1",
                    "subject_id": "joe",
                    "setup_state": "Joe plans to leave.",
                    "outcome_state": "Joe stays.",
                    "inference": "Joe changes his mind.",
                    "confidence": 0.85,
                    "hypothesis_assessments": [{"option_id": "B", "status": "supported"}],
                }],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="narrative_bridge_r1_001",
        goal="Observe the setup and outcome.",
        segment_id="seg_0001",
        time_range=(0.0, 20.0),
        inspection_mode="narrative_bridge",
        expected_evidence="setup and outcome",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 2
    assert "canonical narrative bridge" not in api.calls[0]["prompt"]
    assert "canonical narrative bridge" in api.calls[1]["prompt"]
    assert report.evidence[0].operation_metadata["narrative_facts"][0]["fact_id"] == "bridge_1"


def test_entity_association_uses_reference_in_detail_stage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"summary": "A possible racer is visible.", "confidence": 0.7, "need_detail": True},
            {
                "summary": "The later racer matches the reference.",
                "confidence": 0.9,
                "entities": [{
                    "local_id": "target_1",
                    "description": "A racer in a green jacket and black helmet.",
                    "visual_signature": "green jacket; black helmet",
                    "frame_indices": [0],
                    "attributes": {"clothing_color": "green"},
                    "supports_question_relation": True,
                }],
                "entity_associations": [{
                    "source_participant_id": "second racer",
                    "target_local_id": "target_1",
                    "entity_hypothesis_id": "second_racer_anchor",
                    "status": "supported",
                    "confidence": 0.9,
                    "shared_attributes": {"jacket": "green", "helmet": "black"},
                }],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="participant_link_r1_001",
        goal="Re-identify the second racer.",
        segment_id="seg_0001",
        time_range=(0.0, 20.0),
        inspection_mode="entity_association",
        expected_evidence="cross-window association",
        reference_entities=({
            "participant_id": "second racer",
            "entity_hypothesis_id": "second_racer_anchor",
            "source_event_key": "second overtake",
        },),
    )

    report = investigator.run_batch((task,))[0]

    assert "targeted cross-window entity association" in api.calls[1]["prompt"]
    assert report.evidence[0].operation_metadata["entity_associations"][0]["status"] == "supported"


def test_anchor_event_window_uses_compact_schema_and_keeps_anchor(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"summary": "An overtake may occur here.", "confidence": 0.7, "need_detail": True},
            {
                "summary": "A green racer overtakes the recorder.",
                "confidence": 0.9,
                "events": [{
                    "event_key": "green racer overtakes recorder",
                    "description": "A green racer overtakes the recorder.",
                    "start_sec": 10.0,
                    "end_sec": 11.0,
                    "supports_anchor_event": True,
                    "participant_ids": ["recorder", "green racer"],
                    "participants": [{
                        "participant_id": "green racer",
                        "role": "overtaker",
                        "visual_signature": "green jacket; black helmet",
                        "attributes": {"clothing_color": "green", "helmet_color": "black"},
                    }],
                }],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="participant_anchor_r1_001",
        goal="Locate the overtaking anchor.",
        segment_id="seg_0001",
        time_range=(0.0, 20.0),
        inspection_mode="event_window",
        inspection_intent="event_participant_anchor_discovery",
        expected_evidence="anchor event and participants",
    )

    report = investigator.run_batch((task,))[0]

    assert "event that identifies a participant" in api.calls[1]["prompt"]
    assert "state_before" not in api.calls[1]["prompt"]
    assert report.evidence[0].operation_metadata["events"][0]["participant_ids"] == [
        "recorder", "green racer",
    ]


def test_model_investigator_adapts_not_found_across_fps_and_phase(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"summary": "Target absent on the first grid.", "finding": "not_found", "confidence": 0.9},
            {"summary": "Target absent on the shifted grid.", "finding": "not_found", "confidence": 0.9},
            {
                "summary": "The brief target is visible after densification.",
                "finding": "found",
                "confidence": 0.95,
                "structured_slots": [{"slot_id": "target_state", "value": "visible", "frame_indices": [0]}],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_adaptive",
        goal="Locate the requested brief visible evidence.",
        segment_id="seg_0001",
        time_range=(0.0, 30.0),
    )

    report = investigator.run_batch((task,))[0]
    policy = report.evidence[0].operation_metadata["sampling_policy"]

    assert report.cost["tool_trace"] == (
        "open_segment",
        "inspect_window:0.5",
        "inspect_window:1.0:phase=0.500",
        "inspect_window:2.0:phase=0.333",
    )
    assert report.evidence[0].sampling_fps == 2.0
    assert report.evidence[0].observation_polarity == "positive"
    assert policy["adaptive_attempt_count"] == 3
    assert policy["adaptive_trigger_reasons"] == ["not_found"]


def test_model_investigator_caps_repeated_negative_window_at_three_attempts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        tuple(
            {
                "summary": f"Negative phase {index}.", "finding": "not_found", "confidence": 0.9,
                "visibility_status": "clear",
            }
            for index in range(3)
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_negative_cap", goal="Locate the requested brief evidence.",
        segment_id="seg_0001", time_range=(0.0, 30.0), expected_event_dwell_sec=2.0,
    )

    first = investigator.run_batch((task,))[0]
    repeated = investigator.run_batch((replace(task, query_id="q_negative_cap_repeat"),))[0]

    assert len(api.calls) == 3
    assert first.evidence[0].operation_metadata["qualified_absence"] is True
    assert repeated.cost["reused"] is True
    assert repeated.cost["tool_trace"] == ("open_segment", "adaptive_attempt_cap")


def test_model_investigator_blocks_unresolved_structured_slot_conflict(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {
                "summary": "The jacket appears blue.", "finding": "found", "confidence": 0.9,
                "structured_slots": [{"slot_id": "jacket_color", "value": "blue", "frame_indices": [0]}],
            },
            {
                "summary": "The jacket appears black.", "finding": "found", "confidence": 0.9,
                "structured_slots": [{"slot_id": "jacket_color", "value": "black", "frame_indices": [0]}],
            },
            {
                "summary": "The denser pass cannot settle the color.", "finding": "uncertain", "confidence": 0.8,
                "structured_slots": [{"slot_id": "jacket_color", "value": "black", "frame_indices": [0]}],
                "slot_arbitration": [{"slot_id": "jacket_color", "verdict": "cannot_determine", "frame_indices": [0]}],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    base = InvestigationTask(
        query_id="q_color_a", goal="Observe the jacket color.", segment_id="seg_0001",
        time_range=(0.0, 30.0), sampling_floor_fps=0.5,
        temporal_resolution_rationale="The clothing remains visible for several seconds.",
    )
    denser = replace(base, query_id="q_color_b", sampling_floor_fps=1.0)

    first, second = investigator.run_batch((base, denser))

    assert first.evidence[0].operation_metadata["evidence_state"] == "active"
    assert second.evidence[0].operation_metadata["evidence_state"] == "conflicted"
    assert second.evidence[0].operation_metadata["conflicted_slot_ids"] == ["jacket_color"]
    assert "structured_slot_conflict" in second.evidence[0].operation_metadata["sampling_policy"]["adaptive_trigger_reasons"]


def test_observation_local_slots_at_different_times_do_not_conflict() -> None:
    rows = (
        {"structured_slots": _interactive._normalize_structured_slots([
            {"slot_id": "joe_pose", "slot_type": "observation_attribute", "entity_id": "joe", "attribute": "pose", "value": "sitting", "time_scope": [10.0, 12.0]},
        ])},
        {"structured_slots": _interactive._normalize_structured_slots([
            {"slot_id": "joe_pose", "slot_type": "observation_attribute", "entity_id": "joe", "attribute": "pose", "value": "standing", "time_scope": [20.0, 22.0]},
        ])},
    )

    assert _interactive._conflicted_slot_ids(rows) == ()


def test_structured_slot_placeholders_are_discarded() -> None:
    slots = _interactive._normalize_structured_slots([
        {"slot_id": "joe_pose", "value": "observable unknown"},
        {"slot_id": "joe_location", "value": "placeholder"},
    ])

    assert slots == {}


def test_entity_hypothesis_fields_survive_observation_normalization() -> None:
    entities = _interactive._normalize_entities(
        [{
            "local_id": "racer_1",
            "entity_hypothesis_id": "Racer Green 01",
            "association_confidence": 0.82,
            "description": "A racer in a green jacket and black helmet.",
            "visual_signature": "green jacket; black helmet",
            "attributes": {"jacket_color": "green", "helmet_color": "black"},
            "frame_indices": [0],
            "supports_question_relation": True,
        }],
        frame_paths=("frame.jpg",),
        frame_times=(12.0,),
        observation_id="obs_1",
        window_duration_sec=2.0,
    )

    assert entities[0]["entity_hypothesis_id"] == "racer_green_01"
    assert entities[0]["association_confidence"] == 0.82
    assert entities[0]["attributes"]["jacket_color"] == "green"


def test_event_participant_hypothesis_and_role_survive_normalization() -> None:
    events = _interactive._normalize_events(
        [{
            "event_key": "green racer overtakes recorder",
            "event_class": "other",
            "counting_unit": "overtake_episode",
            "description": "A green-jacket racer overtakes the recorder.",
            "start_sec": 10.0,
            "end_sec": 11.0,
            "supports_question_event": True,
            "participants": [{
                "participant_id": "green racer",
                "entity_hypothesis_id": "Racer Green 01",
                "role": "overtaker",
                "visual_signature": "green jacket; black helmet",
                "attributes": {"jacket_color": "green", "helmet_color": "black"},
                "association_confidence": 0.84,
            }],
        }],
        (9.0, 12.0),
    )

    participant = events[0]["participants"][0]
    assert participant["entity_hypothesis_id"] == "racer_green_01"
    assert participant["role"] == "overtaker"
    assert "green racer" in events[0]["participant_ids"]


def test_anchor_discovery_accepts_anchor_event_without_relabeling_normal_tasks() -> None:
    raw = [{
        "event_key": "second racer overtakes recorder",
        "description": "The second racer overtakes the recorder.",
        "start_sec": 10.0,
        "end_sec": 11.0,
        "supports_question_event": False,
        "supports_anchor_event": True,
    }]

    assert _interactive._normalize_events(raw, (9.0, 12.0)) == ()
    anchor = _interactive._normalize_events(raw, (9.0, 12.0), anchor_discovery=True)

    assert anchor[0]["supports_question_event"] is True


def test_entity_association_requires_exact_reference_and_two_shared_attributes() -> None:
    entities = ({
        "local_id": "target_1",
        "entity_observation_id": "obs:target_1",
        "attributes": {"clothing_color": "green"},
    },)
    reference = ({
        "participant_id": "second racer",
        "entity_hypothesis_id": "second_racer_anchor",
        "source_event_key": "second racer overtakes recorder",
    },)
    rows = _interactive._normalize_entity_associations(
        [{
            "source_participant_id": "second racer",
            "target_local_id": "target_1",
            "entity_hypothesis_id": "second_racer_anchor",
            "status": "supported",
            "confidence": 0.88,
            "shared_attributes": {"jacket": "green", "helmet": "black"},
            "distinguishing_attributes": {"clothing_color": "green"},
        }],
        reference_entities=reference,
        entities=entities,
    )
    weak = _interactive._normalize_entity_associations(
        [{
            "source_participant_id": "second racer",
            "target_local_id": "target_1",
            "entity_hypothesis_id": "second_racer_anchor",
            "status": "supported",
            "confidence": 0.88,
            "shared_attributes": {"jacket": "green"},
        }],
        reference_entities=reference,
        entities=entities,
    )

    assert rows[0]["status"] == "supported"
    assert rows[0]["target_entity_observation_id"] == "obs:target_1"
    assert weak[0]["status"] == "unknown"
    assert weak[0]["confidence"] == 0.59


def test_incomplete_narrative_fact_cannot_support_an_option() -> None:
    facts = _interactive._normalize_narrative_facts(
        [{
            "fact_id": "bridge_1",
            "setup_state": "Joe plans to leave.",
            "outcome_state": "",
            "inference": "Joe changes his mind.",
            "confidence": 0.9,
            "hypothesis_assessments": [{"option_id": "B", "status": "supported", "reason": "possible"}],
        }],
        options={"A": "He leaves.", "B": "He stays."},
    )

    assert facts[0]["confidence"] == 0.59
    assert facts[0]["hypothesis_assessments"][0]["status"] == "unknown"


def test_negative_observation_is_not_reused_as_positive_support(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = GeminiInvestigator(
        workspace,
        api=ScriptedVisionClient(()),
        trace_path=workspace.root_dir / "interactions.jsonl",
    )
    task = InvestigationTask(
        query_id="q_negative_cache", goal="Locate the requested target.", segment_id="seg_0001",
        time_range=(0.0, 30.0),
    )
    negative = EvidenceRecord(
        evidence_id="ev_negative", beat_id="", start_sec=0.0, end_sec=30.0,
        modality="visual", pointer="virtual://negative", verbatim="Target not found.",
        frame_refs=("frame.jpg",), observation_polarity="negative", sampling_fps=2.0,
        source_lineage=({
            "segment_id": "seg_0001", "source_video_id": "source",
            "source_time_range": [0.0, 30.0], "virtual_time_range": [0.0, 30.0],
        },),
    )
    investigator._remember_evidence(task, negative)

    reused = investigator._find_reusable_evidence(task, 0.0, 30.0, required_fps=0.5)

    assert reused is None


def test_model_investigator_does_not_force_detail_from_keywords(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        ({"summary": "The number is already readable.", "confidence": 0.95, "need_detail": False},)
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_number",
        goal="Read the number and track the visible transition.",
        segment_id="seg_0001",
        time_range=(0.0, 30.0),
        modality_hint=("visual", "ocr", "motion"),
        expected_evidence="visible number and transition",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 1
    assert report.cost["detail_frames"] == 0
    assert report.evidence[0].sampling_fps == 0.5


def test_conditional_event_requires_explicit_preconditions(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {
                "summary": "Two visually similar passes occur, but only one follows the required lead state.",
                "confidence": 0.9,
                "need_detail": False,
                "events": [
                    {
                        "local_id": "event_1",
                        "event_key": "qualified pass",
                        "subject_id": "camera racer",
                        "object_id": "yellow racer",
                        "state_before": "camera racer is leading",
                        "transition": "yellow racer passes camera racer",
                        "state_after": "camera racer is behind yellow racer",
                        "preconditions_met": True,
                        "description": "The yellow racer overtakes the leading camera racer.",
                        "start_sec": 10.0,
                        "end_sec": 12.0,
                        "supports_question_event": True,
                    },
                    {
                        "local_id": "event_2",
                        "event_key": "nonqualifying pass",
                        "subject_id": "camera racer",
                        "object_id": "black racer",
                        "state_before": "camera racer is already behind",
                        "transition": "black racer passes camera racer",
                        "state_after": "camera racer remains behind",
                        "preconditions_met": False,
                        "description": "A later pass occurs without the required lead state.",
                        "start_sec": 20.0,
                        "end_sec": 22.0,
                        "supports_question_event": True,
                    },
                ],
                "supports_answer_event": True,
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_conditional_count",
        goal="Count overtakes after the tracked subject reaches the lead.",
        segment_id="seg_0001",
        modality_hint=("visual", "motion"),
        inspection_mode="enumerate_events",
    )

    report = investigator.run_batch((task,))[0]
    events = report.evidence[0].operation_metadata["events"]

    assert len(events) == 2
    assert events[0]["event_key"] == "qualified pass"
    assert events[0]["preconditions_met"] is True
    assert events[1]["event_key"] == "nonqualifying pass"
    assert events[1]["preconditions_met"] is False
    assert events[1]["supports_question_event"] is False
    assert events[1]["qualification_status"] == "unqualified_precondition"


def test_model_investigator_preserves_candidate_provenance(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(({"summary": "A dog chases a man.", "confidence": 0.9, "need_detail": False},))
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_candidate", goal="Inspect this candidate.", segment_id="seg_0001", time_range=(0.0, 10.0),
        source_candidate_ids=("ev_candidate",), inspection_intent="verify dog/father hypothesis",
    )
    report = investigator.run_batch((task,))[0]
    metadata = report.evidence[0].operation_metadata
    assert metadata["source_candidate_ids"] == ["ev_candidate"]
    assert metadata["inspection_intent"] == "verify dog/father hypothesis"


def test_model_investigator_repairs_truncated_structured_observation_once(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient((
        '```json\n{"summary":"He applies lipstick, creates spots, then leaves on a stretcher",',
        {
            "summary": "He applies lipstick, creates spots, then leaves on a stretcher.",
            "confidence": 0.9, "need_detail": False,
            "events": [
                {"local_id": "e1", "description": "applies lipstick", "start_sec": 1, "end_sec": 2, "supports_question_event": True},
                {"local_id": "e2", "description": "leaves on a stretcher", "start_sec": 3, "end_sec": 4, "supports_question_event": True},
            ],
            "supports_answer_event": True,
        },
    ))
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_truncated", goal="Inspect the visible sequence.", segment_id="seg_0001",
        time_range=(0.0, 10.0), modality_hint=("visual",),
    )
    report = investigator.run_batch((task,))[0]
    assert len(api.calls) == 2
    metadata = report.evidence[0].operation_metadata
    assert metadata["structured_parse_status"] == "repaired"
    assert metadata["structured_parse_error"]
    assert len(metadata["events"]) == 2
    assert report.cost["vlm_calls"] == 2


def test_model_investigator_recovers_closed_summary_when_json_repair_stays_truncated(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient((
        '{"summary":"A complete explicit observation.","events":[',
        '```json\n{"summary":"A complete explicit observation.","events":[',
    ))
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_fallback", goal="Inspect the visible sequence.", segment_id="seg_0001",
        time_range=(0.0, 10.0), modality_hint=("visual",),
    )
    report = investigator.run_batch((task,))[0]
    metadata = report.evidence[0].operation_metadata
    assert metadata["structured_parse_status"] == "fallback_extracted"
    assert report.evidence[0].verbatim == "A complete explicit observation."


def test_model_investigator_falls_back_to_explicit_measurement_text(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace = VirtualVideoWorkspace.create(
        workspace.root_dir,
        manifest=workspace.manifest,
        case=_interactive.VirtualVideoCase(
            case_id="measurement", question="What total diameter is stated in light-years?",
            options={"A": "23 trillion", "B": "30 quintillion"}, gold="B",
            target_segment_id="seg_0001", target_virtual_interval=(0.0, 10.0),
        ),
    )
    api = ScriptedVisionClient((
        {"summary": "The stated diameter is 30 quintillion light-years.", "confidence": 0.9, "need_detail": False},
    ))
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="q_measurement", goal="Inspect the stated diameter display.", segment_id="seg_0001",
        time_range=(0.0, 10.0), modality_hint=("visual",),
    )
    report = investigator.run_batch((task,))[0]
    metadata = report.evidence[0].operation_metadata
    assert metadata["structured_parse_status"] == "fallback_extracted"
    assert metadata["measurements"][0]["value"] == 30_000_000_000_000_000_000
    assert metadata["measurements"][0]["quantity_type"] == "diameter"
    assert metadata["measurements"][0]["binding_status"] == "explicit"


def test_model_investigator_reports_partial_gap_instead_of_frame_success(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {
                "summary": "The board is visible, but the final value is unreadable.",
                "confidence": 0.6,
                "need_detail": False,
                "resolution": "resolved",
                "target_presence": {"target": "final board", "status": "present", "confidence": 0.8},
                "condition_results": [
                    {
                        "condition_id": "gap_final_value_c1",
                        "status": "satisfied",
                        "observation": "The final board is visible.",
                    },
                    {
                        "condition_id": "gap_final_value_c2",
                        "status": "unknown",
                        "observation": "The value remains unreadable.",
                    },
                ],
                "failure_reason": "text is too small",
            },
            {
                "summary": "The detail frames still do not make the final value readable.",
                "confidence": 0.6,
                "resolution": "resolved",
                "target_presence": {"target": "final board", "status": "present", "confidence": 0.9},
                "condition_results": [
                    {
                        "condition_id": "gap_final_value_c1",
                        "status": "satisfied",
                        "observation": "The final board is visible.",
                    },
                    {
                        "condition_id": "gap_final_value_c2",
                        "status": "unknown",
                        "observation": "No value or unit can be read.",
                    },
                ],
                "failure_reason": "text remains unreadable after detail inspection",
            },
            {
                "summary": "A final phase-shifted pass still cannot resolve the value.",
                "confidence": 0.6,
                "finding": "uncertain",
                "target_presence": {"target": "final board", "status": "present", "confidence": 0.9},
                "condition_results": [
                    {"condition_id": "gap_final_value_c1", "status": "satisfied", "observation": "Board visible."},
                    {"condition_id": "gap_final_value_c2", "status": "unknown", "observation": "Value unreadable."},
                ],
                "failure_reason": "text remains unreadable after adaptive sampling",
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_gap",
        goal="Read the final value and unit.",
        segment_id="seg_0001",
        time_range=(0.0, 30.0),
        modality_hint=("visual",),
        gap_id="gap_final_value",
        success_conditions=("locate the final board", "read the final value and unit"),
    )

    report = investigator.run_batch((task,))[0]

    assert report.status == "satisfied"
    assert report.resolution == "partial"
    assert report.resolved_conditions == ("locate the final board",)
    assert report.unresolved_conditions == ("read the final value and unit",)
    assert report.failure_reason == "text remains unreadable after adaptive sampling"
    assert len(api.calls) == 3
    assert report.evidence[0].operation_metadata["investigation"]["resolution"] == "partial"
    assert report.condition_results[0].condition_id == "gap_final_value_c1"
    assert report.goal_progress == ("condition_satisfied:gap_final_value_c1",)


def test_model_investigator_materializes_region_crops_inside_inspect_window(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {
                "summary": "A small scoreboard is visible in the upper-left corner.",
                "confidence": 0.7,
                "need_detail": True,
                "detail_start_sec": 4.0,
                "detail_end_sec": 10.0,
                "region_hint": "scoreboard clock and scores",
                "region_box": [0.0, 0.0, 0.55, 0.45],
                "resolution": "partial",
                "unresolved_conditions": ["read the displayed clock and scores"],
            },
            {
                "summary": "The enlarged region makes the displayed clock and scores readable.",
                "confidence": 0.9,
                "resolution": "resolved",
                "target_presence": {"target": "scoreboard", "status": "present", "confidence": 0.95},
                "measurements": [
                    {"value": 8.13, "unit": "game_clock", "raw_text": "8:13"},
                    {"value": 10, "unit": "point", "subject_id": "home"},
                    {"value": 10, "unit": "point", "subject_id": "guest"},
                ],
                "condition_results": [
                    {
                        "condition_id": "gap_scoreboard_c1",
                        "status": "satisfied",
                        "observation": "The scoreboard reads 8:13 and 10-10.",
                    }
                ],
            },
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_scoreboard",
        goal="Read the displayed clock and scores.",
        segment_id="seg_0001",
        time_range=(0.0, 30.0),
        modality_hint=("visual", "ocr"),
        gap_id="gap_scoreboard",
        success_conditions=("read the displayed clock and scores",),
        region_hint="scoreboard",
    )

    report = investigator.run_batch((task,))[0]

    assert len(api.calls) == 2
    assert any("/observations/regions/" in path for path in api.calls[1]["image_paths"])
    assert "gap_scoreboard_c1" in api.calls[1]["prompt"]
    assert "target_presence" in api.calls[1]["prompt"]
    assert "measurements" in api.calls[1]["prompt"]
    assert "ordered_image_groups" in api.calls[1]["prompt"]
    assert report.cost["region_frames"] > 0
    assert report.resolution == "resolved"
    assert "region_observed" in report.progress_flags
    region = report.evidence[0].operation_metadata["region_observation"]
    assert region["normalized_box"] == [0.0, 0.0, 0.55, 0.45]
    assert {row["region_kind"] for row in region["frames"]} == {"model_box", "coarse_tile"}
    assert report.evidence[0].operation_metadata["target_presence"]["status"] == "present"
    assert len(report.evidence[0].operation_metadata["measurements"]) == 3


def test_repeated_query_ids_get_distinct_observation_and_frame_ids(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"start_sec": 0.0, "end_sec": 30.0},
            {"summary": "First pass.", "finding": "found", "confidence": 0.8, "need_detail": False},
            {"start_sec": 0.0, "end_sec": 30.0},
            {"summary": "Second pass.", "finding": "found", "confidence": 0.8, "need_detail": False},
        )
    )
    investigator = GeminiInvestigator(workspace, api=api, trace_path=workspace.root_dir / "interactions.jsonl")
    investigator.sampler = _sampler
    task = InvestigationTask(
        query_id="r1_t1",
        goal="Inspect the visible board.",
        segment_id="seg_0001",
        modality_hint=("visual",),
        expected_evidence="visible board content",
    )

    reports = investigator.run_batch((task, task))
    manifest_rows = [
        json.loads(line)
        for line in (workspace.root_dir / "observations" / "window_frame_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert reports[0].evidence[0].evidence_id == "ev_r1_t1_c01_001"
    assert reports[1].evidence[0].evidence_id == reports[0].evidence[0].evidence_id
    assert reports[1].cost["reused"] is True
    assert len(api.calls) == 3
    frame_ids = [row["frame_id"] for row in manifest_rows]
    assert len(frame_ids) == len(set(frame_ids))
    ledger_rows = [
        json.loads(line)
        for line in (workspace.root_dir / "exploration_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ledger_rows[-1]["reused_from"] == reports[0].evidence[0].evidence_id


def test_trace_viewer_renders_every_reasoner_round(tmp_path: Path) -> None:
    workspace = tmp_path / "run" / "workspaces" / "case-1"
    workspace.mkdir(parents=True)
    (workspace / "case.json").write_text(
        json.dumps({"question": "Q?", "options": {"A": "one"}, "gold": "A"}), encoding="utf-8"
    )
    (workspace / "virtual_timeline.json").write_text(
        json.dumps({"duration_sec": 120.0, "segments": []}), encoding="utf-8"
    )
    (workspace / "run_summary.json").write_text(
        json.dumps({"answer": "A. one", "correct": True, "rounds": 2, "accepted_investigations": 2, "evidence": []}),
        encoding="utf-8",
    )
    (workspace / "beat_index.json").write_text(json.dumps({"beats": []}), encoding="utf-8")
    (workspace / "evidence.jsonl").write_text(
        json.dumps({"evidence_id": "ev_r1_t1_001", "modality": "visual", "sampling_fps": 0.5}) + "\n",
        encoding="utf-8",
    )
    (workspace / "exploration_ledger.jsonl").write_text(
        json.dumps({"visit_id": "visit_0001", "status": "reused", "reused_from": "ev_r1_t1_001"}) + "\n",
        encoding="utf-8",
    )
    trace = (
        {
            "type": "reasoner_investigate",
            "round": 1,
            "prompt": "round one prompt",
            "raw": "round one raw",
            "parsed": {"tasks": [{"query_id": "r1_t1", "segment_id": "seg_0001"}]},
        },
        {"type": "reasoner_investigate", "round": 2, "prompt": "round two prompt", "raw": "round two raw", "parsed": {"tasks": [{"query_id": "r2_t1", "segment_id": "seg_0002"}] }},
        {"type": "reasoner_answer", "round": 3, "prompt": "answer prompt", "raw": "answer raw", "parsed": {"answer": "A. one"}},
    )
    (workspace / "interactions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8"
    )

    assets = tmp_path / "viewer" / "assets"
    bundle = AssetBundler(run_root=tmp_path / "run", assets_dir=assets, case_id="case-1")
    html, _ = _render_case(workspace, bundle)

    assert "Reasoner Round 1" in html
    assert "Reasoner Round 2" in html
    assert "r1_t1" in html
    assert "r2_t1" in html
    assert "round one raw" in html
    assert "round two raw" in html
    assert "Structured Evidence Store" in html
    assert "Exploration Ledger" in html
    assert "visit_0001" in html
    assert "reused_from" in html


def test_trace_viewer_light_bundle_omits_raw_frames(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frame_path = workspace.root_dir / "frame.jpg"
    Image.new("RGB", (64, 36), color=(80, 90, 100)).save(frame_path)
    observations = workspace.root_dir / "observations"
    observations.mkdir()
    (observations / "window_frame_manifest.jsonl").write_text(
        json.dumps(
            {
                "frame_id": "frame_1",
                "path": str(frame_path),
                "virtual_time_sec": 10.0,
                "segment_id": "seg_0001",
                "source_video_id": "source",
                "source_path": "source.mp4",
                "source_time_sec": 10.0,
                "fps_level": "low",
                "sampling_fps": 0.5,
                "query_id": "q1_preview",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace.root_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    (workspace.root_dir / "interactions.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "type": "reasoner_investigate",
                    "round": 1,
                    "parsed": {"tasks": [{"query_id": "q1", "segment_id": "seg_0001"}]},
                },
                {"type": "investigator_preview", "query_id": "q1", "preview_query_id": "q1_preview"},
            )
        ),
        encoding="utf-8",
    )
    assets = tmp_path / "viewer" / "assets"
    bundle = AssetBundler(run_root=tmp_path, assets_dir=assets, case_id=workspace.root_dir.name)

    html, _ = _render_case(workspace.root_dir, bundle, light=True)

    assert "images omitted in light bundle" in html
    assert not any(assets.rglob("frame.jpg"))
