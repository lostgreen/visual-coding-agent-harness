from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
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
_matching_claim_assessment = _interactive._matching_claim_assessment
_followup_prompt = _interactive._followup_prompt


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
        query_requirements={},
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


def test_gemini_reasoner_preserves_verdict_from_truncated_answer_audit(tmp_path: Path) -> None:
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
        query_contract={"required_scope": "window", "aggregation": "none"},
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
    assert decision.support_status == "contradicted"
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
        "query_contract": {"required_scope": "full_video", "aggregation": "deduplicate"},
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


def test_gemini_reasoner_adopts_directly_supported_revised_answer_from_audit(tmp_path: Path) -> None:
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
        query_contract={"required_scope": "full_video", "aggregation": "deduplicate"},
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
    assert decision.answer == "D. Six."
    assert decision.citations == ("ev_1", "ev_2")
    assert decision.support_status == "supported"


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
        query_requirements={},
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
        query_requirements={},
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


def test_answer_audit_targets_relation_risk_without_rechecking_simple_quantities() -> None:
    assert _should_audit_answer(
        {
            "question": "Why did the person give away the money?",
            "query_contract": {"required_scope": "window", "aggregation": "none"},
            "query_requirements": {},
        }
    )
    assert _should_audit_answer(
        {
            "question": "How many people were interviewed in total?",
            "query_contract": {"required_scope": "full_video", "aggregation": "deduplicate"},
            "query_requirements": {},
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


class ScriptedVisionClient:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self.responses = [dict(item) for item in responses]
        self.calls: list[dict[str, Any]] = []

    def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 900) -> str:
        self.calls.append({"prompt": prompt, "image_paths": tuple(image_paths), "max_tokens": max_tokens})
        return json.dumps(self.responses.pop(0))


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
    assert isinstance(report.evidence[0], EvidenceRecord)
    assert (report.evidence[0].start_sec, report.evidence[0].end_sec) == (40.0, 60.0)
    assert report.evidence[0].sampling_fps == 2.0
    assert report.evidence[0].attestation_model
    assert report.evidence[0].source_lineage[0]["source_video_id"] == "source"
    assert report.evidence[0].evidence_kind == "event_observation"
    assert report.evidence[0].operation_metadata["supports_answer_event"] is True
    assert report.evidence[0].operation_metadata["entities"][0]["local_id"] == "person_1"
    assert len(report.evidence[0].frame_refs) == 16
    assert "frame_40.000.jpg" in report.evidence[0].frame_refs[0]
    assert "frame_60.000.jpg" in report.evidence[0].frame_refs[-1]


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
    assert entities[0]["question_relation"] == "points to the diagram"
    assert entities[0]["supports_question_relation"] is True


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
                "summary": "No title card appears in the first minute.",
                "confidence": 0.9,
                "events": [],
                "supports_answer_event": False,
                "need_detail": False,
            },
            {
                "summary": "A title card appears in the second minute.",
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
            {
                "summary": "No title card appears in the final minute.",
                "confidence": 0.9,
                "events": [],
                "supports_answer_event": False,
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

    assert len(api.calls) == 3
    assert '"entities"' not in api.calls[0]["prompt"]
    assert api.calls[0]["max_tokens"] >= 1000
    assert "Prior adjacent-window ending events" in api.calls[2]["prompt"]
    assert "opening title card" in api.calls[2]["prompt"]
    assert len(report.evidence) == 3
    assert [(record.start_sec, record.end_sec) for record in report.evidence] == [
        (0.0, 60.0),
        (60.0, 120.0),
        (120.0, 180.0),
    ]
    assert sum(len(record.operation_metadata["events"]) for record in report.evidence) == 1
    assert report.cost["beat_windows"] == 3


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


def test_model_investigator_reports_partial_gap_instead_of_frame_success(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {
                "summary": "The board is visible, but the final value is unreadable.",
                "confidence": 0.6,
                "need_detail": False,
                "resolution": "partial",
                "resolved_conditions": ["locate the final board"],
                "unresolved_conditions": ["read the final value and unit"],
                "failure_reason": "text is too small",
            },
            {
                "summary": "The detail frames still do not make the final value readable.",
                "confidence": 0.6,
                "resolution": "partial",
                "resolved_conditions": ["locate the final board"],
                "unresolved_conditions": ["read the final value and unit"],
                "failure_reason": "text remains unreadable after detail inspection",
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
    assert report.failure_reason == "text remains unreadable after detail inspection"
    assert report.evidence[0].operation_metadata["investigation"]["resolution"] == "partial"


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
                "resolved_conditions": ["read the displayed clock and scores"],
                "unresolved_conditions": [],
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
    assert report.cost["region_frames"] > 0
    assert report.resolution == "resolved"
    assert "region_observed" in report.progress_flags
    region = report.evidence[0].operation_metadata["region_observation"]
    assert region["normalized_box"] == [0.0, 0.0, 0.55, 0.45]


def test_repeated_query_ids_get_distinct_observation_and_frame_ids(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    api = ScriptedVisionClient(
        (
            {"start_sec": 0.0, "end_sec": 30.0},
            {"summary": "First pass.", "confidence": 0.8, "need_detail": False},
            {"start_sec": 0.0, "end_sec": 30.0},
            {"summary": "Second pass.", "confidence": 0.8, "need_detail": False},
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
