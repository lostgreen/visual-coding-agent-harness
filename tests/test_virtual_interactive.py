from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pytest

from vcah.interactive_agents import VisionInvestigator, WorkspaceReasoner
from vcah.investigator import ObservationAttempt
from vcah.model_client import ImageAttachmentError, OpenAICompatibleClient
from vcah.multiround import InvestigationTask
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)
from vcah.workspace import ObservationLog, stable_attempt_id


def _load_runner() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "run_virtual_videomme_interactive.py"
    spec = importlib.util.spec_from_file_location("workspace_interactive_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        "seg_0001",
        "video-a",
        "video-a.mp4",
        0.0,
        20.0,
        0.0,
        20.0,
        "target",
    )
    return VirtualVideoWorkspace.create(
        tmp_path,
        manifest=VirtualVideoManifest("interactive", (segment,)),
        case=VirtualVideoCase(
            "interactive",
            "What does the person raise?",
            {"A": "A book", "B": "A cup"},
            "B",
            segment.segment_id,
            (0.0, 20.0),
        ),
    )


class FakeAPI:
    model = "fake-model"

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.last_response_metadata: dict[str, Any] = {}

    def chat(self, prompt: str, *, image_paths: Sequence[str] = (), max_tokens: int = 0) -> str:
        self.calls.append({"prompt": prompt, "image_paths": tuple(image_paths), "max_tokens": max_tokens})
        self.last_response_metadata = {
            "images_requested": len(image_paths),
            "images_attached": len(image_paths),
            "images_dropped": 0,
            "finish_reason": "stop",
        }
        return self.responses.pop(0)


def test_reasoner_uses_only_workspace_protocol(tmp_path: Path) -> None:
    api = FakeAPI(
        (
            json.dumps(
                {
                    "action": "investigate",
                    "tasks": [
                        {
                            "query_id": "observe_1",
                            "goal": "Describe the raised object.",
                            "segment_id": "seg_0001",
                            "inspection_mode": "verify_claim",
                            "sampling_floor_fps": 1.0,
                        }
                    ],
                }
            ),
        )
    )
    reasoner = WorkspaceReasoner(api, trace_path=tmp_path / "trace.jsonl")

    decision = reasoner.decide(
        question="What does the person raise?",
        options={"A": "A book", "B": "A cup"},
        remaining_budget=4,
        force_finalize=False,
        mechanical_status={},
        working_document_view="ACTIVE CLAIMS",
        workspace_overview={},
    )

    assert decision.action == "investigate"
    assert not decision.tasks
    prompt = api.calls[0]["prompt"]
    assert "sole semantic decision maker" in prompt
    assert "Working view" in prompt
    assert '"action":"investigate","tasks":[' in prompt
    assert "qualification" not in prompt.casefold()
    assert "option_verdict" not in prompt
    assert "answer audit" not in prompt.casefold()


@pytest.mark.parametrize("wrapper", ("response", "responses", "items"))
def test_reasoner_unwraps_valid_decision_wrappers(tmp_path: Path, wrapper: str) -> None:
    payload = {
        "action": "investigate",
        "tasks": [
            {
                "query_id": "inspect_transition",
                "goal": "Inspect the brief transition.",
                "segment_id": "seg_0001",
                "time_range": [5.0, 7.0],
                "inspection_mode": "window",
                "sampling_floor_fps": 2.0,
            }
        ],
    }
    api = FakeAPI((json.dumps({wrapper: [payload]}),))
    trace_path = tmp_path / f"{wrapper}.jsonl"
    reasoner = WorkspaceReasoner(api, trace_path=trace_path)

    decision = reasoner.decide(
        question="What changes?",
        options={"A": "Nothing", "B": "The color changes"},
        remaining_budget=4,
        force_finalize=False,
        mechanical_status={},
        working_document_view="",
        workspace_overview={},
    )

    assert decision.action == "investigate"
    assert decision.tasks[0].time_range == (5.0, 7.0)
    assert decision.tasks[0].sampling_floor_fps == 2.0
    assert len(api.calls) == 1
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["schema_unwrapped"] is True
    assert trace["repair_failed"] is False


def test_reasoner_repairs_schema_invalid_json_object(tmp_path: Path) -> None:
    api = FakeAPI(
        (
            json.dumps({"response": [{"note": "missing action"}]}),
            json.dumps(
                {
                    "action": "answer",
                    "answer": "B",
                    "supporting_claim_ids": ["claim_cup"],
                }
            ),
        )
    )
    trace_path = tmp_path / "trace.jsonl"
    reasoner = WorkspaceReasoner(api, trace_path=trace_path)

    decision = reasoner.decide(
        question="What does the person raise?",
        options={"A": "A book", "B": "A cup"},
        remaining_budget=0,
        force_finalize=True,
        mechanical_status={},
        working_document_view="",
        workspace_overview={},
    )

    assert decision.answer == "B. A cup"
    assert len(api.calls) == 2
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == ["reasoner_json_repair", "reasoner_workspace"]
    assert rows[-1]["format_repaired"] is True
    assert rows[-1]["repair_attempt_count"] == 1


def test_reasoner_retries_when_first_json_repair_uses_nested_legacy_schema(tmp_path: Path) -> None:
    api = FakeAPI(
        (
            json.dumps({"note": "missing action"}),
            json.dumps(
                {
                    "decision": "Inspect the visible transition.",
                    "action": {
                        "action_type": "investigate",
                        "questions": ["What changes in the supplied window?"],
                    },
                }
            ),
            json.dumps(
                {
                    "action": "investigate",
                    "tasks": [
                        {
                            "query_id": "r1_t1",
                            "goal": "Inspect the visible transition.",
                            "inspection_mode": "window",
                            "segment_id": "seg_0001",
                        }
                    ],
                    "workspace_ops": [],
                }
            ),
        )
    )
    trace_path = tmp_path / "trace.jsonl"
    reasoner = WorkspaceReasoner(api, trace_path=trace_path)

    decision = reasoner.decide(
        question="What changes?",
        options={},
        remaining_budget=4,
        force_finalize=False,
        mechanical_status={},
        working_document_view="",
        workspace_overview={},
    )

    assert decision.action == "investigate"
    assert decision.tasks[0].goal == "Inspect the visible transition."
    assert len(api.calls) == 3
    assert "top-level action must be one of" in api.calls[2]["prompt"]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == [
        "reasoner_json_repair",
        "reasoner_json_repair",
        "reasoner_workspace",
    ]
    assert [row["repair_attempt"] for row in rows[:2]] == [1, 2]
    assert rows[0]["repair_succeeded"] is False
    assert rows[1]["repair_succeeded"] is True
    assert rows[-1]["repair_attempt_count"] == 2


def test_reasoner_prompt_retains_first_and_last_overviews(tmp_path: Path) -> None:
    api = FakeAPI((json.dumps({"action": "answer", "answer": "B"}),))
    reasoner = WorkspaceReasoner(api, trace_path=tmp_path / "trace.jsonl")
    overviews = [
        {
            "overview_id": f"overview_{index:04d}",
            "segment_ids": [f"seg_{index:04d}"],
            "virtual_time_range": [index * 20.0, (index + 1) * 20.0],
            "asr_short_summary": "summary " * 100,
        }
        for index in range(40)
    ]

    reasoner.decide(
        question="What does the person raise?",
        options={"A": "A book", "B": "A cup"},
        remaining_budget=0,
        force_finalize=True,
        mechanical_status={},
        working_document_view="",
        workspace_overview={"segment_overviews": overviews},
    )

    prompt = api.calls[0]["prompt"]
    assert "overview_0000" in prompt
    assert "overview_0039" in prompt
    assert "seg_0039" in prompt
    assert '"residual_uncertainty":""' in prompt
    assert "support_status" not in prompt
    assert "read_observations, update_workspace, or answer" in prompt


def test_final_repair_prompt_requires_an_answer(tmp_path: Path) -> None:
    api = FakeAPI((json.dumps({"action": "update_workspace"}),))
    reasoner = WorkspaceReasoner(api, trace_path=tmp_path / "trace.jsonl")

    reasoner.decide(
        question="What does the person raise?",
        options={"A": "A book", "B": "A cup"},
        remaining_budget=0,
        force_finalize=True,
        final_attempt=2,
        mechanical_status={},
        working_document_view="",
        workspace_overview={},
    )

    prompt = api.calls[0]["prompt"]
    assert "Return action=answer only" in prompt
    assert "workspace-only updates are closed" in prompt


def test_reasoner_preserves_its_answer_and_workspace_operations(tmp_path: Path) -> None:
    api = FakeAPI(
        (
            json.dumps(
                {
                    "action": "answer",
                    "answer": {"option": "B"},
                    "workspace_ops": [
                        {
                            "op": "add_claim",
                            "claim_id": "c1",
                            "text": "The person raises a cup.",
                            "source": "observation",
                            "cites": ["attempt_a"],
                        }
                    ],
                    "supporting_claim_ids": ["c1"],
                    "residual_uncertainty": "The object is briefly occluded.",
                }
            ),
        )
    )
    reasoner = WorkspaceReasoner(api, trace_path=tmp_path / "trace.jsonl")

    decision = reasoner.decide(
        question="What does the person raise?",
        options={"A": "A book", "B": "A cup"},
        remaining_budget=0,
        force_finalize=True,
        mechanical_status={},
        working_document_view="",
        workspace_overview={},
    )

    assert decision.answer == "B. A cup"
    assert decision.workspace_ops[0]["claim_id"] == "c1"
    assert decision.supporting_claim_ids == ("c1",)
    assert decision.residual_uncertainty == "The object is briefly occluded."
    assert "investigate is closed" in api.calls[0]["prompt"]
    assert "Investigate schema" in api.calls[0]["prompt"]


def test_reasoner_does_not_force_or_restore_an_answer(tmp_path: Path) -> None:
    api = FakeAPI(
        (
            json.dumps({"action": "answer", "answer": "B"}),
            json.dumps({"action": "update_workspace"}),
        )
    )
    reasoner = WorkspaceReasoner(api, trace_path=tmp_path / "trace.jsonl")
    common = {
        "question": "What does the person raise?",
        "options": {"A": "A book", "B": "A cup"},
        "remaining_budget": 0,
        "mechanical_status": {},
        "working_document_view": "",
        "workspace_overview": {},
    }

    first = reasoner.decide(force_finalize=False, **common)
    final = reasoner.decide(force_finalize=True, **common)

    assert first.answer == "B. A cup"
    assert final.action == "update_workspace"
    assert not final.answer


def test_investigator_preserves_raw_observation_without_semantic_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    frame_paths = (tmp_path / "frame-5.jpg", tmp_path / "frame-6.jpg")
    for path in frame_paths:
        path.write_bytes(b"frame")
    window = {
        "virtual_time_range": [5.0, 6.0],
        "sampling": {"fps": 1.0, "max_frames": 2, "actual_frames": 2},
        "frames": [
            {"path": str(frame_paths[0]), "virtual_time_sec": 5.0},
            {"path": str(frame_paths[1]), "virtual_time_sec": 6.0},
        ],
        "asr_cues": [],
        "source_lineage": [{"source_video_id": "video-a", "segment_id": "seg_0001"}],
    }
    raw = json.dumps(
        {
            "summary": "The person raises a cup.",
            "observations": [{"time_sec": 5.5, "description": "A cup rises above the table."}],
            "entities": [],
            "events": [],
            "uncertainties": ["The handle is partly hidden."],
        }
    )
    api = FakeAPI((raw,))
    investigator = VisionInvestigator(workspace, api=api, trace_path=tmp_path / "trace.jsonl")
    monkeypatch.setattr(investigator, "inspect_window", lambda *args, **kwargs: window)
    task = InvestigationTask(
        query_id="observe_cup",
        goal="Describe the raised object.",
        segment_id="seg_0001",
        time_range=(5.0, 6.0),
        sampling_floor_fps=1.0,
    )

    report = investigator.run_batch((task,))[0]

    assert report.status == "completed"
    assert report.attempts[0].raw_output == raw
    assert report.evidence[0].operation_metadata["observation_payload"]["summary"] == "The person raises a cup."
    forbidden = {"qualification", "condition_results", "option_verdicts", "claim_assessment"}
    assert forbidden.isdisjoint(report.evidence[0].operation_metadata)
    assert "Do not select an answer option" in api.calls[0]["prompt"]


def test_wide_sparse_scan_is_a_locator_not_full_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    segment = VirtualVideoSegment(
        "seg_long",
        "video-long",
        "video-long.mp4",
        0.0,
        300.0,
        0.0,
        300.0,
        "target",
    )
    workspace = VirtualVideoWorkspace.create(
        tmp_path / "long",
        manifest=VirtualVideoManifest("long", (segment,)),
        case=VirtualVideoCase("long", "What changes?", {}, "", segment.segment_id, (0.0, 300.0)),
    )
    frame_times = (0.0, 120.0, 240.0)
    frame_paths = tuple(tmp_path / f"frame-{int(value)}.jpg" for value in frame_times)
    for path in frame_paths:
        path.write_bytes(b"frame")
    window = {
        "virtual_time_range": [0.0, 240.0],
        "sampling": {"fps": 1.0, "max_frames": 96, "actual_frames": 3},
        "frames": [
            {"path": str(path), "virtual_time_sec": time_sec}
            for path, time_sec in zip(frame_paths, frame_times)
        ],
        "asr_cues": [],
        "source_lineage": [{"source_video_id": "video-long", "segment_id": "seg_long"}],
    }
    raw = json.dumps({"summary": "Sparse locator scan.", "observations": []})
    investigator = VisionInvestigator(
        workspace,
        api=FakeAPI((raw,)),
        trace_path=tmp_path / "wide-trace.jsonl",
    )
    monkeypatch.setattr(investigator, "inspect_window", lambda *args, **kwargs: window)
    task = InvestigationTask(
        query_id="wide-locator",
        goal="Locate the change.",
        segment_id="seg_long",
        time_range=(0.0, 240.0),
        sampling_floor_fps=1.0,
    )

    report = investigator.run_batch((task,))[0]
    attempt = report.attempts[0]
    manifest = attempt.sampling_config["sampling_manifest"]

    assert attempt.requested_range == (0.0, 240.0)
    assert attempt.inspected_ranges != (attempt.requested_range,)
    assert report.coverage_delta == attempt.inspected_ranges
    assert attempt.evidence_role == "candidate"
    assert manifest["requires_refinement"] is True
    assert manifest["coverage_ratio"] < 0.01
    assert manifest["max_gap"] == 120.0
    assert report.cost["requires_refinement"] is True
    assert report.evidence[0].sampling_coverage == "sparse"
    assert len(report.evidence[0].coverage_manifest) == 3


def test_empty_asr_search_attempt_id_is_prompt_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    investigator = VisionInvestigator(
        _workspace(tmp_path),
        api=FakeAPI(()),
        trace_path=tmp_path / "trace.jsonl",
    )
    monkeypatch.setattr(
        investigator,
        "search_asr",
        lambda terms, **kwargs: {"query_terms": list(terms), "clusters": []},
    )
    tasks = (
        InvestigationTask(
            query_id="search_one",
            goal="Locate the first phrase.",
            inspection_mode="search_asr",
            search_terms=("first phrase",),
        ),
        InvestigationTask(
            query_id="search_two",
            goal="Locate a different phrase.",
            inspection_mode="search_asr",
            search_terms=("different phrase",),
        ),
    )

    reports = investigator.run_batch(tasks)

    assert reports[0].attempts[0].attempt_id == reports[1].attempts[0].attempt_id
    assert reports[0].attempts[0].source_video_ids == ("video-a",)
    assert not reports[0].attempts[0].frame_refs
    assert reports[0].cost["consumes_budget"] is False
    assert reports[0].failure_reason == "asr_zero_hits_use_visual_modality"


def test_asr_search_forwards_scope_and_zero_hits_do_not_consume_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    investigator = VisionInvestigator(
        _workspace(tmp_path),
        api=FakeAPI(()),
        trace_path=tmp_path / "trace.jsonl",
    )
    captured: dict[str, Any] = {}

    def search(terms: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
        captured.update({"terms": tuple(terms), **kwargs})
        return {"terms": list(terms), "clusters": []}

    monkeypatch.setattr(investigator, "search_asr", search)
    report = investigator.run_batch(
        (
            InvestigationTask(
                query_id="scoped_asr",
                goal="Search only the requested interval.",
                segment_id="seg_0001",
                time_range=(7.0, 9.0),
                inspection_mode="search_asr",
                search_terms=("alarm",),
            ),
        )
    )[0]

    assert captured["segment_id"] == "seg_0001"
    assert captured["time_range"] == (7.0, 9.0)
    assert report.cost["consumes_budget"] is False
    assert report.failure_reason == "asr_zero_hits_use_visual_modality"


def test_duplicate_asr_material_is_reused_without_spending_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    investigator = VisionInvestigator(
        _workspace(tmp_path),
        api=FakeAPI(()),
        trace_path=tmp_path / "trace.jsonl",
    )
    packet = {
        "terms": ["alarm"],
        "clusters": [
            {
                "segment_id": "seg_0001",
                "virtual_time_range": [7.0, 9.0],
                "matched_terms": ["alarm"],
                "excerpt": "an alarm sounds",
                "source_lineage": [{"source_video_id": "video-a", "segment_id": "seg_0001"}],
                "hit_count": 1,
            }
        ],
    }
    monkeypatch.setattr(investigator, "search_asr", lambda terms, **kwargs: packet)
    tasks = tuple(
        InvestigationTask(
            query_id=f"search_{index}",
            goal="Locate the alarm.",
            segment_id="seg_0001",
            time_range=(7.0, 9.0),
            inspection_mode="search_asr",
            search_terms=("alarm",),
        )
        for index in (1, 2)
    )

    first, duplicate = investigator.run_batch(tasks)

    assert first.cost["consumes_budget"] is True
    assert duplicate.cost["consumes_budget"] is False
    assert duplicate.cost["reused"] is True
    assert not duplicate.attempts
    assert not duplicate.evidence
    assert duplicate.failure_reason == "near_duplicate_asr_query_reused"


def test_near_duplicate_zero_hit_search_is_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    investigator = VisionInvestigator(
        _workspace(tmp_path),
        api=FakeAPI(()),
        trace_path=tmp_path / "trace.jsonl",
    )
    calls = 0

    def search(terms: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        return {"terms": list(terms), "clusters": []}

    monkeypatch.setattr(investigator, "search_asr", search)
    common = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
    first = InvestigationTask(
        query_id="zero-1",
        goal="Locate an event.",
        inspection_mode="search_asr",
        search_terms=(f"{common} twenty",),
    )
    near = InvestigationTask(
        query_id="zero-2",
        goal="Locate the same event with one synonym.",
        inspection_mode="search_asr",
        search_terms=(f"{common} score",),
    )

    first_report, near_report = investigator.run_batch((first, near))

    assert first_report.cost["zero_hits"] is True
    assert near_report.cost["reused"] is True
    assert near_report.cost["zero_hits"] is True
    assert calls == 1
    assert investigator.mechanical_status()["duplicate_search_count"] == 1
    assert investigator.mechanical_status()["empty_search_streak"] == 2


def test_identical_visual_window_reuses_vlm_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    frame = tmp_path / "frame-5.jpg"
    frame.write_bytes(b"frame")
    window = {
        "virtual_time_range": [5.0, 6.0],
        "sampling": {"fps": 1.0, "max_frames": 1, "actual_frames": 1},
        "frames": [{"path": str(frame), "virtual_time_sec": 5.0}],
        "asr_cues": [],
        "source_lineage": [{"source_video_id": "video-a", "segment_id": "seg_0001"}],
    }
    api = FakeAPI(('{"summary":"A cup is visible."}',))
    investigator = VisionInvestigator(workspace, api=api, trace_path=tmp_path / "trace.jsonl")
    monkeypatch.setattr(investigator, "inspect_window", lambda *args, **kwargs: window)
    task = InvestigationTask(
        query_id="visual-reuse",
        goal="Describe the visible object.",
        segment_id="seg_0001",
        time_range=(5.0, 6.0),
        sampling_floor_fps=1.0,
    )

    first, repeated = investigator.run_batch((task, task))

    assert first.cost["vlm_calls"] == 1
    assert repeated.cost["vlm_calls"] == 0
    assert repeated.cost["reused"] is True
    assert repeated.cost["saved_frames"] == 1
    assert len(api.calls) == 1


def test_same_frame_arbitration_reuses_material_identity(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frame_paths = (tmp_path / "frame-5.jpg", tmp_path / "frame-6.jpg")
    for path in frame_paths:
        path.write_bytes(b"frame")
    attempt_id = stable_attempt_id(
        source_video_ids=("video-a",),
        frame_times=(5.0, 6.0),
        sampling_fps=1.0,
        modality="visual",
    )
    log = ObservationLog(tmp_path / "observation_log.jsonl")
    common = {
        "attempt_id": attempt_id,
        "task_id": "first_read",
        "requested_range": (5.0, 6.0),
        "inspected_ranges": ((5.0, 6.0),),
        "attached_frame_times": (5.0, 6.0),
        "sampling_config": {"fps": 1.0, "modality": "visual"},
        "images_requested": 2,
        "images_attached": 2,
        "frame_refs": tuple(str(path) for path in frame_paths),
        "source_video_ids": ("video-a",),
    }
    log.append_attempt(
        ObservationAttempt(**common, raw_output='{"summary":"a cup"}'),
        round_id=1,
        source_lineage=({"source_video_id": "video-a"},),
    )
    log.append_attempt(
        ObservationAttempt(**common, raw_output='{"summary":"possibly a book"}'),
        round_id=2,
        source_lineage=({"source_video_id": "video-a"},),
    )
    api = FakeAPI(('{"summary":"The object is a cup.","uncertainties":[]}',))
    investigator = VisionInvestigator(workspace, api=api, trace_path=tmp_path / "trace.jsonl")
    task = InvestigationTask(
        query_id="arbitrate_1",
        goal="Resolve whether the object is a cup or book.",
        inspection_mode="arbitrate_observation",
        arbitration_attempt_id=attempt_id,
    )

    report = investigator.run_batch((task,))[0]

    assert report.status == "completed"
    assert report.attempts[0].attempt_id == attempt_id
    assert api.calls[0]["image_paths"] == tuple(str(path) for path in frame_paths)
    assert "a cup" in api.calls[0]["prompt"]
    assert "possibly a book" in api.calls[0]["prompt"]


def test_client_retries_length_with_larger_reasoning_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Response:
        status_code = 200
        headers: Mapping[str, str] = {"x-request-id": "request-1"}
        text = ""

        def __init__(self, payload: Mapping[str, Any]) -> None:
            self.payload = payload

        def json(self) -> Mapping[str, Any]:
            return self.payload

    responses = [
        Response(
            {
                "choices": [{"finish_reason": "length", "message": {"content": ""}}],
                "usage": {"completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 100}},
            }
        ),
        Response(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"B"}'}}],
                "usage": {"completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 10}},
            }
        ),
    ]

    def post(url: str, **kwargs: Any) -> Response:
        calls.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr("vcah.model_client.requests.post", post)
    client = OpenAICompatibleClient(
        {"base": "https://example.invalid/v1", "model": "gpt-5-test", "api_key": "secret", "max_retries": 0}
    )

    assert client.chat("answer", max_tokens=200) == '{"answer":"B"}'
    assert calls[0][1]["json"]["max_completion_tokens"] == 200
    assert calls[1][1]["json"]["max_completion_tokens"] == 4096
    assert client.last_response_metadata["truncated_then_retried"]
    assert client.last_response_metadata["reasoning_tokens"] == 110


def test_client_rejects_missing_images_before_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "vcah.model_client.requests.post",
        lambda *args, **kwargs: pytest.fail("network request should not run"),
    )
    client = OpenAICompatibleClient(
        {"base": "https://example.invalid/v1", "model": "vision", "api_key": "secret", "max_retries": 0}
    )

    with pytest.raises(ImageAttachmentError):
        client.chat("inspect", image_paths=(str(tmp_path / "missing.jpg"),))


def test_client_can_interleave_image_labels_and_place_prompt_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Mapping[str, Any]] = []
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class Response:
        status_code = 200
        headers: Mapping[str, str] = {}
        text = ""

        def json(self) -> Mapping[str, Any]:
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": "caption"}}],
                "usage": {},
            }

    def post(url: str, **kwargs: Any) -> Response:
        calls.append({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("vcah.model_client.requests.post", post)
    client = OpenAICompatibleClient(
        {"base": "https://example.invalid/v1", "model": "vision", "api_key": "secret", "max_retries": 0}
    )

    assert client.chat(
        "caption prompt",
        image_paths=(str(first), str(second)),
        image_labels=("[00:00:00]", "[00:00:05]"),
        prompt_position="last",
    ) == "caption"
    content = calls[0]["json"]["messages"][0]["content"]

    assert [item["type"] for item in content] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert [item["text"] for item in content if item["type"] == "text"] == [
        "[00:00:00]",
        "[00:00:05]",
        "caption prompt",
    ]
    assert client.last_response_metadata["image_label_count"] == 2
    assert client.last_response_metadata["prompt_position"] == "last"


def test_runner_helpers_keep_order_and_parse_case_groups(tmp_path: Path) -> None:
    group = tmp_path / "group.json"
    group.write_text(
        json.dumps(
            {
                "group_id": "small",
                "construction": "source_only",
                "cases": [{"case_id": "b"}, {"case_id": "a"}],
            }
        ),
        encoding="utf-8",
    )

    parsed = RUNNER._load_case_group(group)
    rows = RUNNER._run_case_batch(parsed["case_ids"], lambda case_id: {"case_id": case_id}, workers=2)

    assert parsed["case_ids"] == ("b", "a")
    assert [row["case_id"] for row in rows] == ["b", "a"]
    assert RUNNER._options_mapping(("A. first", "second")) == {"A": "first", "B": "second"}
