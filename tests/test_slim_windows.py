from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from vcah.agent import VideoAgent
from vcah.index import ColdIndex, TextIndex, VisualIndex
from vcah.memory import AgentMemory, EvidenceStore
from vcah.model import ScriptedModel
from vcah.tools import AgentTools
from vcah.types import (
    Beat,
    Chapter,
    Frame,
    IndexDiagnostics,
    InvestigatorOutputEmpty,
    InvestigatorOutputInvalid,
    ToolAction,
    Window,
    investigator_input_has_hypothesis,
    validate_investigator_output,
    verify_final_answer,
    window_overlap_ratio,
)


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=(20, 40, 230)).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec, path=str(path)),)


class AttestSpyModel(ScriptedModel):
    def __init__(self) -> None:
        super().__init__()
        self.seen_paths: tuple[str, ...] = ()
        self.vision_model = "vision-spy"

    def attest(self, image_paths: Sequence[str], prompt: str) -> tuple[str, ...]:
        del prompt
        self.seen_paths = tuple(image_paths)
        return tuple(f"Visible clue in {Path(path).name}." for path in image_paths)


def test_window_overlap_ratio() -> None:
    assert window_overlap_ratio(Window(100, 200), (Window(90, 210),)) == 1.0
    assert window_overlap_ratio(Window(100, 200), (Window(100, 150),)) == 0.5
    assert window_overlap_ratio(Window(100, 200), (Window(300, 400),)) == 0.0
    assert window_overlap_ratio(Window(100, 200), (Window(100, 170), Window(130, 200))) == 1.0
    assert window_overlap_ratio(Window(100, 200), (Window(100, 160), Window(130, 170))) == 0.7


def test_tool_action_parses_inspect_windows() -> None:
    action = ToolAction.from_mapping(
        {
            "type": "inspect_window",
            "inspect_windows": [{"start": "00:01:40", "end": "00:02:00"}],
            "modalities": ["asr", "frames"],
        }
    )

    assert action.windows == (Window(100.0, 120.0),)
    assert action.modalities == ("asr", "frames")


def test_inspect_window_executes_requested_window_and_traces_coverage(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "start_sec": 110, "end_sec": 150, "modalities": ["asr"]},
            {"type": "answer", "answer": "The requested segment mentions the bridge.", "citations": ["ev_0001"]},
        ]
    )
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "What is discussed in the requested window?",
        run_dir=tmp_path,
        duration_sec=300.0,
        asr_cues=({"start": 100.0, "end": 200.0, "text": "the bridge is discussed"},),
        range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.citations == ("ev_0001",)
    assert trace[0]["action"]["windows"] == [{"start_sec": 110.0, "end_sec": 150.0}]
    assert trace[0]["requested_windows"] == [{"start_sec": 110.0, "end_sec": 150.0}]
    assert trace[0]["window_coverage_report"][0]["coverage"] == 1.0
    assert trace[0]["fallback_used"] is False
    assert trace[0]["actual_windows"][0]["start_sec"] == 100.0
    assert trace[1]["final_verification"]["passed"] is True
    evidence = json.loads((tmp_path / "run" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert evidence["start_sec"] == 110.0
    assert evidence["end_sec"] == 150.0
    evidence_window = [item for item in trace[0]["actual_windows"] if item.get("evidence_id") == "ev_0001"][0]["evidence_window"]
    assert evidence_window == {"start_sec": 110.0, "end_sec": 150.0}


def test_inspect_window_verbatim_is_clipped_to_timed_cues(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "start_sec": 110, "end_sec": 150, "modalities": ["asr"]},
            {"type": "answer", "answer": "A", "citations": ["ev_0001"]},
        ]
    )
    agent = VideoAgent(model=model, max_steps=3)

    agent.ask(
        "/videos/demo.mp4",
        "What is discussed in the requested window?",
        run_dir=tmp_path,
        duration_sec=300.0,
        asr_cues=(
            {"start": 100.0, "end": 120.0, "text": "irrelevant setup"},
            {"start": 170.0, "end": 190.0, "text": "answer clue outside requested window"},
        ),
        range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        keyframe_sampler=_sampler,
    )
    evidence = json.loads((tmp_path / "run" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()[0])
    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    evidence_window = [item for item in trace[0]["actual_windows"] if item.get("evidence_id") == "ev_0001"][0]

    assert evidence["verbatim"] == "irrelevant setup"
    assert "answer clue" not in evidence["verbatim"]
    assert evidence_window["verbatim_is_window_local"] is True


def test_inspect_window_does_not_create_window_evidence_from_whole_beat_fallback(tmp_path: Path) -> None:
    text_index = TextIndex()
    text_index.add("bt00001", "whole beat transcript includes outside-window facts", modality="asr")
    model = ScriptedModel()
    index = ColdIndex(
        video_path="/videos/demo.mp4",
        duration_sec=200.0,
        chapters=(Chapter("ch01", 100.0, 200.0, ("bt00001",)),),
        beats=(
            Beat(
                "bt00001",
                "ch01",
                100.0,
                200.0,
                "",
                asr_text="whole beat transcript includes outside-window facts",
                asr_cues=(),
            ),
        ),
        text_index=text_index,
        visual_index=VisualIndex(model),
        diagnostics=IndexDiagnostics(200.0, 1, 1, 100.0, 100.0, 0, 0.0, "test", "fast"),
    )
    evidence = EvidenceStore.empty(tmp_path / "evidence.jsonl")
    tools = AgentTools(index, AgentMemory.empty(tmp_path / "memory.json"), evidence, tmp_path)

    result = tools.inspect_window((Window(120.0, 140.0),), ("asr",))

    assert result.evidence_ids == ()
    assert result.payload["evidence_created"] is False
    assert result.payload["actual_windows"][-1]["source"] == "asr"
    assert result.payload["actual_windows"][-1]["verbatim_is_window_local"] is False
    assert result.payload["actual_windows"][-1]["skipped_reason"] == "non_window_local_verbatim"
    assert (tmp_path / "evidence.jsonl").read_text(encoding="utf-8") == ""


def test_inspect_window_filters_out_of_window_frame_refs(tmp_path: Path) -> None:
    text_index = TextIndex()
    model = AttestSpyModel()
    index = ColdIndex(
        video_path="/videos/demo.mp4",
        duration_sec=40.0,
        chapters=(Chapter("ch01", 0.0, 40.0, ("bt00001",)),),
        beats=(
            Beat(
                "bt00001",
                "ch01",
                0.0,
                40.0,
                "",
                frame_paths=("frame_012.jpg", "frame_999.jpg"),
            ),
        ),
        text_index=text_index,
        visual_index=VisualIndex(model),
        diagnostics=IndexDiagnostics(40.0, 1, 1, 40.0, 40.0, 0, 0.0, "test", "fast"),
    )
    evidence = EvidenceStore.empty(tmp_path / "evidence.jsonl")
    tools = AgentTools(index, AgentMemory.empty(tmp_path / "memory.json"), evidence, tmp_path)

    result = tools.inspect_window((Window(10.0, 20.0),), ("frames",))

    assert model.seen_paths == ("frame_012.jpg",)
    assert result.evidence_ids == ("ev_0001",)
    assert evidence.records[0].frame_refs == ("frame_012.jpg",)


def test_inspect_window_fails_closed_when_coverage_is_low(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "start_sec": 500, "end_sec": 600, "modalities": ["asr"]},
            {"type": "answer", "answer": "Unsupported.", "citations": ["ev_0001"]},
        ]
    )
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "What happens later?",
        run_dir=tmp_path,
        duration_sec=300.0,
        asr_cues=({"start": 100.0, "end": 200.0, "text": "the bridge is discussed"},),
        range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        keyframe_sampler=_sampler,
    )
    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.answer == "Insufficient verified evidence."
    assert trace[0]["result"]["text"] == "window_coverage_failed"
    assert trace[0]["window_coverage_report"][0]["coverage"] == 0.0
    assert (tmp_path / "run" / "evidence.jsonl").read_text(encoding="utf-8") == ""


def test_multi_window_partial_fail_creates_no_evidence(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {
                "type": "inspect_window",
                "inspect_windows": [{"start_sec": 110, "end_sec": 150}, {"start_sec": 500, "end_sec": 600}],
                "modalities": ["asr"],
            },
            {"type": "answer", "answer": "Unsupported.", "citations": ["ev_0001"]},
        ]
    )
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "What happens in two windows?",
        run_dir=tmp_path,
        duration_sec=300.0,
        asr_cues=({"start": 100.0, "end": 200.0, "text": "first window transcript"},),
        range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        keyframe_sampler=_sampler,
    )
    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.answer == "Insufficient verified evidence."
    assert trace[0]["result"]["text"] == "window_coverage_failed"
    assert (tmp_path / "run" / "evidence.jsonl").read_text(encoding="utf-8") == ""


def test_same_video_different_questions_get_different_actual_windows(tmp_path: Path) -> None:
    actual_windows = []
    for idx, window in enumerate(((10.0, 20.0), (110.0, 120.0), (210.0, 220.0)), start=1):
        model = ScriptedModel(actions=[{"type": "inspect_window", "start_sec": window[0], "end_sec": window[1], "modalities": ["asr"]}])
        agent = VideoAgent(model=model, max_steps=1)
        run_dir = tmp_path / f"q{idx}"
        agent.ask(
            "/videos/demo.mp4",
            f"Question {idx}",
            run_dir=run_dir,
            duration_sec=300.0,
            asr_cues=(
                {"start": 0.0, "end": 100.0, "text": "early"},
                {"start": 100.0, "end": 200.0, "text": "middle"},
                {"start": 200.0, "end": 300.0, "text": "late"},
            ),
            range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
            keyframe_sampler=_sampler,
        )
        trace = [json.loads(line) for line in (run_dir / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        actual_windows.append(
            tuple((item["start_sec"], item["end_sec"]) for item in trace[0]["actual_windows"] if item.get("source") == "beat")
        )

    assert len(set(actual_windows)) == 3


def test_investigator_hypothesis_and_output_validation() -> None:
    assert investigator_input_has_hypothesis({"reasoner_request": "The likely answer is B"})
    assert investigator_input_has_hypothesis({"answer_hypothesis": "B"})

    try:
        validate_investigator_output({})
    except InvestigatorOutputEmpty:
        pass
    else:
        raise AssertionError("empty investigator output should fail")

    invalid = {"A": {"status": "maybe", "support": [], "contradict": []}}
    try:
        validate_investigator_output(invalid)
    except InvestigatorOutputInvalid:
        pass
    else:
        raise AssertionError("invalid investigator output should fail")


def test_final_verifier_respects_question_polarity() -> None:
    table = {
        "A": {"status": "supported", "support": [], "contradict": []},
        "B": {"status": "contradicted", "support": [], "contradict": []},
    }

    assert verify_final_answer("Which statement is correct?", table, "A")["passed"]
    assert verify_final_answer("Which statement is not correct?", table, "B")["passed"]
    assert not verify_final_answer("Which statement is not correct?", table, "A")["passed"]


def test_agent_final_verifier_blocks_contradicted_positive_option(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "start_sec": 110, "end_sec": 150, "modalities": ["asr"]},
            {
                "type": "answer",
                "selected": "B",
                "answer": "B",
                "citations": ["ev_0001"],
                "evidence_table": {
                    "B": {"status": "contradicted", "support": [], "contradict": [{"text": "William won."}]}
                },
            },
        ]
    )
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "Which statement is correct?",
        run_dir=tmp_path,
        duration_sec=300.0,
        asr_cues=({"start": 100.0, "end": 200.0, "text": "William won."},),
        range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        keyframe_sampler=_sampler,
    )
    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.answer == "Insufficient verified evidence."
    assert trace[1]["final_verification"]["reason"] == "invalid_evidence_table"


def test_agent_requires_evidence_table_for_selected_answer(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "start_sec": 110, "end_sec": 150, "modalities": ["asr"]},
            {"type": "answer", "selected": "A", "answer": "A", "citations": ["ev_0001"]},
        ]
    )
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "Which statement is correct?",
        run_dir=tmp_path,
        duration_sec=300.0,
        asr_cues=({"start": 100.0, "end": 200.0, "text": "supported"},),
        range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        keyframe_sampler=_sampler,
    )
    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.answer == "Insufficient verified evidence."
    assert trace[1]["final_verification"]["reason"] == "missing_evidence_table"


def test_agent_final_guard_blocks_investigator_hypothesis_payload(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "start_sec": 110, "end_sec": 150, "modalities": ["asr"]},
            {
                "type": "answer",
                "selected": "A",
                "answer": "A",
                "citations": ["ev_0001"],
                "investigator_payload": {"answer_hypothesis": "A"},
                "evidence_table": {"A": {"status": "supported", "support": [{"text": "supported"}], "contradict": []}},
            },
        ]
    )
    agent = VideoAgent(model=model, max_steps=3)

    answer = agent.ask(
        "/videos/demo.mp4",
        "Which statement is correct?",
        run_dir=tmp_path,
        duration_sec=300.0,
        asr_cues=({"start": 100.0, "end": 200.0, "text": "supported"},),
        range_detector=lambda _video_path, _duration: ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        keyframe_sampler=_sampler,
    )
    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.answer == "Insufficient verified evidence."
    assert trace[1]["investigator_received_hypothesis"] is True
    assert trace[1]["final_verification"]["reason"] == "investigator_input_contains_hypothesis"
