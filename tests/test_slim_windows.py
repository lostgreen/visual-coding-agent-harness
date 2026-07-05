from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from vcah.agent import VideoAgent
from vcah.model import ScriptedModel
from vcah.types import (
    Frame,
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


def test_window_overlap_ratio() -> None:
    assert window_overlap_ratio(Window(100, 200), (Window(90, 210),)) == 1.0
    assert window_overlap_ratio(Window(100, 200), (Window(100, 150),)) == 0.5
    assert window_overlap_ratio(Window(100, 200), (Window(300, 400),)) == 0.0


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
