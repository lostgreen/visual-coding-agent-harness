from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vcah.agent import VideoAgent
from vcah.model import ScriptedModel
from vcah.types import Frame


def _sampler(video_path: str, start_sec: float, end_sec: float, n_frames: int, out_dir: Path) -> tuple[Frame, ...]:
    del video_path, end_sec, n_frames
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_{int(start_sec):03d}.jpg"
    Image.new("RGB", (32, 18), color=(20, 40, 230)).save(path)
    return (Frame(frame_id="fr001", time_sec=start_sec, path=str(path)),)


def test_window_lineage_tracks_materialized_request_ids(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "inspect_windows": [{"start_sec": 10, "end_sec": 20}], "modalities": ["asr"]},
        ]
    )
    VideoAgent(model=model, max_steps=1).ask(
        "/videos/demo.mp4",
        "What happens?",
        run_dir=tmp_path,
        duration_sec=30.0,
        asr_cues=({"start": 10.0, "end": 20.0, "text": "The bridge is mentioned."},),
        range_detector=lambda _video_path, _duration: ((0.0, 30.0),),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    lineage = trace[0]["window_lineage"]

    assert lineage["raw_requested_ids"] == lineage["parsed_requested_ids"] == lineage["dispatched_request_ids"]
    assert lineage["executed_request_ids"] == lineage["raw_requested_ids"]
    assert lineage["materialized_request_ids"] == lineage["raw_requested_ids"]
    assert lineage["error"] is None


def test_reasoner_request_id_is_preserved_in_lineage(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {
                "type": "inspect_window",
                "inspect_windows": [{"request_id": "win_reasoner_01", "start_sec": 10, "end_sec": 20}],
                "modalities": ["asr"],
            },
        ]
    )
    VideoAgent(model=model, max_steps=1).ask(
        "/videos/demo.mp4",
        "What happens?",
        run_dir=tmp_path,
        duration_sec=30.0,
        asr_cues=({"start": 10.0, "end": 20.0, "text": "The bridge is mentioned."},),
        range_detector=lambda _video_path, _duration: ((0.0, 30.0),),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert trace[0]["action"]["windows"][0]["request_id"] == "win_reasoner_01"
    assert trace[0]["window_lineage"]["raw_requested_ids"] == ["win_reasoner_01"]


def test_window_lineage_reports_execution_loss(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "inspect_windows": [{"start_sec": 50, "end_sec": 60}], "modalities": ["asr"]},
        ]
    )
    VideoAgent(model=model, max_steps=1).ask(
        "/videos/demo.mp4",
        "What happens later?",
        run_dir=tmp_path,
        duration_sec=30.0,
        asr_cues=({"start": 10.0, "end": 20.0, "text": "The bridge is mentioned."},),
        range_detector=lambda _video_path, _duration: ((0.0, 30.0),),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    lineage = trace[0]["window_lineage"]

    assert lineage["executed_request_ids"] == []
    assert lineage["dropped_request_ids"] == lineage["raw_requested_ids"]
    assert lineage["error"] == "window_request_execution_loss"


def test_window_lineage_reports_raw_to_parsed_loss(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {
                "type": "inspect_window",
                "inspect_windows": [
                    {"start_sec": 1, "end_sec": 2},
                    {"start_sec": 3, "end_sec": 4},
                    {"start_sec": 5},
                    {"start_sec": 6, "end_sec": 7},
                ],
                "modalities": ["asr"],
            },
        ]
    )
    VideoAgent(model=model, max_steps=1).ask(
        "/videos/demo.mp4",
        "What happens?",
        run_dir=tmp_path,
        duration_sec=10.0,
        asr_cues=(
            {"start": 1.0, "end": 2.0, "text": "first"},
            {"start": 3.0, "end": 4.0, "text": "second"},
            {"start": 6.0, "end": 7.0, "text": "third"},
        ),
        range_detector=lambda _video_path, _duration: ((0.0, 10.0),),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    lineage = trace[0]["window_lineage"]

    assert len(lineage["raw_requested_ids"]) == 4
    assert len(lineage["parsed_requested_ids"]) == 3
    assert lineage["error"] == "window_request_parse_loss"
    assert lineage["parse_errors"]


def test_window_lineage_reports_materialization_loss(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "inspect_windows": [{"start_sec": 10, "end_sec": 20}], "modalities": ["asr"]},
        ]
    )
    VideoAgent(model=model, max_steps=1).ask(
        "/videos/demo.mp4",
        "What happens?",
        run_dir=tmp_path,
        duration_sec=30.0,
        asr_cues=(),
        range_detector=lambda _video_path, _duration: ((0.0, 30.0),),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    lineage = trace[0]["window_lineage"]

    assert lineage["executed_request_ids"] == lineage["raw_requested_ids"]
    assert lineage["materialized_request_ids"] == []
    assert lineage["error"] == "window_request_materialization_loss"


def test_lineage_error_latches_and_blocks_final_answer(tmp_path: Path) -> None:
    model = ScriptedModel(
        actions=[
            {"type": "inspect_window", "inspect_windows": [{"start_sec": 10, "end_sec": 20}], "modalities": ["asr"]},
            {"type": "answer", "answer": "A", "selected": "A", "citations": ["ev_missing"]},
        ]
    )
    answer = VideoAgent(model=model, max_steps=2).ask(
        "/videos/demo.mp4",
        "Which statement is correct?\nA. A bridge is mentioned.\nB. A tower is mentioned.",
        run_dir=tmp_path,
        duration_sec=30.0,
        asr_cues=(),
        range_detector=lambda _video_path, _duration: ((0.0, 30.0),),
        keyframe_sampler=_sampler,
    )

    trace = [json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert answer.answer == "Insufficient verified evidence."
    assert trace[1]["final_verification"]["reason"] == "run_integrity_failure"
    assert trace[1]["final_verification"]["integrity_failures"]
