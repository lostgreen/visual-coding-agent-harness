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
