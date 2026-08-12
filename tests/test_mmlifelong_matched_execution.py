from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from vcah.interactive_agents import VisionInvestigator
from vcah.multiround import InvestigationTask
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
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


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        "seg_0001", "video-a", "video-a.mp4", 0.0, 20.0, 0.0, 20.0, "target"
    )
    return VirtualVideoWorkspace.create(
        tmp_path,
        manifest=VirtualVideoManifest("interactive", (segment,)),
        case=VirtualVideoCase(
            "interactive", "What happens?", {}, "", segment.segment_id, (0.0, 20.0)
        ),
    )


def test_forced_anchor_execution_adds_exact_frame_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    frame_paths = {
        5.0: tmp_path / "frame-5.jpg",
        7.5: tmp_path / "frame-7.5.jpg",
        10.0: tmp_path / "frame-10.jpg",
    }
    for path in frame_paths.values():
        path.write_bytes(b"frame")
    calls: list[tuple[float, float]] = []

    def inspect(start_sec: float, end_sec: float, **_: Any) -> Mapping[str, Any]:
        calls.append((start_sec, end_sec))
        times = (7.5,) if start_sec == end_sec == 7.5 else (5.0, 10.0)
        return {
            "virtual_time_range": [start_sec, end_sec],
            "sampling": {"fps": 1.0, "max_frames": len(times), "actual_frames": len(times)},
            "frames": [
                {"path": str(frame_paths[value]), "virtual_time_sec": value}
                for value in times
            ],
            "asr_cues": [],
            "source_lineage": [
                {"source_video_id": "video-a", "segment_id": "seg_0001"}
            ],
        }

    api = FakeAPI(('{"summary":"The anchor frame is visible."}',))
    investigator = VisionInvestigator(
        workspace,
        api=api,
        trace_path=tmp_path / "trace.jsonl",
        anchor_execution_policy="force_if_requested",
    )
    investigator._oracle_guidance = {"anchor_timestamps_sec": [7.5, 15.0]}
    monkeypatch.setattr(investigator, "inspect_window", inspect)
    task = InvestigationTask(
        query_id="inspect_anchor",
        goal="Inspect the requested occurrence.",
        segment_id="seg_0001",
        time_range=(5.0, 10.0),
        sampling_floor_fps=1.0,
    )

    report = investigator.run_batch((task,))[0]
    attempt = report.attempts[0]

    assert calls == [(5.0, 10.0), (7.5, 7.5)]
    assert attempt.attached_frame_times == (5.0, 7.5, 10.0)
    assert attempt.sampling_config["forced_anchor_timestamps_sec"] == [7.5]
    assert api.calls[0]["image_paths"] == tuple(
        str(frame_paths[value]) for value in (5.0, 7.5, 10.0)
    )
