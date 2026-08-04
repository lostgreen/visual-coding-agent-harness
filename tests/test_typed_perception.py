from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from vcah.interactive_agents import VisionInvestigator
from vcah.multiround import InvestigationTask
from vcah.sampling import evidence_sampling_profile
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

    def chat(
        self,
        prompt: str,
        *,
        image_paths: Sequence[str] = (),
        max_tokens: int = 0,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "image_paths": tuple(image_paths),
                "max_tokens": max_tokens,
            }
        )
        self.last_response_metadata = {
            "images_requested": len(image_paths),
            "images_attached": len(image_paths),
            "images_dropped": 0,
            "finish_reason": "stop",
        }
        return self.responses.pop(0)


def _workspace(tmp_path: Path, *, duration: float = 300.0) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        "seg_0001",
        "video-a",
        "video-a.mp4",
        0.0,
        duration,
        0.0,
        duration,
    )
    return VirtualVideoWorkspace.create(
        tmp_path,
        manifest=VirtualVideoManifest("typed", (segment,)),
        case=VirtualVideoCase("typed", "What is visible?"),
    )


def _window(tmp_path: Path, frame_times: Sequence[float]) -> dict[str, object]:
    paths = tuple(tmp_path / f"frame-{index}.jpg" for index in range(len(frame_times)))
    for path in paths:
        path.write_bytes(b"frame")
    return {
        "virtual_time_range": [min(frame_times), max(frame_times)],
        "sampling": {"actual_frames": len(paths)},
        "frames": [
            {"path": str(path), "virtual_time_sec": time_sec}
            for path, time_sec in zip(paths, frame_times)
        ],
        "asr_cues": [],
        "source_lineage": [
            {"source_video_id": "video-a", "segment_id": "seg_0001"}
        ],
    }


def test_sampling_profiles_encode_evidence_specific_contracts() -> None:
    assert evidence_sampling_profile("text_exact").same_material_second_read
    assert evidence_sampling_profile("ui_text").max_window_sec == 20.0
    assert evidence_sampling_profile("persistent_state").max_probe_count == 12
    assert evidence_sampling_profile("transient_event").perception_mode == "cue_refinement"
    assert evidence_sampling_profile("relation").fps >= 1.0


def test_text_exact_runs_two_interpretations_on_identical_material(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "workspace", duration=20.0)
    window = _window(tmp_path, (5.0, 6.0))
    responses = (
        json.dumps(
            {
                "summary": "CODE-123",
                "items": [
                    {
                        "time_anchor": [5.0, 5.0],
                        "text": "CODE-123",
                        "item_kind": "text",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "CODE-123",
                "items": [
                    {
                        "time_anchor": [5.0, 5.0],
                        "text": "CODE-123",
                        "item_kind": "text",
                    }
                ],
            }
        ),
    )
    api = FakeAPI(responses)
    investigator = VisionInvestigator(
        workspace,
        api=api,
        trace_path=tmp_path / "trace.jsonl",
    )
    monkeypatch.setattr(investigator, "inspect_window", lambda *args, **kwargs: window)
    report = investigator.run_batch(
        (
            InvestigationTask(
                query_id="read_text",
                goal="Read the exact code.",
                segment_id="seg_0001",
                time_range=(5.0, 6.0),
                evidence_kind="text_exact",
            ),
        )
    )[0]

    assert len(report.attempts) == 2
    primary, reread = report.attempts
    assert primary.attempt_id == reread.attempt_id
    assert primary.frame_refs == reread.frame_refs
    assert primary.attached_frame_times == reread.attached_frame_times
    assert primary.interpretation_purpose == "primary"
    assert reread.interpretation_purpose == "manual_reread"
    assert report.cost["vlm_calls"] == 2
    assert api.calls[0]["image_paths"] == api.calls[1]["image_paths"]


def test_persistent_state_uses_probe_coverage_instead_of_fps_fidelity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "workspace")
    frame_times = tuple(float(value) for value in range(0, 300, 40))
    window = _window(tmp_path, frame_times)
    api = FakeAPI((json.dumps({"summary": "The state persists."}),))
    investigator = VisionInvestigator(
        workspace,
        api=api,
        trace_path=tmp_path / "trace.jsonl",
    )
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def inspect(*args: Any, **kwargs: Any) -> dict[str, object]:
        calls.append((args, kwargs))
        return window

    monkeypatch.setattr(investigator, "inspect_window", inspect)
    report = investigator.run_batch(
        (
            InvestigationTask(
                query_id="probe_state",
                goal="Check whether the state persists.",
                segment_id="seg_0001",
                time_range=(0.0, 300.0),
                evidence_kind="persistent_state",
            ),
        )
    )[0]
    config = report.attempts[0].sampling_config

    assert calls[0][1]["fps"] == 0.5
    assert calls[0][1]["max_frames"] == 12
    assert config["probe_coverage_requirement"] == 6
    assert config["probe_coverage_satisfied"] is True
    assert config["requires_refinement"] is False
    assert report.attempts[0].evidence_role == "unclassified"


def test_ui_text_mechanically_bounds_wide_window_and_requests_separate_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "workspace", duration=100.0)
    window = _window(tmp_path, (40.0, 60.0))
    response = json.dumps(
        {
            "summary": "Menu",
            "items": [
                {"time_anchor": [40.0, 40.0], "text": "Start", "item_kind": "ui_label"},
                {
                    "time_anchor": [40.0, 40.0],
                    "text": "A highlighted menu entry.",
                    "item_kind": "ui_description",
                },
            ],
        }
    )
    api = FakeAPI((response, response))
    investigator = VisionInvestigator(
        workspace,
        api=api,
        trace_path=tmp_path / "trace.jsonl",
    )
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def inspect(*args: Any, **kwargs: Any) -> dict[str, object]:
        calls.append((args, kwargs))
        return window

    monkeypatch.setattr(investigator, "inspect_window", inspect)
    investigator.run_batch(
        (
            InvestigationTask(
                query_id="read_ui",
                goal="Read the menu entry.",
                segment_id="seg_0001",
                time_range=(0.0, 100.0),
                evidence_kind="ui_text",
            ),
        )
    )

    assert calls[0][0][:2] == (40.0, 60.0)
    assert calls[0][1]["fps"] == 2.0
    assert calls[0][1]["max_frames"] == 40
    assert "item_kind=ui_label" in api.calls[0]["prompt"]
