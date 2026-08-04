from __future__ import annotations

from pathlib import Path

from vcah.multiround import InvestigationTask, _resolve_tasks
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        segment_id="seg_0001",
        source_video_id="video-a",
        source_path="video-a.mp4",
        source_start_sec=0.0,
        source_end_sec=20.0,
        virtual_start_sec=0.0,
        virtual_end_sec=20.0,
    )
    return VirtualVideoWorkspace.create(
        tmp_path,
        manifest=VirtualVideoManifest("case-cue", (segment,)),
        case=VirtualVideoCase(case_id="case-cue", question="What happens?"),
    )


def _parent_row() -> dict[str, object]:
    return {
        "attempt_id": "attempt_parent",
        "observation_cues": [
            {
                "cue_id": "cue_exact",
                "attempt_id": "attempt_parent",
                "interpretation_id": "interpretation_parent",
                "item_id": "item_parent",
                "source_frame_ref": "frame-8.jpg",
                "virtual_time": 8.0,
                "cue_kind": "event",
            }
        ],
    }


def _task() -> InvestigationTask:
    return InvestigationTask(
        query_id="refine",
        goal="Inspect the event around the cue.",
        time_range=(18.0, 19.0),
        parent_attempt_id="attempt_parent",
        cue_id="cue_exact",
        window_radius_sec=3.0,
        sampling_floor_fps=0.5,
    )


def test_unverified_cue_forces_exact_same_frame_verification(tmp_path: Path) -> None:
    tasks = _resolve_tasks(
        _workspace(tmp_path),
        (_task(),),
        limit=1,
        observation_rows=(_parent_row(),),
    )

    assert len(tasks) == 1
    resolved = tasks[0]
    assert resolved.time_range == (8.0, 8.0)
    assert resolved.segment_id == "seg_0001"
    assert resolved.sampling_floor_fps == 2.0
    assert resolved.cue_stage == "cue_verification"
    assert resolved.interpretation_purpose == "cue_verification"


def test_verified_cue_opens_only_its_narrow_child_window(tmp_path: Path) -> None:
    tasks = _resolve_tasks(
        _workspace(tmp_path),
        (_task(),),
        limit=1,
        observation_rows=(_parent_row(),),
        cue_states={"cue_exact": {"status": "verified"}},
    )

    assert len(tasks) == 1
    resolved = tasks[0]
    assert resolved.time_range == (5.0, 11.0)
    assert resolved.cue_stage == "child_refinement"
    assert resolved.cue_virtual_time == 8.0
    assert resolved.interpretation_purpose == "manual_reread"


def test_rejected_or_unbound_cue_is_an_explicit_resolution_error(tmp_path: Path) -> None:
    errors: list[dict[str, object]] = []
    tasks = _resolve_tasks(
        _workspace(tmp_path),
        (_task(),),
        limit=1,
        errors=errors,
        observation_rows=(_parent_row(),),
        cue_states={"cue_exact": {"status": "rejected"}},
    )

    assert not tasks
    assert errors[0]["code"] == "cue_rejected"
