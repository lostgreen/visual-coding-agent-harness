from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.contracts.report import VerifyRequest
from visual_coding_agent_harness.tools.vlm_tools import explore
from visual_coding_agent_harness.tools.vlm_tools import verify_window
from visual_coding_agent_harness.video.index import Frame, Scene, Shot, VideoIndex


class RecordingBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.response)


def _write_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)
    return str(path)


def _index(image_dir: Path | None = None) -> VideoIndex:
    shots = []
    for idx in range(1, 4):
        shot_id = f"sc01_sh{idx:03d}"
        lowres_grid_path = f"/grids/{shot_id}.jpg"
        if image_dir is not None:
            lowres_grid_path = _write_image(image_dir / f"{shot_id}.jpg")
        shots.append(
            Shot(
                shot_id=shot_id,
                scene_id="sc01",
                start_sec=float(idx * 10),
                end_sec=float(idx * 10 + 8),
                frames=(Frame(frame_id=f"{shot_id}_fr001", time_sec=float(idx * 10), thumb_path=f"/thumbs/{shot_id}.jpg"),),
                visual_caption=f"Shot {idx} visual caption.",
                asr_text=f"Shot {idx} transcript.",
                ocr_lines=(),
                entities=("car",) if idx == 2 else (),
                lowres_grid_path=lowres_grid_path,
            )
        )
    return VideoIndex(
        video_path="/videos/demo.mp4",
        duration_sec=60.0,
        scenes=(
            Scene(
                scene_id="sc01",
                start_sec=0.0,
                end_sec=60.0,
                title="Street scene",
                summary="Cars appear on a street.",
                shots=tuple(shots),
                scene_thumb_path="/thumbs/sc01.jpg",
            ),
        ),
    )


def _query() -> ScopedQuery:
    return ScopedQuery(
        query_id="q1",
        goal_id="g1",
        natural_query="Find the car.",
        scope=QueryScope(scene_ids=("sc01",), entity_hints=("car",), modality_hint=("visual",)),
        expected_evidence="A car is visible.",
        budget=QueryBudget(max_shots_to_verify=2, max_frames=16),
    )


def test_explore_batches_lowres_grids_and_sorts_candidate_picks(tmp_path: Path) -> None:
    backend = RecordingBackend(
        json.dumps(
            {
                "picks": [
                    {"shot_id": "sc01_sh001", "score": 0.2, "reason": "weak match"},
                    {"shot_id": "sc01_sh002", "score": 0.92, "reason": "car visible"},
                ]
            }
        )
    )

    picks = explore(query=_query(), index=_index(tmp_path), backend=backend, batch_size=16)

    assert [pick.shot_id for pick in picks] == ["sc01_sh002", "sc01_sh001"]
    assert picks.batch_count == 1
    assert picks.degraded is False
    assert backend.requests[0].task == "multi_v3_explore"
    assert [Path(path).name for path in backend.requests[0].frames] == ["sc01_sh001.jpg", "sc01_sh002.jpg", "sc01_sh003.jpg"]
    assert "ShotMeta" in backend.requests[0].prompt
    assert "ExpectedEvidence" not in backend.requests[0].prompt


def test_explore_filters_non_image_grid_paths_and_marks_degraded(tmp_path: Path) -> None:
    index = _index()
    scene = index.scenes[0]
    shots = (
        replace(scene.shots[0], lowres_grid_path=_write_image(tmp_path / "ok.jpg")),
        replace(scene.shots[1], lowres_grid_path="/grids/bad.json"),
        replace(scene.shots[2], lowres_grid_path=""),
    )
    index = replace(index, scenes=(replace(scene, shots=shots),))
    backend = RecordingBackend(json.dumps({"picks": []}))

    explore(query=_query(), index=index, backend=backend)

    assert [Path(path).name for path in backend.requests[0].frames] == ["ok.jpg"]
    assert backend.requests[0].media_type == "image"
    assert backend.requests[0].metadata["degraded"] is True


def test_explore_reports_actual_batch_count(tmp_path: Path) -> None:
    backend = RecordingBackend(json.dumps({"picks": []}))

    picks = explore(query=_query(), index=_index(tmp_path), backend=backend, batch_size=2)

    assert picks.batch_count == 2
    assert len(backend.requests) == 2


def test_verify_window_sends_focused_high_resolution_request(tmp_path: Path) -> None:
    backend = RecordingBackend(
        json.dumps(
            {
                "findings": [
                    {
                        "summary": "The car is visible.",
                        "supports_options": ["A"],
                        "refutes_options": ["C"],
                        "citation_ids": ["ev_0001"],
                        "confidence": 0.86,
                    }
                ]
            }
        )
    )
    request = VerifyRequest(
        shot_id="sc01_sh002",
        time_range=(20.0, 28.0),
        focus_claim="A car is visible.",
        sampling={"fps": 2, "max_frames": 16, "resolution": "high"},
        checks=({"target_id": "g1", "claim": "car visible", "polarity": "presence"},),
    )

    frame_1 = _write_image(tmp_path / "20.jpg")
    frame_2 = _write_image(tmp_path / "21.jpg")

    findings = verify_window(query_id="q1", request=request, frame_paths=(frame_1, frame_2), backend=backend)

    assert findings[0].finding_id == "ev_0001"
    assert findings[0].shot_id == "sc01_sh002"
    assert backend.requests[0].task == "multi_v3_verify_window"
    assert backend.requests[0].metadata["shot_id"] == "sc01_sh002"
    assert [Path(path).name for path in backend.requests[0].frames] == ["20.jpg", "21.jpg"]


def test_verify_window_filters_non_image_frames(tmp_path: Path) -> None:
    backend = RecordingBackend(json.dumps({"findings": []}))
    request = VerifyRequest(
        shot_id="sc01_sh002",
        time_range=(20.0, 28.0),
        focus_claim="A car is visible.",
        sampling={"fps": 2, "max_frames": 16, "resolution": "high"},
        checks=(),
    )

    verify_window(query_id="q1", request=request, frame_paths=("/frames/20.json", _write_image(tmp_path / "21.png")), backend=backend)

    assert [Path(path).name for path in backend.requests[0].frames] == ["21.png"]
    assert backend.requests[0].media_type == "image"


def test_verify_window_omits_image_media_type_when_no_image_frames() -> None:
    backend = RecordingBackend(json.dumps({"findings": []}))
    request = VerifyRequest(
        shot_id="sc01_sh002",
        time_range=(20.0, 28.0),
        focus_claim="A car is visible.",
        sampling={},
        checks=(),
    )

    verify_window(query_id="q1", request=request, frame_paths=("/frames/20.json",), backend=backend)

    assert list(backend.requests[0].frames) == []
    assert backend.requests[0].media_type is None
