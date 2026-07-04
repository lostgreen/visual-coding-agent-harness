from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.contracts.report import VerifyRequest
from visual_coding_agent_harness.tools.vlm_tools import explore
from visual_coding_agent_harness.tools.vlm_tools import verify_window
from visual_coding_agent_harness.workspace.text_index import InvertedIndex
from visual_coding_agent_harness.workspace.video_workspace import Beat, Chapter, VideoWorkspace
from visual_coding_agent_harness.workspace.visual_index import VisualIndex


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


class EmptyEmbeddingBackend:
    embedding_dim = 1

    def encode_images(self, paths):
        raise AssertionError("not used")

    def encode_text(self, queries):
        raise AssertionError("not used")


def _workspace(image_dir: Path | None = None) -> VideoWorkspace:
    beats = []
    for idx in range(1, 4):
        beat_id = f"bt{idx:05d}"
        keyframe_path = f"/grids/{beat_id}.jpg"
        if image_dir is not None:
            keyframe_path = _write_image(image_dir / f"{beat_id}.jpg")
        beats.append(
            Beat(
                beat_id=beat_id,
                chapter_id="ch01",
                start_sec=float(idx * 10),
                end_sec=float(idx * 10 + 8),
                keyframe_path=keyframe_path,
                asr_verbatim=f"Beat {idx} transcript.",
                ocr_verbatim=(),
                shot_ids=(f"sc01_sh{idx:03d}",),
            )
        )
    return VideoWorkspace(
        video_path="/videos/demo.mp4",
        duration_sec=60.0,
        chapters=(
            Chapter("ch01", 0.0, 60.0, tuple(beat.beat_id for beat in beats), beats[0].keyframe_path),
        ),
        beats=tuple(beats),
        text_index=InvertedIndex(),
        visual_index=VisualIndex(EmptyEmbeddingBackend()),
    )


def _query() -> ScopedQuery:
    return ScopedQuery(
        query_id="q1",
        goal_id="g1",
        natural_query="Find the car.",
        scope=QueryScope(chapter_ids=("ch01",), entity_hints=("car",), modality_hint=("visual",)),
        expected_evidence="A car is visible.",
        budget=QueryBudget(max_shots_to_verify=2, max_frames=16),
    )


def test_explore_batches_lowres_grids_and_sorts_candidate_picks(tmp_path: Path) -> None:
    backend = RecordingBackend(
        json.dumps(
            {
                "picks": [
                    {"shot_id": "sc01_sh001", "score": 0.2, "reason": "weak match"},
                    {"shot_id": "bt00002", "score": 0.92, "reason": "car visible"},
                ]
            }
        )
    )

    picks = explore(query=_query(), workspace=_workspace(tmp_path), backend=backend, batch_size=16)

    assert [pick.shot_id for pick in picks] == ["sc01_sh002", "sc01_sh001"]
    assert picks.batch_count == 1
    assert picks.degraded is False
    assert backend.requests[0].task == "multi_v3_explore"
    assert [Path(path).name for path in backend.requests[0].frames] == ["bt00001.jpg", "bt00002.jpg", "bt00003.jpg"]
    assert "BeatMeta" in backend.requests[0].prompt
    assert "ExpectedEvidence" not in backend.requests[0].prompt


def test_explore_filters_non_image_grid_paths_and_marks_degraded(tmp_path: Path) -> None:
    workspace = _workspace()
    beats = (
        Beat("bt00001", "ch01", 10.0, 18.0, _write_image(tmp_path / "ok.jpg"), "Beat 1 transcript.", (), ("sc01_sh001",)),
        Beat("bt00002", "ch01", 20.0, 28.0, "/grids/bad.json", "Beat 2 transcript.", (), ("sc01_sh002",)),
        Beat("bt00003", "ch01", 30.0, 38.0, "", "Beat 3 transcript.", (), ("sc01_sh003",)),
    )
    workspace.beats = beats
    workspace.chapters = (Chapter("ch01", 0.0, 60.0, tuple(beat.beat_id for beat in beats), beats[0].keyframe_path),)
    backend = RecordingBackend(json.dumps({"picks": []}))

    explore(query=_query(), workspace=workspace, backend=backend)

    assert [Path(path).name for path in backend.requests[0].frames] == ["ok.jpg"]
    assert backend.requests[0].media_type == "image"
    assert backend.requests[0].metadata["degraded"] is True


def test_explore_reports_actual_batch_count(tmp_path: Path) -> None:
    backend = RecordingBackend(json.dumps({"picks": []}))

    picks = explore(query=_query(), workspace=_workspace(tmp_path), backend=backend, batch_size=2)

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
