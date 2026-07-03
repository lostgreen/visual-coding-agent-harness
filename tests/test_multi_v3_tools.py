from __future__ import annotations

import json

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.contracts.report import VerifyRequest
from visual_coding_agent_harness.tools.explore import explore
from visual_coding_agent_harness.tools.verify import verify_window
from visual_coding_agent_harness.video.index import Frame, Scene, Shot, VideoIndex


class RecordingBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.response)


def _index() -> VideoIndex:
    shots = []
    for idx in range(1, 4):
        shot_id = f"sc01_sh{idx:03d}"
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
                lowres_grid_path=f"/grids/{shot_id}.jpg",
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


def test_explore_batches_lowres_grids_and_sorts_candidate_picks() -> None:
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

    picks = explore(query=_query(), index=_index(), backend=backend, batch_size=16)

    assert [pick.shot_id for pick in picks] == ["sc01_sh002", "sc01_sh001"]
    assert backend.requests[0].task == "multi_v3_explore"
    assert list(backend.requests[0].frames) == ["/grids/sc01_sh001.jpg", "/grids/sc01_sh002.jpg", "/grids/sc01_sh003.jpg"]
    assert "ShotMeta" in backend.requests[0].prompt


def test_verify_window_sends_focused_high_resolution_request() -> None:
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

    findings = verify_window(query_id="q1", request=request, frame_paths=("/frames/20.jpg", "/frames/21.jpg"), backend=backend)

    assert findings[0].finding_id == "ev_0001"
    assert findings[0].shot_id == "sc01_sh002"
    assert backend.requests[0].task == "multi_v3_verify_window"
    assert backend.requests[0].metadata["shot_id"] == "sc01_sh002"
    assert list(backend.requests[0].frames) == ["/frames/20.jpg", "/frames/21.jpg"]
