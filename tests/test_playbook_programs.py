from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from visual_coding_agent_harness.agents.playbook_programs import PROGRAMS
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.playbook import Playbook
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.workspace.video_workspace import Beat, Chapter
from visual_coding_agent_harness.workspace.visual_index import BeatHit
from visual_coding_agent_harness.workspace.memo import MemoStore


class RecordingBackend:
    def __init__(self, *, explore_response: str | None = None) -> None:
        self.requests: list[BackendRequest] = []
        self.explore_response = explore_response

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if request.task == "playbook_explore" and self.explore_response is not None:
            return BackendResponse(text=self.explore_response)
        return BackendResponse(
            text=json.dumps(
                {
                    "findings": [
                        {
                            "summary": "supporting evidence found",
                            "supports_options": ["A"],
                            "citation_ids": [f"ev_{len(self.requests)}"],
                            "confidence": 0.9,
                        }
                    ]
                }
            )
        )


class SpyWorkspace:
    def __init__(self, beats: Sequence[Beat]) -> None:
        self.video_path = "/videos/demo.mp4"
        self.duration_sec = 4.0
        self.beats = tuple(beats)
        self.chapters = (Chapter("ch01", 0.0, 4.0, tuple(beat.beat_id for beat in beats), beats[0].keyframe_path),)
        self.calls: list[str] = []
        self.text_queries: list[str] = []
        self.visual_queries: list[str] = []

    def search_text(self, query: str, *, modality=("asr", "ocr")):
        del modality
        self.calls.append("text")
        self.text_queries.append(query)
        return tuple(BeatHit(beat.beat_id, 1.0, "text") for beat in self.beats)

    def search_visual(self, query: str, k: int = 20):
        del k
        self.calls.append("visual")
        self.visual_queries.append(query)
        return tuple(BeatHit(beat.beat_id, 1.0, "visual") for beat in self.beats)

    def get_beat(self, beat_id: str) -> Beat:
        for beat in self.beats:
            if beat.beat_id == beat_id:
                return beat
        raise ValueError(beat_id)


def _image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)
    return str(path)


def _beats(tmp_path: Path) -> tuple[Beat, Beat]:
    return (
        Beat("bt00001", "ch01", 0.0, 2.0, _image(tmp_path / "one.jpg"), "red car", (), ("sh001",)),
        Beat("bt00002", "ch01", 2.0, 4.0, _image(tmp_path / "two.jpg"), "red car again", (), ("sh002",)),
    )


def _query(playbook: Playbook) -> ScopedQuery:
    return ScopedQuery(
        query_id="q1",
        goal_id="g1",
        playbook=playbook,
        natural_query="red car",
        scope=QueryScope(chapter_ids=("ch01",)),
        expected_evidence="red car",
        budget=QueryBudget(max_beats_to_verify=2, max_frames=4),
    )


def test_playbook_programs_use_expected_search_order(tmp_path: Path) -> None:
    expected_first = {
        Playbook.LOCATE_STATEMENT: "text",
        Playbook.READ_TEXT: "text",
        Playbook.ORDER_ACTIONS: "visual",
        Playbook.IDENTIFY_VISUAL: "visual",
        Playbook.COUNT: "visual",
        Playbook.COMPARE: "text",
    }
    for playbook, first_call in expected_first.items():
        workspace = SpyWorkspace(_beats(tmp_path / playbook.value))
        backend = RecordingBackend()

        PROGRAMS[playbook].execute(query=_query(playbook), workspace=workspace, backend=backend)

        assert workspace.calls[0] == first_call


def test_count_playbook_does_not_stop_after_first_supporting_finding(tmp_path: Path) -> None:
    workspace = SpyWorkspace(_beats(tmp_path))
    backend = RecordingBackend()

    report = PROGRAMS[Playbook.COUNT].execute(query=_query(Playbook.COUNT), workspace=workspace, backend=backend)

    assert report.cost["verify_calls"] == 2
    assert report.verified_shots == ("sh001", "sh002")


def test_playbook_program_writes_and_reuses_observation_memos(tmp_path: Path) -> None:
    workspace = SpyWorkspace(_beats(tmp_path))
    store = MemoStore(tmp_path / "memos.jsonl")
    backend = RecordingBackend(
        explore_response=json.dumps(
            {
                "observations": [
                    {"beat_id": "bt00001", "observation": "a red car parked near the curb"},
                ]
            }
        )
    )

    PROGRAMS[Playbook.IDENTIFY_VISUAL].execute(
        query=_query(Playbook.IDENTIFY_VISUAL),
        workspace=workspace,
        backend=backend,
        memo_store=store,
    )
    assert store.get("bt00001")[0].observation == "a red car parked near the curb"

    second_backend = RecordingBackend(explore_response=json.dumps({"observations": []}))
    PROGRAMS[Playbook.IDENTIFY_VISUAL].execute(
        query=_query(Playbook.IDENTIFY_VISUAL),
        workspace=workspace,
        backend=second_backend,
        memo_store=store,
    )

    assert "previous observation" in second_backend.requests[0].prompt


def test_playbook_program_routes_split_queries_to_matching_indexes(tmp_path: Path) -> None:
    workspace = SpyWorkspace(_beats(tmp_path))
    backend = RecordingBackend()
    query = ScopedQuery(
        query_id="q_split",
        goal_id="g1",
        playbook=Playbook.LOCATE_STATEMENT,
        natural_query="fallback text",
        text_queries=("spoken phrase near the podium",),
        visual_queries=("wide shot of the podium",),
        scope=QueryScope(chapter_ids=("ch01",)),
        expected_evidence="The podium is visible.",
        budget=QueryBudget(max_beats_to_verify=2, max_frames=4),
    )

    PROGRAMS[Playbook.LOCATE_STATEMENT].execute(query=query, workspace=workspace, backend=backend)

    assert workspace.text_queries == ["spoken phrase near the podium"]
    assert workspace.visual_queries == ["wide shot of the podium"]


def test_dense_playbook_passes_resolution_and_dense_sampling_to_frame_sampler(tmp_path: Path) -> None:
    workspace = SpyWorkspace(_beats(tmp_path))
    backend = RecordingBackend()
    sampler_calls: list[tuple[str, int, str, bool]] = []
    verify_frames: list[tuple[str, ...]] = []

    def sampler(beat: Beat, max_frames: int, *, resolution: str, dense: bool) -> tuple[str, ...]:
        sampler_calls.append((beat.beat_id, max_frames, resolution, dense))
        return tuple(f"/frames/{beat.beat_id}_{idx}.jpg" for idx in range(max_frames))

    def verify_fn(*, query_id: str, request, frame_paths, backend):
        del query_id, backend
        verify_frames.append(tuple(frame_paths))
        assert request.sampling["resolution"] == "high"
        assert request.sampling["dense"] is True
        return ()

    PROGRAMS[Playbook.COUNT].execute(
        query=_query(Playbook.COUNT),
        workspace=workspace,
        backend=backend,
        frame_sampler=sampler,
        verify_fn=verify_fn,
    )

    assert sampler_calls[0] == ("bt00001", 4, "high", True)
    assert len(verify_frames[0]) == 4
