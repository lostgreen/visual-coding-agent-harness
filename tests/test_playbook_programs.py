from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from visual_coding_agent_harness.agents.playbook_programs import PROGRAMS
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.playbook import Playbook
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.contracts.report import Finding
from visual_coding_agent_harness.workspace.investigator_ws import EvidenceRecordLedger
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
        Playbook.COVERAGE: "text",
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


def test_coverage_mode_collects_multiple_supporting_beats(tmp_path: Path) -> None:
    workspace = SpyWorkspace(_beats(tmp_path))
    backend = RecordingBackend()

    report = PROGRAMS[Playbook.COVERAGE].execute(query=_query(Playbook.COVERAGE), workspace=workspace, backend=backend)

    assert report.cost["verify_calls"] == 2
    assert report.verified_shots == ("sh001", "sh002")
    assert workspace.calls[0] == "text"


def test_coverage_mode_diversifies_candidates_by_chapter(tmp_path: Path) -> None:
    beats = (
        Beat("bt00001", "ch01", 0.0, 2.0, _image(tmp_path / "one.jpg"), "theme", (), ("sh001",)),
        Beat("bt00002", "ch01", 2.0, 4.0, _image(tmp_path / "two.jpg"), "theme", (), ("sh002",)),
        Beat("bt00003", "ch02", 4.0, 6.0, _image(tmp_path / "three.jpg"), "theme", (), ("sh003",)),
        Beat("bt00004", "ch03", 6.0, 8.0, _image(tmp_path / "four.jpg"), "theme", (), ("sh004",)),
    )
    workspace = SpyWorkspace(beats)
    backend = RecordingBackend()
    query = ScopedQuery(
        query_id="q_coverage",
        goal_id="g1",
        playbook=Playbook.COVERAGE,
        natural_query="theme",
        scope=QueryScope(chapter_ids=("ch01", "ch02", "ch03")),
        expected_evidence="distributed support across chapters",
        budget=QueryBudget(max_beats_to_verify=3, max_frames=2),
    )

    report = PROGRAMS[Playbook.COVERAGE].execute(query=query, workspace=workspace, backend=backend)

    assert report.verified_shots == ("sh001", "sh003", "sh004")
    assert report.cost["operator_count"] >= 3


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
    assert "Task query: red car" in second_backend.requests[0].prompt
    assert "Expected evidence: red car" in second_backend.requests[0].prompt
    assert "Playbook: identify_visual" in second_backend.requests[0].prompt


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


def test_locate_statement_creates_stable_asr_evidence_when_asr_supports_claim(tmp_path: Path) -> None:
    image = _image(tmp_path / "one.jpg")
    beat = Beat("bt00001", "ch01", 0.0, 2.0, image, "The mayor says the bridge is closed.", (), ("sh001",))
    workspace = SpyWorkspace((beat,))
    backend = RecordingBackend()
    ledger = EvidenceRecordLedger(tmp_path / "evidence_records.jsonl")
    query = ScopedQuery(
        query_id="q_asr",
        goal_id="g1",
        playbook=Playbook.LOCATE_STATEMENT,
        natural_query="bridge closed",
        text_queries=("bridge closed",),
        scope=QueryScope(chapter_ids=("ch01",)),
        expected_evidence="The bridge is closed.",
        budget=QueryBudget(max_beats_to_verify=1, max_frames=3),
    )

    def verify_fn(*, query_id: str, request, frame_paths, backend):
        del request, frame_paths, backend
        return (
            Finding("vlm_finding_1", query_id, "sh001", "The bridge closure is mentioned.", citation_ids=("dup",)),
            Finding("vlm_finding_2", query_id, "sh001", "The same quote supports the claim.", citation_ids=("dup",)),
        )

    PROGRAMS[Playbook.LOCATE_STATEMENT].execute(
        query=query,
        workspace=workspace,
        backend=backend,
        frame_sampler=lambda beat, max_frames: (image,),
        verify_fn=verify_fn,
        evidence_ledger=ledger,
    )

    records = ledger.read_all()
    assert [record.evidence_id for record in records] == ["ev_q_asr_bt00001_asr_001", "ev_q_asr_bt00001_asr_002"]
    assert {record.modality for record in records} == {"asr"}
    assert records[0].pointer == "bt00001"
    assert records[0].verbatim == "The mayor says the bridge is closed."


def test_read_text_creates_ocr_evidence_when_ocr_supports_claim(tmp_path: Path) -> None:
    image = _image(tmp_path / "one.jpg")
    beat = Beat("bt00001", "ch01", 0.0, 2.0, image, "", ("GATE 12",), ("sh001",))
    workspace = SpyWorkspace((beat,))
    backend = RecordingBackend()
    ledger = EvidenceRecordLedger(tmp_path / "evidence_records.jsonl")
    query = ScopedQuery(
        query_id="q_ocr",
        goal_id="g1",
        playbook=Playbook.READ_TEXT,
        natural_query="gate 12",
        text_queries=("gate 12",),
        scope=QueryScope(chapter_ids=("ch01",)),
        expected_evidence="The sign reads GATE 12.",
        budget=QueryBudget(max_beats_to_verify=1, max_frames=4),
    )

    PROGRAMS[Playbook.READ_TEXT].execute(
        query=query,
        workspace=workspace,
        backend=backend,
        frame_sampler=lambda beat, max_frames: (image,),
        verify_fn=lambda *, query_id, request, frame_paths, backend: (
            Finding("vlm_finding", query_id, "sh001", "The sign reads GATE 12.", citation_ids=("vlm_citation",)),
        ),
        evidence_ledger=ledger,
    )

    record = ledger.read_all()[0]
    assert record.evidence_id == "ev_q_ocr_bt00001_ocr_001"
    assert record.modality == "ocr"
    assert record.pointer == "bt00001"
    assert record.verbatim == "GATE 12"


def test_compare_playbook_searches_and_verifies_scope_b(tmp_path: Path) -> None:
    beats = (
        Beat("bt00001", "ch01", 0.0, 2.0, _image(tmp_path / "one.jpg"), "first car", (), ("sh001",)),
        Beat("bt00002", "ch02", 2.0, 4.0, _image(tmp_path / "two.jpg"), "second car", (), ("sh002",)),
    )
    workspace = SpyWorkspace(beats)
    backend = RecordingBackend()
    verified: list[str] = []
    query = ScopedQuery(
        query_id="q_compare",
        goal_id="g1",
        playbook=Playbook.COMPARE,
        natural_query="compare cars",
        scope=QueryScope(chapter_ids=("ch01",)),
        scope_b=QueryScope(chapter_ids=("ch02",)),
        expected_evidence="Compare the two cars.",
        budget=QueryBudget(max_beats_to_verify=2, max_frames=3),
    )

    def verify_fn(*, query_id: str, request, frame_paths, backend):
        del query_id, frame_paths, backend
        verified.append(request.shot_id)
        return ()

    report = PROGRAMS[Playbook.COMPARE].execute(query=query, workspace=workspace, backend=backend, verify_fn=verify_fn)

    assert set(verified) == {"sh001", "sh002"}
    assert set(report.explored_shots) == {"sh001", "sh002"}
