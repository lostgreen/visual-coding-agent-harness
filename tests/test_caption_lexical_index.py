from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from vcah.caption_lexical_index import CaptionLexicalIndex, render_caption_hits
from vcah.caption_schema import CaptionPassageV1, passage_to_dict
from vcah.interactive_agents import VisionInvestigator, _rema_caption_queries
from vcah.multiround import InvestigationTask
from vcah.virtual_index import build_workspace_overview
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)
from vcah.workspace import ObservationLog


class DummyApi:
    model = "unused"


def _passages() -> tuple[CaptionPassageV1, ...]:
    return (
        CaptionPassageV1(
            "p0",
            "c0",
            "The player opens the red temple door.",
            10.0,
            20.0,
            10.0,
            0,
            {"interval_precision": "anchor", "source_segments": ["seg_0001"]},
        ),
        CaptionPassageV1(
            "p1",
            "c0",
            "A small creature runs through the doorway.",
            20.0,
            30.0,
            20.0,
            1,
            {"interval_precision": "anchor", "source_segments": ["seg_0002"]},
        ),
        CaptionPassageV1(
            "p2",
            "c1",
            "The hero rides a cloud above the mountain.",
            100.0,
            120.0,
            100.0,
            0,
            {"interval_precision": "anchor", "source_segments": ["seg_0002"]},
        ),
        CaptionPassageV1(
            "p3",
            "c2",
            "角色打开红色寺庙大门。",
            200.0,
            210.0,
            200.0,
            0,
            {"interval_precision": "anchor", "source_segments": ["seg_0002"]},
        ),
    )


def _workspace(
    tmp_path: Path,
    *,
    with_captions: bool = True,
    question: str = "Where is the red door?",
) -> VirtualVideoWorkspace:
    manifest = VirtualVideoManifest(
        workspace_id="mmlifelong-game",
        segments=(
            VirtualVideoSegment("seg_0001", "video-a", "video-a.mp4", 0.0, 20.0, 0.0, 20.0),
            VirtualVideoSegment("seg_0002", "video-b", "video-b.mp4", 0.0, 280.0, 20.0, 300.0),
        ),
    )
    case = VirtualVideoCase(case_id="case", question=question)
    workspace = VirtualVideoWorkspace.create(tmp_path / "case", manifest=manifest, case=case)
    if with_captions:
        captions = workspace.asset_root / "captions"
        captions.mkdir()
        (captions / "passages.cfg.jsonl").write_text(
            "".join(json.dumps(passage_to_dict(row), ensure_ascii=False) + "\n" for row in _passages()),
            encoding="utf-8",
        )
    return workspace


def test_lexical_search_exact_phrase_time_filter_and_chinese() -> None:
    index = CaptionLexicalIndex(_passages(), config_digest="cfg")

    exact = index.search(("red temple door",), top_k=3)
    filtered = index.search(("red temple door",), top_k=3, time_range=(90.0, 130.0))
    chinese = index.search(("红色寺庙大门",), top_k=2)

    assert exact[0].passage_id == "p0"
    assert exact[0].lexical_score == exact[0].fused_score
    assert filtered == ()
    assert chinese[0].passage_id == "p3"


def test_lexical_neighbor_expansion_and_query_fingerprint_are_deterministic() -> None:
    index = CaptionLexicalIndex(_passages(), config_digest="cfg")

    hits = index.search(("red temple door",), top_k=1, expand_neighbors=1)
    first = index.query_fingerprint(
        ("red temple door",),
        top_k=1,
        time_range=None,
        expand_neighbors=1,
    )
    repeated = index.query_fingerprint(
        ("red temple door",),
        top_k=1,
        time_range=None,
        expand_neighbors=1,
    )
    changed_index = CaptionLexicalIndex((*_passages(), _passages()[0]), config_digest="cfg-v2")
    changed = changed_index.query_fingerprint(
        ("red temple door",),
        top_k=1,
        time_range=None,
        expand_neighbors=1,
    )

    assert [hit.passage_id for hit in hits] == ["p0", "p1"]
    assert hits[1].metadata["neighbor_of"] == "p0"
    assert first == repeated
    assert changed != first
    assert len(render_caption_hits(hits, char_limit=120)) <= 120


def test_lexical_segment_scope_filters_hits_neighbors_and_fingerprint() -> None:
    index = CaptionLexicalIndex(_passages(), config_digest="cfg")
    queries = ("red temple door", "红色寺庙大门")

    first_segment = index.search(
        queries,
        top_k=3,
        segment_ids=("seg_0001",),
        expand_neighbors=1,
    )
    second_segment = index.search(
        queries,
        top_k=3,
        segment_ids=("seg_0002",),
        expand_neighbors=1,
    )
    first_fingerprint = index.query_fingerprint(
        queries,
        top_k=3,
        time_range=None,
        segment_ids=("seg_0001",),
        expand_neighbors=1,
    )
    second_fingerprint = index.query_fingerprint(
        queries,
        top_k=3,
        time_range=None,
        segment_ids=("seg_0002",),
        expand_neighbors=1,
    )

    assert [hit.passage_id for hit in first_segment] == ["p0"]
    assert [hit.passage_id for hit in second_segment] == ["p3"]
    assert first_fingerprint != second_fingerprint


def test_search_caption_creates_one_locator_attempt_and_caches_zero_hits(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = VisionInvestigator(
        workspace,
        api=DummyApi(),
        trace_path=tmp_path / "trace.jsonl",
    )
    task = InvestigationTask(
        query_id="caption-1",
        goal="Locate the red door.",
        inspection_mode="search_caption",
        caption_queries=("red temple door",),
        top_k=3,
        index_mode="lexical",
    )

    report = investigator._investigate_task(task)
    attempt = report.attempts[0]
    row = ObservationLog(tmp_path / "observation_log.jsonl").append_attempt(attempt, round_id=1)
    repeated = investigator._investigate_task(task)
    zero_workspace = VirtualVideoWorkspace.create(
        tmp_path / "zero-case",
        manifest=workspace.manifest,
        case=VirtualVideoCase(
            case_id="zero-caption-case",
            question="xylophonequasar nebularift",
            options={},
            gold="",
            target_segment_id="seg_0001",
            target_virtual_interval=(0.0, 30.0),
        ),
        asset_root=workspace.asset_root,
    )
    zero_investigator = VisionInvestigator(
        zero_workspace,
        api=DummyApi(),
        trace_path=tmp_path / "zero-trace.jsonl",
    )
    zero_task = InvestigationTask(
        query_id="caption-zero",
        goal="Locate a submarine.",
        inspection_mode="search_caption",
        caption_queries=("xylophonequasar nebularift",),
    )
    zero = zero_investigator._investigate_task(zero_task)
    zero_repeated = zero_investigator._investigate_task(zero_task)

    assert report.status == "completed"
    assert report.evidence == ()
    assert attempt.modality == "caption_search"
    assert attempt.inspected_ranges == ()
    assert attempt.sampling_config["queries"][0] == workspace.case.question
    assert attempt.sampling_config["hits"][0]["passage_id"] == "p0"
    assert row["attempt_id"] == attempt.attempt_id
    assert repeated.cost["reused"] is True
    assert repeated.attempts == ()
    assert zero.cost["zero_hits"] is True
    assert zero.cost["consumes_budget"] is False
    assert zero_repeated.cost["reused"] is True


def test_search_caption_scope_is_forwarded_and_not_reused_across_segments(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    investigator = VisionInvestigator(
        workspace,
        api=DummyApi(),
        trace_path=tmp_path / "trace-scoped.jsonl",
    )
    common = {
        "goal": "Locate the red door.",
        "inspection_mode": "search_caption",
        "caption_queries": ("red temple door", "红色寺庙大门"),
        "top_k": 3,
        "index_mode": "lexical",
    }

    first = investigator._investigate_task(
        InvestigationTask(query_id="caption-seg-1", segment_id="seg_0001", **common)
    )
    second = investigator._investigate_task(
        InvestigationTask(query_id="caption-seg-2", segment_id="seg_0002", **common)
    )

    first_attempt = first.attempts[0]
    second_attempt = second.attempts[0]
    assert first.cost["reused"] is False
    assert second.cost["reused"] is False
    assert first_attempt.sampling_config["segment_ids"] == ["seg_0001"]
    assert second_attempt.sampling_config["segment_ids"] == ["seg_0002"]
    assert first_attempt.sampling_config["source_video_ids"] == ["video-a"]
    assert second_attempt.sampling_config["source_video_ids"] == ["video-b"]
    assert [hit["passage_id"] for hit in first_attempt.sampling_config["hits"]] == ["p0"]
    second_ids = [hit["passage_id"] for hit in second_attempt.sampling_config["hits"]]
    assert second_ids[0] == "p3"
    assert "p0" not in second_ids


def test_rema_caption_queries_exclude_full_question_and_allow_refinement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    investigator = VisionInvestigator(
        workspace,
        api=DummyApi(),
        trace_path=tmp_path / "trace-rema.jsonl",
        caption_query_strategy="rema",
    )
    calls: list[tuple[str, ...]] = []
    top_ks: list[int] = []

    def search(queries: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
        normalized = tuple(str(query) for query in queries)
        calls.append(normalized)
        top_ks.append(int(kwargs["top_k"]))
        fingerprint = "-".join(normalized)
        return {
            "hits": [],
            "query_fingerprint": fingerprint,
            "index_digest": "rema-index",
            "config_digest": "fixture",
            "rendered": "",
            "segment_ids": [],
            "source_video_ids": [],
            "query_strategy": "rema",
        }

    monkeypatch.setattr(investigator, "search_caption", search)
    first = investigator._investigate_task(
        InvestigationTask(
            query_id="rema-1",
            goal="Locate the target.",
            inspection_mode="search_caption",
            caption_queries=("red temple door", "红色寺庙大门"),
            top_k=5,
        )
    )
    refined = investigator._investigate_task(
        InvestigationTask(
            query_id="rema-2",
            goal="Refine the target.",
            inspection_mode="search_caption",
            caption_queries=("red temple doorway", "红色寺庙大门"),
            top_k=5,
        )
    )

    assert calls == [
        ("red temple door", "红色寺庙大门", "the target"),
        ("red temple doorway", "红色寺庙大门", "the target"),
    ]
    assert top_ks == [8, 8]
    assert workspace.case.question not in calls[0]
    assert first.cost["reused"] is False
    assert refined.cost["reused"] is False
    assert first.attempts[0].sampling_config["query_strategy"] == "rema"
    assert first.attempts[0].sampling_config["top_k"] == 8
    assert first.attempts[0].sampling_config["requested_top_k"] == 5


def test_rema_caption_queries_split_temporal_goal_before_entity_terms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    investigator = VisionInvestigator(
        workspace,
        api=DummyApi(),
        trace_path=tmp_path / "trace-rema-temporal.jsonl",
        caption_query_strategy="rema",
    )
    captured: list[str] = []

    def search(queries: Sequence[str], **_kwargs: Any) -> Mapping[str, Any]:
        captured.extend(queries)
        return {
            "hits": [],
            "query_fingerprint": "temporal",
            "index_digest": "rema-index",
            "config_digest": "fixture",
            "rendered": "",
            "segment_ids": [],
            "source_video_ids": [],
            "query_strategy": "rema",
        }

    monkeypatch.setattr(investigator, "search_caption", search)
    investigator._investigate_task(
        InvestigationTask(
            query_id="rema-temporal",
            goal="Locate the chapter start and the first fight against Yin Tiger.",
            inspection_mode="search_caption",
            caption_queries=("Yin Tiger", "Flaming Mountains"),
        )
    )

    assert captured == [
        "the player first fights Yin Tiger",
        "the first fight against Yin Tiger",
        "Yin Tiger",
        "Flaming Mountains",
        "the chapter start",
    ]


def test_rema_caption_queries_extract_temporal_contract_from_question() -> None:
    queries = _rema_caption_queries(
        "Find the Flaming Mountains chapter.",
        ("Flaming Mountains",),
        fallback=(
            "After the player enters the Flaming Mountains chapter, what are the values "
            "before the first challenge against Yin Tiger?"
        ),
    )

    assert queries[:3] == (
        "the player first challenges Yin Tiger",
        "the player enters the Flaming Mountains chapter",
        "the first challenge against Yin Tiger",
    )


def test_rema_temporal_locator_selects_first_target_after_shared_chapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    question = (
        "After the player enters the Flaming Mountains chapter, what are the values "
        "before the first challenge against Yin Tiger?"
    )
    workspace = _workspace(tmp_path, question=question)
    investigator = VisionInvestigator(
        workspace,
        api=DummyApi(),
        trace_path=tmp_path / "trace-rema-locator.jsonl",
        caption_query_strategy="rema",
    )
    calls: list[dict[str, Any]] = []

    def hit(
        passage_id: str,
        start: float,
        end: float,
        text: str,
        query: str,
        rank: int,
    ) -> dict[str, Any]:
        return {
            "passage_id": passage_id,
            "caption_id": "caption",
            "rank": rank,
            "lexical_score": 1.0,
            "dense_score": 1.0,
            "fused_score": 1.0 / (60 + rank),
            "virtual_start_sec": start,
            "virtual_end_sec": end,
            "wall_clock_begin": None,
            "wall_clock_end": None,
            "text": text,
            "interval_precision": "anchor",
            "source_pointer": f"caption://fixture/{passage_id}",
            "metadata": {"query_matches": [{"query": query.casefold(), "rank": rank}]},
        }

    target_query = "the player first challenges Yin Tiger"
    targets = [
        hit("late", 68920.0, 68967.0, "The later tiger fight continues.", target_query, 1),
        hit("early", 17844.0, 17868.0, "An earlier tiger fight continues.", target_query, 2),
        hit("correct", 66215.0, 66300.0, "The sparring match begins.", target_query, 3),
    ]
    chapter = hit(
        "chapter-5",
        63135.0,
        63144.0,
        "White calligraphy appears, reading Chapter 5: Sunset in the Mortal World.",
        "a chapter title appears",
        2,
    )

    def search(queries: Sequence[str], **kwargs: Any) -> Mapping[str, Any]:
        calls.append({"queries": tuple(queries), **kwargs})
        scoped_end = (kwargs.get("time_range") or (None, None))[1]
        hits = targets if scoped_end is None else ([chapter] if scoped_end in {66215.0, 68920.0} else [])
        return {
            "hits": hits,
            "query_fingerprint": f"search-{len(calls)}",
            "index_digest": "rema-index",
            "config_digest": "fixture",
            "rendered": "",
            "segment_ids": [],
            "source_video_ids": [],
            "query_strategy": "rema",
        }

    monkeypatch.setattr(investigator, "search_caption", search)
    report = investigator._investigate_task(
        InvestigationTask(
            query_id="rema-temporal-locator",
            goal="Find the Flaming Mountains chapter.",
            inspection_mode="search_caption",
            caption_queries=("Flaming Mountains",),
            top_k=5,
        )
    )

    locator = report.attempts[0].sampling_config["temporal_locator"]
    recommended = locator["recommended"]
    assert calls[0]["top_k"] == 12
    assert [call["top_k"] for call in calls[1:]] == [6, 6, 6]
    assert recommended["scope_anchor"]["time_range"] == [63135.0, 63144.0]
    assert recommended["target_event"]["time_range"] == [66215.0, 66300.0]
    assert recommended["inspection_range"] == [66125.0, 66190.0]
    assert recommended["target_candidate_count"] == 2


def test_no_caption_workspace_does_not_advertise_caption_navigation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, with_captions=False)

    overview = build_workspace_overview(workspace, frame_refs=())

    assert overview["available_navigation"] == ["search_asr"]
    assert not workspace.cold_index_dir.exists()
