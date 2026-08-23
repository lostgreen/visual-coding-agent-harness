from __future__ import annotations

from vcah.caption_context import expand_query_conditioned_context
from vcah.caption_evidence_bundle import build_caption_evidence_bundle_set
from vcah.caption_lexical_index import CaptionLexicalIndex
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1
from vcah.investigator import VirtualVideoInvestigator
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)


def _passage(
    passage_id: str,
    caption_id: str,
    start: float,
    end: float,
    segment_id: str,
) -> CaptionPassageV1:
    return CaptionPassageV1(
        passage_id=passage_id,
        caption_id=caption_id,
        text=f"Visible event in {passage_id}.",
        virtual_start_sec=start,
        virtual_end_sec=end,
        anchor_virtual_sec=start,
        ordinal=0,
        metadata={
            "interval_precision": "anchor",
            "source_segments": [segment_id],
        },
    )


def _hit(passage: CaptionPassageV1) -> CaptionHitV1:
    return CaptionHitV1(
        passage_id=passage.passage_id,
        caption_id=passage.caption_id,
        rank=1,
        lexical_score=1.0,
        dense_score=None,
        fused_score=1.0,
        virtual_start_sec=passage.virtual_start_sec,
        virtual_end_sec=passage.virtual_end_sec,
        wall_clock_begin=None,
        wall_clock_end=None,
        text=passage.text,
        interval_precision="anchor",
        source_pointer=f"caption://cfg/{passage.passage_id}",
        metadata={
            **dict(passage.metadata),
            "source_video_ids": ["video-a"],
        },
    )


def test_context_expansion_crosses_caption_chunks_on_same_source() -> None:
    defeat = _passage("cap59:p5", "cap59", 17_809.0, 17_844.0, "seg-a")
    singing = _passage("cap60:p0", "cap60", 18_000.0, 18_047.0, "seg-b")
    other_source = _passage("other:p0", "other", 18_048.0, 18_060.0, "seg-c")

    expanded = expand_query_conditioned_context(
        (defeat, singing, other_source),
        (_hit(defeat),),
        distance=1,
        time_range=None,
        index_digest="index",
        config_digest="cfg",
        source_video_id_by_segment={
            "seg-a": "video-a",
            "seg-b": "video-a",
            "seg-c": "video-b",
        },
        max_gap_sec=180.0,
    )

    assert [hit.passage_id for hit in expanded] == ["cap59:p5", "cap60:p0"]
    assert expanded[0].metadata.get("context_expansion_contract") is None
    assert expanded[1].metadata["neighbor_of"] == "cap59:p5"
    assert expanded[1].metadata["cross_caption"] is True
    assert expanded[1].metadata["candidate_only"] is True
    assert expanded[1].metadata["context_links"] == [
        {
            "seed_passage_id": "cap59:p5",
            "seed_rank": 1,
            "offset": 1,
            "edge_gap_sec": 156.0,
            "same_source_timeline": True,
            "source_match_basis": "source_video_id",
            "shared_source_video_ids": ["video-a"],
            "shared_segment_ids": [],
        }
    ]

    bundle_set = build_caption_evidence_bundle_set(expanded)
    bundle = bundle_set["bundles"][0]
    assert bundle["seed_passage_ids"] == ["cap59:p5"]
    assert bundle["context_passage_ids"] == ["cap60:p0"]
    assert [row["role"] for row in bundle["member_passages"]] == [
        "seed",
        "context",
    ]
    assert bundle["event_boundaries_preserved"] is True
    assert bundle["semantic_claim"] == "temporally_related_evidence_not_single_event"
    assert bundle["member_passages"][1]["context_links"][0][
        "same_source_timeline"
    ] is True


def test_context_expansion_respects_gap_and_source_boundaries() -> None:
    seed = _passage("seed", "cap-a", 100.0, 110.0, "seg-a")
    too_far = _passage("far", "cap-b", 300.0, 310.0, "seg-b")
    wrong_source = _passage("wrong", "cap-c", 111.0, 120.0, "seg-c")

    expanded = expand_query_conditioned_context(
        (seed, wrong_source, too_far),
        (_hit(seed),),
        distance=3,
        time_range=None,
        index_digest="index",
        config_digest="cfg",
        source_video_id_by_segment={
            "seg-a": "video-a",
            "seg-b": "video-a",
            "seg-c": "video-b",
        },
        max_gap_sec=120.0,
    )

    assert [hit.passage_id for hit in expanded] == ["seed"]


def test_context_expansion_keeps_all_seeds_ahead_of_context() -> None:
    first = _passage("first", "cap-a", 10.0, 20.0, "seg-a")
    middle = _passage("middle", "cap-b", 21.0, 30.0, "seg-a")
    last = _passage("last", "cap-c", 31.0, 40.0, "seg-a")
    first_hit = _hit(first)
    last_hit = CaptionHitV1(
        **{
            **first_hit.__dict__,
            "passage_id": last.passage_id,
            "caption_id": last.caption_id,
            "rank": 2,
            "virtual_start_sec": last.virtual_start_sec,
            "virtual_end_sec": last.virtual_end_sec,
            "text": last.text,
        }
    )

    expanded = expand_query_conditioned_context(
        (first, middle, last),
        (first_hit, last_hit),
        distance=1,
        time_range=None,
        index_digest="index",
        config_digest="cfg",
        source_video_id_by_segment={"seg-a": "video-a"},
    )

    assert [hit.passage_id for hit in expanded] == ["first", "last", "middle"]
    assert expanded[2].metadata["context_seed_passage_ids"] == ["first", "last"]


def test_context_expansion_can_be_directional() -> None:
    before = _passage("before", "cap-a", 10.0, 20.0, "seg-a")
    anchor = _passage("anchor", "cap-a", 21.0, 30.0, "seg-a")
    after = _passage("after", "cap-a", 31.0, 40.0, "seg-a")

    expanded = expand_query_conditioned_context(
        (before, anchor, after),
        (_hit(anchor),),
        distance=1,
        time_range=None,
        index_digest="index",
        config_digest="cfg",
        source_video_id_by_segment={"seg-a": "video-a"},
        direction="after",
    )

    assert [hit.passage_id for hit in expanded] == ["anchor", "after"]
    assert expanded[1].metadata["neighbor_offset"] == 1
    assert expanded[1].metadata["context_direction"] == "after"


def test_caption_search_context_is_opt_in_and_emits_bundle(tmp_path) -> None:
    defeat = CaptionPassageV1(
        "cap59:p5",
        "cap59",
        "Tiger Vanguard is defeated and dissolves into blood.",
        17_809.0,
        17_844.0,
        17_809.0,
        5,
        {"interval_precision": "anchor", "source_segments": ["seg-a"]},
    )
    singing = CaptionPassageV1(
        "cap60:p0",
        "cap60",
        "The Headless One appears and sings the first line.",
        18_000.0,
        18_047.0,
        18_000.0,
        0,
        {"interval_precision": "anchor", "source_segments": ["seg-b"]},
    )
    workspace = VirtualVideoWorkspace.create(
        tmp_path / "case",
        manifest=VirtualVideoManifest(
            workspace_id="game",
            segments=(
                VirtualVideoSegment(
                    "seg-a", "video-a", "video.mp4", 0.0, 17_850.0, 0.0, 17_850.0
                ),
                VirtualVideoSegment(
                    "seg-b",
                    "video-a",
                    "video.mp4",
                    17_850.0,
                    18_100.0,
                    17_850.0,
                    18_100.0,
                ),
            ),
        ),
        case=VirtualVideoCase(
            case_id="tiger",
            question="After defeating Tiger Vanguard, what does the Headless One sing?",
        ),
    )
    investigator = VirtualVideoInvestigator(
        workspace,
        caption_index=CaptionLexicalIndex((defeat, singing), config_digest="cfg"),
    )

    baseline = investigator.search_caption(
        ("Tiger Vanguard defeated",), top_k=1, index_mode="lexical"
    )
    expanded = investigator.search_caption(
        ("Tiger Vanguard defeated",),
        top_k=1,
        context_neighbors=1,
        context_max_gap_sec=180.0,
        index_mode="lexical",
    )

    assert [row["passage_id"] for row in baseline["hits"]] == ["cap59:p5"]
    assert "evidence_bundle_set" not in baseline
    assert [row["passage_id"] for row in expanded["seed_hits"]] == ["cap59:p5"]
    assert [row["passage_id"] for row in expanded["hits"]] == [
        "cap59:p5",
        "cap60:p0",
    ]
    assert expanded["query_fingerprint"] != baseline["query_fingerprint"]
    bundle = expanded["evidence_bundle_set"]["bundles"][0]
    assert bundle["seed_passage_ids"] == ["cap59:p5"]
    assert bundle["context_passage_ids"] == ["cap60:p0"]
