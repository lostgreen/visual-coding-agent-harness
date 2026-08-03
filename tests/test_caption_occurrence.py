from __future__ import annotations

from dataclasses import asdict

from vcah.caption_occurrence import build_caption_occurrence_set
from vcah.caption_schema import CaptionHitV1


def _hit(
    passage_id: str,
    start_sec: float,
    end_sec: float,
    *,
    rank: int,
    source_video_id: str = "video-a",
    query: str = "target event",
) -> CaptionHitV1:
    return CaptionHitV1(
        passage_id=passage_id,
        caption_id="caption",
        rank=rank,
        lexical_score=1.0,
        dense_score=None,
        fused_score=1.0 / rank,
        virtual_start_sec=start_sec,
        virtual_end_sec=end_sec,
        wall_clock_begin=None,
        wall_clock_end=None,
        text="A repeated event is shown.",
        interval_precision="anchor",
        source_pointer=f"caption://fixture/{passage_id}",
        metadata={
            "source_segments": [f"segment-{source_video_id}"],
            "source_video_ids": [source_video_id],
            "query_matches": [{"query": query, "rank": rank}],
        },
    )


def test_distant_repeated_events_remain_competing_occurrences() -> None:
    hits = (
        _hit("early-a", 17_809.0, 17_812.0, rank=1),
        _hit("early-b", 17_820.0, 17_824.0, rank=3),
        _hit("late-a", 19_950.0, 19_952.0, rank=2),
    )

    occurrence_set = build_caption_occurrence_set(hits, gap_sec=120.0)

    assert occurrence_set["status"] == "competing_candidates"
    assert occurrence_set["occurrence_ambiguous"] is True
    assert occurrence_set["candidate_count"] == 2
    assert occurrence_set["candidates"][0]["passage_ids"] == ["early-a", "early-b"]
    assert occurrence_set["candidates"][1]["passage_ids"] == ["late-a"]
    assert all(candidate["evidence_role"] == "candidate" for candidate in occurrence_set["candidates"])
    assert occurrence_set["selected_occurrence_id"] is None


def test_occurrence_identity_ignores_query_interpretation_and_hit_order() -> None:
    original = (
        _hit("p1", 100.0, 105.0, rank=1, query="first wording"),
        _hit("p2", 110.0, 115.0, rank=2, query="first wording"),
    )
    reinterpreted = tuple(
        CaptionHitV1(
            **{
                **asdict(hit),
                "metadata": {
                    **dict(hit.metadata),
                    "query_matches": [{"query": "different wording", "rank": hit.rank}],
                },
            }
        )
        for hit in reversed(original)
    )

    first = build_caption_occurrence_set(original)
    second = build_caption_occurrence_set(reinterpreted)

    assert first["candidates"][0]["occurrence_id"] == second["candidates"][0]["occurrence_id"]
    assert first["candidates"][0]["query_matches"] != second["candidates"][0]["query_matches"]


def test_overlapping_times_from_different_sources_do_not_merge() -> None:
    occurrence_set = build_caption_occurrence_set(
        (
            _hit("source-a", 100.0, 110.0, rank=1, source_video_id="video-a"),
            _hit("source-b", 102.0, 112.0, rank=2, source_video_id="video-b"),
        )
    )

    assert occurrence_set["candidate_count"] == 2
    assert [candidate["source_video_ids"] for candidate in occurrence_set["candidates"]] == [
        ["video-a"],
        ["video-b"],
    ]
