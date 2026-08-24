from __future__ import annotations

import pytest

from vcah.caption_schema import CaptionPassageV1
from vcah.relation_evidence_search import (
    RELATION_EVIDENCE_SEARCH_CONTRACT,
    build_oracle_fixed_window,
    relation_guided_evidence_search,
    select_oracle_anchor_passage,
)


def _passage(
    passage_id: str,
    start: float,
    end: float,
    text: str,
) -> CaptionPassageV1:
    return CaptionPassageV1(
        passage_id=passage_id,
        caption_id="caption",
        text=text,
        virtual_start_sec=start,
        virtual_end_sec=end,
        anchor_virtual_sec=start,
        ordinal=int(start),
        metadata={"source_segments": ["segment"]},
    )


def test_bounded_search_stops_at_target_then_reads_numeric_ocr() -> None:
    anchor = _passage("anchor", 0.0, 10.0, "boss challenge begins")
    noise = _passage("noise", 10.0, 20.0, "the player walks")
    target = _passage("target", 20.0, 30.0, "opens a treasure chest")
    result = relation_guided_evidence_search(
        (anchor, noise, target),
        anchor_intervals=((1.0, 9.0),),
        direction="after",
        target_event_term_groups=(("chest",),),
        evidence_channels=("numeric_ocr",),
        ocr_rows=(
            {
                "text": "x2762",
                "regions": ["hud_number"],
                "virtual_times_sec": [25.0],
                "segment_ids": ["segment"],
            },
        ),
        index_digest="index",
        config_digest="config",
    )

    assert result["contract"] == RELATION_EVIDENCE_SEARCH_CONTRACT
    assert result["stop_success"] is True
    assert result["visited_passage_count"] == 2
    assert result["stop_hit"].passage_id == "target"
    assert result["stop_hit"].metadata["evidence_channels_observed"] == [
        "caption",
        "visible_ocr",
        "numeric_ocr",
    ]
    bundle = result["evidence_bundle_set"]["bundles"][0]
    assert bundle["seed_passage_ids"] == ["anchor"]
    assert bundle["context_passage_ids"] == ["target"]


def test_ocr_does_not_trigger_stop_when_target_source_is_caption_only() -> None:
    anchor = _passage("anchor", 0.0, 10.0, "boss challenge begins")
    noise = _passage("noise", 10.0, 20.0, "the player walks")
    target = _passage("target", 20.0, 30.0, "opens the chest")
    result = relation_guided_evidence_search(
        (anchor, noise, target),
        anchor_intervals=((1.0, 9.0),),
        direction="after",
        target_event_term_groups=(("chest",),),
        evidence_channels=("visible_ocr",),
        target_text_sources=("caption",),
        ocr_rows=(
            {
                "text": "treasure chest",
                "regions": ["item_popup"],
                "virtual_times_sec": [15.0],
                "segment_ids": ["segment"],
            },
        ),
        index_digest="index",
        config_digest="config",
    )

    assert result["stop_hit"].passage_id == "target"
    assert result["visited_passage_count"] == 2


def test_before_search_stops_at_nearest_matching_event() -> None:
    older = _passage("older", 0.0, 10.0, "opens a treasure chest")
    nearest = _passage("nearest", 10.0, 20.0, "opens another chest")
    anchor = _passage("anchor", 20.0, 30.0, "boss fight begins")
    result = relation_guided_evidence_search(
        (older, nearest, anchor),
        anchor_intervals=((21.0, 29.0),),
        direction="before",
        target_event_term_groups=(("chest",),),
        evidence_channels=("caption",),
        index_digest="index",
        config_digest="config",
    )

    assert result["stop_hit"].passage_id == "nearest"
    assert result["visited_passage_count"] == 1


def test_fixed_window_uses_oracle_anchor_and_direction() -> None:
    before = _passage("before", 0.0, 10.0, "before")
    anchor = _passage("anchor", 10.0, 20.0, "anchor")
    after_one = _passage("after-one", 20.0, 30.0, "after one")
    after_two = _passage("after-two", 30.0, 40.0, "after two")
    result = build_oracle_fixed_window(
        (before, anchor, after_one, after_two),
        anchor_intervals=((11.0, 19.0),),
        direction="after",
        evidence_channels=("caption",),
        distance=2,
        index_digest="index",
        config_digest="config",
    )

    assert [hit.passage_id for hit in result["hits"]] == [
        "anchor",
        "after-one",
        "after-two",
    ]
    assert result["visited_passage_count"] == 2


def test_anchor_selection_prefers_semantic_interval_overlap() -> None:
    broad = _passage("broad", 0.0, 30.0, "broad")
    exact = _passage("exact", 10.0, 20.0, "exact")
    selected = select_oracle_anchor_passage(
        (broad, exact),
        ((11.0, 19.0),),
    )
    assert selected is not None
    assert selected.passage_id == "exact"


def test_empty_target_term_group_is_rejected() -> None:
    anchor = _passage("anchor", 0.0, 10.0, "anchor")
    with pytest.raises(ValueError, match="term groups"):
        relation_guided_evidence_search(
            (anchor,),
            anchor_intervals=((1.0, 9.0),),
            direction="after",
            target_event_term_groups=((),),
            evidence_channels=("caption",),
            index_digest="index",
            config_digest="config",
        )
