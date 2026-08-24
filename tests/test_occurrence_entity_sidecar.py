from __future__ import annotations

from vcah.caption_schema import CaptionPassageV1
from vcah.occurrence_entity_sidecar import (
    admit_global_entity_rows,
    build_entity_sidecar_passages,
    fixed3_passage_targets,
    global_entity_duplicate_stats,
    global_entity_ocr_prompt,
    parse_global_entity_ocr_response_diagnostic,
    select_hashed_passages,
)


def _passage(passage_id: str, start: float = 10.0, end: float = 20.0) -> CaptionPassageV1:
    return CaptionPassageV1(
        passage_id=passage_id,
        caption_id="caption-a",
        text="A generic boss falls.",
        virtual_start_sec=start,
        virtual_end_sec=end,
        anchor_virtual_sec=start,
        ordinal=0,
        metadata={"source_segments": ["seg-a"]},
    )


def test_entity_prompt_is_blind_and_forbids_inference() -> None:
    prompt = global_entity_ocr_prompt(("frame_01", "frame_02"))
    assert "entity-oriented OCR" in prompt
    assert "guess an alias" in prompt
    assert "question" in prompt.casefold() and "unavailable" in prompt.casefold()
    assert "official intervals are unavailable" in prompt


def test_parser_and_admission_keep_consensus_with_lineage() -> None:
    raw = (
        '{"frames":['
        '{"frame_label":"frame_01","entities":['
        '{"text":"赤髯龙","entity_type":"boss_name",'
        '"ui_region":"boss_name_bar","confidence":"high"}]},'
        '{"frame_label":"frame_02","entities":['
        '{"text":"赤髯龙","entity_type":"boss_name",'
        '"ui_region":"boss_name_bar","confidence":"medium"}]}'
        "]}"
    )
    parsed = parse_global_entity_ocr_response_diagnostic(
        raw, allowed_frame_labels=("frame_01", "frame_02")
    )
    assert parsed["status"] == "success"
    admission = admit_global_entity_rows(
        parsed["rows"],
        passage_id="p1",
        frame_metadata={
            "frame_01": {
                "frame_id": "f1",
                "virtual_time_sec": 11.0,
                "segment_id": "seg-a",
                "source_video_id": "video-a",
            },
            "frame_02": {
                "frame_id": "f2",
                "virtual_time_sec": 12.0,
                "segment_id": "seg-a",
                "source_video_id": "video-a",
            },
        },
    )
    row = admission["admitted_rows"][0]
    assert row["text"] == "赤髯龙"
    assert row["support_count"] == 2
    assert row["admission_reason"] == "multi_frame_consensus"
    assert row["frame_ids"] == ["f1", "f2"]


def test_single_high_value_region_admitted_but_noise_rejected() -> None:
    admission = admit_global_entity_rows(
        (
            {
                "frame_label": "f1",
                "text": "火焰山",
                "entity_type": "location",
                "ui_region": "location_title",
                "confidence": "high",
            },
            {
                "frame_label": "f1",
                "text": "999",
                "entity_type": "other_named_entity",
                "ui_region": "other",
                "confidence": "high",
            },
            {
                "frame_label": "f1",
                "text": "确认",
                "entity_type": "other_named_entity",
                "ui_region": "menu_title",
                "confidence": "high",
            },
            {
                "frame_label": "f1",
                "text": "Tiger",
                "entity_type": "boss_name",
                "ui_region": "boss_name_bar",
                "confidence": "high",
            },
        ),
        passage_id="p1",
        frame_metadata={"f1": {"virtual_time_sec": 11.0}},
    )
    assert [row["text"] for row in admission["admitted_rows"]] == ["火焰山"]
    assert admission["rejection_counts"] == {
        "blocked_text": 1,
        "insufficient_english_tokens": 1,
        "numeric_only": 1,
    }


def test_fixed3_targets_and_hash_selection_are_deterministic() -> None:
    passage = _passage("p1", 10.0, 20.0)
    targets = fixed3_passage_targets(passage)
    assert [row["virtual_time_sec"] for row in targets] == [10.0, 15.0, 19.999]
    short = fixed3_passage_targets(_passage("short", 10.0, 10.001))
    assert len(short) == 1
    assert short[0]["sample_positions"] == (
        "start",
        "midpoint",
        "end_minus_1ms",
    )
    passages = tuple(_passage(f"p{index}") for index in range(10))
    first = select_hashed_passages(passages, seed="seed", count=2)
    second = select_hashed_passages(tuple(reversed(passages)), seed="seed", count=2)
    assert [row.passage_id for row in first] == [row.passage_id for row in second]


def test_sidecar_passages_and_duplicate_stats_preserve_passage_ids() -> None:
    passages = (_passage("p1"), _passage("p2"))
    rows = (
        {"passage_id": "p1", "text": "赤髯龙", "normalized": "赤髯龙"},
        {"passage_id": "p2", "text": "赤髯龙", "normalized": "赤髯龙"},
        {"passage_id": "p2", "text": "火焰山", "normalized": "火焰山"},
    )
    sidecar = build_entity_sidecar_passages(passages, rows)
    assert [row.passage_id for row in sidecar] == ["p1", "p2"]
    assert sidecar[0].text == "赤髯龙"
    stats = global_entity_duplicate_stats(rows)
    assert stats["unique_entity_text_count"] == 2
    assert stats["duplicate_entity_text_count"] == 1
