from __future__ import annotations

from vcah.caption_schema import CaptionPassageV1
from vcah.occurrence_ocr import (
    deduplicate_ocr_rows,
    enrich_caption_passages_with_ocr,
    fuse_caption_hit_ranks,
    gemini_ocr_prompt,
    ocr_sidecar_passages,
    ocr_query_overlap,
    ocr_text_has_query_evidence,
    parse_gemini_ocr_response,
    parse_gemini_ocr_response_diagnostic,
)


def test_ui_aware_prompt_is_blind_and_pixel_grounded() -> None:
    prompt = gemini_ocr_prompt(("frame_01", "frame_02"), variant="ui_aware_v1")
    assert "boss HP-bar" in prompt
    assert "Copy only text that is visibly supported by pixels" in prompt
    assert "question" not in prompt.casefold()
    assert "answer" not in prompt.casefold()


def test_parser_requires_every_bound_frame_and_valid_enums() -> None:
    raw = (
        '{"frames":['
        '{"frame_label":"frame_01","visible_text":['
        '{"text":"虎先锋","region":"boss_name","confidence":"high"}]},'
        '{"frame_label":"frame_02","visible_text":[]}'
        "]}"
    )
    rows = parse_gemini_ocr_response(
        raw, allowed_frame_labels=("frame_01", "frame_02")
    )
    assert rows is not None
    assert rows[0]["text"] == "虎先锋"
    missing = parse_gemini_ocr_response_diagnostic(
        '{"frames":[{"frame_label":"frame_01","visible_text":[]}]}',
        allowed_frame_labels=("frame_01", "frame_02"),
    )
    assert missing["status"] == "success"
    assert missing["normalization_counts"]["implicit_empty_frame"] == 1


def test_parser_normalizes_noncritical_schema_variation() -> None:
    raw = (
        '{"frames":[{"frame_label":"frame_01 (first)","visible_text":['
        '{"text":"缩地青符","region":"item_name","confidence":0.91}]}]}'
    )
    diagnostic = parse_gemini_ocr_response_diagnostic(
        raw, allowed_frame_labels=("frame_01", "frame_02")
    )
    assert diagnostic["status"] == "success"
    assert diagnostic["rows"][0]["region"] == "other"
    assert diagnostic["rows"][0]["confidence"] == "high"
    assert diagnostic["normalization_counts"] == {
        "confidence_normalized": 1,
        "frame_label_alias": 1,
        "implicit_empty_frame": 1,
        "unknown_region_to_other": 1,
    }


def test_parser_accepts_frames_as_root_array_with_audit_count() -> None:
    diagnostic = parse_gemini_ocr_response_diagnostic(
        '[{"frame_label":"frame_01","visible_text":[]}]',
        allowed_frame_labels=("frame_01",),
    )
    assert diagnostic["status"] == "success"
    assert diagnostic["normalization_counts"] == {"root_array_to_frames": 1}


def test_temporal_dedup_keeps_lineage_and_highest_confidence() -> None:
    rows = deduplicate_ocr_rows(
        (
            {
                "frame_label": "frame_01",
                "text": "得胜",
                "region": "victory_defeat",
                "confidence": "medium",
            },
            {
                "frame_label": "frame_02",
                "text": "得胜",
                "region": "victory_defeat",
                "confidence": "high",
            },
        ),
        frame_metadata={
            "frame_01": {"virtual_time_sec": 10.0, "segment_id": "seg-a"},
            "frame_02": {"virtual_time_sec": 11.0, "segment_id": "seg-a"},
        },
    )
    assert len(rows) == 1
    assert rows[0]["frame_labels"] == ["frame_01", "frame_02"]
    assert rows[0]["max_confidence"] == "high"


def test_ocr_query_overlap_uses_posthoc_tokens() -> None:
    result = ocr_query_overlap(({"text": "虎先锋"},), ("击败虎先锋之后",))
    assert "虎先" in result["matched_tokens"]
    assert result["matched_token_count"] > 0
    assert ocr_text_has_query_evidence("虎先锋", ("击败虎先锋之后",)) is True
    assert ocr_text_has_query_evidence("虎", ("击败虎先锋之后",)) is False


def test_enrichment_binds_nearest_passage_without_changing_ids() -> None:
    passages = (
        CaptionPassageV1(
            passage_id="p1",
            caption_id="c1",
            text="A boss fight continues.",
            virtual_start_sec=10.0,
            virtual_end_sec=20.0,
            anchor_virtual_sec=10.0,
            ordinal=0,
            metadata={"source_segments": ["seg-a"]},
        ),
        CaptionPassageV1(
            passage_id="p2",
            caption_id="c1",
            text="The player walks away.",
            virtual_start_sec=30.0,
            virtual_end_sec=40.0,
            anchor_virtual_sec=30.0,
            ordinal=1,
            metadata={"source_segments": ["seg-a"]},
        ),
    )
    enriched = enrich_caption_passages_with_ocr(
        passages,
        (
            {
                "text": "虎先锋",
                "virtual_times_sec": [15.0],
                "segment_ids": ["seg-a"],
            },
        ),
    )
    assert [row.passage_id for row in enriched] == ["p1", "p2"]
    assert "虎先锋" in enriched[0].text
    assert enriched[1].text == passages[1].text
    sidecar = ocr_sidecar_passages(enriched)
    assert sidecar[0].text == "虎先锋"
    assert sidecar[1].text == ""


def test_rank_fusion_can_promote_ocr_only_passage() -> None:
    baseline = (
        {"passage_id": "p1", "rank": 1, "metadata": {}},
        {"passage_id": "p2", "rank": 2, "metadata": {}},
    )
    ocr = (
        {"passage_id": "p3", "rank": 1, "metadata": {}},
        {"passage_id": "p2", "rank": 2, "metadata": {}},
    )
    fused = fuse_caption_hit_ranks(baseline, ocr, top_k=3)
    assert fused[0]["passage_id"] == "p2"
    assert {row["passage_id"] for row in fused} == {"p1", "p2", "p3"}
