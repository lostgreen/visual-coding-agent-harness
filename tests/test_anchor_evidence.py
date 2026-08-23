from __future__ import annotations

from vcah.anchor_evidence import (
    ANCHOR_EVIDENCE_CONTRACT,
    expand_anchor_conditioned_evidence,
    infer_anchor_evidence_request,
)
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1


def _passage(
    passage_id: str, start: float, end: float, segment_id: str
) -> CaptionPassageV1:
    return CaptionPassageV1(
        passage_id=passage_id,
        caption_id="caption",
        text=f"event {passage_id}",
        virtual_start_sec=start,
        virtual_end_sec=end,
        anchor_virtual_sec=start,
        ordinal=int(start),
        metadata={"source_segments": [segment_id]},
    )


def _hit(passage: CaptionPassageV1) -> CaptionHitV1:
    return CaptionHitV1(
        passage_id=passage.passage_id,
        caption_id=passage.caption_id,
        rank=1,
        lexical_score=1.0,
        dense_score=1.0,
        fused_score=1.0,
        virtual_start_sec=passage.virtual_start_sec,
        virtual_end_sec=passage.virtual_end_sec,
        wall_clock_begin=None,
        wall_clock_end=None,
        text=passage.text,
        interval_precision="anchor",
        source_pointer=f"caption://cfg/{passage.passage_id}",
        metadata=dict(passage.metadata),
    )


def test_request_recognizes_direction_and_channel_without_entities() -> None:
    request = infer_anchor_evidence_request(
        "After defeating the boss, what line does the singer sing?"
    )
    assert request.eligible is True
    assert request.direction == "after"
    assert request.evidence_channels == ("subtitle",)

    aggregate = infer_anchor_evidence_request(
        "How many enemies were defeated before entering the temple?"
    )
    assert aggregate.eligible is False
    assert aggregate.reason == "aggregate_question_requires_stateful_evidence"

    numeric_ocr = infer_anchor_evidence_request(
        "How many units of Spirit Essence were obtained before the boss fight?"
    )
    assert numeric_ocr.eligible is True
    assert numeric_ocr.direction == "before"
    assert numeric_ocr.evidence_channels == ("numeric_ocr",)

    following = infer_anchor_evidence_request(
        "Which of the following statements is correct?"
    )
    assert following.eligible is False
    assert following.direction is None

    multi_anchor = infer_anchor_evidence_request(
        "Before fighting which of the following characters is mana not full?"
    )
    assert multi_anchor.eligible is False
    assert multi_anchor.direction == "before"
    assert multi_anchor.reason == "multi_candidate_relation_requires_separate_anchors"

    unknown_anchor = infer_anchor_evidence_request(
        "After defeating which boss was the armor unlocked?"
    )
    assert unknown_anchor.eligible is False
    assert unknown_anchor.reason == "anchor_is_unknown_answer_target"


def test_expansion_bypasses_packet_scope_only_after_frozen_anchor() -> None:
    anchor = _passage("anchor", 10.0, 20.0, "seg-a")
    target = _passage("target", 21.0, 30.0, "seg-b")
    result = expand_anchor_conditioned_evidence(
        (anchor, target),
        (_hit(anchor),),
        question="After the fight, what item was obtained?",
        ocr_rows=(
            {
                "text": "上清宝葫芦",
                "regions": ["item_popup"],
                "max_confidence": "high",
                "virtual_times_sec": [25.0],
                "segment_ids": ["seg-b"],
            },
        ),
        distance=1,
        index_digest="index",
        config_digest="cfg",
        source_video_id_by_segment={"seg-a": "video-a", "seg-b": "video-a"},
    )

    assert result["contract"] == ANCHOR_EVIDENCE_CONTRACT
    assert [hit.passage_id for hit in result["seed_hits"]] == ["anchor"]
    assert [hit.passage_id for hit in result["hits"]] == ["anchor", "target"]
    metadata = result["hits"][1].metadata
    assert metadata["packet_scope_bypassed_after_anchor"] is True
    assert metadata["evidence_channels_observed"] == ["caption", "visible_ocr"]
    bundle = result["evidence_bundle_set"]["bundles"][0]
    assert bundle["seed_passage_ids"] == ["anchor"]
    assert bundle["context_passage_ids"] == ["target"]


def test_ineligible_question_preserves_seed_identity() -> None:
    anchor = _passage("anchor", 10.0, 20.0, "seg-a")
    result = expand_anchor_conditioned_evidence(
        (anchor,),
        (_hit(anchor),),
        question="What color was the boss?",
        distance=3,
        index_digest="index",
        config_digest="cfg",
    )
    assert result["request"]["eligible"] is False
    assert result["hits"] == result["seed_hits"]
