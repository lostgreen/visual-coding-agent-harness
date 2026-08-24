from __future__ import annotations

from tools.diagnose_mmlifelong_occurrence_field_ablation import (
    VARIANTS,
    _passage_ids_overlap_rank,
    render_markdown,
)
from vcah.caption_schema import CaptionPassageV1
from vcah.occurrence_field_ablation import build_field_ablation_report


def test_renderer_marks_oracle_boundary() -> None:
    ranks = {variant: 1 for variant in VARIANTS}
    report = build_field_ablation_report(
        (
            {
                "case_id": "mmlifelong-game-test-0001",
                "anchor_description": "anchor",
                "oracle_passage_id": "p1",
                "field_names": ["entity", "event", "state"],
                "fields_frozen_before_ablation_outcomes": True,
                "oracle_gold_occurrence_only": True,
                "target_evidence_and_answer_excluded": True,
                "ranks": ranks,
            },
        ),
        expected_cases=1,
        variant_order=VARIANTS,
    )
    markdown = render_markdown(report)
    assert "Oracle Field Ablation" in markdown
    assert "perfect extraction" in markdown
    assert "Non-gold field false positives" in markdown


def test_rank_accepts_any_passage_overlapping_anchor_interval() -> None:
    passages = (
        CaptionPassageV1(
            passage_id="same-occurrence-neighbor",
            caption_id="caption-a",
            text="neighbor",
            virtual_start_sec=9.0,
            virtual_end_sec=12.0,
            anchor_virtual_sec=9.0,
            ordinal=0,
        ),
        CaptionPassageV1(
            passage_id="selected-oracle",
            caption_id="caption-a",
            text="selected",
            virtual_start_sec=12.0,
            virtual_end_sec=20.0,
            anchor_virtual_sec=12.0,
            ordinal=1,
        ),
    )
    assert (
        _passage_ids_overlap_rank(
            ("same-occurrence-neighbor", "selected-oracle"),
            passages=passages,
            anchor_intervals=((10.0, 18.0),),
        )
        == 1
    )
