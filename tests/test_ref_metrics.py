from __future__ import annotations

import pytest

from vcah.caption_schema import CaptionHitV1
from vcah.mmlifelong_metrics import (
    answer_judge_prompt,
    agent_run_metrics,
    bins_for_interval,
    caption_hits_from_observation_rows,
    export_supporting_intervals,
    interval_iou,
    judge_free_form_answer,
    parse_answer_judge_response,
    ref_score,
    ref_scores,
    retrieval_metrics,
    smooth_score,
)
from vcah.workspace import Claim, WorkingDocument


def _hit(passage_id: str, rank: int, start: float, end: float) -> CaptionHitV1:
    return CaptionHitV1(
        passage_id=passage_id,
        caption_id="caption-1",
        rank=rank,
        lexical_score=1.0,
        dense_score=None,
        fused_score=1.0,
        virtual_start_sec=start,
        virtual_end_sec=end,
        wall_clock_begin=None,
        wall_clock_end=None,
        text="caption passage",
        interval_precision="anchor",
        source_pointer=f"caption://test/{passage_id}",
    )


def test_ref_bins_use_half_open_end_boundary() -> None:
    assert bins_for_interval(0.0, 60.0, 60) == {0}
    assert bins_for_interval(60.0, 120.0, 60) == {1}
    assert bins_for_interval(59.0, 61.0, 60) == {0, 1}
    assert bins_for_interval(10.0, 10.0, 60) == set()
    with pytest.raises(ValueError, match="positive"):
        bins_for_interval(0.0, 1.0, 0)


def test_ref_score_reports_quantized_iou() -> None:
    assert ref_score(((0.0, 60.0),), ((0.0, 60.0),), bucket_size=60) == 100.0
    assert ref_score(((0.0, 60.0),), ((60.0, 120.0),), bucket_size=60) == 0.0
    assert ref_score(((0.0, 120.0),), ((60.0, 120.0),), bucket_size=60) == 50.0
    assert set(ref_scores(((0.0, 1.0),), ((0.0, 1.0),))) == {
        "Ref@60",
        "Ref@300",
        "Ref@600",
    }


def test_supporting_interval_export_prefers_anchor_and_filters_candidates() -> None:
    document = WorkingDocument(
        claims={
            "anchored": Claim(
                claim_id="anchored",
                text="Directly anchored fact",
                source="observation",
                cites=("attempt_anchor",),
                time_anchor=(10.0, 20.0),
            ),
            "fallback": Claim(
                claim_id="fallback",
                text="Fact without an explicit anchor",
                source="observation",
                cites=("attempt_visual", "attempt_caption", "attempt_negative"),
            ),
            "inactive": Claim(
                claim_id="inactive",
                text="Retracted fact",
                source="observation",
                cites=("attempt_inactive",),
                status="retracted",
            ),
        }
    )
    rows = (
        {"attempt_id": "attempt_anchor", "inspected_ranges": [[100.0, 200.0]], "modality": "visual"},
        {"attempt_id": "attempt_visual", "inspected_ranges": [[20.0, 25.0], [24.0, 30.0]], "modality": "visual"},
        {"attempt_id": "attempt_caption", "inspected_ranges": [[40.0, 50.0]], "modality": "caption_search"},
        {"attempt_id": "attempt_negative", "inspected_ranges": [[60.0, 70.0]], "modality": "visual", "evidence_role": "negative"},
        {"attempt_id": "attempt_inactive", "inspected_ranges": [[80.0, 90.0]], "modality": "visual"},
    )

    assert export_supporting_intervals(
        document,
        ("anchored", "fallback", "inactive"),
        rows,
    ) == ((10.0, 30.0),)


def test_supporting_interval_export_inherits_derived_claim_anchors() -> None:
    document = WorkingDocument(
        claims={
            "event": Claim(
                claim_id="event",
                text="The event occurred.",
                source="observation",
                cites=("attempt_visual",),
                time_anchor=(19950.0, 19952.027),
            ),
            "item": Claim(
                claim_id="item",
                text="The item appeared immediately afterward.",
                source="observation",
                cites=("attempt_visual",),
                time_anchor=(19950.0, 19954.053),
            ),
            "answer": Claim(
                claim_id="answer",
                text="The event yielded the item.",
                source="derived",
                derived_from=("event", "item"),
            ),
        }
    )
    rows = (
        {
            "attempt_id": "attempt_visual",
            "inspected_ranges": [[19950.0, 19988.0]],
            "modality": "visual",
        },
    )

    assert export_supporting_intervals(document, ("answer",), rows) == (
        (19950.0, 19954.053),
    )


def test_retrieval_metrics_cover_single_and_multi_clue_cases() -> None:
    hits = (
        _hit("p0", 1, 0.0, 10.0),
        _hit("p1", 2, 100.0, 110.0),
        _hit("p2", 3, 200.0, 210.0),
    )
    metrics = retrieval_metrics(hits, ((102.0, 108.0), (202.0, 208.0)), ks=(1, 2, 3))

    assert metrics["FirstClueRank"] == 2
    assert metrics["ClueRecall@1"] == 0.0
    assert metrics["ClueRecall@2"] == 1.0
    assert metrics["AllCluesRecall@2"] == 0.0
    assert metrics["AllCluesRecall@3"] == 1.0
    assert metrics["ClueIoU@3"] == pytest.approx(12.0 / 30.0)
    assert interval_iou(((0.0, 10.0),), ((5.0, 15.0),)) == pytest.approx(1.0 / 3.0)


def test_answer_judge_result_preserves_raw_and_smoothed_scores() -> None:
    prompt = answer_judge_prompt(
        question="What was raised?",
        reference_answer="A cup",
        predicted_answer="The person raised a cup.",
    )
    result = parse_answer_judge_response(
        '```json\n{"score": 4, "rationale": "Semantically correct."}\n```',
        judge_model="judge-v1",
        prompt=prompt,
        retry_count=1,
    )

    assert result.raw_score == 4
    assert result.smoothed_score == 1.0
    assert result.parse_status == "parsed"
    assert result.retry_count == 1
    assert smooth_score(3) == 0.5
    assert smooth_score(2) == 0.0
    with pytest.raises(ValueError, match="between 0 and 5"):
        smooth_score(6)


def test_answer_judge_retries_parse_failure_and_preserves_metadata() -> None:
    responses = iter(("not-json", '{"score": 3, "rationale": "Partly correct."}'))
    metadata = {"finish_reason": "stop", "completion_tokens": 20}

    result = judge_free_form_answer(
        lambda prompt: next(responses),
        question="What happened?",
        reference_answer="The player opened a chest.",
        predicted_answer="A chest was opened.",
        judge_model="judge-v1",
        max_retries=2,
        response_metadata=lambda: metadata,
    )

    assert result.raw_score == 3
    assert result.smoothed_score == 0.5
    assert result.retry_count == 1
    assert result.response_metadata["completion_tokens"] == 20


def test_caption_and_agent_metrics_are_derived_from_logged_candidates() -> None:
    rows = (
        {
            "attempt_id": "caption-attempt",
            "modality": "caption_search",
            "sampling_config": {
                "mode": "search_caption",
                "hits": [
                    {"passage_id": "p1", "range": [100.0, 110.0], "score": 0.9},
                    {"passage_id": "p2", "range": [200.0, 210.0], "score": 0.8},
                ],
            },
            "frame_times": [],
        },
        {
            "attempt_id": "visual-attempt",
            "modality": "visual",
            "sampling_config": {"mode": "window"},
            "frame_times": [101.0, 102.0],
        },
    )
    trace = (
        {"type": "reasoner_decision", "action": "investigate"},
        {"type": "reasoner_decision", "action": "read_observations"},
        {
            "type": "investigator_batch",
            "outcomes": [{"query_id": "q1", "reused": True}],
        },
    )

    hits = caption_hits_from_observation_rows(rows)
    metrics = agent_run_metrics(
        trace,
        rows,
        answer_present=True,
        reference_valid=True,
        supporting_intervals=((102.0, 108.0),),
    )

    assert [hit.passage_id for hit in hits] == ["p1", "p2"]
    assert metrics["caption_searches"] == 1
    assert metrics["visual_confirmations"] == 1
    assert metrics["visual_frames_inspected"] == 2
    assert metrics["dedicated_read_rounds"] == 1
    assert metrics["duplicate_search_count"] == 1
    assert metrics["candidate_to_support_conversion"] == 0.5
