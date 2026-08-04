from __future__ import annotations

from runs.mger_locator_ablation import evaluate_hits, select_promotion


def test_locator_metrics_measure_hit_and_occurrence_recall() -> None:
    hits = (
        {
            "passage_id": "p1",
            "caption_id": "c1",
            "rank": 1,
            "range": [10.0, 20.0],
            "score": 1.0,
            "metadata": {"source_video_ids": ["v1"]},
        },
        {
            "passage_id": "p2",
            "caption_id": "c1",
            "rank": 2,
            "range": [30.0, 40.0],
            "score": 0.5,
            "metadata": {"source_video_ids": ["v1"]},
        },
    )

    metrics = evaluate_hits(hits, ((15.0, 16.0), (100.0, 101.0)))

    assert metrics["candidate_clue_recall"] == 0.5
    assert metrics["occurrence_candidate_recall"] == 0.5
    assert metrics["candidate_count"] == 2
    assert metrics["recall_per_candidate"] == 0.25


def test_locator_promotion_requires_recall_improvement_and_prefers_lower_cost() -> None:
    variants = (
        {
            "name": "current_hybrid",
            "status": "ok",
            "candidate_clue_recall": 0.0,
            "occurrence_candidate_recall": 0.0,
            "candidate_count": 12,
            "recall_per_candidate": 0.0,
        },
        {
            "name": "wide",
            "status": "ok",
            "candidate_clue_recall": 1.0,
            "occurrence_candidate_recall": 1.0,
            "candidate_count": 40,
            "recall_per_candidate": 0.025,
        },
        {
            "name": "focused",
            "status": "ok",
            "candidate_clue_recall": 1.0,
            "occurrence_candidate_recall": 1.0,
            "candidate_count": 16,
            "recall_per_candidate": 0.0625,
        },
    )

    result = select_promotion(variants)

    assert result["promotion_eligible"] is True
    assert result["selected_variant"] == "focused"


def test_locator_promotion_rejects_candidate_growth_without_recall_gain() -> None:
    variants = (
        {
            "name": "current_hybrid",
            "status": "ok",
            "candidate_clue_recall": 0.0,
            "occurrence_candidate_recall": 0.0,
            "candidate_count": 12,
        },
        {
            "name": "larger_k",
            "status": "ok",
            "candidate_clue_recall": 0.0,
            "occurrence_candidate_recall": 0.0,
            "candidate_count": 32,
        },
    )

    result = select_promotion(variants)

    assert result["promotion_eligible"] is False
    assert result["reason"] == "no_offline_candidate_recall_improvement"
