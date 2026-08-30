from __future__ import annotations

import pytest

from vcah.wp17_week_split import build_week_query_manifests


_TYPE_COUNTS = {
    "Attribute Recognition": 8,
    "Causal Reasoning": 17,
    "Counting": 59,
    "Entity Recognition": 23,
    "Event Recognition": 17,
    "Event Tracking": 6,
    "Hallucination Detection": 10,
    "Language Content Recall": 15,
    "Social Interaction": 9,
    "Temporal Reasoning": 36,
}


def _cases() -> list[dict[str, str]]:
    rows = []
    index = 0
    for question_type, count in _TYPE_COUNTS.items():
        for _ in range(count):
            rows.append(
                {
                    "case_id": f"mmlifelong-week-test-{index:04d}",
                    "question_type": question_type,
                    "case_sha256": f"sha-{index}",
                    "question": "must not be persisted",
                    "answer": "must not be persisted",
                    "clue_intervals": [[1, 2]],
                }
            )
            index += 1
    return rows


def test_week_split_is_deterministic_disjoint_and_complete() -> None:
    first = build_week_query_manifests(_cases())
    second = build_week_query_manifests(list(reversed(_cases())))

    assert first == second
    assert first["protocol"]["decision"] == "WP17_WEEK_QUERY_SPLIT_FROZEN"
    assert first["protocol"]["structural_gate_passed"] is True
    dev_ids = {row["case_id"] for row in first["week_dev"]["cases"]}
    holdout_ids = {row["case_id"] for row in first["week_holdout"]["cases"]}
    assert len(dev_ids) == 60
    assert len(holdout_ids) == 140
    assert not dev_ids & holdout_ids
    assert len(dev_ids | holdout_ids) == 200


def test_week_split_uses_largest_remainder_question_type_quotas() -> None:
    result = build_week_query_manifests(_cases())
    quotas = result["week_dev"]["question_type_quota"]

    assert {key: row["development"] for key, row in quotas.items()} == {
        "Attribute Recognition": 2,
        "Causal Reasoning": 5,
        "Counting": 18,
        "Entity Recognition": 7,
        "Event Recognition": 5,
        "Event Tracking": 2,
        "Hallucination Detection": 3,
        "Language Content Recall": 4,
        "Social Interaction": 3,
        "Temporal Reasoning": 11,
    }


def test_week_split_does_not_persist_question_answer_or_clues() -> None:
    result = build_week_query_manifests(_cases())

    assert all(
        set(row) == {"case_id", "question_type", "case_sha256"}
        for name in ("week_dev", "week_holdout")
        for row in result[name]["cases"]
    )
    assert result["week_dev"]["method_selection_allowed"] is True
    assert result["week_holdout"]["method_selection_allowed"] is False
    assert result["week_holdout"]["shared_video_corpus_with_development"] is True
    assert result["week_holdout"]["eligible_for_unseen_video_claim"] is False


def test_week_split_rejects_bad_universe_and_duplicates() -> None:
    with pytest.raises(ValueError, match="expected 200"):
        build_week_query_manifests(_cases()[:-1])

    duplicate = _cases()
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="duplicate"):
        build_week_query_manifests(duplicate)
