from __future__ import annotations

from vcah.occurrence_anchor_recall import build_anchor_recall_report


VARIANTS = (
    "frozen_seed_best_packet",
    "full_question_hybrid",
    "anchor_only_hybrid",
    "anchor_event_lexical",
    "anchor_event_hybrid",
)


def _case(case_id: str, ranks: dict[str, int | None]) -> dict[str, object]:
    return {
        "case_id": case_id,
        "anchor_description": "reference occurrence",
        "anchor_intervals": [[10.0, 20.0]],
        "queries_frozen_before_anchor_outcomes": True,
        "ranks": ranks,
    }


def test_anchor_event_upper_bound_drives_automation_decision() -> None:
    cases = []
    for index in range(10):
        cases.append(
            _case(
                f"mmlifelong-game-test-{index:04d}",
                {
                    "frozen_seed_best_packet": None,
                    "full_question_hybrid": 4 if index < 3 else None,
                    "anchor_only_hybrid": 2 if index < 5 else None,
                    "anchor_event_lexical": 6 if index < 7 else None,
                    "anchor_event_hybrid": 3 if index < 8 else None,
                },
            )
        )

    report = build_anchor_recall_report(
        cases,
        expected_cases=10,
        variant_order=VARIANTS,
    )

    assert report["structural_gate_passed"] is True
    assert report["decision"] == "AUTOMATE_ANCHOR_EVENT_REPRESENTATION"
    assert report["variants"]["anchor_only_hybrid"]["recall"]["at_5"] == {
        "count": 5,
        "case_count": 10,
        "rate": 0.5,
    }
    comparison = report["comparisons"][
        "anchor_event_hybrid_vs_full_question_hybrid_at5"
    ]
    assert len(comparison["recovered_case_ids"]) == 5
    assert comparison["regressed_case_ids"] == []
    assert comparison["net_recovery"] == 5


def test_anchor_only_query_can_meet_target_directly() -> None:
    cases = [
        _case(
            f"mmlifelong-game-test-{index:04d}",
            {
                variant: (1 if variant == "anchor_only_hybrid" else None)
                for variant in VARIANTS
            },
        )
        for index in range(10)
    ]

    report = build_anchor_recall_report(
        cases,
        expected_cases=10,
        variant_order=VARIANTS,
    )

    assert report["decision"] == "ANCHOR_ONLY_QUERY_SUFFICIENT"
    assert report["variants"]["anchor_only_hybrid"]["mean_reciprocal_rank"] == 1.0


def test_structural_gate_rejects_multi_occurrence_case_and_bad_rank() -> None:
    case = _case(
        "mmlifelong-game-test-0115",
        {variant: None for variant in VARIANTS},
    )
    case["ranks"]["anchor_event_hybrid"] = 0

    report = build_anchor_recall_report(
        [case],
        expected_cases=1,
        variant_order=VARIANTS,
    )

    assert report["structural_gate_passed"] is False
    assert report["structural_checks"]["single_anchor_subset_excludes_0115"] is False
    assert report["structural_checks"]["rank_domain_valid"] is False
