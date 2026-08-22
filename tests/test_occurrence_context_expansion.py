from __future__ import annotations

from vcah.occurrence_context_expansion import (
    build_occurrence_context_expansion_report,
)


def _bundle(
    bundle_id: str,
    intervals: list[list[float]],
    *,
    rank: int = 1,
) -> dict:
    return {
        "bundle_id": bundle_id,
        "rank": rank,
        "seed_passage_ids": [f"seed-{bundle_id}"],
        "context_passage_ids": [],
        "source_video_ids": ["video-a"],
        "event_boundaries_preserved": True,
        "member_passages": [
            {
                "passage_id": f"p-{bundle_id}-{index}",
                "time_range": interval,
                "role": "seed" if index == 0 else "context",
            }
            for index, interval in enumerate(intervals)
        ],
    }


def _variant(seed_id: str, bundles: list[dict], *, context_count: int = 0) -> dict:
    return {
        "seed_hit_ids": [seed_id],
        "seed_hit_count": 1,
        "context_hit_count": context_count,
        "cross_caption_context_count": context_count,
        "bundle_set": {"bundles": bundles},
    }


def _case(case_id: str, *, baseline_gold: bool, expanded_gold: bool) -> dict:
    miss = _bundle(f"{case_id}-miss", [[300.0, 310.0]])
    baseline = _bundle(
        f"{case_id}-base",
        [[101.0, 105.0]] if baseline_gold else [[300.0, 310.0]],
    )
    expanded = _bundle(
        f"{case_id}-expanded",
        (
            [[300.0, 310.0], [102.0, 106.0]]
            if expanded_gold and not baseline_gold
            else [[101.0, 105.0]]
            if baseline_gold
            else [[300.0, 310.0]]
        ),
    )
    return {
        "case_id": case_id,
        "clues": [[100.0, 110.0]],
        "packets": [
            {
                "variants": {
                    "baseline": _variant("seed", [baseline]),
                    "neighbors_1": _variant(
                        "seed", [expanded], context_count=int(expanded_gold)
                    ),
                    "neighbors_2": _variant("seed", [expanded, miss], context_count=2),
                }
            }
        ],
    }


def test_context_expansion_selects_smallest_variant_meeting_target() -> None:
    cases = (
        _case("present", baseline_gold=True, expanded_gold=True),
        _case("recovered", baseline_gold=False, expanded_gold=True),
    )

    report = build_occurrence_context_expansion_report(
        cases,
        expected_cases=2,
        variant_order=("neighbors_1", "neighbors_2"),
        target_recall_count=2,
        target_recovery_count=1,
    )

    assert report["structural_gate_passed"] is True
    assert report["baseline"]["at_5"]["count"] == 1
    assert report["variants"]["neighbors_1"]["at_5"]["count"] == 2
    assert (
        report["variants"]["neighbors_1"]["recovered_from_baseline_absent_count"]
        == 1
    )
    assert report["selected_variant"] == "neighbors_1"
    assert report["decision"] == "PROCEED_TO_RUNTIME_CANARY"


def test_context_expansion_rejects_seed_drift() -> None:
    case = _case("drift", baseline_gold=False, expanded_gold=True)
    case["packets"][0]["variants"]["neighbors_1"]["seed_hit_ids"] = ["changed"]

    report = build_occurrence_context_expansion_report(
        (case,),
        expected_cases=1,
        variant_order=("neighbors_1", "neighbors_2"),
        target_recall_count=1,
        target_recovery_count=1,
    )

    assert report["structural_gate_passed"] is False
    assert report["decision"] == "STOP_STRUCTURAL_GATE_FAILED"
    assert any("seed_drift" in error for error in report["structural_errors"])
