from __future__ import annotations

from vcah.occurrence_relation_evidence import (
    RELATION_EVIDENCE_REPORT_CONTRACT,
    build_relation_evidence_report,
)


def _hit(start: float, end: float, channels: list[str]) -> dict:
    return {
        "passage_id": f"p-{start}",
        "time_range": [start, end],
        "evidence_channels_observed": channels,
    }


def _case(case_id: str, *, bounded_start: float) -> dict:
    anchor = _hit(10.0, 20.0, ["caption"])
    fixed_target = _hit(30.0, 40.0, ["caption", "visible_ocr"])
    bounded_target = _hit(
        bounded_start,
        bounded_start + 10.0,
        ["caption", "visible_ocr"],
    )
    return {
        "case_id": case_id,
        "question": "After the anchor, what item is obtained?",
        "anchor_description": "anchor event",
        "anchor_intervals": [[10.0, 20.0]],
        "relation": "after",
        "target_event_description": "item acquisition",
        "target_event_term_groups": [["item"]],
        "target_evidence_type": ["visible_ocr"],
        "evidence_intervals": [[30.0, 40.0]],
        "labels_frozen_before_primary_outcomes": True,
        "variants": {
            "fixed_d20": {
                "anchor_hit": anchor,
                "stop_hit": None,
                "hits": [anchor, fixed_target],
                "visited_passage_count": 20,
                "stop_success": False,
                "stop_reason": "fixed_distance_complete",
            },
            "bounded_search": {
                "anchor_hit": anchor,
                "stop_hit": bounded_target,
                "hits": [anchor, bounded_target],
                "visited_passage_count": 3,
                "stop_success": True,
                "stop_reason": "target_event_found",
                "matched_target_terms": [["item"]],
            },
        },
    }


def test_report_selects_anchor_investment_when_oracle_ceiling_is_high() -> None:
    report = build_relation_evidence_report(
        (
            _case("mmlifelong-game-test-0010", bounded_start=30.0),
            _case("mmlifelong-game-test-0028", bounded_start=30.0),
        ),
        expected_cases=2,
    )

    assert report["contract"] == RELATION_EVIDENCE_REPORT_CONTRACT
    assert report["structural_gate_passed"] is True
    assert report["decision"] == "INVEST_IN_ANCHOR_RETRIEVAL"
    bounded = report["variants"]["bounded_search"]
    assert bounded["evidence_recall"]["count"] == 2
    assert bounded["bound_evidence_recall"]["count"] == 2
    assert bounded["passages_visited"] == {"mean": 3, "p95": 3, "max": 3}


def test_report_counts_wrong_stop_without_using_it_as_a_gate() -> None:
    report = build_relation_evidence_report(
        (_case("mmlifelong-game-test-0010", bounded_start=50.0),),
        expected_cases=1,
    )

    bounded = report["variants"]["bounded_search"]
    assert bounded["stop_success_rate"]["count"] == 1
    assert bounded["wrong_stop_rate"]["count"] == 1
    assert report["structural_gate_passed"] is True
    assert report["endpoint_values_are_gates"] is False


def test_report_rejects_multi_occurrence_case_from_single_anchor_subset() -> None:
    report = build_relation_evidence_report(
        (_case("mmlifelong-game-test-0115", bounded_start=30.0),),
        expected_cases=1,
    )
    assert report["structural_gate_passed"] is False
    assert report["structural_checks"]["single_anchor_subset_excludes_0115"] is False
