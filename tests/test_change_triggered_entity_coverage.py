from vcah.change_triggered_entity_coverage import (
    build_change_triggered_coverage_report,
    build_tier0_miss_audit_report,
    matching_entity_occurrences,
)


def _occurrence(identifier: str, text: str, start: float, end: float) -> dict:
    return {
        "occurrence_id": identifier,
        "text": text,
        "occurrence_start_sec": start,
        "occurrence_end_sec": end,
    }


def test_matching_entity_occurrences_requires_text_and_time() -> None:
    matches = matching_entity_occurrences(
        (
            _occurrence("near", "黑风大王", 95.0, 96.0),
            _occurrence("wrong-text", "赤髯龙", 100.0, 101.0),
            _occurrence("far", "黑风大王", 300.0, 301.0),
        ),
        query_terms=("Black Wind King", "黑风大王"),
        anchor_intervals=((100.0, 110.0),),
        tolerance_sec=10.0,
    )
    assert [row["occurrence_id"] for row in matches] == ["near"]


def test_coverage_report_uses_yes_only_primary_and_paired_decision() -> None:
    cases = []
    a1 = []
    a2 = []
    for index in range(10):
        case_id = f"case-{index}"
        expectation = "yes" if index < 8 else ("no" if index == 8 else "uncertain")
        cases.append(
            {
                "case_id": case_id,
                "anchor_text_expected": expectation,
                "entity_query": (f"entity-{index}",),
                "anchor_intervals": ((index * 100.0, index * 100.0 + 5.0),),
            }
        )
        if index in {0, 1}:
            a1.append(
                _occurrence(
                    f"a1-{index}", f"entity-{index}", index * 100.0, index * 100.0
                )
            )
        if index in {0, 1, 2, 3}:
            a2.append(
                _occurrence(
                    f"a2-{index}", f"entity-{index}", index * 100.0, index * 100.0
                )
            )
    report = build_change_triggered_coverage_report(
        case_specs=cases,
        arm_occurrences={"a1_uniform": a1, "a2_change": a2},
        tolerance_sec=60.0,
        structural_checks={"runtime_inputs_valid": True},
    )
    assert report["decision"] == "GO_TO_PHASE_7B"
    assert report["strict_text_expected_yes"]["a1_uniform"]["count"] == 2
    assert report["strict_text_expected_yes"]["a2_change"]["count"] == 4
    assert report["paired_a2_minus_a1"]["wins_ties_losses"] == {
        "wins": 2,
        "ties": 6,
        "losses": 0,
    }
    assert report["paired_a2_minus_a1"]["mcnemar_exact_two_sided_p"] == 0.5
    assert report["structural_gate_passed"] is True


def test_endpoint_values_do_not_override_structural_failure() -> None:
    cases = [
        {
            "case_id": f"case-{index}",
            "anchor_text_expected": "yes" if index < 8 else "no",
            "entity_query": (f"entity-{index}",),
            "anchor_intervals": ((0.0, 1.0),),
        }
        for index in range(10)
    ]
    report = build_change_triggered_coverage_report(
        case_specs=cases,
        arm_occurrences={"a1_uniform": (), "a2_change": ()},
        tolerance_sec=60.0,
        structural_checks={"runtime_inputs_valid": False},
    )
    assert report["decision"] == "STRUCTURAL_FAILURE"
    assert report["endpoint_values_were_not_structural_gates"] is True


def test_tier0_miss_audit_routes_recovered_text_without_becoming_endpoint() -> None:
    rows = (
        {
            "case_id": "case-yes",
            "anchor_text_expected": "yes",
            "entity_query": ("Black Wind King", "黑风大王"),
            "anchor_intervals": ((100.0, 110.0),),
        },
        {
            "case_id": "case-no",
            "anchor_text_expected": "no",
            "entity_query": ("fourth meditation point",),
            "anchor_intervals": ((200.0, 210.0),),
        },
    )
    report = build_tier0_miss_audit_report(
        case_rows=rows,
        diagnostic_occurrences=(
            _occurrence("visible", "黑风大王", 103.0, 104.0),
        ),
        structural_checks={"runtime_inputs_valid": True},
    )
    assert report["decision"] == "CONTINUE_READER_OR_SAMPLING_REPAIR"
    assert report["strict_text_expected_yes"]["recovered_count"] == 1
    assert report["category_counts"] == {
        "no_ui_text_visual_event_or_state": 1,
        "ui_text_exists_reader_or_resolution_failure": 1,
    }
    assert report["endpoint_evaluation"] is False
    assert report["upper_bound_claim"] is False
