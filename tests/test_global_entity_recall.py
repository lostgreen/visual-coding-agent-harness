from __future__ import annotations

from vcah.global_entity_recall import build_global_entity_recall_report


def _cases(fused_hits: int, coverage: int) -> list[dict[str, object]]:
    rows = []
    for index in range(10):
        rows.append(
            {
                "case_id": f"case-{index}",
                "entity_query": ["named entity"],
                "anchor_intervals": [[10.0, 20.0]],
                "baseline_rank": 1 if index == 0 else None,
                "entity_rank": 1 if index < fused_hits else None,
                "fused_rank": 2 if index < fused_hits else None,
                "gold_anchor_entity_covered": index < coverage,
                "same_entity_occurrence_count": 2,
                "non_gold_entity_document_rate": 0.5,
            }
        )
    return rows


def test_global_entity_report_go_is_an_endpoint_decision() -> None:
    report = build_global_entity_recall_report(
        _cases(8, 8),
        expected_cases=10,
        extraction_gate_passed=True,
        duplicate_stats={"duplicate_entity_rate": 0.1},
    )
    assert report["structural_gate_passed"] is True
    assert report["decision"] == "GO_GLOBAL_ENTITY_SIDECAR"
    assert report["retrieval"]["caption_entity_rrf"]["at_5"]["count"] == 8
    assert report["endpoint_values_were_not_structural_gates"] is True


def test_global_entity_report_separates_low_coverage_from_confusion() -> None:
    low = build_global_entity_recall_report(
        _cases(4, 4),
        expected_cases=10,
        extraction_gate_passed=True,
        duplicate_stats={},
    )
    confusion = build_global_entity_recall_report(
        _cases(4, 7),
        expected_cases=10,
        extraction_gate_passed=True,
        duplicate_stats={},
    )
    assert low["decision"] == "STOP_FIXED3_LOW_ENTITY_COVERAGE"
    assert confusion["decision"] == "RETRIEVAL_CONFUSION_WITH_ENTITY_COVERAGE"


def test_structural_failure_overrides_endpoint() -> None:
    report = build_global_entity_recall_report(
        _cases(10, 10),
        expected_cases=10,
        extraction_gate_passed=False,
        duplicate_stats={},
    )
    assert report["decision"] == "STRUCTURAL_FAILURE"
