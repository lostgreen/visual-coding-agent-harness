from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence


RECALL_KS = (1, 3, 5, 10)


def build_global_entity_recall_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    extraction_gate_passed: bool,
    duplicate_stats: Mapping[str, Any],
) -> dict[str, Any]:
    rows = tuple(dict(row) for row in cases)
    case_ids = tuple(str(row.get("case_id", "") or "") for row in rows)
    ranks_valid = all(
        _rank(row.get(key)) is not False
        for row in rows
        for key in ("baseline_rank", "entity_rank", "fused_rank")
    )
    gates = {
        "expected_case_count": len(rows) == int(expected_cases),
        "case_ids_unique_and_nonempty": len(set(case_ids)) == len(rows)
        and all(case_ids),
        "extraction_structural_gate_passed": bool(extraction_gate_passed),
        "all_rank_values_valid": ranks_valid,
        "all_cases_have_entity_query": all(
            tuple(row.get("entity_query", ()) or ()) for row in rows
        ),
        "all_cases_have_anchor_intervals": all(
            tuple(row.get("anchor_intervals", ()) or ()) for row in rows
        ),
    }
    gates["structural_gate_passed"] = all(gates.values())
    baseline = _recall_summary(rows, "baseline_rank")
    entity = _recall_summary(rows, "entity_rank")
    fused = _recall_summary(rows, "fused_rank")
    coverage_count = sum(bool(row.get("gold_anchor_entity_covered")) for row in rows)
    fused_r5 = int(fused["at_5"]["count"])
    if fused_r5 >= 8:
        decision = "GO_GLOBAL_ENTITY_SIDECAR"
    elif fused_r5 >= 5:
        decision = "PARTIAL_DIAGNOSE_COVERAGE_VS_CONFUSION"
    elif coverage_count <= 4:
        decision = "STOP_FIXED3_LOW_ENTITY_COVERAGE"
    else:
        decision = "RETRIEVAL_CONFUSION_WITH_ENTITY_COVERAGE"
    if not gates["structural_gate_passed"]:
        decision = "STRUCTURAL_FAILURE"
    non_gold_rates = [
        float(row.get("non_gold_entity_document_rate", 0.0) or 0.0) for row in rows
    ]
    same_entity_counts = [
        int(row.get("same_entity_occurrence_count", 0) or 0) for row in rows
    ]
    recovered = [
        row["case_id"]
        for row in rows
        if _within(row.get("fused_rank"), 5)
        and not _within(row.get("baseline_rank"), 5)
    ]
    regressed = [
        row["case_id"]
        for row in rows
        if _within(row.get("baseline_rank"), 5)
        and not _within(row.get("fused_rank"), 5)
    ]
    return {
        "schema_version": "MMLifelongGlobalEntityRecallReportV1",
        "case_count": len(rows),
        "decision": decision,
        "structural_gate_passed": gates["structural_gate_passed"],
        "gates": gates,
        "retrieval": {
            "caption_baseline": baseline,
            "entity_lexical": entity,
            "caption_entity_rrf": fused,
            "paired_delta_at_5": {
                "count": fused_r5 - int(baseline["at_5"]["count"]),
                "rate": float(fused["at_5"]["rate"])
                - float(baseline["at_5"]["rate"]),
            },
            "recovered_case_ids": recovered,
            "regressed_case_ids": regressed,
        },
        "entity_coverage": {
            "count": coverage_count,
            "case_count": len(rows),
            "rate": coverage_count / len(rows) if rows else 0.0,
        },
        "false_positive_diagnostics": {
            "mean_same_entity_occurrence_count": (
                statistics.fmean(same_entity_counts) if same_entity_counts else 0.0
            ),
            "maximum_same_entity_occurrence_count": max(same_entity_counts, default=0),
            "mean_non_gold_entity_document_rate": (
                statistics.fmean(non_gold_rates) if non_gold_rates else 0.0
            ),
            "duplicate_stats": dict(duplicate_stats),
        },
        "case_level": list(rows),
        "endpoint_values_were_not_structural_gates": True,
        "bounded_search_run": False,
        "qa_run": False,
        "judge_calls": 0,
    }


def _recall_summary(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    return {
        f"at_{k}": {
            "count": sum(_within(row.get(key), k) for row in rows),
            "case_count": len(rows),
            "rate": (
                sum(_within(row.get(key), k) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        }
        for k in RECALL_KS
    }


def _within(value: Any, limit: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= limit


def _rank(value: Any) -> int | None | bool:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return False
    return value
