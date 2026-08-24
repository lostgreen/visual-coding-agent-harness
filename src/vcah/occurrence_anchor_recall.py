from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ANCHOR_RECALL_REPORT_CONTRACT = "WP16-4-anchor-recall-report-v1"
ANCHOR_RECALL_KS = (1, 3, 5, 10)


def build_anchor_recall_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    variant_order: Sequence[str],
) -> dict[str, Any]:
    rows = tuple(dict(case) for case in cases)
    variants = tuple(dict.fromkeys(str(value) for value in variant_order))
    case_ids = tuple(str(case.get("case_id", "") or "") for case in rows)
    checks = {
        "expected_case_count": len(rows) == int(expected_cases),
        "unique_case_ids": len(set(case_ids)) == len(case_ids),
        "single_anchor_subset_excludes_0115": all(
            not case_id.endswith("-0115") for case_id in case_ids
        ),
        "manual_queries_frozen_before_anchor_outcomes": all(
            bool(case.get("queries_frozen_before_anchor_outcomes")) for case in rows
        ),
        "anchor_identity_labeled": all(
            bool(case.get("anchor_description")) and bool(case.get("anchor_intervals"))
            for case in rows
        ),
        "query_variants_complete": all(
            set(variants) == set(dict(case.get("ranks", {}))) for case in rows
        ),
        "rank_domain_valid": all(
            rank is None or _valid_rank(rank)
            for case in rows
            for rank in dict(case.get("ranks", {})).values()
        ),
    }
    summaries = {variant: _variant_summary(rows, variant) for variant in variants}
    full_name = "full_question_hybrid"
    anchor_name = "anchor_only_hybrid"
    event_name = "anchor_event_hybrid"
    full_at5 = _recall_count(summaries.get(full_name, {}), 5)
    anchor_at5 = _recall_count(summaries.get(anchor_name, {}), 5)
    event_at5 = _recall_count(summaries.get(event_name, {}), 5)
    target = math.ceil(0.8 * max(1, len(rows)))
    limited = math.floor(0.5 * len(rows))
    if anchor_at5 >= target:
        decision = "ANCHOR_ONLY_QUERY_SUFFICIENT"
    elif event_at5 >= target:
        decision = "AUTOMATE_ANCHOR_EVENT_REPRESENTATION"
    elif event_at5 <= limited:
        decision = "ANCHOR_RECALL_REQUIRES_GLOBAL_MULTIMODAL_INDEX"
    else:
        decision = "MIXED_ANCHOR_RECALL"

    comparisons = {}
    if full_name in summaries:
        for variant in variants:
            if variant == full_name:
                continue
            comparisons[f"{variant}_vs_{full_name}_at5"] = _paired_at_k(
                rows,
                treatment=variant,
                control=full_name,
                k=5,
            )
    return {
        "contract": ANCHOR_RECALL_REPORT_CONTRACT,
        "case_count": len(rows),
        "case_ids": list(case_ids),
        "structural_checks": checks,
        "structural_gate_passed": all(checks.values()),
        "endpoint_values_are_gates": False,
        "decision": decision,
        "decision_thresholds": {
            "target_recall_at5_count": target,
            "limited_recall_at5_count": limited,
        },
        "variants": summaries,
        "comparisons": comparisons,
        "per_case": [
            {
                "case_id": str(case.get("case_id", "") or ""),
                "anchor_description": str(case.get("anchor_description", "") or ""),
                "ranks": dict(case.get("ranks", {})),
            }
            for case in rows
        ],
        "diagnostics": {
            "full_question_at5": full_at5,
            "anchor_only_at5": anchor_at5,
            "anchor_event_at5": event_at5,
            "manual_event_queries_are_oracle_enriched": True,
        },
    }


def _variant_summary(
    cases: Sequence[Mapping[str, Any]],
    variant: str,
) -> dict[str, Any]:
    ranks = tuple(dict(case.get("ranks", {})).get(variant) for case in cases)
    reciprocal = tuple(1.0 / int(rank) for rank in ranks if _valid_rank(rank))
    return {
        "recall": {
            f"at_{k}": {
                "count": sum(_valid_rank(rank) and int(rank) <= k for rank in ranks),
                "case_count": len(cases),
                "rate": (
                    sum(_valid_rank(rank) and int(rank) <= k for rank in ranks)
                    / len(cases)
                    if cases
                    else None
                ),
            }
            for k in ANCHOR_RECALL_KS
        },
        "mean_reciprocal_rank": (sum(reciprocal) / len(cases) if cases else None),
        "not_found_count": sum(rank is None for rank in ranks),
    }


def _paired_at_k(
    cases: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
    k: int,
) -> dict[str, Any]:
    recovered: list[str] = []
    regressed: list[str] = []
    retained: list[str] = []
    missed: list[str] = []
    for case in cases:
        ranks = dict(case.get("ranks", {}))
        control_hit = _valid_rank(ranks.get(control)) and int(ranks[control]) <= k
        treatment_hit = _valid_rank(ranks.get(treatment)) and int(ranks[treatment]) <= k
        case_id = str(case.get("case_id", "") or "")
        if treatment_hit and not control_hit:
            recovered.append(case_id)
        elif control_hit and not treatment_hit:
            regressed.append(case_id)
        elif control_hit and treatment_hit:
            retained.append(case_id)
        else:
            missed.append(case_id)
    return {
        "recovered_case_ids": recovered,
        "regressed_case_ids": regressed,
        "retained_case_ids": retained,
        "missed_case_ids": missed,
        "net_recovery": len(recovered) - len(regressed),
    }


def _recall_count(summary: Mapping[str, Any], k: int) -> int:
    return int(
        dict(dict(summary.get("recall", {})).get(f"at_{k}", {})).get("count", 0) or 0
    )


def _valid_rank(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1
