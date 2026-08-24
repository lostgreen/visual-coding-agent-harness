from __future__ import annotations

import math
from statistics import mean
from typing import Any, Callable, Mapping, Sequence


RELATION_EVIDENCE_REPORT_CONTRACT = "WP16-4-oracle-relation-evidence-report-v1"


def build_relation_evidence_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    variant_order: Sequence[str] = ("fixed_d20", "bounded_search"),
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
        "manual_labels_frozen_before_search_outcomes": all(
            bool(case.get("labels_frozen_before_search_outcomes")) for case in rows
        ),
        "anchor_identity_labeled": all(
            bool(case.get("anchor_description")) and bool(case.get("anchor_intervals"))
            for case in rows
        ),
        "target_event_labeled": all(
            bool(case.get("target_event_description"))
            and bool(case.get("target_event_term_groups"))
            for case in rows
        ),
        "variant_set_complete": all(
            set(variants) <= set(dict(case.get("variants", {}))) for case in rows
        ),
        "oracle_anchor_resolved": all(
            bool(dict(case.get("variants", {})).get(variant, {}).get("anchor_hit"))
            for case in rows
            for variant in variants
        ),
    }

    per_case: list[dict[str, Any]] = []
    for case in rows:
        metrics = {
            variant: _case_variant_metrics(case, variant) for variant in variants
        }
        per_case.append(
            {
                "case_id": str(case.get("case_id", "") or ""),
                "question": str(case.get("question", "") or ""),
                "anchor_description": str(case.get("anchor_description", "") or ""),
                "relation": str(case.get("relation", "") or ""),
                "target_event_description": str(
                    case.get("target_event_description", "") or ""
                ),
                "target_evidence_type": list(
                    case.get("target_evidence_type", ()) or ()
                ),
                "metrics": metrics,
            }
        )

    summaries = {variant: _variant_summary(per_case, variant) for variant in variants}
    bounded = summaries.get("bounded_search", {})
    recovered = int(dict(bounded.get("evidence_recall", {})).get("count", 0) or 0)
    invest_threshold = math.ceil(0.8 * max(1, len(rows)))
    limited_threshold = math.floor(0.5 * len(rows))
    if recovered >= invest_threshold:
        decision = "INVEST_IN_ANCHOR_RETRIEVAL"
    elif recovered <= limited_threshold:
        decision = "SINGLE_ANCHOR_RELATION_CEILING_LIMITED"
    else:
        decision = "MIXED_ORACLE_CEILING"

    return {
        "contract": RELATION_EVIDENCE_REPORT_CONTRACT,
        "case_count": len(rows),
        "case_ids": list(case_ids),
        "structural_checks": checks,
        "structural_gate_passed": all(checks.values()),
        "endpoint_values_are_gates": False,
        "decision": decision,
        "decision_thresholds": {
            "invest_in_anchor_retrieval_min_evidence_cases": (invest_threshold),
            "limited_ceiling_max_evidence_cases": limited_threshold,
        },
        "variants": summaries,
        "per_case": per_case,
    }


def _case_variant_metrics(
    case: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    row = dict(dict(case.get("variants", {})).get(variant, {}) or {})
    evidence_intervals = tuple(case.get("evidence_intervals", ()) or ())
    requested = {
        str(value)
        for value in tuple(case.get("target_evidence_type", ()) or ())
        if str(value)
    }
    stop = row.get("stop_hit")
    candidates = (
        (dict(stop),)
        if isinstance(stop, Mapping)
        else tuple(
            dict(hit)
            for hit in tuple(row.get("hits", ()) or ())[1:]
            if isinstance(hit, Mapping)
        )
    )
    evidence_hits = tuple(
        hit
        for hit in candidates
        if _overlaps_any(hit.get("time_range"), evidence_intervals)
    )
    evidence = bool(evidence_hits)
    channel = any(
        requested
        <= {
            str(value)
            for value in tuple(hit.get("evidence_channels_observed", ()) or ())
            if str(value)
        }
        for hit in evidence_hits
    )
    stop_success = bool(row.get("stop_success"))
    return {
        "anchor_found": bool(row.get("anchor_hit")),
        "evidence": evidence,
        "channel": channel,
        "bound": evidence and channel,
        "stop_success": stop_success,
        "wrong_stop": stop_success and not evidence,
        "visited_passage_count": int(row.get("visited_passage_count", 0) or 0),
        "stop_reason": str(row.get("stop_reason", "") or ""),
        "stop_time_range": (
            list(stop.get("time_range", ()) or ())
            if isinstance(stop, Mapping)
            else None
        ),
        "matched_target_terms": list(row.get("matched_target_terms", ()) or ()),
    }


def _variant_summary(
    per_case: Sequence[Mapping[str, Any]],
    variant: str,
) -> dict[str, Any]:
    metrics = tuple(dict(case["metrics"][variant]) for case in per_case)
    visited = tuple(int(row["visited_passage_count"]) for row in metrics)
    stopped = tuple(row for row in metrics if row["stop_success"])
    return {
        "evidence_recall": _count_metric(metrics, lambda row: row["evidence"]),
        "channel_evidence_recall": _count_metric(metrics, lambda row: row["channel"]),
        "bound_evidence_recall": _count_metric(metrics, lambda row: row["bound"]),
        "stop_success_rate": _count_metric(metrics, lambda row: row["stop_success"]),
        "wrong_stop_rate": _count_metric(metrics, lambda row: row["wrong_stop"]),
        "wrong_stop_given_stop": _ratio(
            sum(bool(row["wrong_stop"]) for row in stopped),
            len(stopped),
        ),
        "passages_visited": {
            "mean": mean(visited) if visited else 0.0,
            "p95": _nearest_rank_percentile(visited, 0.95),
            "max": max(visited, default=0),
        },
    }


def _count_metric(
    rows: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    count = sum(bool(predicate(row)) for row in rows)
    return {
        "count": count,
        "case_count": len(rows),
        "rate": count / len(rows) if rows else None,
    }


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": int(numerator),
        "denominator": int(denominator),
        "rate": numerator / denominator if denominator else None,
    }


def _nearest_rank_percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


def _overlaps_any(
    raw_interval: Any,
    intervals: Sequence[Sequence[float]],
) -> bool:
    if not isinstance(raw_interval, Sequence) or len(raw_interval) != 2:
        return False
    start, end = sorted((float(raw_interval[0]), float(raw_interval[1])))
    return any(
        min(end, max(float(value[0]), float(value[1])))
        > max(start, min(float(value[0]), float(value[1])))
        for value in intervals
        if isinstance(value, Sequence) and len(value) == 2
    )
