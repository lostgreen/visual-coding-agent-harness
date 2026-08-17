from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import stable_digest
from vcah.occurrence_visual_probe import (
    VISUAL_PROBE_CONTRACT,
    VISUAL_PROBE_VERDICTS,
)


VISUAL_GEOMETRY_CONTRACT = "wp15_zero_model_paired_visual_geometry_v1"
JOINT_STRONG_MIN_WINS = 10
JOINT_STRONG_MAX_LOSSES = 1
COVERAGE_DOMINANT_MIN_CASES_PER_STRATUM = 3
COVERAGE_DOMINANT_MIN_SUPPORT_GAP = 0.30

PAIR_CATEGORIES = (
    "matched_only_supported",
    "both_supported",
    "mismatched_only_supported",
    "neither_supported",
)


def build_visual_geometry_report(
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    evaluation_records: Mapping[str, Mapping[str, Any]],
    *,
    expected_cases: int,
    expected_items: int,
    joint_min_wins: int = JOINT_STRONG_MIN_WINS,
    joint_max_losses: int = JOINT_STRONG_MAX_LOSSES,
    coverage_min_cases_per_stratum: int = COVERAGE_DOMINANT_MIN_CASES_PER_STRATUM,
    coverage_min_support_gap: float = COVERAGE_DOMINANT_MIN_SUPPORT_GAP,
) -> dict[str, Any]:
    cases = tuple(
        dict(row)
        for row in tuple(manifest.get("cases", ()) or ())
        if isinstance(row, Mapping) and bool(row.get("eligible"))
    )
    items = {
        str(item.get("item_id", "") or ""): dict(item)
        for case in cases
        for item in tuple(case.get("items", ()) or ())
        if isinstance(item, Mapping) and str(item.get("item_id", "") or "")
    }
    result_by_id: dict[str, dict[str, Any]] = {}
    duplicate_result_ids = 0
    for raw in results:
        item_id = str(raw.get("item_id", "") or "")
        if item_id in result_by_id:
            duplicate_result_ids += 1
        result_by_id[item_id] = dict(raw)

    group_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item_id, item in items.items():
        result = result_by_id.get(item_id)
        if not isinstance(result, Mapping):
            continue
        pair_kind = str(item.get("pair_kind", "") or "")
        group_id = str(item.get("pair_group_id", "") or "")
        if pair_kind in {"matched", "mismatched"} and group_id:
            group_rows[group_id][pair_kind] = {
                **item,
                "verdict": str(result.get("verdict", "") or ""),
            }

    paired_rows = []
    for group_id in sorted(group_rows):
        pair = group_rows[group_id]
        matched = pair.get("matched")
        mismatched = pair.get("mismatched")
        if not isinstance(matched, Mapping) or not isinstance(mismatched, Mapping):
            continue
        matched_supported = matched.get("verdict") == "supported"
        mismatched_supported = mismatched.get("verdict") == "supported"
        category = _pair_category(matched_supported, mismatched_supported)
        paired_rows.append(
            {
                "pair_group_id": group_id,
                "case_id": str(matched.get("case_id", "") or ""),
                "constraint_id": str(matched.get("constraint_id", "") or ""),
                "constraint_type": str(matched.get("constraint_type", "") or ""),
                "matched_verdict": str(matched.get("verdict", "") or ""),
                "mismatched_verdict": str(mismatched.get("verdict", "") or ""),
                "support_geometry": category,
            }
        )

    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    overall = Counter()
    for row in paired_rows:
        category = str(row["support_geometry"])
        overall[category] += 1
        by_type[str(row["constraint_type"])][category] += 1

    paired_geometry = {
        "count": len(paired_rows),
        "counts": _complete_counts(overall),
        "rates": _rates(overall, len(paired_rows)),
        "by_constraint_type": {
            constraint_type: {
                "count": sum(counts.values()),
                "counts": _complete_counts(counts),
                "rates": _rates(counts, sum(counts.values())),
            }
            for constraint_type, counts in sorted(by_type.items())
        },
        "by_constraint": paired_rows,
    }

    case_rows = []
    paired_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        paired_by_case[str(row["case_id"])].append(row)
    joint_outcomes = Counter()
    margins = Counter()
    for case_id in sorted(paired_by_case):
        rows = paired_by_case[case_id]
        matched_count = sum(row["matched_verdict"] == "supported" for row in rows)
        mismatched_count = sum(
            row["mismatched_verdict"] == "supported" for row in rows
        )
        margin = matched_count - mismatched_count
        outcome = "win" if margin > 0 else "loss" if margin < 0 else "tie"
        joint_outcomes[outcome] += 1
        margins[str(margin)] += 1
        case_rows.append(
            {
                "case_id": case_id,
                "constraint_count": len(rows),
                "matched_supported_count": matched_count,
                "mismatched_supported_count": mismatched_count,
                "margin": margin,
                "outcome": outcome,
            }
        )
    case_joint = {
        "case_count": len(case_rows),
        "wins": joint_outcomes["win"],
        "ties": joint_outcomes["tie"],
        "losses": joint_outcomes["loss"],
        "margin_distribution": dict(
            sorted(margins.items(), key=lambda row: int(row[0]))
        ),
        "cases": case_rows,
    }

    coverage_rows = []
    case_by_id = {
        str(case.get("case_id", "") or ""): case for case in cases
    }
    for case_id in sorted(case_by_id):
        case = case_by_id[case_id]
        evaluation = evaluation_records.get(case_id, {})
        clues = _intervals(evaluation.get("clue_intervals", ()))
        matched_windows = tuple(
            row
            for row in tuple(case.get("windows", ()) or ())
            if isinstance(row, Mapping) and row.get("pair_kind") == "matched"
        )
        frames = tuple(
            row
            for window in matched_windows
            for row in tuple(window.get("frames", ()) or ())
            if isinstance(row, Mapping)
            and isinstance(row.get("virtual_time_sec"), (int, float))
        )
        frame_times = tuple(float(row["virtual_time_sec"]) for row in frames)
        clue_hit_count = sum(
            any(start <= timestamp <= end for start, end in clues)
            for timestamp in frame_times
        )
        nearest_distance = (
            min(_distance_to_intervals(timestamp, clues) for timestamp in frame_times)
            if frame_times and clues
            else None
        )
        support_rows = paired_by_case.get(case_id, ())
        supported_count = sum(
            row["matched_verdict"] == "supported" for row in support_rows
        )
        coverage_rows.append(
            {
                "case_id": case_id,
                "clue_interval_count": len(clues),
                "sampled_frame_count": len(frame_times),
                "clue_hit_frame_count": clue_hit_count,
                "has_clue_frame": clue_hit_count > 0,
                "nearest_clue_distance_sec": nearest_distance,
                "constraint_count": len(support_rows),
                "matched_supported_count": supported_count,
                "matched_supported_rate": (
                    supported_count / len(support_rows) if support_rows else None
                ),
            }
        )
    covered = _coverage_stratum(coverage_rows, covered=True)
    uncovered = _coverage_stratum(coverage_rows, covered=False)
    coverage_gap = _difference(
        covered["matched_supported_rate"], uncovered["matched_supported_rate"]
    )
    frame_coverage = {
        "case_count": len(coverage_rows),
        "covered": covered,
        "uncovered": uncovered,
        "covered_minus_uncovered_supported_rate": coverage_gap,
        "cases": coverage_rows,
    }

    structural_checks = {
        "wp14_contract_matches": manifest.get("contract") == VISUAL_PROBE_CONTRACT,
        "eligible_case_count_matches_expected": len(cases) == expected_cases,
        "item_count_matches_expected": len(items) == expected_items,
        "result_item_set_exact": set(result_by_id) == set(items),
        "result_ids_unique": duplicate_result_ids == 0,
        "all_results_successful": all(
            row.get("status") == "success" for row in result_by_id.values()
        ),
        "all_verdicts_valid": all(
            row.get("verdict") in VISUAL_PROBE_VERDICTS
            for row in result_by_id.values()
        ),
        "matched_mismatched_pairs_complete": len(paired_rows) * 3 == len(items),
        "case_joint_coverage_complete": len(case_rows) == len(cases),
        "evaluation_records_complete": set(case_by_id).issubset(evaluation_records),
        "matched_frame_records_complete": all(
            row["clue_interval_count"] > 0 and row["sampled_frame_count"] > 0
            for row in coverage_rows
        ),
        "zero_model_calls": True,
        "endpoint_values_not_validity_gates": True,
    }
    structural_gate_passed = bool(cases) and all(structural_checks.values())
    joint_strong = bool(
        case_joint["wins"] >= joint_min_wins
        and case_joint["losses"] <= joint_max_losses
    )
    coverage_dominant = bool(
        covered["case_count"] >= coverage_min_cases_per_stratum
        and uncovered["case_count"] >= coverage_min_cases_per_stratum
        and coverage_gap is not None
        and coverage_gap >= coverage_min_support_gap
    )
    if not structural_gate_passed:
        decision = "STOP_INVALID_WP15_0_INPUT"
    elif joint_strong:
        decision = "PROCEED_TO_BLIND_COMPARATIVE_VISUAL_PROBE"
    elif coverage_dominant:
        decision = "PROCEED_TO_CONTEXT_AWARE_FIXED_SAMPLING"
    else:
        decision = "STOP_AND_REDESIGN_VISUAL_EVIDENCE"
    return {
        "schema_version": "MMLifelongOccurrenceVisualGeometryReportV1",
        "contract": VISUAL_GEOMETRY_CONTRACT,
        "study": "WP15-0 zero-model paired visual evidence diagnosis",
        "scope": "frozen39 mechanism-development exploratory diagnostic",
        "zero_model_calls": True,
        "qa_judge_run": False,
        "source_manifest_digest": stable_digest(manifest),
        "source_results_digest": stable_digest(
            sorted(
                (dict(row) for row in results),
                key=lambda row: str(row.get("item_id", "")),
            )
        ),
        "structural_checks": structural_checks,
        "structural_gate_passed": structural_gate_passed,
        "paired_support_geometry": paired_geometry,
        "case_level_joint_evidence": case_joint,
        "matched_frame_coverage": frame_coverage,
        "branch_thresholds": {
            "joint_min_wins": joint_min_wins,
            "joint_max_losses": joint_max_losses,
            "coverage_min_cases_per_stratum": coverage_min_cases_per_stratum,
            "coverage_min_support_gap": coverage_min_support_gap,
            "frozen_before_wp15_0_outcomes": True,
            "wp14_gap_threshold_changed": False,
        },
        "branch_checks": {
            "joint_evidence_strong": joint_strong,
            "frame_coverage_dominant": coverage_dominant,
        },
        "decision": decision,
        "interpretation": {
            "candidate_support_is_candidate_sufficiency": False,
            "candidate_sufficiency_is_discriminative_evidence": False,
            "unary_visual_verification_direction_rejected": False,
            "wp14_behavioral_stop_preserved": True,
        },
        "underpowered": True,
        "day_test140_accessed": False,
        "week_accessed": False,
    }


def _pair_category(matched: bool, mismatched: bool) -> str:
    if matched and mismatched:
        return "both_supported"
    if matched:
        return "matched_only_supported"
    if mismatched:
        return "mismatched_only_supported"
    return "neither_supported"


def _complete_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {category: int(counts.get(category, 0)) for category in PAIR_CATEGORIES}


def _rates(counts: Mapping[str, int], total: int) -> dict[str, float | None]:
    return {
        category: (int(counts.get(category, 0)) / total if total else None)
        for category in PAIR_CATEGORIES
    }


def _intervals(values: Any) -> tuple[tuple[float, float], ...]:
    rows = []
    for value in tuple(values or ()):
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
            and all(isinstance(item, (int, float)) for item in value)
        ):
            rows.append(tuple(sorted((float(value[0]), float(value[1])))))
    return tuple(rows)


def _distance_to_intervals(
    timestamp: float, intervals: Sequence[tuple[float, float]]
) -> float | None:
    if not intervals:
        return None
    return min(
        0.0
        if start <= timestamp <= end
        else min(abs(timestamp - start), abs(timestamp - end))
        for start, end in intervals
    )


def _coverage_stratum(
    rows: Sequence[Mapping[str, Any]], *, covered: bool
) -> dict[str, Any]:
    selected = [row for row in rows if bool(row.get("has_clue_frame")) is covered]
    constraints = sum(int(row.get("constraint_count", 0) or 0) for row in selected)
    supported = sum(
        int(row.get("matched_supported_count", 0) or 0) for row in selected
    )
    return {
        "case_count": len(selected),
        "constraint_count": constraints,
        "matched_supported_count": supported,
        "matched_supported_rate": supported / constraints if constraints else None,
    }


def _difference(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left) - float(right)
