from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from vcah.occurrence_ocr import ocr_text_has_query_evidence


def matching_entity_occurrences(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    query_terms: Sequence[str],
    anchor_intervals: Sequence[Sequence[float]],
    tolerance_sec: float,
) -> tuple[dict[str, Any], ...]:
    tolerance = float(tolerance_sec)
    if tolerance < 0.0:
        raise ValueError("coverage tolerance cannot be negative")
    intervals = _normalized_intervals(anchor_intervals)
    if not tuple(str(value).strip() for value in query_terms if str(value).strip()):
        raise ValueError("entity query terms cannot be empty")
    matches = []
    for raw in occurrences:
        text = str(raw.get("text", "") or "").strip()
        start = raw.get("occurrence_start_sec")
        end = raw.get("occurrence_end_sec")
        if (
            not text
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            continue
        if not ocr_text_has_query_evidence(text, query_terms):
            continue
        if not _overlaps_with_tolerance(
            float(start), float(end), intervals=intervals, tolerance=tolerance
        ):
            continue
        matches.append(
            {
                "occurrence_id": str(raw.get("occurrence_id", "") or ""),
                "text": text,
                "start_sec": float(start),
                "end_sec": float(end),
            }
        )
    return tuple(
        sorted(
            matches,
            key=lambda row: (
                float(row["start_sec"]),
                float(row["end_sec"]),
                str(row["occurrence_id"]),
            ),
        )
    )


def build_change_triggered_coverage_report(
    *,
    case_specs: Sequence[Mapping[str, Any]],
    arm_occurrences: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerance_sec: float,
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    arms = tuple(sorted(str(value) for value in arm_occurrences))
    if arms != ("a1_uniform", "a2_change"):
        raise ValueError("coverage report requires a1_uniform and a2_change")
    rows = []
    seen = set()
    for raw in case_specs:
        case_id = str(raw.get("case_id", "") or "")
        expectation = str(raw.get("anchor_text_expected", "") or "")
        if not case_id or case_id in seen:
            raise ValueError("case IDs must be nonempty and unique")
        if expectation not in {"yes", "no", "uncertain"}:
            raise ValueError(f"{case_id}: invalid anchor text expectation")
        seen.add(case_id)
        entity_query = tuple(raw.get("entity_query", ()) or ())
        intervals = tuple(raw.get("anchor_intervals", ()) or ())
        arm_values = {}
        for arm in arms:
            matches = matching_entity_occurrences(
                arm_occurrences[arm],
                query_terms=entity_query,
                anchor_intervals=intervals,
                tolerance_sec=tolerance_sec,
            )
            arm_values[arm] = {
                "covered": bool(matches),
                "match_count": len(matches),
                "matching_occurrences": list(matches),
            }
        rows.append(
            {
                "case_id": case_id,
                "anchor_text_expected": expectation,
                "entity_query": list(entity_query),
                "anchor_intervals": [list(value) for value in intervals],
                "arms": arm_values,
            }
        )

    strict_rows = tuple(row for row in rows if row["anchor_text_expected"] == "yes")
    strict = {arm: _coverage_summary(strict_rows, arm=arm) for arm in arms}
    all_cases = {arm: _coverage_summary(rows, arm=arm) for arm in arms}
    wins = sum(
        row["arms"]["a2_change"]["covered"] and not row["arms"]["a1_uniform"]["covered"]
        for row in strict_rows
    )
    losses = sum(
        row["arms"]["a1_uniform"]["covered"] and not row["arms"]["a2_change"]["covered"]
        for row in strict_rows
    )
    ties = len(strict_rows) - wins - losses
    paired_delta = int(strict["a2_change"]["count"]) - int(
        strict["a1_uniform"]["count"]
    )
    paired = {
        "count_delta": paired_delta,
        "rate_delta": float(strict["a2_change"]["rate"])
        - float(strict["a1_uniform"]["rate"]),
        "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(wins, losses),
        "mcnemar_is_report_only": True,
    }
    checks = {str(key): bool(value) for key, value in structural_checks.items()}
    checks.update(
        {
            "expected_frozen10_case_count": len(rows) == 10,
            "expected_text_yes_primary_count": len(strict_rows) == 8,
            "all_cases_have_entity_query": all(row["entity_query"] for row in rows),
            "all_cases_have_anchor_intervals": all(
                row["anchor_intervals"] for row in rows
            ),
        }
    )
    checks["structural_gate_passed"] = all(checks.values())
    a2_count = int(strict["a2_change"]["count"])
    if paired_delta >= 2 and a2_count >= 4:
        decision = "GO_TO_PHASE_7B"
    elif paired_delta == 1 or a2_count == 3:
        decision = "PARTIAL_RUN_TIER0_MISS_AUDIT"
    else:
        decision = "NO_GO_PENDING_TIER0_MISS_AUDIT"
    if not checks["structural_gate_passed"]:
        decision = "STRUCTURAL_FAILURE"
    return {
        "schema_version": "MMLifelongChangeTriggeredEntityCoverageReportV1",
        "decision": decision,
        "case_count": len(rows),
        "primary_denominator": "anchor_text_expected_yes_only",
        "coverage_anchor_tolerance_sec": float(tolerance_sec),
        "strict_text_expected_yes": strict,
        "all_case_secondary": all_cases,
        "paired_a2_minus_a1": paired,
        "case_level": rows,
        "gates": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "endpoint_values_were_not_structural_gates": True,
        "frozen10_is_underpowered": True,
        "retrieval_run": False,
        "qa_run": False,
        "judge_calls": 0,
    }


def build_tier0_miss_audit_report(
    *,
    case_rows: Sequence[Mapping[str, Any]],
    diagnostic_occurrences: Sequence[Mapping[str, Any]],
    structural_checks: Mapping[str, bool],
) -> dict[str, Any]:
    rows = []
    category_counts: Counter[str] = Counter()
    seen = set()
    for raw in case_rows:
        case_id = str(raw.get("case_id", "") or "")
        expectation = str(raw.get("anchor_text_expected", "") or "")
        if not case_id or case_id in seen:
            raise ValueError("miss-audit case IDs must be nonempty and unique")
        if expectation not in {"yes", "no", "uncertain"}:
            raise ValueError(f"{case_id}: invalid anchor text expectation")
        seen.add(case_id)
        matches = matching_entity_occurrences(
            diagnostic_occurrences,
            query_terms=tuple(raw.get("entity_query", ()) or ()),
            anchor_intervals=tuple(raw.get("anchor_intervals", ()) or ()),
            tolerance_sec=0.0,
        )
        if matches:
            category = "ui_text_exists_reader_or_resolution_failure"
        elif expectation == "no":
            category = "no_ui_text_visual_event_or_state"
        elif expectation == "uncertain":
            category = "annotation_uncertain"
        else:
            category = "other"
        category_counts[category] += 1
        rows.append(
            {
                "case_id": case_id,
                "anchor_text_expected": expectation,
                "entity_query": list(raw.get("entity_query", ()) or ()),
                "anchor_intervals": [
                    list(value) for value in tuple(raw.get("anchor_intervals", ()) or ())
                ],
                "category": category,
                "diagnostic_match_count": len(matches),
                "diagnostic_matching_occurrences": list(matches),
            }
        )
    strict_rows = tuple(row for row in rows if row["anchor_text_expected"] == "yes")
    strict_recovered = sum(
        row["diagnostic_match_count"] > 0 for row in strict_rows
    )
    checks = {str(key): bool(value) for key, value in structural_checks.items()}
    checks.update(
        {
            "nonempty_miss_case_set": bool(rows),
            "all_cases_assigned_one_category": sum(category_counts.values())
            == len(rows),
            "diagnostic_is_not_endpoint": True,
            "diagnostic_is_not_upper_bound": True,
        }
    )
    checks["structural_gate_passed"] = all(checks.values())
    if not checks["structural_gate_passed"]:
        decision = "STRUCTURAL_FAILURE"
    elif strict_recovered:
        decision = "CONTINUE_READER_OR_SAMPLING_REPAIR"
    else:
        decision = "STOP_NO_VISIBLE_ENTITY_RECOVERY_AT_TIER0"
    return {
        "schema_version": "MMLifelongTier0MissAuditReportV1",
        "decision": decision,
        "case_count": len(rows),
        "strict_text_expected_yes": {
            "recovered_count": strict_recovered,
            "case_count": len(strict_rows),
            "rate": strict_recovered / len(strict_rows) if strict_rows else 0.0,
        },
        "category_counts": dict(sorted(category_counts.items())),
        "case_level": rows,
        "gates": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "endpoint_evaluation": False,
        "upper_bound_claim": False,
        "retrieval_run": False,
        "qa_run": False,
        "judge_calls": 0,
    }


def _coverage_summary(rows: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    count = sum(bool(row["arms"][arm]["covered"]) for row in rows)
    return {
        "count": count,
        "case_count": len(rows),
        "rate": count / len(rows) if rows else 0.0,
    }


def _exact_mcnemar_p(wins: int, losses: int) -> float:
    first = max(0, int(wins))
    second = max(0, int(losses))
    discordant = first + second
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(first, second) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _normalized_intervals(
    values: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    intervals = []
    for value in values:
        if len(value) != 2:
            raise ValueError("anchor intervals must be [start, end]")
        start, end = float(value[0]), float(value[1])
        if end < start:
            raise ValueError("anchor interval end precedes start")
        intervals.append((start, end))
    if not intervals:
        raise ValueError("anchor intervals cannot be empty")
    return tuple(intervals)


def _overlaps_with_tolerance(
    start: float,
    end: float,
    *,
    intervals: Sequence[tuple[float, float]],
    tolerance: float,
) -> bool:
    return any(
        min(float(end), interval_end + tolerance)
        >= max(float(start), interval_start - tolerance)
        for interval_start, interval_end in intervals
    )
