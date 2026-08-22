from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence


OCCURRENCE_CONTEXT_EXPANSION_CONTRACT = "WP16-1-query-conditioned-context-v1"
RECALL_KS = (1, 3, 5)


def build_occurrence_context_expansion_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    variant_order: Sequence[str],
    target_recall_count: int = 22,
    target_recovery_count: int = 8,
) -> dict[str, Any]:
    variants = tuple(str(value) for value in variant_order if str(value))
    errors = _structural_errors(cases, expected_cases=expected_cases, variants=variants)
    baseline = _coverage_summary(cases, variant="baseline")
    baseline_absent = tuple(
        case for case in cases if not _case_has_gold(case, variant="baseline", k=5)
    )
    variant_reports: dict[str, dict[str, Any]] = {}
    selected_variant: str | None = None
    for variant in variants:
        summary = _coverage_summary(cases, variant=variant)
        recovered = tuple(
            str(case.get("case_id", ""))
            for case in baseline_absent
            if _case_has_gold(case, variant=variant, k=5)
        )
        regressed = tuple(
            str(case.get("case_id", ""))
            for case in cases
            if _case_has_gold(case, variant="baseline", k=5)
            and not _case_has_gold(case, variant=variant, k=5)
        )
        cost = _context_cost(cases, variant=variant)
        meets_target = bool(
            not regressed
            and (
                int(summary["at_5"]["count"]) >= int(target_recall_count)
                or len(recovered) >= int(target_recovery_count)
            )
        )
        variant_reports[variant] = {
            **summary,
            "recovered_from_baseline_absent_count": len(recovered),
            "recovered_from_baseline_absent_rate": _rate(
                len(recovered), len(baseline_absent)
            ),
            "recovered_case_ids": list(recovered),
            "regressed_case_count": len(regressed),
            "regressed_case_ids": list(regressed),
            "context_cost": cost,
            "meets_exploratory_target": meets_target,
        }
        if selected_variant is None and meets_target:
            selected_variant = variant

    structural_passed = not errors
    if not structural_passed:
        decision = "STOP_STRUCTURAL_GATE_FAILED"
    elif selected_variant is not None:
        decision = "PROCEED_TO_RUNTIME_CANARY"
    else:
        decision = "STOP_CONTEXT_EXPANSION_COVERAGE_INSUFFICIENT"
    return {
        "contract": OCCURRENCE_CONTEXT_EXPANSION_CONTRACT,
        "structural_gate_passed": structural_passed,
        "structural_errors": errors,
        "cohort": {
            "case_count": len(cases),
            "baseline_candidate_present_at_5": int(baseline["at_5"]["count"]),
            "baseline_candidate_absent_at_5": len(baseline_absent),
        },
        "evaluation_contract": {
            "seed_top_k": 5,
            "gold_coverage_uses_member_passage_intervals": True,
            "bundle_time_range_is_never_used_for_gold_coverage": True,
            "target_recall_count": int(target_recall_count),
            "target_recovery_count": int(target_recovery_count),
            "smallest_passing_variant_is_selected": True,
            "exploratory_frozen39": True,
        },
        "baseline": baseline,
        "variants": variant_reports,
        "selected_variant": selected_variant,
        "decision": decision,
        "notes": [
            "Retrieval seeds are frozen across variants; only post-retrieval context changes.",
            "A bundle relates seed and neighboring passages without asserting one semantic event.",
            "No Reasoner, VLM, or judge calls are used.",
        ],
    }


def _structural_errors(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    variants: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    if len(cases) != int(expected_cases):
        errors.append(f"case_count:{len(cases)}!={int(expected_cases)}")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if len(set(case_ids)) != len(case_ids) or any(not value for value in case_ids):
        errors.append("case_ids_not_unique_and_complete")
    for case in cases:
        case_id = str(case.get("case_id", ""))
        packets = tuple(case.get("packets", ()) or ())
        if not packets:
            errors.append(f"{case_id}:missing_packets")
            continue
        for packet_index, packet in enumerate(packets):
            values = packet.get("variants", {})
            if not isinstance(values, Mapping):
                errors.append(f"{case_id}:packet{packet_index}:variants_invalid")
                continue
            expected = {"baseline", *variants}
            if set(values) != expected:
                errors.append(f"{case_id}:packet{packet_index}:variant_set_mismatch")
                continue
            baseline_seed_ids = tuple(values["baseline"].get("seed_hit_ids", ()))
            for variant in variants:
                row = values[variant]
                if tuple(row.get("seed_hit_ids", ())) != baseline_seed_ids:
                    errors.append(
                        f"{case_id}:packet{packet_index}:{variant}:seed_drift"
                    )
                bundle_set = row.get("bundle_set", {})
                for bundle in tuple(bundle_set.get("bundles", ()) or ()):
                    if bundle.get("event_boundaries_preserved") is not True:
                        errors.append(
                            f"{case_id}:packet{packet_index}:{variant}:boundary_not_preserved"
                        )
                    for member in tuple(bundle.get("member_passages", ()) or ()):
                        if str(member.get("role", "")) != "context":
                            continue
                        links = tuple(member.get("context_links", ()) or ())
                        if not links or any(
                            not isinstance(link, Mapping)
                            or link.get("same_source_timeline") is not True
                            for link in links
                        ):
                            errors.append(
                                f"{case_id}:packet{packet_index}:{variant}:"
                                "context_source_link_unproven"
                            )
    return sorted(set(errors))


def _coverage_summary(
    cases: Sequence[Mapping[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    return {
        f"at_{k}": {
            "count": sum(_case_has_gold(case, variant=variant, k=k) for case in cases),
            "rate": _rate(
                sum(_case_has_gold(case, variant=variant, k=k) for case in cases),
                len(cases),
            ),
        }
        for k in RECALL_KS
    }


def _case_has_gold(case: Mapping[str, Any], *, variant: str, k: int) -> bool:
    clues = tuple(case.get("clues", ()) or ())
    for packet in tuple(case.get("packets", ()) or ()):
        variants = packet.get("variants", {})
        row = variants.get(variant, {}) if isinstance(variants, Mapping) else {}
        bundle_set = row.get("bundle_set", {}) if isinstance(row, Mapping) else {}
        for bundle in tuple(bundle_set.get("bundles", ()) or ()):
            if int(bundle.get("rank", 0) or 0) > int(k):
                continue
            for member in tuple(bundle.get("member_passages", ()) or ()):
                interval = _interval(member.get("time_range"))
                if interval is not None and any(
                    _overlap(interval, clue) > 0.0 for clue in clues
                ):
                    return True
    return False


def _context_cost(
    cases: Sequence[Mapping[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    rows = [
        packet.get("variants", {}).get(variant, {})
        for case in cases
        for packet in tuple(case.get("packets", ()) or ())
    ]
    rows = [row for row in rows if isinstance(row, Mapping)]
    return {
        "packet_count": len(rows),
        "mean_seed_hit_count": (
            mean(int(row.get("seed_hit_count", 0) or 0) for row in rows)
            if rows
            else 0.0
        ),
        "mean_context_hit_count": (
            mean(int(row.get("context_hit_count", 0) or 0) for row in rows)
            if rows
            else 0.0
        ),
        "mean_cross_caption_context_count": (
            mean(
                int(row.get("cross_caption_context_count", 0) or 0)
                for row in rows
            )
            if rows
            else 0.0
        ),
        "total_context_hit_count": sum(
            int(row.get("context_hit_count", 0) or 0) for row in rows
        ),
        "total_cross_caption_context_count": sum(
            int(row.get("cross_caption_context_count", 0) or 0) for row in rows
        ),
    }


def _interval(value: Any) -> tuple[float, float] | None:
    try:
        values = tuple(value or ())
        if len(values) != 2:
            return None
        start, end = sorted((float(values[0]), float(values[1])))
        return start, end
    except (TypeError, ValueError):
        return None


def _overlap(left: tuple[float, float], right_value: Any) -> float:
    right = _interval(right_value)
    if right is None:
        return 0.0
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None
