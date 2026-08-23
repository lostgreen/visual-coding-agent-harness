from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence


ANCHOR_EVIDENCE_REPORT_CONTRACT = "WP16-3-anchor-evidence-report-v1"
RECALL_KS = (1, 3, 5)


def build_anchor_evidence_report(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    variant_order: Sequence[str],
) -> dict[str, Any]:
    variants = tuple(str(value) for value in variant_order if str(value))
    errors = _structural_errors(
        cases,
        expected_cases=expected_cases,
        variants=variants,
    )
    summaries = {
        variant: _variant_summary(cases, variant=variant)
        for variant in ("baseline", *variants)
    }
    selected = max(
        variants,
        key=lambda variant: (
            summaries[variant]["bound_evidence_at_5"]["count"],
            summaries[variant]["channel_evidence_at_5"]["count"],
            -summaries[variant]["mean_context_hit_count"],
        ),
        default=None,
    )
    baseline = summaries["baseline"]
    treatment = summaries.get(selected, baseline)
    signal = bool(
        selected
        and (
            treatment["bound_evidence_at_5"]["count"]
            > baseline["bound_evidence_at_5"]["count"]
            or treatment["channel_evidence_at_5"]["count"]
            > baseline["channel_evidence_at_5"]["count"]
        )
    )
    return {
        "schema_version": ANCHOR_EVIDENCE_REPORT_CONTRACT,
        "decision": (
            "DIAGNOSTIC_MECHANISM_SIGNAL"
            if not errors and signal
            else "DIAGNOSTIC_NO_SIGNAL"
            if not errors
            else "INVALID"
        ),
        "structural_gate_passed": not errors,
        "structural_errors": errors,
        "case_count": len(cases),
        "eligible_case_count": sum(
            bool(case.get("request", {}).get("eligible", False))
            for case in cases
        ),
        "anchor_labeled_case_count": sum(
            bool(case.get("anchor_intervals")) for case in cases
        ),
        "selected_variant": selected,
        "variants": summaries,
        "scope_diagnostics": _scope_diagnostics(cases),
        "per_case": [_case_summary(case, variants=variants) for case in cases],
        "claims": {
            "anchor_recall_requires_independent_anchor_labels": True,
            "unlabeled_cases_report_directional_path_only": True,
            "official_intervals_enter_evaluation_only": True,
            "oracle_localized_ocr_is_formal_retrieval_improvement": False,
            "endpoint_values_are_structural_gates": False,
        },
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
    case_ids = [str(case.get("case_id", "") or "") for case in cases]
    if any(not value for value in case_ids) or len(case_ids) != len(set(case_ids)):
        errors.append("case_ids_not_unique_and_complete")
    for case in cases:
        case_id = str(case.get("case_id", "") or "")
        request = case.get("request", {})
        if not isinstance(request, Mapping):
            errors.append(f"{case_id}:request_invalid")
            continue
        direction = str(request.get("direction", "") or "")
        for packet_index, packet in enumerate(tuple(case.get("packets", ()) or ())):
            rows = packet.get("variants", {})
            if not isinstance(rows, Mapping):
                errors.append(f"{case_id}:packet{packet_index}:variants_invalid")
                continue
            expected = {"baseline", *variants}
            if set(rows) != expected:
                errors.append(f"{case_id}:packet{packet_index}:variant_set_mismatch")
                continue
            seed_ids = tuple(rows["baseline"].get("seed_hit_ids", ()) or ())
            for variant in variants:
                row = rows[variant]
                if tuple(row.get("seed_hit_ids", ()) or ()) != seed_ids:
                    errors.append(f"{case_id}:packet{packet_index}:{variant}:seed_drift")
                for bundle in tuple(row.get("bundle_set", {}).get("bundles", ()) or ()):
                    for member in tuple(bundle.get("member_passages", ()) or ()):
                        if str(member.get("role", "") or "") != "context":
                            continue
                        links = tuple(member.get("context_links", ()) or ())
                        if not links or any(
                            link.get("same_source_timeline") is not True
                            for link in links
                            if isinstance(link, Mapping)
                        ):
                            errors.append(
                                f"{case_id}:packet{packet_index}:{variant}:source_unproven"
                            )
                        offsets = [
                            int(link.get("offset", 0) or 0)
                            for link in links
                            if isinstance(link, Mapping)
                        ]
                        if direction == "after" and any(value <= 0 for value in offsets):
                            errors.append(
                                f"{case_id}:packet{packet_index}:{variant}:direction_violation"
                            )
                        if direction == "before" and any(value >= 0 for value in offsets):
                            errors.append(
                                f"{case_id}:packet{packet_index}:{variant}:direction_violation"
                            )
    return sorted(set(errors))


def _variant_summary(
    cases: Sequence[Mapping[str, Any]], *, variant: str
) -> dict[str, Any]:
    eligible = tuple(
        case
        for case in cases
        if bool(case.get("request", {}).get("eligible", False))
    )
    anchor_labeled = tuple(case for case in eligible if case.get("anchor_intervals"))
    rows: dict[str, Any] = {
        "eligible_case_count": len(eligible),
        "anchor_labeled_case_count": len(anchor_labeled),
    }
    for k in RECALL_KS:
        rows[f"anchor_seed_at_{k}"] = _count_metric(
            anchor_labeled,
            lambda case: _case_metric(case, variant=variant, k=k, metric="anchor"),
        )
        rows[f"evidence_at_{k}"] = _count_metric(
            eligible,
            lambda case: _case_metric(case, variant=variant, k=k, metric="evidence"),
        )
        rows[f"channel_evidence_at_{k}"] = _count_metric(
            eligible,
            lambda case: _case_metric(case, variant=variant, k=k, metric="channel"),
        )
        rows[f"bound_evidence_at_{k}"] = _count_metric(
            anchor_labeled,
            lambda case: _case_metric(case, variant=variant, k=k, metric="bound"),
        )
    context_counts = [
        int(packet.get("variants", {}).get(variant, {}).get("context_hit_count", 0) or 0)
        for case in eligible
        for packet in tuple(case.get("packets", ()) or ())
    ]
    rows["mean_context_hit_count"] = mean(context_counts) if context_counts else 0.0
    return rows


def _case_metric(
    case: Mapping[str, Any], *, variant: str, k: int, metric: str
) -> bool:
    anchors = tuple(case.get("anchor_intervals", ()) or ())
    evidence = tuple(case.get("evidence_intervals", ()) or ())
    requested = set(case.get("request", {}).get("evidence_channels", ()) or ())
    for packet in tuple(case.get("packets", ()) or ()):
        row = packet.get("variants", {}).get(variant, {})
        bundles = tuple(row.get("bundle_set", {}).get("bundles", ()) or ())
        for bundle in bundles:
            if int(bundle.get("rank", 0) or 0) > int(k):
                continue
            members = tuple(bundle.get("member_passages", ()) or ())
            seed_anchor = any(
                str(member.get("role", "") or "") == "seed"
                and _overlaps_any(member.get("time_range"), anchors)
                for member in members
            )
            any_evidence = any(
                _overlaps_any(member.get("time_range"), evidence) for member in members
            )
            channel_evidence = any(
                str(member.get("role", "") or "") == "context"
                and _overlaps_any(member.get("time_range"), evidence)
                and _channel_matches(
                    requested,
                    set(member.get("evidence_channels_observed", ()) or ()),
                )
                for member in members
            )
            if metric == "anchor" and seed_anchor:
                return True
            if metric == "evidence" and any_evidence:
                return True
            if metric == "channel" and channel_evidence:
                return True
            if metric == "bound" and seed_anchor and channel_evidence:
                return True
    return False


def _channel_matches(requested: set[str], observed: set[str]) -> bool:
    normalized = {
        "visual_caption" if value == "caption" else value for value in observed
    } | observed
    return bool(requested & normalized) if requested else bool(observed)


def _scope_diagnostics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        packet.get("scope_diagnostic", {})
        for case in cases
        for packet in tuple(case.get("packets", ()) or ())
        if isinstance(packet.get("scope_diagnostic"), Mapping)
    ]
    return {
        "packet_count": len(rows),
        "scope_blocked_semantic_packet_count": sum(
            bool(row.get("scope_blocked_semantic_evidence", False)) for row in rows
        ),
        "weak_numeric_only_packet_count": sum(
            bool(row.get("global_overlap", {}).get("weak_numeric_only", False))
            for row in rows
        ),
        "scope_blocked_case_ids": sorted(
            {
                str(case.get("case_id", "") or "")
                for case in cases
                if any(
                    bool(
                        packet.get("scope_diagnostic", {}).get(
                            "scope_blocked_semantic_evidence", False
                        )
                    )
                    for packet in tuple(case.get("packets", ()) or ())
                )
            }
        ),
        "weak_numeric_only_case_ids": sorted(
            {
                str(case.get("case_id", "") or "")
                for case in cases
                if any(
                    bool(
                        packet.get("scope_diagnostic", {})
                        .get("global_overlap", {})
                        .get("weak_numeric_only", False)
                    )
                    for packet in tuple(case.get("packets", ()) or ())
                )
            }
        ),
    }


def _case_summary(case: Mapping[str, Any], *, variants: Sequence[str]) -> dict[str, Any]:
    return {
        "case_id": str(case.get("case_id", "") or ""),
        "request": dict(case.get("request", {}) or {}),
        "anchor_labeled": bool(case.get("anchor_intervals")),
        "metrics_at_5": {
            variant: {
                metric: _case_metric(case, variant=variant, k=5, metric=metric)
                for metric in ("anchor", "evidence", "channel", "bound")
            }
            for variant in ("baseline", *variants)
        },
    }


def _count_metric(
    cases: Sequence[Mapping[str, Any]], predicate: Any
) -> dict[str, Any]:
    count = sum(bool(predicate(case)) for case in cases)
    return {
        "count": count,
        "case_count": len(cases),
        "rate": float(count) / float(len(cases)) if cases else None,
    }


def _overlaps_any(value: Any, intervals: Sequence[Any]) -> bool:
    left = _interval(value)
    return bool(
        left
        and any(
            (right := _interval(interval)) is not None
            and min(left[1], right[1]) > max(left[0], right[0])
            for interval in intervals
        )
    )


def _interval(value: Any) -> tuple[float, float] | None:
    try:
        values = tuple(value or ())
        if len(values) != 2:
            return None
        start, end = sorted((float(values[0]), float(values[1])))
        return start, end
    except (TypeError, ValueError):
        return None
