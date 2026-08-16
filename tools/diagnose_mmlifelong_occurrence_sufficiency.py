#!/usr/bin/env python3
"""Diagnose WP8 sufficiency decisions from frozen runtime artifacts only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


BLOCKER_CATEGORIES = (
    "explicit_unsupported",
    "declared_unknown",
    "implicit_unknown",
    "partial_support",
    "mixed",
)
SUPPORT_CATEGORIES = (
    "supported",
    "partial_support",
    "declared_unknown",
    "implicit_unknown",
    "contradicted",
)

# Frozen before inspecting WP8 outcomes. These thresholds require a large effect,
# consistency across constraint types, and a stable positive case-cluster bootstrap.
STRONG_SUPPORT_GAP = 0.15
STRONG_POSITIVE_TYPE_FRACTION = 2 / 3
STRONG_BOOTSTRAP_POSITIVE_PROBABILITY = 0.90
TOP_K_RETENTION_THRESHOLD = 0.95


def load_diagnostic_cases(
    run_root: Path, *, evaluation_record_root: Path
) -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for prediction_path in sorted(Path(run_root).glob("cases/*/prediction.json")):
        run_dir = prediction_path.parent
        prediction = _read_json(prediction_path)
        case_id = str(prediction.get("case_id", run_dir.name) or run_dir.name)
        runtime = _read_json(run_dir / "runtime_summary.json")
        state = _read_json(run_dir / "occurrence_resolution_state.json")
        observations = _read_jsonl(run_dir / "observation_log.jsonl")
        record = _read_json(
            Path(evaluation_record_root) / case_id / "evaluation_case.json"
        )
        clues = tuple(
            (float(value[0]), float(value[1]))
            for value in tuple(record.get("clue_intervals", ()) or ())
            if _is_interval(value)
        )
        trace = tuple(
            dict(row)
            for row in tuple(runtime.get("trace", ()) or ())
            if isinstance(row, Mapping)
        )
        candidate_sets = _candidate_sets(observations, state)
        events = _expanded_sufficiency_events(trace, candidate_sets)
        cases.append(
            {
                "case_id": case_id,
                "events": events,
                "candidate_sets": candidate_sets,
                "clues": clues,
            }
        )
    return tuple(cases)


def classify_candidate_blockers(
    event: Mapping[str, Any],
) -> dict[str, str]:
    blockers: dict[str, set[str]] = defaultdict(set)
    constraints = tuple(event.get("constraints_checked", ()) or ())
    candidate_ids = list(
        dict.fromkeys(
            str(row.get("occurrence_id", "") or "").strip()
            for constraint in constraints
            if isinstance(constraint, Mapping)
            for row in tuple(constraint.get("support", ()) or ())
            if isinstance(row, Mapping)
            and str(row.get("occurrence_id", "") or "").strip()
        )
    )
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            continue
        implicit_ids = {
            str(value)
            for value in tuple(
                constraint.get("implicit_unknown_occurrence_ids", ()) or ()
            )
            if str(value)
        }
        seen: set[str] = set()
        for row in tuple(constraint.get("support", ()) or ()):
            if not isinstance(row, Mapping):
                continue
            occurrence_id = str(row.get("occurrence_id", "") or "").strip()
            if not occurrence_id:
                continue
            seen.add(occurrence_id)
            category = _support_category(
                str(row.get("status", "") or ""),
                implicit=occurrence_id in implicit_ids,
            )
            if category != "supported":
                blockers[occurrence_id].add(_blocker_category(category))
        for occurrence_id in candidate_ids:
            if occurrence_id not in seen:
                blockers[occurrence_id].add("implicit_unknown")

    classified: dict[str, str] = {}
    for occurrence_id in candidate_ids:
        categories = blockers.get(occurrence_id, set())
        if not categories:
            classified[occurrence_id] = "unblocked"
        elif len(categories) == 1:
            classified[occurrence_id] = next(iter(categories))
        else:
            classified[occurrence_id] = "mixed"
    return classified


def build_blocker_diagnosis(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    decision_rows: list[dict[str, Any]] = []
    total_events = 0
    sufficient_events = 0
    for case in cases:
        for event_index, event in enumerate(tuple(case.get("events", ()) or ())):
            total_events += 1
            if str(event.get("verdict", "") or "") != "insufficient":
                sufficient_events += 1
                continue
            classified = classify_candidate_blockers(event)
            counts = Counter(classified.values())
            candidate_counts.update(
                value for value in classified.values() if value != "unblocked"
            )
            distinct = {value for value in classified.values() if value != "unblocked"}
            decision_class = next(iter(distinct)) if len(distinct) == 1 else "mixed"
            decision_counts[decision_class] += 1
            decision_rows.append(
                {
                    "case_id": str(case.get("case_id", "") or ""),
                    "set_id": str(event.get("set_id", "") or ""),
                    "round": int(event.get("round", 0) or 0),
                    "event_index": event_index,
                    "decision_class": decision_class,
                    "candidate_class_counts": {
                        category: counts.get(category, 0)
                        for category in (*BLOCKER_CATEGORIES, "unblocked")
                        if counts.get(category, 0)
                    },
                }
            )

    blocked_candidates = sum(candidate_counts.values())
    insufficient_events = sum(decision_counts.values())
    return {
        "event_count": total_events,
        "sufficient_event_count": sufficient_events,
        "insufficient_event_count": insufficient_events,
        "blocked_candidate_count": blocked_candidates,
        "candidate_class_counts": {
            category: candidate_counts.get(category, 0)
            for category in BLOCKER_CATEGORIES
        },
        "candidate_implicit_unknown_primary_rate": _ratio(
            candidate_counts.get("implicit_unknown", 0), blocked_candidates
        ),
        "decision_class_counts": {
            category: decision_counts.get(category, 0)
            for category in BLOCKER_CATEGORIES
        },
        "decision_implicit_unknown_primary_rate": _ratio(
            decision_counts.get("implicit_unknown", 0), insufficient_events
        ),
        "decision_rows": decision_rows,
    }


def build_support_discrimination(
    cases: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_type: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"gold": Counter(), "non_gold": Counter()}
    )
    overall = {"gold": Counter(), "non_gold": Counter()}
    case_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "gold_supported": 0,
            "gold_total": 0,
            "non_gold_supported": 0,
            "non_gold_total": 0,
        }
    )
    candidate_present_event_count = 0
    missing_set_metadata_count = 0

    for case in cases:
        case_id = str(case.get("case_id", "") or "")
        candidate_sets = dict(case.get("candidate_sets", {}) or {})
        clues = tuple(case.get("clues", ()) or ())
        for event in tuple(case.get("events", ()) or ()):
            set_id = str(event.get("set_id", "") or "")
            candidates = tuple(candidate_sets.get(set_id, ()) or ())
            gold_ids = {
                str(candidate.get("occurrence_id", "") or "")
                for candidate in candidates
                if _candidate_is_gold(candidate, clues)
            }
            if not candidates:
                missing_set_metadata_count += 1
                continue
            if not gold_ids:
                continue
            candidate_present_event_count += 1
            for constraint in tuple(event.get("constraints_checked", ()) or ()):
                if not isinstance(constraint, Mapping):
                    continue
                constraint_type = str(
                    constraint.get("constraint_type", "unknown") or "unknown"
                ).casefold()
                implicit_ids = {
                    str(value)
                    for value in tuple(
                        constraint.get("implicit_unknown_occurrence_ids", ()) or ()
                    )
                    if str(value)
                }
                for row in tuple(constraint.get("support", ()) or ()):
                    if not isinstance(row, Mapping):
                        continue
                    occurrence_id = str(row.get("occurrence_id", "") or "")
                    if not occurrence_id:
                        continue
                    group = "gold" if occurrence_id in gold_ids else "non_gold"
                    category = _support_category(
                        str(row.get("status", "") or ""),
                        implicit=occurrence_id in implicit_ids,
                    )
                    by_type[constraint_type][group][category] += 1
                    overall[group][category] += 1
                    prefix = "gold" if group == "gold" else "non_gold"
                    case_counts[case_id][f"{prefix}_total"] += 1
                    if category == "supported":
                        case_counts[case_id][f"{prefix}_supported"] += 1

    type_rows: dict[str, Any] = {}
    eligible_gaps: list[float] = []
    for constraint_type in sorted(by_type):
        gold = _support_summary(by_type[constraint_type]["gold"])
        non_gold = _support_summary(by_type[constraint_type]["non_gold"])
        gap = _difference(gold["supported_rate"], non_gold["supported_rate"])
        if gap is not None:
            eligible_gaps.append(gap)
        type_rows[constraint_type] = {
            "gold": gold,
            "non_gold": non_gold,
            "support_gap": gap,
        }

    overall_gold = _support_summary(overall["gold"])
    overall_non_gold = _support_summary(overall["non_gold"])
    overall_gap = _difference(
        overall_gold["supported_rate"], overall_non_gold["supported_rate"]
    )
    bootstrap = _case_cluster_bootstrap_gap(
        case_counts,
        samples=bootstrap_samples,
        seed=seed,
    )
    positive_type_fraction = _ratio(
        sum(gap > 0 for gap in eligible_gaps), len(eligible_gaps)
    )
    strong = bool(
        overall_gap is not None
        and overall_gap >= STRONG_SUPPORT_GAP
        and positive_type_fraction is not None
        and positive_type_fraction >= STRONG_POSITIVE_TYPE_FRACTION
        and bootstrap["positive_probability"] is not None
        and bootstrap["positive_probability"] >= STRONG_BOOTSTRAP_POSITIVE_PROBABILITY
    )
    return {
        "candidate_present_event_count": candidate_present_event_count,
        "missing_set_metadata_event_count": missing_set_metadata_count,
        "overall": {
            "gold": overall_gold,
            "non_gold": overall_non_gold,
            "support_gap": overall_gap,
        },
        "by_constraint_type": type_rows,
        "positive_constraint_type_fraction": positive_type_fraction,
        "case_cluster_bootstrap": bootstrap,
        "decision_policy": {
            "minimum_support_gap": STRONG_SUPPORT_GAP,
            "minimum_positive_constraint_type_fraction": (
                STRONG_POSITIVE_TYPE_FRACTION
            ),
            "minimum_bootstrap_positive_probability": (
                STRONG_BOOTSTRAP_POSITIVE_PROBABILITY
            ),
        },
        "strong_gold_non_gold_discrimination": strong,
    }


def build_gold_at_k(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique_sets: dict[tuple[str, str], tuple[Mapping[str, Any], ...]] = {}
    clues_by_case: dict[str, tuple[tuple[float, float], ...]] = {}
    for case in cases:
        case_id = str(case.get("case_id", "") or "")
        clues_by_case[case_id] = tuple(case.get("clues", ()) or ())
        candidate_sets = dict(case.get("candidate_sets", {}) or {})
        for event in tuple(case.get("events", ()) or ()):
            set_id = str(event.get("set_id", "") or "")
            if set_id and (case_id, set_id) not in unique_sets:
                unique_sets[(case_id, set_id)] = tuple(
                    candidate_sets.get(set_id, ()) or ()
                )

    counts: list[int] = []
    full_gold = 0
    hits = {1: 0, 3: 0, 5: 0}
    missing_metadata = 0
    rank_source_counts: Counter[str] = Counter()
    for (case_id, _), candidates in unique_sets.items():
        if not candidates:
            missing_metadata += 1
            continue
        ranked, rank_source = _ranked_candidates(candidates)
        rank_source_counts[rank_source] += 1
        counts.append(len(ranked))
        clues = clues_by_case.get(case_id, ())
        gold_positions = [
            index
            for index, candidate in enumerate(ranked, start=1)
            if _candidate_is_gold(candidate, clues)
        ]
        if gold_positions:
            full_gold += 1
            first_gold = min(gold_positions)
            for k in hits:
                hits[k] += first_gold <= k

    analyzed_sets = len(counts)
    gold_at_k = {
        str(k): {
            "hit_count": hits[k],
            "unconditional_rate": _ratio(hits[k], analyzed_sets),
            "conditional_retention_rate": _ratio(hits[k], full_gold),
        }
        for k in sorted(hits)
    }
    safe_values = [
        k
        for k in sorted(hits)
        if _at_least(
            gold_at_k[str(k)]["conditional_retention_rate"],
            TOP_K_RETENTION_THRESHOLD,
        )
    ]
    return {
        "unit": "unique_case_set",
        "set_count": len(unique_sets),
        "analyzed_set_count": analyzed_sets,
        "missing_candidate_metadata_set_count": missing_metadata,
        "full_set_gold_count": full_gold,
        "full_set_gold_rate": _ratio(full_gold, analyzed_sets),
        "gold_at_k": gold_at_k,
        "candidate_count_distribution": _distribution(counts),
        "rank_source_counts": dict(sorted(rank_source_counts.items())),
        "top_k_retention_threshold": TOP_K_RETENTION_THRESHOLD,
        "recommended_top_k": min(safe_values) if safe_values else None,
    }


def build_selection_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row.get("arm", "none") or "none")].append(row)
    result: dict[str, Any] = {}
    for arm in sorted(by_arm):
        metrics = _selection_metrics(by_arm[arm])
        result[arm] = {
            "observed": metrics,
            "always_abstain": _always_abstain_metrics(
                candidate_present=metrics["candidate_present_count"],
                candidate_absent=metrics["candidate_absent_count"],
            ),
        }
    return result


def build_diagnosis(
    cases: Sequence[Mapping[str, Any]],
    *,
    selection_rows: Sequence[Mapping[str, Any]] = (),
    expected_cases: int | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 20260816,
) -> dict[str, Any]:
    if expected_cases is not None and len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(cases)}")
    blockers = build_blocker_diagnosis(cases)
    discrimination = build_support_discrimination(
        cases,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    gold_at_k = build_gold_at_k(cases)
    support_positive = bool(discrimination["strong_gold_non_gold_discrimination"])
    recommendation = (
        "PROCEED_A4_1_SUPPORT_SURFACE_CALIBRATION"
        if support_positive
        else "STOP_THRESHOLD_CALIBRATION_REDESIGN_EVIDENCE"
    )
    return {
        "schema_version": "OccurrenceSufficiencyOfflineDiagnosisV1",
        "case_count": len(cases),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "d1_blocker_decomposition": blockers,
        "d2_support_discrimination": discrimination,
        "d3_gold_at_k": gold_at_k,
        "selection_diagnostics": build_selection_diagnostics(selection_rows),
        "recommendation": {
            "decision": recommendation,
            "support_surface_calibration_authorized": support_positive,
            "top_k_authorized": gold_at_k["recommended_top_k"] is not None,
            "recommended_top_k": gold_at_k["recommended_top_k"],
            "runtime_or_model_calls_used": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    d1 = report["d1_blocker_decomposition"]
    d2 = report["d2_support_discrimination"]
    d3 = report["d3_gold_at_k"]
    recommendation = report["recommendation"]
    lines = [
        "# WP8 Sufficiency Offline Diagnosis",
        "",
        f"Cases: {report['case_count']}; sufficiency events: {d1['event_count']}.",
        "No runtime or model calls were used.",
        "",
        "## Decision",
        "",
        f"**{recommendation['decision']}**",
        "",
        (
            "Top-K recommendation: "
            + (
                f"K={recommendation['recommended_top_k']}"
                if recommendation["recommended_top_k"] is not None
                else "none"
            )
            + "."
        ),
        "",
        "## D1: Insufficient blockers",
        "",
        "| Category | Blocked candidates | Decisions |",
        "|---|---:|---:|",
    ]
    for category in BLOCKER_CATEGORIES:
        lines.append(
            f"| {category} | {d1['candidate_class_counts'][category]} | "
            f"{d1['decision_class_counts'][category]} |"
        )
    lines.extend(
        [
            "",
            (
                "Implicit-unknown-only rate: "
                f"{_fmt(d1['candidate_implicit_unknown_primary_rate'])} by candidate; "
                f"{_fmt(d1['decision_implicit_unknown_primary_rate'])} by decision."
            ),
            "",
            "## D2: Gold vs non-gold support",
            "",
            (
                f"Overall support gap: {_fmt(d2['overall']['support_gap'])}; "
                "case-cluster bootstrap 95% CI "
                f"[{_fmt(d2['case_cluster_bootstrap']['ci95'][0])}, "
                f"{_fmt(d2['case_cluster_bootstrap']['ci95'][1])}]; "
                "positive probability "
                f"{_fmt(d2['case_cluster_bootstrap']['positive_probability'])}."
            ),
            "",
            "| Constraint | Gold supported | Non-gold supported | Gap | Gold explicit | Non-gold explicit |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for constraint_type, row in d2["by_constraint_type"].items():
        lines.append(
            f"| {constraint_type} | {_fmt(row['gold']['supported_rate'])} | "
            f"{_fmt(row['non_gold']['supported_rate'])} | "
            f"{_fmt(row['support_gap'])} | "
            f"{_fmt(row['gold']['explicit_coverage'])} | "
            f"{_fmt(row['non_gold']['explicit_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## D3: Gold@K",
            "",
            "| K | Unconditional Gold@K | Conditional retention |",
            "|---:|---:|---:|",
        ]
    )
    for k, row in d3["gold_at_k"].items():
        lines.append(
            f"| {k} | {_fmt(row['unconditional_rate'])} | "
            f"{_fmt(row['conditional_retention_rate'])} |"
        )
    distribution = d3["candidate_count_distribution"]
    lines.extend(
        [
            "",
            (
                "Candidate counts: "
                f"min={distribution['min']}, median={distribution['median']}, "
                f"p75={distribution['p75']}, max={distribution['max']}, "
                f"mean={_fmt(distribution['mean'])}."
            ),
            "",
            "## Selection diagnostics",
            "",
            "| Arm | Policy | Precision | Recall | F1 | Balanced accuracy | False commit | No-match accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, policies in report["selection_diagnostics"].items():
        for policy_name in ("observed", "always_abstain"):
            metrics = policies[policy_name]
            lines.append(
                f"| {arm} | {policy_name} | {_fmt(metrics['precision'])} | "
                f"{_fmt(metrics['recall'])} | {_fmt(metrics['f1'])} | "
                f"{_fmt(metrics['balanced_accuracy'])} | "
                f"{_fmt(metrics['false_commit_rate'])} | "
                f"{_fmt(metrics['no_match_accuracy'])} |"
            )
    lines.extend(
        [
            "",
            "The 39-case cohort is underpowered. Endpoint values were not validity gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def _expanded_sufficiency_events(
    trace: Sequence[Mapping[str, Any]],
    candidate_sets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    compact_events = {
        (
            int(row.get("round", 0) or 0),
            int(row.get("occurrence_op_index", 0) or 0),
        ): row
        for row in trace
        if row.get("type") == "occurrence_sufficiency_decision"
    }
    states = {
        set_id: {
            str(candidate.get("occurrence_id", "") or ""): "active"
            for candidate in candidates
            if str(candidate.get("occurrence_id", "") or "")
        }
        for set_id, candidates in candidate_sets.items()
    }
    expanded: list[dict[str, Any]] = []
    for decision in trace:
        if (
            decision.get("type") != "reasoner_decision"
            or decision.get("occurrence_ops_accepted") is False
        ):
            continue
        round_id = int(decision.get("round", 0) or 0)
        for operation_index, operation in enumerate(
            tuple(decision.get("occurrence_ops", ()) or ())
        ):
            if not isinstance(operation, Mapping):
                continue
            op = str(operation.get("op", operation.get("type", "")) or "").casefold()
            set_id = str(
                operation.get("set_id", operation.get("locator_attempt_id", "")) or ""
            )
            set_states = states.get(set_id, {})
            if op == "assess_sufficiency":
                compact = compact_events.get((round_id, operation_index))
                if compact is None:
                    raise ValueError(
                        "missing compact sufficiency event for "
                        f"round={round_id}, op={operation_index}"
                    )
                if not set_states:
                    raise ValueError(
                        f"missing candidate metadata for sufficiency set {set_id}"
                    )
                viable_ids = tuple(
                    occurrence_id
                    for occurrence_id, status in set_states.items()
                    if status != "eliminated"
                )
                recorded_scope = tuple(
                    str(value)
                    for value in tuple(compact.get("scope_occurrence_ids", ()) or ())
                    if str(value)
                )
                scope_ids = recorded_scope or viable_ids
                if any(occurrence_id not in viable_ids for occurrence_id in scope_ids):
                    raise ValueError(
                        f"sufficiency scope contains non-viable ID for set {set_id}"
                    )
                normalized_constraints = _normalize_raw_constraints(
                    operation.get("constraints_checked", ()), viable_ids=scope_ids
                )
                sufficient_ids = tuple(
                    occurrence_id
                    for occurrence_id in scope_ids
                    if normalized_constraints
                    and all(
                        next(
                            str(row.get("status", "") or "")
                            for row in tuple(constraint.get("support", ()) or ())
                            if str(row.get("occurrence_id", "") or "") == occurrence_id
                        )
                        == "supported"
                        for constraint in normalized_constraints
                    )
                )
                implicit_count = sum(
                    len(
                        tuple(
                            constraint.get("implicit_unknown_occurrence_ids", ()) or ()
                        )
                    )
                    for constraint in normalized_constraints
                )
                _validate_reconstructed_event(
                    compact,
                    normalized_constraints=normalized_constraints,
                    sufficient_ids=sufficient_ids,
                    implicit_count=implicit_count,
                    scope_ids=scope_ids,
                    out_of_scope_ids=tuple(
                        occurrence_id
                        for occurrence_id in viable_ids
                        if occurrence_id not in scope_ids
                    ),
                )
                expanded.append(
                    {
                        **compact,
                        "constraints_checked": normalized_constraints,
                        "reconstructed_from_accepted_operation": True,
                    }
                )
                continue
            occurrence_id = str(operation.get("occurrence_id", "") or "")
            if occurrence_id not in set_states:
                continue
            if op == "eliminate":
                set_states[occurrence_id] = "eliminated"
            elif op in {"keep", "reopen"}:
                set_states[occurrence_id] = "active"

    if len(expanded) != len(compact_events):
        raise ValueError(
            "sufficiency event reconstruction count mismatch: "
            f"expanded={len(expanded)}, compact={len(compact_events)}"
        )
    return tuple(expanded)


def _normalize_raw_constraints(
    raw_constraints: Any, *, viable_ids: Sequence[str]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_constraint in tuple(raw_constraints or ()):
        if not isinstance(raw_constraint, Mapping):
            continue
        support: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_row in tuple(raw_constraint.get("support", ()) or ()):
            if not isinstance(raw_row, Mapping):
                continue
            occurrence_id = str(raw_row.get("occurrence_id", "") or "")
            if occurrence_id not in viable_ids or occurrence_id in seen:
                continue
            seen.add(occurrence_id)
            support.append(dict(raw_row))
        missing = [
            occurrence_id for occurrence_id in viable_ids if occurrence_id not in seen
        ]
        support.extend(
            {
                "occurrence_id": occurrence_id,
                "status": "unknown",
                "evidence_passage_ids": [],
            }
            for occurrence_id in missing
        )
        normalized.append(
            {
                **raw_constraint,
                "support": support,
                "implicit_unknown_occurrence_ids": missing,
            }
        )
    return normalized


def _validate_reconstructed_event(
    compact: Mapping[str, Any],
    *,
    normalized_constraints: Sequence[Mapping[str, Any]],
    sufficient_ids: Sequence[str],
    implicit_count: int,
    scope_ids: Sequence[str],
    out_of_scope_ids: Sequence[str],
) -> None:
    expected_constraint_ids = [
        str(row.get("constraint_id", "") or "") for row in normalized_constraints
    ]
    expected_constraint_types = [
        str(row.get("constraint_type", "") or "") for row in normalized_constraints
    ]
    checks = {
        "constraint_ids": (
            list(compact.get("constraints_checked", ()) or ()),
            expected_constraint_ids,
        ),
        "constraint_types": (
            list(compact.get("constraint_types", ()) or ()),
            expected_constraint_types,
        ),
        "sufficient_occurrence_ids": (
            list(compact.get("sufficient_occurrence_ids", ()) or ()),
            list(sufficient_ids),
        ),
        "implicit_unknown_support_count": (
            int(compact.get("implicit_unknown_support_count", 0) or 0),
            implicit_count,
        ),
    }
    if "scope_occurrence_ids" in compact:
        checks["scope_occurrence_ids"] = (
            list(compact.get("scope_occurrence_ids", ()) or ()),
            list(scope_ids),
        )
        checks["out_of_scope_occurrence_ids"] = (
            list(compact.get("out_of_scope_occurrence_ids", ()) or ()),
            list(out_of_scope_ids),
        )
        checks["support_complete"] = (
            bool(compact.get("support_complete")),
            True,
        )
    mismatches = [
        name for name, (actual, expected) in checks.items() if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "sufficiency event reconstruction mismatch: " + ",".join(mismatches)
        )


def _candidate_sets(
    observations: Sequence[Mapping[str, Any]], state: Mapping[str, Any]
) -> dict[str, tuple[dict[str, Any], ...]]:
    sets: dict[str, list[dict[str, Any]]] = {}
    for observation_index, observation in enumerate(observations):
        config = observation.get("sampling_config", {})
        occurrence_set = (
            config.get("occurrence_set") if isinstance(config, Mapping) else None
        )
        if not isinstance(occurrence_set, Mapping):
            continue
        set_id = str(
            observation.get(
                "attempt_id",
                occurrence_set.get("attempt_id", occurrence_set.get("set_id", "")),
            )
            or f"observation_{observation_index:04d}"
        )
        _merge_candidates(sets, set_id, occurrence_set.get("candidates", ()))
    for raw_set in tuple(state.get("sets", ()) or ()):
        if not isinstance(raw_set, Mapping):
            continue
        set_id = str(raw_set.get("set_id", raw_set.get("locator_attempt_id", "")) or "")
        if set_id:
            _merge_candidates(sets, set_id, raw_set.get("candidates", ()))
    return {set_id: tuple(candidates) for set_id, candidates in sets.items()}


def _merge_candidates(
    sets: dict[str, list[dict[str, Any]]], set_id: str, raw_candidates: Any
) -> None:
    destination = sets.setdefault(set_id, [])
    positions = {
        str(candidate.get("occurrence_id", "") or ""): index
        for index, candidate in enumerate(destination)
    }
    for raw_index, raw_candidate in enumerate(tuple(raw_candidates or ())):
        if not isinstance(raw_candidate, Mapping):
            continue
        occurrence_id = str(raw_candidate.get("occurrence_id", "") or "").strip()
        if not occurrence_id:
            continue
        normalized = dict(raw_candidate)
        normalized["occurrence_id"] = occurrence_id
        normalized.setdefault("source_order", raw_index + 1)
        if occurrence_id not in positions:
            positions[occurrence_id] = len(destination)
            destination.append(normalized)
            continue
        current = destination[positions[occurrence_id]]
        for key, value in normalized.items():
            if key not in current or current[key] in (None, "", [], ()):
                current[key] = value


def _ranked_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    ranks: list[float] = []
    for candidate in candidates:
        try:
            ranks.append(float(candidate.get("rank")))
        except (TypeError, ValueError):
            ranks = []
            break
    if len(ranks) == len(candidates) and len(set(ranks)) == len(ranks):
        return (
            tuple(
                candidate
                for _, candidate in sorted(
                    zip(ranks, candidates), key=lambda item: item[0]
                )
            ),
            "explicit_rank",
        )
    return tuple(candidates), "source_order"


def _candidate_is_gold(
    candidate: Mapping[str, Any], clues: Sequence[tuple[float, float]]
) -> bool:
    raw_range = candidate.get("time_range", ())
    if not _is_interval(raw_range):
        return False
    interval = (float(raw_range[0]), float(raw_range[1]))
    return any(_overlap(interval, clue) > 0 for clue in clues)


def _support_category(status: str, *, implicit: bool) -> str:
    normalized = str(status or "").strip().casefold()
    if normalized == "supported":
        return "supported"
    if normalized == "partial":
        return "partial_support"
    if normalized == "contradicted":
        return "contradicted"
    if normalized == "unknown":
        return "implicit_unknown" if implicit else "declared_unknown"
    return "declared_unknown"


def _blocker_category(support_category: str) -> str:
    return (
        "explicit_unsupported"
        if support_category == "contradicted"
        else support_category
    )


def _support_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    total = sum(int(counts.get(category, 0)) for category in SUPPORT_CATEGORIES)
    status_counts = {
        category: int(counts.get(category, 0)) for category in SUPPORT_CATEGORIES
    }
    return {
        "total": total,
        "counts": status_counts,
        "rates": {
            category: _ratio(count, total) for category, count in status_counts.items()
        },
        "supported_rate": _ratio(status_counts["supported"], total),
        "explicit_coverage": _ratio(total - status_counts["implicit_unknown"], total),
    }


def _case_cluster_bootstrap_gap(
    case_counts: Mapping[str, Mapping[str, int]], *, samples: int, seed: int
) -> dict[str, Any]:
    case_ids = sorted(case_counts)
    if not case_ids or samples <= 0:
        return {
            "samples_requested": samples,
            "samples_valid": 0,
            "ci95": [None, None],
            "positive_probability": None,
        }
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        gold_supported = sum(
            int(case_counts[case_id]["gold_supported"]) for case_id in sampled
        )
        gold_total = sum(int(case_counts[case_id]["gold_total"]) for case_id in sampled)
        non_supported = sum(
            int(case_counts[case_id]["non_gold_supported"]) for case_id in sampled
        )
        non_total = sum(
            int(case_counts[case_id]["non_gold_total"]) for case_id in sampled
        )
        if not gold_total or not non_total:
            continue
        gaps.append(gold_supported / gold_total - non_supported / non_total)
    gaps.sort()
    return {
        "samples_requested": samples,
        "samples_valid": len(gaps),
        "ci95": [
            _quantile(gaps, 0.025) if gaps else None,
            _quantile(gaps, 0.975) if gaps else None,
        ],
        "positive_probability": (
            sum(gap > 0 for gap in gaps) / len(gaps) if gaps else None
        ),
    }


def _selection_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_present = sum(
        bool(row.get("candidate_recall_resolved_set")) for row in rows
    )
    candidate_absent = len(rows) - candidate_present
    selected = sum(
        str(row.get("final_resolution", "") or "") == "selected" for row in rows
    )
    tp = sum(
        bool(row.get("candidate_recall_resolved_set")) and row.get("osa_strict") is True
        for row in rows
    )
    fp = selected - tp
    fn = candidate_present - tp
    tn = sum(
        not bool(row.get("candidate_recall_resolved_set"))
        and str(row.get("final_resolution", "") or "") == "no_match"
        for row in rows
    )
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, candidate_absent)
    return {
        "case_count": len(rows),
        "candidate_present_count": candidate_present,
        "candidate_absent_count": candidate_absent,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "specificity": specificity,
        "balanced_accuracy": _mean_optional((recall, specificity)),
        "false_commit_rate": _ratio(
            sum(
                not bool(row.get("candidate_recall_resolved_set"))
                and str(row.get("final_resolution", "") or "") == "selected"
                for row in rows
            ),
            candidate_absent,
        ),
        "no_match_accuracy": specificity,
        "false_abstention_rate": _ratio(
            sum(
                bool(row.get("candidate_recall_resolved_set"))
                and str(row.get("final_resolution", "") or "") == "no_match"
                for row in rows
            ),
            candidate_present,
        ),
    }


def _always_abstain_metrics(
    *, candidate_present: int, candidate_absent: int
) -> dict[str, Any]:
    recall = 0.0 if candidate_present else None
    specificity = 1.0 if candidate_absent else None
    return {
        "case_count": candidate_present + candidate_absent,
        "candidate_present_count": candidate_present,
        "candidate_absent_count": candidate_absent,
        "tp": 0,
        "fp": 0,
        "fn": candidate_present,
        "tn": candidate_absent,
        "precision": None,
        "recall": recall,
        "f1": 0.0 if candidate_present else None,
        "specificity": specificity,
        "balanced_accuracy": _mean_optional((recall, specificity)),
        "false_commit_rate": 0.0 if candidate_absent else None,
        "no_match_accuracy": specificity,
        "false_abstention_rate": 1.0 if candidate_present else None,
    }


def _load_selection_rows(
    run_roots: Sequence[Path], *, evaluation_record_root: Path
) -> tuple[dict[str, Any], ...]:
    module_path = Path(__file__).with_name("analyze_mmlifelong_occurrence_agent.py")
    spec = importlib.util.spec_from_file_location(
        "occurrence_analysis_for_sufficiency_diagnosis", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load occurrence analyzer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(
        module.collect_rows(
            tuple(Path(root) for root in run_roots),
            evaluation_record_root=Path(evaluation_record_root),
        )
    )


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower]) * (1 - fraction) + float(values[upper]) * fraction


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _is_interval(value: Any) -> bool:
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    )


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _mean_optional(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if recall == 0 and precision is None:
        return 0.0
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not Path(path).is_file():
        return ()
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--control-run-root")
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    evaluation_root = Path(args.evaluation_record_root)
    cases = load_diagnostic_cases(run_root, evaluation_record_root=evaluation_root)
    selection_roots = [run_root]
    if args.control_run_root:
        selection_roots.insert(0, Path(args.control_run_root))
    selection_rows = _load_selection_rows(
        selection_roots, evaluation_record_root=evaluation_root
    )
    report = build_diagnosis(
        cases,
        selection_rows=selection_rows,
        expected_cases=args.expected_cases,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "event_count": report["d1_blocker_decomposition"]["event_count"],
                "decision": report["recommendation"]["decision"],
                "recommended_top_k": report["recommendation"]["recommended_top_k"],
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
