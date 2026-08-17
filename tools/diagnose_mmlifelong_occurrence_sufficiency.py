#!/usr/bin/env python3
"""Diagnose WP8 sufficiency decisions from frozen runtime artifacts only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from math import ceil
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from vcah.occurrence_sufficiency import (
    ANSWER_TARGET_CONSTRAINT_TYPES,
    REFERENT_IDENTIFYING_CONSTRAINT_TYPES,
)


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
TARGET_MIN_RECALL = 0.70
TARGET_MAX_FALSE_COMMIT = 0.20

# Frozen from the clean WP10.1 control before winner-level signed analysis.
WP10_1_NEGATIVE_CASE_COUNT = 26
WP10_1_FALSE_COMMIT_COUNT = 12
WP10_1_POSITIVE_CASE_COUNT = 13
WP10_1_CORRECT_COMMIT_COUNT = 8
WP11_HARD_GUARD_MAX_FALSE_COMMIT_RATE = 0.30
WP11_HARD_GUARD_MIN_COMMIT_RECALL = 0.60


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
    labeled_rows, labeling = _labeled_support_rows(cases)
    for row in labeled_rows:
        case_id = str(row["case_id"])
        constraint_type = str(row["constraint_type"])
        group = str(row["group"])
        category = str(row["category"])
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
            "semantic_group": _constraint_semantic_group(constraint_type),
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
        "candidate_present_event_count": labeling["candidate_present_event_count"],
        "missing_set_metadata_event_count": labeling[
            "missing_set_metadata_event_count"
        ],
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


def build_signed_evidence_diagnostic(
    cases: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Measure whether explicit contradiction discriminates non-gold candidates."""
    labeled_rows, labeling = _labeled_support_rows(cases)
    overall = {"gold": Counter(), "non_gold": Counter()}
    by_type: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"gold": Counter(), "non_gold": Counter()}
    )
    case_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "gold_contradicted": 0,
            "gold_total": 0,
            "non_gold_contradicted": 0,
            "non_gold_total": 0,
        }
    )
    for row in labeled_rows:
        case_id = str(row["case_id"])
        constraint_type = str(row["constraint_type"])
        group = str(row["group"])
        category = str(row["category"])
        overall[group][category] += 1
        by_type[constraint_type][group][category] += 1
        prefix = "gold" if group == "gold" else "non_gold"
        case_counts[case_id][f"{prefix}_total"] += 1
        if category == "contradicted":
            case_counts[case_id][f"{prefix}_contradicted"] += 1

    overall_gold = _contradiction_summary(overall["gold"])
    overall_non_gold = _contradiction_summary(overall["non_gold"])
    type_rows: dict[str, Any] = {}
    for constraint_type in sorted(by_type):
        gold = _contradiction_summary(by_type[constraint_type]["gold"])
        non_gold = _contradiction_summary(by_type[constraint_type]["non_gold"])
        type_rows[constraint_type] = {
            "semantic_group": _constraint_semantic_group(constraint_type),
            "gold": gold,
            "non_gold": non_gold,
            "non_gold_minus_gold_gap": _difference(
                non_gold["contradicted_rate"], gold["contradicted_rate"]
            ),
        }
    bootstrap = _case_cluster_bootstrap_rate_gap(
        case_counts,
        samples=bootstrap_samples,
        seed=seed,
        left_success="non_gold_contradicted",
        left_total="non_gold_total",
        right_success="gold_contradicted",
        right_total="gold_total",
    )
    return {
        "unit": "candidate_constraint_support_row",
        "candidate_present_event_count": labeling["candidate_present_event_count"],
        "missing_set_metadata_event_count": labeling[
            "missing_set_metadata_event_count"
        ],
        "overall": {
            "gold": overall_gold,
            "non_gold": overall_non_gold,
            "non_gold_minus_gold_gap": _difference(
                overall_non_gold["contradicted_rate"],
                overall_gold["contradicted_rate"],
            ),
            "non_gold_to_gold_rate_ratio": (
                overall_non_gold["contradicted_rate"]
                / overall_gold["contradicted_rate"]
                if overall_gold["contradicted_rate"]
                and overall_non_gold["contradicted_rate"] is not None
                else None
            ),
        },
        "by_constraint_type": type_rows,
        "case_cluster_bootstrap": bootstrap,
        "diagnostic_only": True,
        "runtime_scoring_changed": False,
    }


def _labeled_support_rows(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, str], ...], dict[str, int]]:
    rows: list[dict[str, str]] = []
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
                for support in tuple(constraint.get("support", ()) or ()):
                    if not isinstance(support, Mapping):
                        continue
                    occurrence_id = str(support.get("occurrence_id", "") or "")
                    if not occurrence_id:
                        continue
                    rows.append(
                        {
                            "case_id": case_id,
                            "constraint_type": constraint_type,
                            "group": (
                                "gold" if occurrence_id in gold_ids else "non_gold"
                            ),
                            "category": _support_category(
                                str(support.get("status", "") or ""),
                                implicit=occurrence_id in implicit_ids,
                            ),
                        }
                    )
    return tuple(rows), {
        "candidate_present_event_count": candidate_present_event_count,
        "missing_set_metadata_event_count": missing_set_metadata_count,
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


def build_aggregation_rule_sweep(
    cases: Sequence[Mapping[str, Any]],
    *,
    observed_selection_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Replay fixed support matrices with deterministic, gold-blind rules."""
    snapshots = tuple(_final_sufficiency_snapshot(case) for case in cases)
    if len(snapshots) != len(cases):
        raise ValueError("every diagnosis case must have a sufficiency snapshot")

    max_constraint_count = max(
        (
            int(candidate["constraint_count"])
            for snapshot in snapshots
            for candidate in snapshot["candidates"]
        ),
        default=0,
    )
    ratio_thresholds = sorted(
        {
            0.0,
            1.0,
            *(
                float(candidate["supported_ratio"])
                for snapshot in snapshots
                for candidate in snapshot["candidates"]
            ),
        }
    )
    rule_specs: list[dict[str, Any]] = [
        {
            "rule_id": "R0",
            "variant_id": "R0:all_supported",
            "description": "all constraints are supported",
            "parameters": {},
        },
        {
            "rule_id": "R1",
            "variant_id": "R1:supported_or_partial",
            "description": "all constraints are supported or partial",
            "parameters": {},
        },
        {
            "rule_id": "R2",
            "variant_id": "R2:referent_all_supported",
            "description": "all present referent-identifying constraints are supported",
            "parameters": {},
        },
    ]
    rule_specs.extend(
        {
            "rule_id": "R3",
            "variant_id": f"R3:min_supported={minimum}",
            "description": "no contradiction and at least m supported constraints",
            "parameters": {"minimum_supported": minimum},
        }
        for minimum in range(1, max_constraint_count + 1)
    )
    rule_specs.extend(
        {
            "rule_id": "R4",
            "variant_id": f"R4:min_ratio={threshold:.6g}",
            "description": "supported-constraint ratio meets theta",
            "parameters": {"minimum_supported_ratio": threshold},
        }
        for threshold in ratio_thresholds
    )
    rule_specs.extend(
        (
            {
                "rule_id": "R5",
                "variant_id": "R5:support_margin=1",
                "description": "unique best candidate leads runner-up by one support",
                "parameters": {"minimum_support_margin": 1},
            },
            {
                "rule_id": "R6",
                "variant_id": "R6:referent_and_margin",
                "description": "R2 referent sufficiency and R5 comparative margin",
                "parameters": {"minimum_support_margin": 1},
            },
        )
    )

    variants: list[dict[str, Any]] = []
    for spec in rule_specs:
        rows = tuple(
            _simulate_snapshot(snapshot, spec["rule_id"], spec["parameters"])
            for snapshot in snapshots
        )
        metrics = _selection_metrics(rows)
        target_met = bool(
            _at_least(metrics.get("recall"), TARGET_MIN_RECALL)
            and metrics.get("false_commit_rate") is not None
            and float(metrics["false_commit_rate"]) <= TARGET_MAX_FALSE_COMMIT
        )
        variants.append({**spec, "metrics": metrics, "target_met": target_met})

    r0 = next(row for row in variants if row["rule_id"] == "R0")
    observed_a4 = tuple(
        row for row in observed_selection_rows if str(row.get("arm", "") or "") == "a4"
    )
    observed_metrics = _selection_metrics(observed_a4) if observed_a4 else None
    parity_fields = ("case_count", "tp", "fp", "fn", "tn")
    parity = {
        "applicable": observed_metrics is not None,
        "passed": (
            all(
                r0["metrics"].get(field) == observed_metrics.get(field)
                for field in parity_fields
            )
            if observed_metrics is not None
            else None
        ),
        "fields": list(parity_fields),
        "replayed": {field: r0["metrics"].get(field) for field in parity_fields},
        "observed": (
            {field: observed_metrics.get(field) for field in parity_fields}
            if observed_metrics is not None
            else None
        ),
    }
    target_points = [row["variant_id"] for row in variants if row["target_met"]]
    return {
        "unit": "final_sufficiency_event_per_case",
        "selection_policy": (
            "choose the highest-ranked eligible candidate; R5/R6 require a unique "
            "support-count winner"
        ),
        "case_count": len(snapshots),
        "source_event_count": sum(
            int(snapshot["source_event_count"]) for snapshot in snapshots
        ),
        "final_event_count": len(snapshots),
        "max_constraint_count": max_constraint_count,
        "target": {
            "minimum_recall": TARGET_MIN_RECALL,
            "maximum_false_commit_rate": TARGET_MAX_FALSE_COMMIT,
        },
        "target_achievable": bool(target_points),
        "target_working_points": target_points,
        "r0_observed_parity": parity,
        "variants": variants,
        "best_f1": _best_rule(variants, "f1"),
        "best_balanced_accuracy": _best_rule(variants, "balanced_accuracy"),
        "best_youden_j": _best_rule(variants, "youden_j"),
    }


def build_r5_error_geometry(
    cases: Sequence[Mapping[str, Any]],
    *,
    selection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe live R5 decisions without changing evidence or replaying a model."""
    snapshots = {
        str(snapshot["case_id"]): snapshot
        for snapshot in (_final_sufficiency_snapshot(case) for case in cases)
    }
    a4_rows = {
        str(row.get("case_id", "") or ""): row
        for row in selection_rows
        if str(row.get("arm", "") or "") == "a4"
    }
    missing = sorted(set(snapshots) - set(a4_rows))
    if missing:
        raise ValueError("missing A4 selection rows: " + ",".join(missing))

    rows: list[dict[str, Any]] = []
    for case_id in sorted(snapshots):
        snapshot = snapshots[case_id]
        selection = a4_rows[case_id]
        candidates = sorted(
            tuple(snapshot.get("candidates", ()) or ()),
            key=lambda row: (
                -int(row.get("supported_count", 0) or 0),
                int(row.get("rank", 0) or 0),
            ),
        )
        best_count = (
            int(candidates[0].get("supported_count", 0) or 0) if candidates else 0
        )
        runner_up_count = (
            int(candidates[1].get("supported_count", 0) or 0)
            if len(candidates) > 1
            else 0
        )
        margin = best_count - runner_up_count
        selected_ids = tuple(
            str(value)
            for value in tuple(selection.get("selected_occurrence_ids", ()) or ())
            if str(value)
        )
        selected_id = selected_ids[0] if len(selected_ids) == 1 else None
        by_id = {
            str(candidate.get("occurrence_id", "") or ""): candidate
            for candidate in candidates
        }
        gold_ids = set(snapshot.get("gold_occurrence_ids", ()) or ())
        candidate_present = bool(gold_ids)
        final_resolution = str(selection.get("final_resolution", "") or "")
        selected_candidate_gold = (
            selected_id in gold_ids if selected_id is not None else None
        )
        selected_statuses = tuple(by_id.get(selected_id, {}).get("statuses", ()) or ())
        selected_contradictions = tuple(
            row
            for row in selected_statuses
            if str(row.get("status", "") or "") == "contradicted"
        )
        if not candidate_present and final_resolution == "selected":
            outcome = "false_commit"
        elif candidate_present and final_resolution != "selected":
            outcome = "false_abstention"
        elif candidate_present and selected_candidate_gold is True:
            outcome = "correct_commit"
        elif candidate_present and final_resolution == "selected":
            outcome = "resolver_error"
        elif not candidate_present and final_resolution == "no_match":
            outcome = "correct_no_match"
        else:
            outcome = "other"
        if best_count == 0:
            geometry = "all_zero"
        elif margin == 0:
            geometry = "positive_tie"
        elif best_count == 1 and margin >= 1:
            geometry = "weak_unique_leader"
        elif margin >= 1:
            geometry = "strong_unique_leader"
        else:
            geometry = "positive_ambiguous"
        gold_counts = [
            int(by_id[occurrence_id].get("supported_count", 0) or 0)
            for occurrence_id in gold_ids
            if occurrence_id in by_id
        ]
        rows.append(
            {
                "case_id": case_id,
                "outcome": outcome,
                "geometry": geometry,
                "candidate_present": candidate_present,
                "final_resolution": final_resolution,
                "best_support_count": best_count,
                "runner_up_support_count": runner_up_count,
                "margin": margin,
                "constraint_count": max(
                    (
                        int(candidate.get("constraint_count", 0) or 0)
                        for candidate in candidates
                    ),
                    default=0,
                ),
                "candidate_count": len(candidates),
                "selected_candidate_gold": selected_candidate_gold,
                "selected_occurrence_id": selected_id,
                "winner_contradicted": bool(selected_contradictions),
                "winner_contradiction_constraint_types": sorted(
                    {
                        str(row.get("constraint_type", "unknown") or "unknown")
                        for row in selected_contradictions
                    }
                ),
                "winner_contradiction_passage_ids": sorted(
                    {
                        str(passage_id)
                        for row in selected_contradictions
                        for passage_id in tuple(
                            row.get("evidence_passage_ids", ()) or ()
                        )
                        if str(passage_id)
                    }
                ),
                "gold_supported_count_max": max(gold_counts) if gold_counts else None,
                "selected_supported_count": (
                    int(by_id[selected_id].get("supported_count", 0) or 0)
                    if selected_id in by_id
                    else None
                ),
            }
        )

    outcome_counts = Counter(str(row["outcome"]) for row in rows)
    geometry_by_outcome: dict[str, dict[str, int]] = {}
    for outcome in sorted(outcome_counts):
        counts = Counter(
            str(row["geometry"]) for row in rows if row["outcome"] == outcome
        )
        geometry_by_outcome[outcome] = dict(sorted(counts.items()))
    error_outcomes = {"false_commit", "false_abstention", "resolver_error"}
    return {
        "applicable": True,
        "unit": "final_live_r5_decision_per_case",
        "case_count": len(rows),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "geometry_by_outcome": geometry_by_outcome,
        "error_case_count": sum(row["outcome"] in error_outcomes for row in rows),
        "error_rows": [row for row in rows if row["outcome"] in error_outcomes],
        "correct_commit_case_count": outcome_counts.get("correct_commit", 0),
        "correct_commit_rows": [
            row for row in rows if row["outcome"] == "correct_commit"
        ],
    }


def build_winner_guard_potential(
    error_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Test whether winner contradictions meet the frozen hard-veto bar."""
    false_rows = tuple(error_geometry.get("error_rows", ()) or ())
    false_commit_rows = tuple(
        row for row in false_rows if row.get("outcome") == "false_commit"
    )
    correct_commit_rows = tuple(error_geometry.get("correct_commit_rows", ()) or ())
    false_winners_contradicted = sum(
        bool(row.get("winner_contradicted")) for row in false_commit_rows
    )
    correct_winners_contradicted = sum(
        bool(row.get("winner_contradicted")) for row in correct_commit_rows
    )

    maximum_false_commits = int(
        WP10_1_NEGATIVE_CASE_COUNT * WP11_HARD_GUARD_MAX_FALSE_COMMIT_RATE
    )
    required_false_blocks = max(0, WP10_1_FALSE_COMMIT_COUNT - maximum_false_commits)
    minimum_correct_commits = ceil(
        WP10_1_POSITIVE_CASE_COUNT * WP11_HARD_GUARD_MIN_COMMIT_RECALL
    )
    allowed_correct_blocks = max(
        0, WP10_1_CORRECT_COMMIT_COUNT - minimum_correct_commits
    )

    rows = sorted(
        (
            {
                "case_id": str(row.get("case_id", "") or ""),
                "candidate_status": (
                    "candidate_absent"
                    if row.get("outcome") == "false_commit"
                    else "candidate_present"
                ),
                "commit_outcome": str(row.get("outcome", "") or ""),
                "r5_winner": row.get("selected_occurrence_id"),
                "winner_contradicted": bool(row.get("winner_contradicted")),
                "constraint_types": tuple(
                    row.get("winner_contradiction_constraint_types", ()) or ()
                ),
                "passage_ids": tuple(
                    row.get("winner_contradiction_passage_ids", ()) or ()
                ),
            }
            for row in (*false_commit_rows, *correct_commit_rows)
        ),
        key=lambda row: row["case_id"],
    )
    qualified = (
        false_winners_contradicted >= required_false_blocks
        and correct_winners_contradicted <= allowed_correct_blocks
    )
    return {
        "applicable": bool(false_commit_rows or correct_commit_rows),
        "unit": "final_r5_winner_per_commit_case",
        "false_commit_winner_count": len(false_commit_rows),
        "false_winner_contradicted_count": false_winners_contradicted,
        "false_winner_contradiction_coverage": _ratio(
            false_winners_contradicted, len(false_commit_rows)
        ),
        "correct_commit_winner_count": len(correct_commit_rows),
        "correct_winner_contradicted_count": correct_winners_contradicted,
        "correct_winner_contradiction_rate": _ratio(
            correct_winners_contradicted, len(correct_commit_rows)
        ),
        "frozen_qualification": {
            "negative_case_count": WP10_1_NEGATIVE_CASE_COUNT,
            "baseline_false_commit_count": WP10_1_FALSE_COMMIT_COUNT,
            "maximum_false_commit_rate": WP11_HARD_GUARD_MAX_FALSE_COMMIT_RATE,
            "maximum_false_commits": maximum_false_commits,
            "required_false_blocks": required_false_blocks,
            "positive_case_count": WP10_1_POSITIVE_CASE_COUNT,
            "baseline_correct_commit_count": WP10_1_CORRECT_COMMIT_COUNT,
            "minimum_commit_recall": WP11_HARD_GUARD_MIN_COMMIT_RECALL,
            "minimum_correct_commits": minimum_correct_commits,
            "allowed_correct_blocks": allowed_correct_blocks,
        },
        "hard_veto_qualified_on_this_repeat": qualified,
        "case_rows": rows,
    }


def _final_sufficiency_snapshot(case: Mapping[str, Any]) -> dict[str, Any]:
    events = tuple(case.get("events", ()) or ())
    if not events:
        raise ValueError(f"case {case.get('case_id')} has no sufficiency event")
    event = events[-1]
    set_id = str(event.get("set_id", "") or "")
    candidate_sets = dict(case.get("candidate_sets", {}) or {})
    ranked, _ = _ranked_candidates(tuple(candidate_sets.get(set_id, ()) or ()))
    support_ids = tuple(
        dict.fromkeys(
            str(row.get("occurrence_id", "") or "")
            for constraint in tuple(event.get("constraints_checked", ()) or ())
            if isinstance(constraint, Mapping)
            for row in tuple(constraint.get("support", ()) or ())
            if isinstance(row, Mapping) and str(row.get("occurrence_id", "") or "")
        )
    )
    recorded_scope = tuple(
        str(value)
        for value in tuple(event.get("scope_occurrence_ids", ()) or ())
        if str(value)
    )
    scope_ids = recorded_scope or support_ids
    scope_set = set(scope_ids)
    ranked_ids = [
        str(candidate.get("occurrence_id", "") or "")
        for candidate in ranked
        if str(candidate.get("occurrence_id", "") or "") in scope_set
    ]
    missing_metadata = scope_set - set(ranked_ids)
    if missing_metadata:
        raise ValueError(
            f"missing candidate metadata for case {case.get('case_id')} set {set_id}"
        )
    constraints = tuple(event.get("constraints_checked", ()) or ())
    candidates: list[dict[str, Any]] = []
    for rank, occurrence_id in enumerate(ranked_ids, start=1):
        statuses: list[dict[str, str]] = []
        for constraint in constraints:
            if not isinstance(constraint, Mapping):
                continue
            match = next(
                (
                    row
                    for row in tuple(constraint.get("support", ()) or ())
                    if isinstance(row, Mapping)
                    and str(row.get("occurrence_id", "") or "") == occurrence_id
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"incomplete reconstructed support for case {case.get('case_id')}"
                )
            statuses.append(
                {
                    "constraint_type": str(
                        constraint.get("constraint_type", "unknown") or "unknown"
                    ).casefold(),
                    "status": str(
                        match.get("status", "unknown") or "unknown"
                    ).casefold(),
                    "evidence_passage_ids": tuple(
                        str(value)
                        for value in tuple(match.get("evidence_passage_ids", ()) or ())
                        if str(value)
                    ),
                }
            )
        supported_count = sum(row["status"] == "supported" for row in statuses)
        candidates.append(
            {
                "occurrence_id": occurrence_id,
                "rank": rank,
                "statuses": statuses,
                "constraint_count": len(statuses),
                "supported_count": supported_count,
                "supported_ratio": _ratio(supported_count, len(statuses)) or 0.0,
            }
        )
    clues = tuple(case.get("clues", ()) or ())
    metadata_by_id = {
        str(candidate.get("occurrence_id", "") or ""): candidate for candidate in ranked
    }
    gold_ids = {
        occurrence_id
        for occurrence_id in ranked_ids
        if _candidate_is_gold(metadata_by_id[occurrence_id], clues)
    }
    return {
        "case_id": str(case.get("case_id", "") or ""),
        "set_id": set_id,
        "source_event_count": len(events),
        "candidates": candidates,
        "gold_occurrence_ids": gold_ids,
    }


def _simulate_snapshot(
    snapshot: Mapping[str, Any], rule_id: str, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = tuple(snapshot.get("candidates", ()) or ())
    selected_id: str | None = None
    if rule_id in {"R5", "R6"}:
        winner = _comparative_winner(candidates)
        if winner is not None and (
            rule_id == "R5" or _candidate_eligible(winner, "R2", parameters)
        ):
            selected_id = str(winner["occurrence_id"])
    else:
        selected = next(
            (
                candidate
                for candidate in candidates
                if _candidate_eligible(candidate, rule_id, parameters)
            ),
            None,
        )
        if selected is not None:
            selected_id = str(selected["occurrence_id"])
    gold_ids = set(snapshot.get("gold_occurrence_ids", ()) or ())
    return {
        "case_id": str(snapshot.get("case_id", "") or ""),
        "candidate_recall_resolved_set": bool(gold_ids),
        "final_resolution": "selected" if selected_id is not None else "no_match",
        "osa_strict": selected_id in gold_ids if selected_id is not None else False,
    }


def _candidate_eligible(
    candidate: Mapping[str, Any], rule_id: str, parameters: Mapping[str, Any]
) -> bool:
    statuses = tuple(candidate.get("statuses", ()) or ())
    raw_statuses = tuple(str(row.get("status", "") or "") for row in statuses)
    if not raw_statuses:
        return False
    if rule_id == "R0":
        return all(status == "supported" for status in raw_statuses)
    if rule_id == "R1":
        return all(status in {"supported", "partial"} for status in raw_statuses)
    if rule_id == "R2":
        referent = tuple(
            str(row.get("status", "") or "")
            for row in statuses
            if str(row.get("constraint_type", "") or "")
            in REFERENT_IDENTIFYING_CONSTRAINT_TYPES
        )
        return bool(referent) and all(status == "supported" for status in referent)
    if rule_id == "R3":
        return "contradicted" not in raw_statuses and int(
            candidate.get("supported_count", 0) or 0
        ) >= int(parameters["minimum_supported"])
    if rule_id == "R4":
        return float(candidate.get("supported_ratio", 0.0) or 0.0) >= float(
            parameters["minimum_supported_ratio"]
        )
    raise ValueError(f"unsupported aggregation rule: {rule_id}")


def _comparative_winner(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda row: (
            -int(row.get("supported_count", 0) or 0),
            int(row.get("rank", 0) or 0),
        ),
    )
    best = int(ordered[0].get("supported_count", 0) or 0)
    runner_up = (
        int(ordered[1].get("supported_count", 0) or 0) if len(ordered) > 1 else 0
    )
    if best <= 0 or best - runner_up < 1:
        return None
    if len(ordered) > 1 and best == runner_up:
        return None
    return ordered[0]


def _best_rule(
    variants: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in variants
        if isinstance(dict(row.get("metrics", {}) or {}).get(metric), (int, float))
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda row: float(row["metrics"][metric]))
    return {
        "variant_id": best["variant_id"],
        "value": best["metrics"][metric],
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
    selection_diagnostics = build_selection_diagnostics(selection_rows)
    aggregation_sweep = build_aggregation_rule_sweep(
        cases,
        observed_selection_rows=selection_rows,
    )
    a4_selection_rows = tuple(
        row for row in selection_rows if str(row.get("arm", "") or "") == "a4"
    )
    error_geometry = (
        build_r5_error_geometry(cases, selection_rows=a4_selection_rows)
        if a4_selection_rows
        else {
            "applicable": False,
            "reason": "no A4 selection rows were supplied",
            "error_case_count": 0,
            "error_rows": [],
            "correct_commit_case_count": 0,
            "correct_commit_rows": [],
        }
    )
    signed_evidence = build_signed_evidence_diagnostic(
        cases,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    winner_guard = build_winner_guard_potential(error_geometry)
    target_achievable = bool(aggregation_sweep["target_achievable"])
    recommendation = (
        "PROCEED_WITH_OFFLINE_SELECTED_AGGREGATION_RULE"
        if target_achievable
        else "REPRESENTATION_INSUFFICIENT_FOR_TARGET_WORKING_POINT"
    )
    return {
        "schema_version": "OccurrenceSufficiencyOfflineDiagnosisV4",
        "case_count": len(cases),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "d1_blocker_decomposition": blockers,
        "d2_support_discrimination": discrimination,
        "d3_gold_at_k": gold_at_k,
        "d4_aggregation_rule_sweep": aggregation_sweep,
        "d5_r5_error_geometry": error_geometry,
        "d6_signed_evidence_diagnostic": signed_evidence,
        "d7_winner_guard_potential": winner_guard,
        "selection_diagnostics": selection_diagnostics,
        "recommendation": {
            "decision": recommendation,
            "target_working_point_found": target_achievable,
            "target_working_points": aggregation_sweep["target_working_points"],
            "a4_1_support_surface_calibration_authorized": False,
            "a4_2_requires_separate_representation_test": not target_achievable,
            "top_k_authorized": gold_at_k["recommended_top_k"] is not None,
            "recommended_top_k": gold_at_k["recommended_top_k"],
            "runtime_or_model_calls_used": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    d1 = report["d1_blocker_decomposition"]
    d2 = report["d2_support_discrimination"]
    d3 = report["d3_gold_at_k"]
    d4 = report["d4_aggregation_rule_sweep"]
    d5 = report["d5_r5_error_geometry"]
    d6 = report["d6_signed_evidence_diagnostic"]
    d7 = report["d7_winner_guard_potential"]
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
            "| Constraint | Group | Gold supported | Non-gold supported | Gap | Gold explicit | Non-gold explicit |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for constraint_type, row in d2["by_constraint_type"].items():
        lines.append(
            f"| {constraint_type} | {row['semantic_group']} | "
            f"{_fraction(row['gold']['counts']['supported'], row['gold']['total'])} "
            f"({_fmt(row['gold']['supported_rate'])}) | "
            f"{_fraction(row['non_gold']['counts']['supported'], row['non_gold']['total'])} "
            f"({_fmt(row['non_gold']['supported_rate'])}) | "
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
            "## D4: Aggregation-rule sweep",
            "",
            (
                f"R0 observed parity: **{_pass_fail(d4['r0_observed_parity']['passed'])}**. "
                f"Target recall >= {_fmt(d4['target']['minimum_recall'])} and "
                f"false commit <= {_fmt(d4['target']['maximum_false_commit_rate'])}: "
                f"**{'ACHIEVABLE' if d4['target_achievable'] else 'NOT ACHIEVABLE'}**."
            ),
            "",
            "| Variant | Precision | Recall | Specificity | Youden J | F1 | Balanced acc. | False commit | False abstention | Target |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for variant in d4["variants"]:
        metrics = variant["metrics"]
        lines.append(
            f"| {variant['variant_id']} | {_fmt(metrics['precision'])} | "
            f"{_fmt(metrics['recall'])} | {_fmt(metrics['specificity'])} | "
            f"{_fmt(metrics['youden_j'])} | {_fmt(metrics['f1'])} | "
            f"{_fmt(metrics['balanced_accuracy'])} | "
            f"{_fmt(metrics['false_commit_rate'])} | "
            f"{_fmt(metrics['false_abstention_rate'])} | "
            f"{'yes' if variant['target_met'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## D5: Live R5 error geometry",
            "",
        ]
    )
    if not d5["applicable"]:
        lines.append(f"Not applicable: {d5['reason']}.")
    else:
        lines.extend(
            [
                (
                    f"Errors: {d5['error_case_count']}; correct commits: "
                    f"{d5['correct_commit_case_count']}."
                ),
                "",
                "| Case | Outcome | Geometry | Best | Runner-up | Margin | Constraints | Candidates | Gold support | Selected support | Selected gold |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in (*d5["error_rows"], *d5["correct_commit_rows"]):
            lines.append(
                f"| {row['case_id']} | {row['outcome']} | {row['geometry']} | "
                f"{row['best_support_count']} | {row['runner_up_support_count']} | "
                f"{row['margin']} | {row['constraint_count']} | "
                f"{row['candidate_count']} | "
                f"{_fmt(row['gold_supported_count_max'])} | "
                f"{_fmt(row['selected_supported_count'])} | "
                f"{row['selected_candidate_gold']} |"
            )
    signed = d6["overall"]
    lines.extend(
        [
            "",
            "## D6: Dense signed-evidence diagnostic",
            "",
            (
                "Explicit contradiction rate: gold "
                f"{_fmt(signed['gold']['contradicted_rate'])}, non-gold "
                f"{_fmt(signed['non_gold']['contradicted_rate'])}; non-gold minus "
                f"gold gap {_fmt(signed['non_gold_minus_gold_gap'])}."
            ),
            (
                "Case-cluster bootstrap 95% CI "
                f"[{_fmt(d6['case_cluster_bootstrap']['ci95'][0])}, "
                f"{_fmt(d6['case_cluster_bootstrap']['ci95'][1])}]; this is a "
                "diagnostic only and does not change runtime scoring."
            ),
            "",
            "| Constraint | Gold contradicted | Non-gold contradicted | Gap |",
            "|---|---:|---:|---:|",
        ]
    )
    for constraint_type, row in d6["by_constraint_type"].items():
        lines.append(
            f"| {constraint_type} | "
            f"{_fmt(row['gold']['contradicted_rate'])} | "
            f"{_fmt(row['non_gold']['contradicted_rate'])} | "
            f"{_fmt(row['non_gold_minus_gold_gap'])} |"
        )
    qualification = d7["frozen_qualification"]
    lines.extend(
        [
            "",
            "## D7: Winner-level hard-guard potential",
            "",
            (
                "False-winner contradiction coverage: "
                f"{_fraction(d7['false_winner_contradicted_count'], d7['false_commit_winner_count'])} "
                f"({_fmt(d7['false_winner_contradiction_coverage'])}); correct-winner "
                "collateral: "
                f"{_fraction(d7['correct_winner_contradicted_count'], d7['correct_commit_winner_count'])} "
                f"({_fmt(d7['correct_winner_contradiction_rate'])})."
            ),
            (
                "Frozen hard-veto requirement: block at least "
                f"{qualification['required_false_blocks']}/"
                f"{qualification['baseline_false_commit_count']} false commits and "
                "at most "
                f"{qualification['allowed_correct_blocks']}/"
                f"{qualification['baseline_correct_commit_count']} correct commits. "
                "Repeat result: **"
                f"{_pass_fail(d7['hard_veto_qualified_on_this_repeat'])}**."
            ),
            "",
            "| Case | Status | Outcome | R5 winner | Contradicted | Constraint types | Passage IDs |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in d7["case_rows"]:
        lines.append(
            f"| {row['case_id']} | {row['candidate_status']} | "
            f"{row['commit_outcome']} | {row['r5_winner']} | "
            f"{row['winner_contradicted']} | "
            f"{', '.join(row['constraint_types']) or '-'} | "
            f"{', '.join(row['passage_ids']) or '-'} |"
        )
    lines.extend(
        [
            "",
            (
                "The sweep replays fixed final support matrices only. It does not "
                "counterfactually alter search, evidence acquisition, or model outputs."
            ),
            "",
            "## Selection diagnostics",
            "",
            "| Arm | Policy | Commit precision | Commit recall | Commit F1 | Specificity | Balanced accuracy | OSA given commit | False commit | No-match accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, policies in report["selection_diagnostics"].items():
        for policy_name in ("observed", "always_abstain"):
            metrics = policies[policy_name]
            lines.append(
                f"| {arm} | {policy_name} | {_fmt(metrics['precision'])} | "
                f"{_fmt(metrics['recall'])} | {_fmt(metrics['f1'])} | "
                f"{_fmt(metrics['specificity'])} | "
                f"{_fmt(metrics['balanced_accuracy'])} | "
                f"{_fmt(metrics['osa_given_commit'])} | "
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
    evidence_events = tuple(
        row for row in trace if row.get("type") == "occurrence_evidence_declaration"
    )
    if evidence_events:
        expanded_signed: list[dict[str, Any]] = []
        for event in evidence_events:
            set_id = str(event.get("set_id", "") or "")
            scope_ids = tuple(
                str(value)
                for value in tuple(event.get("scope_occurrence_ids", ()) or ())
                if str(value)
            )
            candidate_ids = {
                str(candidate.get("occurrence_id", "") or "")
                for candidate in tuple(candidate_sets.get(set_id, ()) or ())
                if isinstance(candidate, Mapping)
            }
            if not scope_ids or not set(scope_ids) <= candidate_ids:
                raise ValueError(
                    f"missing signed-evidence candidate metadata for set {set_id}"
                )
            normalized = _normalize_signed_constraints(
                event.get("constraints", ()), scope_ids=scope_ids
            )
            expanded_signed.append(
                {
                    **dict(event),
                    "constraints_checked": normalized,
                    "reconstructed_from_evidence_declaration": True,
                }
            )
        return tuple(expanded_signed)

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
                support_counts = {
                    occurrence_id: sum(
                        next(
                            str(row.get("status", "") or "")
                            for row in tuple(constraint.get("support", ()) or ())
                            if str(row.get("occurrence_id", "") or "") == occurrence_id
                        )
                        == "supported"
                        for constraint in normalized_constraints
                    )
                    for occurrence_id in scope_ids
                }
                if compact.get("aggregation_rule") == "unique_supported_count_margin":
                    ranked_counts = sorted(
                        support_counts.items(),
                        key=lambda item: (-item[1], scope_ids.index(item[0])),
                    )
                    best = ranked_counts[0][1] if ranked_counts else 0
                    runner_up = ranked_counts[1][1] if len(ranked_counts) > 1 else 0
                    minimum_margin = int(compact.get("minimum_support_margin", 1) or 1)
                    sufficient_ids = (
                        (ranked_counts[0][0],)
                        if ranked_counts
                        and best > 0
                        and best - runner_up >= minimum_margin
                        else ()
                    )
                else:
                    sufficient_ids = tuple(
                        occurrence_id
                        for occurrence_id in scope_ids
                        if normalized_constraints
                        and support_counts[occurrence_id] == len(normalized_constraints)
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


def _normalize_signed_constraints(
    raw_constraints: Any, *, scope_ids: Sequence[str]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    scope = tuple(dict.fromkeys(str(value) for value in scope_ids if str(value)))
    for raw_constraint in tuple(raw_constraints or ()):
        if not isinstance(raw_constraint, Mapping):
            continue
        supported = {
            str(row.get("occurrence_id", "") or ""): dict(row)
            for row in tuple(raw_constraint.get("supported_candidates", ()) or ())
            if isinstance(row, Mapping)
            and str(row.get("occurrence_id", "") or "") in scope
        }
        contradicted = {
            str(row.get("occurrence_id", "") or ""): dict(row)
            for row in tuple(raw_constraint.get("contradicted_candidates", ()) or ())
            if isinstance(row, Mapping)
            and str(row.get("occurrence_id", "") or "") in scope
        }
        conflict = set(supported) & set(contradicted)
        if conflict:
            raise ValueError(
                "signed-evidence polarity conflict: " + ",".join(sorted(conflict))
            )
        support_rows: list[dict[str, Any]] = []
        implicit_ids: list[str] = []
        for occurrence_id in scope:
            if occurrence_id in supported:
                support_rows.append(
                    {
                        **supported[occurrence_id],
                        "occurrence_id": occurrence_id,
                        "status": "supported",
                    }
                )
            elif occurrence_id in contradicted:
                support_rows.append(
                    {
                        **contradicted[occurrence_id],
                        "occurrence_id": occurrence_id,
                        "status": "contradicted",
                    }
                )
            else:
                implicit_ids.append(occurrence_id)
                support_rows.append(
                    {
                        "occurrence_id": occurrence_id,
                        "status": "unknown",
                        "evidence_passage_ids": [],
                    }
                )
        normalized.append(
            {
                **dict(raw_constraint),
                "support": support_rows,
                "implicit_unknown_occurrence_ids": implicit_ids,
            }
        )
    return normalized


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


def _constraint_semantic_group(constraint_type: str) -> str:
    normalized = str(constraint_type or "").strip().casefold()
    if normalized in REFERENT_IDENTIFYING_CONSTRAINT_TYPES:
        return "referent_identifying"
    if normalized in ANSWER_TARGET_CONSTRAINT_TYPES:
        return "answer_target"
    return "unmapped"


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


def _contradiction_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    total = sum(int(counts.get(category, 0)) for category in SUPPORT_CATEGORIES)
    contradicted = int(counts.get("contradicted", 0))
    return {
        "total": total,
        "contradicted_count": contradicted,
        "contradicted_rate": _ratio(contradicted, total),
    }


def _case_cluster_bootstrap_gap(
    case_counts: Mapping[str, Mapping[str, int]], *, samples: int, seed: int
) -> dict[str, Any]:
    return _case_cluster_bootstrap_rate_gap(
        case_counts,
        samples=samples,
        seed=seed,
        left_success="gold_supported",
        left_total="gold_total",
        right_success="non_gold_supported",
        right_total="non_gold_total",
    )


def _case_cluster_bootstrap_rate_gap(
    case_counts: Mapping[str, Mapping[str, int]],
    *,
    samples: int,
    seed: int,
    left_success: str,
    left_total: str,
    right_success: str,
    right_total: str,
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
        left_hits = sum(int(case_counts[case_id][left_success]) for case_id in sampled)
        left_count = sum(int(case_counts[case_id][left_total]) for case_id in sampled)
        right_hits = sum(
            int(case_counts[case_id][right_success]) for case_id in sampled
        )
        right_count = sum(int(case_counts[case_id][right_total]) for case_id in sampled)
        if not left_count or not right_count:
            continue
        gaps.append(left_hits / left_count - right_hits / right_count)
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
    tp = sum(
        bool(row.get("candidate_recall_resolved_set"))
        and str(row.get("final_resolution", "") or "") == "selected"
        for row in rows
    )
    fp = sum(
        not bool(row.get("candidate_recall_resolved_set"))
        and str(row.get("final_resolution", "") or "") == "selected"
        for row in rows
    )
    fn = candidate_present - tp
    tn = sum(
        not bool(row.get("candidate_recall_resolved_set"))
        and str(row.get("final_resolution", "") or "") == "no_match"
        for row in rows
    )
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    strict_correct_commits = sum(
        bool(row.get("candidate_recall_resolved_set"))
        and str(row.get("final_resolution", "") or "") == "selected"
        and row.get("osa_strict") is True
        for row in rows
    )
    specificity = _ratio(tn, candidate_absent)
    youden_j = (
        recall + specificity - 1
        if recall is not None and specificity is not None
        else None
    )
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
        "youden_j": youden_j,
        "balanced_accuracy": _mean_optional((recall, specificity)),
        "osa_given_commit": _ratio(strict_correct_commits, tp),
        "strict_correct_commit_count": strict_correct_commits,
        "wrong_occurrence_commit_count": tp - strict_correct_commits,
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
        "youden_j": (
            recall + specificity - 1
            if recall is not None and specificity is not None
            else None
        ),
        "balanced_accuracy": _mean_optional((recall, specificity)),
        "osa_given_commit": None,
        "strict_correct_commit_count": 0,
        "wrong_occurrence_commit_count": 0,
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


def _fraction(numerator: int, denominator: int) -> str:
    return f"{int(numerator)}/{int(denominator)}"


def _pass_fail(value: Any) -> str:
    if value is None:
        return "NA"
    return "PASS" if value is True else "FAIL"


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
