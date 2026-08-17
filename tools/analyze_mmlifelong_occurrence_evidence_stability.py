#!/usr/bin/env python3
"""Compare repeated occurrence evidence declarations before gate aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Mapping, Sequence


MIN_WINNER_AGREEMENT = 0.90
MIN_GATE_AGREEMENT = 0.90
MAX_FALSE_COMMIT = 0.30
MIN_COMMIT_RECALL = 0.60
MIN_OSA_GIVEN_COMMIT = 0.85


def collect_run(root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for prediction_path in sorted(Path(root).glob("cases/*/prediction.json")):
        case_id = prediction_path.parent.name
        prediction = _read_json(prediction_path)
        runtime = _read_json(prediction_path.parent / "runtime_summary.json")
        trace = tuple(
            row
            for row in tuple(runtime.get("trace", ()) or ())
            if isinstance(row, Mapping)
        )
        case = _extract_case(trace)
        if "answer_present" in prediction:
            case["answer_present"] = bool(prediction.get("answer_present"))
        elif "answer_present" in runtime:
            case["answer_present"] = bool(runtime.get("answer_present"))
        else:
            case["answer_present"] = any(
                row.get("type") == "reasoner_decision"
                and row.get("action") == "answer"
                for row in trace
            )
        case["terminal_occurrence_failure_count"] = sum(
            row.get("type") == "decision_control_exhausted"
            and any(
                isinstance(error, Mapping)
                and str(error.get("code", "")).startswith("occurrence_")
                for error in tuple(row.get("errors", ()) or ())
            )
            for row in trace
        )
        case["contradictory_gate_state_count"] = sum(
            row.get("type") == "contradictory_gate_state" for row in trace
        )
        case["full_structural_valid"] = bool(
            case["transaction_structural_valid"]
            and case["contradictory_gate_state_count"] == 0
        )
        case["working_method_valid"] = bool(
            case["full_structural_valid"]
            and case["answer_present"]
            and case["terminal_occurrence_failure_count"] == 0
        )
        cases[case_id] = case
    return cases


def build_stability_report(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    repeat_labels: Sequence[str],
    expected_cases: int,
    performance: Mapping[str, Mapping[str, Any]] | None = None,
    performance_cases: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ] | None = None,
    baseline_supported_row_agreement: float | None = None,
) -> dict[str, Any]:
    if len(repeat_labels) != 2:
        raise ValueError("exactly two independent repeat labels are required")
    if len(set(repeat_labels)) != 2:
        raise ValueError("repeat labels must be distinct")
    missing = [label for label in repeat_labels if label not in runs]
    if missing:
        raise ValueError(f"missing runs: {', '.join(missing)}")
    left_label, right_label = repeat_labels
    left = runs[left_label]
    right = runs[right_label]
    aligned_ids = tuple(sorted(set(left) & set(right)))
    if len(aligned_ids) != int(expected_cases):
        raise ValueError(
            f"aligned case count {len(aligned_ids)} != expected {expected_cases}"
        )

    per_case: dict[str, dict[str, Any]] = {}
    for case_id in aligned_ids:
        first = left[case_id]
        second = right[case_id]
        evidence_pair_valid = bool(
            first.get("evidence_declaration_valid")
            and first.get("mechanical_gate_valid")
            and second.get("evidence_declaration_valid")
            and second.get("mechanical_gate_valid")
        )
        structural_pair_valid = bool(
            first.get("full_structural_valid")
            and second.get("full_structural_valid")
        )
        working_method_pair_valid = bool(
            first.get("working_method_valid")
            and second.get("working_method_valid")
        )
        row: dict[str, Any] = {
            "evidence_pair_valid": evidence_pair_valid,
            "structural_pair_valid": structural_pair_valid,
            "working_method_pair_valid": working_method_pair_valid,
            "first_evidence_declaration_valid": bool(
                first.get("evidence_declaration_valid")
            ),
            "second_evidence_declaration_valid": bool(
                second.get("evidence_declaration_valid")
            ),
            "first_mechanical_gate_valid": bool(
                first.get("mechanical_gate_valid")
            ),
            "second_mechanical_gate_valid": bool(
                second.get("mechanical_gate_valid")
            ),
            "first_answer_present": bool(first.get("answer_present")),
            "second_answer_present": bool(second.get("answer_present")),
            "first_terminal_occurrence_failure_count": int(
                first.get("terminal_occurrence_failure_count", 0) or 0
            ),
            "second_terminal_occurrence_failure_count": int(
                second.get("terminal_occurrence_failure_count", 0) or 0
            ),
            "first_gate": first["gate"],
            "second_gate": second["gate"],
            "first_winner": first["winner"],
            "second_winner": second["winner"],
        }
        if evidence_pair_valid:
            support_ids = sorted(
                set(first["support_counts"]) | set(second["support_counts"])
            )
            count_errors = [
                abs(
                    int(first["support_counts"].get(occurrence_id, 0))
                    - int(second["support_counts"].get(occurrence_id, 0))
                )
                for occurrence_id in support_ids
            ]
            row.update(
                {
                    "supported_row_jaccard": _jaccard(
                        first["supported_rows"], second["supported_rows"]
                    ),
                    "strict_supported_row_jaccard": _jaccard(
                        first["strict_supported_rows"],
                        second["strict_supported_rows"],
                    ),
                    "candidate_passage_jaccard": _jaccard(
                        first["candidate_passage_rows"],
                        second["candidate_passage_rows"],
                    ),
                    "support_count_mae": (
                        mean(count_errors) if count_errors else 0.0
                    ),
                    "winner_agrees": first["winner"] == second["winner"],
                    "gate_agrees": first["gate"] == second["gate"],
                }
            )
        else:
            row.update(
                {
                    "supported_row_jaccard": None,
                    "strict_supported_row_jaccard": None,
                    "candidate_passage_jaccard": None,
                    "support_count_mae": None,
                    "winner_agrees": None,
                    "gate_agrees": None,
                }
            )
        per_case[case_id] = row

    evidence_rows = tuple(
        row for row in per_case.values() if row["evidence_pair_valid"]
    )
    supported_row_agreement = _mean_metric(
        evidence_rows, "supported_row_jaccard"
    )
    winner_agreement = _mean_metric(evidence_rows, "winner_agrees")
    gate_agreement = _mean_metric(evidence_rows, "gate_agrees")
    evidence_passed = bool(
        supported_row_agreement is not None
        and (
            baseline_supported_row_agreement is None
            or supported_row_agreement > baseline_supported_row_agreement
        )
    )
    stability_criteria = {
        "supported_row_agreement_improves_over_baseline": {
            "value": supported_row_agreement,
            "baseline": baseline_supported_row_agreement,
            "passed": evidence_passed,
            "required_for_working_method": False,
        },
        "winner_agreement_at_least_90pct": {
            "value": winner_agreement,
            "threshold": MIN_WINNER_AGREEMENT,
            "passed": bool(
                winner_agreement is not None
                and winner_agreement >= MIN_WINNER_AGREEMENT
            ),
            "required_for_working_method": True,
        },
        "gate_agreement_at_least_90pct": {
            "value": gate_agreement,
            "threshold": MIN_GATE_AGREEMENT,
            "passed": bool(
                gate_agreement is not None
                and gate_agreement >= MIN_GATE_AGREEMENT
            ),
            "required_for_working_method": True,
        },
    }

    performance_by_run = dict(performance or {})
    performance_criteria: dict[str, dict[str, Any]] = {}
    for label in repeat_labels:
        metrics = performance_by_run.get(label)
        if metrics is None:
            continue
        checks = {
            "false_commit_at_most_30pct": (
                metrics.get("false_commit_rate"), MAX_FALSE_COMMIT, "max"
            ),
            "commit_recall_at_least_60pct": (
                metrics.get("commit_recall"), MIN_COMMIT_RECALL, "min"
            ),
            "osa_given_commit_at_least_85pct": (
                metrics.get("osa_given_commit"), MIN_OSA_GIVEN_COMMIT, "min"
            ),
        }
        performance_criteria[label] = {
            name: {
                "value": value,
                "threshold": threshold,
                "passed": (
                    isinstance(value, (int, float))
                    and (
                        float(value) <= threshold
                        if direction == "max"
                        else float(value) >= threshold
                    )
                ),
            }
            for name, (value, threshold, direction) in checks.items()
        }

    performance_passed = bool(performance_criteria) and all(
        check["passed"]
        for run_checks in performance_criteria.values()
        for check in run_checks.values()
    ) and len(performance_criteria) == len(repeat_labels)
    stability_passed = all(
        criterion["passed"]
        for criterion in stability_criteria.values()
        if criterion["required_for_working_method"]
    )
    evidence_valid_pair_count = len(evidence_rows)
    structural_valid_pair_count = sum(
        row["structural_pair_valid"] for row in per_case.values()
    )
    working_method_valid_pair_count = sum(
        row["working_method_pair_valid"] for row in per_case.values()
    )
    structural_reliability_passed = bool(
        evidence_valid_pair_count == int(expected_cases)
        and structural_valid_pair_count == int(expected_cases)
        and working_method_valid_pair_count == int(expected_cases)
    )
    validity = {
        "all_aligned_case_count": len(aligned_ids),
        "evidence_valid_pair_count": evidence_valid_pair_count,
        "structural_valid_pair_count": structural_valid_pair_count,
        "working_method_valid_pair_count": working_method_valid_pair_count,
        "structural_invalid_pair_count": (
            len(aligned_ids) - structural_valid_pair_count
        ),
        "working_method_invalid_pair_count": (
            len(aligned_ids) - working_method_valid_pair_count
        ),
        "missing_evidence_event_count": {
            label: sum(
                int(row.get("evidence_event_count", 0) or 0) == 0
                for row in runs[label].values()
            )
            for label in repeat_labels
        },
        "missing_gate_event_count": {
            label: sum(
                int(row.get("gate_event_count", 0) or 0) == 0
                for row in runs[label].values()
            )
            for label in repeat_labels
        },
        "missing_answer_count": {
            label: sum(
                not bool(row.get("answer_present"))
                for row in runs[label].values()
            )
            for label in repeat_labels
        },
        "terminal_occurrence_failure_count": {
            label: sum(
                int(row.get("terminal_occurrence_failure_count", 0) or 0)
                for row in runs[label].values()
            )
            for label in repeat_labels
        },
    }
    return {
        "schema_version": "MMLifelongOccurrenceEvidenceStabilityV3",
        "protocol": {
            "repeat_labels": list(repeat_labels),
            "expected_cases": int(expected_cases),
            "supported_row_key": "set_id + constraint_type + occurrence_id",
            "aggregation_policy_hidden_from_reasoner": True,
            "signed_evidence_enabled": False,
        },
        "source_case_counts": {label: len(runs[label]) for label in repeat_labels},
        "aligned_case_count": len(aligned_ids),
        "validity": validity,
        "metric_denominators": {
            "evidence_stability": evidence_valid_pair_count,
            "working_method": working_method_valid_pair_count,
        },
        "metrics": {
            "supported_row_jaccard_macro": supported_row_agreement,
            "strict_supported_row_jaccard_macro": _mean_metric(
                evidence_rows, "strict_supported_row_jaccard"
            ),
            "candidate_passage_jaccard_macro": _mean_metric(
                evidence_rows, "candidate_passage_jaccard"
            ),
            "support_count_mae_macro": _mean_metric(
                evidence_rows, "support_count_mae"
            ),
            "winner_agreement": winner_agreement,
            "gate_agreement": gate_agreement,
            "gate_drift_case_count": sum(
                row["gate_agrees"] is False for row in per_case.values()
            ),
            "winner_drift_case_count": sum(
                row["winner_agrees"] is False for row in per_case.values()
            ),
            "no_match_to_selected_case_count": sum(
                row["evidence_pair_valid"]
                and
                row["first_gate"] == "insufficient"
                and row["second_gate"] == "sufficient"
                for row in per_case.values()
            ),
            "selected_to_no_match_case_count": sum(
                row["evidence_pair_valid"]
                and
                row["first_gate"] == "sufficient"
                and row["second_gate"] == "insufficient"
                for row in per_case.values()
            ),
        },
        "stability_criteria": stability_criteria,
        "performance_by_run": performance_by_run,
        "scope_size_diagnostic": build_scope_size_diagnostic(
            runs,
            repeat_labels=repeat_labels,
            performance_cases=performance_cases or {},
        ),
        "error_attribution": build_error_attribution(
            runs,
            repeat_labels=repeat_labels,
            performance_cases=performance_cases or {},
        ),
        "performance_criteria": performance_criteria,
        "stability_passed": stability_passed,
        "structural_reliability_passed": structural_reliability_passed,
        "performance_guardrails_passed": performance_passed,
        "working_method_passed": (
            structural_reliability_passed
            and stability_passed
            and performance_passed
        ),
        "gate_drift_case_ids": [
            case_id
            for case_id, row in per_case.items()
            if row["gate_agrees"] is False
        ],
        "winner_drift_case_ids": [
            case_id
            for case_id, row in per_case.items()
            if row["winner_agrees"] is False
        ],
        "evidence_invalid_pair_case_ids": [
            case_id
            for case_id, row in per_case.items()
            if not row["evidence_pair_valid"]
        ],
        "working_method_invalid_pair_case_ids": [
            case_id
            for case_id, row in per_case.items()
            if not row["working_method_pair_valid"]
        ],
        "per_case": per_case,
    }


def build_error_attribution(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    repeat_labels: Sequence[str],
    performance_cases: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Separate repeat-stable gate errors from one-repeat drift.

    A correct commit is deliberately strict: both the gold occurrence is in
    scope and the committed occurrence is correct (``osa_strict is True``).
    Support stability is measured over unique candidate-constraint rows. The
    decision-support view keeps only rows attached to each repeat's winner.
    """

    if len(repeat_labels) != 2 or len(set(repeat_labels)) != 2:
        raise ValueError("error attribution requires exactly two repeat labels")
    left_label, right_label = repeat_labels
    left_run = runs.get(left_label, {})
    right_run = runs.get(right_label, {})
    left_performance = performance_cases.get(left_label, {})
    right_performance = performance_cases.get(right_label, {})
    aligned_ids = tuple(sorted(set(left_run) & set(right_run)))

    left_false = {
        case_id
        for case_id in aligned_ids
        if left_performance.get(case_id, {}).get("false_commit") is True
    }
    right_false = {
        case_id
        for case_id in aligned_ids
        if right_performance.get(case_id, {}).get("false_commit") is True
    }
    left_correct = {
        case_id
        for case_id in aligned_ids
        if left_performance.get(case_id, {}).get("osa_strict") is True
    }
    right_correct = {
        case_id
        for case_id in aligned_ids
        if right_performance.get(case_id, {}).get("osa_strict") is True
    }

    category_ids = {
        "shared_false_commits": tuple(sorted(left_false & right_false)),
        f"{left_label}_only_false_commits": tuple(
            sorted(left_false - right_false)
        ),
        f"{right_label}_only_false_commits": tuple(
            sorted(right_false - left_false)
        ),
        "shared_correct_commits": tuple(sorted(left_correct & right_correct)),
    }
    false_union = left_false | right_false

    per_case: dict[str, dict[str, Any]] = {}
    for case_id in aligned_ids:
        left = left_run[case_id]
        right = right_run[case_id]
        left_rows = set(left.get("supported_rows", set()) or set())
        right_rows = set(right.get("supported_rows", set()) or set())
        stable_rows = left_rows & right_rows
        unstable_rows = left_rows ^ right_rows
        left_winner = str(left.get("winner", "") or "")
        right_winner = str(right.get("winner", "") or "")
        left_decision_rows = {
            row for row in left_rows if len(row) >= 3 and row[2] == left_winner
        }
        right_decision_rows = {
            row for row in right_rows if len(row) >= 3 and row[2] == right_winner
        }
        stable_decision_rows = left_decision_rows & right_decision_rows
        unstable_decision_rows = left_decision_rows ^ right_decision_rows
        left_present = _candidate_present(left_performance.get(case_id, {}))
        right_present = _candidate_present(right_performance.get(case_id, {}))
        union_rows = left_rows | right_rows
        candidate_ids = sorted(
            set(str(key) for key in left.get("support_counts", {}))
            | set(str(key) for key in right.get("support_counts", {}))
        )
        per_case[case_id] = {
            "candidate_present": (
                left_present if left_present == right_present else None
            ),
            "candidate_present_by_run": {
                left_label: left_present,
                right_label: right_present,
            },
            "gate": {
                left_label: left.get("gate"),
                right_label: right.get("gate"),
            },
            "winner": {
                left_label: left_winner,
                right_label: right_winner,
            },
            "support_counts": {
                left_label: dict(left.get("support_counts", {}) or {}),
                right_label: dict(right.get("support_counts", {}) or {}),
            },
            "supported_row_intersection": _serialize_support_rows(stable_rows),
            "supported_row_xor": _serialize_support_rows(unstable_rows),
            "decision_support_intersection": _serialize_support_rows(
                stable_decision_rows
            ),
            "decision_support_xor": _serialize_support_rows(
                unstable_decision_rows
            ),
            "constraint_types": sorted(
                {str(row[1]) for row in union_rows if len(row) >= 3}
            ),
            "candidate_ids": candidate_ids,
            "outcomes": {
                left_label: _commit_outcome(left_performance.get(case_id, {})),
                right_label: _commit_outcome(right_performance.get(case_id, {})),
            },
        }

    category_summaries = {
        category: _summarize_support_stability(
            case_ids,
            left_run=left_run,
            right_run=right_run,
        )
        for category, case_ids in category_ids.items()
    }
    return {
        "diagnostic_only": True,
        "aggregation_changed": False,
        "correct_commit_definition": "osa_strict is true in both repeats",
        "support_stability_definition": (
            "stable rows are candidate-constraint support keys declared in both "
            "repeats; unstable rows are their symmetric difference"
        ),
        "false_commit_count_by_run": {
            left_label: len(left_false),
            right_label: len(right_false),
        },
        "shared_false_commit_count": len(left_false & right_false),
        "false_commit_union_count": len(false_union),
        "shared_false_commit_fraction_of_union": (
            len(left_false & right_false) / len(false_union)
            if false_union
            else None
        ),
        "candidate_presence_mismatch_case_ids": [
            case_id
            for case_id, row in per_case.items()
            if row["candidate_present"] is None
            and any(
                value is not None
                for value in row["candidate_present_by_run"].values()
            )
        ],
        "case_ids": {key: list(value) for key, value in category_ids.items()},
        "category_summaries": category_summaries,
        "per_case": per_case,
    }


def _summarize_support_stability(
    case_ids: Sequence[str],
    *,
    left_run: Mapping[str, Mapping[str, Any]],
    right_run: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stable_rows: list[tuple[str, str, str]] = []
    unstable_rows: list[tuple[str, str, str]] = []
    stable_decision_rows: list[tuple[str, str, str]] = []
    unstable_decision_rows: list[tuple[str, str, str]] = []
    for case_id in case_ids:
        left = left_run[case_id]
        right = right_run[case_id]
        left_rows = set(left.get("supported_rows", set()) or set())
        right_rows = set(right.get("supported_rows", set()) or set())
        stable_rows.extend(left_rows & right_rows)
        unstable_rows.extend(left_rows ^ right_rows)
        left_winner = str(left.get("winner", "") or "")
        right_winner = str(right.get("winner", "") or "")
        left_decision = {
            row for row in left_rows if len(row) >= 3 and row[2] == left_winner
        }
        right_decision = {
            row for row in right_rows if len(row) >= 3 and row[2] == right_winner
        }
        stable_decision_rows.extend(left_decision & right_decision)
        unstable_decision_rows.extend(left_decision ^ right_decision)
    return {
        "case_count": len(case_ids),
        "all_positive_support": _support_stability_counts(
            stable_rows, unstable_rows
        ),
        "decision_positive_support": _support_stability_counts(
            stable_decision_rows, unstable_decision_rows
        ),
    }


def _support_stability_counts(
    stable_rows: Sequence[tuple[str, str, str]],
    unstable_rows: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    types = sorted(
        {str(row[1]) for row in (*stable_rows, *unstable_rows) if len(row) >= 3}
    )
    stable_count = len(stable_rows)
    unstable_count = len(unstable_rows)
    return {
        "stable_supported_row_count": stable_count,
        "unstable_supported_row_count": unstable_count,
        "stable_supported_rate": _ratio(stable_count, stable_count + unstable_count),
        "by_constraint_type": {
            constraint_type: {
                "stable_supported_row_count": sum(
                    len(row) >= 3 and str(row[1]) == constraint_type
                    for row in stable_rows
                ),
                "unstable_supported_row_count": sum(
                    len(row) >= 3 and str(row[1]) == constraint_type
                    for row in unstable_rows
                ),
                "stable_supported_rate": _ratio(
                    sum(
                        len(row) >= 3 and str(row[1]) == constraint_type
                        for row in stable_rows
                    ),
                    sum(
                        len(row) >= 3 and str(row[1]) == constraint_type
                        for row in (*stable_rows, *unstable_rows)
                    ),
                ),
            }
            for constraint_type in types
        },
    }


def _candidate_present(performance: Mapping[str, Any]) -> bool | None:
    if isinstance(performance.get("false_abstention"), bool):
        return True
    if isinstance(performance.get("false_commit"), bool):
        return False
    return None


def _commit_outcome(performance: Mapping[str, Any]) -> str:
    if performance.get("false_commit") is True:
        return "false_commit"
    if performance.get("osa_strict") is True:
        return "correct_commit"
    if performance.get("false_abstention") is True:
        return "false_abstention"
    if performance.get("false_commit") is False:
        return "correct_no_match"
    if performance.get("osa_strict") is False:
        return "wrong_occurrence_commit"
    return "unclassified"


def _serialize_support_rows(
    rows: Sequence[tuple[str, str, str]] | set[tuple[str, str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "set_id": str(set_id),
            "constraint_type": str(constraint_type),
            "occurrence_id": str(occurrence_id),
        }
        for set_id, constraint_type, occurrence_id in sorted(rows)
    ]


def build_scope_size_diagnostic(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    repeat_labels: Sequence[str],
    performance_cases: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    by_run: dict[str, dict[str, Any]] = {}
    for label in repeat_labels:
        run = runs.get(label, {})
        case_metrics = performance_cases.get(label, {})
        strata: dict[str, Any] = {}
        for scope_size in range(1, 6):
            rows: list[dict[str, Any]] = []
            for case_id, evidence in run.items():
                counts = {
                    str(key): int(value)
                    for key, value in dict(
                        evidence.get("support_counts", {}) or {}
                    ).items()
                }
                if len(counts) != scope_size:
                    continue
                ordered = sorted(counts.values(), reverse=True)
                best = ordered[0] if ordered else 0
                runner_up = ordered[1] if len(ordered) > 1 else 0
                performance = case_metrics.get(case_id, {})
                false_commit = performance.get("false_commit")
                false_abstention = performance.get("false_abstention")
                rows.append(
                    {
                        "false_commit": (
                            bool(false_commit)
                            if isinstance(false_commit, bool)
                            else None
                        ),
                        "commit": (
                            not bool(false_abstention)
                            if isinstance(false_abstention, bool)
                            else None
                        ),
                        "selected": evidence.get("gate") == "sufficient",
                        "total_support_count": sum(counts.values()),
                        "best_support_count": best,
                        "winner_margin": best - runner_up,
                    }
                )
            absent = [
                row["false_commit"]
                for row in rows
                if row["false_commit"] is not None
            ]
            present = [
                row["commit"] for row in rows if row["commit"] is not None
            ]
            strata[str(scope_size)] = {
                "n": len(rows),
                "candidate_absent_n": len(absent),
                "candidate_present_n": len(present),
                "false_commit_rate": _mean_values(absent),
                "commit_recall": _mean_values(present),
                "selected_rate": _mean_values(
                    [row["selected"] for row in rows]
                ),
                "mean_total_support_count": _mean_values(
                    [row["total_support_count"] for row in rows]
                ),
                "mean_best_support_count": _mean_values(
                    [row["best_support_count"] for row in rows]
                ),
                "mean_winner_margin": _mean_values(
                    [row["winner_margin"] for row in rows]
                ),
            }
        by_run[label] = {
            "case_count": len(run),
            "performance_case_count": len(case_metrics),
            "by_scope_size": strata,
        }
    return {
        "diagnostic_only": True,
        "aggregation_changed": False,
        "scope_sizes": [1, 2, 3, 4, 5],
        "by_run": by_run,
    }


def _extract_case(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evidence_events = tuple(
        row
        for row in trace
        if row.get("type") == "occurrence_evidence_declaration"
    )
    gate_events = tuple(
        row
        for row in trace
        if row.get("type") == "occurrence_sufficiency_gate_decision"
    )
    resolution_events = tuple(
        row
        for row in trace
        if row.get("type") == "occurrence_gate_resolution_committed"
    )
    decoupled = bool(evidence_events or gate_events or resolution_events)
    supported_rows: set[tuple[str, str, str]] = set()
    strict_rows: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    candidate_passage_rows: set[tuple[str, str, str]] = set()
    if evidence_events:
        for event in evidence_events:
            set_id = str(event.get("set_id", "") or "")
            for constraint in tuple(event.get("constraints", ()) or ()):
                if not isinstance(constraint, Mapping):
                    continue
                constraint_type = str(
                    constraint.get("constraint_type", "") or ""
                ).casefold()
                description = _normalize_text(constraint.get("description", ""))
                for row in tuple(
                    constraint.get("supported_candidates", ()) or ()
                ):
                    if not isinstance(row, Mapping):
                        continue
                    occurrence_id = str(row.get("occurrence_id", "") or "")
                    passages = tuple(
                        sorted(str(value) for value in row.get("evidence_passage_ids", ()) or ())
                    )
                    supported_rows.add((set_id, constraint_type, occurrence_id))
                    strict_rows.add(
                        (set_id, constraint_type, description, occurrence_id, passages)
                    )
                    candidate_passage_rows.update(
                        (set_id, occurrence_id, passage_id) for passage_id in passages
                    )
    else:
        accepted_assessment_count = 0
        for decision in trace:
            if (
                decision.get("type") != "reasoner_decision"
                or decision.get("occurrence_ops_accepted") is False
            ):
                continue
            for operation in tuple(decision.get("occurrence_ops", ()) or ()):
                if not isinstance(operation, Mapping) or str(
                    operation.get("op", operation.get("type", "")) or ""
                ).casefold() != "assess_sufficiency":
                    continue
                accepted_assessment_count += 1
                set_id = str(operation.get("set_id", "") or "")
                for constraint in tuple(
                    operation.get("constraints_checked", ()) or ()
                ):
                    if not isinstance(constraint, Mapping):
                        continue
                    constraint_type = str(
                        constraint.get("constraint_type", "") or ""
                    ).casefold()
                    description = _normalize_text(
                        constraint.get("description", "")
                    )
                    for row in tuple(constraint.get("support", ()) or ()):
                        if not isinstance(row, Mapping) or str(
                            row.get("status", "") or ""
                        ).casefold() != "supported":
                            continue
                        occurrence_id = str(row.get("occurrence_id", "") or "")
                        passages = tuple(
                            sorted(
                                str(value)
                                for value in row.get("evidence_passage_ids", ()) or ()
                            )
                        )
                        supported_rows.add((set_id, constraint_type, occurrence_id))
                        strict_rows.add(
                            (
                                set_id,
                                constraint_type,
                                description,
                                occurrence_id,
                                passages,
                            )
                        )
                        candidate_passage_rows.update(
                            (set_id, occurrence_id, passage_id)
                            for passage_id in passages
                        )
        gate_events = tuple(
            row
            for row in trace
            if row.get("type") == "occurrence_sufficiency_decision"
        )
    final_gate = gate_events[-1] if gate_events else {}
    support_counts = final_gate.get("support_count_by_occurrence", {})
    if decoupled:
        evidence_by_position = {
            _event_position(row): row for row in evidence_events
        }
        gate_by_position = {_event_position(row): row for row in gate_events}
        resolution_by_position = {
            _event_position(row): row for row in resolution_events
        }
        evidence_declaration_valid = bool(
            evidence_events
            and all(_evidence_event_valid(row) for row in evidence_events)
        )
        mechanical_gate_valid = bool(
            gate_events
            and len(gate_events) == len(evidence_events)
            and set(gate_by_position) == set(evidence_by_position)
            and all(
                _mechanical_gate_event_valid(gate)
                and str(gate.get("evidence_report_digest", "") or "")
                == str(
                    evidence_by_position[position].get(
                        "evidence_report_digest", ""
                    )
                    or ""
                )
                for position, gate in gate_by_position.items()
            )
        )
        mechanical_resolution_valid = bool(
            resolution_events
            and len(resolution_events) == len(gate_events)
            and set(resolution_by_position) == set(gate_by_position)
            and all(
                _mechanical_resolution_valid(
                    resolution,
                    gate_by_position[position],
                )
                for position, resolution in resolution_by_position.items()
            )
        )
    else:
        evidence_declaration_valid = bool(
            accepted_assessment_count
            and accepted_assessment_count == len(gate_events)
        )
        mechanical_gate_valid = bool(
            gate_events
            and all(
                str(row.get("verdict", "") or "")
                in {"sufficient", "insufficient"}
                for row in gate_events
            )
        )
        mechanical_resolution_valid = mechanical_gate_valid
    return {
        "supported_rows": supported_rows,
        "strict_supported_rows": strict_rows,
        "candidate_passage_rows": candidate_passage_rows,
        "support_counts": (
            {str(key): int(value) for key, value in support_counts.items()}
            if isinstance(support_counts, Mapping)
            else {}
        ),
        "winner": str(
            final_gate.get("winner_occurrence_id")
            or next(
                iter(final_gate.get("sufficient_occurrence_ids", ()) or ()), ""
            )
        ),
        "gate": str(final_gate.get("verdict", "") or ""),
        "evidence_event_count": (
            len(evidence_events) if decoupled else accepted_assessment_count
        ),
        "gate_event_count": len(gate_events),
        "resolution_event_count": (
            len(resolution_events) if decoupled else len(gate_events)
        ),
        "evidence_declaration_valid": evidence_declaration_valid,
        "mechanical_gate_valid": mechanical_gate_valid,
        "mechanical_resolution_valid": mechanical_resolution_valid,
        "transaction_structural_valid": bool(
            evidence_declaration_valid
            and mechanical_gate_valid
            and mechanical_resolution_valid
        ),
    }


def _event_position(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("round", 0) or 0),
        int(row.get("occurrence_op_index", 0) or 0),
        str(row.get("set_id", "") or ""),
    )


def _evidence_event_valid(event: Mapping[str, Any]) -> bool:
    scope = tuple(str(value) for value in event.get("scope_occurrence_ids", ()) or ())
    constraints = tuple(
        row
        for row in tuple(event.get("constraints", ()) or ())
        if isinstance(row, Mapping)
    )
    if not (
        str(event.get("set_id", "") or "")
        and str(event.get("evidence_report_digest", "") or "")
        and event.get("rule_blind") is True
        and not bool(event.get("model_verdict_present"))
        and event.get("support_complete") is True
        and event.get("support_contract")
        == "rule_blind_sparse_positive_evidence_v1"
        and 1 <= len(scope) <= 5
        and constraints
    ):
        return False
    scope_set = set(scope)
    return all(
        str(candidate.get("occurrence_id", "") or "") in scope_set
        and bool(tuple(candidate.get("evidence_passage_ids", ()) or ()))
        for constraint in constraints
        for candidate in tuple(
            constraint.get("supported_candidates", ()) or ()
        )
        if isinstance(candidate, Mapping)
    )


def _mechanical_gate_event_valid(event: Mapping[str, Any]) -> bool:
    raw_counts = event.get("support_count_by_occurrence", {})
    if (
        event.get("decision_owner") != "runtime"
        or bool(event.get("model_verdict_present"))
        or event.get("support_contract")
        != "rule_blind_sparse_positive_evidence_v1"
        or event.get("aggregation_rule") != "unique_supported_count_margin"
        or not isinstance(raw_counts, Mapping)
        or not raw_counts
    ):
        return False
    try:
        counts = {str(key): int(value) for key, value in raw_counts.items()}
        best_recorded = int(event.get("best_support_count", -1))
        runner_up_recorded = int(event.get("runner_up_support_count", -1))
        minimum_margin = int(event.get("minimum_support_margin", 1) or 1)
    except (TypeError, ValueError):
        return False
    if minimum_margin != 1:
        return False
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    best = ordered[0][1]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    leaders = [key for key, value in ordered if value == best]
    expected_winner = (
        leaders[0]
        if best > 0 and len(leaders) == 1 and best - runner_up >= minimum_margin
        else ""
    )
    return bool(
        best_recorded == best
        and runner_up_recorded == runner_up
        and str(event.get("winner_occurrence_id", "") or "")
        == expected_winner
        and str(event.get("verdict", "") or "")
        == ("sufficient" if expected_winner else "insufficient")
    )


def _mechanical_resolution_valid(
    resolution: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> bool:
    expected_op = "select" if gate.get("verdict") == "sufficient" else "no_match"
    op = str(resolution.get("op", resolution.get("type", "")) or "").casefold()
    if op != expected_op:
        return False
    if expected_op == "select":
        return str(resolution.get("occurrence_id", "") or "") == str(
            gate.get("winner_occurrence_id", "") or ""
        )
    return not str(resolution.get("occurrence_id", "") or "")


def _load_performance(report_path: Path) -> Mapping[str, Any]:
    report = _read_json(report_path)
    arms = report.get("arms", {})
    arm = arms.get("a4", {}) if isinstance(arms, Mapping) else {}
    if not isinstance(arm, Mapping):
        return {}
    return {
        "false_commit_rate": arm.get("false_commit_rate"),
        "commit_recall": arm.get("commit_recall"),
        "osa_given_commit": arm.get("osa_given_commit"),
        "selected_locator_usage_rate": arm.get("selected_locator_usage_rate"),
        "bound_visual_clue_recall": arm.get("bound_visual_clue_recall"),
        "exact_correct_rate": arm.get("exact_correct_rate"),
        "verified_correct_rate": arm.get("verified_correct_rate"),
    }


def _load_performance_cases(
    report_path: Path,
) -> Mapping[str, Mapping[str, Any]]:
    report = _read_json(report_path)
    cases = report.get("cases", {})
    if not isinstance(cases, Mapping):
        return {}
    return {
        str(case_id): dict(arms.get("a4", {}))
        for case_id, arms in cases.items()
        if isinstance(arms, Mapping)
        and isinstance(arms.get("a4"), Mapping)
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    validity = report["validity"]
    attribution = report["error_attribution"]
    evidence_n = report["metric_denominators"]["evidence_stability"]
    lines = [
        "# MM-Lifelong Evidence Elicitation Stability",
        "",
        f"Aligned cases: {report['aligned_case_count']}; evidence-valid pairs: {evidence_n}; full-method-valid pairs: {validity['working_method_valid_pair_count']}.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Supported-row Jaccard (macro, n={evidence_n}) | {_format_metric(metrics['supported_row_jaccard_macro'])} |",
        f"| Strict supported-row Jaccard (macro, n={evidence_n}) | {_format_metric(metrics['strict_supported_row_jaccard_macro'])} |",
        f"| Candidate-passage Jaccard (macro, n={evidence_n}) | {_format_metric(metrics['candidate_passage_jaccard_macro'])} |",
        f"| Support-count MAE (macro, n={evidence_n}) | {_format_metric(metrics['support_count_mae_macro'])} |",
        f"| Winner agreement (n={evidence_n}) | {_format_metric(metrics['winner_agreement'])} |",
        f"| Gate agreement (n={evidence_n}) | {_format_metric(metrics['gate_agreement'])} |",
        "",
        f"Structural reliability passed: `{report['structural_reliability_passed']}`. Stability passed: `{report['stability_passed']}`. Performance guardrails passed in both repeats: `{report['performance_guardrails_passed']}`. Working method passed: `{report['working_method_passed']}`.",
        "",
        "## WP11-0 Error Attribution",
        "",
        f"Shared false commits: {attribution['shared_false_commit_count']}/{attribution['false_commit_union_count']} unique false-commit cases ({_format_metric(attribution['shared_false_commit_fraction_of_union'])}).",
        "",
        "| Outcome group | Cases | Stable decision-support rows | Unstable decision-support rows | Stable rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, summary in attribution["category_summaries"].items():
        support = summary["decision_positive_support"]
        lines.append(
            f"| {category} | {summary['case_count']} | "
            f"{support['stable_supported_row_count']} | "
            f"{support['unstable_supported_row_count']} | "
            f"{_format_metric(support['stable_supported_rate'])} |"
        )
    lines.append("")
    for label, run in report["scope_size_diagnostic"]["by_run"].items():
        lines.extend(
            [
                f"## Scope-Size Diagnostic: {label}",
                "",
                "| Scope | n | Absent n | False commit | Present n | Commit recall | Best support | Winner margin |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for scope_size, row in run["by_scope_size"].items():
            lines.append(
                f"| {scope_size} | {row['n']} | {row['candidate_absent_n']} | "
                f"{_format_metric(row['false_commit_rate'])} | "
                f"{row['candidate_present_n']} | {_format_metric(row['commit_recall'])} | "
                f"{_format_metric(row['mean_best_support_count'])} | "
                f"{_format_metric(row['mean_winner_margin'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _mean_metric(
    rows: Sequence[Mapping[str, Any]], key: str
) -> float | None:
    values = [row.get(key) for row in rows]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def _mean_values(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _format_metric(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "n/a"


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs=2, action="append", metavar=("LABEL", "RUN_ROOT"), required=True
    )
    parser.add_argument("--repeat-label", action="append", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument(
        "--performance-report",
        nargs=2,
        action="append",
        metavar=("LABEL", "REPORT_JSON"),
    )
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs = {label: collect_run(Path(root)) for label, root in args.run}
    performance_reports = tuple(args.performance_report or ())
    performance = {
        label: _load_performance(Path(path))
        for label, path in performance_reports
    }
    performance_cases = {
        label: _load_performance_cases(Path(path))
        for label, path in performance_reports
    }
    baseline = None
    if args.baseline_report:
        baseline_report = _read_json(args.baseline_report)
        baseline = baseline_report.get("metrics", {}).get(
            "supported_row_jaccard_macro"
        )
    report = build_stability_report(
        runs,
        repeat_labels=tuple(args.repeat_label),
        expected_cases=args.expected_cases,
        performance=performance,
        performance_cases=performance_cases,
        baseline_supported_row_agreement=(
            float(baseline) if isinstance(baseline, (int, float)) else None
        ),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"aligned={report['aligned_case_count']} evidence_valid="
        f"{report['validity']['evidence_valid_pair_count']} gate_agreement="
        f"{_format_metric(report['metrics']['gate_agreement'])} "
        f"working_method_passed={report['working_method_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
