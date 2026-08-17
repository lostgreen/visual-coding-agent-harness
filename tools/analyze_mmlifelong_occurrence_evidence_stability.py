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
        runtime = _read_json(prediction_path.parent / "runtime_summary.json")
        trace = tuple(
            row
            for row in tuple(runtime.get("trace", ()) or ())
            if isinstance(row, Mapping)
        )
        cases[case_id] = _extract_case(trace)
    return cases


def build_stability_report(
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    repeat_labels: Sequence[str],
    expected_cases: int,
    performance: Mapping[str, Mapping[str, Any]] | None = None,
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
        per_case[case_id] = {
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
            "support_count_mae": mean(count_errors) if count_errors else 0.0,
            "winner_agrees": first["winner"] == second["winner"],
            "gate_agrees": first["gate"] == second["gate"],
            "first_gate": first["gate"],
            "second_gate": second["gate"],
            "first_winner": first["winner"],
            "second_winner": second["winner"],
        }

    supported_row_agreement = mean(
        row["supported_row_jaccard"] for row in per_case.values()
    )
    winner_agreement = mean(row["winner_agrees"] for row in per_case.values())
    gate_agreement = mean(row["gate_agrees"] for row in per_case.values())
    evidence_passed = (
        baseline_supported_row_agreement is None
        or supported_row_agreement > baseline_supported_row_agreement
    )
    stability_criteria = {
        "supported_row_agreement_improves_over_baseline": {
            "value": supported_row_agreement,
            "baseline": baseline_supported_row_agreement,
            "passed": evidence_passed,
        },
        "winner_agreement_at_least_90pct": {
            "value": winner_agreement,
            "threshold": MIN_WINNER_AGREEMENT,
            "passed": winner_agreement >= MIN_WINNER_AGREEMENT,
        },
        "gate_agreement_at_least_90pct": {
            "value": gate_agreement,
            "threshold": MIN_GATE_AGREEMENT,
            "passed": gate_agreement >= MIN_GATE_AGREEMENT,
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
        criterion["passed"] for criterion in stability_criteria.values()
    )
    return {
        "schema_version": "MMLifelongOccurrenceEvidenceStabilityV1",
        "protocol": {
            "repeat_labels": list(repeat_labels),
            "expected_cases": int(expected_cases),
            "supported_row_key": "set_id + constraint_type + occurrence_id",
            "aggregation_policy_hidden_from_reasoner": True,
            "signed_evidence_enabled": False,
        },
        "source_case_counts": {label: len(runs[label]) for label in repeat_labels},
        "aligned_case_count": len(aligned_ids),
        "metrics": {
            "supported_row_jaccard_macro": supported_row_agreement,
            "strict_supported_row_jaccard_macro": mean(
                row["strict_supported_row_jaccard"] for row in per_case.values()
            ),
            "candidate_passage_jaccard_macro": mean(
                row["candidate_passage_jaccard"] for row in per_case.values()
            ),
            "support_count_mae_macro": mean(
                row["support_count_mae"] for row in per_case.values()
            ),
            "winner_agreement": winner_agreement,
            "gate_agreement": gate_agreement,
            "gate_drift_case_count": sum(
                not row["gate_agrees"] for row in per_case.values()
            ),
            "winner_drift_case_count": sum(
                not row["winner_agrees"] for row in per_case.values()
            ),
            "no_match_to_selected_case_count": sum(
                row["first_gate"] == "insufficient"
                and row["second_gate"] == "sufficient"
                for row in per_case.values()
            ),
            "selected_to_no_match_case_count": sum(
                row["first_gate"] == "sufficient"
                and row["second_gate"] == "insufficient"
                for row in per_case.values()
            ),
        },
        "stability_criteria": stability_criteria,
        "performance_by_run": performance_by_run,
        "performance_criteria": performance_criteria,
        "stability_passed": stability_passed,
        "performance_guardrails_passed": performance_passed,
        "working_method_passed": stability_passed and performance_passed,
        "gate_drift_case_ids": [
            case_id for case_id, row in per_case.items() if not row["gate_agrees"]
        ],
        "winner_drift_case_ids": [
            case_id
            for case_id, row in per_case.items()
            if not row["winner_agrees"]
        ],
        "per_case": per_case,
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
    }


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


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# MM-Lifelong Evidence Elicitation Stability",
        "",
        f"Aligned cases: {report['aligned_case_count']}.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Supported-row Jaccard (macro) | {metrics['supported_row_jaccard_macro']:.4f} |",
        f"| Strict supported-row Jaccard (macro) | {metrics['strict_supported_row_jaccard_macro']:.4f} |",
        f"| Candidate-passage Jaccard (macro) | {metrics['candidate_passage_jaccard_macro']:.4f} |",
        f"| Support-count MAE (macro) | {metrics['support_count_mae_macro']:.4f} |",
        f"| Winner agreement | {metrics['winner_agreement']:.4f} |",
        f"| Gate agreement | {metrics['gate_agreement']:.4f} |",
        "",
        f"Stability passed: `{report['stability_passed']}`. Performance guardrails passed in both repeats: `{report['performance_guardrails_passed']}`. Working method passed: `{report['working_method_passed']}`.",
        "",
    ]
    return "\n".join(lines)


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
    performance = {
        label: _load_performance(Path(path))
        for label, path in tuple(args.performance_report or ())
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
        f"aligned={report['aligned_case_count']} gate_agreement="
        f"{report['metrics']['gate_agreement']:.4f} "
        f"working_method_passed={report['working_method_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
