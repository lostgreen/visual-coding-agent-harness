#!/usr/bin/env python3
"""Analyze two OOB negative-only sidecar repeats against frozen winners."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


REQUIRED_FALSE_BLOCKS = 5
ALLOWED_POSITIVE_BLOCKS = 0
EXPECTED_FALSE_COMMITS = 12
EXPECTED_POSITIVE_COMMITS = 8


def collect_repeat(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).glob("cases/*/sidecar_result.json")):
        row = _read_json(path)
        case_id = str(row.get("case_id", path.parent.name) or path.parent.name)
        results[case_id] = row
    if not results:
        raise FileNotFoundError(f"no sidecar results under {root}")
    return results


def build_report(
    repeats: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    frozen_rows: Sequence[Mapping[str, Any]],
    expected_cases: int,
    expected_false_commits: int = EXPECTED_FALSE_COMMITS,
    expected_positive_commits: int = EXPECTED_POSITIVE_COMMITS,
    required_false_blocks: int = REQUIRED_FALSE_BLOCKS,
    allowed_positive_blocks: int = ALLOWED_POSITIVE_BLOCKS,
    structural_only: bool = False,
) -> dict[str, Any]:
    if len(repeats) != 2:
        raise ValueError("exactly two sidecar repeats are required")
    labels = tuple(repeats)
    left_label, right_label = labels
    left = repeats[left_label]
    right = repeats[right_label]
    aligned_ids = tuple(sorted(set(left) & set(right)))
    if len(left) != expected_cases or len(right) != expected_cases:
        raise ValueError("each repeat must contain the expected case count")
    if len(aligned_ids) != expected_cases:
        raise ValueError("sidecar repeat case sets are not aligned")

    frozen = {
        str(row.get("case_id", "") or ""): row
        for row in frozen_rows
        if str(row.get("arm", "") or "") == "a4"
    }
    if not set(aligned_ids) <= set(frozen):
        raise ValueError("sidecar cases are missing from frozen selections")
    frozen = {case_id: frozen[case_id] for case_id in aligned_ids}
    false_ids = tuple(
        case_id
        for case_id in aligned_ids
        if frozen[case_id].get("candidate_recall_resolved_set") is False
        and frozen[case_id].get("final_resolution") == "selected"
    )
    positive_ids = tuple(
        case_id
        for case_id in aligned_ids
        if frozen[case_id].get("candidate_recall_resolved_set") is True
        and frozen[case_id].get("final_resolution") == "selected"
    )
    strict_correct_ids = tuple(
        case_id for case_id in positive_ids if frozen[case_id].get("osa_strict") is True
    )

    gates = _structural_gates(repeats, aligned_ids=aligned_ids)
    gates["checks"]["frozen_denominators_match"] = bool(
        len(false_ids) == int(expected_false_commits)
        and len(positive_ids) == int(expected_positive_commits)
    )
    gates["passed"] = all(gates["checks"].values())
    normalized = {
        label: _normalize_repeat(
            repeats[label],
            frozen=frozen,
            false_ids=false_ids,
            positive_ids=positive_ids,
        )
        for label in labels
    }
    per_repeat = {
        label: _repeat_metrics(
            normalized[label],
            false_ids=false_ids,
            positive_ids=positive_ids,
            strict_correct_ids=strict_correct_ids,
        )
        for label in labels
    }
    stability = _stability_metrics(
        normalized[left_label],
        normalized[right_label],
        aligned_ids=aligned_ids,
        false_ids=false_ids,
        positive_ids=positive_ids,
    )
    coverage_passed = all(
        row["false_winner_contradicted_count"] >= int(required_false_blocks)
        for row in per_repeat.values()
    )
    collateral_passed = all(
        row["positive_winner_contradicted_count"] <= int(allowed_positive_blocks)
        for row in per_repeat.values()
    )
    stable = bool(stability["winner_flag_exact_agreement"])
    if structural_only:
        decision = "STRUCTURAL_CANARY_ONLY"
    elif coverage_passed and collateral_passed and stable:
        decision = "QUALIFIES_FOR_HARD_GUARD_EXPERIMENT"
    elif coverage_passed and not collateral_passed:
        decision = "CONTRADICTION_AS_PENDING_EVIDENCE_TRIGGER_ONLY"
    else:
        decision = "STOP_SIGNED_EVIDENCE_LINE"

    case_rows = []
    for case_id in (*false_ids, *positive_ids):
        first = normalized[left_label][case_id]
        second = normalized[right_label][case_id]
        case_rows.append(
            {
                "case_id": case_id,
                "candidate_status": (
                    "candidate_absent" if case_id in false_ids else "candidate_present"
                ),
                "osa_strict": bool(frozen[case_id].get("osa_strict")),
                "frozen_winner": first["winner"],
                f"{left_label}_winner_contradicted": first["winner_contradicted"],
                f"{right_label}_winner_contradicted": second["winner_contradicted"],
                f"{left_label}_constraint_types": first["winner_constraint_types"],
                f"{right_label}_constraint_types": second["winner_constraint_types"],
                "winner_flag_agrees": (
                    first["winner_contradicted"] == second["winner_contradicted"]
                ),
            }
        )
    actual_models = sorted(
        {
            str(row.get("actual_model", "") or "")
            for repeat in repeats.values()
            for row in repeat.values()
            if str(row.get("actual_model", "") or "")
        }
    )
    return {
        "schema_version": "MMLifelongOccurrenceNegativeSidecarAnalysisV1",
        "repeat_labels": list(labels),
        "case_count": len(aligned_ids),
        "frozen_counts": {
            "false_commit_count": len(false_ids),
            "positive_commit_count": len(positive_ids),
            "strict_correct_commit_count": len(strict_correct_ids),
        },
        "qualification": {
            "required_false_blocks": int(required_false_blocks),
            "expected_false_commits": int(expected_false_commits),
            "allowed_positive_blocks": int(allowed_positive_blocks),
            "expected_positive_commits": int(expected_positive_commits),
            "coverage_passed": coverage_passed,
            "collateral_passed": collateral_passed,
            "exact_winner_flag_stability_passed": stable,
            "structural_only": structural_only,
        },
        "structural_gates": gates,
        "actual_models": actual_models,
        "per_repeat": per_repeat,
        "stability": stability,
        "case_rows": case_rows,
        "decision": decision,
        "endpoint_values_were_gates": False,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    labels = tuple(report["repeat_labels"])
    lines = [
        "# WP11 OOB Negative-Only Sidecar",
        "",
        f"Decision: **{report['decision']}**",
        "",
        (
            f"Structural gates: **{'PASS' if report['structural_gates']['passed'] else 'FAIL'}**. "
            "Endpoint values were not structural gates."
        ),
        f"Model: {', '.join(report['actual_models']) or 'unknown'}.",
        "",
        "| Repeat | False-winner coverage | Positive-commit collateral | Strict-correct collision | Identity false blocks | Event false blocks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = report["per_repeat"][label]
        lines.append(
            f"| {label} | {row['false_winner_contradicted_count']}/"
            f"{row['false_commit_count']} | "
            f"{row['positive_winner_contradicted_count']}/"
            f"{row['positive_commit_count']} | "
            f"{row['strict_correct_winner_contradicted_count']}/"
            f"{row['strict_correct_commit_count']} | "
            f"{row['false_winner_contradiction_by_type'].get('identity', 0)} | "
            f"{row['false_winner_contradiction_by_type'].get('event', 0)} |"
        )
    stability = report["stability"]
    constraint_types = sorted(
        {
            constraint_type
            for label in labels
            for key in (
                "false_winner_contradiction_by_type",
                "positive_winner_contradiction_by_type",
            )
            for constraint_type in report["per_repeat"][label][key]
        }
    )
    lines.extend(
        [
            "",
            "## Winner contradiction by constraint type",
            "",
            f"| Type | {labels[0]} false | {labels[0]} positive | {labels[1]} false | {labels[1]} positive |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for constraint_type in constraint_types:
        lines.append(
            f"| {constraint_type} | "
            f"{report['per_repeat'][labels[0]]['false_winner_contradiction_by_type'].get(constraint_type, 0)} | "
            f"{report['per_repeat'][labels[0]]['positive_winner_contradiction_by_type'].get(constraint_type, 0)} | "
            f"{report['per_repeat'][labels[1]]['false_winner_contradiction_by_type'].get(constraint_type, 0)} | "
            f"{report['per_repeat'][labels[1]]['positive_winner_contradiction_by_type'].get(constraint_type, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Stability",
            "",
            (
                f"Row/candidate/constraint/strict-passage Jaccard: "
                f"{_fmt(stability['row_jaccard'])} / "
                f"{_fmt(stability['candidate_jaccard'])} / "
                f"{_fmt(stability['constraint_jaccard'])} / "
                f"{_fmt(stability['strict_passage_jaccard'])}."
            ),
            (
                f"Winner contradiction flag agreement: "
                f"{stability['winner_flag_agreement_count']}/"
                f"{stability['winner_flag_case_count']} "
                f"({_fmt(stability['winner_flag_agreement_rate'])}); exact="
                f"{stability['winner_flag_exact_agreement']}."
            ),
            (
                "False/positive winner-flag agreement: "
                f"{_fmt(stability['false_winner_flag_agreement_rate'])} / "
                f"{_fmt(stability['positive_winner_flag_agreement_rate'])}."
            ),
            "",
            "## Frozen winner cases",
            "",
            f"| Case | Status | OSA | {labels[0]} contradicted | {labels[1]} contradicted | Agreement |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in report["case_rows"]:
        lines.append(
            f"| {row['case_id']} | {row['candidate_status']} | "
            f"{row['osa_strict']} | "
            f"{row[f'{labels[0]}_winner_contradicted']} | "
            f"{row[f'{labels[1]}_winner_contradicted']} | "
            f"{row['winner_flag_agrees']} |"
        )
    lines.extend(
        [
            "",
            "Hard-guard qualification uses the declared false-block, collateral, "
            "and winner-flag stability thresholds. Structural-only runs do not make "
            "a mechanism decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def _normalize_repeat(
    repeat: Mapping[str, Mapping[str, Any]],
    *,
    frozen: Mapping[str, Mapping[str, Any]],
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    commit_ids = set((*false_ids, *positive_ids))
    normalized: dict[str, dict[str, Any]] = {}
    for case_id, result in repeat.items():
        frozen_row = frozen[case_id]
        selected = tuple(
            str(value)
            for value in tuple(frozen_row.get("selected_occurrence_ids", ()) or ())
            if str(value)
        )
        winner = selected[0] if len(selected) == 1 else ""
        contradiction_rows = tuple(
            row
            for row in tuple(result.get("contradiction_rows", ()) or ())
            if isinstance(row, Mapping)
        )
        winner_rows = tuple(
            row
            for row in contradiction_rows
            if winner and str(row.get("occurrence_id", "") or "") == winner
        )
        normalized[case_id] = {
            "winner": winner if case_id in commit_ids else "",
            "winner_contradicted": bool(winner_rows),
            "winner_constraint_types": sorted(
                {
                    str(row.get("constraint_type", "unknown") or "unknown")
                    for row in winner_rows
                }
            ),
            "winner_passage_ids": sorted(
                {
                    str(passage_id)
                    for row in winner_rows
                    for passage_id in tuple(row.get("evidence_passage_ids", ()) or ())
                    if str(passage_id)
                }
            ),
            "rows": contradiction_rows,
        }
    return normalized


def _repeat_metrics(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
    strict_correct_ids: Sequence[str],
) -> dict[str, Any]:
    false_hits = tuple(
        case_id for case_id in false_ids if rows[case_id]["winner_contradicted"]
    )
    positive_hits = tuple(
        case_id for case_id in positive_ids if rows[case_id]["winner_contradicted"]
    )
    strict_hits = tuple(
        case_id
        for case_id in strict_correct_ids
        if rows[case_id]["winner_contradicted"]
    )
    return {
        "false_commit_count": len(false_ids),
        "false_winner_contradicted_count": len(false_hits),
        "false_winner_contradiction_coverage": _ratio(len(false_hits), len(false_ids)),
        "positive_commit_count": len(positive_ids),
        "positive_winner_contradicted_count": len(positive_hits),
        "positive_winner_contradiction_rate": _ratio(
            len(positive_hits), len(positive_ids)
        ),
        "strict_correct_commit_count": len(strict_correct_ids),
        "strict_correct_winner_contradicted_count": len(strict_hits),
        "strict_correct_winner_contradiction_rate": _ratio(
            len(strict_hits), len(strict_correct_ids)
        ),
        "false_winner_contradiction_by_type": _type_counts(rows, false_hits),
        "positive_winner_contradiction_by_type": _type_counts(rows, positive_hits),
        "false_winner_contradicted_case_ids": list(false_hits),
        "positive_winner_contradicted_case_ids": list(positive_hits),
    }


def _stability_metrics(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    aligned_ids: Sequence[str],
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
) -> dict[str, Any]:
    left_sets = _evidence_sets(left)
    right_sets = _evidence_sets(right)
    commit_ids = (*false_ids, *positive_ids)
    agreement = sum(
        left[case_id]["winner_contradicted"] == right[case_id]["winner_contradicted"]
        for case_id in commit_ids
    )
    false_agreement = sum(
        left[case_id]["winner_contradicted"] == right[case_id]["winner_contradicted"]
        for case_id in false_ids
    )
    positive_agreement = sum(
        left[case_id]["winner_contradicted"] == right[case_id]["winner_contradicted"]
        for case_id in positive_ids
    )
    return {
        "row_jaccard": _jaccard(left_sets["row"], right_sets["row"]),
        "candidate_jaccard": _jaccard(left_sets["candidate"], right_sets["candidate"]),
        "constraint_jaccard": _jaccard(
            left_sets["constraint"], right_sets["constraint"]
        ),
        "strict_passage_jaccard": _jaccard(left_sets["strict"], right_sets["strict"]),
        "mean_case_row_jaccard": mean(
            _case_row_jaccard(left[case_id], right[case_id]) for case_id in aligned_ids
        ),
        "winner_flag_case_count": len(commit_ids),
        "winner_flag_agreement_count": agreement,
        "winner_flag_agreement_rate": _ratio(agreement, len(commit_ids)),
        "winner_flag_exact_agreement": agreement == len(commit_ids),
        "false_winner_flag_agreement_rate": _ratio(false_agreement, len(false_ids)),
        "positive_winner_flag_agreement_rate": _ratio(
            positive_agreement, len(positive_ids)
        ),
    }


def _structural_gates(
    repeats: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    aligned_ids: Sequence[str],
) -> dict[str, Any]:
    labels = tuple(repeats)
    actual_models = {
        str(row.get("actual_model", "") or "")
        for repeat in repeats.values()
        for row in repeat.values()
    }
    checks: dict[str, bool] = {
        "aligned_case_sets": all(
            set(repeats[label]) == set(aligned_ids) for label in labels
        ),
        "all_results_successful": all(
            row.get("status") == "success"
            for repeat in repeats.values()
            for row in repeat.values()
        ),
        "repeat_labels_valid": all(
            row.get("repeat_label") == label
            for label, repeat in repeats.items()
            for row in repeat.values()
        ),
        "actual_model_consistent": len(actual_models) == 1 and "" not in actual_models,
        "live_model_calls": all(
            row.get("live_model_call") is True
            for repeat in repeats.values()
            for row in repeat.values()
        ),
        "snapshot_inputs_matched": all(
            repeats[labels[0]][case_id].get("snapshot_digest")
            == repeats[labels[1]][case_id].get("snapshot_digest")
            for case_id in aligned_ids
        ),
        "no_oracle_input_gate": all(
            row.get("no_oracle_input_gate_passed") is True
            for repeat in repeats.values()
            for row in repeat.values()
        ),
        "negative_only_visibility": all(
            row.get("positive_support_visible_to_model") is False
            and row.get("selection_state_visible_to_model") is False
            for repeat in repeats.values()
            for row in repeat.values()
        ),
        "out_of_band_isolation": all(
            row.get("workspace_write_enabled") is False
            and row.get("reasoner_context_write_enabled") is False
            for repeat in repeats.values()
            for row in repeat.values()
        ),
        "no_raw_response_or_prompt_persisted": all(
            row.get("raw_response_persisted") is False
            and row.get("prompt_persisted") is False
            for repeat in repeats.values()
            for row in repeat.values()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _evidence_sets(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[tuple[str, ...]]]:
    result = {"row": set(), "candidate": set(), "constraint": set(), "strict": set()}
    for case_id, case in rows.items():
        for row in tuple(case.get("rows", ()) or ()):
            constraint_id = str(row.get("constraint_id", "") or "")
            occurrence_id = str(row.get("occurrence_id", "") or "")
            result["row"].add((case_id, constraint_id, occurrence_id))
            result["candidate"].add((case_id, occurrence_id))
            result["constraint"].add((case_id, constraint_id))
            for passage_id in tuple(row.get("evidence_passage_ids", ()) or ()):
                result["strict"].add(
                    (case_id, constraint_id, occurrence_id, str(passage_id))
                )
    return result


def _case_row_jaccard(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    def keys(row: Mapping[str, Any]) -> set[tuple[str, str]]:
        return {
            (
                str(value.get("constraint_id", "") or ""),
                str(value.get("occurrence_id", "") or ""),
            )
            for value in tuple(row.get("rows", ()) or ())
        }

    return _jaccard(keys(left), keys(right))


def _type_counts(
    rows: Mapping[str, Mapping[str, Any]], case_ids: Sequence[str]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case_id in case_ids:
        counts.update(set(rows[case_id]["winner_constraint_types"]))
    return dict(sorted(counts.items()))


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _load_frozen_rows(
    run_root: Path, *, evaluation_record_root: Path
) -> tuple[dict[str, Any], ...]:
    module_path = Path(__file__).with_name("analyze_mmlifelong_occurrence_agent.py")
    spec = importlib.util.spec_from_file_location(
        "sidecar_frozen_analysis", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load occurrence analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(
        module.collect_rows(
            (Path(run_root),),
            evaluation_record_root=Path(evaluation_record_root),
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_repeat(value: str) -> tuple[str, Path]:
    label, separator, path = str(value).partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("repeat must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", action="append", type=_parse_repeat, required=True)
    parser.add_argument("--frozen-run-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument(
        "--expected-false-commits", type=int, default=EXPECTED_FALSE_COMMITS
    )
    parser.add_argument(
        "--expected-positive-commits", type=int, default=EXPECTED_POSITIVE_COMMITS
    )
    parser.add_argument(
        "--required-false-blocks", type=int, default=REQUIRED_FALSE_BLOCKS
    )
    parser.add_argument(
        "--allowed-positive-blocks", type=int, default=ALLOWED_POSITIVE_BLOCKS
    )
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    repeats = {label: collect_repeat(path) for label, path in args.repeat}
    frozen_rows = _load_frozen_rows(
        Path(args.frozen_run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
    )
    report = build_report(
        repeats,
        frozen_rows=frozen_rows,
        expected_cases=args.expected_cases,
        expected_false_commits=args.expected_false_commits,
        expected_positive_commits=args.expected_positive_commits,
        required_false_blocks=args.required_false_blocks,
        allowed_positive_blocks=args.allowed_positive_blocks,
        structural_only=args.structural_only,
    )
    _write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"SIDECAR_ANALYSIS_DONE decision={report['decision']} "
        f"structural_gate={report['structural_gates']['passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
