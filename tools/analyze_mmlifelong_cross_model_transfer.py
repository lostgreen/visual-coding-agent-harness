#!/usr/bin/env python3
"""Analyze one-factor cross-model transfer of the O1 causal chain."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


_LADDER_PATH = Path(__file__).with_name("analyze_mmlifelong_oracle_ladder.py")
_SPEC = importlib.util.spec_from_file_location("mmlifelong_oracle_ladder", _LADDER_PATH)
assert _SPEC and _SPEC.loader
LADDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LADDER)

ARMS = ("o1", "o1.5", "o1.75")
DELTA_SPECS = (
    ("selection", "o1.5", "o1"),
    ("anchor", "o1.75", "o1.5"),
)


def collect_stack_rows(
    bindings: Mapping[str, Mapping[str, Path]],
    *,
    evaluation_record_root: Path,
    case_ids: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for stack, arm_roots in bindings.items():
        for declared_arm, root in arm_roots.items():
            root_rows = LADDER.collect_rows(
                (Path(root),),
                evaluation_record_root=evaluation_record_root,
                case_ids=case_ids,
            )
            for raw in root_rows:
                row = dict(raw)
                if row["arm"] != declared_arm:
                    raise ValueError(
                        f"declared {stack}:{declared_arm} contains {row['arm']}"
                    )
                row["stack"] = stack
                rows.append(row)
    return tuple(rows)


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    bootstrap_samples: int = 10_000,
    seed: int = 20260812,
    minimum_effect: float = 0.02,
) -> dict[str, Any]:
    manifest_cases = tuple(
        str(row["case_id"])
        for row in tuple(manifest.get("cases", ()) or ())
        if isinstance(row, Mapping) and row.get("case_id")
    )
    expected_cases = set(manifest_cases)
    by_stack: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        stack, arm, case_id = (
            str(row["stack"]),
            str(row["arm"]),
            str(row["case_id"]),
        )
        if case_id in by_stack[stack][arm]:
            raise ValueError(f"duplicate row: {stack}:{arm}:{case_id}")
        by_stack[stack][arm][case_id] = row
    stacks = tuple(sorted(by_stack, key=lambda value: (value != "base", value)))

    stack_case_checks = {
        stack: {
            arm: set(by_stack[stack].get(arm, {})) == expected_cases for arm in ARMS
        }
        for stack in stacks
    }
    model_pairs = {
        stack: _stack_model_pair(by_stack[stack], expected_cases) for stack in stacks
    }
    base_models = model_pairs.get("base", {})
    factor_checks: dict[str, bool] = {}
    for stack, models in model_pairs.items():
        if stack == "base":
            continue
        if stack == "r1_i0":
            factor_checks[stack] = (
                models.get("investigator") == base_models.get("investigator")
                and models.get("reasoner") != base_models.get("reasoner")
            )
        elif stack == "r0_i1":
            factor_checks[stack] = (
                models.get("reasoner") == base_models.get("reasoner")
                and models.get("investigator") != base_models.get("investigator")
            )
        elif stack == "r1_i1":
            factor_checks[stack] = (
                models.get("reasoner") != base_models.get("reasoner")
                and models.get("investigator") != base_models.get("investigator")
            )
        else:
            factor_checks[stack] = False

    runtime_checks = {
        "base_stack_present": "base" in by_stack,
        "manifest_is_outcome_independent": manifest.get(
            "selection_is_outcome_independent"
        )
        is True,
        "manifest_case_ids_unique": bool(manifest_cases)
        and len(manifest_cases) == len(expected_cases),
        "all_stack_arm_case_sets_match_manifest": all(
            value
            for stack_checks in stack_case_checks.values()
            for value in stack_checks.values()
        ),
        "frozen_runtime_aligned_except_models": _frozen_runtime_aligned(
            by_stack, expected_cases
        ),
        "model_factor_isolation_valid": bool(factor_checks)
        and all(factor_checks.values()),
        "o1_family_candidate_pools_aligned": _candidate_pools_aligned(
            by_stack, expected_cases
        ),
        "natural_caption_retrieval_aligned": _natural_retrieval_aligned(
            by_stack, expected_cases
        ),
        "oracle_recall_complete": all(
            float(row.get("audit", {}).get("final_clue_recall", -1.0)) == 1.0
            for stack in stacks
            for arm in ARMS
            for row in by_stack[stack][arm].values()
        ),
    }
    evaluation_checks = {
        "all_evaluated": all(
            isinstance(row.get("score"), (int, float)) for row in rows
        ),
        "all_judge_responses_parsed": all(
            row.get("parse_status") == "parsed" for row in rows
        ),
        "judge_model_frozen": len(
            {str(row.get("judge_model")) for row in rows if row.get("judge_model")}
        )
        == 1,
    }

    deltas: list[dict[str, Any]] = []
    per_case_deltas: dict[tuple[str, str], dict[str, float]] = {}
    for stack_index, stack in enumerate(stacks):
        for delta_index, (name, left, right) in enumerate(DELTA_SPECS):
            values = {
                case_id: float(by_stack[stack][left][case_id]["score"])
                - float(by_stack[stack][right][case_id]["score"])
                for case_id in sorted(expected_cases)
                if isinstance(by_stack[stack][left][case_id].get("score"), (int, float))
                and isinstance(by_stack[stack][right][case_id].get("score"), (int, float))
            }
            per_case_deltas[(stack, name)] = values
            low, high = LADDER._bootstrap_ci(
                tuple(values.values()),
                samples=bootstrap_samples,
                seed=seed + stack_index * 10 + delta_index,
            )
            deltas.append(
                _delta_row(stack, name, values, low=low, high=high)
            )

    difference_of_differences: list[dict[str, Any]] = []
    for stack_index, stack in enumerate(stacks):
        if stack == "base":
            continue
        for delta_index, (name, _, _) in enumerate(DELTA_SPECS):
            alt = per_case_deltas.get((stack, name), {})
            base = per_case_deltas.get(("base", name), {})
            values = {
                case_id: alt[case_id] - base[case_id]
                for case_id in sorted(set(alt) & set(base))
            }
            low, high = LADDER._bootstrap_ci(
                tuple(values.values()),
                samples=bootstrap_samples,
                seed=seed + 100 + stack_index * 10 + delta_index,
            )
            difference_of_differences.append(
                _delta_row(stack, f"{name}_dod_vs_base", values, low=low, high=high)
            )

    arm_metrics = [
        _arm_metrics(stack, arm, tuple(by_stack[stack][arm].values()))
        for stack in stacks
        for arm in ARMS
    ]
    strata = _stratified_deltas(by_stack, expected_cases)
    acceptance = {}
    for stack in stacks:
        if stack == "base":
            continue
        stack_deltas = {
            row["effect"]: row for row in deltas if row["stack"] == stack
        }
        single_selection = next(
            (
                row
                for row in strata
                if row["stack"] == stack
                and row["effect"] == "selection"
                and row["dimension"] == "clue_count"
                and row["bucket"] == "single"
            ),
            None,
        )
        acceptance[stack] = {
            "selection_positive_and_nontrivial": _positive_nontrivial(
                stack_deltas.get("selection"), minimum_effect
            ),
            "anchor_positive_and_nontrivial": _positive_nontrivial(
                stack_deltas.get("anchor"), minimum_effect
            ),
            "single_clue_selection_positive": bool(single_selection)
            and float(single_selection["mean_score_delta"]) > 0.0,
        }
        acceptance[stack]["passed"] = all(acceptance[stack].values())

    checks = {**runtime_checks, **evaluation_checks}
    return {
        "schema_version": "MMLifelongCrossModelTransferReportV1",
        "manifest_digest": _digest(manifest),
        "expected_cases": len(expected_cases),
        "stacks": list(stacks),
        "model_pairs": model_pairs,
        "model_factor_checks": factor_checks,
        "runtime_gate_passed": all(runtime_checks.values()),
        "runtime_gate_checks": runtime_checks,
        "gate_passed": all(checks.values()),
        "gate_checks": checks,
        "acceptance_minimum_effect": float(minimum_effect),
        "acceptance": acceptance,
        "judge_models": sorted(
            {str(row.get("judge_model")) for row in rows if row.get("judge_model")}
        ),
        "arm_metrics": arm_metrics,
        "paired_deltas": deltas,
        "difference_of_differences": difference_of_differences,
        "stratified_deltas": strata,
    }


def _stack_model_pair(
    rows_by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> dict[str, str]:
    variants = {
        json.dumps(
            row.get("frozen_config", {}).get("models", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        for arm in ARMS
        for case_id, row in rows_by_arm.get(arm, {}).items()
        if case_id in cases
    }
    if len(variants) != 1:
        return {}
    value = json.loads(next(iter(variants)))
    return {
        "reasoner": str(value.get("reasoner", "")),
        "investigator": str(value.get("investigator", "")),
    }


def _frozen_runtime_aligned(
    by_stack: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    cases: set[str],
) -> bool:
    for case_id in cases:
        signatures = []
        for stack in by_stack:
            for arm in ARMS:
                row = by_stack[stack].get(arm, {}).get(case_id)
                if row is None:
                    return False
                config = dict(row.get("frozen_config", {}) or {})
                config.pop("models", None)
                signatures.append(_digest(config))
        if len(set(signatures)) != 1:
            return False
    return bool(cases)


def _candidate_pools_aligned(
    by_stack: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    cases: set[str],
) -> bool:
    for case_id in cases:
        signatures = []
        for stack in by_stack:
            for arm in ARMS:
                row = by_stack[stack].get(arm, {}).get(case_id)
                audit = row.get("audit") if row else None
                if not isinstance(audit, Mapping):
                    return False
                signatures.append(
                    _digest(
                        {
                            "candidate_passage_ids": audit.get("candidate_passage_ids"),
                            "candidate_intervals": audit.get("candidate_intervals"),
                            "shuffle_seed_digest": audit.get("shuffle_seed_digest"),
                        }
                    )
                )
        if len(set(signatures)) != 1:
            return False
    return bool(cases)


def _natural_retrieval_aligned(
    by_stack: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    cases: set[str],
) -> bool:
    for case_id in cases:
        signatures = []
        for stack in by_stack:
            for arm in ARMS:
                row = by_stack[stack].get(arm, {}).get(case_id)
                audit = row.get("audit") if row else None
                if not isinstance(audit, Mapping):
                    return False
                signatures.append(
                    (
                        audit.get("caption_config_digest"),
                        audit.get("intervention_digest"),
                        audit.get("natural_candidate_count"),
                        audit.get("natural_clue_recall"),
                    )
                )
        if len(set(signatures)) != 1:
            return False
    return bool(cases)


def _delta_row(
    stack: str,
    effect: str,
    values: Mapping[str, float],
    *,
    low: float | None,
    high: float | None,
) -> dict[str, Any]:
    sequence = tuple(values.values())
    return {
        "stack": stack,
        "effect": effect,
        "case_count": len(sequence),
        "mean_score_delta": mean(sequence) if sequence else None,
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(value > 0 for value in sequence),
        "ties": sum(value == 0 for value in sequence),
        "losses": sum(value < 0 for value in sequence),
    }


def _arm_metrics(
    stack: str, arm: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "stack": stack,
        "arm": arm,
        "case_count": len(rows),
        "mean_score": _optional_mean(row.get("score") for row in rows),
        "exact_correct_rate": _optional_mean(
            float(row["score"] == 1.0)
            for row in rows
            if isinstance(row.get("score"), (int, float))
        ),
        "selected_candidate_request_recall": _optional_mean(
            row.get("selected_candidate_request_recall") for row in rows
        ),
        "selected_candidate_inspection_recall": _optional_mean(
            row.get("selected_candidate_inspection_recall") for row in rows
        ),
        "anchor_request_recall": _optional_mean(
            row.get("anchor_request_recall") for row in rows
        ),
        "anchor_inspection_recall": _optional_mean(
            row.get("anchor_inspection_recall") for row in rows
        ),
        "clue_center_visual_recall": _optional_mean(
            row.get("clue_center_visual_recall") for row in rows
        ),
        "visual_frames": _optional_mean(row.get("visual_frames") for row in rows),
        "vlm_calls": _optional_mean(row.get("vlm_calls") for row in rows),
        "occurrence_candidate_count": _optional_mean(
            row.get("occurrence_candidate_count") for row in rows
        ),
    }


def _stratified_deltas(
    by_stack: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    cases: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for stack in sorted(by_stack, key=lambda value: (value != "base", value)):
        for effect, left, right in DELTA_SPECS:
            grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
            for case_id in sorted(cases):
                left_row = by_stack[stack][left][case_id]
                right_row = by_stack[stack][right][case_id]
                if not isinstance(left_row.get("score"), (int, float)) or not isinstance(
                    right_row.get("score"), (int, float)
                ):
                    continue
                delta = float(left_row["score"]) - float(right_row["score"])
                audit = right_row.get("audit", {}) or {}
                buckets = {
                    "clue_count": (
                        "single" if int(right_row.get("clue_count", 0)) == 1 else "multi"
                    ),
                    "question_type": str(right_row.get("question_type", "Unknown")),
                    "clue_duration": LADDER._duration_bucket(
                        float(right_row.get("clue_duration_sec", 0.0))
                    ),
                    "candidate_count": str(audit.get("final_candidate_count", "unknown")),
                    "occurrence_count": str(
                        right_row.get("occurrence_candidate_count", "unknown")
                    ),
                }
                for dimension, bucket in buckets.items():
                    grouped[(dimension, bucket)].append(delta)
            rows.extend(
                {
                    "stack": stack,
                    "effect": effect,
                    "dimension": dimension,
                    "bucket": bucket,
                    "case_count": len(values),
                    "mean_score_delta": mean(values),
                }
                for (dimension, bucket), values in sorted(grouped.items())
            )
    return rows


def _positive_nontrivial(row: Mapping[str, Any] | None, threshold: float) -> bool:
    return bool(row) and isinstance(row.get("mean_score_delta"), (int, float)) and float(
        row["mean_score_delta"]
    ) >= float(threshold)


def _optional_mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong Cross-Model Transfer",
        "",
        f"Runtime gate: `{str(report['runtime_gate_passed']).lower()}`",
        f"Evaluation gate: `{str(report['gate_passed']).lower()}`",
        "",
        "| Stack | Arm | N | Mean | Exact | Selected req. | Selected inspect | Anchor req. | Anchor inspect | Clue center | Frames | VLM calls |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["arm_metrics"]:
        values = (
            row["stack"], row["arm"], row["case_count"], row["mean_score"],
            row["exact_correct_rate"], row["selected_candidate_request_recall"],
            row["selected_candidate_inspection_recall"], row["anchor_request_recall"],
            row["anchor_inspection_recall"], row["clue_center_visual_recall"],
            row["visual_frames"], row["vlm_calls"],
        )
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(("", "| Stack | Effect | N | Delta | 95% CI | W/T/L |", "| --- | --- | ---: | ---: | ---: | ---: |"))
    for row in (*report["paired_deltas"], *report["difference_of_differences"]):
        lines.append(
            f"| {row['stack']} | {row['effect']} | {row['case_count']} | "
            f"{_fmt(row['mean_score_delta'])} | [{_fmt(row['ci95_low'])}, "
            f"{_fmt(row['ci95_high'])}] | {row['wins']}/{row['ties']}/{row['losses']} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_binding(value: str) -> tuple[str, str, Path]:
    left, separator, raw_path = value.partition("=")
    stack, colon, arm = left.partition(":")
    if not separator or not colon or not stack or arm not in ARMS or not raw_path:
        raise argparse.ArgumentTypeError("expected STACK:ARM=PATH")
    return stack, arm, Path(raw_path)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=_parse_binding, required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--minimum-effect", type=float, default=0.02)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    bindings: dict[str, dict[str, Path]] = defaultdict(dict)
    for stack, arm, path in args.run_root:
        if arm in bindings[stack]:
            raise ValueError(f"duplicate binding: {stack}:{arm}")
        bindings[stack][arm] = path
    manifest = json.loads(Path(args.case_manifest).read_text(encoding="utf-8"))
    case_ids = frozenset(str(row["case_id"]) for row in manifest["cases"])
    report = build_report(
        collect_stack_rows(
            bindings,
            evaluation_record_root=Path(args.evaluation_record_root),
            case_ids=case_ids,
        ),
        manifest=manifest,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        minimum_effect=args.minimum_effect,
    )
    _write(Path(args.out_json), json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write(Path(args.out_md), render_markdown(report))
    print(
        json.dumps(
            {
                "runtime_gate_passed": report["runtime_gate_passed"],
                "gate_passed": report["gate_passed"],
                "stacks": report["stacks"],
                "model_pairs": report["model_pairs"],
                "acceptance": report["acceptance"],
                "out_json": args.out_json,
                "out_md": args.out_md,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
