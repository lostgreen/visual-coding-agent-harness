#!/usr/bin/env python3
"""Reconcile repeated occurrence-control runs on one complete case cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


METRICS = (
    "exact_correct_rate",
    "verified_correct_rate",
    "grounded_correct_ref300_rate",
    "osa_strict",
    "false_commit_rate",
    "selected_locator_usage_rate",
    "bound_visual_clue_recall",
    "mean_visual_frames",
    "mean_vlm_calls",
    "mean_semantic_rounds_used",
)


def _load_analyzer() -> Any:
    module_path = Path(__file__).with_name("analyze_mmlifelong_occurrence_agent.py")
    spec = importlib.util.spec_from_file_location(
        "occurrence_analysis_for_repeated_controls", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load occurrence analyzer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_analyzer()


def build_repeated_control_report(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    repeat_labels: Sequence[str],
    effect_pair: tuple[str, str],
) -> dict[str, Any]:
    if len(repeat_labels) < 2:
        raise ValueError("at least two repeat labels are required")
    required = set(repeat_labels) | set(effect_pair)
    missing = sorted(required - set(runs))
    if missing:
        raise ValueError(f"missing run labels: {', '.join(missing)}")

    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for label, rows in runs.items():
        by_case: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("case_id", "") or "")
            if not case_id:
                raise ValueError(f"run {label} contains a row without case_id")
            if case_id in by_case:
                raise ValueError(f"duplicate row in {label}: {case_id}")
            by_case[case_id] = row
        indexed[label] = by_case

    aligned = set.intersection(*(set(indexed[label]) for label in required))
    common_complete = {
        case_id
        for case_id in aligned
        if all(
            ANALYZER._row_frozen_replay_complete(indexed[label][case_id])
            for label in required
        )
    }
    common_ids = tuple(sorted(common_complete))
    metrics_by_run = {
        label: ANALYZER._aggregate_arm(
            tuple(indexed[label][case_id] for case_id in common_ids)
        )
        for label in sorted(required)
    }

    repeat_variation: dict[str, Any] = {}
    for metric in METRICS:
        values = {
            label: metrics_by_run[label].get(metric) for label in repeat_labels
        }
        numeric = {
            label: float(value)
            for label, value in values.items()
            if isinstance(value, (int, float))
        }
        repeat_variation[metric] = {
            "values": values,
            "minimum": min(numeric.values()) if numeric else None,
            "maximum": max(numeric.values()) if numeric else None,
            "range": (
                max(numeric.values()) - min(numeric.values()) if numeric else None
            ),
        }

    treatment_label, baseline_label = effect_pair
    effects: dict[str, Any] = {}
    for metric in METRICS:
        treatment = metrics_by_run[treatment_label].get(metric)
        baseline = metrics_by_run[baseline_label].get(metric)
        delta = (
            float(treatment) - float(baseline)
            if isinstance(treatment, (int, float))
            and isinstance(baseline, (int, float))
            else None
        )
        variation = repeat_variation[metric]["range"]
        effects[metric] = {
            "treatment": treatment,
            "baseline": baseline,
            "delta": delta,
            "repeat_range": variation,
            "absolute_effect_to_variation_ratio": (
                abs(delta) / float(variation)
                if delta is not None
                and isinstance(variation, (int, float))
                and variation > 0
                else None
            ),
            "zero_repeat_variation": variation == 0,
        }

    return {
        "schema_version": "MMLifelongOccurrenceRepeatedControlsV1",
        "source_case_counts": {
            label: len(indexed[label]) for label in sorted(required)
        },
        "source_complete_counts": {
            label: sum(
                ANALYZER._row_frozen_replay_complete(row)
                for row in indexed[label].values()
            )
            for label in sorted(required)
        },
        "aligned_case_count": len(aligned),
        "common_complete_case_count": len(common_ids),
        "common_complete_case_ids": list(common_ids),
        "excluded_from_common_complete": {
            label: sorted(
                case_id
                for case_id in aligned
                if not ANALYZER._row_frozen_replay_complete(
                    indexed[label][case_id]
                )
            )
            for label in sorted(required)
        },
        "repeat_labels": list(repeat_labels),
        "effect_pair": {
            "treatment": treatment_label,
            "baseline": baseline_label,
        },
        "metrics_by_run": metrics_by_run,
        "repeat_variation": repeat_variation,
        "effect_to_variation": effects,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong Repeated-Control Reconciliation",
        "",
        (
            f"Aligned cases: {report['aligned_case_count']}; common frozen-complete "
            f"cases: {report['common_complete_case_count']}."
        ),
        "",
        "| Run | Source N | Complete N | Common exact | OSA strict | False commit | Locator use | Bound visual recall | Frames | VLM calls | Rounds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in report["metrics_by_run"].items():
        lines.append(
            f"| {label} | {report['source_case_counts'][label]} | "
            f"{report['source_complete_counts'][label]} | "
            f"{_fmt(metrics['exact_correct_rate'])} | "
            f"{_fmt(metrics['osa_strict'])} | "
            f"{_fmt(metrics['false_commit_rate'])} | "
            f"{_fmt(metrics['selected_locator_usage_rate'])} | "
            f"{_fmt(metrics['bound_visual_clue_recall'])} | "
            f"{_fmt(metrics['mean_visual_frames'])} | "
            f"{_fmt(metrics['mean_vlm_calls'])} | "
            f"{_fmt(metrics['mean_semantic_rounds_used'])} |"
        )
    pair = report["effect_pair"]
    lines.extend(
        [
            "",
            (
                f"Effect is `{pair['treatment']} - {pair['baseline']}`; repeat range "
                f"uses {', '.join(report['repeat_labels'])}."
            ),
            "",
            "| Metric | Effect | Repeat range | |Effect| / range |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, row in report["effect_to_variation"].items():
        lines.append(
            f"| {metric} | {_fmt(row['delta'])} | "
            f"{_fmt(row['repeat_range'])} | "
            f"{_fmt(row['absolute_effect_to_variation_ratio'])} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("LABEL", "RUN_ROOT", "EVALUATION_ROOT"),
        required=True,
    )
    parser.add_argument("--repeat-label", action="append", required=True)
    parser.add_argument(
        "--effect-pair",
        nargs=2,
        required=True,
        metavar=("TREATMENT", "BASELINE"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs: dict[str, Sequence[Mapping[str, Any]]] = {}
    for label, run_root, evaluation_root in args.run:
        if label in runs:
            raise ValueError(f"duplicate run label: {label}")
        runs[label] = ANALYZER.collect_rows(
            (Path(run_root),), evaluation_record_root=Path(evaluation_root)
        )
    report = build_repeated_control_report(
        runs,
        repeat_labels=tuple(args.repeat_label),
        effect_pair=tuple(args.effect_pair),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"common_complete={report['common_complete_case_count']} "
        f"repeat_labels={','.join(report['repeat_labels'])}"
    )
    return 0


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
