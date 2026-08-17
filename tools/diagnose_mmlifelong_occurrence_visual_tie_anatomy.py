#!/usr/bin/env python3
"""Run the zero-model WP15-1 tie/loss anatomy diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_visual_tie_anatomy import build_visual_tie_anatomy_report


def render_markdown(report: Mapping[str, Any]) -> str:
    cases = report["case_level"]
    anatomy = report["residual_anatomy"]
    lines = [
        "# MM-Lifelong WP15-1 Tie/Loss Anatomy",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "This is a zero-model, zero-judge decomposition of the frozen WP15-0 geometry.",
        "",
        f"Structural gate passed: **{report['structural_gate_passed']}**",
        f"Case W/T/L: **{cases['wins']}/{cases['ties']}/{cases['losses']}**",
        "",
        "## Residual Anatomy",
        "",
        "| Tie class | Count | Rate |",
        "|---|---:|---:|",
    ]
    for tie_class, count in anatomy["tie_class_counts"].items():
        lines.append(
            f"| {tie_class} | {count} | "
            f"{_fmt(anatomy['tie_class_rates'][tie_class])} |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Outcome | Margin | Matched only | Both | Mismatched only | Neither | Residual class |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in cases["cases"]:
        counts = row["support_geometry_counts"]
        lines.append(
            f"| {row['case_id']} | {row['outcome']} | {row['margin']} | "
            f"{counts['matched_only_supported']} | {counts['both_supported']} | "
            f"{counts['mismatched_only_supported']} | "
            f"{counts['neither_supported']} | {row['residual_class'] or ''} |"
        )
    lines.extend(
        [
            "",
            "## Constraint-Type Discriminativeness",
            "",
            "| Type | N | Matched only | Both | Mismatched only | Neither | Ambiguity | Disc.-adverse |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["constraint_type_discriminativeness"]:
        lines.append(
            f"| {row['constraint_type']} | {row['count']} | "
            f"{_fmt(row['matched_only_rate'])} | "
            f"{_fmt(row['both_supported_rate'])} | "
            f"{_fmt(row['mismatched_only_rate'])} | "
            f"{_fmt(row['neither_supported_rate'])} | "
            f"{_fmt(row['ambiguity_rate'])} | "
            f"{_fmt(row['discriminative_minus_adverse_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Branch Rule",
            "",
            (
                "A WP16 branch requires one tie class to cover at least "
                f"{_fmt(report['branch_thresholds']['dominant_tie_share_min'])} "
                "of ties and no more than "
                f"{report['branch_thresholds']['max_loss_cases_for_followup']} loss."
            ),
            "No constraint-type weights were learned. Endpoint values were not validity gates.",
            "",
            "frozen39 is underpowered and repeatedly used for mechanism development. "
            "Day-test140 and Week remain sealed.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{float(value):.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-report", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_visual_tie_anatomy_report(
        _read_json(Path(args.geometry_report)),
        expected_cases=int(args.expected_cases),
        expected_pairs=int(args.expected_pairs),
    )
    _write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    anatomy = report["residual_anatomy"]
    print(
        "VISUAL_TIE_ANATOMY_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"ties={anatomy['tie_class_counts']} "
        f"losses={anatomy['loss_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
