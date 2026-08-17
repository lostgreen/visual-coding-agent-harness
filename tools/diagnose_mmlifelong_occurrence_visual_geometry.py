#!/usr/bin/env python3
"""Run the zero-model WP15-0 paired visual evidence diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_visual_geometry import build_visual_geometry_report


def render_markdown(report: Mapping[str, Any]) -> str:
    geometry = report["paired_support_geometry"]
    joint = report["case_level_joint_evidence"]
    coverage = report["matched_frame_coverage"]
    lines = [
        "# MM-Lifelong WP15-0 Visual Evidence Geometry",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "This is a zero-model, zero-judge diagnosis over the frozen WP14 verdicts.",
        "",
        f"Structural gate passed: **{report['structural_gate_passed']}**",
        "",
        "## Paired Support Geometry",
        "",
        "| Geometry | Count | Rate |",
        "|---|---:|---:|",
    ]
    for category, count in geometry["counts"].items():
        lines.append(
            f"| {category} | {count} | {_fmt(geometry['rates'][category])} |"
        )
    lines.extend(
        [
            "",
            "## Case-Level Joint Evidence",
            "",
            (
                f"Matched versus mismatched supported-count W/T/L: "
                f"**{joint['wins']}/{joint['ties']}/{joint['losses']}** "
                f"across {joint['case_count']} cases."
            ),
            f"Margin distribution: `{joint['margin_distribution']}`",
            "",
            "## Matched Frame Coverage",
            "",
            "| Stratum | Cases | Constraints | Supported rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in ("covered", "uncovered"):
        row = coverage[label]
        lines.append(
            f"| {label} | {row['case_count']} | {row['constraint_count']} | "
            f"{_fmt(row['matched_supported_rate'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Covered minus uncovered support-rate gap: "
                f"**{_fmt(coverage['covered_minus_uncovered_supported_rate'])}**."
            ),
            "",
            "## Frozen Branch Rule",
            "",
            (
                "Comparative probing requires at least "
                f"{report['branch_thresholds']['joint_min_wins']} case wins and at "
                f"most {report['branch_thresholds']['joint_max_losses']} loss."
            ),
            (
                "Sampling repair requires at least "
                f"{report['branch_thresholds']['coverage_min_cases_per_stratum']} "
                "cases in each coverage stratum and a covered-minus-uncovered support "
                f"gap of at least "
                f"{_fmt(report['branch_thresholds']['coverage_min_support_gap'])}."
            ),
            "",
            "WP14's 40pp unary gap threshold is unchanged. Candidate support, "
            "candidate sufficiency, and discriminative evidence remain distinct.",
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


def _read_results(root: Path) -> tuple[dict[str, Any], ...]:
    return tuple(_read_json(path) for path in sorted(Path(root).glob("items/*.json")))


def _evaluation_records(
    root: Path, case_ids: set[str]
) -> dict[str, dict[str, Any]]:
    return {
        case_id: _read_json(Path(root) / case_id / "evaluation_case.json")
        for case_id in sorted(case_ids)
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    probe_root = Path(args.probe_root)
    manifest = _read_json(probe_root / "probe_manifest.json")
    case_ids = {
        str(row.get("case_id", "") or "")
        for row in tuple(manifest.get("cases", ()) or ())
        if isinstance(row, Mapping) and bool(row.get("eligible"))
    }
    report = build_visual_geometry_report(
        manifest,
        _read_results(Path(args.result_root)),
        _evaluation_records(Path(args.evaluation_record_root), case_ids),
        expected_cases=int(args.expected_cases),
        expected_items=int(args.expected_items),
    )
    _write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        "VISUAL_GEOMETRY_DONE "
        f"decision={report['decision']} "
        f"structural={report['structural_gate_passed']} "
        f"wtl={report['case_level_joint_evidence']['wins']}/"
        f"{report['case_level_joint_evidence']['ties']}/"
        f"{report['case_level_joint_evidence']['losses']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
