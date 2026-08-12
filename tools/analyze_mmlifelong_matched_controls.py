#!/usr/bin/env python3
"""Analyze O1.75-forced and O2-center as matched-action controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

ARMS = ("o1.75", "o1.75-forced", "o2", "o2-center")
COMPARISONS = (
    ("o1.75-forced-o1.75", "o1.75-forced", "o1.75"),
    ("o2-center-o2", "o2-center", "o2"),
    ("o2-center-o1.75", "o2-center", "o1.75"),
    ("o2-center-o1.75-forced", "o2-center", "o1.75-forced"),
)


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    bootstrap_samples: int = 10_000,
    seed: int = 20260812,
) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm = str(row["arm"])
        if arm in ARMS:
            by_arm[arm][str(row["case_id"])] = row
    case_sets = {arm: set(by_arm.get(arm, {})) for arm in ARMS}
    common = set.intersection(*(case_sets[arm] for arm in ARMS))
    aligned = all(case_sets[arm] == common for arm in ARMS)

    comparisons = []
    for index, (name, left, right) in enumerate(COMPARISONS):
        differences = [
            float(by_arm[left][case_id]["score"])
            - float(by_arm[right][case_id]["score"])
            for case_id in sorted(common)
            if isinstance(by_arm[left][case_id].get("score"), (int, float))
            and isinstance(by_arm[right][case_id].get("score"), (int, float))
        ]
        low, high = LADDER._bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=seed + index,
        )
        comparisons.append(
            {
                "comparison": name,
                "case_count": len(differences),
                "mean_score_delta": mean(differences) if differences else None,
                "ci95_low": low,
                "ci95_high": high,
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }
        )

    arm_metrics = []
    for arm in ARMS:
        group = tuple(by_arm[arm].values())
        arm_metrics.append(
            {
                "arm": arm,
                "case_count": len(group),
                "mean_score": _optional_mean(row.get("score") for row in group),
                "exact_correct_rate": _optional_mean(
                    float(row["score"] == 1.0)
                    for row in group
                    if isinstance(row.get("score"), (int, float))
                ),
                "clue_visual_recall": _optional_mean(
                    row.get("clue_visual_recall") for row in group
                ),
                "clue_center_visual_recall": _optional_mean(
                    row.get("clue_center_visual_recall") for row in group
                ),
                "anchor_request_recall": _optional_mean(
                    row.get("anchor_request_recall") for row in group
                ),
                "anchor_inspection_recall": _optional_mean(
                    row.get("anchor_inspection_recall") for row in group
                ),
                "anchor_frame_recall": _optional_mean(
                    row.get("anchor_frame_recall") for row in group
                ),
                "execution_fidelity": _aggregate_execution_fidelity(group),
                "exact_frame_execution_fidelity": (
                    _aggregate_exact_execution_fidelity(group)
                ),
                "forced_anchor_count": sum(
                    int(row.get("forced_anchor_count", 0) or 0) for row in group
                ),
                "visual_frames": _optional_mean(
                    row.get("visual_frames") for row in group
                ),
                "visual_window_count": _optional_mean(
                    row.get("visual_window_count") for row in group
                ),
            }
        )

    runtime_checks = {
        "all_arms_present": all(bool(by_arm.get(arm)) for arm in ARMS),
        "case_sets_aligned": aligned,
        "expected_case_count": aligned and len(common) == int(expected_cases),
        "frozen_configs_aligned_except_intervention": _frozen_configs_aligned(
            by_arm,
            common,
        ),
        "natural_caption_retrieval_aligned": _natural_retrieval_aligned(
            by_arm,
            common,
        ),
        "o175_visible_guidance_matched": _o175_visible_guidance_matched(
            by_arm,
            common,
        ),
        "o175_candidate_pools_matched": _candidate_pools_matched(
            by_arm,
            common,
            ("o1.75", "o1.75-forced"),
        ),
        "o2_exact_locators_matched": _candidate_pools_matched(
            by_arm,
            common,
            ("o2", "o2-center"),
        ),
        "o2_center_guidance_valid": _o2_center_guidance_valid(by_arm, common),
        "forced_requested_anchors_executed": all(
            int(by_arm["o1.75-forced"][case_id].get("anchor_requested_count", 0))
            == int(by_arm["o1.75-forced"][case_id].get("anchor_exact_frame_count", 0))
            for case_id in common
        ),
        "forced_anchor_attachments_complete": all(
            int(
                by_arm["o1.75-forced"][case_id].get(
                    "anchor_attachment_failure_count",
                    0,
                )
                or 0
            )
            == 0
            for case_id in common
        ),
    }
    evaluation_checks = {
        "all_evaluated": all(
            isinstance(row.get("score"), (int, float))
            for arm in ARMS
            for row in by_arm[arm].values()
        ),
        "all_judge_responses_parsed": all(
            row.get("parse_status") == "parsed"
            for arm in ARMS
            for row in by_arm[arm].values()
        ),
    }
    checks = {**runtime_checks, **evaluation_checks}
    return {
        "schema_version": "MMLifelongMatchedActionControlReportV1",
        "expected_cases": int(expected_cases),
        "common_case_count": len(common),
        "runtime_gate_passed": all(runtime_checks.values()),
        "runtime_gate_checks": runtime_checks,
        "gate_passed": all(checks.values()),
        "gate_checks": checks,
        "judge_models": sorted(
            {
                str(row["judge_model"])
                for arm in ARMS
                for row in by_arm[arm].values()
                if row.get("judge_model")
            }
        ),
        "arms": arm_metrics,
        "paired_comparisons": comparisons,
    }


def _aggregate_execution_fidelity(rows: Sequence[Mapping[str, Any]]) -> float | None:
    requested = sum(int(row.get("anchor_requested_count", 0) or 0) for row in rows)
    inspected = sum(int(row.get("anchor_inspected_count", 0) or 0) for row in rows)
    return inspected / requested if requested else None


def _aggregate_exact_execution_fidelity(
    rows: Sequence[Mapping[str, Any]],
) -> float | None:
    requested = sum(int(row.get("anchor_requested_count", 0) or 0) for row in rows)
    executed = sum(int(row.get("anchor_exact_frame_count", 0) or 0) for row in rows)
    return executed / requested if requested else None


def _frozen_configs_aligned(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    for case_id in cases:
        configs = []
        for arm in ARMS:
            config = dict(by_arm[arm][case_id].get("frozen_config", {}) or {})
            config.pop("anchor_execution_policy", None)
            configs.append(json.dumps(config, sort_keys=True, separators=(",", ":")))
        if len(set(configs)) != 1:
            return False
    return bool(cases)


def _natural_retrieval_aligned(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    for case_id in cases:
        signatures = []
        for arm in ARMS:
            audit = by_arm[arm][case_id].get("audit")
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


def _candidate_pools_matched(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
    arms: Sequence[str],
) -> bool:
    for case_id in cases:
        signatures = []
        for arm in arms:
            audit = by_arm[arm][case_id].get("audit")
            if not isinstance(audit, Mapping):
                return False
            signatures.append(
                (
                    json.dumps(audit.get("candidate_passage_ids"), sort_keys=True),
                    json.dumps(audit.get("candidate_intervals"), sort_keys=True),
                    audit.get("shuffle_seed_digest"),
                )
            )
        if len(set(signatures)) != 1:
            return False
    return bool(cases)


def _o175_visible_guidance_matched(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    for case_id in cases:
        audits = [by_arm[arm][case_id].get("audit") for arm in ARMS[:2]]
        if not all(isinstance(audit, Mapping) for audit in audits):
            return False
        left, right = audits
        visible_keys = (
            "guidance_type",
            "exact_boundaries_visible",
            "anchor_count",
            "selected_candidate_ranks",
            "selected_candidate_passage_ids",
            "selected_candidate_intervals",
            "point_anchor_candidate_ranks",
            "point_anchor_candidate_passage_ids",
        )
        if any(left.get(key) != right.get(key) for key in visible_keys):
            return False
        if tuple(by_arm["o1.75"][case_id].get("anchor_timestamps_sec", ()) or ()) != tuple(
            by_arm["o1.75-forced"][case_id].get("anchor_timestamps_sec", ()) or ()
        ):
            return False
        if (left.get("anchor_execution_policy") or "agent_controlled") != "agent_controlled":
            return False
        if right.get("anchor_execution_policy") != "force_if_requested":
            return False
    return bool(cases)


def _o2_center_guidance_valid(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    for case_id in cases:
        row = by_arm["o2-center"][case_id]
        audit = row.get("audit")
        if not isinstance(audit, Mapping):
            return False
        clue_count = int(row.get("clue_count", 0) or 0)
        if audit.get("guidance_type") != "exact_locators_with_point_anchors":
            return False
        if audit.get("exact_boundaries_visible") is not True:
            return False
        if int(audit.get("exact_locator_count", -1)) != clue_count:
            return False
        if int(audit.get("anchor_count", -1)) != clue_count:
            return False
        intervals = tuple(audit.get("candidate_intervals", ()) or ())
        anchors = tuple(audit.get("anchor_timestamps_sec", ()) or ())
        ranks = tuple(audit.get("point_anchor_candidate_ranks", ()) or ())
        if (
            len(intervals) != clue_count
            or len(anchors) != clue_count
            or len(ranks) != clue_count
        ):
            return False
        if any(
            int(rank) < 1
            or int(rank) > len(intervals)
            or abs(
                float(anchor)
                - (
                    float(intervals[int(rank) - 1][0])
                    + float(intervals[int(rank) - 1][1])
                )
                / 2.0
            )
            > 0.001
            for anchor, rank in zip(anchors, ranks)
        ):
            return False
    return bool(cases)


def _optional_mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "| Arm | N | Mean | Exact | Clue overlap | Clue center | Anchor request | Anchor inspect | Anchor exact frame | EF | Exact-frame EF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["arms"]:
        values = (
            row["arm"], row["case_count"], row["mean_score"],
            row["exact_correct_rate"], row["clue_visual_recall"],
            row["clue_center_visual_recall"], row["anchor_request_recall"],
            row["anchor_inspection_recall"], row["anchor_frame_recall"],
            row["execution_fidelity"], row["exact_frame_execution_fidelity"],
        )
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(("", "| Comparison | N | Delta | 95% CI | W/T/L |", "| --- | ---: | ---: | ---: | ---: |"))
    for row in report["paired_comparisons"]:
        lines.append(
            f"| {row['comparison']} | {row['case_count']} | {_fmt(row['mean_score_delta'])} | "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    rows = LADDER.collect_rows(
        tuple(Path(value) for value in args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
        case_ids=(frozenset(args.case_id) if args.case_id else None),
    )
    report = build_report(
        rows,
        expected_cases=args.expected_cases,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    LADDER._write_text(Path(args.out_json), json.dumps(report, indent=2, sort_keys=True) + "\n")
    LADDER._write_text(Path(args.out_md), render_markdown(report))
    print(json.dumps({
        "common_case_count": report["common_case_count"],
        "gate_passed": report["gate_passed"],
        "runtime_gate_passed": report["runtime_gate_passed"],
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
