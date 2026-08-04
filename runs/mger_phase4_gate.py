#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


EXPECTED_CASE_COUNT = 10
FROZEN_BASELINE_MEAN_FRAMES = 55.1
MAX_MEAN_FRAMES = FROZEN_BASELINE_MEAN_FRAMES * 1.25
REGRESSION_CASE_SUFFIX = "0117"
JUDGE_PARSE_STATUS = "parsed"


def collect_cases(root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for evaluation_path in sorted(Path(root).rglob("mmlifelong_eval.json")):
        case_root = evaluation_path.parent.parent
        evaluation = _read_json(evaluation_path)
        runtime = _read_json(case_root / "runtime_summary.json")
        config = _read_json(case_root / "run_config.json")
        case_id = str(evaluation.get("case_id", case_root.name) or case_root.name)
        if case_id in cases:
            raise ValueError(f"duplicate case_id under one run root: {case_id}")
        answer = evaluation.get("answer", {})
        answer_metrics = answer if isinstance(answer, Mapping) else {}
        runtime_metrics = runtime.get("runtime_metrics", {})
        metrics = runtime_metrics if isinstance(runtime_metrics, Mapping) else {}
        score = answer_metrics.get("score")
        cases[case_id] = {
            "case_id": case_id,
            "score": float(score) if isinstance(score, (int, float)) else None,
            "parse_status": str(answer_metrics.get("parse_status", "") or ""),
            "judge_model": str(answer_metrics.get("judge_model", "") or ""),
            "answer_present": bool(runtime.get("answer_present")),
            "reference_valid": bool(runtime.get("reference_valid")),
            "evidence_control_mode": str(
                config.get("evidence_control_mode", "strict") or "strict"
            ),
            "evidence_state_mode": str(
                config.get("evidence_state_mode", "llm_authored") or "llm_authored"
            ),
            "visual_frames_inspected": int(
                metrics.get("visual_frames_inspected", 0) or 0
            ),
            "silently_dropped_acquisition_count": int(
                metrics.get("silently_dropped_acquisition_count", 0) or 0
            ),
            "decision_repair_count": int(
                metrics.get("decision_repair_count", 0) or 0
            ),
            "task_resolution_error_count": int(
                metrics.get("task_resolution_error_count", 0) or 0
            ),
            "state_mutation_op_count": int(
                metrics.get("state_mutation_op_count", 0) or 0
            ),
            "prompt_schema_token_cost": int(
                metrics.get("prompt_schema_token_cost", 0) or 0
            ),
        }
    if not cases:
        raise ValueError(f"no evaluated MM-Lifelong cases found under {root}")
    return cases


def reliability_category(case: Mapping[str, Any]) -> str:
    if not bool(case.get("answer_present")):
        return "MissingAnswer"
    correct = case.get("score") == 1.0
    reference_valid = bool(case.get("reference_valid"))
    if correct and reference_valid:
        return "GroundedCorrect"
    if not correct and reference_valid:
        return "WrongButVerified"
    if correct:
        return "CorrectButUngrounded"
    return "WrongAndUngrounded"


def evaluate_root(
    cases: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
    stage: str = "first",
    run_identity: str = "",
    expected_case_count: int = EXPECTED_CASE_COUNT,
    max_mean_frames: float = MAX_MEAN_FRAMES,
    expected_judge_model: str = "",
) -> dict[str, Any]:
    normalized_stage = str(stage or "first").strip().casefold()
    if normalized_stage not in {"first", "strict"}:
        raise ValueError(f"unsupported Phase 4 gate stage: {normalized_stage}")
    rows = tuple(dict(case) for _, case in sorted(cases.items()))
    categories = {
        name: sum(reliability_category(case) == name for case in rows)
        for name in (
            "GroundedCorrect",
            "WrongButVerified",
            "CorrectButUngrounded",
            "WrongAndUngrounded",
            "MissingAnswer",
        )
    }
    case_count = len(rows)
    answer_rate = mean(float(bool(case.get("answer_present"))) for case in rows)
    mean_frames = mean(
        float(case.get("visual_frames_inspected", 0) or 0) for case in rows
    )
    silent_drops = _sum(rows, "silently_dropped_acquisition_count")
    decision_repairs = _sum(rows, "decision_repair_count")
    task_resolution_errors = _sum(rows, "task_resolution_error_count")
    parse_failures = sorted(
        str(case.get("case_id", ""))
        for case in rows
        if case.get("parse_status") != JUDGE_PARSE_STATUS
    )
    judge_mismatches = sorted(
        str(case.get("case_id", ""))
        for case in rows
        if expected_judge_model
        and str(case.get("judge_model", "")) != expected_judge_model
    )
    expected_control_mode = "strict" if normalized_stage == "strict" else "shadow"
    control_mode_mismatches = sorted(
        str(case.get("case_id", ""))
        for case in rows
        if str(case.get("evidence_control_mode", "")) != expected_control_mode
    )
    regression = next(
        (
            case
            for case in rows
            if str(case.get("case_id", "")).endswith(REGRESSION_CASE_SUFFIX)
        ),
        None,
    )
    checks = {
        "case_count_is_10": case_count == int(expected_case_count),
        "judge_parse_complete": not parse_failures,
        "judge_model_consistent": not judge_mismatches,
        "evidence_control_mode_consistent": not control_mode_mismatches,
        "mean_frames_within_limit": mean_frames <= float(max_mean_frames),
        "silent_drops_zero": silent_drops == 0,
    }
    if normalized_stage == "first":
        checks.update(
            {
                "answer_rate_at_least_0_9": answer_rate >= 0.9,
                "decision_repairs_at_most_5": decision_repairs <= 5,
                "task_resolution_errors_at_most_2": task_resolution_errors <= 2,
                "case_0117_no_regression": bool(
                    regression
                    and regression.get("answer_present")
                    and regression.get("score") == 1.0
                ),
                "wrong_but_verified_at_most_4": categories["WrongButVerified"] <= 4,
                "grounded_correct_at_least_1": categories["GroundedCorrect"] >= 1,
            }
        )
    else:
        checks.update(
            {
                "answer_rate_at_least_0_8": answer_rate >= 0.8,
                "wrong_but_verified_at_most_2": categories["WrongButVerified"] <= 2,
                "grounded_correct_above_baseline": categories["GroundedCorrect"] > 1,
            }
        )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "label": str(label),
        "stage": normalized_stage,
        "run_identity": str(run_identity or label),
        "case_count": case_count,
        "categories": categories,
        "answer_rate": answer_rate,
        "mean_frames": mean_frames,
        "silent_drops": silent_drops,
        "decision_repairs": decision_repairs,
        "task_resolution_errors": task_resolution_errors,
        "state_mutation_ops": _sum(rows, "state_mutation_op_count"),
        "prompt_schema_token_cost": _sum(rows, "prompt_schema_token_cost"),
        "parse_failures": parse_failures,
        "judge_model_mismatches": judge_mismatches,
        "evidence_control_mode_mismatches": control_mode_mismatches,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "cases": [
            {**case, "reliability_category": reliability_category(case)}
            for case in rows
        ],
    }


def evaluate_phase4(
    roots: Sequence[Mapping[str, Any]],
    *,
    stage: str = "first",
    min_independent_roots: int = 2,
) -> dict[str, Any]:
    reports = tuple(dict(report) for report in roots)
    identities = {
        str(report.get("run_identity") or report.get("label") or index)
        for index, report in enumerate(reports, start=1)
    }
    required = max(2, int(min_independent_roots))
    failures = []
    if len(identities) < required:
        failures.append(f"insufficient_independent_roots:{len(identities)}<{required}")
    failures.extend(
        f"root_failed:{report.get('label', index)}"
        for index, report in enumerate(reports, start=1)
        if not bool(report.get("passed"))
    )
    return {
        "schema_version": "MGERPhase4GateV1",
        "stage": str(stage),
        "passed": not failures,
        "decision": "GO" if not failures else "NO-GO",
        "failures": failures,
        "independent_root_count": len(identities),
        "required_independent_roots": required,
        "roots": list(reports),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the MGER Phase 4 protocol, reliability, and cost gates."
    )
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--stage", choices=("first", "strict"), default="first")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--max-mean-frames", type=float, default=MAX_MEAN_FRAMES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    reports = []
    for index, value in enumerate(args.run_root, start=1):
        root = Path(value).resolve()
        reports.append(
            evaluate_root(
                collect_cases(root),
                label=f"root_{index}",
                stage=args.stage,
                run_identity=str(root),
                max_mean_frames=args.max_mean_frames,
                expected_judge_model=args.judge_model,
            )
        )
    result = evaluate_phase4(reports, stage=args.stage)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(int(case.get(key, 0) or 0) for case in rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


if __name__ == "__main__":
    raise SystemExit(main())
