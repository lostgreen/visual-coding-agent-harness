#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


ROOT_CONFIG_KEYS = (
    "phase5_arm",
    "controller_mode",
    "controller_evidence_visibility",
    "measurement_control",
    "answer_policy",
    "models",
)
TIER2_EXCLUDED_FROM_CROSS_ARM_GATE = (
    "decision_repair_count",
    "task_resolution_error_count",
    "state_mutation_op_count",
    "reference_valid_rate",
    "reference_integrity_ok",
    "material_support_ok",
    "occurrence_binding_ok",
    "obligation_coverage_ok",
    "prompt_schema_token_cost",
)


def collect_root(root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_dir in sorted((Path(root) / "cases").glob("*")):
        if not case_dir.is_dir():
            continue
        evaluation_path = case_dir / "evaluation" / "mmlifelong_eval.json"
        config_path = case_dir / "run_config.json"
        runtime_path = case_dir / "runtime_summary.json"
        if not (evaluation_path.is_file() and config_path.is_file() and runtime_path.is_file()):
            continue
        evaluation = _read_json(evaluation_path)
        config = _read_json(config_path)
        runtime = _read_json(runtime_path)
        answer = _mapping(evaluation.get("answer"))
        grounding = _mapping(evaluation.get("reference_grounding"))
        metrics = _mapping(runtime.get("runtime_metrics"))
        score = answer.get("score")
        cases.append(
            {
                "case_id": str(evaluation.get("case_id", case_dir.name)),
                "score": float(score) if isinstance(score, (int, float)) else None,
                "parse_status": str(answer.get("parse_status", "")),
                "judge_model": str(answer.get("judge_model", "")),
                "official_judge_config_match": bool(
                    answer.get("official_judge_config_match", False)
                ),
                "answer_rate": _number(metrics.get("answer_rate")),
                "observed_case_rate": _number(
                    metrics.get(
                        "observed_case_rate",
                        float(_number(metrics.get("visual_interpretation_count")) > 0),
                    )
                ),
                "visual_frames": _number(metrics.get("visual_frames_inspected")),
                "caption_searches": _number(metrics.get("caption_searches")),
                "requested_acquisitions": _number(
                    metrics.get("requested_acquisition_count")
                ),
                "silent_drops": _number(
                    metrics.get("silently_dropped_acquisition_count")
                ),
                "malformed_count": _number(metrics.get("malformed_decision_count")),
                "decision_attempt_count": _number(
                    metrics.get("reasoner_decision_attempt_count")
                ),
                "ref_60": _ref_score(grounding, 60),
                "ref_300": _ref_score(grounding, 300),
                "ref_600": _ref_score(grounding, 600),
                "config": config,
            }
        )
    return _aggregate(Path(root), cases)


def gate0(
    blind: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    expected_case_count: int = 10,
    minimum_score_delta: float = 0.15,
    frozen_reference_frames: float = 55.1,
    frozen_frame_tolerance_ratio: float = 0.25,
) -> dict[str, Any]:
    score_delta = _number(frozen.get("mean_score")) - _number(
        blind.get("mean_score")
    )
    frame_delta = abs(
        _number(frozen.get("mean_frames")) - float(frozen_reference_frames)
    )
    checks = {
        "blind_arm_config": blind.get("phase5_arm") == "blind_prior",
        "frozen_arm_config": frozen.get("phase5_arm") == "frozen_baseline",
        "case_count": (
            int(blind.get("case_count", 0)) == expected_case_count
            and int(frozen.get("case_count", 0)) == expected_case_count
        ),
        "paired_case_ids": blind.get("case_ids") == frozen.get("case_ids"),
        "root_config_consistency": bool(blind.get("root_config_consistent"))
        and bool(frozen.get("root_config_consistent")),
        "official_judge_config_match": bool(
            blind.get("official_judge_config_match")
        )
        and bool(frozen.get("official_judge_config_match")),
        "judge_parse_complete": bool(blind.get("judge_parse_complete"))
        and bool(frozen.get("judge_parse_complete")),
        "blind_has_no_tool_use": (
            _number(blind.get("observed_case_rate")) == 0.0
            and _number(blind.get("mean_caption_searches")) == 0.0
            and _number(blind.get("mean_requested_acquisitions")) == 0.0
        ),
        "frozen_observed_case_rate": _number(frozen.get("observed_case_rate"))
        == 1.0,
        "frozen_silent_drops": _number(frozen.get("total_silent_drops")) == 0.0,
        "frozen_frame_reproduction": frame_delta
        <= float(frozen_reference_frames) * float(frozen_frame_tolerance_ratio),
        "measurement_score_delta": score_delta >= float(minimum_score_delta),
    }
    return {
        "schema_version": "MGERPhase5Gate0V1",
        "stage": "gate0_measurement_validity",
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "screening_only": True,
        "statistical_significance_claim": False,
        "tier2_metrics_excluded": list(TIER2_EXCLUDED_FROM_CROSS_ARM_GATE),
        "thresholds": {
            "expected_case_count": expected_case_count,
            "minimum_score_delta": minimum_score_delta,
            "frozen_reference_frames": frozen_reference_frames,
            "frozen_frame_tolerance_ratio": frozen_frame_tolerance_ratio,
        },
        "comparisons": {
            "mean_score_delta_frozen_minus_blind": score_delta,
            "frozen_frame_absolute_delta": frame_delta,
        },
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "arms": {"blind_prior": dict(blind), "frozen_baseline": dict(frozen)},
    }


def _aggregate(root: Path, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    configs = [dict(case.get("config", {})) for case in cases]
    config_values = {
        key: {_stable_value(config.get(key)) for config in configs}
        for key in ROOT_CONFIG_KEYS
    }
    scores = [case["score"] for case in cases if case.get("score") is not None]
    observed = [case["visual_frames"] for case in cases if case["observed_case_rate"] > 0]
    malformed = sum(case["malformed_count"] for case in cases)
    attempts = sum(case["decision_attempt_count"] for case in cases)
    phase5_arms = {str(config.get("phase5_arm", "")) for config in configs}
    return {
        "root": str(root),
        "phase5_arm": next(iter(phase5_arms)) if len(phase5_arms) == 1 else "mixed",
        "case_count": len(cases),
        "case_ids": sorted(str(case["case_id"]) for case in cases),
        "root_config_consistent": all(len(values) == 1 for values in config_values.values()),
        "config_mismatches": sorted(
            key for key, values in config_values.items() if len(values) != 1
        ),
        "official_judge_config_match": bool(cases)
        and all(case["official_judge_config_match"] for case in cases),
        "judge_parse_complete": bool(cases)
        and all(case["parse_status"] == "parsed" for case in cases),
        "judge_models": sorted({str(case["judge_model"]) for case in cases}),
        "mean_score": mean(scores) if scores else None,
        "exact_correct_rate": (
            mean(float(score == 1.0) for score in scores) if scores else None
        ),
        "answer_rate": _mean(case["answer_rate"] for case in cases),
        "observed_case_rate": _mean(case["observed_case_rate"] for case in cases),
        "mean_frames": _mean(case["visual_frames"] for case in cases),
        "conditional_mean_frames": mean(observed) if observed else None,
        "mean_caption_searches": _mean(case["caption_searches"] for case in cases),
        "mean_requested_acquisitions": _mean(
            case["requested_acquisitions"] for case in cases
        ),
        "total_silent_drops": sum(case["silent_drops"] for case in cases),
        "malformed_decision_rate": malformed / attempts if attempts else 0.0,
        "ref_60": _mean(case["ref_60"] for case in cases),
        "ref_300": _mean(case["ref_300"] for case in cases),
        "ref_600": _mean(case["ref_600"] for case in cases),
        "scores": {str(case["case_id"]): case["score"] for case in cases},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the MM-Lifelong Phase 5 measurement-validity gate."
    )
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=10)
    parser.add_argument("--minimum-score-delta", type=float, default=0.15)
    parser.add_argument("--frozen-reference-frames", type=float, default=55.1)
    parser.add_argument("--frozen-frame-tolerance-ratio", type=float, default=0.25)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = gate0(
        collect_root(args.blind_root),
        collect_root(args.frozen_root),
        expected_case_count=args.expected_case_count,
        minimum_score_delta=args.minimum_score_delta,
        frozen_reference_frames=args.frozen_reference_frames,
        frozen_frame_tolerance_ratio=args.frozen_frame_tolerance_ratio,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["decision"] == "GO" else 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _mean(values: Sequence[float] | Any) -> float | None:
    rows = [float(value) for value in values]
    return mean(rows) if rows else None


def _ref_score(grounding: Mapping[str, Any], bucket: int) -> float:
    value = grounding.get(f"Ref@{bucket}", grounding.get(f"ref_{bucket}", 0.0))
    return _number(value)


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
