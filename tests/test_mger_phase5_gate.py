from __future__ import annotations

import json

from runs.mger_phase5_gate import (
    TIER2_EXCLUDED_FROM_CROSS_ARM_GATE,
    collect_runtime_root,
    frozen_reproduction_check,
    gate0,
)


def _arm(name: str, *, mean_score: float, observed: float) -> dict[str, object]:
    return {
        "phase5_arm": name,
        "case_count": 10,
        "case_ids": [f"case-{index}" for index in range(10)],
        "root_config_consistent": True,
        "official_judge_config_match": True,
        "judge_parse_complete": True,
        "mean_score": mean_score,
        "observed_case_rate": observed,
        "mean_frames": 55.1 if name == "frozen_baseline" else 0.0,
        "mean_caption_searches": 0.0,
        "mean_requested_acquisitions": 0.0,
        "total_silent_drops": 0,
    }


def test_gate_uses_mean_score_with_half_credit() -> None:
    blind = _arm("blind_prior", mean_score=0.0, observed=0.0)
    frozen = _arm("frozen_baseline", mean_score=0.15, observed=1.0)

    result = gate0(blind, frozen)

    assert result["decision"] == "GO"
    assert result["comparisons"]["mean_score_delta_frozen_minus_blind"] == 0.15


def test_gate_rejects_non_official_judge_model() -> None:
    blind = _arm("blind_prior", mean_score=0.0, observed=0.0)
    frozen = _arm("frozen_baseline", mean_score=0.2, observed=1.0)
    frozen["official_judge_config_match"] = False

    result = gate0(blind, frozen)

    assert result["decision"] == "NO-GO"
    assert "official_judge_config_match" in result["failed_checks"]


def test_tier2_metrics_are_excluded_from_cross_arm_gate() -> None:
    blind = _arm("blind_prior", mean_score=0.0, observed=0.0)
    frozen = _arm("frozen_baseline", mean_score=0.2, observed=1.0)
    blind["reference_valid_rate"] = 1.0
    frozen["reference_valid_rate"] = 0.0
    blind["task_resolution_error_count"] = 0
    frozen["task_resolution_error_count"] = 100

    result = gate0(blind, frozen)

    assert result["decision"] == "GO"
    assert "reference_valid_rate" in TIER2_EXCLUDED_FROM_CROSS_ARM_GATE
    assert "task_resolution_error_count" in TIER2_EXCLUDED_FROM_CROSS_ARM_GATE


def test_runtime_reproduction_check_stops_on_frame_drift(tmp_path) -> None:
    root = tmp_path / "frozen"
    for index in range(10):
        case_dir = root / "cases" / f"case-{index}"
        case_dir.mkdir(parents=True)
        (case_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "phase5_arm": "frozen_baseline",
                    "controller_mode": "frozen_baseline",
                    "controller_evidence_visibility": "none",
                    "measurement_control": "none",
                    "answer_policy": "benchmark_best_effort",
                    "models": {"reasoner": "test"},
                }
            ),
            encoding="utf-8",
        )
        (case_dir / "runtime_summary.json").write_text(
            json.dumps(
                {
                    "case_id": f"case-{index}",
                    "answer_present": True,
                    "runtime_metrics": {
                        "answer_rate": 1.0,
                        "observed_case_rate": 1.0,
                        "visual_frames_inspected": 85.2,
                        "silently_dropped_acquisition_count": 0,
                        "malformed_decision_count": 0,
                        "reasoner_decision_attempt_count": 5,
                    },
                }
            ),
            encoding="utf-8",
        )

    result = frozen_reproduction_check(collect_runtime_root(root))

    assert result["decision"] == "STOP"
    assert result["official_judge_config_status"] == "not_evaluated"
    assert result["measurements"]["mean_frames"] == 85.2
    assert result["thresholds"]["maximum_mean_frames"] == 68.875
    assert result["failed_checks"] == ["frame_reproduction"]
