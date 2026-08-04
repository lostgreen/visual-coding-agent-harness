from __future__ import annotations

from runs.mger_phase5_gate import TIER2_EXCLUDED_FROM_CROSS_ARM_GATE, gate0


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
