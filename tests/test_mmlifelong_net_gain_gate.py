from __future__ import annotations

from runs.mmlifelong_net_gain_gate import evaluate_net_gain, evaluate_run_pair


def _case(
    *,
    accuracy: float,
    reference_valid: float = 1.0,
    frames: float = 10.0,
    coverage: float = 1.0,
    anchor: float = 1.0,
) -> dict[str, float]:
    return {
        "accuracy_score": accuracy,
        "reference_valid": reference_valid,
        "visual_frames_inspected": frames,
        "clue_frame_coverage": coverage,
        "retrieval_dedup_rate": 0.0,
        "sampling_fidelity_mean": 1.0,
        "sampling_fidelity_min": 1.0,
        "anchor_consistency": anchor,
    }


def test_net_gain_requires_two_passing_independent_repeats() -> None:
    baseline = {
        "case-a": _case(accuracy=0.0),
        "case-b": _case(accuracy=1.0),
    }
    candidate = {
        "case-a": _case(accuracy=1.0, frames=12.0),
        "case-b": _case(accuracy=1.0, frames=12.0),
    }
    pair = evaluate_run_pair(
        baseline,
        candidate,
        label="repeat-1",
        run_identity="run-1",
    )

    assert pair["passed"] is True
    assert pair["accuracy_delta"] == 0.5
    assert pair["visual_frame_ratio"] == 1.2
    single = evaluate_net_gain((pair,))
    assert single["passed"] is False
    assert "insufficient_independent_repeats:1<2" in single["failures"]

    duplicate = evaluate_net_gain((pair, pair))
    assert duplicate["passed"] is False
    assert duplicate["submitted_pair_count"] == 2
    assert duplicate["independent_repeat_count"] == 1

    repeated = evaluate_net_gain(
        (pair, {**pair, "label": "repeat-2", "run_identity": "run-2"})
    )
    assert repeated["passed"] is True


def test_net_gain_reports_reference_and_cost_regressions() -> None:
    baseline = {"case-a": _case(accuracy=0.0, frames=10.0)}
    candidate = {
        "case-a": _case(
            accuracy=1.0,
            reference_valid=0.0,
            frames=13.0,
        )
    }

    pair = evaluate_run_pair(baseline, candidate, label="regressed")

    assert pair["passed"] is False
    assert "reference_non_regression" in pair["failures"]
    assert "frame_cost_within_limit" in pair["failures"]
    result = evaluate_net_gain((pair, pair))
    assert result["passed"] is False
    assert "paired_repeat_failed:regressed" in result["failures"]


def test_pair_gate_rejects_case_set_drift() -> None:
    pair = evaluate_run_pair(
        {"case-a": _case(accuracy=0.0), "case-b": _case(accuracy=0.0)},
        {"case-a": _case(accuracy=1.0)},
        label="missing-case",
    )

    assert pair["passed"] is False
    assert "missing_candidate_cases:case-b" in pair["failures"]
