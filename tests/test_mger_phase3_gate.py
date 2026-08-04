from __future__ import annotations

from runs.mger_phase3_gate import (
    evaluate_phase3,
    evaluate_root,
    reliability_category,
)


def _case(
    case_id: str,
    *,
    score: float = 0.0,
    reference_valid: bool = False,
    answer_present: bool = True,
    frames: int = 10,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "score": score,
        "parse_status": "parsed",
        "judge_model": "gpt-5.5",
        "answer_present": answer_present,
        "reference_valid": reference_valid,
        "visual_frames_inspected": frames,
        "silently_dropped_acquisition_count": 0,
        "decision_repair_count": 0,
    }


def _passing_cases() -> dict[str, dict[str, object]]:
    cases = {
        f"mmlifelong-game-test-{index:04d}": _case(
            f"mmlifelong-game-test-{index:04d}"
        )
        for index in range(1, 11)
    }
    cases["mmlifelong-game-test-0001"].update(score=1.0, reference_valid=True)
    cases["mmlifelong-game-test-0002"].update(score=1.0, reference_valid=True)
    cases["mmlifelong-game-test-0003"].update(score=1.0, reference_valid=True)
    cases["mmlifelong-game-test-0117"] = _case(
        "mmlifelong-game-test-0117",
        score=1.0,
        reference_valid=True,
        frames=40,
    )
    cases.pop("mmlifelong-game-test-0010")
    return cases


def test_reliability_categories_are_mutually_exclusive() -> None:
    assert reliability_category(_case("a", score=1.0, reference_valid=True)) == "GroundedCorrect"
    assert reliability_category(_case("a", score=0.0, reference_valid=True)) == "WrongButVerified"
    assert reliability_category(_case("a", score=1.0)) == "CorrectButUngrounded"
    assert reliability_category(_case("a", answer_present=False)) == "MissingAnswer"


def test_phase3_gate_requires_two_independent_passing_roots() -> None:
    root_1 = evaluate_root(
        _passing_cases(),
        label="r1",
        run_identity="run-1",
        expected_judge_model="gpt-5.5",
    )
    root_2 = evaluate_root(
        _passing_cases(),
        label="r2",
        run_identity="run-2",
        expected_judge_model="gpt-5.5",
    )

    assert root_1["passed"] is True
    assert root_1["categories"]["GroundedCorrect"] == 4
    assert evaluate_phase3((root_1,))["passed"] is False
    paired = evaluate_phase3((root_1, root_2))
    assert paired["passed"] is True
    assert paired["decision"] == "GO"


def test_phase3_gate_reports_reliability_and_cost_failures() -> None:
    cases = _passing_cases()
    for case in cases.values():
        case["visual_frames_inspected"] = 100
    for case_id in (
        "mmlifelong-game-test-0001",
        "mmlifelong-game-test-0002",
        "mmlifelong-game-test-0003",
    ):
        cases[case_id].update(score=0.0, reference_valid=True)
    cases["mmlifelong-game-test-0004"]["answer_present"] = False
    cases["mmlifelong-game-test-0005"]["answer_present"] = False

    report = evaluate_root(cases, label="failed")

    assert report["passed"] is False
    assert report["categories"]["WrongButVerified"] == 3
    assert "wrong_but_verified_at_most_1" in report["failures"]
    assert "grounded_correct_above_2" in report["failures"]
    assert "answer_rate_at_least_0_9" in report["failures"]
    assert "mean_frames_within_limit" in report["failures"]
