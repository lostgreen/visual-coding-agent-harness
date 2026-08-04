from __future__ import annotations

from runs.mger_phase4_gate import evaluate_phase4, evaluate_root


def _cases(
    *,
    grounded_correct: int = 1,
    wrong_but_verified: int = 0,
    answer_count: int = 10,
    decision_repairs: int = 5,
    task_errors: int = 2,
    control_mode: str = "shadow",
) -> dict[str, dict[str, object]]:
    case_ids = [f"case-{index:04d}" for index in range(9)] + ["case-0117"]
    rows: dict[str, dict[str, object]] = {}
    for index, case_id in enumerate(case_ids):
        answer_present = index < answer_count
        score = 1.0 if index < grounded_correct else 0.0
        reference_valid = index < grounded_correct or (
            grounded_correct <= index < grounded_correct + wrong_but_verified
        )
        if case_id == "case-0117":
            answer_present = True
            score = 1.0
        rows[case_id] = {
            "case_id": case_id,
            "score": score,
            "parse_status": "parsed",
            "judge_model": "gpt-5.5",
            "answer_present": answer_present,
            "reference_valid": reference_valid,
            "evidence_control_mode": control_mode,
            "visual_frames_inspected": 55,
            "silently_dropped_acquisition_count": 0,
            "decision_repair_count": 1 if index < decision_repairs else 0,
            "task_resolution_error_count": 1 if index < task_errors else 0,
            "state_mutation_op_count": 0,
            "prompt_schema_token_cost": 0,
        }
    return rows


def test_first_gate_accepts_protocol_stable_shadow_root() -> None:
    report = evaluate_root(
        _cases(),
        label="root-1",
        stage="first",
        expected_judge_model="gpt-5.5",
    )

    assert report["passed"] is True
    assert report["answer_rate"] == 1.0
    assert report["decision_repairs"] == 5
    assert report["task_resolution_errors"] == 2


def test_first_gate_reports_protocol_and_regression_failures() -> None:
    cases = _cases(answer_count=7, decision_repairs=6, task_errors=3)
    cases["case-0117"]["score"] = 0.0

    report = evaluate_root(cases, label="root-1", stage="first")

    assert report["passed"] is False
    assert "answer_rate_at_least_0_9" in report["failures"]
    assert "decision_repairs_at_most_5" in report["failures"]
    assert "task_resolution_errors_at_most_2" in report["failures"]
    assert "case_0117_no_regression" in report["failures"]


def test_strict_gate_uses_second_round_reliability_targets() -> None:
    report = evaluate_root(
        _cases(
            grounded_correct=2,
            wrong_but_verified=2,
            answer_count=8,
            control_mode="strict",
        ),
        label="strict-root",
        stage="strict",
    )

    assert report["passed"] is True
    assert report["categories"]["GroundedCorrect"] >= 2
    assert report["categories"]["WrongButVerified"] <= 2


def test_phase4_gate_requires_two_independent_passing_roots() -> None:
    root_1 = evaluate_root(_cases(), label="root-1", run_identity="run-1")
    root_2 = evaluate_root(_cases(), label="root-2", run_identity="run-2")

    assert evaluate_phase4((root_1,))["passed"] is False
    paired = evaluate_phase4((root_1, root_2))
    assert paired["passed"] is True
    assert paired["independent_root_count"] == 2
