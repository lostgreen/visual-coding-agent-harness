from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analyze_mmlifelong_occurrence_agent.py"
)
SPEC = importlib.util.spec_from_file_location("occurrence_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)

AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mmlifelong_occurrence_canary.py"
)
AUDIT_SPEC = importlib.util.spec_from_file_location("occurrence_audit", AUDIT_PATH)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def _row(
    arm: str,
    case_id: str,
    *,
    score: float,
    signature: list[dict],
    digest: str = "digest",
) -> dict:
    return {
        "arm": arm,
        "case_id": case_id,
        "score": score,
        "exact_correct": score == 1.0,
        "verified_correct": False,
        "correct_and_ref_300": score == 1.0,
        "parse_status": "parsed",
        "judge_model": "judge",
        "candidate_recall": True,
        "osa_eligible": arm == "a2",
        "osa_correct": arm == "a2",
        "occurrence_handle_usage_rate": 1.0,
        "premature_occurrence_commit": False,
        "visual_frames": 3,
        "visual_windows": 1,
        "vlm_calls": 1,
        "pre_treatment_signature": signature,
        "treatment_exposure": {
            "visible_excerpt_digest": digest,
            "visible_text_digest": digest,
            "visible_excerpt_count": 2,
        },
        "no_oracle_gate_passed": True,
        "frozen_config": {"controller_mode": "frozen_baseline"},
    }


def test_report_separates_pre_treatment_divergence_and_a2_osa() -> None:
    same = [{"action": "investigate", "tasks": []}]
    different = [{"action": "update_workspace", "tasks": []}]
    rows = (
        _row("a0", "c1", score=0.0, signature=same),
        _row("a0", "c2", score=0.0, signature=same),
        _row("a1", "c1", score=1.0, signature=same),
        _row("a1", "c2", score=1.0, signature=different),
        _row("a1-flat", "c1", score=0.0, signature=same),
        _row("a1-flat", "c2", score=0.0, signature=different),
        _row("a2", "c1", score=1.0, signature=same),
        _row("a2", "c2", score=1.0, signature=same),
    )

    report = ANALYSIS.build_report(
        rows, expected_cases=2, bootstrap_samples=100, seed=7
    )

    assert report["comparisons"]["a1-a0"][
        "pre_treatment_divergence_rate"
    ] == 0.5
    assert report["comparisons"]["a1-a0"][
        "matched_pre_treatment_subset"
    ]["paired_n"] == 1
    assert report["arms"]["a1"]["occurrence_selection_accuracy"] is None
    assert report["arms"]["a2"]["occurrence_selection_accuracy"] == 1.0
    assert report["text_budget_parity"]["comparable_n"] == 2
    assert report["text_budget_parity"]["passed"] is True
    assert report["structural_gate_passed"] is True


def test_a1_flat_text_mismatch_fails_structural_gate() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    rows = (
        _row("a1", "c1", score=0.0, signature=signature, digest="left"),
        _row(
            "a1-flat", "c1", score=0.0, signature=signature, digest="right"
        ),
    )

    report = ANALYSIS.build_report(
        rows, expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["text_budget_parity"]["passed"] is False
    assert report["structural_gate_passed"] is False


def test_a1_flat_text_parity_is_not_comparable_after_trajectory_divergence() -> None:
    rows = (
        _row(
            "a1",
            "c1",
            score=0.0,
            signature=[{"action": "investigate", "tasks": []}],
            digest="left",
        ),
        _row(
            "a1-flat",
            "c1",
            score=0.0,
            signature=[{"action": "answer", "tasks": []}],
            digest="right",
        ),
    )

    report = ANALYSIS.build_report(
        rows, expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["text_budget_parity"]["passed"] is None
    assert report["text_budget_parity"]["status"] == "not_comparable"
    assert report["structural_gate_passed"] is True


def test_canary_audit_checks_structured_protocol_artifacts(tmp_path) -> None:
    bindings = {}
    for arm in ("a0", "a1-flat", "a1", "a2"):
        root = tmp_path / arm
        case = root / "cases" / "case-1"
        case.mkdir(parents=True)
        (case / "prediction.json").write_text(
            json.dumps({"case_id": "case-1"}), encoding="utf-8"
        )
        (case / "run_config.json").write_text(
            json.dumps(
                {
                    "occurrence_method_arm": arm,
                    "models": {
                        "reasoner": "pa/gmn-2.5-pr",
                        "investigator": "pa/gmn-2.5-pr",
                    },
                }
            ),
            encoding="utf-8",
        )
        trace = [
            {
                "type": "occurrence_treatment_eligible",
                "round": 2,
                "visible_occurrence_count": 2 if arm == "a2" else 1,
            },
            {
                "type": "reasoner_decision",
                "occurrence_ops": (
                    [{"op": "select", "occurrence_id": "occ_1"}]
                    if arm == "a2"
                    else []
                ),
                "occurrence_ops_accepted": True,
            },
        ]
        if arm == "a2":
            trace.insert(
                1,
                {
                    "type": "decision_schema_error",
                    "errors": [
                        {"code": "occurrence_selection_required"}
                    ],
                },
            )
        if arm != "a0":
            trace.append({"type": "occurrence_treatment_exposed", "round": 2})
        (case / "runtime_summary.json").write_text(
            json.dumps(
                {
                    "no_oracle_runtime_gate": {
                        "no_oracle_runtime_gate_passed": True
                    },
                    "trace": trace,
                }
            ),
            encoding="utf-8",
        )
        if arm == "a2":
            (case / "occurrence_resolution_state.json").write_text(
                "{}", encoding="utf-8"
            )
        bindings[arm] = root

    report = AUDIT.audit_roots(bindings, expected_cases=1)

    assert report["structural_gate_passed"] is True
    assert report["per_arm"]["a2"]["occurrence_op_count"] == 1
    assert report["per_arm"]["a2"]["selection_case_count"] == 1
    assert report["per_arm"]["a2"]["selection_required_case_count"] == 1
    assert report["per_arm"]["a2"]["selection_required_retry_count"] == 1
    assert report["per_arm"]["a2"]["occurrence_validation_error_count"] == 0
    assert report["per_arm"]["a2"]["state_file_count"] == 1
