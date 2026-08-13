from __future__ import annotations

import importlib.util
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
