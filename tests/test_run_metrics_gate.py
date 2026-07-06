from __future__ import annotations

import json
from pathlib import Path

from runs.regression_gate import evaluate_gate
from runs.report_metrics import report_metrics


def test_report_metrics_counts_refusal_and_new_wrong(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "1", "gold": "A", "final_answer": "A", "correct": True},
                    {"case_id": "2", "gold": "B", "final_answer": "B", "correct": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    current.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "1", "gold": "A", "final_answer": "C", "correct": False},
                    {"case_id": "2", "gold": "B", "final_answer": "insufficient_verified_evidence", "correct": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    metrics = report_metrics(current, baseline_path=baseline)

    assert metrics["new_wrong_count"] == 1
    assert metrics["refusal_count"] == 1
    assert metrics["refusal_rate_delta"] == 0.5


def test_regression_gate_blocks_known_target_wrong_answer(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "1810", "gold": "C", "final_answer": "B", "correct": False},
                    {"case_id": "1822", "gold": "C", "final_answer": "C", "correct": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    gate = evaluate_gate(summary)

    assert gate["passed"] is False
    assert "target_case_still_wrong:1810" in gate["failures"]
