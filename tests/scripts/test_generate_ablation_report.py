import json
from pathlib import Path

from scripts import generate_ablation_report


def test_report_contains_all_runs_and_completeness_section(tmp_path: Path):
    run_root = tmp_path / "agent_v2"
    run_root.mkdir()
    evidence_path = run_root / "evidence_chains.jsonl"
    evidence_path.write_text('{"id":"chain1"}\n{"id":"chain2"}\n', encoding="utf-8")
    trajectory_path = run_root / "traj.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "TrainingTrajectoryV1",
                "case_id": "605-1",
                "final_decision": "final",
                "selected_option": "D",
                "is_correct": True,
                "tool_calls": [{"step": 1, "tool": "vision_read"}],
                "evidence_chain_ids": ["chain1"],
                "frame_set_ids": ["fs1"],
                "context_budget_reports": [{"overflow": False}],
            }
        ),
        encoding="utf-8",
    )
    summary_path = run_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "accuracy": 1.0,
                "final_rate": 1.0,
                "need_more_evidence_rate": 0.0,
                "unsupported_final_rate": 0.0,
                "followup_success_rate": 1.0,
                "evidence_chains_path": str(evidence_path),
                "cases": [
                    {
                        "question_id": "605-1",
                        "raw_artifacts": {"training_trajectories": {"agent_v2": str(trajectory_path)}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    matrix_path = tmp_path / "index.json"
    matrix_path.write_text(
        json.dumps(
            {
                "matrix_id": "unit",
                "runs": [{"id": "agent_v2", "summary_path": str(summary_path), "status": "done", "exit_code": 0}],
            }
        ),
        encoding="utf-8",
    )

    report = generate_ablation_report.build_report(matrix_json=matrix_path)
    rendered = generate_ablation_report.render_markdown(report)

    assert report["runs"][0]["evidence_chain_rows"] == 2
    assert report["runs"][0]["trajectory_audit_failures"] == 0
    assert "| agent_v2 |" in rendered
    assert "## Completeness" in rendered
