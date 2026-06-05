import json
import os
import subprocess
import sys
from pathlib import Path


def test_audit_runs_on_smoke_case(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "TrainingTrajectoryV1",
                "case_id": "case_001",
                "final_decision": "final",
                "selected_option": "B",
                "is_correct": True,
                "tool_calls": [{"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}}],
                "evidence_chain_ids": [["ev_distilled_1", "ev_ledger_2", "ev_mapped_3"]],
                "frame_set_ids": ["fs_001"],
                "context_budget_reports": [{"overflow": False}],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [sys.executable, "scripts/audit_trajectory.py", str(trajectory_path)],
        cwd=repo_root,
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert "case_id: case_001" in completed.stdout
    assert "evidence_chains: 1" in completed.stdout


def test_audit_rejects_final_without_evidence_chain(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "TrainingTrajectoryV1",
                "case_id": "case_001",
                "final_decision": "final",
                "tool_calls": [{"step": 1, "tool": "vision_read", "arguments": {}}],
                "evidence_chain_ids": [],
                "frame_set_ids": [],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [sys.executable, "scripts/audit_trajectory.py", str(trajectory_path)],
        cwd=repo_root,
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert completed.returncode == 1
    assert "no evidence chains" in completed.stdout
