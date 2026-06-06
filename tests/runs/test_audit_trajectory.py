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


def test_audit_checks_tool_results_are_visible_in_later_planner_turns(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "TrainingTrajectoryV1",
                "case_id": "case_001",
                "final_decision": "max_rounds_reached",
                "tool_calls": [{"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}}],
                "tool_results": [
                    {
                        "step": 1,
                        "tool": "vision_read",
                        "observation_id": "obs_0001",
                        "claim": "The local window supports option B.",
                        "source_round": 1,
                        "visible_in_planner_rounds": [2],
                    }
                ],
                "planner_turns": [
                    {"round": 1, "prompt_artifact": {"sha256": "aaa"}, "evidence_observation_ids": []},
                    {"round": 2, "prompt_artifact": {"sha256": "bbb"}, "evidence_observation_ids": ["obs_0001"]},
                ],
                "evidence_chain_ids": [],
                "frame_set_ids": ["fs_001"],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    status, output = _run_audit(trajectory_path)

    assert status == 0
    assert "planner_turns: 2" in output
    assert "tool_results: 1" in output
    assert "tool_result_visibility: 1/1" in output


def test_audit_rejects_missing_tool_result_claim_and_prompt_visibility(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "TrainingTrajectoryV1",
                "case_id": "case_001",
                "final_decision": "max_rounds_reached",
                "tool_calls": [{"step": 1, "tool": "global_gist", "arguments": {}}],
                "tool_results": [
                    {
                        "step": 1,
                        "tool": "global_gist",
                        "observation_id": "obs_0001",
                        "claim": "",
                        "source_round": 1,
                        "visible_in_planner_rounds": [],
                    }
                ],
                "planner_turns": [
                    {"round": 1, "prompt_artifact": {"sha256": "aaa"}, "evidence_observation_ids": []},
                    {"round": 2, "prompt_artifact": {"sha256": "bbb"}, "evidence_observation_ids": []},
                ],
                "evidence_chain_ids": [],
                "frame_set_ids": [],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    status, output = _run_audit(trajectory_path)

    assert status == 1
    assert "empty claim for obs_0001" in output
    assert "not visible in later planner prompt: obs_0001" in output


def _run_audit(trajectory_path: Path) -> tuple[int, str]:
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
    return completed.returncode, completed.stdout
