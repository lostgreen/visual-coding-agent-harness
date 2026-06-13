import json
import subprocess
import sys
from pathlib import Path

from visual_coding_agent_harness.evals.videomme.trajectory_markdown import render_trajectory_markdown


def test_render_trajectory_markdown_groups_planner_io_and_tool_results(tmp_path):
    prompt = tmp_path / "artifacts" / "planner_io" / "round_0001_prompt.txt"
    response = tmp_path / "artifacts" / "planner_io" / "round_0001_response.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("## Task\nWhich object is visible?\n## Evidence\n(none)\n", encoding="utf-8")
    response.write_text(
        '<think>The user wants hidden reasoning.</think>{"status":"continue","program":[{"tool":"vision_read"}]}',
        encoding="utf-8",
    )
    trajectory = {
        "schema_version": "TrainingTrajectoryV1",
        "case_id": "case_001",
        "question": "Which object is visible?",
        "options": ["A. blue car", "B. red car"],
        "ground_truth": "B",
        "selected_option": "B",
        "is_correct": True,
        "workspace_root": tmp_path.as_posix(),
        "planner_turns": [
            {
                "round": 1,
                "prompt_artifact": {"path": "artifacts/planner_io/round_0001_prompt.txt"},
                "response_artifact": {"path": "artifacts/planner_io/round_0001_response.txt"},
                "response_excerpt": '{"program":[{"tool":"vision_read"}],"status":"continue"}',
                "evidence_observation_ids": [],
            }
        ],
        "planner_plans": [
            {
                "round": 1,
                "rationale": "inspect the visible object",
                "program": [{"tool": "vision_read", "args": {"segment_id": "seg_0001"}}],
            }
        ],
        "route_repairs": [
            {
                "round": 1,
                "requested_tool": "vision_read",
                "resolved_tool": "global_gist",
                "reason": "main idea floor",
            }
        ],
        "tool_calls": [
            {
                "step": 1,
                "source_round": 1,
                "tool": "global_gist",
                "arguments": {"segment_id": "seg_0001"},
            }
        ],
        "tool_results": [
            {
                "step": 1,
                "source_round": 1,
                "tool": "global_gist",
                "observation_id": "obs_0001",
                "claim": "The window shows a red car.",
                "confidence": 0.9,
                "visible_in_planner_rounds": [2],
            }
        ],
    }

    markdown = render_trajectory_markdown(trajectory, trajectory_path=tmp_path / "case_001.json")

    assert "# Trajectory case_001" in markdown
    assert "## Round 1" in markdown
    assert "### Planner input" in markdown
    assert "prompt_artifact: path=artifacts/planner_io/round_0001_prompt.txt" in markdown
    assert "## Task" not in markdown
    assert "### Planner output" in markdown
    assert '"status":"continue"' in markdown
    assert "The user wants hidden reasoning" not in markdown
    assert "### Tool results" in markdown
    assert "obs_0001" in markdown
    assert "The window shows a red car." in markdown
    assert "vision_read -> global_gist" in markdown


def test_render_trajectory_markdown_explains_runs_without_planner_turns(tmp_path):
    trajectory = {
        "schema_version": "TrainingTrajectoryV1",
        "case_id": "case_002",
        "question": "What is the video mainly about?",
        "selected_option": "B",
        "workspace_root": tmp_path.as_posix(),
        "planner_turns": [],
        "tool_calls": [
            {"step": 1, "tool": "global_gist", "arguments": {"ask_for": "main idea"}, "source_round": 0}
        ],
        "tool_results": [
            {
                "step": 1,
                "tool": "global_gist",
                "observation_id": "obs_0001",
                "claim": "The video discusses imperial collapse.",
                "confidence": 0.76,
                "source_round": 0,
            }
        ],
    }

    markdown = render_trajectory_markdown(trajectory, trajectory_path=tmp_path / "case_002.json")

    assert "## Planner turns" in markdown
    assert "No planner turns were recorded" in markdown
    assert "## Non-planner tool activity" in markdown
    assert "The video discusses imperial collapse." in markdown


def test_trajectory_markdown_cli_writes_default_neighbor_file(tmp_path):
    trajectory_path = tmp_path / "case_001_agent_v2.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "TrainingTrajectoryV1",
                "case_id": "case_001",
                "question": "Which object is visible?",
                "workspace_root": tmp_path.as_posix(),
                "planner_turns": [],
                "tool_calls": [],
                "tool_results": [],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "scripts/trajectory_markdown.py", str(trajectory_path)],
        cwd=Path(__file__).resolve().parents[2],
        env={"PYTHONPATH": "src:."},
        check=False,
        text=True,
        capture_output=True,
        timeout=20,
    )

    output_path = tmp_path / "case_001_agent_v2.md"
    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert output_path.as_posix() in completed.stdout
