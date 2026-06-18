import json
from pathlib import Path

from scripts import run_ablation


def test_matrix_parses_and_builds_dry_run_commands(tmp_path: Path):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "matrix_id": "unit",
                "common_args": ["--strategy", "workspace_v2", "--cases", "605-1"],
                "runs": [
                    {"id": "base", "args": ["--max-rounds", "2"]},
                    {"id": "media", "planner_receives_media": True},
                    {"id": "no_followup", "enable_followup": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    matrix = run_ablation.load_matrix(matrix_path)
    entries = run_ablation.build_entries(matrix=matrix, output_dir=tmp_path / "out", python="python")
    index_path = run_ablation.write_index(output_dir=tmp_path / "out", matrix=matrix, entries=entries)

    assert entries[0]["id"] == "base"
    assert "--max-rounds" in entries[0]["argv"]
    assert "--planner-receives-media" in entries[1]["argv"]
    assert "--disable-followup" in entries[2]["argv"]
    assert json.loads(index_path.read_text(encoding="utf-8"))["schema_version"] == "AblationMatrixV1"


def test_matrix_rejects_empty_runs(tmp_path: Path):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps({"runs": []}), encoding="utf-8")

    try:
        run_ablation.load_matrix(matrix_path)
    except ValueError as exc:
        assert "runs" in str(exc)
    else:
        raise AssertionError("expected ValueError")
