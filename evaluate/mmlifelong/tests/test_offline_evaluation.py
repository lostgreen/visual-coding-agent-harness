from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from benchmarks.schema import EvaluationRecord, RuntimeQuestion
from benchmarks.mmlifelong.runner import prediction_artifact


def test_cli_reparses_existing_prediction_without_rerunning_agent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    question = RuntimeQuestion(
        case_id="mmlifelong-game-test-0031",
        question="What was raised?",
        subset="game",
        split="test",
        runtime_metadata={"source_subset": "day", "source_index": 31},
    )
    prediction = prediction_artifact(
        question,
        answer="A cup",
        selected_option="",
        supporting_intervals=((10.0, 20.0),),
        supporting_attempt_ids=("attempt-1",),
        answer_present=True,
        duration_sec=100.0,
    )
    (run_dir / "prediction.json").write_text(
        json.dumps(prediction),
        encoding="utf-8",
    )
    record = EvaluationRecord(
        case_id=question.case_id,
        reference_answer="A cup",
        clue_intervals=((12.0, 18.0),),
        evaluation_metadata={"question": "What was raised?"},
    )
    record_path = tmp_path / "evaluation_case.json"
    record_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    response_path = tmp_path / "judge.txt"
    response_path.write_text(
        "Analysis:\nThe answers are equivalent.\n\nFinal Score:\n5\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluate.mmlifelong.cli",
            "--run-dir",
            str(run_dir),
            "--evaluation-record",
            str(record_path),
            "--judge-response-file",
            str(response_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )

    assert completed.returncode == 0, completed.stderr
    evaluation = json.loads(
        (run_dir / "evaluation" / "mmlifelong_eval.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (run_dir / "evaluation" / "eval_provenance.json").read_text(encoding="utf-8")
    )
    assert evaluation["answer"]["raw_score"] == 5
    assert evaluation["answer"]["score"] == 1.0
    assert evaluation["reference_grounding"]["ref_60"] == 1.0
    assert provenance["prediction_artifact_sha256"]
