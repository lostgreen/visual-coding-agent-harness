from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


def _load_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "evaluate_mmlifelong_batch.py"
    spec = importlib.util.spec_from_file_location("evaluate_mmlifelong_batch", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BATCH = _load_module()


def test_discovers_same_case_in_distinct_oracle_arms(tmp_path: Path) -> None:
    roots = []
    for arm in ("c0", "o1"):
        root = tmp_path / arm
        run = root / "cases" / "case-1"
        run.mkdir(parents=True)
        (run / "prediction.json").write_text(
            json.dumps({"case_id": "case-1"}), encoding="utf-8"
        )
        (run / "run_config.json").write_text(
            json.dumps({"oracle_arm": arm}), encoding="utf-8"
        )
        roots.append(root)

    discovered = BATCH.discover_runs(roots)

    assert [BATCH._run_key(run) for run in discovered] == ["c0:case-1", "o1:case-1"]


def test_discovers_same_o0_case_in_distinct_occurrence_method_arms(
    tmp_path: Path,
) -> None:
    roots = []
    for method_arm in ("a0", "a1"):
        root = tmp_path / method_arm
        run = root / "cases" / "case-1"
        run.mkdir(parents=True)
        (run / "prediction.json").write_text(
            json.dumps({"case_id": "case-1"}), encoding="utf-8"
        )
        (run / "run_config.json").write_text(
            json.dumps(
                {"oracle_arm": "o0", "occurrence_method_arm": method_arm}
            ),
            encoding="utf-8",
        )
        roots.append(root)

    discovered = BATCH.discover_runs(roots)

    assert [BATCH._run_key(run) for run in discovered] == [
        "o0:a0:case-1",
        "o0:a1:case-1",
    ]


def test_command_uses_prepared_evaluator_only_record(tmp_path: Path) -> None:
    record = tmp_path / "records" / "case-1" / "evaluation_case.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        evaluation_record_root=str(tmp_path / "records"),
        judge_max_retries=2,
        max_completion_tokens=4096,
        judge_response_file=None,
        config="api.yaml",
        judge_section="judge_api",
        overwrite=False,
    )

    command = BATCH.evaluation_command(
        {"case_id": "case-1", "oracle_arm": "o1", "run_dir": "/run"},
        args,
    )

    assert command[command.index("--evaluation-record") + 1] == str(record)
    assert command[command.index("--judge-section") + 1] == "judge_api"
    assert "--overwrite" not in command


def test_worker_limit_is_sixteen() -> None:
    assert BATCH.MAX_WORKERS == 16
