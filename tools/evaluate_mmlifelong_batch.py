#!/usr/bin/env python3
"""Evaluate one or more MM-Lifelong run roots with bounded concurrency."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence


MAX_WORKERS = 16


def discover_runs(run_roots: Sequence[Path]) -> tuple[dict[str, str], ...]:
    runs: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in run_roots:
        for prediction_path in sorted(Path(root).glob("cases/*/prediction.json")):
            run_dir = prediction_path.parent.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            prediction = _read_json(prediction_path)
            config_path = run_dir / "run_config.json"
            config = _read_json(config_path) if config_path.is_file() else {}
            runs.append(
                {
                    "case_id": str(prediction["case_id"]),
                    "oracle_arm": str(config.get("oracle_arm", "o0")),
                    "run_dir": str(run_dir),
                }
            )
    if not runs:
        raise FileNotFoundError("no prediction.json files found under the run roots")
    return tuple(runs)


def evaluation_command(
    run: Mapping[str, str],
    args: argparse.Namespace,
) -> list[str]:
    record = (
        Path(args.evaluation_record_root)
        / str(run["case_id"])
        / "evaluation_case.json"
    )
    if not record.is_file():
        raise FileNotFoundError(f"missing evaluation record: {record}")
    command = [
        sys.executable,
        "-m",
        "evaluate.mmlifelong.cli",
        "--run-dir",
        str(run["run_dir"]),
        "--evaluation-record",
        str(record),
        "--judge-max-retries",
        str(args.judge_max_retries),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
    ]
    if args.judge_response_file:
        command.extend(("--judge-response-file", str(args.judge_response_file)))
    else:
        command.extend(
            (
                "--config",
                str(args.config),
                "--judge-section",
                str(args.judge_section),
            )
        )
    if args.overwrite:
        command.append("--overwrite")
    return command


def evaluate_batch(args: argparse.Namespace) -> Path:
    if not args.judge_response_file and not args.config:
        raise ValueError("--config is required for live judging")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    logs_root = out_root / "logs"
    logs_root.mkdir(exist_ok=True)
    runs = discover_runs(tuple(Path(value) for value in args.run_root))
    if args.case_ids:
        requested = set(args.case_ids)
        runs = tuple(run for run in runs if run["case_id"] in requested)
        found = {run["case_id"] for run in runs}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested case IDs are missing: {', '.join(missing)}")
    workers = max(1, min(MAX_WORKERS, int(args.workers), len(runs)))
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_evaluate_one, run, args, logs_root): _run_key(run)
            for run in runs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "run_key": key,
                    "status": "orchestrator_failed",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            with lock:
                results[key] = result
                _write_summary(out_root, workers, runs, results)
            print(
                f"EVAL_DONE run_key={key} status={result['status']} "
                f"score={result.get('score')}",
                flush=True,
            )
    summary_path = _write_summary(out_root, workers, runs, results)
    if Counter(result["status"] for result in results.values()).get("success", 0) != len(runs):
        raise SystemExit(1)
    return summary_path


def _evaluate_one(
    run: Mapping[str, str],
    args: argparse.Namespace,
    logs_root: Path,
) -> dict[str, Any]:
    run_dir = Path(run["run_dir"])
    evaluation_path = run_dir / "evaluation" / "mmlifelong_eval.json"
    if args.resume and evaluation_path.is_file():
        return _successful_result(run, evaluation_path, 0.0, resumed=True)
    log_path = logs_root / f"{_run_key(run).replace(':', '-')}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            evaluation_command(run, args),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=_subprocess_env(),
            timeout=max(1, int(args.case_timeout_sec)),
        )
    duration = round(time.monotonic() - started, 3)
    if completed.returncode != 0 or not evaluation_path.is_file():
        return {
            "run_key": _run_key(run),
            **dict(run),
            "status": "failed",
            "returncode": int(completed.returncode),
            "duration_sec": duration,
            "log_path": str(log_path),
        }
    return _successful_result(run, evaluation_path, duration)


def _successful_result(
    run: Mapping[str, str],
    evaluation_path: Path,
    duration_sec: float,
    *,
    resumed: bool = False,
) -> dict[str, Any]:
    evaluation = _read_json(evaluation_path)
    answer = evaluation.get("answer", {})
    return {
        "run_key": _run_key(run),
        **dict(run),
        "status": "success",
        "resumed": resumed,
        "duration_sec": duration_sec,
        "score": answer.get("score"),
        "raw_score": answer.get("raw_score"),
        "parse_status": answer.get("parse_status"),
        "official_judge_model_match": answer.get("official_judge_model_match"),
        "evaluation_path": str(evaluation_path),
    }


def _write_summary(
    out_root: Path,
    workers: int,
    runs: Sequence[Mapping[str, str]],
    results: Mapping[str, Mapping[str, Any]],
) -> Path:
    path = out_root / "evaluation_batch_summary.json"
    status_counts = Counter(result["status"] for result in results.values())
    payload = {
        "schema_version": "MMLifelongEvaluationBatchV1",
        "selected_count": len(runs),
        "completed_count": len(results),
        "workers": workers,
        "status_counts": dict(sorted(status_counts.items())),
        "oracle_arm_counts": dict(
            sorted(Counter(run["oracle_arm"] for run in runs).items())
        ),
        "results": [results[key] for key in sorted(results)],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _run_key(run: Mapping[str, str]) -> str:
    return f"{run['oracle_arm']}:{run['case_id']}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _subprocess_env() -> dict[str, str]:
    environment = os.environ.copy()
    repository_root = Path(__file__).resolve().parents[1]
    required = (str(repository_root), str(repository_root / "src"))
    existing = tuple(
        item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item
    )
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys((*required, *existing)))
    return environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MM-Lifelong batch predictions with up to 16 workers."
    )
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--judge-section", default="judge_api")
    parser.add_argument("--judge-response-file")
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--case-timeout-sec", type=int, default=600)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary_path = evaluate_batch(_parse_args())
    print(f"EVALUATION_BATCH_DONE summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
