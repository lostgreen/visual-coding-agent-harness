#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence


SELECTION_STRATEGY = "question_type_temporal_quantiles"


def discover_cases(case_root: Path) -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for path in sorted(Path(case_root).glob("*/case.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        interval = payload.get("target_virtual_interval") or ()
        if len(interval) != 2:
            raise ValueError(f"Case is missing target_virtual_interval: {path}")
        cases.append(
            {
                "case_id": str(payload["case_id"]),
                "question_type": str(payload.get("question_type") or "Unknown"),
                "target_virtual_interval": [float(interval[0]), float(interval[1])],
                "case_workspace": str(path.parent),
            }
        )
    if not cases:
        raise FileNotFoundError(f"No MM-Lifelong cases found under {case_root}")
    return tuple(cases)


def select_stratified_cases(
    cases: Sequence[Mapping[str, Any]],
    limit: int,
) -> tuple[dict[str, Any], ...]:
    requested = int(limit)
    if requested <= 0:
        raise ValueError("limit must be positive")
    requested = min(requested, len(cases))
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["question_type"])].append(case)
    types = sorted(grouped)
    base, remainder = divmod(requested, len(types))
    quotas = {
        question_type: min(base + (index < remainder), len(grouped[question_type]))
        for index, question_type in enumerate(types)
    }
    unallocated = requested - sum(quotas.values())
    while unallocated:
        changed = False
        for question_type in types:
            if quotas[question_type] >= len(grouped[question_type]):
                continue
            quotas[question_type] += 1
            unallocated -= 1
            changed = True
            if not unallocated:
                break
        if not changed:
            raise ValueError("Could not allocate the requested number of cases")

    selected: list[dict[str, Any]] = []
    for question_type in types:
        group = sorted(
            grouped[question_type],
            key=lambda item: (
                float(item["target_virtual_interval"][0]),
                str(item["case_id"]),
            ),
        )
        quota = quotas[question_type]
        indices = tuple(((index + 1) * len(group)) // (quota + 1) for index in range(quota))
        selected.extend(dict(group[index]) for index in indices)
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                str(item["question_type"]),
                float(item["target_virtual_interval"][0]),
                str(item["case_id"]),
            ),
        )
    )


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"batch output is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)
    (out_root / "cases").mkdir(exist_ok=True)

    candidates = discover_cases(Path(args.case_root))
    selected = select_stratified_cases(candidates, args.limit)
    selection = {
        "schema_version": 1,
        "strategy": SELECTION_STRATEGY,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "caption_config_digest": args.caption_config_digest,
        "caption_index_mode": args.caption_index_mode,
        "caption_query_strategy": args.caption_query_strategy,
        "answer_policy": args.answer_policy,
        "workers": max(1, min(int(args.workers), len(selected))),
        "question_type_counts": dict(Counter(case["question_type"] for case in selected)),
        "cases": [
            {
                "case_id": case["case_id"],
                "question_type": case["question_type"],
                "target_virtual_interval": case["target_virtual_interval"],
            }
            for case in selected
        ],
    }
    _write_json(out_root / "selection.json", selection)

    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, min(int(args.workers), len(selected)))) as executor:
        futures = {
            executor.submit(_run_case, case, args, out_root): str(case["case_id"])
            for case in selected
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "case_id": case_id,
                    "status": "orchestrator_failed",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            with lock:
                results[case_id] = result
                _write_batch_summary(out_root, selection, results)
            print(
                f"CASE_DONE case_id={case_id} status={result['status']} "
                f"duration_sec={result.get('duration_sec', 0.0)}",
                flush=True,
            )

    summary_path = _write_batch_summary(out_root, selection, results)
    status_counts = Counter(result["status"] for result in results.values())
    print(
        "BATCH_DONE "
        f"selected={len(selected)} status_counts={dict(status_counts)} summary={summary_path}",
        flush=True,
    )
    if status_counts.get("success", 0) != len(selected):
        raise SystemExit(1)


def _run_case(
    case: Mapping[str, Any],
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    out_dir = out_root / "cases" / case_id
    log_path = out_root / "logs" / f"{case_id}.log"
    metrics_path = out_dir / "mmlifelong_metrics.json"
    if args.resume and metrics_path.is_file():
        return _successful_result(case, metrics_path, out_dir, log_path, duration_sec=0.0, resumed=True)

    command = _case_command(case, args, out_dir)
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
                timeout=max(1, int(args.case_timeout_sec)),
            )
        duration = round(time.monotonic() - started, 3)
    except subprocess.TimeoutExpired:
        return {
            "case_id": case_id,
            "question_type": case["question_type"],
            "target_virtual_interval": case["target_virtual_interval"],
            "status": "timeout",
            "duration_sec": round(time.monotonic() - started, 3),
            "out_dir": str(out_dir),
            "log_path": str(log_path),
        }
    if completed.returncode != 0 or not metrics_path.is_file():
        return {
            "case_id": case_id,
            "question_type": case["question_type"],
            "target_virtual_interval": case["target_virtual_interval"],
            "status": "failed",
            "returncode": int(completed.returncode),
            "duration_sec": duration,
            "out_dir": str(out_dir),
            "log_path": str(log_path),
        }
    return _successful_result(case, metrics_path, out_dir, log_path, duration_sec=duration)


def _successful_result(
    case: Mapping[str, Any],
    metrics_path: Path,
    out_dir: Path,
    log_path: Path,
    *,
    duration_sec: float,
    resumed: bool = False,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "case_id": str(case["case_id"]),
        "question_type": case["question_type"],
        "target_virtual_interval": case["target_virtual_interval"],
        "status": "success",
        "resumed": bool(resumed),
        "duration_sec": duration_sec,
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "metrics": {
            "accuracy_score": metrics.get("accuracy_score"),
            "answer_present": metrics.get("answer_present"),
            "reference_valid": metrics.get("reference_valid"),
            "ref": metrics.get("ref"),
            "retrieval": metrics.get("retrieval"),
            "agent": metrics.get("agent"),
        },
    }


def _case_command(
    case: Mapping[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[str]:
    runner = Path(__file__).with_name("run_mmlifelong_interactive.py")
    command = [
        sys.executable,
        str(runner),
        "--case-workspace",
        str(case["case_workspace"]),
        "--out-dir",
        str(out_dir),
        "--config",
        str(args.config),
        "--reasoner-section",
        str(args.reasoner_section),
        "--investigator-section",
        str(args.investigator_section),
        "--judge-section",
        str(args.judge_section),
        "--answer-policy",
        str(args.answer_policy),
        "--max-rounds",
        str(args.max_rounds),
        "--max-investigations",
        str(args.max_investigations),
        "--max-tasks-per-round",
        str(args.max_tasks_per_round),
        "--caption-index-mode",
        str(args.caption_index_mode),
        "--caption-query-strategy",
        str(args.caption_query_strategy),
        "--caption-config-digest",
        str(args.caption_config_digest),
        "--embedding-model",
        str(args.embedding_model),
        "--embedding-device",
        str(args.embedding_device),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        "--judge-max-retries",
        str(args.judge_max_retries),
        "--judge" if args.judge else "--no-judge",
    ]
    if args.embedding_revision:
        command.extend(("--embedding-revision", str(args.embedding_revision)))
    return command


def _write_batch_summary(
    out_root: Path,
    selection: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> Path:
    path = out_root / "batch_summary.json"
    status_counts = Counter(result["status"] for result in results.values())
    _write_json(
        path,
        {
            "schema_version": 1,
            "selection_strategy": selection["strategy"],
            "selected_count": selection["selected_count"],
            "completed_count": len(results),
            "status_counts": dict(status_counts),
            "caption_config_digest": selection.get("caption_config_digest"),
            "results": [results[case_id] for case_id in sorted(results)],
        },
    )
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a stratified MM-Lifelong case batch.")
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--caption-config-digest", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    parser.add_argument("--reasoner-section", default="investigator_api")
    parser.add_argument("--investigator-section", default="investigator_api")
    parser.add_argument("--judge-section", default="investigator_api")
    parser.add_argument("--answer-policy", choices=("strict", "benchmark_best_effort"), default="benchmark_best_effort")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-investigations", type=int, default=12)
    parser.add_argument("--max-tasks-per-round", type=int, default=4)
    parser.add_argument("--caption-index-mode", choices=("lexical", "dense", "hybrid"), default="hybrid")
    parser.add_argument(
        "--caption-query-strategy",
        choices=("joint", "rema", "adaptive"),
        default="joint",
    )
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
