#!/usr/bin/env python3
"""Run or dry-run an eval ablation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PAIRED_BOOLEAN_FLAGS = {
    "enable_query_context",
    "enable_followup",
    "enable_context_budget",
    "enable_map_reflux",
    "enable_evidence_staging",
    "planner_receives_media",
}


def load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("matrix root must be an object")
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("matrix must contain a non-empty runs list")
    for item in runs:
        if not isinstance(item, Mapping) or not str(item.get("id", "")).strip():
            raise ValueError("each run must be an object with id")
    return dict(payload)


def build_entries(*, matrix: Mapping[str, Any], output_dir: Path, python: str = sys.executable) -> list[dict[str, Any]]:
    common_args = _string_list(matrix.get("common_args", []))
    entries = []
    for item in matrix.get("runs", []):
        run = dict(item)
        run_id = str(run["id"])
        run_root = output_dir / run_id
        argv = [
            python,
            "runs/eval_runner.py",
            *common_args,
            "--run-root",
            str(run_root),
            *_variant_args(run),
        ]
        entries.append(
            {
                "id": run_id,
                "argv": argv,
                "run_root": str(run_root),
                "summary_path": str(run_root / "summary.json"),
                "status": "pending",
                "exit_code": None,
                "stdout_path": str(output_dir / f"{run_id}.stdout.log"),
                "stderr_path": str(output_dir / f"{run_id}.stderr.log"),
            }
        )
    return entries


def write_index(*, output_dir: Path, matrix: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "AblationMatrixV1",
                "matrix_id": str(matrix.get("matrix_id", output_dir.name)),
                "generated_at": time.time(),
                "runs": list(entries),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return index_path


def run_entries(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        entry["started_at"] = time.time()
        stdout_path = Path(str(entry["stdout_path"]))
        stderr_path = Path(str(entry["stderr_path"]))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(entry["argv"], stdout=stdout, stderr=stderr, check=False)
        entry["ended_at"] = time.time()
        entry["exit_code"] = int(proc.returncode)
        entry["status"] = "done" if proc.returncode == 0 and Path(str(entry["summary_path"])).exists() else "failed"


def _variant_args(run: Mapping[str, Any]) -> list[str]:
    if "args" in run:
        return _string_list(run["args"])
    args = []
    for key, value in run.items():
        if key == "id":
            continue
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif str(key) in PAIRED_BOOLEAN_FLAGS:
                args.append("--disable-" + str(key).removeprefix("enable_").replace("_", "-"))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("args must be a list of strings")
    return [str(item) for item in value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or dry-run a visual harness ablation matrix.")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    matrix = load_matrix(args.matrix)
    entries = build_entries(matrix=matrix, output_dir=args.output_dir)
    if args.dry_run:
        for entry in entries:
            print(" ".join(entry["argv"]))
        write_index(output_dir=args.output_dir, matrix=matrix, entries=entries)
        return
    run_entries(entries)
    index_path = write_index(output_dir=args.output_dir, matrix=matrix, entries=entries)
    print(f"DONE index={index_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
