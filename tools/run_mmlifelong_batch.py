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

from benchmarks.mmlifelong.oracle import ORACLE_ARMS
from vcah.occurrence_agent import (
    OCCURRENCE_METHOD_ARMS,
    validate_occurrence_method_configuration,
)
from vcah.phase5 import Phase5Protocol
from vcah.replay import file_checksum


SELECTION_STRATEGY = "question_type_temporal_quantiles"


def discover_cases(case_root: Path) -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for path in sorted(Path(case_root).glob("*/case.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("runtime_metadata", payload.get("metadata", {}))
        source_index = metadata.get("source_index", 0) if isinstance(metadata, Mapping) else 0
        cases.append(
            {
                "case_id": str(payload["case_id"]),
                "question_type": str(payload.get("question_type") or "Unknown"),
                "selection_coordinate": float(source_index),
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
                _selection_coordinate(item),
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
                _selection_coordinate(item),
                str(item["case_id"]),
            ),
        )
    )


def main() -> None:
    args = _parse_args()
    occurrence_method_arm = validate_occurrence_method_configuration(
        method_arm=args.occurrence_method_arm,
        oracle_arm=args.oracle_arm,
        oracle_intervention=args.oracle_intervention_root,
    )
    if args.oracle_arm == "o0":
        if args.oracle_intervention_root:
            raise ValueError("O0 must not load --oracle-intervention-root")
    elif not args.oracle_intervention_root:
        raise ValueError(f"{args.oracle_arm} requires --oracle-intervention-root")
    protocol = Phase5Protocol(
        controller_mode=args.controller_mode,
        controller_evidence_visibility=args.controller_evidence_visibility,
        measurement_control=args.measurement_control,
    )
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"batch output is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)
    (out_root / "cases").mkdir(exist_ok=True)

    candidates = discover_cases(Path(args.case_root))
    case_manifest_digest = None
    if args.case_manifest:
        manifest_path = Path(args.case_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_cases = manifest.get("cases", ()) if isinstance(manifest, Mapping) else ()
        manifest_case_ids = [
            str(row["case_id"])
            for row in raw_cases
            if isinstance(row, Mapping) and row.get("case_id")
        ]
        if not manifest_case_ids or len(manifest_case_ids) != len(set(manifest_case_ids)):
            raise ValueError("case manifest must contain unique non-empty case IDs")
        by_id = {str(case["case_id"]): case for case in candidates}
        missing = [case_id for case_id in manifest_case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"manifest case IDs are missing: {', '.join(missing)}")
        selected = tuple(dict(by_id[case_id]) for case_id in manifest_case_ids)
        selection_strategy = "explicit_case_manifest"
        case_manifest_digest = file_checksum(manifest_path)["sha256"]
    elif args.case_ids:
        by_id = {str(case["case_id"]): case for case in candidates}
        missing = [case_id for case_id in args.case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"requested case IDs are missing: {', '.join(missing)}")
        selected = tuple(dict(by_id[case_id]) for case_id in args.case_ids)
        selection_strategy = "explicit_case_ids"
    else:
        selected = select_stratified_cases(candidates, args.limit)
        selection_strategy = SELECTION_STRATEGY
    selection = {
        "schema_version": 1,
        **protocol.to_dict(),
        "strategy": selection_strategy,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "case_manifest_sha256": case_manifest_digest,
        "caption_config_digest": args.caption_config_digest,
        "caption_index_mode": args.caption_index_mode,
        "caption_query_strategy": args.caption_query_strategy,
        "answer_policy": args.answer_policy,
        "evidence_control_mode": args.evidence_control_mode,
        "evidence_state_mode": args.evidence_state_mode,
        "workers": max(1, min(int(args.workers), len(selected))),
        "phase5r_mode": "recorded_replay" if args.recorded_fixture_root else "live",
        "api_bindings": {
            "reasoner": {
                "config_name": Path(args.reasoner_config or args.config).name,
                "section": args.reasoner_section,
            },
            "investigator": {
                "config_name": Path(args.investigator_config or args.config).name,
                "section": args.investigator_section,
            },
        },
        "oracle_arm": args.oracle_arm,
        "occurrence_method_arm": occurrence_method_arm,
        "oracle_intervention_root": str(args.oracle_intervention_root or ""),
        "recorded_fixture_root": str(args.recorded_fixture_root or ""),
        "recorded_fixture_manifest": (
            file_checksum(Path(args.recorded_fixture_root) / "manifest.json")
            if args.recorded_fixture_root
            else None
        ),
        "question_type_counts": dict(Counter(case["question_type"] for case in selected)),
        "cases": [
            {
                "case_id": case["case_id"],
                "question_type": case["question_type"],
                "selection_coordinate": _selection_coordinate(case),
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
    prediction_path = out_dir / "prediction.json"
    if args.resume and prediction_path.is_file():
        return _successful_result(
            case,
            prediction_path,
            out_dir,
            log_path,
            duration_sec=0.0,
            resumed=True,
        )

    command = _case_command(case, args, out_dir)
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=_subprocess_env(),
                timeout=max(1, int(args.case_timeout_sec)),
            )
        duration = round(time.monotonic() - started, 3)
    except subprocess.TimeoutExpired:
        return {
            "case_id": case_id,
            "question_type": case["question_type"],
            "selection_coordinate": _selection_coordinate(case),
            "status": "timeout",
            "duration_sec": round(time.monotonic() - started, 3),
            "out_dir": str(out_dir),
            "log_path": str(log_path),
        }
    if completed.returncode != 0 or not prediction_path.is_file():
        return {
            "case_id": case_id,
            "question_type": case["question_type"],
            "selection_coordinate": _selection_coordinate(case),
            "status": "failed",
            "returncode": int(completed.returncode),
            "duration_sec": duration,
            "out_dir": str(out_dir),
            "log_path": str(log_path),
        }
    return _successful_result(case, prediction_path, out_dir, log_path, duration_sec=duration)


def _successful_result(
    case: Mapping[str, Any],
    prediction_path: Path,
    out_dir: Path,
    log_path: Path,
    *,
    duration_sec: float,
    resumed: bool = False,
) -> dict[str, Any]:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    runtime_path = out_dir / "runtime_summary.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    replay_path = out_dir / "phase5r_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.is_file() else {}
    return {
        "case_id": str(case["case_id"]),
        "question_type": case["question_type"],
        "selection_coordinate": _selection_coordinate(case),
        "status": "success",
        "resumed": bool(resumed),
        "duration_sec": duration_sec,
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "runtime": {
            "answer_present": prediction.get("answer_present"),
            "verification_status": prediction.get("verification_status"),
            "reference_valid": runtime.get("reference_valid"),
            "runtime_metrics": runtime.get("runtime_metrics"),
        },
        "oracle_arm": runtime.get("oracle_arm", "o0"),
        "oracle_intervention_audit": runtime.get("oracle_intervention_audit"),
        "phase5r_replay": {
            "decision": replay.get("decision"),
            "failed_checks": replay.get("failed_checks", []),
        }
        if replay
        else None,
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
        "--answer-policy",
        str(args.answer_policy),
        "--controller-mode",
        str(args.controller_mode),
        "--controller-evidence-visibility",
        str(args.controller_evidence_visibility),
        "--measurement-control",
        str(args.measurement_control),
        "--evidence-control-mode",
        str(args.evidence_control_mode),
        "--evidence-state-mode",
        str(args.evidence_state_mode),
        "--max-rounds",
        str(args.max_rounds),
        "--max-investigations",
        str(args.max_investigations),
        "--max-tasks-per-round",
        str(args.max_tasks_per_round),
        "--control-retry-budget",
        str(args.control_retry_budget),
        "--caption-index-mode",
        str(args.caption_index_mode),
        "--caption-query-strategy",
        str(args.caption_query_strategy),
        "--caption-config-digest",
        str(args.caption_config_digest),
        "--oracle-arm",
        str(args.oracle_arm),
        "--occurrence-method-arm",
        str(getattr(args, "occurrence_method_arm", "none")),
        "--embedding-model",
        str(args.embedding_model),
        "--embedding-device",
        str(args.embedding_device),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
    ]
    if getattr(args, "reasoner_config", None):
        command.extend(("--reasoner-config", str(args.reasoner_config)))
    if getattr(args, "investigator_config", None):
        command.extend(("--investigator-config", str(args.investigator_config)))
    if args.embedding_revision:
        command.extend(("--embedding-revision", str(args.embedding_revision)))
    if args.oracle_intervention_root:
        intervention_path = (
            Path(args.oracle_intervention_root)
            / "cases"
            / f"{case['case_id']}.json"
        )
        if not intervention_path.is_file():
            raise FileNotFoundError(
                f"missing oracle intervention: {intervention_path}"
            )
        command.extend(("--oracle-intervention", str(intervention_path)))
    if args.recorded_fixture_root:
        fixture_path = (
            Path(args.recorded_fixture_root)
            / "cases"
            / f"{case['case_id']}.json"
        )
        if not fixture_path.is_file():
            raise FileNotFoundError(f"missing recorded fixture: {fixture_path}")
        command.extend(("--recorded-decisions", str(fixture_path)))
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
            "phase5_arm": selection.get("phase5_arm"),
            "controller_mode": selection.get("controller_mode"),
            "controller_evidence_visibility": selection.get(
                "controller_evidence_visibility"
            ),
            "measurement_control": selection.get("measurement_control"),
            "phase5r_mode": selection.get("phase5r_mode"),
            "oracle_arm": selection.get("oracle_arm"),
            "occurrence_method_arm": selection.get("occurrence_method_arm"),
            "recorded_fixture_manifest": selection.get(
                "recorded_fixture_manifest"
            ),
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


def _subprocess_env() -> dict[str, str]:
    environment = os.environ.copy()
    repository_root = Path(__file__).resolve().parents[1]
    required = (str(repository_root), str(repository_root / "src"))
    existing = tuple(
        item
        for item in environment.get("PYTHONPATH", "").split(os.pathsep)
        if item
    )
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys((*required, *existing)))
    return environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a stratified MM-Lifelong case batch.")
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reasoner-config")
    parser.add_argument("--investigator-config")
    parser.add_argument("--caption-config-digest", required=True)
    parser.add_argument("--oracle-arm", choices=ORACLE_ARMS, default="o0")
    parser.add_argument("--oracle-intervention-root")
    parser.add_argument(
        "--occurrence-method-arm",
        choices=OCCURRENCE_METHOD_ARMS,
        default="none",
    )
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--case-manifest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    parser.add_argument("--reasoner-section", default="investigator_api")
    parser.add_argument("--investigator-section", default="investigator_api")
    parser.add_argument("--answer-policy", choices=("strict", "benchmark_best_effort"), default="benchmark_best_effort")
    parser.add_argument(
        "--controller-mode",
        choices=("frozen_baseline", "minimal_tool", "mger"),
        default="mger",
    )
    parser.add_argument(
        "--controller-evidence-visibility",
        choices=("none", "candidates_only", "full"),
        default="full",
    )
    parser.add_argument(
        "--measurement-control",
        choices=("none", "blind_prior", "caption_only"),
        default="none",
    )
    parser.add_argument("--evidence-control-mode", choices=("shadow", "strict"), default="shadow")
    parser.add_argument(
        "--evidence-state-mode",
        choices=("llm_authored", "runtime_derived"),
        default="runtime_derived",
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-investigations", type=int, default=12)
    parser.add_argument("--max-tasks-per-round", type=int, default=4)
    parser.add_argument("--control-retry-budget", type=int, default=2)
    parser.add_argument("--caption-index-mode", choices=("lexical", "dense", "hybrid"), default="hybrid")
    parser.add_argument(
        "--caption-query-strategy",
        choices=("joint", "rema", "adaptive"),
        default="joint",
    )
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--recorded-fixture-root")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.case_ids and args.case_manifest:
        parser.error("--case-ids and --case-manifest are mutually exclusive")
    return args


def _selection_coordinate(case: Mapping[str, Any]) -> float:
    return float(case.get("selection_coordinate", 0.0) or 0.0)


if __name__ == "__main__":
    main()
