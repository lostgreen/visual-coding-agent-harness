#!/usr/bin/env python3
"""Run an out-of-band negative-only occurrence evidence sidecar."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

from vcah.model_client import OpenAICompatibleClient
from vcah.occurrence_negative_sidecar import (
    NEGATIVE_SIDECAR_CONTRACT,
    file_sha256,
    load_negative_sidecar_snapshot,
    negative_sidecar_prompt,
    parse_negative_sidecar_response,
    safe_response_metadata,
    stable_digest,
    validate_negative_sidecar_output,
)


MAX_WORKERS = 16


def run_batch(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"sidecar output is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cases").mkdir(exist_ok=True)
    case_ids = _manifest_case_ids(Path(args.case_manifest))
    if args.case_ids:
        requested = set(args.case_ids)
        missing = sorted(requested - set(case_ids))
        if missing:
            raise ValueError("requested IDs are outside the manifest")
        case_ids = tuple(case_id for case_id in case_ids if case_id in requested)
    if args.expected_cases is not None and len(case_ids) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} cases, selected {len(case_ids)}"
        )
    if not case_ids:
        raise ValueError("no sidecar cases selected")

    client = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.section)
    workers = max(1, min(MAX_WORKERS, int(args.workers), len(case_ids)))
    run_manifest = {
        "schema_version": "MMLifelongOccurrenceNegativeSidecarRunV1",
        "contract": NEGATIVE_SIDECAR_CONTRACT,
        "repeat_label": str(args.repeat_label),
        "case_count": len(case_ids),
        "case_manifest_sha256": file_sha256(Path(args.case_manifest)),
        "positive_run_root": str(Path(args.positive_run_root)),
        "replay_fixture_root": str(Path(args.replay_fixture_root)),
        "config_sha256": file_sha256(Path(args.config)),
        "api_section": str(args.section),
        "actual_model": str(client.model),
        "max_completion_tokens": int(args.max_completion_tokens),
        "workers": workers,
        "workspace_write_enabled": False,
        "reasoner_context_write_enabled": False,
    }
    _write_json_atomic(out_root / "run_manifest.json", run_manifest)

    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one, case_id, client, args): case_id
            for case_id in case_ids
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "case_id": case_id,
                    "status": "orchestrator_failed",
                    "error_type": type(exc).__name__,
                }
            with lock:
                results[case_id] = result
                _write_summary(out_root, case_ids, results, run_manifest)
            print(
                f"SIDECAR_DONE case_id={case_id} status={result['status']} "
                f"rows={result.get('contradiction_row_count', 0)}",
                flush=True,
            )
    summary_path = _write_summary(out_root, case_ids, results, run_manifest)
    if Counter(row["status"] for row in results.values()).get("success", 0) != len(
        case_ids
    ):
        raise SystemExit(1)
    return summary_path


def _run_one(
    case_id: str,
    client: OpenAICompatibleClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_path = Path(args.out_root) / "cases" / case_id / "sidecar_result.json"
    positive_run_dir = Path(args.positive_run_root) / "cases" / case_id
    replay_path = Path(args.replay_fixture_root) / "cases" / f"{case_id}.json"
    try:
        snapshot = load_negative_sidecar_snapshot(
            positive_run_dir,
            replay_fixture_path=replay_path,
        )
    except Exception as exc:
        result = {
            "schema_version": "MMLifelongOccurrenceNegativeSidecarCaseV1",
            "contract": NEGATIVE_SIDECAR_CONTRACT,
            "case_id": case_id,
            "repeat_label": str(args.repeat_label),
            "status": "source_failed",
            "error_type": type(exc).__name__,
        }
        _write_json_atomic(result_path, result)
        return result
    if args.resume and result_path.is_file():
        prior = _read_json(result_path)
        if (
            prior.get("status") == "success"
            and prior.get("snapshot_digest") == snapshot.digest
            and prior.get("actual_model") == client.model
        ):
            return {**prior, "resumed": True}

    prompt = negative_sidecar_prompt(snapshot)
    started = time.monotonic()
    try:
        raw = client.chat(
            prompt,
            max_tokens=max(4096, int(args.max_completion_tokens)),
            response_format={"type": "json_object"},
        )
        response_metadata = safe_response_metadata(client.last_response_metadata)
    except Exception as exc:
        result = {
            "schema_version": "MMLifelongOccurrenceNegativeSidecarCaseV1",
            "contract": NEGATIVE_SIDECAR_CONTRACT,
            "case_id": case_id,
            "repeat_label": str(args.repeat_label),
            "status": "model_failed",
            "error_type": type(exc).__name__,
            "snapshot_digest": snapshot.digest,
            "prompt_digest": stable_digest(prompt),
            "actual_model": str(client.model),
            "duration_sec": round(time.monotonic() - started, 3),
        }
        _write_json_atomic(result_path, result)
        return result

    parsed = parse_negative_sidecar_response(raw)
    normalized, errors = validate_negative_sidecar_output(
        parsed,
        snapshot=snapshot,
    )
    status = "success" if not errors else "validation_failed"
    result = {
        "schema_version": "MMLifelongOccurrenceNegativeSidecarCaseV1",
        "contract": NEGATIVE_SIDECAR_CONTRACT,
        "case_id": case_id,
        "repeat_label": str(args.repeat_label),
        "status": status,
        "resumed": False,
        "snapshot_digest": snapshot.digest,
        "prompt_digest": stable_digest(prompt),
        "model_response_digest": stable_digest(raw),
        "actual_model": str(client.model),
        "duration_sec": round(time.monotonic() - started, 3),
        "input_counts": {
            "constraint_count": len(snapshot.constraints),
            "candidate_count": len(snapshot.candidates),
            "visible_passage_count": sum(
                len(tuple(row.get("representative_passages", ()) or ()))
                for row in snapshot.candidates
            ),
        },
        "source_digests": {
            "case_sha256": snapshot.source_case_sha256,
            "runtime_sha256": snapshot.source_runtime_sha256,
            "replay_fixture_sha256": snapshot.replay_fixture_sha256,
        },
        "contradiction_rows": list(normalized),
        "contradiction_row_count": len(normalized),
        "validation_error_codes": list(errors),
        "response_metadata": response_metadata,
        "raw_response_persisted": False,
        "prompt_persisted": False,
        "no_oracle_input_gate_passed": True,
        "live_model_call": True,
        "positive_support_visible_to_model": False,
        "selection_state_visible_to_model": False,
        "workspace_write_enabled": False,
        "reasoner_context_write_enabled": False,
    }
    _write_json_atomic(result_path, result)
    return result


def _write_summary(
    out_root: Path,
    case_ids: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> Path:
    path = Path(out_root) / "sidecar_batch_summary.json"
    status_counts = Counter(str(row.get("status", "")) for row in results.values())
    payload = {
        "schema_version": "MMLifelongOccurrenceNegativeSidecarBatchV1",
        "contract": NEGATIVE_SIDECAR_CONTRACT,
        "repeat_label": run_manifest["repeat_label"],
        "selected_count": len(case_ids),
        "completed_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "contradiction_row_count": sum(
            int(row.get("contradiction_row_count", 0) or 0) for row in results.values()
        ),
        "actual_model": run_manifest["actual_model"],
        "case_results": [
            {
                "case_id": case_id,
                "status": results[case_id].get("status"),
                "contradiction_row_count": int(
                    results[case_id].get("contradiction_row_count", 0) or 0
                ),
                "resumed": bool(results[case_id].get("resumed")),
            }
            for case_id in case_ids
            if case_id in results
        ],
    }
    _write_json_atomic(path, payload)
    return path


def _manifest_case_ids(path: Path) -> tuple[str, ...]:
    manifest = _read_json(path)
    case_ids = tuple(
        str(row.get("case_id", "") or "")
        for row in tuple(manifest.get("cases", ()) or ())
        if isinstance(row, Mapping) and str(row.get("case_id", "") or "")
    )
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case manifest must contain unique case IDs")
    return case_ids


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-run-root", required=True)
    parser.add_argument("--replay-fixture-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="planner_api")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--repeat-label", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run_batch(_parse_args())
    print(f"SIDECAR_BATCH_DONE summary={summary}", flush=True)


if __name__ == "__main__":
    main()
