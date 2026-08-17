#!/usr/bin/env python3
"""Run the blind WP14 visual discriminability probe."""

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
    file_sha256,
    safe_response_metadata,
    stable_digest,
)
from vcah.occurrence_visual_probe import (
    VISUAL_PROBE_CONTRACT,
    audit_visual_probe_manifest,
    parse_visual_probe_response,
    visual_probe_prompt,
)


MAX_WORKERS = 16


def run_probe(args: argparse.Namespace) -> Path:
    probe_root = Path(args.probe_root)
    manifest_path = probe_root / "probe_manifest.json"
    manifest = _read_json(manifest_path)
    audit = audit_visual_probe_manifest(manifest, root=probe_root)
    if not audit["structural_gate_passed"]:
        raise ValueError("visual probe provenance gate did not pass")
    cases, items, windows = _indexes(manifest)
    if args.expected_cases is not None and len(cases) != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} cases, found {len(cases)}")
    if args.expected_items is not None and len(items) != args.expected_items:
        raise ValueError(f"expected {args.expected_items} items, found {len(items)}")

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"visual probe output is not empty: {out_root}")
    (out_root / "items").mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.section)
    workers = max(1, min(MAX_WORKERS, int(args.workers), len(items)))
    run_manifest = {
        "schema_version": "MMLifelongVisualDiscriminabilityProbeRunV1",
        "contract": VISUAL_PROBE_CONTRACT,
        "probe_manifest_sha256": file_sha256(manifest_path),
        "provenance_structural_gate_passed": True,
        "eligible_case_count": len(cases),
        "item_count": len(items),
        "actual_model": str(client.model),
        "config_sha256": file_sha256(Path(args.config)),
        "api_section": str(args.section),
        "max_completion_tokens": max(4096, int(args.max_completion_tokens)),
        "temperature": client.replay_settings.get("temperature"),
        "top_p": client.replay_settings.get("top_p"),
        "requested_seed": client.replay_settings.get("requested_seed"),
        "provider_seed_supported": client.replay_settings.get(
            "provider_seed_supported"
        ),
        "provider_reported_seed_support": client.replay_settings.get(
            "provider_reported_seed_support"
        ),
        "response_format": {"type": "json_object"},
        "workers": workers,
        "agent_behavior_changed": False,
        "workspace_write_enabled": False,
        "reasoner_context_write_enabled": False,
        "prompt_persisted": False,
        "raw_response_persisted": False,
    }
    _write_json_atomic(out_root / "run_manifest.json", run_manifest)

    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                item,
                window=windows[str(item["visual_observation_id"])],
                probe_root=probe_root,
                out_root=out_root,
                client=client,
                args=args,
            ): str(item["item_id"])
            for item in items.values()
        }
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": "MMLifelongVisualProbeItemResultV1",
                    "contract": VISUAL_PROBE_CONTRACT,
                    "item_id": item_id,
                    "status": "orchestrator_failed",
                    "error_type": type(exc).__name__,
                }
            with lock:
                results[item_id] = result
                _write_summary(out_root, tuple(items), results, run_manifest)
            print(
                f"VISUAL_PROBE_DONE item_id={item_id} status={result['status']}",
                flush=True,
            )
    summary_path = _write_summary(out_root, tuple(items), results, run_manifest)
    if sum(row.get("status") == "success" for row in results.values()) != len(items):
        raise SystemExit(1)
    return summary_path


def _run_one(
    item: Mapping[str, Any],
    *,
    window: Mapping[str, Any],
    probe_root: Path,
    out_root: Path,
    client: OpenAICompatibleClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    item_id = str(item.get("item_id", "") or "")
    result_path = out_root / "items" / f"{item_id}.json"
    prior = _read_json(result_path) if args.resume and result_path.is_file() else None
    if (
        prior is not None
        and prior.get("status") == "success"
        and prior.get("actual_model") == client.model
        and prior.get("item_digest") == stable_digest(dict(item))
    ):
        return {**prior, "resume_reused_success": True}
    frame_paths = tuple(
        str(probe_root / str(row.get("path", "") or ""))
        for row in tuple(window.get("frames", ()) or ())
        if isinstance(row, Mapping)
    )
    if not frame_paths or any(not Path(path).is_file() for path in frame_paths):
        raise ValueError(f"{item_id}: materialized frames are incomplete")
    prompt = visual_probe_prompt(item)
    started = time.monotonic()
    attempts = []
    verdict = None
    response_metadata: dict[str, Any] = {}
    raw = ""
    for parse_attempt in range(2):
        call_prompt = prompt
        if parse_attempt:
            call_prompt += "\nYour previous response was invalid. Return only the exact JSON schema."
        try:
            raw = client.chat(
                call_prompt,
                image_paths=frame_paths,
                max_tokens=max(4096, int(args.max_completion_tokens)),
                response_format={"type": "json_object"},
            )
            response_metadata = safe_response_metadata(client.last_response_metadata)
        except Exception as exc:
            attempts.append(
                {
                    "attempt_index": parse_attempt + 1,
                    "status": "model_failed",
                    "error_type": type(exc).__name__,
                }
            )
            break
        verdict = parse_visual_probe_response(raw)
        attempts.append(
            {
                "attempt_index": parse_attempt + 1,
                "status": "success" if verdict is not None else "invalid_json",
                "model_response_digest": stable_digest(raw),
                "response_metadata": response_metadata,
            }
        )
        if verdict is not None:
            break
    status = "success" if verdict is not None else attempts[-1]["status"]
    result = {
        "schema_version": "MMLifelongVisualProbeItemResultV1",
        "contract": VISUAL_PROBE_CONTRACT,
        "item_id": item_id,
        "visual_observation_id": str(item.get("visual_observation_id", "") or ""),
        "status": status,
        "verdict": verdict,
        "actual_model": str(client.model),
        "item_digest": stable_digest(dict(item)),
        "prompt_digest": stable_digest(prompt),
        "model_response_digest": stable_digest(raw),
        "frame_count": len(frame_paths),
        "duration_sec": round(time.monotonic() - started, 3),
        "attempt_count": len(attempts),
        "parse_retry_count": max(0, len(attempts) - 1),
        "attempt_history": attempts,
        "response_metadata": response_metadata,
        "resume_reused_success": False,
        "prompt_persisted": False,
        "raw_response_persisted": False,
        "pair_kind_visible_to_model": False,
        "gold_visible_to_model": False,
        "answer_visible_to_model": False,
        "r5_state_visible_to_model": False,
    }
    _write_json_atomic(result_path, result)
    return result


def _indexes(manifest: Mapping[str, Any]):
    cases = {}
    items = {}
    windows = {}
    for case in tuple(manifest.get("cases", ()) or ()):
        if not isinstance(case, Mapping) or not bool(case.get("eligible")):
            continue
        case_id = str(case.get("case_id", "") or "")
        cases[case_id] = case
        for window in tuple(case.get("windows", ()) or ()):
            if isinstance(window, Mapping):
                windows[str(window.get("visual_observation_id", "") or "")] = window
        for item in tuple(case.get("items", ()) or ()):
            if isinstance(item, Mapping):
                items[str(item.get("item_id", "") or "")] = item
    return cases, items, windows


def _write_summary(
    out_root: Path,
    item_ids: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> Path:
    statuses = Counter(str(row.get("status", "") or "") for row in results.values())
    verdicts = Counter(str(row.get("verdict", "") or "") for row in results.values())
    payload = {
        "schema_version": "MMLifelongVisualProbeBatchSummaryV1",
        "contract": VISUAL_PROBE_CONTRACT,
        "selected_count": len(item_ids),
        "completed_count": len(results),
        "status_counts": dict(sorted(statuses.items())),
        "verdict_counts": dict(sorted(verdicts.items())),
        "actual_model": run_manifest["actual_model"],
        "case_results": [
            {
                "item_id": item_id,
                "status": results[item_id].get("status"),
                "verdict": results[item_id].get("verdict"),
                "attempt_count": int(results[item_id].get("attempt_count", 0) or 0),
                "resume_reused_success": bool(
                    results[item_id].get("resume_reused_success")
                ),
            }
            for item_id in item_ids
            if item_id in results
        ],
    }
    path = out_root / "visual_probe_batch_summary.json"
    _write_json_atomic(path, payload)
    return path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
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
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="planner_api")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--expected-items", type=int)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run_probe(_parse_args())
    print(f"VISUAL_PROBE_BATCH_DONE summary={summary}", flush=True)


if __name__ == "__main__":
    main()
