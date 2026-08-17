#!/usr/bin/env python3
"""Judge blinded negative-sidecar rows one item per independent model call."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import threading
import time
from typing import Any, Mapping, Sequence

from vcah.model_client import OpenAICompatibleClient
from vcah.occurrence_negative_sidecar import (
    file_sha256,
    safe_response_metadata,
    stable_digest,
)


MAX_WORKERS = 16
VALID_VERDICTS = frozenset({"true_contradiction", "false_contradiction", "unclear"})


def run_batch(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()) and not args.resume:
        raise FileExistsError(f"judgment output is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tasks").mkdir(exist_ok=True)
    blind = _read_json(Path(args.items_json))
    key = _read_json(Path(args.key_json))
    _validate_bundle(blind, key)
    tasks = _judgment_tasks(blind, key, seed=int(args.seed))
    client = OpenAICompatibleClient.from_yaml(Path(args.config), section=args.section)
    workers = max(1, min(MAX_WORKERS, int(args.workers), len(tasks)))
    protocol_digest = str(blind["judgment_protocol_digest"])
    manifest = {
        "schema_version": "MMLifelongOccurrenceNegativeRowJudgeRunV1",
        "items_sha256": file_sha256(Path(args.items_json)),
        "key_sha256": file_sha256(Path(args.key_json)),
        "config_sha256": file_sha256(Path(args.config)),
        "api_section": str(args.section),
        "actual_model": str(client.model),
        "judgment_protocol_digest": protocol_digest,
        "primary_item_count": int(blind["item_count"]),
        "reliability_item_count": int(key["reliability_sample_count"]),
        "task_count": len(tasks),
        "one_item_per_call": True,
        "shared_conversation_context": False,
        "max_completion_tokens": max(4096, int(args.max_completion_tokens)),
        "judge_max_retries": max(0, int(args.judge_max_retries)),
        "workers": workers,
        "seed": int(args.seed),
        "temperature": client.replay_settings.get("temperature"),
        "top_p": client.replay_settings.get("top_p"),
        "requested_seed": client.replay_settings.get("requested_seed"),
        "provider_seed_supported": client.replay_settings.get(
            "provider_seed_supported"
        ),
        "provider_reported_seed_support": client.replay_settings.get(
            "provider_reported_seed_support"
        ),
        "raw_response_persisted": False,
        "prompt_persisted": False,
    }
    _write_json_atomic(out_root / "run_manifest.json", manifest)
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_judge_task, task, client, blind, args): task["task_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "status": "orchestrator_failed",
                    "error_type": type(exc).__name__,
                }
            with lock:
                results[task_id] = result
                _write_outputs(out_root, tasks, results, manifest)
            print(
                f"ROW_JUDGE_DONE task_id={task_id} status={result['status']}",
                flush=True,
            )
    output_path = _write_outputs(out_root, tasks, results, manifest)
    if Counter(row["status"] for row in results.values()).get("success", 0) != len(
        tasks
    ):
        raise SystemExit(1)
    return output_path


def _validate_bundle(blind: Mapping[str, Any], key: Mapping[str, Any]) -> None:
    items = tuple(blind.get("items", ()) or ())
    if len(items) != int(blind.get("item_count", -1)):
        raise ValueError("blind item count mismatch")
    if len(items) != int(key.get("item_count", -1)):
        raise ValueError("blind/key item count mismatch")
    if stable_digest(items) != str(key.get("blind_items_digest", "") or ""):
        raise ValueError("blind/key item digest mismatch")
    if str(blind.get("judgment_protocol_digest", "") or "") != str(
        key.get("judgment_protocol_digest", "") or ""
    ):
        raise ValueError("blind/key judgment protocol mismatch")
    item_ids = {
        str(row.get("audit_item_id", "") or "")
        for row in items
        if isinstance(row, Mapping)
    }
    sampled = {
        str(value)
        for value in tuple(key.get("reliability_sample_item_ids", ()) or ())
        if str(value)
    }
    if not sampled <= item_ids:
        raise ValueError("reliability sample is outside blind items")


def _judgment_tasks(
    blind: Mapping[str, Any], key: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, Any], ...]:
    items = {
        str(row.get("audit_item_id", "") or ""): dict(row)
        for row in tuple(blind.get("items", ()) or ())
        if isinstance(row, Mapping)
    }
    tasks: list[dict[str, Any]] = []
    for item_id, item in items.items():
        tasks.append(_task(item_id, item, kind="primary"))
    for item_id in tuple(key.get("reliability_sample_item_ids", ()) or ()):
        normalized_id = str(item_id)
        tasks.append(_task(normalized_id, items[normalized_id], kind="reliability"))
    random.Random(seed).shuffle(tasks)
    return tuple(tasks)


def _task(audit_item_id: str, item: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    task_id = stable_digest({"audit_item_id": audit_item_id, "judgment_kind": kind})[
        :24
    ]
    return {
        "task_id": task_id,
        "audit_item_id": audit_item_id,
        "judgment_kind": kind,
        "item": dict(item),
        "item_digest": stable_digest(item),
    }


def _judge_task(
    task: Mapping[str, Any],
    client: OpenAICompatibleClient,
    blind: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    result_path = Path(args.out_root) / "tasks" / f"{task_id}.json"
    prior = _read_json(result_path) if args.resume and result_path.is_file() else None
    protocol_digest = str(blind["judgment_protocol_digest"])
    if prior is not None and all(
        (
            prior.get("status") == "success",
            prior.get("actual_model") == client.model,
            prior.get("item_digest") == task["item_digest"],
            prior.get("judgment_protocol_digest") == protocol_digest,
        )
    ):
        return {**prior, "resume_reused_success": True}
    prompt = _judgment_prompt(
        task["item"],
        protocol=blind["judgment_protocol"],
        instance_nonce=task_id,
    )
    started = time.monotonic()
    attempt_history: list[dict[str, Any]] = []
    verdict = ""
    for attempt_index in range(max(0, int(args.judge_max_retries)) + 1):
        try:
            raw = client.chat(
                prompt,
                max_tokens=max(4096, int(args.max_completion_tokens)),
                response_format={"type": "json_object"},
            )
            metadata = safe_response_metadata(client.last_response_metadata)
        except Exception as exc:
            attempt_history.append(
                {
                    "attempt_index": attempt_index + 1,
                    "status": "model_failed",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        verdict = _parse_verdict(raw)
        attempt_history.append(
            {
                "attempt_index": attempt_index + 1,
                "status": "success" if verdict else "invalid_verdict",
                "model_response_digest": stable_digest(raw),
                "response_metadata": metadata,
            }
        )
        if verdict:
            break
    result = {
        "schema_version": "MMLifelongOccurrenceNegativeRowJudgmentV1",
        "task_id": task_id,
        "audit_item_id": str(task["audit_item_id"]),
        "judgment_kind": str(task["judgment_kind"]),
        "status": "success" if verdict else "failed",
        "verdict": verdict,
        "actual_model": str(client.model),
        "item_digest": str(task["item_digest"]),
        "judgment_protocol_digest": protocol_digest,
        "prompt_digest": stable_digest(prompt),
        "duration_sec": round(time.monotonic() - started, 3),
        "attempt_count": len(attempt_history),
        "attempt_history": attempt_history,
        "raw_response_persisted": False,
        "prompt_persisted": False,
    }
    _write_json_atomic(result_path, result)
    return result


def _judgment_prompt(
    item: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    instance_nonce: str,
) -> str:
    blinded_item = {
        key: value
        for key, value in item.items()
        if key not in {"audit_item_id", "allowed_verdicts"}
    }
    payload = {
        "protocol": dict(protocol),
        "independent_instance": instance_nonce,
        "blinded_item": blinded_item,
    }
    return (
        "Judge one blinded citation-level contradiction claim. Use only the "
        "cited visible caption passages. Absence of support is not "
        "contradiction. Return exactly one JSON object with no explanation: "
        '{"verdict":"true_contradiction|false_contradiction|unclear"}.\n\n'
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _parse_verdict(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, Mapping) or set(payload) != {"verdict"}:
        return ""
    verdict = str(payload.get("verdict", "") or "").strip().casefold()
    return verdict if verdict in VALID_VERDICTS else ""


def _write_outputs(
    out_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Path:
    successful = [row for row in results.values() if row.get("status") == "success"]
    path = out_root / "judgments.json"
    payload = {
        "schema_version": "MMLifelongOccurrenceNegativeRowJudgmentsV1",
        "judgment_protocol_digest": manifest["judgment_protocol_digest"],
        "actual_model": manifest["actual_model"],
        "primary_judgment_count": sum(
            row.get("judgment_kind") == "primary" for row in successful
        ),
        "reliability_judgment_count": sum(
            row.get("judgment_kind") == "reliability" for row in successful
        ),
        "judgments": [
            _compact_judgment(row)
            for row in successful
            if row.get("judgment_kind") == "primary"
        ],
        "reliability_judgments": [
            _compact_judgment(row)
            for row in successful
            if row.get("judgment_kind") == "reliability"
        ],
    }
    payload["judgments"].sort(key=lambda row: row["audit_item_id"])
    payload["reliability_judgments"].sort(key=lambda row: row["audit_item_id"])
    _write_json_atomic(path, payload)
    _write_json_atomic(
        out_root / "summary.json",
        {
            "schema_version": "MMLifelongOccurrenceNegativeRowJudgeSummaryV1",
            "selected_task_count": len(tasks),
            "completed_task_count": len(results),
            "status_counts": dict(
                sorted(
                    Counter(
                        row.get("status", "unknown") for row in results.values()
                    ).items()
                )
            ),
            "primary_success_count": payload["primary_judgment_count"],
            "reliability_success_count": payload["reliability_judgment_count"],
            "actual_model": manifest["actual_model"],
            "judgment_protocol_digest": manifest["judgment_protocol_digest"],
        },
    )
    return path


def _compact_judgment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audit_item_id": str(row["audit_item_id"]),
        "judgment_task_id": str(row["task_id"]),
        "verdict": str(row["verdict"]),
        "actual_model": str(row["actual_model"]),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-json", required=True)
    parser.add_argument("--key-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="planner_api")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = run_batch(args)
    print(f"NEGATIVE_ROW_JUDGE_DONE output={output}", flush=True)


if __name__ == "__main__":
    main()
