from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FIVE_SEED_CASE_IDS = frozenset({"441-2", "441-4", "445-2", "445-3"})
REPLAY_SCHEMA_VERSION = 3
REPLAY_COMPATIBILITY_FIELDS = (
    "case_id",
    "seed",
    "gold_option",
    "raw_option",
    "final_option",
    "correct",
    "reference_valid",
    "investigation_ordering",
    "execution_health",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_checksum(path: Path, *, block_size: int = 1024 * 1024) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {"path": str(target), "status": "missing", "sha256": "", "bytes": 0}
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return {
        "path": str(target),
        "status": "present",
        "sha256": digest.hexdigest(),
        "bytes": target.stat().st_size,
    }


def git_commit(cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else "unknown"


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"replay-{stamp}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class ImmutableRun:
    run_id: str
    root: Path
    config_hash: str


def create_immutable_run(
    runs_root: Path,
    *,
    run_id: str,
    config: Mapping[str, Any],
) -> ImmutableRun:
    normalized = str(run_id or "").strip()
    if not _RUN_ID_RE.fullmatch(normalized):
        raise ValueError("run_id must contain only letters, digits, '.', '_' or '-' and cannot start with punctuation")
    root = Path(runs_root) / normalized
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    (root / "workspaces").mkdir(exist_ok=False)
    normalized_config = dict(config)
    config_hash = stable_hash(normalized_config)
    _exclusive_json(
        root / "config.json",
        {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "run_id": normalized,
            "config_hash": config_hash,
            **normalized_config,
        },
    )
    return ImmutableRun(normalized, root, config_hash)


def write_immutable_summary(run: ImmutableRun, payload: Mapping[str, Any]) -> Path:
    path = run.root / "summary.json"
    _exclusive_json(path, {"run_id": run.run_id, "config_hash": run.config_hash, **dict(payload)})
    return path


def workspace_input_checksums(workspace: Any) -> dict[str, Any]:
    segments = tuple(getattr(getattr(workspace, "manifest", None), "segments", ()) or ())
    source_rows = [
        file_checksum(Path(path))
        for path in sorted({str(getattr(segment, "source_path", "") or "") for segment in segments})
    ]
    frame_manifest = file_checksum(Path(getattr(workspace, "frame_manifest")))
    window_manifest = file_checksum(
        Path(getattr(workspace, "root_dir")) / "observations" / "window_frame_manifest.jsonl"
    )
    return {
        "source_media": source_rows,
        "source_media_checksum": stable_hash(source_rows),
        "frame_manifest_checksum": str(frame_manifest["sha256"]),
        "window_frame_manifest_checksum": str(window_manifest["sha256"]),
    }


def replay_case_metadata(
    *,
    workspace_root: Path,
    case_summary: Mapping[str, Any],
    input_checksums: Mapping[str, Any],
    seed: int,
    provider_settings: Mapping[str, Mapping[str, Any]],
    gold_option: str,
) -> dict[str, Any]:
    """Build a compact replay record without retaining prompts or model output."""
    root = Path(workspace_root)
    interactions = _read_jsonl(root / "interactions.jsonl")
    prompts = [str(row.get("prompt", "") or "") for row in interactions]
    raw_responses = [str(row.get("raw", "") or "") for row in interactions]
    request_ids: list[str] = []
    retry_count = 0
    token_cost: Counter[str] = Counter()
    visual_input_count = 0
    length_count = 0
    format_repair_count = 0
    repair_failure_count = 0
    images_requested = images_attached = images_dropped = model_response_count = 0
    for row in interactions:
        format_repair_count += int(bool(row.get("format_repaired")))
        repair_failure_count += int(bool(row.get("repair_failed")))
        metadata = dict(row.get("api_response", {}) or {})
        request_id = str(metadata.get("provider_request_id", "") or "")
        if request_id and request_id in request_ids:
            continue
        if request_id:
            request_ids.append(request_id)
        retry_count += int(metadata.get("retry_count", 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            if isinstance(metadata.get(key), (int, float)):
                token_cost[key] += int(metadata[key])
        visual_input_count += len(tuple(row.get("image_paths", row.get("frame_paths", ())) or ()))
        finish_reason = str(metadata.get("finish_reason", "") or "").casefold()
        length_count += int(finish_reason == "length") + int(bool(metadata.get("truncated_then_retried")))
        images_requested += int(metadata.get("images_requested", 0) or 0)
        images_attached += int(metadata.get("images_attached", 0) or 0)
        images_dropped += int(metadata.get("images_dropped", 0) or 0)
        if finish_reason != "image_attachment_failed" and any(
            key in metadata
            for key in ("finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens", "provider_request_id")
        ):
            model_response_count += 1 + int(metadata.get("truncation_retry_count", 0) or 0)

    trace = tuple(
        dict(row) for row in tuple(case_summary.get("trace", ()) or ()) if isinstance(row, Mapping)
    )
    outcome = next((row for row in reversed(trace) if row.get("type") == "answer_outcome"), {})
    investigations = [
        {
            "round": int(row.get("round", 0) or 0),
            "tasks": [dict(task) for task in tuple(row.get("tasks", ()) or ()) if isinstance(task, Mapping)],
        }
        for row in trace
        if row.get("type") == "reasoner_decision" and str(row.get("action", "")) == "investigate"
    ]
    batches = [row for row in trace if row.get("type") == "investigator_batch"]
    proposed_tasks = sum(len(row["tasks"]) for row in investigations)
    dispatched_tasks = sum(int(row.get("requested_tasks", 0) or 0) for row in batches)
    executed_rounds = {int(row.get("round", 0) or 0) for row in batches if int(row.get("requested_tasks", 0) or 0)}
    planned_rounds = {int(row["round"]) for row in investigations if row["tasks"]}
    raw_answer = str(outcome.get("raw_reasoner_answer", "") or "")
    final_answer = str(case_summary.get("answer", outcome.get("answer", "")) or "")
    health = {
        "model_response_count": model_response_count,
        "finish_reason_length_count": length_count,
        "finish_reason_length_ratio": round(length_count / max(1, model_response_count), 6),
        "format_repair_count": format_repair_count,
        "repair_failure_count": repair_failure_count,
        "investigation_count": int(case_summary.get("investigation_count", 0) or 0),
        "reasoner_investigation_round_count": len(planned_rounds),
        "reasoner_executed_round_count": len(planned_rounds & executed_rounds),
        "reasoner_task_execution_rate": round(
            len(planned_rounds & executed_rounds) / max(1, len(planned_rounds)),
            6,
        ),
        "reasoner_task_count": proposed_tasks,
        "reasoner_dispatched_task_count": dispatched_tasks,
        "images_requested": images_requested,
        "images_attached": images_attached,
        "images_dropped": images_dropped,
    }
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "case_id": str(case_summary.get("case_id", root.name) or root.name),
        "seed": int(seed),
        "gold_option": str(gold_option or ""),
        "raw_reasoner_answer": raw_answer,
        "raw_option": _as_option(raw_answer),
        "final_answer": final_answer,
        "final_option": _as_option(final_answer),
        "correct": bool(case_summary.get("correct", False)),
        "reference_valid": bool(case_summary.get("reference_valid", outcome.get("reference_valid", False))),
        "reference_reason": str(case_summary.get("reference_reason", outcome.get("reference_reason", "")) or ""),
        "framework_answer_mutation": bool(outcome.get("framework_answer_mutation", False)),
        "prompt_hash": stable_hash(prompts),
        "raw_response_hash": stable_hash(raw_responses),
        "observation_log_checksum": str(file_checksum(root / "observation_log.jsonl")["sha256"]),
        "working_document_checksum": str(file_checksum(root / "working_document.json")["sha256"]),
        "workspace_ops_checksum": str(file_checksum(root / "workspace_ops.jsonl")["sha256"]),
        "provider_request_ids": request_ids,
        "provider_settings": {str(role): dict(settings) for role, settings in provider_settings.items()},
        "api_retry_count": retry_count,
        "investigation_ordering": investigations,
        "trace_checksum": str(file_checksum(root / "interactions.jsonl")["sha256"]),
        "frame_vlm_cost": {
            "prompt_tokens": int(token_cost["prompt_tokens"]),
            "completion_tokens": int(token_cost["completion_tokens"]),
            "reasoning_tokens": int(token_cost["reasoning_tokens"]),
            "visual_input_count": visual_input_count,
        },
        "execution_health": health,
        **dict(input_checksums),
    }


def compare_replay_records(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    compatibility_fields: Sequence[str] = REPLAY_COMPATIBILITY_FIELDS,
) -> dict[str, Any]:
    missing = [field for field in compatibility_fields if field not in current]
    type_mismatches = [
        field
        for field in compatibility_fields
        if field in baseline and field in current and type(baseline[field]) is not type(current[field])
    ]
    before = dict(baseline.get("execution_health", {}) or {})
    after = dict(current.get("execution_health", {}) or {})
    metric_keys = (
        "finish_reason_length_ratio",
        "format_repair_count",
        "repair_failure_count",
        "reasoner_task_execution_rate",
        "images_dropped",
    )
    delta = {
        key: round(float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0), 6)
        for key in metric_keys
    }
    delta["correct"] = int(bool(current.get("correct"))) - int(bool(baseline.get("correct")))
    delta["reference_valid"] = int(bool(current.get("reference_valid"))) - int(bool(baseline.get("reference_valid")))
    return {
        "format_compatible": not missing and not type_mismatches,
        "missing_fields": missing,
        "type_mismatches": type_mismatches,
        "behavior_delta": delta,
    }


def aggregate_seed_results(case_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in case_records:
        grouped[str(row.get("case_id", "") or "")].append(row)
    per_case = {}
    total_cost: Counter[str] = Counter()
    total_health: Counter[str] = Counter()
    total_correct = total_reference_valid = total_drift = 0
    for case_id, rows in sorted(grouped.items()):
        final_options = [str(row.get("final_option", "") or "") for row in rows]
        by_observation_log: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            checksum = str(row.get("observation_log_checksum", "") or "")
            option = str(row.get("final_option", "") or "")
            if checksum and option:
                by_observation_log[checksum].add(option)
            for key, value in dict(row.get("frame_vlm_cost", {}) or {}).items():
                if isinstance(value, (int, float)):
                    total_cost[str(key)] += int(value)
            for key, value in dict(row.get("execution_health", {}) or {}).items():
                if isinstance(value, (int, float)) and key not in {
                    "finish_reason_length_ratio",
                    "reasoner_task_execution_rate",
                }:
                    total_health[str(key)] += float(value)
        drift = sum(1 for options in by_observation_log.values() if len(options) > 1)
        correct = sum(bool(row.get("correct", False)) for row in rows)
        reference_valid = sum(bool(row.get("reference_valid", False)) for row in rows)
        total_correct += correct
        total_reference_valid += reference_valid
        total_drift += drift
        per_case[case_id] = {
            "run_count": len(rows),
            "final_answer_distribution": _distribution(final_options),
            "same_observation_answer_drift": drift,
            "accuracy": round(correct / max(1, len(rows)), 6),
            "reference_valid_rate": round(reference_valid / max(1, len(rows)), 6),
        }
    protocol = {
        case_id: {
            "required_seed_count": 5 if case_id in _FIVE_SEED_CASE_IDS else 3,
            "actual_seed_count": len({int(row.get("seed", 0) or 0) for row in rows}),
        }
        for case_id, rows in sorted(grouped.items())
    }
    for row in protocol.values():
        row["satisfied"] = row["actual_seed_count"] >= row["required_seed_count"]
    overall_health = dict(sorted(total_health.items()))
    overall_health["finish_reason_length_ratio"] = round(
        float(total_health["finish_reason_length_count"]) / max(1.0, float(total_health["model_response_count"])),
        6,
    )
    overall_health["reasoner_task_execution_rate"] = round(
        float(total_health["reasoner_executed_round_count"])
        / max(1.0, float(total_health["reasoner_investigation_round_count"])),
        6,
    )
    run_count = len(case_records)
    return {
        "per_case": per_case,
        "targeted_seed_protocol": protocol,
        "overall": {
            "case_count": len(per_case),
            "run_count": run_count,
            "correct_count": total_correct,
            "accuracy": round(total_correct / max(1, run_count), 6),
            "reference_valid_count": total_reference_valid,
            "reference_valid_rate": round(total_reference_valid / max(1, run_count), 6),
            "same_observation_answer_drift": total_drift,
            "frame_vlm_cost": dict(sorted(total_cost.items())),
            "execution_health": overall_health,
        },
    }


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return tuple(rows)


def _as_option(answer: Any) -> str:
    match = re.match(r"\s*([A-Z])(?:[.)\s:]|$)", str(answer or "").upper())
    return match.group(1) if match else ""


def _distribution(values: Sequence[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if str(value)).items()))
