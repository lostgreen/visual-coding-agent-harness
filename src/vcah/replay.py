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
REPLAY_SCHEMA_VERSION = 2
REPLAY_COMPATIBILITY_FIELDS = (
    "case_id",
    "seed",
    "gold_option",
    "raw_option",
    "final_option",
    "correct",
    "answer_mode",
    "grounding_status",
    "investigation_ordering",
    "execution_health",
)


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible replay metadata."""
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_checksum(path: Path, *, block_size: int = 1024 * 1024) -> dict[str, Any]:
    """Describe a file without reading it into memory or emitting its contents."""
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


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


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
    """Create a replay root exactly once; an existing run ID is always an error."""
    normalized_id = str(run_id or "").strip()
    if not _RUN_ID_RE.fullmatch(normalized_id):
        raise ValueError("run_id must contain only letters, digits, '.', '_' or '-' and cannot start with punctuation")
    root_parent = Path(runs_root)
    root_parent.mkdir(parents=True, exist_ok=True)
    root = root_parent / normalized_id
    root.mkdir(exist_ok=False)
    (root / "workspaces").mkdir(exist_ok=False)
    normalized_config = dict(config)
    config_hash = stable_hash(normalized_config)
    _exclusive_json(
        root / "config.json",
        {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "run_id": normalized_id,
            "config_hash": config_hash,
            **normalized_config,
        },
    )
    return ImmutableRun(run_id=normalized_id, root=root, config_hash=config_hash)


def write_immutable_summary(run: ImmutableRun, payload: Mapping[str, Any]) -> Path:
    path = run.root / "summary.json"
    _exclusive_json(path, {"run_id": run.run_id, "config_hash": run.config_hash, **dict(payload)})
    return path


def workspace_input_checksums(workspace: Any) -> dict[str, Any]:
    """Hash source media and frame manifests referenced by one virtual workspace."""
    segments = tuple(getattr(getattr(workspace, "manifest", None), "segments", ()) or ())
    source_rows = []
    for path_text in sorted({str(getattr(segment, "source_path", "") or "") for segment in segments}):
        source_rows.append(file_checksum(Path(path_text)))
    frame_manifest = file_checksum(Path(getattr(workspace, "frame_manifest")))
    window_manifest = file_checksum(Path(getattr(workspace, "root_dir")) / "observations" / "window_frame_manifest.jsonl")
    return {
        "source_media": source_rows,
        "source_media_checksum": stable_hash(source_rows),
        "frame_manifest_checksum": str(frame_manifest["sha256"]),
        "window_frame_manifest_checksum": str(window_manifest["sha256"]),
    }


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not Path(path).is_file():
        return ()
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
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


def replay_case_metadata(
    *,
    workspace_root: Path,
    case_summary: Mapping[str, Any],
    input_checksums: Mapping[str, Any],
    seed: int,
    provider_settings: Mapping[str, Mapping[str, Any]],
    gold_option: str,
) -> dict[str, Any]:
    """Build a compact, content-free replay record for a completed case."""
    root = Path(workspace_root)
    interactions = _read_jsonl(root / "interactions.jsonl")
    prompts = [str(row.get("prompt", "") or "") for row in interactions]
    raw_responses = [str(row.get("raw", "") or "") for row in interactions]
    request_ids: list[str] = []
    retry_count = 0
    costs: Counter[str] = Counter()
    visual_input_count = 0
    length_count = 0
    format_repair_count = 0
    repair_failure_count = 0
    images_requested = 0
    images_attached = 0
    images_dropped = 0
    model_response_count = 0
    for row in interactions:
        metadata = dict(row.get("api_response", {}) or {})
        request_id = str(metadata.get("provider_request_id", "") or "")
        duplicate_request = bool(request_id and request_id in request_ids)
        if request_id and request_id not in request_ids:
            request_ids.append(request_id)
        if duplicate_request:
            continue
        retry_count += int(metadata.get("retry_count", 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                costs[key] += int(value)
        visual_input_count += len(tuple(row.get("image_paths", row.get("frame_paths", ())) or ()))
        length_count += int(str(metadata.get("finish_reason", "") or "").casefold() == "length")
        length_count += int(bool(metadata.get("truncated_then_retried")))
        format_repair_count += int(bool(metadata.get("format_repaired")))
        repair_failure_count += int(bool(metadata.get("repair_failed")))
        images_requested += int(metadata.get("images_requested", 0) or 0)
        images_attached += int(metadata.get("images_attached", 0) or 0)
        images_dropped += int(metadata.get("images_dropped", 0) or 0)
        finish_reason = str(metadata.get("finish_reason", "") or "").casefold()
        is_model_response = finish_reason != "image_attachment_failed" and any(
            key in metadata
            for key in (
                "finish_reason",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "provider_request_id",
            )
        )
        if is_model_response:
            model_response_count += 1 + int(metadata.get("truncation_retry_count", 0) or 0)

    logical_trace = tuple(
        dict(row) for row in tuple(case_summary.get("trace", ()) or ()) if isinstance(row, Mapping)
    )
    outcome = next((row for row in reversed(logical_trace) if row.get("type") == "answer_outcome"), {})
    selection_events = [row for row in logical_trace if row.get("type") == "answer_selection_event"]
    investigations = [
        {
            "round": int(row.get("round", 0) or 0),
            "task_count": int(row.get("task_count", 0) or 0),
            "tasks": [dict(task) for task in tuple(row.get("tasks", ()) or ()) if isinstance(task, Mapping)],
        }
        for row in logical_trace
        if row.get("type") == "reasoner_decision" and str(row.get("action", "")) == "investigate"
    ]
    blockers = [
        str(item)
        for item in tuple(outcome.get("final_adjudication_blockers", ()) or ())
        if str(item)
    ]
    raw_answer = str(outcome.get("raw_reasoner_answer", "") or "")
    final_answer = str(case_summary.get("answer", outcome.get("answer", "")) or "")
    raw_option = _as_option(raw_answer)
    final_option = _as_option(final_answer)
    investigator_batches = [row for row in logical_trace if row.get("type") == "investigator_batch"]
    eligible_reasoner_rounds = {
        int(row.get("round", 0) or 0)
        for row in logical_trace
        if row.get("type") == "reasoner_decision"
        and str(row.get("action", "") or "") == "investigate"
        and int(row.get("task_count", 0) or 0) > 0
        and int(row.get("remaining_budget", 0) or 0) > 0
    }
    executed_reasoner_rounds = {
        int(row.get("round", 0) or 0)
        for row in investigator_batches
        if int(row.get("requested_tasks", 0) or 0) > 0
    }
    proposed_reasoner_tasks = sum(
        int(row.get("task_count", 0) or 0)
        for row in logical_trace
        if row.get("type") == "reasoner_decision"
        and str(row.get("action", "") or "") == "investigate"
        and int(row.get("remaining_budget", 0) or 0) > 0
    )
    dispatched_reasoner_tasks = sum(int(row.get("requested_tasks", 0) or 0) for row in investigator_batches)
    system_tasks = sum(int(row.get("system_suggestion_tasks", 0) or 0) for row in investigator_batches)
    accepted_tasks = sum(int(row.get("accepted_tasks", 0) or 0) for row in investigator_batches)
    no_information_gain_tasks = sum(
        int(row.get("no_information_gain_tasks", 0) or 0) for row in investigator_batches
    )
    override_count = sum(
        1
        for row in logical_trace
        if row.get("type") in {"repair_override", "navigation_drilldown_override", "candidate_dispatch_conversion"}
    )
    budget_limit = int(outcome.get("investigation_budget_limit", 0) or 0)
    cumulative_accepted = 0
    budget_exhausted_round = 0
    for row in sorted(investigator_batches, key=lambda item: int(item.get("round", 0) or 0)):
        cumulative_accepted += int(row.get("accepted_tasks", 0) or 0)
        if budget_limit > 0 and cumulative_accepted >= budget_limit:
            budget_exhausted_round = int(row.get("round", 0) or 0)
            break
    execution_health = {
        "model_response_count": model_response_count,
        "finish_reason_length_count": length_count,
        "finish_reason_length_ratio": round(length_count / max(1, model_response_count), 6),
        "format_repair_count": format_repair_count,
        "repair_failure_count": repair_failure_count,
        "accepted_investigation_count": int(case_summary.get("accepted_investigations", accepted_tasks) or 0),
        "no_information_gain_task_count": no_information_gain_tasks,
        "system_suggestion_task_count": system_tasks,
        "system_task_budget_ratio": round(
            float(outcome.get("system_task_budget_ratio_actual", system_tasks / max(1, accepted_tasks)) or 0.0),
            6,
        ),
        "system_override_count": override_count,
        "reasoner_investigation_round_count": len(eligible_reasoner_rounds),
        "reasoner_executed_round_count": len(eligible_reasoner_rounds & executed_reasoner_rounds),
        "reasoner_task_execution_rate": round(
            len(eligible_reasoner_rounds & executed_reasoner_rounds) / max(1, len(eligible_reasoner_rounds)),
            6,
        ),
        "reasoner_task_count": proposed_reasoner_tasks,
        "reasoner_dispatched_task_count": dispatched_reasoner_tasks,
        "budget_exhausted_round": budget_exhausted_round,
        "budget_exhausted_before_round3": bool(budget_exhausted_round and budget_exhausted_round < 3),
        "images_requested": images_requested,
        "images_attached": images_attached,
        "images_dropped": images_dropped,
        "forced_choice": str(case_summary.get("answer_mode", outcome.get("answer_mode", "")) or "") == "forced_choice",
    }
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "case_id": str(case_summary.get("case_id", root.name) or root.name),
        "seed": int(seed),
        "gold_option": str(gold_option or ""),
        "raw_reasoner_answer": raw_answer,
        "raw_option": raw_option,
        "final_answer": final_answer,
        "final_option": final_option,
        "correct": bool(case_summary.get("correct", False)),
        "answer_mode": str(case_summary.get("answer_mode", outcome.get("answer_mode", "")) or ""),
        "grounding_status": str(case_summary.get("grounding_status", outcome.get("grounding_status", "")) or ""),
        "final_selection_source": str(outcome.get("final_selection_source", "") or ""),
        "answer_selection_event_count": len(selection_events),
        "answer_mutation_events": [
            dict(row) for row in tuple(outcome.get("answer_mutation_events", ()) or ()) if isinstance(row, Mapping)
        ],
        "qualified_event_count": int(dict(outcome.get("canonical_fact_counts", {}) or {}).get("qualified_events", 0) or 0),
        "selected_episode_id": str(outcome.get("selected_episode_id", "") or ""),
        "final_adjudication_blockers": blockers,
        "evidence_digest_hash": str(dict(outcome.get("revision_context", {}) or {}).get("evidence_digest_hash", "") or ""),
        "prompt_hash": stable_hash(prompts),
        "raw_response_hash": stable_hash(raw_responses),
        "provider_request_ids": request_ids,
        "provider_settings": {str(role): dict(settings) for role, settings in provider_settings.items()},
        "api_retry_count": retry_count,
        "investigation_ordering": investigations,
        "trace_checksum": str(file_checksum(root / "interactions.jsonl")["sha256"]),
        "frame_vlm_cost": {
            "prompt_tokens": int(costs["prompt_tokens"]),
            "completion_tokens": int(costs["completion_tokens"]),
            "reasoning_tokens": int(costs["reasoning_tokens"]),
            "visual_input_count": visual_input_count,
        },
        "execution_health": execution_health,
        **dict(input_checksums),
    }


def compare_replay_records(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    compatibility_fields: Sequence[str] = REPLAY_COMPATIBILITY_FIELDS,
) -> dict[str, Any]:
    """Compare replay format separately from behavior so schema churn cannot mask regressions."""
    missing = [field for field in compatibility_fields if field not in current]
    type_mismatches = [
        field
        for field in compatibility_fields
        if field in baseline and field in current and type(baseline[field]) is not type(current[field])
    ]
    baseline_health = dict(baseline.get("execution_health", {}) or {})
    current_health = dict(current.get("execution_health", {}) or {})
    metric_keys = (
        "finish_reason_length_ratio",
        "format_repair_count",
        "repair_failure_count",
        "no_information_gain_task_count",
        "system_task_budget_ratio",
        "system_override_count",
        "reasoner_task_execution_rate",
        "budget_exhausted_before_round3",
        "images_dropped",
    )
    semantic_delta = {
        key: round(float(current_health.get(key, 0.0) or 0.0) - float(baseline_health.get(key, 0.0) or 0.0), 6)
        for key in metric_keys
    }
    semantic_delta["correct"] = int(bool(current.get("correct"))) - int(bool(baseline.get("correct")))
    semantic_delta["forced_choice"] = int(bool(current_health.get("forced_choice"))) - int(
        bool(baseline_health.get("forced_choice"))
    )
    return {
        "format_compatible": not missing and not type_mismatches,
        "missing_fields": missing,
        "type_mismatches": type_mismatches,
        "semantic_delta": semantic_delta,
    }


def aggregate_seed_results(case_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate answers and semantic state by case without retaining model text."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in case_records:
        grouped[str(row.get("case_id", "") or "")].append(row)
    per_case: dict[str, dict[str, Any]] = {}
    total_raw_correct_to_final_wrong = 0
    total_same_evidence_drift = 0
    total_cost: Counter[str] = Counter()
    total_health: Counter[str] = Counter()
    total_correct = 0
    total_forced = 0
    for case_id, rows in sorted(grouped.items()):
        raw_answers = [str(row.get("raw_option", "") or "") for row in rows]
        final_answers = [str(row.get("final_option", "") or "") for row in rows]
        qualified_counts = [int(row.get("qualified_event_count", 0) or 0) for row in rows]
        episodes = [str(row.get("selected_episode_id", "") or "") for row in rows]
        blockers = [
            blocker
            for row in rows
            for blocker in tuple(row.get("final_adjudication_blockers", ()) or ())
            if str(blocker)
        ]
        mutation_sources = [
            str(event.get("source", "") or "")
            for row in rows
            for event in tuple(row.get("answer_mutation_events", ()) or ())
            if isinstance(event, Mapping) and str(event.get("source", ""))
        ]
        selection_sources = [str(row.get("final_selection_source", "") or "") for row in rows]
        raw_correct_to_final_wrong = sum(
            1
            for row in rows
            if str(row.get("raw_option", "") or "") == str(row.get("gold_option", "") or "")
            and not bool(row.get("correct", False))
        )
        by_evidence: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            digest = str(row.get("evidence_digest_hash", "") or "")
            answer = str(row.get("final_option", "") or "")
            if digest and answer:
                by_evidence[digest].add(answer)
        same_evidence_drift = sum(1 for answers in by_evidence.values() if len(answers) > 1)
        for row in rows:
            for key, value in dict(row.get("frame_vlm_cost", {}) or {}).items():
                if isinstance(value, (int, float)):
                    total_cost[str(key)] += int(value)
            for key, value in dict(row.get("execution_health", {}) or {}).items():
                if isinstance(value, bool):
                    total_health[str(key)] += int(value)
                elif isinstance(value, (int, float)) and key not in {
                    "finish_reason_length_ratio", "system_task_budget_ratio", "reasoner_task_execution_rate"
                }:
                    total_health[str(key)] += float(value)
            total_correct += int(bool(row.get("correct", False)))
            total_forced += int(str(row.get("answer_mode", "") or "") == "forced_choice")
        total_raw_correct_to_final_wrong += raw_correct_to_final_wrong
        total_same_evidence_drift += same_evidence_drift
        per_case[case_id] = {
            "run_count": len(rows),
            "raw_answer_distribution": _distribution(raw_answers),
            "final_answer_distribution": _distribution(final_answers),
            "qualified_event_count_distribution": _distribution(qualified_counts),
            "selected_episode_distribution": _distribution(episodes),
            "blocker_distribution": _distribution(blockers),
            "answer_mutation_source_distribution": _distribution((*mutation_sources, *selection_sources)),
            "raw_correct_to_final_wrong": raw_correct_to_final_wrong,
            "same_evidence_answer_drift": same_evidence_drift,
            "accuracy": round(sum(bool(row.get("correct", False)) for row in rows) / max(1, len(rows)), 6),
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
    overall_health["system_task_budget_ratio"] = round(
        float(total_health["system_suggestion_task_count"])
        / max(1.0, float(total_health["accepted_investigation_count"])),
        6,
    )
    overall_health["reasoner_task_execution_rate"] = round(
        float(total_health["reasoner_executed_round_count"])
        / max(1.0, float(total_health["reasoner_investigation_round_count"])),
        6,
    )
    return {
        "per_case": per_case,
        "targeted_seed_protocol": protocol,
        "overall": {
            "case_count": len(per_case),
            "run_count": len(case_records),
            "raw_correct_to_final_wrong": total_raw_correct_to_final_wrong,
            "same_evidence_answer_drift": total_same_evidence_drift,
            "correct_count": total_correct,
            "accuracy": round(total_correct / max(1, len(case_records)), 6),
            "forced_choice_count": total_forced,
            "forced_choice_rate": round(total_forced / max(1, len(case_records)), 6),
            "frame_vlm_cost": dict(sorted(total_cost.items())),
            "execution_health": overall_health,
        },
    }
