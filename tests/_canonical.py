from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DROPPED_KEYS = {
    "timestamp",
    "created_at",
    "wall_time_ms",
    "absolute_path",
    "run_id",
    "run_root",
}

UNORDERED_LIST_KEYS = {
    "candidate_segments",
    "target_entities",
}


def canonicalize_event(event: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sorted(event.items()):
        if key in DROPPED_KEYS:
            continue
        out[key] = _canonicalize_value(value, parent_key=key)
    return out


def canonical_trace_diff(actual_path: str | Path, baseline_path: str | Path) -> list[dict[str, Any]]:
    actual_events = _read_canonical_jsonl(Path(actual_path))
    baseline_events = _read_canonical_jsonl(Path(baseline_path))
    diffs: list[dict[str, Any]] = []
    max_len = max(len(actual_events), len(baseline_events))
    for index in range(max_len):
        actual = actual_events[index] if index < len(actual_events) else None
        expected = baseline_events[index] if index < len(baseline_events) else None
        if actual != expected:
            diffs.append({"index": index, "actual": actual, "expected": expected})
    return diffs


def _canonicalize_value(value: Any, *, parent_key: str) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return canonicalize_event(value)
    if isinstance(value, list):
        items = [_canonicalize_value(item, parent_key=parent_key) for item in value]
        if parent_key in UNORDERED_LIST_KEYS:
            return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True))
        return items
    return value


def _read_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            events.append(canonicalize_event(payload))
    return events
