"""Deterministic continuation planning for WP17 slot construction runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


WP17_SLOT_CONTINUATION_CONTRACT = "WP17-slot-construction-continuation-v1"
WP17_SLOT_CONTINUATION_PLAN_CONTRACT = "WP17-slot-continuation-plan-v1"


def continuation_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the result payload excluding hop-specific continuation metadata."""

    payload = dict(row)
    payload.pop("continuation_provenance", None)
    return payload


def cumulative_experiment_model_calls(summary: Mapping[str, Any]) -> int:
    """Return model calls accumulated through all continuation hops."""

    continuation = dict(summary.get("continuation", {}) or {})
    if "total_experiment_model_calls" in continuation:
        return int(continuation["total_experiment_model_calls"] or 0)
    return int(summary.get("model_calls", 0) or 0)


def build_continuation_entries(
    segments: Sequence[Mapping[str, Any]],
    parent_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Choose rows to reuse or rerun without consulting endpoint values.

    E1C0 has no cross-segment dependency. E1C1 carries the previous caption and
    E1C2 carries slot state, so either stateful arm is replayed from its first
    non-success row through the end of that window.
    """

    ordered = tuple(dict(row) for row in segments)
    expected_keys = {
        (str(segment["segment_id"]), arm)
        for segment in ordered
        for arm in ("e1c0", "e1c1", "e1c2")
    }
    if set(parent_rows) != expected_keys:
        missing = len(expected_keys - set(parent_rows))
        extra = len(set(parent_rows) - expected_keys)
        raise ValueError(
            f"WP17 continuation parent result set mismatch: missing={missing} extra={extra}"
        )

    suffix_starts: dict[tuple[str, str], int] = {}
    rerun_reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    for segment in ordered:
        segment_id = str(segment["segment_id"])
        window_id = str(segment["window_id"])
        ordinal = int(segment["window_segment_ordinal"])
        for arm in ("e1c0", "e1c1", "e1c2"):
            row = parent_rows[(segment_id, arm)]
            if row.get("status") == "success":
                continue
            code = str(row.get("failure_code", "non_success"))
            key = (segment_id, arm)
            rerun_reasons[key].add(f"parent_{code}")
            if arm in {"e1c1", "e1c2"}:
                suffix_key = (arm, window_id)
                suffix_starts[suffix_key] = min(
                    ordinal, suffix_starts.get(suffix_key, ordinal)
                )

    for segment in ordered:
        segment_id = str(segment["segment_id"])
        window_id = str(segment["window_id"])
        ordinal = int(segment["window_segment_ordinal"])
        for arm in ("e1c1", "e1c2"):
            first = suffix_starts.get((arm, window_id))
            if first is not None and ordinal >= first:
                rerun_reasons[(segment_id, arm)].add(
                    f"{arm}_window_suffix_from_{first:04d}"
                )

    entries = []
    for segment in ordered:
        segment_id = str(segment["segment_id"])
        for arm in ("e1c0", "e1c1", "e1c2"):
            key = (segment_id, arm)
            reasons = sorted(rerun_reasons.get(key, ()))
            entries.append(
                {
                    "segment_id": segment_id,
                    "arm": arm,
                    "action": "rerun" if reasons else "reuse",
                    "reasons": reasons or ["parent_success_dependency_intact"],
                }
            )
    return tuple(entries)


def index_continuation_entries(
    plan: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    if plan.get("contract") != WP17_SLOT_CONTINUATION_PLAN_CONTRACT:
        raise ValueError("WP17 continuation plan contract mismatch")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in tuple(plan.get("entries", ()) or ()):
        row = dict(raw)
        key = (str(row["segment_id"]), str(row["arm"]))
        if key in rows:
            raise ValueError("WP17 continuation plan has duplicate result keys")
        if row.get("action") not in {"reuse", "rerun"}:
            raise ValueError("WP17 continuation action must be reuse or rerun")
        rows[key] = row
    return rows
