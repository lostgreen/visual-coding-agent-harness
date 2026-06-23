"""Exploration and exclusion ledger for workspace-v2 planning."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


def empty_search_ledger() -> dict[str, Any]:
    return {"records": [], "candidates": [], "options": {}, "answer_options": {}}


def norm_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def update_search_ledger(
    snapshot: Mapping[str, Any] | None,
    *,
    observation_id: str,
    tool_name: str,
    raw_output: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    ledger = _coerce_snapshot(snapshot)
    raw = dict(raw_output)
    events: list[tuple[str, dict[str, Any]]] = []
    if tool_name == "explore":
        raw, explore_events = _maybe_soft_recovery_hint(ledger, observation_id=observation_id, raw_output=raw)
        events.extend(explore_events)
        events.extend(_update_after_explore(ledger, observation_id=observation_id, raw_output=raw))
    elif tool_name == "verify_window":
        events.extend(_update_after_verify(ledger, observation_id=observation_id, raw_output=raw))
    elif tool_name == "synthesize_memory":
        events.extend(_update_after_synthesize(ledger, observation_id=observation_id, raw_output=raw))
    return ledger, raw, events


def render_search_ledger(snapshot: Mapping[str, Any], *, max_items: int = 3) -> str:
    ledger = _coerce_snapshot(snapshot)
    lines: list[str] = ["## Exploration Ledger"]
    records = list(ledger["records"])[-max_items:]
    if records:
        for record in records:
            query = record.get("query_norm") or "(unknown query)"
            candidate_count = len(record.get("candidate_keys") or [])
            lines.append(f'- "{query}" -> {candidate_count} candidate(s), status={record.get("status") or "-"}')
    else:
        lines.append("(none)")

    pending = [item for item in ledger["candidates"] if item.get("status") == "pending"]
    lines.extend(["", "## Pending Candidates"])
    if pending:
        for candidate in pending[:max_items]:
            lines.append(
                "- "
                + str(candidate.get("candidate_key") or candidate.get("event_id") or "-")
                + f" segment={candidate.get('segment_id') or '-'}"
                + f" time={_format_time_range(candidate.get('time_range')) or '-'}"
                + " recommended=verify_window"
            )
    else:
        lines.append("(none)")

    if ledger["options"]:
        lines.extend(["", "## Option Coverage"])
        for option_id, option in sorted(ledger["options"].items())[:max_items]:
            reason = str(option.get("reason") or "").strip()
            suffix = f" reason={reason}" if reason else ""
            lines.append(f"- {option_id}: {option.get('status') or 'unknown'}{suffix}")
        untested = [
            option_id
            for option_id in sorted(ledger.get("answer_options", {}))
            if str(ledger["options"].get(option_id, {}).get("status") or "untested") == "untested"
        ]
        if untested:
            lines.append("Untested: " + ", ".join(untested))

    event_candidates = [item for item in ledger["candidates"] if item.get("event_id") or item.get("event_type")]
    if event_candidates:
        lines.extend(["", "## Counting Ledger"])
        for candidate in event_candidates[:max_items]:
            lines.append(
                "- "
                + str(candidate.get("event_id") or candidate.get("candidate_key") or "-")
                + f" type={candidate.get('event_type') or '-'}"
                + f" status={candidate.get('status') or 'pending'}"
            )

    recommended = _recommended_next_actions(ledger)
    if recommended:
        lines.extend(["", "## Recommended Next"])
        for action in recommended[:max_items]:
            if action.get("candidate_key"):
                lines.append(f"- {action.get('tool')}({action.get('candidate_key')})")
            elif action.get("segment_id") and action.get("time_range"):
                suffix = f": {action.get('focus')}" if action.get("focus") else ""
                lines.append(
                    f"- {action.get('tool')} segment={action.get('segment_id')} "
                    f"time={_format_time_range(action.get('time_range'))}{suffix}"
                )
            elif action.get("focus"):
                lines.append(f"- {action.get('tool')}: {action.get('focus')}")
            else:
                lines.append(f"- {action.get('tool')}")
    return "\n".join(lines)


def _coerce_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    ledger = empty_search_ledger()
    if isinstance(snapshot, Mapping):
        records = snapshot.get("records")
        candidates = snapshot.get("candidates")
        options = snapshot.get("options")
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            ledger["records"] = [dict(item) for item in records if isinstance(item, Mapping)]
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            ledger["candidates"] = [dict(item) for item in candidates if isinstance(item, Mapping)]
        if isinstance(options, Mapping):
            ledger["options"] = {str(key): dict(value) for key, value in options.items() if isinstance(value, Mapping)}
        answer_options = snapshot.get("answer_options")
        if isinstance(answer_options, Mapping):
            ledger["answer_options"] = {str(key): str(value) for key, value in answer_options.items()}
    return ledger


def _maybe_soft_recovery_hint(
    ledger: dict[str, Any],
    *,
    observation_id: str,
    raw_output: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    query_norm = norm_query(raw_output.get("query") or raw_output.get("claim") or raw_output.get("question"))
    if not query_norm:
        return dict(raw_output), []
    repeat_count = sum(1 for record in ledger["records"] if record.get("query_norm") == query_norm)
    if repeat_count <= 0:
        return dict(raw_output), []
    pending = [candidate for candidate in ledger["candidates"] if candidate.get("status") == "pending"]
    events = [
        (
            "repeated_explore_detected",
            {
                "observation_id": observation_id,
                "query": str(raw_output.get("query") or ""),
                "query_norm": query_norm,
                "count": repeat_count + 1,
                "pending_candidate_count": len(pending),
            },
        )
    ]
    return dict(raw_output), events


def _update_after_explore(
    ledger: dict[str, Any],
    *,
    observation_id: str,
    raw_output: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    query = str(raw_output.get("query") or raw_output.get("claim") or "")
    query_norm = norm_query(raw_output.get("query") or query)
    candidates = _candidate_records(observation_id=observation_id, query_norm=query_norm, raw_output=raw_output)
    candidate_keys = [str(item.get("candidate_key") or item.get("event_id") or "") for item in candidates if str(item.get("candidate_key") or item.get("event_id") or "")]
    record = {
        "query": query,
        "query_norm": query_norm,
        "tool": "explore",
        "observation_id": observation_id,
        "mode": str(raw_output.get("mode") or ""),
        "support_status": str(raw_output.get("support_status") or ""),
        "candidate_keys": candidate_keys,
        "segment_ids": _unique([item.get("segment_id") for item in candidates]),
        "time_ranges": [item.get("time_range") for item in candidates if item.get("time_range")],
        "status": "hint" if str(raw_output.get("mode") or "") == "planner_recovery_hint" else "explored",
        "recommended_next_actions": _recommended_next_actions({"candidates": candidates, "records": [], "options": {}}),
        "notes": [],
    }
    ledger["records"].append(record)
    for candidate in candidates:
        _upsert_candidate(ledger, candidate)
    _record_answer_options(ledger, raw_output=raw_output)
    _update_option_status_from_explore(ledger, observation_id=observation_id, raw_output=raw_output)
    return [
        (
            "exploration_ledger_update",
            {"record_type": "explore", "query_norm": query_norm, "candidate_key": ",".join(candidate_keys), "status": record["status"]},
        )
    ]


def _update_after_verify(
    ledger: dict[str, Any],
    *,
    observation_id: str,
    raw_output: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    candidate_key = str(raw_output.get("candidate_key") or "").strip()
    results = raw_output.get("verification_results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        results = ()
    status = "uncertain"
    checked_claims: list[str] = []
    for result in results:
        if not isinstance(result, Mapping):
            continue
        verdict = str(result.get("verdict") or "uncertain")
        checked_claims.append(str(result.get("claim") or ""))
        if verdict == "supported":
            status = "verified_supported"
        elif verdict == "not_found_in_window" and status != "verified_supported":
            status = "verified_negative"
        elif verdict == "contradicted":
            status = "wrong_scope"
    if candidate_key:
        for candidate in ledger["candidates"]:
            if candidate.get("candidate_key") == candidate_key:
                candidate["status"] = status
                candidate["checked_claims"] = [claim for claim in checked_claims if claim]
                candidate["last_verification_observation_id"] = observation_id
                break
    return [
        (
            "exploration_ledger_update",
            {"record_type": "verify_window", "candidate_key": candidate_key, "status": status},
        )
    ]


def _update_after_synthesize(
    ledger: dict[str, Any],
    *,
    observation_id: str,
    raw_output: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    support_ids = {str(item) for item in _sequence_items(raw_output.get("supports") or raw_output.get("previous_memory_refs"))}
    if support_ids:
        for candidate in ledger["candidates"]:
            related = {str(item) for item in _sequence_items(candidate.get("related_memory_ids"))}
            if related & support_ids:
                candidate["status"] = "consumed_for_synthesis"
    return [("exploration_ledger_update", {"record_type": "synthesize_memory", "observation_id": observation_id, "status": "synthesized"})]


def _candidate_records(*, observation_id: str, query_norm: str, raw_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _mapping_list(raw_output.get("candidate_windows")):
        candidate_key = str(item.get("candidate_key") or item.get("candidate_id") or "").strip()
        records.append(
            {
                "candidate_key": candidate_key,
                "source_observation_id": observation_id,
                "segment_id": str(item.get("segment_id") or ""),
                "time_range": _time_range(item.get("time_range")),
                "segment_start_sec": _optional_float(item.get("segment_start_sec")),
                "segment_end_sec": _optional_float(item.get("segment_end_sec")),
                "query_norm": query_norm,
                "status": "pending",
                "checked_claims": [],
                "last_verification_observation_id": None,
            }
        )
    for item in _mapping_list(raw_output.get("event_candidates")):
        event_id = str(item.get("event_id") or "").strip()
        records.append(
            {
                "candidate_key": event_id or f"{observation_id}:event_{len(records) + 1:04d}",
                "event_id": event_id,
                "event_type": str(item.get("event_type") or ""),
                "team": str(item.get("team") or ""),
                "source_observation_id": observation_id,
                "segment_id": str(item.get("segment_id") or ""),
                "time_range": _time_range(item.get("time_range")),
                "query_norm": query_norm,
                "status": "pending",
                "checked_claims": [],
                "last_verification_observation_id": None,
            }
        )
    return records


def _update_option_status_from_explore(
    ledger: dict[str, Any],
    *,
    observation_id: str,
    raw_output: Mapping[str, Any],
) -> None:
    mapping = raw_output.get("answer_mapping") if isinstance(raw_output.get("answer_mapping"), Mapping) else {}
    related_option = str(mapping.get("related_option") or raw_output.get("related_option") or "").strip()
    option_relation = str(mapping.get("option_relation") or raw_output.get("option_relation") or "").strip()
    supports_option = str(mapping.get("supports_option") or raw_output.get("supports_option") or "").strip()
    condition = raw_output.get("condition_match") if isinstance(raw_output.get("condition_match"), Mapping) else {}
    if not related_option and str(condition.get("match_level") or "") == "related_but_wrong_scope":
        related_option = supports_option
        option_relation = "wrong_scope"
    option_id = related_option or supports_option
    if not option_id:
        return
    if option_relation == "wrong_scope" or str(condition.get("match_level") or "") == "related_but_wrong_scope":
        status = "wrong_scope"
    elif supports_option:
        status = "supported"
    else:
        status = option_relation or "weak_related"
    existing = dict(ledger["options"].get(option_id, {}))
    obs_ids = _unique([*(_sequence_items(existing.get("related_observation_ids"))), observation_id])
    existing.update(
        {
            "option_id": option_id,
            "status": status,
            "related_memory_ids": list(_sequence_items(existing.get("related_memory_ids"))),
            "related_observation_ids": obs_ids,
            "reason": str(condition.get("reason") or option_relation or status),
        }
    )
    ledger["options"][option_id] = existing


def _record_answer_options(ledger: dict[str, Any], *, raw_output: Mapping[str, Any]) -> None:
    options = raw_output.get("answer_options")
    if not isinstance(options, Mapping):
        return
    for option_id, text in options.items():
        key = str(option_id).strip().upper()[:1]
        if not key:
            continue
        ledger["answer_options"][key] = str(text)
        ledger["options"].setdefault(
            key,
            {
                "option_id": key,
                "status": "untested",
                "related_memory_ids": [],
                "related_observation_ids": [],
                "reason": None,
            },
        )


def _recommended_next_actions(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    for candidate in ledger.get("candidates", []):
        if isinstance(candidate, Mapping) and candidate.get("status") == "pending":
            return [{"tool": "verify_window", "candidate_key": candidate.get("candidate_key") or candidate.get("event_id")}]
    sweep = _negative_only_sweep_action(ledger)
    if sweep is not None:
        return [sweep]
    options = ledger.get("options")
    if isinstance(options, Mapping) and any(
        isinstance(option, Mapping) and option.get("status") == "untested" for option in options.values()
    ):
        return [{"tool": "explore", "focus": "test untested options against the original condition"}]
    return []


def _negative_only_sweep_action(ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = [candidate for candidate in _mapping_list(ledger.get("candidates")) if str(candidate.get("segment_id") or "").strip()]
    if not candidates:
        return None
    if any(str(candidate.get("status") or "") == "verified_supported" for candidate in candidates):
        return None
    negatives = [candidate for candidate in candidates if str(candidate.get("status") or "") == "verified_negative"]
    if not negatives:
        return None
    seed = negatives[-1]
    segment_id = str(seed.get("segment_id") or "").strip()
    segment_candidates = [candidate for candidate in candidates if str(candidate.get("segment_id") or "").strip() == segment_id]
    covered = sorted(
        time_range
        for time_range in (_time_range(candidate.get("time_range")) for candidate in segment_candidates)
        if time_range is not None
    )
    if not covered:
        return None
    last_width = max(0.1, covered[-1][1] - covered[-1][0])
    segment_start = _first_present_float([candidate.get("segment_start_sec") for candidate in segment_candidates])
    segment_end = _first_present_float([candidate.get("segment_end_sec") for candidate in segment_candidates])
    if segment_start is None:
        segment_start = min(start for start, _end in covered)
    if segment_end is None:
        segment_end = max(end for _start, end in covered) + last_width
    next_range = _largest_uncovered_range(
        covered,
        segment_start=float(segment_start),
        segment_end=float(segment_end),
        preferred_width=last_width,
    )
    if next_range is None:
        start = covered[-1][1]
        next_range = [start, start + last_width]
    return {
        "tool": "verify_window",
        "segment_id": segment_id,
        "time_range": next_range,
        "focus": "extend visual coverage beyond already-verified regions",
    }


def _largest_uncovered_range(
    covered: Sequence[Sequence[float]],
    *,
    segment_start: float,
    segment_end: float,
    preferred_width: float,
) -> list[float] | None:
    cursor = float(segment_start)
    gaps: list[tuple[float, float]] = []
    for start, end in covered:
        start = max(float(segment_start), float(start))
        end = min(float(segment_end), float(end))
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < segment_end:
        gaps.append((cursor, float(segment_end)))
    if not gaps:
        return None
    start, end = max(gaps, key=lambda item: item[1] - item[0])
    width = min(max(0.1, float(preferred_width)), max(0.1, end - start))
    return [round(start, 3), round(min(end, start + width), 3)]


def _upsert_candidate(ledger: dict[str, Any], candidate: Mapping[str, Any]) -> None:
    key = str(candidate.get("candidate_key") or candidate.get("event_id") or "").strip()
    if not key:
        return
    for index, existing in enumerate(ledger["candidates"]):
        if str(existing.get("candidate_key") or existing.get("event_id") or "") == key:
            merged = dict(existing)
            merged.update({key_: deepcopy(value) for key_, value in candidate.items() if value not in (None, "", [])})
            ledger["candidates"][index] = merged
            return
    ledger["candidates"].append(dict(candidate))


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _sequence_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _unique(values: Any) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in _sequence_items(values):
        if value in (None, ""):
            continue
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _time_range(value: Any) -> list[float] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present_float(values: Sequence[Any]) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _format_time_range(value: Any) -> str:
    time_range = _time_range(value)
    if not time_range:
        return ""
    return f"[{time_range[0]:.1f}-{time_range[1]:.1f}]"
