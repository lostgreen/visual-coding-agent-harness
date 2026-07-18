from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_STOP = frozenset({"a", "an", "the", "then", "and", "after", "before", "followed", "by", "is", "was"})


def build_sequence_ledger(
    snapshot: Mapping[str, Any],
    options: Mapping[str, str],
) -> dict[str, Any]:
    events = tuple(
        sorted(
            (
                dict(row)
                for row in tuple(snapshot.get("qualified_events", ()) or ())
                if isinstance(row, Mapping)
            ),
            key=lambda row: _event_time(row),
        )
    )
    rows = [
        {
            "event_id": str(row.get("candidate_id", "") or row.get("event_id", "") or ""),
            "time": _event_time(row),
            "event_keys": [str(item) for item in tuple(row.get("event_keys", ()) or ()) if str(item)],
            "descriptions": [str(item) for item in tuple(row.get("descriptions", ()) or ()) if str(item)],
            "evidence_ids": [str(item) for item in tuple(row.get("evidence_ids", ()) or ()) if str(item)],
        }
        for row in events
    ]
    unresolved = bool(
        tuple(snapshot.get("incomplete_events", ()) or ())
        or tuple(snapshot.get("conflicted_events", ()) or ())
        or tuple(snapshot.get("duplicate_suspect_events", ()) or ())
    )
    option_verdicts = {
        option: _evaluate_option_sequence(str(text or ""), rows, unresolved=unresolved)
        for option, text in options.items()
    }
    return {
        "events": rows,
        "unresolved": unresolved,
        "option_sequence_verdicts": option_verdicts,
        "no_option_exact_match": bool(option_verdicts) and not any(
            row["status"] == "supported" for row in option_verdicts.values()
        ),
    }


def _evaluate_option_sequence(
    option_text: str,
    events: Sequence[Mapping[str, Any]],
    *,
    unresolved: bool,
) -> dict[str, Any]:
    steps = _steps(option_text)
    if len(steps) < 2 or unresolved:
        return {
            "status": "unknown",
            "steps": steps,
            "matched_event_ids": [],
            "reason": "sequence is incomplete or the option does not specify an ordered sequence",
        }
    positions = []
    for step in steps:
        matches = [index for index, event in enumerate(events) if _step_matches(step, event)]
        if not matches:
            return {
                "status": "unknown",
                "steps": steps,
                "matched_event_ids": [],
                "reason": "at least one option event has no qualified canonical match",
            }
        positions.append(matches)
    selected_positions = _increasing_selection(positions)
    if selected_positions is not None:
        return {
            "status": "supported",
            "steps": steps,
            "matched_event_ids": [str(events[index].get("event_id", "") or "") for index in selected_positions],
            "reason": "all option events occur in the stated order in the canonical sequence ledger",
        }
    return {
        "status": "contradicted",
        "steps": steps,
        "matched_event_ids": [str(events[index].get("event_id", "") or "") for matches in positions for index in matches[:1]],
        "reason": "all option events are present but their canonical order conflicts with the option",
    }


def _steps(text: str) -> list[str]:
    parts = re.split(r"\s*(?:→|->|\bthen\b|\bfollowed by\b)\s*", str(text or "").casefold())
    return [part.strip(" .,:;-") for part in parts if _tokens(part)]


def _step_matches(step: str, event: Mapping[str, Any]) -> bool:
    step_tokens = set(_tokens(step))
    if not step_tokens:
        return False
    text = " ".join(
        (
            *tuple(str(item) for item in tuple(event.get("event_keys", ()) or ())),
            *tuple(str(item) for item in tuple(event.get("descriptions", ()) or ())),
        )
    ).casefold()
    event_tokens = set(_tokens(text))
    return bool(
        " ".join(_tokens(step)) in " ".join(_tokens(text))
        or len(step_tokens.intersection(event_tokens)) / max(1, len(step_tokens)) >= 0.6
    )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.findall(r"[a-z0-9]+", str(text or "").casefold()) if token not in _STOP)


def _event_time(row: Mapping[str, Any]) -> float:
    bounds = tuple(row.get("virtual_time_range", ()) or ())
    if len(bounds) == 2:
        try:
            return float(bounds[0])
        except (TypeError, ValueError):
            pass
    try:
        return float(row.get("time", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _increasing_selection(positions: Sequence[Sequence[int]]) -> list[int] | None:
    selected: list[int] = []
    floor = -1
    for candidates in positions:
        next_index = next((index for index in candidates if index > floor), None)
        if next_index is None:
            return None
        selected.append(next_index)
        floor = next_index
    return selected
