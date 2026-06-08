"""Shared v4 sufficiency and verifier predicates over typed evidence rows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


GROUNDING_WEIGHTS = {
    "global_sparse": 0.35,
    "visually_confirmed": 1.0,
    "indexed_transcript": 0.85,
    "inferred": 0.35,
    "weak": 0.2,
    "external_knowledge": 0.1,
}
WEAK_GROUNDING = {"global_sparse", "inferred", "weak", "external_knowledge"}
NAVIGATION_TOOLS = {
    "video_ls",
    "search_segments",
    "target_coverage",
    "read_segment",
    "read_segment_detail",
    "locate_targets_in_segment",
    "expand_window",
    "zoom",
}
VISUAL_GROUNDING = {"visually_confirmed"}


@dataclass(frozen=True)
class PredicateResult:
    name: str
    passed: bool
    reasons: Sequence[str] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)


def gist_supports_single_option(table: Mapping[str, Any]) -> PredicateResult:
    global_rows = [
        row
        for row in _rows(table)
        if _is_global_floor_row(row) and _row_supported_option(row)
    ]
    options = sorted({_row_supported_option(row) for row in global_rows if _row_supported_option(row)})
    passed = len(options) == 1
    return PredicateResult(
        name="gist_supports_single_option",
        passed=passed,
        reasons=() if passed else ("global gist does not support exactly one option",),
        details={"options": options},
    )


def every_event_has_confirmed_timestamp(
    table: Mapping[str, Any],
    *,
    expected_events: Sequence[str] = (),
) -> PredicateResult:
    expected = [str(event).strip() for event in expected_events if str(event).strip()]
    confirmed = _confirmed_timestamped_events(table)
    missing = []
    for event in expected:
        if not any(_events_match(event, observed.get("event", "")) for observed in confirmed):
            missing.append(event)
    passed = bool(expected) and not missing
    return PredicateResult(
        name="every_event_has_confirmed_timestamp",
        passed=passed,
        reasons=() if passed else ("missing confirmed timestamp for event",),
        details={"expected_events": expected, "observed_events": confirmed, "missing_events": missing},
    )


def distinguishing_fact_exists(
    table: Mapping[str, Any],
    *,
    selected_option: str | None = None,
) -> PredicateResult:
    option = selected_option or _top_option(table)
    support_rows = _strong_support_rows(table, option)
    passed = bool(option and support_rows)
    return PredicateResult(
        name="distinguishing_fact_exists",
        passed=passed,
        reasons=() if passed else ("no visually confirmed distinguishing fact for selected option",),
        details={"selected_option": option, "support_obs_ids": [str(row.get("obs_id", "")) for row in support_rows]},
    )


def selected_option_has_structured_support(
    table: Mapping[str, Any],
    *,
    selected_option: str | None = None,
) -> PredicateResult:
    option = selected_option or _top_option(table)
    rows = _option_rows(table, option)
    support_rows = [
        row
        for row in rows
        if str(row.get("tool", "")) not in NAVIGATION_TOOLS
        and not _is_weak_grounding(row)
        and (_row_supports_option(row, option) or _is_global_floor_row(row))
    ]
    passed = bool(option and support_rows)
    return PredicateResult(
        name="selected_option_has_structured_support",
        passed=passed,
        reasons=() if passed else ("selected option lacks structured visual support",),
        details={"selected_option": option, "support_obs_ids": [str(row.get("obs_id", "")) for row in support_rows]},
    )


def no_decisive_weak_grounding(
    table: Mapping[str, Any],
    *,
    selected_option: str | None = None,
) -> PredicateResult:
    option = selected_option or _top_option(table)
    selected_rows = _option_rows(table, option)
    strong_rows = [row for row in selected_rows if not _is_weak_grounding(row)]
    weak_rows = [row for row in selected_rows if _is_weak_grounding(row)]
    weak_score = max([_row_score(row) for row in weak_rows] or [0.0])
    strong_score = max([_row_score(row) for row in strong_rows] or [0.0])
    passed = not option or strong_score >= weak_score or weak_score == 0.0
    return PredicateResult(
        name="no_decisive_weak_grounding",
        passed=passed,
        reasons=() if passed else ("selected option is only decisively supported by weak grounding",),
        details={"selected_option": option, "strong_score": strong_score, "weak_score": weak_score},
    )


def no_unaddressed_conflict(
    table: Mapping[str, Any],
    *,
    selected_option: str | None = None,
    cited_obs_ids: Sequence[str] = (),
) -> PredicateResult:
    option = selected_option or _top_option(table)
    cited = {str(obs_id) for obs_id in cited_obs_ids}
    selected_score = max([_row_score(row) for row in _strong_support_rows(table, option)] or [0.0])
    conflicts = []
    for row in _rows(table):
        row_option = _row_supported_option(row)
        if not row_option or row_option == option or str(row.get("obs_id", "")) in cited:
            continue
        if _is_weak_grounding(row):
            continue
        score = _row_score(row)
        if score > selected_score:
            conflicts.append({"obs_id": str(row.get("obs_id", "")), "option": row_option, "score": score})
    passed = not conflicts
    return PredicateResult(
        name="no_unaddressed_conflict",
        passed=passed,
        reasons=() if passed else ("uncited stronger conflicting option support",),
        details={"selected_option": option, "conflicts": conflicts},
    )


def temporal_order_consistent(
    table: Mapping[str, Any],
    *,
    selected_option: str | None = None,
    expected_events: Sequence[str] = (),
) -> PredicateResult:
    expected = [str(event).strip() for event in expected_events if str(event).strip()]
    if not expected and selected_option:
        expected = _option_event_sequence(_option_text(table, selected_option))
    observed = _confirmed_timestamped_events(table)
    matched = []
    for event in expected:
        match = _match_observed_event(event, observed)
        if match:
            matched.append({"expected": event, "observed": match["event"], "start_sec": match["start_sec"]})
    if len(expected) < 2:
        return PredicateResult(
            name="temporal_order_consistent",
            passed=False,
            reasons=("temporal order needs at least two expected events",),
            details={"expected_events": expected, "observed_events": observed, "matched_events": matched},
        )
    if len(matched) < len(expected):
        return PredicateResult(
            name="temporal_order_consistent",
            passed=False,
            reasons=("not all temporal events are matched to confirmed timestamps",),
            details={"expected_events": expected, "observed_events": observed, "matched_events": matched},
        )
    times = [float(item["start_sec"]) for item in matched]
    passed = times == sorted(times)
    return PredicateResult(
        name="temporal_order_consistent",
        passed=passed,
        reasons=() if passed else ("temporal order contradicts evidence",),
        details={"expected_events": expected, "observed_events": observed, "matched_events": matched},
    )


def direct_floor_holds(
    table: Mapping[str, Any],
    *,
    selected_option: str | None = None,
) -> PredicateResult:
    option = selected_option or _top_option(table)
    global_rows = [
        row
        for row in _rows(table)
        if _is_global_floor_row(row)
        and _row_supported_option(row)
        and not _is_candidate_hint_only_global_row(row)
    ]
    if not global_rows:
        return PredicateResult(name="direct_floor_holds", passed=True, details={"selected_option": option})
    global_rows.sort(key=lambda row: (-_row_score(row), str(row.get("obs_id", ""))))
    floor = global_rows[0]
    floor_option = _row_supported_option(floor)
    passed = not floor_option or floor_option == option
    return PredicateResult(
        name="direct_floor_holds",
        passed=passed,
        reasons=() if passed else ("selected option is below the global gist floor",),
        details={
            "selected_option": option,
            "floor_option": floor_option,
            "floor_obs_id": str(floor.get("obs_id", "")),
            "floor_score": _row_score(floor),
        },
    )


def grounding_quality_floor(
    mapped_records: Sequence[Any],
    *,
    workspace: Any,
    require_visual: bool = True,
) -> str | None:
    if not mapped_records:
        return "no mapped evidence"
    qualities = set()
    for record in mapped_records:
        chain = workspace.evidence_chain(record.evidence_id)
        qualities.update(str(item.grounding_quality) for item in chain)
    if require_visual and not qualities.intersection(VISUAL_GROUNDING):
        return f"final support contains no visually_confirmed evidence (got {sorted(qualities)})"
    return None


def _rows(table: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = table.get("rows", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [row for row in rows if isinstance(row, Mapping)]
    groups = table.get("groups", {})
    if not isinstance(groups, Mapping):
        return []
    flattened = []
    for value in groups.values():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flattened.extend(row for row in value if isinstance(row, Mapping))
    return flattened


def _option_rows(table: Mapping[str, Any], option: str | None) -> list[Mapping[str, Any]]:
    if not option:
        return []
    groups = table.get("groups", {})
    if isinstance(groups, Mapping):
        rows = groups.get(option, [])
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            return [row for row in rows if isinstance(row, Mapping)]
    return [row for row in _rows(table) if _row_supports_option(row, option)]


def _strong_support_rows(table: Mapping[str, Any], option: str | None) -> list[Mapping[str, Any]]:
    return [
        row
        for row in _option_rows(table, option)
        if not _is_weak_grounding(row) and _row_supports_option(row, option)
    ]


def _row_supports_option(row: Mapping[str, Any], option: str | None) -> bool:
    if not option:
        return False
    supported = _row_supported_option(row)
    return supported == option


def _row_supported_option(row: Mapping[str, Any]) -> str:
    supported = str(row.get("supported_option", "") or "").strip().upper()[:1]
    if supported:
        return supported
    for relation in _relations(row):
        relation_name = str(relation.get("relation", "")).strip().lower()
        if relation_name not in {"support", "supports", "supported"}:
            continue
        option = str(relation.get("option", "")).strip().upper()[:1]
        if option:
            return option
    return ""


def _relations(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = row.get("candidate_option_relations", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _top_option(table: Mapping[str, Any]) -> str:
    scores = {}
    for row in _rows(table):
        option = _row_supported_option(row)
        if not option:
            continue
        scores[option] = scores.get(option, 0.0) + _row_score(row)
    if not scores:
        return ""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _row_score(row: Mapping[str, Any]) -> float:
    return float(row.get("confidence", 0.0) or 0.0) * GROUNDING_WEIGHTS.get(
        str(row.get("grounding_quality", "weak")),
        0.2,
    )


def _is_weak_grounding(row: Mapping[str, Any]) -> bool:
    return str(row.get("grounding_quality", "weak")) in WEAK_GROUNDING


def _is_global_floor_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("tool", "")) == "global_gist" or str(row.get("grounding_quality", "")) == "global_sparse"


def _is_candidate_hint_only_global_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("tool", "")) == "global_gist" or str(row.get("grounding_quality", "")) == "global_sparse"


def _confirmed_timestamped_events(table: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    timeline = table.get("timeline")
    if isinstance(timeline, Sequence) and not isinstance(timeline, (str, bytes)):
        return _confirmed_timeline_events(timeline)

    observed = []
    for row in _rows(table):
        if str(row.get("grounding_quality", "")) not in {"visually_confirmed", "indexed_transcript"}:
            continue
        event = str(row.get("event_label", "")).strip()
        if not event:
            continue
        start_value = row.get("observed_at_sec")
        if start_value is None:
            time_range = row.get("time_range")
            if not isinstance(time_range, Sequence) or isinstance(time_range, (str, bytes)) or len(time_range) < 2:
                continue
            start_value = time_range[0]
        try:
            start_sec = float(start_value)
        except (TypeError, ValueError):
            continue
        observed.append(
            {
                "event": event,
                "start_sec": start_sec,
                "obs_id": str(row.get("obs_id", "")),
            }
        )
    return sorted(observed, key=lambda item: (float(item["start_sec"]), str(item.get("obs_id", ""))))


def _confirmed_timeline_events(timeline: Sequence[Any]) -> list[Mapping[str, Any]]:
    observed = []
    for row in timeline:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("confidence_signal", "")).strip().lower() != "visually_confirmed":
            continue
        event = str(row.get("entity") or row.get("event_label") or "").strip()
        observed_at_sec = row.get("observed_at_sec")
        if not event or observed_at_sec is None:
            continue
        try:
            start_sec = float(observed_at_sec)
        except (TypeError, ValueError):
            continue
        observed.append(
            {
                "event": event,
                "start_sec": start_sec,
                "obs_id": str(row.get("obs_id") or row.get("observation_id") or ""),
            }
        )
    return sorted(observed, key=lambda item: (float(item["start_sec"]), str(item.get("obs_id", ""))))


def _match_observed_event(expected_event: str, observed_events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for observed in observed_events:
        if _events_match(expected_event, str(observed.get("event", ""))):
            return observed
    return None


def _events_match(left: str, right: str) -> bool:
    left_tokens = _event_tokens(left)
    right_tokens = _event_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return True
    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return bool(union) and overlap / union >= 0.5


def _event_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "then",
        "before",
        "after",
        "first",
        "second",
        "third",
        "fourth",
        "later",
        "appears",
        "appear",
        "shown",
        "shows",
        "object",
        "event",
        "option",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in stopwords
    }


def _option_text(table: Mapping[str, Any], option: str) -> str:
    options = table.get("options", [])
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return ""
    for index, item in enumerate(options):
        text = str(item).strip()
        match = re.match(r"^([A-Za-z])(?:[\.)]\s*|\s+|$)", text)
        letter = match.group(1).upper() if match else chr(ord("A") + index)
        if letter == option:
            return text
    return ""


def _option_event_sequence(option_text: str) -> list[str]:
    text = _strip_option_prefix(option_text).lower()
    text = text.replace("->", " then ").replace(">", " then ")
    if re.search(r"\bbefore\b", text):
        return _clean_event_sequence(re.split(r"\bbefore\b", text, maxsplit=1))
    if re.search(r"\bafter\b", text):
        return list(reversed(_clean_event_sequence(re.split(r"\bafter\b", text, maxsplit=1))))
    return _clean_event_sequence(re.split(r"\bthen\b|[,;/]+", text))


def _strip_option_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-H](?:[\.)]\s*|\s+)", "", text, flags=re.IGNORECASE).strip()


def _clean_event_sequence(parts: Sequence[str]) -> list[str]:
    events = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", str(part)).strip(" .:-")
        if _event_tokens(cleaned):
            events.append(cleaned)
    return events
