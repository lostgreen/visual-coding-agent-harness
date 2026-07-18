from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_enumeration_manifest(
    *,
    target_segment_id: str,
    required_range: tuple[float, float],
    windows: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    unresolved_candidate_ids: Sequence[str],
    expected_event_dwell_sec: float,
    required_ranges: Sequence[tuple[float, float]] | None = None,
    max_window_sec: float = 15.0,
) -> dict[str, Any]:
    start, end = sorted((float(required_range[0]), float(required_range[1])))
    normalized_required_ranges = _merge_ranges(
        required_ranges or ((start, end),)
    )
    if not normalized_required_ranges:
        normalized_required_ranges = [(start, end)]
    dwell = max(0.05, float(expected_event_dwell_sec or 1.0))
    normalized_windows: list[dict[str, Any]] = []
    valid_ranges: list[tuple[float, float]] = []
    for raw_window in windows:
        bounds = tuple(raw_window.get("range", ()) or ())
        if len(bounds) != 2:
            continue
        window_start, window_end = sorted((float(bounds[0]), float(bounds[1])))
        parse_status = str(raw_window.get("parse_status", "unknown") or "unknown").casefold()
        fps = float(raw_window.get("sampling_fps", 0.0) or 0.0)
        sampling_ok = fps > 0.0 and (1.0 / fps) <= dwell + 1e-9
        short_window = (window_end - window_start) <= float(max_window_sec) + 1e-9
        row = {
            "range": [window_start, window_end],
            "sampling_fps": fps,
            "expected_event_dwell_sec": dwell,
            "parse_status": parse_status,
            "candidate_ids": [str(item) for item in tuple(raw_window.get("candidate_ids", ()) or ()) if str(item)],
            "status": (
                "complete"
                if parse_status == "ok" and sampling_ok and short_window
                else "incomplete"
            ),
            "sampling_ok": sampling_ok,
            "window_size_ok": short_window,
        }
        normalized_windows.append(row)
        if row["status"] == "complete":
            valid_ranges.append((window_start, window_end))
    unprocessed_ranges = [
        gap
        for required in normalized_required_ranges
        for gap in _uncovered_ranges(required, valid_ranges)
    ]
    boundary_gaps = [
        gap
        for required in normalized_required_ranges
        for gap in _boundary_gaps(
            [
                (max(required[0], left), min(required[1], right))
                for left, right in valid_ranges
                if right > required[0] and left < required[1]
            ],
            dwell,
        )
    ]
    reconciliation_complete = not tuple(unresolved_candidate_ids or ())
    enumeration_complete = bool(
        normalized_windows
        and not unprocessed_ranges
        and not boundary_gaps
        and reconciliation_complete
        and all(row["status"] == "complete" for row in normalized_windows)
    )
    return {
        "target_segment_id": str(target_segment_id or ""),
        "required_range": [start, end],
        "required_ranges": [[left, right] for left, right in normalized_required_ranges],
        "windows": normalized_windows,
        "unprocessed_ranges": [[left, right] for left, right in unprocessed_ranges],
        "candidate_ids": [str(item) for item in candidate_ids if str(item)],
        "candidate_reconciliation_status": "complete" if reconciliation_complete else "incomplete",
        "unresolved_candidate_ids": [str(item) for item in unresolved_candidate_ids if str(item)],
        "boundary_gaps": [[left, right] for left, right in boundary_gaps],
        "enumeration_complete": enumeration_complete,
    }


def _uncovered_ranges(
    required: tuple[float, float],
    ranges: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    start, end = required
    merged = _merge_ranges(
        (max(start, left), min(end, right))
        for left, right in ranges
        if right > start and left < end
    )
    cursor = start
    gaps: list[tuple[float, float]] = []
    for left, right in merged:
        if left > cursor + 1e-6:
            gaps.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end - 1e-6:
        gaps.append((cursor, end))
    return gaps


def _boundary_gaps(ranges: Sequence[tuple[float, float]], dwell: float) -> list[tuple[float, float]]:
    ordered = _merge_ranges(ranges)
    return [
        (left_end, right_start)
        for (_, left_end), (right_start, _) in zip(ordered, ordered[1:])
        if right_start - left_end > dwell
    ]


def _merge_ranges(ranges: Sequence[tuple[float, float]] | Any) -> list[tuple[float, float]]:
    ordered = sorted((float(left), float(right)) for left, right in ranges if right > left)
    merged: list[tuple[float, float]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1] + 1e-6:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged
