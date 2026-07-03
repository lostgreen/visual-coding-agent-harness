"""Scene aggregation for detected multi_v3 shot ranges."""

from __future__ import annotations

from typing import Sequence


def aggregate_shot_ranges_by_duration(
    shot_ranges: Sequence[tuple[float, float]],
    *,
    max_scene_sec: float = 600.0,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    groups: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_start: float | None = None
    limit = max(1.0, float(max_scene_sec))
    for start_sec, end_sec in shot_ranges:
        start = float(start_sec)
        end = float(end_sec)
        if current and current_start is not None and end - current_start > limit:
            groups.append(current)
            current = []
            current_start = None
        if current_start is None:
            current_start = start
        current.append((start, end))
    if current:
        groups.append(current)
    return tuple(tuple(group) for group in groups)
