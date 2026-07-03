"""Shot-boundary detection helpers for multi_v3 video indexing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Sequence


def detect_shots_ffmpeg(
    video_path: str,
    *,
    min_shot_sec: float = 2.0,
    scene_threshold: float = 0.3,
) -> Sequence[tuple[float, float]]:
    duration = _probe_duration(video_path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{float(scene_threshold)})',showinfo",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to detect shot boundaries") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg shot detection failed: {' | '.join(tail)}")
    cuts = sorted(
        {
            time_sec
            for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr or "")
            if (time_sec := float(match.group(1))) > 0.0
        }
    )
    return _ranges_from_cuts(cuts, duration_sec=duration, min_shot_sec=min_shot_sec)


def detect_shots_uniform(duration_sec: float, *, window: float = 15.0) -> Sequence[tuple[float, float]]:
    duration = max(0.0, float(duration_sec))
    if duration <= 0.0:
        return ()
    step = max(0.1, float(window))
    ranges = []
    start = 0.0
    while start < duration:
        end = min(duration, start + step)
        ranges.append((round(start, 3), round(end, 3)))
        start = end
    return tuple(ranges)


def _probe_duration(video_path: str) -> float:
    if not Path(video_path).exists():
        raise RuntimeError(f"video does not exist: {video_path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(command, text=True, timeout=20).strip()
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to detect shot boundaries") from exc
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"ffprobe failed for {video_path}") from exc
    return max(0.0, float(out))


def _ranges_from_cuts(
    cuts: Sequence[float],
    *,
    duration_sec: float,
    min_shot_sec: float,
) -> tuple[tuple[float, float], ...]:
    boundaries = [0.0]
    for cut in cuts:
        if cut - boundaries[-1] >= float(min_shot_sec):
            boundaries.append(float(cut))
    duration = max(0.0, float(duration_sec))
    if not boundaries or duration - boundaries[-1] >= 0.001:
        boundaries.append(duration)
    ranges = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end > start:
            ranges.append((round(start, 3), round(end, 3)))
    return tuple(ranges)
