"""FrameSet contract for answer-grade visual evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSet:
    frame_set_id: str
    video_path: str
    segment_id: str
    time_range: tuple[float, float]
    fps: float
    nframes: int
    max_pixels: int
    frame_paths: tuple[str, ...]
    sha256: str = ""
    created_at: float = 0.0
