"""Lightweight video scene index used by iterative visual agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class VideoSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    keyframe_path: str = ""
    low_fps_caption: str = ""
    source: str = "fixed_window"

    def to_dict(self) -> Mapping[str, object]:
        return {
            "segment_id": self.segment_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "keyframe_path": self.keyframe_path,
            "low_fps_caption": self.low_fps_caption,
            "source": self.source,
        }


@dataclass(frozen=True)
class SceneIndex:
    video_path: str
    duration_sec: float
    segments: Sequence[VideoSegment] = field(default_factory=list)

    def get(self, segment_id: str) -> VideoSegment:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise ValueError(f"Unknown segment_id: {segment_id}")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "video_path": self.video_path,
            "duration_sec": self.duration_sec,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def summary(self, max_segments: int = 16) -> str:
        if not self.segments:
            return "(no segments indexed)"

        lines = []
        for segment in self.segments[:max_segments]:
            caption = segment.low_fps_caption or segment.keyframe_path or "no coarse caption yet"
            lines.append(f"{segment.segment_id} [{segment.start_sec:.1f}-{segment.end_sec:.1f}s] {caption}")
        remaining = len(self.segments) - max_segments
        if remaining > 0:
            lines.append(f"... {remaining} more segments omitted")
        return "\n".join(lines)


def fixed_window_scene_index(
    *,
    video_path: str,
    duration_sec: float,
    window_sec: float = 30.0,
    source: str = "fixed_window",
) -> SceneIndex:
    """Create a deterministic fallback scene index without decoding the video."""

    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")

    segments = []
    start = 0.0
    index = 1
    while start < duration_sec:
        end = min(start + window_sec, duration_sec)
        segments.append(
            VideoSegment(
                segment_id=f"seg_{index:04d}",
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                source=source,
            )
        )
        start = end
        index += 1
    return SceneIndex(video_path=video_path, duration_sec=duration_sec, segments=segments)
