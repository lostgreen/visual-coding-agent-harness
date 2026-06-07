"""Lightweight video scene index used by iterative visual agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class VideoSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    keyframe_path: str = ""
    low_fps_caption: str = ""
    source: str = "fixed_window"
    source_segment_id: Optional[str] = None
    visual_caption: str = ""
    visual_caption_source: str = ""
    asr_summary: str = ""
    asr_summary_source: str = ""
    raw_asr_ref: str = ""
    stage_tags: Sequence[str] = field(default_factory=tuple)
    entities: Sequence[str] = field(default_factory=tuple)
    topic_tags: Sequence[str] = field(default_factory=tuple)
    confidence: Optional[float] = None
    grounding_quality: str = ""
    citation_provenance: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "segment_id": self.segment_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "keyframe_path": self.keyframe_path,
            "low_fps_caption": self.low_fps_caption,
            "source": self.source,
            "source_segment_id": self.source_segment_id,
            "visual_caption": self.visual_caption,
            "visual_caption_source": self.visual_caption_source,
            "asr_summary": self.asr_summary,
            "asr_summary_source": self.asr_summary_source,
            "raw_asr_ref": self.raw_asr_ref,
            "stage_tags": list(self.stage_tags),
            "entities": list(self.entities),
            "topic_tags": list(self.topic_tags),
            "confidence": self.confidence,
            "grounding_quality": self.grounding_quality,
            "citation_provenance": dict(self.citation_provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VideoSegment":
        return cls(
            segment_id=str(value["segment_id"]),
            start_sec=float(value["start_sec"]),
            end_sec=float(value["end_sec"]),
            keyframe_path=str(value.get("keyframe_path") or ""),
            low_fps_caption=str(value.get("low_fps_caption") or ""),
            source=str(value.get("source") or "fixed_window"),
            source_segment_id=value.get("source_segment_id") if value.get("source_segment_id") is not None else None,
            visual_caption=str(value.get("visual_caption") or ""),
            visual_caption_source=str(value.get("visual_caption_source") or ""),
            asr_summary=str(value.get("asr_summary") or ""),
            asr_summary_source=str(value.get("asr_summary_source") or ""),
            raw_asr_ref=str(value.get("raw_asr_ref") or ""),
            stage_tags=tuple(str(item) for item in value.get("stage_tags") or ()),
            entities=tuple(str(item) for item in value.get("entities") or ()),
            topic_tags=tuple(str(item) for item in value.get("topic_tags") or ()),
            confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
            grounding_quality=str(value.get("grounding_quality") or ""),
            citation_provenance={str(k): str(v) for k, v in dict(value.get("citation_provenance") or {}).items()},
        )


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SceneIndex":
        return cls(
            video_path=str(value["video_path"]),
            duration_sec=float(value["duration_sec"]),
            segments=[VideoSegment.from_dict(item) for item in value.get("segments", [])],
        )

    def summary(self, max_segments: int = 16, max_caption_chars: int = 240) -> str:
        if not self.segments:
            return "(no segments indexed)"

        lines = []
        for segment in self.segments[:max_segments]:
            parts = []
            if segment.visual_caption:
                parts.append(f"Visual: {_bounded_text(segment.visual_caption, max_caption_chars)}")
            if segment.asr_summary:
                parts.append(f"ASR: {_bounded_text(segment.asr_summary, max_caption_chars)}")
            tags = _unique_texts([*segment.topic_tags, *segment.stage_tags])
            if tags:
                parts.append(f"Tags: {', '.join(tags)}")
            entities = _unique_texts(segment.entities)
            if entities:
                parts.append(f"Entities: {', '.join(entities)}")
            if not parts:
                caption = segment.low_fps_caption or segment.keyframe_path or "no coarse caption yet"
                parts.append(_bounded_text(caption, max_caption_chars))
            lines.append(f"{segment.segment_id} [{segment.start_sec:.1f}-{segment.end_sec:.1f}s] {' | '.join(parts)}")
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


def _bounded_text(value: str, limit: int) -> str:
    value = " ".join(str(value or "").split())
    if limit <= 0 or len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _unique_texts(values: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
