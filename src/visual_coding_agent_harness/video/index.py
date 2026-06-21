"""Lightweight video scene index used by iterative visual agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence


_TIMELINE_BEAT_MODALITIES = frozenset({"visual", "asr", "ocr", "temporal"})


@dataclass(frozen=True)
class TimelineBeat:
    beat_id: str
    start_sec: float
    end_sec: float
    summary: str
    entity_hints: Sequence[str] = field(default_factory=tuple)
    modality_hints: Sequence[str] = field(default_factory=tuple)
    confidence: Optional[float] = None
    frame_refs: Sequence[str] = field(default_factory=tuple)
    limitations: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError("TimelineBeat end_sec must be greater than or equal to start_sec")
        invalid = [modality for modality in self.modality_hints if modality not in _TIMELINE_BEAT_MODALITIES]
        if invalid:
            valid = "|".join(sorted(_TIMELINE_BEAT_MODALITIES))
            raise ValueError(f"TimelineBeat modality_hints must be one of {valid}; got {invalid[0]!r}")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "beat_id": self.beat_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "summary": self.summary,
            "entity_hints": list(self.entity_hints),
            "modality_hints": list(self.modality_hints),
            "confidence": self.confidence,
            "frame_refs": list(self.frame_refs),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimelineBeat":
        return cls(
            beat_id=str(value.get("beat_id") or ""),
            start_sec=float(value["start_sec"]),
            end_sec=float(value["end_sec"]),
            summary=str(value.get("summary") or ""),
            entity_hints=tuple(str(item) for item in value.get("entity_hints") or ()),
            modality_hints=tuple(str(item) for item in value.get("modality_hints") or ()),
            confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
            frame_refs=tuple(str(item) for item in value.get("frame_refs") or ()),
            limitations=tuple(str(item) for item in value.get("limitations") or ()),
        )


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
    map_summary: str = ""
    raw_asr_ref: str = ""
    stage_tags: Sequence[str] = field(default_factory=tuple)
    entities: Sequence[str] = field(default_factory=tuple)
    topic_tags: Sequence[str] = field(default_factory=tuple)
    confidence: Optional[float] = None
    grounding_quality: str = ""
    citation_provenance: Mapping[str, str] = field(default_factory=dict)
    asr_sentences: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    ocr_frames: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    limitations: Sequence[str] = field(default_factory=tuple)
    index_level: Literal["root", "refined"] = "root"
    parent_segment_id: Optional[str] = None
    root_segment_id: Optional[str] = None
    timeline_beats: Sequence[TimelineBeat] = field(default_factory=tuple)
    refinement_state: Literal["coarse", "refined"] = "coarse"
    index_provenance: Mapping[str, object] = field(default_factory=dict)

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
            "map_summary": self.map_summary,
            "raw_asr_ref": self.raw_asr_ref,
            "stage_tags": list(self.stage_tags),
            "entities": list(self.entities),
            "topic_tags": list(self.topic_tags),
            "confidence": self.confidence,
            "grounding_quality": self.grounding_quality,
            "citation_provenance": dict(self.citation_provenance),
            "asr_sentences": [dict(item) for item in self.asr_sentences],
            "ocr_frames": [dict(item) for item in self.ocr_frames],
            "limitations": list(self.limitations),
            "index_level": self.index_level,
            "parent_segment_id": self.parent_segment_id,
            "root_segment_id": self.root_segment_id,
            "timeline_beats": [beat.to_dict() for beat in self.timeline_beats],
            "refinement_state": self.refinement_state,
            "index_provenance": dict(self.index_provenance),
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
            map_summary=str(value.get("map_summary") or ""),
            raw_asr_ref=str(value.get("raw_asr_ref") or ""),
            stage_tags=tuple(str(item) for item in value.get("stage_tags") or ()),
            entities=tuple(str(item) for item in value.get("entities") or ()),
            topic_tags=tuple(str(item) for item in value.get("topic_tags") or ()),
            confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
            grounding_quality=str(value.get("grounding_quality") or ""),
            citation_provenance={str(k): str(v) for k, v in dict(value.get("citation_provenance") or {}).items()},
            asr_sentences=tuple(dict(item) for item in value.get("asr_sentences") or ()),
            ocr_frames=tuple(dict(item) for item in value.get("ocr_frames") or ()),
            limitations=tuple(str(item) for item in value.get("limitations") or ()),
            index_level=_coerce_literal(value.get("index_level"), allowed=("root", "refined"), default="root"),
            parent_segment_id=value.get("parent_segment_id") if value.get("parent_segment_id") is not None else None,
            root_segment_id=value.get("root_segment_id") if value.get("root_segment_id") is not None else None,
            timeline_beats=tuple(TimelineBeat.from_dict(item) for item in value.get("timeline_beats") or ()),
            refinement_state=_coerce_literal(
                value.get("refinement_state"),
                allowed=("coarse", "refined"),
                default="coarse",
            ),
            index_provenance=dict(value.get("index_provenance") or {}),
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

    def summary(
        self,
        max_segments: int = 16,
        max_caption_chars: int = 240,
        target_hints: Sequence[str] = (),
    ) -> str:
        if not self.segments:
            return "(no segments indexed)"

        lines = []
        for segment in self.segments[:max_segments]:
            caption = segment.map_summary or segment.low_fps_caption or segment.keyframe_path or "no coarse caption yet"
            lines.append(
                f"{segment.segment_id} [{segment.start_sec:.1f}-{segment.end_sec:.1f}s] "
                f"{_bounded_text(caption, max_caption_chars)}"
            )
            mentions = _target_asr_mentions(segment=segment, targets=target_hints)
            if mentions:
                lines.append("  asr mentions: " + ", ".join(mentions))
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


def _target_asr_mentions(*, segment: VideoSegment, targets: Sequence[str]) -> list[str]:
    mentions: list[str] = []
    for target in [str(item).strip() for item in targets if str(item).strip()]:
        for sentence in segment.asr_sentences:
            if not isinstance(sentence, Mapping):
                continue
            if not _target_phrase_in_text(target=target, text=str(sentence.get("text") or "")):
                continue
            timestamp = float(sentence.get("start_sec", segment.start_sec) or segment.start_sec)
            mentions.append(f"{target} @ ~{timestamp:.1f}s")
            break
    return mentions


def _target_phrase_in_text(*, target: str, text: str) -> bool:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", str(target or ""))]
    if not tokens:
        return False
    patterns = [tokens]
    if tokens[0] == "the" and len(tokens) > 1:
        patterns.append(tokens[1:])
    for pattern_tokens in patterns:
        regex = r"\b" + r"[\W_]+".join(re.escape(token) for token in pattern_tokens) + r"\b"
        if re.search(regex, str(text or ""), flags=re.IGNORECASE):
            return True
    return False


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


def _coerce_literal(value: Any, *, allowed: Sequence[str], default: str) -> Any:
    text = str(value or default)
    return text if text in allowed else default
