"""Lightweight video scene index used by iterative visual agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence


_TIMELINE_BEAT_MODALITIES = frozenset({"visual", "asr", "ocr", "temporal"})


@dataclass(frozen=True)
class Frame:
    frame_id: str
    time_sec: float
    thumb_path: str
    thumb_embedding: bytes = b""
    ocr_text: str = ""

    def to_dict(self) -> Mapping[str, object]:
        return {
            "frame_id": self.frame_id,
            "time_sec": float(self.time_sec),
            "thumb_path": self.thumb_path,
            "thumb_embedding": self.thumb_embedding.hex(),
            "ocr_text": self.ocr_text,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Frame":
        embedding = value.get("thumb_embedding") or ""
        return cls(
            frame_id=str(value.get("frame_id") or ""),
            time_sec=float(value.get("time_sec", 0.0) or 0.0),
            thumb_path=str(value.get("thumb_path") or ""),
            thumb_embedding=bytes.fromhex(str(embedding)) if str(embedding) else b"",
            ocr_text=str(value.get("ocr_text") or ""),
        )


@dataclass(frozen=True)
class Shot:
    shot_id: str
    scene_id: str
    start_sec: float
    end_sec: float
    frames: Sequence[Frame] = field(default_factory=tuple)
    visual_caption: str = ""
    asr_text: str = ""
    ocr_lines: Sequence[str] = field(default_factory=tuple)
    entities: Sequence[str] = field(default_factory=tuple)
    lowres_grid_path: str = ""
    source_segment_id: str = ""

    def __post_init__(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError("Shot end_sec must be greater than or equal to start_sec")
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "ocr_lines", _clean_text_tuple(self.ocr_lines))
        object.__setattr__(self, "entities", _clean_text_tuple(self.entities))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "shot_id": self.shot_id,
            "scene_id": self.scene_id,
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "frames": [frame.to_dict() for frame in self.frames],
            "visual_caption": self.visual_caption,
            "asr_text": self.asr_text,
            "ocr_lines": list(self.ocr_lines),
            "entities": list(self.entities),
            "lowres_grid_path": self.lowres_grid_path,
            "source_segment_id": self.source_segment_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Shot":
        return cls(
            shot_id=str(value.get("shot_id") or ""),
            scene_id=str(value.get("scene_id") or ""),
            start_sec=float(value.get("start_sec", 0.0) or 0.0),
            end_sec=float(value.get("end_sec", 0.0) or 0.0),
            frames=tuple(Frame.from_dict(item) for item in value.get("frames") or ()),
            visual_caption=str(value.get("visual_caption") or ""),
            asr_text=str(value.get("asr_text") or ""),
            ocr_lines=_clean_text_tuple(value.get("ocr_lines") or ()),
            entities=_clean_text_tuple(value.get("entities") or ()),
            lowres_grid_path=str(value.get("lowres_grid_path") or ""),
            source_segment_id=str(value.get("source_segment_id") or ""),
        )


@dataclass(frozen=True)
class Scene:
    scene_id: str
    start_sec: float
    end_sec: float
    title: str
    summary: str
    shots: Sequence[Shot] = field(default_factory=tuple)
    dominant_entities: Sequence[str] = field(default_factory=tuple)
    dominant_topics: Sequence[str] = field(default_factory=tuple)
    scene_thumb_path: str = ""
    source_segment_id: str = ""

    def __post_init__(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError("Scene end_sec must be greater than or equal to start_sec")
        shots = tuple(self.shots)
        for shot in shots:
            if shot.scene_id != self.scene_id:
                raise ValueError("Shot scene_id must match parent scene")
            if shot.start_sec < self.start_sec or shot.end_sec > self.end_sec:
                raise ValueError("Shot time range must be within scene time range")
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "dominant_entities", _clean_text_tuple(self.dominant_entities))
        object.__setattr__(self, "dominant_topics", _clean_text_tuple(self.dominant_topics))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "scene_id": self.scene_id,
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "title": self.title,
            "summary": self.summary,
            "shots": [shot.to_dict() for shot in self.shots],
            "dominant_entities": list(self.dominant_entities),
            "dominant_topics": list(self.dominant_topics),
            "scene_thumb_path": self.scene_thumb_path,
            "source_segment_id": self.source_segment_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Scene":
        return cls(
            scene_id=str(value.get("scene_id") or ""),
            start_sec=float(value.get("start_sec", 0.0) or 0.0),
            end_sec=float(value.get("end_sec", 0.0) or 0.0),
            title=str(value.get("title") or ""),
            summary=str(value.get("summary") or ""),
            shots=tuple(Shot.from_dict(item) for item in value.get("shots") or ()),
            dominant_entities=_clean_text_tuple(value.get("dominant_entities") or ()),
            dominant_topics=_clean_text_tuple(value.get("dominant_topics") or ()),
            scene_thumb_path=str(value.get("scene_thumb_path") or ""),
            source_segment_id=str(value.get("source_segment_id") or ""),
        )


@dataclass(frozen=True)
class VideoIndex:
    video_path: str
    duration_sec: float
    scenes: Sequence[Scene] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenes", tuple(self.scenes))
        if self.duration_sec < 0:
            raise ValueError("duration_sec must be non-negative")

    def get_scene(self, scene_id: str) -> Scene:
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise ValueError(f"Unknown scene_id: {scene_id}")

    def get_shot(self, shot_id: str) -> Shot:
        for scene in self.scenes:
            for shot in scene.shots:
                if shot.shot_id == shot_id:
                    return shot
        raise ValueError(f"Unknown shot_id: {shot_id}")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "video_path": self.video_path,
            "duration_sec": float(self.duration_sec),
            "scenes": [scene.to_dict() for scene in self.scenes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VideoIndex":
        return cls(
            video_path=str(value.get("video_path") or ""),
            duration_sec=float(value.get("duration_sec", 0.0) or 0.0),
            scenes=tuple(Scene.from_dict(item) for item in value.get("scenes") or ()),
        )

    def to_scene_index(self) -> "SceneIndex":
        segments = []
        for scene in self.scenes:
            shot = scene.shots[0] if scene.shots else None
            segments.append(
                VideoSegment(
                    segment_id=scene.source_segment_id or (shot.source_segment_id if shot else "") or scene.scene_id,
                    start_sec=scene.start_sec,
                    end_sec=scene.end_sec,
                    keyframe_path=scene.scene_thumb_path,
                    low_fps_caption=scene.summary,
                    visual_caption=shot.visual_caption if shot else "",
                    asr_summary=shot.asr_text if shot else "",
                    map_summary=scene.summary,
                    entities=scene.dominant_entities or (shot.entities if shot else ()),
                    topic_tags=scene.dominant_topics,
                    index_provenance={"source": "multi_v3_video_index", "scene_id": scene.scene_id},
                )
            )
        return SceneIndex(video_path=self.video_path, duration_sec=self.duration_sec, segments=segments)

    def summary(self, max_scenes: int = 64, max_summary_chars: int = 240) -> str:
        if not self.scenes:
            return "(no scenes indexed)"
        lines = []
        for scene in self.scenes[:max_scenes]:
            entities = f" | entities: {', '.join(scene.dominant_entities)}" if scene.dominant_entities else ""
            lines.append(
                f"{scene.scene_id} [{scene.start_sec:.1f}-{scene.end_sec:.1f}s] "
                f"{scene.title}: {_bounded_text(scene.summary, max_summary_chars)}{entities}"
            )
        remaining = len(self.scenes) - max_scenes
        if remaining > 0:
            lines.append(f"... {remaining} more scenes omitted")
        return "\n".join(lines)


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


def _clean_text_tuple(values: object) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes)):
        sequence = () if values is None else (values,)
    else:
        try:
            sequence = tuple(values)  # type: ignore[arg-type]
        except TypeError:
            sequence = (values,)
    return tuple(text for item in sequence if (text := " ".join(str(item or "").split())))


def _coerce_literal(value: Any, *, allowed: Sequence[str], default: str) -> Any:
    text = str(value or default)
    return text if text in allowed else default
