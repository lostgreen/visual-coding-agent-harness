"""Structured video workspace index for navigation-style agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .video_index import SceneIndex


@dataclass(frozen=True)
class VideoMapSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    source: str = "fixed_window"
    keyframe_paths: Sequence[str] = field(default_factory=list)
    low_fps_caption: str = ""
    asr_text: str = ""
    ocr_text: str = ""
    entities: Sequence[str] = field(default_factory=list)
    embedding_refs: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "segment_id": self.segment_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "source": self.source,
            "keyframe_paths": list(self.keyframe_paths),
            "low_fps_caption": self.low_fps_caption,
            "asr_text": self.asr_text,
            "ocr_text": self.ocr_text,
            "entities": list(self.entities),
            "embedding_refs": list(self.embedding_refs),
        }

    def compact_text(self) -> str:
        parts = [
            self.low_fps_caption,
            self.asr_text,
            self.ocr_text,
            " ".join(self.entities),
        ]
        return " | ".join(part for part in parts if part)


@dataclass(frozen=True)
class VideoSearchResult:
    segment: VideoMapSegment
    score: float
    matched_fields: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "segment_id": self.segment.segment_id,
            "start_sec": self.segment.start_sec,
            "end_sec": self.segment.end_sec,
            "score": self.score,
            "matched_fields": list(self.matched_fields),
            "summary": self.segment.compact_text(),
        }


@dataclass(frozen=True)
class VideoMap:
    video_path: str
    duration_sec: float
    segments: Sequence[VideoMapSegment] = field(default_factory=list)

    @classmethod
    def from_scene_index(cls, scene_index: SceneIndex) -> "VideoMap":
        return cls(
            video_path=scene_index.video_path,
            duration_sec=scene_index.duration_sec,
            segments=[
                VideoMapSegment(
                    segment_id=segment.segment_id,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    source=segment.source,
                    keyframe_paths=[segment.keyframe_path] if segment.keyframe_path else [],
                    low_fps_caption=segment.low_fps_caption,
                )
                for segment in scene_index.segments
            ],
        )

    def get(self, segment_id: str) -> VideoMapSegment:
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
            compact = segment.compact_text() or "no index text yet"
            lines.append(f"{segment.segment_id} [{segment.start_sec:.1f}-{segment.end_sec:.1f}s] {compact}")
        remaining = len(self.segments) - max_segments
        if remaining > 0:
            lines.append(f"... {remaining} more segments omitted")
        return "\n".join(lines)

    def search(self, query: str, top_k: int = 5, modalities: Sequence[str] = ()) -> Sequence[VideoSearchResult]:
        query_terms = _tokens(query)
        if not query_terms:
            return []
        allowed_fields = set(modalities or ["low_fps_caption", "asr_text", "ocr_text", "entities"])
        results = []
        for segment in self.segments:
            score = 0.0
            matched_fields = []
            for field_name, field_value in _search_fields(segment).items():
                if field_name not in allowed_fields:
                    continue
                field_terms = _tokens(field_value)
                overlap = query_terms.intersection(field_terms)
                if overlap:
                    matched_fields.append(field_name)
                    score += len(overlap) / max(len(query_terms), 1)
            if score > 0:
                results.append(
                    VideoSearchResult(
                        segment=segment,
                        score=round(score, 3),
                        matched_fields=matched_fields,
                    )
                )
        return sorted(results, key=lambda result: (-result.score, result.segment.start_sec))[:top_k]


def _search_fields(segment: VideoMapSegment) -> Mapping[str, str]:
    return {
        "low_fps_caption": segment.low_fps_caption,
        "asr_text": segment.asr_text,
        "ocr_text": segment.ocr_text,
        "entities": " ".join(segment.entities),
    }


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)}
