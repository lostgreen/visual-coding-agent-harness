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
        for segment in self._sample_segments(max_segments):
            compact = segment.compact_text() or "no index text yet"
            lines.append(f"{segment.segment_id} [{segment.start_sec:.1f}-{segment.end_sec:.1f}s] {compact}")
        remaining = len(self.segments) - max_segments
        if remaining > 0:
            lines.append(f"... {remaining} more segments omitted")
        return "\n".join(lines)

    def coverage(self) -> Mapping[str, object]:
        field_counts = {
            "keyframe_paths": sum(1 for segment in self.segments if segment.keyframe_paths),
            "low_fps_caption": sum(1 for segment in self.segments if segment.low_fps_caption),
            "asr_text": sum(1 for segment in self.segments if segment.asr_text),
            "ocr_text": sum(1 for segment in self.segments if segment.ocr_text),
            "entities": sum(1 for segment in self.segments if segment.entities),
            "embedding_refs": sum(1 for segment in self.segments if segment.embedding_refs),
        }
        indexed_segment_count = sum(1 for segment in self.segments if segment.compact_text())
        return {
            "segment_count": len(self.segments),
            "duration_sec": self.duration_sec,
            "field_counts": field_counts,
            "available_indexes": [name for name, count in field_counts.items() if count > 0],
            "indexed_segment_count": indexed_segment_count,
            "empty_segment_count": len(self.segments) - indexed_segment_count,
        }

    def outline(self, max_segments: int = 16) -> Sequence[Mapping[str, object]]:
        return [
            {
                "segment_id": segment.segment_id,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "source": segment.source,
                "summary": segment.compact_text() or "no index text yet",
                "index_fields": _populated_fields(segment),
            }
            for segment in self._sample_segments(max_segments)
        ]

    def candidates(self, query: str = "", top_k: int = 5) -> Sequence[VideoSearchResult]:
        if query.strip():
            searched = self.search(query=query, top_k=top_k)
            if searched:
                return searched
        return self.anchor_segments(max_segments=top_k)

    def anchor_segments(self, max_segments: int = 5) -> Sequence[VideoSearchResult]:
        if not self.segments or max_segments <= 0:
            return []

        selected: list[VideoMapSegment] = []
        for segment in self._sample_segments(min(3, max_segments)):
            if segment not in selected:
                selected.append(segment)

        rich_segments = sorted(
            self.segments,
            key=lambda segment: (_text_richness(segment), -segment.start_sec),
            reverse=True,
        )
        for segment in rich_segments:
            if segment not in selected:
                selected.append(segment)
            if len(selected) >= max_segments:
                break

        results = []
        for index, segment in enumerate(selected[:max_segments]):
            matched_fields = _populated_fields(segment) or ["timeline_anchor"]
            score = max(0.1, round(1.0 - (index * 0.1), 3))
            results.append(VideoSearchResult(segment=segment, score=score, matched_fields=matched_fields))
        return results

    def overview(self, query: str = "", max_segments: int = 16, top_k: int = 5) -> Mapping[str, object]:
        candidates = self.candidates(query=query, top_k=top_k)
        return {
            "coverage": self.coverage(),
            "outline": self.outline(max_segments=max_segments),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "recommended_next_tools": _recommended_next_tools(query=query, candidates=candidates),
        }

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

    def _sample_segments(self, max_segments: int) -> Sequence[VideoMapSegment]:
        if max_segments <= 0:
            return []
        if len(self.segments) <= max_segments:
            return list(self.segments)
        if max_segments == 1:
            return [self.segments[0]]

        last_index = len(self.segments) - 1
        selected_indexes = []
        for slot in range(max_segments):
            index = round(slot * last_index / (max_segments - 1))
            if index not in selected_indexes:
                selected_indexes.append(index)
        return [self.segments[index] for index in selected_indexes]


def _search_fields(segment: VideoMapSegment) -> Mapping[str, str]:
    return {
        "low_fps_caption": segment.low_fps_caption,
        "asr_text": segment.asr_text,
        "ocr_text": segment.ocr_text,
        "entities": " ".join(segment.entities),
    }


def _populated_fields(segment: VideoMapSegment) -> Sequence[str]:
    fields = []
    if segment.keyframe_paths:
        fields.append("keyframe_paths")
    for field_name, field_value in _search_fields(segment).items():
        if field_value:
            fields.append(field_name)
    if segment.embedding_refs:
        fields.append("embedding_refs")
    return fields


def _text_richness(segment: VideoMapSegment) -> int:
    return len(_tokens(segment.compact_text()))


def _recommended_next_tools(*, query: str, candidates: Sequence[VideoSearchResult]) -> Sequence[Mapping[str, object]]:
    if not candidates:
        return [
            {
                "tool": "caption_segment",
                "reason": "No indexed candidate was found; inspect a temporal anchor visually.",
            }
        ]

    best = candidates[0].segment
    next_tools = [
        {
            "tool": "read_segment",
            "args": {"segment_id": best.segment_id},
            "reason": "Read compact metadata before spending VLM budget.",
        },
        {
            "tool": "caption_segment",
            "args": {"segment_id": best.segment_id},
            "reason": "Use the shared VLM to verify the top candidate visually.",
        },
    ]
    if query.strip():
        next_tools.append(
            {
                "tool": "qa_segment",
                "args": {"segment_id": best.segment_id, "question": query},
                "reason": "Ask a targeted question on the best localized segment.",
            }
        )
    next_tools.append(
        {
            "tool": "expand_window",
            "args": {"segment_id": best.segment_id, "before_sec": 30.0, "after_sec": 30.0},
            "reason": "Expand temporal context if the local observation is incomplete.",
        }
    )
    return next_tools


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)}
