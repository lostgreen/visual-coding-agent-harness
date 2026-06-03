"""Navigation tools over a structured VideoMap workspace."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..registry import ToolRegistry, tool
from ..video_map import VideoMap, VideoMapSegment


def build_video_navigation_registry(video_map: VideoMap) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="video_ls", description="Summarize the indexed video workspace.")
    def video_ls() -> Mapping[str, object]:
        indexed_fields = _available_indexes(video_map.segments)
        claim = (
            f"Video {video_map.video_path} has {len(video_map.segments)} segments "
            f"over {video_map.duration_sec:.1f} seconds. Available indexes: {', '.join(indexed_fields) or 'none'}."
        )
        return {
            "claim": claim,
            "confidence": 1.0,
            "input_artifacts": [video_map.video_path],
            "regions": [
                {
                    "segment_count": len(video_map.segments),
                    "duration_sec": video_map.duration_sec,
                    "available_indexes": indexed_fields,
                }
            ],
            "raw_video_map": video_map.to_dict(),
        }

    @tool(name="search_segments", description="Search indexed video segments by text query.")
    def search_segments(query: str, top_k: int = 5, modalities: Sequence[str] = ()) -> Mapping[str, object]:
        results = video_map.search(query=query, top_k=top_k, modalities=modalities)
        if results:
            ids = ", ".join(result.segment.segment_id for result in results)
            claim = f"Search for '{query}' returned candidate segments: {ids}."
        else:
            claim = f"Search for '{query}' returned no candidate segments."
        return {
            "claim": claim,
            "confidence": 0.85 if results else 0.2,
            "input_artifacts": [video_map.video_path],
            "regions": [result.to_dict() for result in results],
            "limitations": "Lexical VideoMap search; embedding retrieval should replace this for semantic recall.",
        }

    @tool(name="read_segment", description="Read compact indexed metadata for one segment.")
    def read_segment(segment_id: str) -> Mapping[str, object]:
        segment = video_map.get(segment_id)
        claim = _segment_claim(segment)
        return {
            "claim": claim,
            "confidence": 1.0,
            "input_artifacts": [video_map.video_path],
            "regions": [segment.to_dict()],
        }

    @tool(name="expand_window", description="Return a bounded temporal window around a segment.")
    def expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0) -> Mapping[str, object]:
        segment = video_map.get(segment_id)
        start_sec = max(0.0, segment.start_sec - before_sec)
        end_sec = min(video_map.duration_sec, segment.end_sec + after_sec)
        return {
            "claim": (
                f"Expanded {segment.segment_id} from [{segment.start_sec:.1f}, {segment.end_sec:.1f}] "
                f"to [{start_sec:.1f}, {end_sec:.1f}] seconds."
            ),
            "confidence": 1.0,
            "input_artifacts": [video_map.video_path],
            "regions": [
                {
                    "segment_id": segment.segment_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_start_sec": segment.start_sec,
                    "source_end_sec": segment.end_sec,
                }
            ],
        }

    registry.register(video_ls)
    registry.register(search_segments)
    registry.register(read_segment)
    registry.register(expand_window)
    return registry


def _available_indexes(segments: Sequence[VideoMapSegment]) -> Sequence[str]:
    indexes = []
    checks = [
        ("keyframes", lambda segment: bool(segment.keyframe_paths)),
        ("captions", lambda segment: bool(segment.low_fps_caption)),
        ("asr", lambda segment: bool(segment.asr_text)),
        ("ocr", lambda segment: bool(segment.ocr_text)),
        ("entities", lambda segment: bool(segment.entities)),
        ("embeddings", lambda segment: bool(segment.embedding_refs)),
    ]
    for name, predicate in checks:
        if any(predicate(segment) for segment in segments):
            indexes.append(name)
    return indexes


def _segment_claim(segment: VideoMapSegment) -> str:
    parts = [
        f"{segment.segment_id} covers {segment.start_sec:.1f}-{segment.end_sec:.1f}s.",
        f"caption: {segment.low_fps_caption}" if segment.low_fps_caption else "",
        f"ASR: {segment.asr_text}" if segment.asr_text else "",
        f"OCR: {segment.ocr_text}" if segment.ocr_text else "",
        f"entities: {', '.join(segment.entities)}" if segment.entities else "",
    ]
    return " ".join(part for part in parts if part)
