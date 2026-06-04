"""Navigation tools over a structured VideoMap workspace."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..registry import ToolRegistry, tool
from ..video_map import VideoMap, VideoMapSegment, VideoMapStore


def build_video_navigation_registry(video_map: VideoMap | VideoMapStore) -> ToolRegistry:
    registry = ToolRegistry()
    video_map_store = video_map if isinstance(video_map, VideoMapStore) else VideoMapStore(video_map)

    @tool(name="video_ls", description="Build a compact map-first overview of the indexed video workspace.")
    def video_ls(query: str = "", max_segments: int = 16, top_k: int = 5) -> Mapping[str, object]:
        current = video_map_store.current
        indexed_fields = _available_indexes(current.segments)
        overview = current.overview(query=query, max_segments=max_segments, top_k=top_k)
        candidate_ids = [str(candidate["segment_id"]) for candidate in overview["candidates"]]
        candidate_text = ", ".join(candidate_ids) if candidate_ids else "none"
        claim = (
            f"map-first video_ls: Video {current.video_path} has {len(current.segments)} segments "
            f"over {current.duration_sec:.1f} seconds. Available indexes: {', '.join(indexed_fields) or 'none'}. "
            f"Candidate segments: {candidate_text}."
        )
        return {
            "claim": claim,
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "regions": [
                {
                    "segment_count": len(current.segments),
                    "duration_sec": current.duration_sec,
                    "available_indexes": indexed_fields,
                }
            ],
            "coverage": overview["coverage"],
            "outline": overview["outline"],
            "candidates": overview["candidates"],
            "recommended_next_tools": overview["recommended_next_tools"],
            "raw_video_map": current.to_dict(),
        }

    @tool(name="search_segments", description="Search indexed video segments by text query.")
    def search_segments(query: str, top_k: int = 5, modalities: Sequence[str] = ()) -> Mapping[str, object]:
        current = video_map_store.current
        results = current.search(query=query, top_k=top_k, modalities=modalities)
        if results:
            ids = ", ".join(result.segment.segment_id for result in results)
            claim = f"Search for '{query}' returned candidate segments: {ids}."
        else:
            claim = f"Search for '{query}' returned no candidate segments."
        return {
            "claim": claim,
            "confidence": 0.85 if results else 0.2,
            "input_artifacts": [current.video_path],
            "regions": [result.to_dict() for result in results],
            "modalities": _modality_results(current=current, query=query, top_k=top_k, modalities=modalities),
            "limitations": (
                "Training-free VideoMap retrieval over caption/ASR/OCR/entity indexes; "
                "embedding retrieval can replace the scoring backend without changing this contract."
            ),
        }

    @tool(name="read_segment", description="Read compact indexed metadata for one segment.")
    def read_segment(segment_id: str) -> Mapping[str, object]:
        current = video_map_store.current
        segment = current.get(segment_id)
        claim = _segment_claim(segment)
        return {
            "claim": claim,
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "regions": [segment.to_dict()],
        }

    @tool(name="expand_window", description="Return a bounded temporal window around a segment.")
    def expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0) -> Mapping[str, object]:
        current = video_map_store.current
        segment = current.get(segment_id)
        start_sec = max(0.0, segment.start_sec - before_sec)
        end_sec = min(current.duration_sec, segment.end_sec + after_sec)
        return {
            "claim": (
                f"Expanded {segment.segment_id} from [{segment.start_sec:.1f}, {segment.end_sec:.1f}] "
                f"to [{start_sec:.1f}, {end_sec:.1f}] seconds."
            ),
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
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

    @tool(name="zoom", description="Materialize finer child segments for a coarse VideoMap segment.")
    def zoom(segment_id: str, target_granularity_sec: float = 60.0) -> Mapping[str, object]:
        current = video_map_store.current
        parent = current.get(segment_id)
        children = video_map_store.materialize_zoom(
            segment_id,
            target_granularity_sec=target_granularity_sec,
        )
        child_ids = ", ".join(child.segment_id for child in children)
        return {
            "claim": (
                f"Materialized {len(children)} child segment{'s' if len(children) != 1 else ''} "
                f"from {segment_id}: {child_ids}."
            ),
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "regions": [
                {
                    "segment_id": parent.segment_id,
                    "start_sec": parent.start_sec,
                    "end_sec": parent.end_sec,
                    "target_granularity_sec": target_granularity_sec,
                    "child_segments": [child.to_dict() for child in children],
                }
            ],
            "recommended_next_tools": [
                {
                    "tool": "inspect_segment",
                    "args": {
                        "segment_id": child.segment_id,
                        "start_sec": child.start_sec,
                        "end_sec": child.end_sec,
                    },
                    "reason": "Delegate localized inspection on this finer temporal window.",
                }
                for child in children
            ],
        }

    registry.register(video_ls)
    registry.register(search_segments)
    registry.register(read_segment)
    registry.register(expand_window)
    registry.register(zoom)
    return registry


def _current_video_map(video_map: VideoMap | VideoMapStore) -> VideoMap:
    if isinstance(video_map, VideoMapStore):
        return video_map.current
    return video_map


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


def _modality_results(
    *,
    current: VideoMap,
    query: str,
    top_k: int,
    modalities: Sequence[str],
) -> Mapping[str, Sequence[Mapping[str, object]]]:
    requested = [str(modality).lower() for modality in modalities]
    channels = requested or ["caption", "asr", "ocr", "entities"]
    grouped = {}
    for channel in channels:
        results = current.search(query=query, top_k=top_k, modalities=[channel])
        grouped[channel] = [result.to_dict() for result in results]
    return grouped


def _segment_claim(segment: VideoMapSegment) -> str:
    parts = [
        f"{segment.segment_id} covers {segment.start_sec:.1f}-{segment.end_sec:.1f}s.",
        f"caption: {segment.low_fps_caption}" if segment.low_fps_caption else "",
        f"ASR: {segment.asr_text}" if segment.asr_text else "",
        f"OCR: {segment.ocr_text}" if segment.ocr_text else "",
        f"entities: {', '.join(segment.entities)}" if segment.entities else "",
    ]
    return " ".join(part for part in parts if part)
