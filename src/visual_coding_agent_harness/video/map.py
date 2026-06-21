"""Structured video workspace index for navigation-style agents."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from .text_norm import unique_tokens
from .index import SceneIndex, TimelineBeat

FrameSampler = Callable[[str, float, float, int], Sequence[str]]


_CANONICAL_SEARCH_MODALITIES = ("caption", "asr", "ocr", "entities")
_SEARCH_MODALITY_VALID_TEXT = "caption|asr|ocr|entities"
_SEARCH_MODALITY_TO_FIELDS = {
    "caption": ("low_fps_caption",),
    "asr": ("asr_text",),
    "ocr": ("ocr_text",),
    "entities": ("entities",),
}
_SEARCH_MODALITY_ALIASES = {
    "caption": ("caption",),
    "captions": ("caption",),
    "visual": ("caption", "entities"),
    "asr": ("asr",),
    "audio": ("asr",),
    "speech": ("asr",),
    "ocr": ("ocr",),
    "screen": ("ocr",),
    "text": ("caption", "asr", "ocr"),
    "entities": ("entities",),
    "objects": ("entities",),
    "low_fps_caption": ("caption",),
    "asr_text": ("asr",),
    "ocr_text": ("ocr",),
}


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
    asr_sentences: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    ocr_frames: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    limitations: Sequence[str] = field(default_factory=tuple)
    index_level: Literal["root", "refined"] = "root"
    parent_segment_id: str | None = None
    root_segment_id: str | None = None
    timeline_beats: Sequence[TimelineBeat] = field(default_factory=tuple)
    refinement_state: Literal["coarse", "refined"] = "coarse"
    index_provenance: Mapping[str, object] = field(default_factory=dict)

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
    matches: Sequence[Mapping[str, object]] = field(default_factory=list)
    relevance_reason: str = ""

    def to_dict(self) -> Mapping[str, object]:
        return {
            "segment_id": self.segment.segment_id,
            "start_sec": self.segment.start_sec,
            "end_sec": self.segment.end_sec,
            "score": self.score,
            "matched_fields": list(self.matched_fields),
            "matches": [dict(match) for match in self.matches],
            "summary": self.segment.compact_text(),
            "relevance_reason": self.relevance_reason or _relevance_reason(self.matched_fields),
        }


@dataclass(frozen=True)
class IndexRefinementPatch:
    patch_id: str
    parent_segment_id: str
    requested_start_sec: float
    requested_end_sec: float
    resolution: Literal["coarse", "medium", "dense"]
    children: Sequence[VideoMapSegment]
    cache_hit: bool
    cost: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "patch_id": self.patch_id,
            "parent_segment_id": self.parent_segment_id,
            "requested_start_sec": self.requested_start_sec,
            "requested_end_sec": self.requested_end_sec,
            "resolution": self.resolution,
            "children": [child.to_dict() for child in self.children],
            "cache_hit": self.cache_hit,
            "cost": dict(self.cost),
            "provenance": dict(self.provenance),
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
                    low_fps_caption=segment.visual_caption or segment.low_fps_caption,
                    asr_text=segment.asr_summary,
                    asr_sentences=tuple(dict(item) for item in segment.asr_sentences),
                    ocr_frames=tuple(dict(item) for item in segment.ocr_frames),
                    limitations=tuple(str(item) for item in segment.limitations),
                    index_level=segment.index_level,
                    parent_segment_id=segment.parent_segment_id,
                    root_segment_id=segment.root_segment_id,
                    timeline_beats=tuple(segment.timeline_beats),
                    refinement_state=segment.refinement_state,
                    index_provenance=dict(segment.index_provenance),
                    entities=_unique_texts([*segment.entities, *segment.topic_tags, *segment.stage_tags]),
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
            results.append(
                VideoSearchResult(
                    segment=segment,
                    score=score,
                    matched_fields=matched_fields,
                    relevance_reason="Timeline anchor selected for broad coverage.",
                )
            )
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
        allowed_fields = _resolve_search_fields(modalities)
        results = []
        for segment in self.segments:
            score = 0.0
            matched_fields = []
            matches = []
            for field_name, field_value in _search_fields(segment).items():
                if field_name not in allowed_fields:
                    continue
                field_terms = _tokens(field_value)
                overlap = query_terms.intersection(field_terms)
                if overlap:
                    matched_fields.append(field_name)
                    field_score = len(overlap) / max(len(query_terms), 1)
                    score += field_score
                    matches.append(
                        {
                            "modality": _field_modality(field_name),
                            "field": field_name,
                            "score": round(field_score, 3),
                            "matched_terms": sorted(overlap),
                            "evidence": _evidence_snippet(field_value),
                        }
                    )
            if score > 0:
                results.append(
                    VideoSearchResult(
                        segment=segment,
                        score=round(score, 3),
                        matched_fields=matched_fields,
                        matches=matches,
                        relevance_reason=_match_reason(matches),
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


def _resolve_search_fields(modalities: Sequence[str]) -> set[str]:
    fields: set[str] = set()
    for modality in _resolve_search_modalities(modalities):
        fields.update(_SEARCH_MODALITY_TO_FIELDS[modality])
    return fields


def _resolve_search_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    if not modalities:
        return _CANONICAL_SEARCH_MODALITIES

    resolved: list[str] = []
    for modality in modalities:
        normalized = str(modality).strip().lower()
        for canonical in _SEARCH_MODALITY_ALIASES.get(normalized, ()):
            if canonical not in resolved:
                resolved.append(canonical)
    if not resolved:
        return _CANONICAL_SEARCH_MODALITIES
    return tuple(resolved)


def search_modality_limitations(modalities: Sequence[str]) -> tuple[str, ...]:
    unknown = []
    for modality in modalities:
        raw = str(modality).strip()
        if not raw:
            continue
        if raw.lower() in _SEARCH_MODALITY_ALIASES:
            continue
        if raw not in unknown:
            unknown.append(raw)
    return tuple(
        f"unknown modality '{modality}' ignored; valid: {_SEARCH_MODALITY_VALID_TEXT}" for modality in unknown
    )


def _field_modality(field_name: str) -> str:
    return {
        "low_fps_caption": "caption",
        "asr_text": "asr",
        "ocr_text": "ocr",
        "entities": "entities",
    }.get(field_name, field_name)


def _evidence_snippet(text: str, max_chars: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _match_reason(matches: Sequence[Mapping[str, object]]) -> str:
    modalities = []
    terms = []
    for match in matches:
        modality = str(match.get("modality", ""))
        if modality and modality not in modalities:
            modalities.append(modality)
        for term in match.get("matched_terms", []):
            term_text = str(term)
            if term_text not in terms:
                terms.append(term_text)
    if not matches:
        return "No query-specific index match."
    return f"Matched query terms {', '.join(terms)} in {', '.join(modalities)} indexes."


def _relevance_reason(matched_fields: Sequence[str]) -> str:
    if not matched_fields:
        return "No indexed fields matched."
    return f"Selected from indexed fields: {', '.join(matched_fields)}."


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
    ]
    if query.strip():
        next_tools.append(
            {
                "tool": "inspect_segment",
                "args": {"segment_id": best.segment_id, "question": query},
                "reason": "Delegate targeted visual inspection on the best localized segment.",
            }
        )
    next_tools.append(
        {
            "tool": "caption_segment",
            "args": {"segment_id": best.segment_id},
            "reason": "Use the shared VLM to verify the top candidate visually.",
        }
    )
    next_tools.append(
        {
            "tool": "zoom",
            "args": {"segment_id": best.segment_id, "target_granularity_sec": 60.0},
            "reason": "Materialize finer child segments when the local observation is too coarse.",
        }
    )
    return next_tools


def _tokens(text: str) -> set[str]:
    return unique_tokens(text)


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


class VideoMapStore:
    """Mutable holder for an evolving VideoMap workspace."""

    def __init__(self, video_map: VideoMap) -> None:
        self.current = video_map
        self._refinement_registry: dict[tuple[str, float, float, str], IndexRefinementPatch] = {}

    @property
    def latest_refinement_patch(self) -> IndexRefinementPatch | None:
        if not self._refinement_registry:
            return None
        return next(reversed(self._refinement_registry.values()))

    def update_segment(
        self,
        segment_id: str,
        *,
        low_fps_caption: str | None = None,
        asr_text: str | None = None,
        ocr_text: str | None = None,
        entities: Sequence[str] | None = None,
        keyframe_paths: Sequence[str] | None = None,
        embedding_refs: Sequence[str] | None = None,
    ) -> VideoMapSegment:
        updated_segments = []
        updated_segment = None
        for segment in self.current.segments:
            if segment.segment_id != segment_id:
                updated_segments.append(segment)
                continue
            updated_segment = replace(
                segment,
                low_fps_caption=segment.low_fps_caption if low_fps_caption is None else low_fps_caption,
                asr_text=segment.asr_text if asr_text is None else asr_text,
                ocr_text=segment.ocr_text if ocr_text is None else ocr_text,
                entities=segment.entities if entities is None else list(entities),
                keyframe_paths=segment.keyframe_paths if keyframe_paths is None else list(keyframe_paths),
                embedding_refs=segment.embedding_refs if embedding_refs is None else list(embedding_refs),
            )
            updated_segments.append(updated_segment)
        if updated_segment is None:
            raise ValueError(f"Unknown segment_id: {segment_id}")
        self.current = replace(self.current, segments=updated_segments)
        return updated_segment

    def materialize_zoom(self, segment_id: str, *, target_granularity_sec: float) -> Sequence[VideoMapSegment]:
        if target_granularity_sec <= 0:
            raise ValueError("target_granularity_sec must be positive")

        parent = self.current.get(segment_id)
        duration = max(0.0, parent.end_sec - parent.start_sec)
        if duration <= 0:
            raise ValueError(f"Segment {segment_id} has empty duration")

        children: list[VideoMapSegment] = []
        child_start = parent.start_sec
        child_index = 1
        while child_start < parent.end_sec:
            child_end = min(parent.end_sec, child_start + target_granularity_sec)
            children.append(
                VideoMapSegment(
                    segment_id=f"{parent.segment_id}_z{child_index:02d}",
                    start_sec=round(child_start, 3),
                    end_sec=round(child_end, 3),
                    source=f"structural_zoom:{parent.segment_id}",
                    keyframe_paths=list(parent.keyframe_paths),
                    low_fps_caption=parent.low_fps_caption,
                    asr_text=parent.asr_text,
                    ocr_text=parent.ocr_text,
                    entities=list(parent.entities),
                    embedding_refs=list(parent.embedding_refs),
                    index_level=parent.index_level,
                    parent_segment_id=parent.segment_id,
                    root_segment_id=parent.root_segment_id or parent.segment_id,
                    limitations=(
                        "structural child only; no local perception was performed",
                    ),
                )
            )
            child_start = child_end
            child_index += 1

        existing_by_id = {segment.segment_id: segment for segment in self.current.segments}
        updated_segments = list(self.current.segments)
        for child in children:
            if child.segment_id in existing_by_id:
                updated_segments = [
                    child if segment.segment_id == child.segment_id else segment for segment in updated_segments
                ]
            else:
                updated_segments.append(child)
        self.current = replace(self.current, segments=updated_segments)
        return children

    def apply_refinement(
        self,
        *,
        parent_segment_id: str,
        requested_start_sec: float,
        requested_end_sec: float,
        resolution: Literal["coarse", "medium", "dense"],
        children: Sequence[VideoMapSegment],
        provenance: Mapping[str, object],
    ) -> IndexRefinementPatch:
        parent = self.current.get(parent_segment_id)
        if parent.index_level != "root":
            raise ValueError("Index refinement requires a root parent")
        start = round(float(requested_start_sec), 3)
        end = round(float(requested_end_sec), 3)
        if end <= start:
            raise ValueError("requested refinement range must be non-empty")
        if start < float(parent.start_sec) or end > float(parent.end_sec):
            raise ValueError("requested refinement range must stay within parent segment")
        key = (parent.segment_id, start, end, str(resolution))
        cached = self._refinement_registry.get(key)
        if cached is not None:
            return replace(cached, cache_hit=True)

        normalized_children = []
        last_end = start
        for child in children:
            if child.index_level != "refined":
                raise ValueError("refinement children must have index_level='refined'")
            if child.end_sec <= child.start_sec:
                raise ValueError("refinement children must have non-empty duration")
            if child.start_sec < last_end:
                raise ValueError("refinement children must be chronological and non-overlapping")
            if child.start_sec < parent.start_sec or child.end_sec > parent.end_sec:
                raise ValueError("refinement children must stay within parent segment")
            if child.start_sec < start or child.end_sec > end:
                raise ValueError("refinement children must stay within requested range")
            if child.low_fps_caption and child.low_fps_caption == parent.low_fps_caption:
                raise ValueError("refinement child must not copy parent caption")
            normalized_children.append(
                replace(
                    child,
                    parent_segment_id=parent.segment_id,
                    root_segment_id=parent.root_segment_id or parent.segment_id,
                    refinement_state="refined",
                    index_level="refined",
                )
            )
            last_end = child.end_sec
        patch = IndexRefinementPatch(
            patch_id=_refinement_patch_id(parent.segment_id, start, end, str(resolution)),
            parent_segment_id=parent.segment_id,
            requested_start_sec=start,
            requested_end_sec=end,
            resolution=resolution,
            children=tuple(normalized_children),
            cache_hit=False,
            provenance=dict(provenance),
        )
        existing_by_id = {segment.segment_id: segment for segment in self.current.segments}
        updated_segments = list(self.current.segments)
        for child in normalized_children:
            if child.segment_id in existing_by_id:
                updated_segments = [
                    child if segment.segment_id == child.segment_id else segment for segment in updated_segments
                ]
            else:
                updated_segments.append(child)
        self.current = replace(self.current, segments=updated_segments)
        self._refinement_registry[key] = patch
        return patch


class IndexRefiner:
    """Fresh local perception for one root range, using only an existing frame sampler."""

    def __init__(
        self,
        *,
        backend: VisionLanguageBackend,
        frame_sampler: FrameSampler | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.frame_sampler = frame_sampler
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None

    def refine(
        self,
        store: VideoMapStore,
        *,
        parent_segment_id: str,
        requested_start_sec: float,
        requested_end_sec: float,
        resolution: Literal["coarse", "medium", "dense"] = "medium",
        focus: Sequence[str] = (),
    ) -> IndexRefinementPatch:
        parent = store.current.get(parent_segment_id)
        if parent.index_level != "root":
            raise ValueError("Index refinement requires a root parent")
        start = float(requested_start_sec)
        end = float(requested_end_sec)
        max_frames = max(1, int(round(end - start)))
        frames = tuple(self.frame_sampler(store.current.video_path, start, end, max_frames)) if self.frame_sampler else ()
        prompt = (
            "Create a navigation-only local DVC index for this root video range. "
            "Return JSON with children. Each child needs start_sec, end_sec, summary, optional beats, "
            "entity_hints, modality_hints, and limitations. Do not answer the question; this is not evidence.\n"
            f"Parent: {parent.segment_id} [{parent.start_sec:.3f}, {parent.end_sec:.3f}]\n"
            f"Requested: [{start:.3f}, {end:.3f}], resolution={resolution}\n"
            f"Focus: {', '.join(str(item) for item in focus)}"
        )
        metadata = {
            "tool": "read_segment",
            "mode": "refine",
            "parent_segment_id": parent.segment_id,
            "start_sec": start,
            "end_sec": end,
            "resolution": resolution,
            "frame_cache_policy": "existing_cache_only",
            "frame_count": len(frames),
        }
        artifact_prefix = _refinement_artifact_prefix(
            self.artifact_root,
            parent_segment_id=parent.segment_id,
            start_sec=start,
            end_sec=end,
            resolution=resolution,
        )
        _write_refinement_request_artifact(
            artifact_prefix,
            {
                "task": "refine_segment_index",
                "prompt": prompt,
                "media_type": "video",
                "frames": list(frames),
                "max_new_tokens": 512,
                "metadata": metadata,
            },
        )
        response = self.backend.generate(
            BackendRequest(
                task="refine_segment_index",
                prompt=prompt,
                media_type="video",
                frames=frames,
                max_new_tokens=512,
                metadata=metadata,
            )
        )
        _write_refinement_response_artifact(artifact_prefix, response.text)
        data = _parse_json_mapping(response.text)
        try:
            children = _refinement_children_from_payload(
                payload=data,
                parent=parent,
                requested_start_sec=start,
                requested_end_sec=end,
                resolution=resolution,
            )
        except ValueError as exc:
            _write_refinement_validation_artifact(
                artifact_prefix,
                {
                    "valid": False,
                    "error": str(exc),
                    "parsed_keys": sorted(str(key) for key in data.keys()),
                    "child_count": 0,
                },
            )
            raise
        _write_refinement_validation_artifact(
            artifact_prefix,
            {
                "valid": True,
                "child_count": len(children),
                "parsed_keys": sorted(str(key) for key in data.keys()),
            },
        )
        return store.apply_refinement(
            parent_segment_id=parent.segment_id,
            requested_start_sec=start,
            requested_end_sec=end,
            resolution=resolution,
            children=children,
            provenance={
                "backend_task": "refine_segment_index",
                "fresh_local_perception": True,
                "frame_cache_only": True,
                "frame_count": len(frames),
            },
        )


def _refinement_children_from_payload(
    *,
    payload: Mapping[str, Any],
    parent: VideoMapSegment,
    requested_start_sec: float,
    requested_end_sec: float,
    resolution: str,
) -> tuple[VideoMapSegment, ...]:
    raw_value = payload.get("children")
    raw_children = raw_value if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)) else ()
    children = []
    for index, item in enumerate(raw_children, start=1):
        if not isinstance(item, Mapping):
            continue
        start = float(item.get("start_sec", requested_start_sec))
        end = float(item.get("end_sec", requested_end_sec))
        summary = _clean_text(item.get("summary") or item.get("caption") or "")
        if not summary:
            continue
        child_id = _refined_child_id(parent.segment_id, str(resolution), start, end)
        children.append(
            VideoMapSegment(
                segment_id=child_id,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                source="dvc_refined_v1",
                low_fps_caption=summary,
                asr_text=str(item.get("asr_text") or ""),
                ocr_text=str(item.get("ocr_text") or ""),
                entities=_unique_texts(
                    [*_clean_list(item.get("entities")), *_clean_list(item.get("entity_hints"))]
                ),
                asr_sentences=_rows_in_range(parent.asr_sentences, start, end),
                ocr_frames=_rows_in_range(parent.ocr_frames, start, end),
                limitations=tuple(_clean_list(item.get("limitations"))),
                index_level="refined",
                parent_segment_id=parent.segment_id,
                root_segment_id=parent.root_segment_id or parent.segment_id,
                timeline_beats=_beats_from_payload(item.get("beats") or (), segment_id=child_id),
                refinement_state="refined",
                index_provenance={
                    "schema_version": "dvc_refined_v1",
                    "fresh_local_perception": True,
                    "resolution": str(resolution),
                },
            )
        )
    if children:
        return tuple(children)
    raise ValueError(
        "refinement_output_invalid: backend returned no valid children[*].summary; no index patch was applied"
    )


def _refinement_artifact_prefix(
    artifact_root: Path | None,
    *,
    parent_segment_id: str,
    start_sec: float,
    end_sec: float,
    resolution: str,
) -> Path | None:
    if artifact_root is None:
        return None
    safe_parent = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(parent_segment_id)).strip("_") or "segment"
    return artifact_root / f"{safe_parent}_{_sec_id(start_sec)}_{_sec_id(end_sec)}_{resolution}"


def _write_refinement_request_artifact(prefix: Path | None, payload: Mapping[str, Any]) -> None:
    if prefix is None:
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "_request.json").write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _write_refinement_response_artifact(prefix: Path | None, text: str) -> None:
    if prefix is None:
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "_response.txt").write_text(str(text or ""), encoding="utf-8")


def _write_refinement_validation_artifact(prefix: Path | None, payload: Mapping[str, Any]) -> None:
    if prefix is None:
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "_validation.json").write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _beats_from_payload(value: Any, *, segment_id: str) -> tuple[TimelineBeat, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    beats = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        beats.append(
            TimelineBeat(
                beat_id=str(item.get("beat_id") or f"{segment_id}_b{index:02d}"),
                start_sec=float(item.get("start_sec", 0.0)),
                end_sec=float(item.get("end_sec", 0.0)),
                summary=_clean_text(item.get("summary") or ""),
                entity_hints=tuple(_clean_list(item.get("entity_hints"))),
                modality_hints=tuple(
                    modality
                    for modality in _clean_list(item.get("modality_hints"))
                    if modality in {"visual", "asr", "ocr", "temporal"}
                ),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                frame_refs=tuple(_clean_list(item.get("frame_refs"))),
                limitations=tuple(_clean_list(item.get("limitations"))),
            )
        )
    return tuple(beats)


def _refinement_patch_id(parent_segment_id: str, start_sec: float, end_sec: float, resolution: str) -> str:
    return f"{parent_segment_id}_patch_{resolution}_{_sec_id(start_sec)}_{_sec_id(end_sec)}"


def _refined_child_id(parent_segment_id: str, resolution: str, start_sec: float, end_sec: float) -> str:
    return f"{parent_segment_id}_r_{resolution}_{_sec_id(start_sec)}_{_sec_id(end_sec)}"


def _sec_id(value: float) -> str:
    return f"{int(round(float(value) * 1000)):07d}"


def _parse_json_mapping(text: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    return _unique_texts([_clean_text(item) for item in items])


def _rows_in_range(rows: Sequence[Mapping[str, object]], start_sec: float, end_sec: float) -> tuple[Mapping[str, object], ...]:
    filtered = []
    for row in rows:
        row_start = float(row.get("start_sec", start_sec) or start_sec)
        row_end = float(row.get("end_sec", row_start) or row_start)
        if row_start < end_sec and row_end > start_sec:
            filtered.append(dict(row))
    return tuple(filtered)
