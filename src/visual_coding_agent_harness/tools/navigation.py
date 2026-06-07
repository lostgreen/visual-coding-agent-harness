"""Navigation tools over a structured VideoMap workspace."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from ..registry import ToolRegistry, tool
from ..video_map import VideoMap, VideoMapSegment, VideoMapStore
from ..workspace import EvidenceWorkspace, MapUpdateProposal


def build_video_navigation_registry(
    video_map: VideoMap | VideoMapStore,
    *,
    workspace: EvidenceWorkspace | None = None,
) -> ToolRegistry:
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

    @tool(name="target_coverage", description="Build a target-to-segment coverage matrix from indexed caption/ASR/OCR/entity fields.")
    def target_coverage(targets: Sequence[str], top_k: int = 3, modalities: Sequence[str] = ()) -> Mapping[str, object]:
        current = video_map_store.current
        rows = []
        for index, target in enumerate([str(item).strip() for item in targets if str(item).strip()], start=1):
            results = current.search(query=target, top_k=top_k, modalities=modalities)
            candidates = [_coverage_candidate(result) for result in results]
            status = "candidate" if candidates else "missing"
            rows.append(
                {
                    "target_id": f"T{index}",
                    "target": target,
                    "status": status,
                    "candidates": candidates,
                    "missing_confirmation": not bool(candidates),
                }
            )
        summary = "; ".join(
            f"{row['target_id']} {row['target']}: "
            + (
                ", ".join(candidate["segment_id"] for candidate in row["candidates"])
                if row["candidates"]
                else "missing"
            )
            for row in rows
        )
        return {
            "claim": f"Target coverage matrix: {summary or 'no targets supplied'}.",
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "coverage": rows,
            "limitations": "Index coverage only; use read_segment_detail and visual tools to confirm facts before final answers.",
        }

    @tool(name="ground_question", description="Ground a question or event into candidate video windows without answering.")
    def ground_question(query: str, top_k: int = 5, modalities: Sequence[str] = ()) -> Mapping[str, object]:
        current = video_map_store.current
        normalized_query = _normalize_grounding_query(query)
        results = current.search(query=normalized_query, top_k=top_k, modalities=modalities) if normalized_query else []
        if not results:
            results = current.anchor_segments(max_segments=top_k)
        candidates = [_grounding_candidate(result) for result in results]
        ids = ", ".join(str(candidate["segment_id"]) for candidate in candidates) if candidates else "none"
        return {
            "claim": f"Grounding query '{query}' returned candidate windows: {ids}.",
            "confidence": max([float(candidate["confidence"]) for candidate in candidates] or [0.0]),
            "input_artifacts": [current.video_path],
            "regions": candidates,
            "candidates": candidates,
            "normalized_query": normalized_query,
            "recommended_next_tools": [
                {
                    "tool": "vision_read",
                    "args": {
                        "segment_id": candidate["segment_id"],
                        "start_sec": candidate["start_sec"],
                        "end_sec": candidate["end_sec"],
                        "ask_for": query,
                    },
                    "reason": "Read typed visual facts from this grounded candidate window.",
                }
                for candidate in candidates
            ],
            "limitations": "Grounding only localizes candidates from indexes; it does not choose MCQ options or produce final answers.",
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

    @tool(name="read_segment_detail", description="Read full indexed details for one segment, including target hits.")
    def read_segment_detail(segment_id: str, targets: Sequence[str] = ()) -> Mapping[str, object]:
        current = video_map_store.current
        segment = current.get(segment_id)
        target_hits = [_target_hit_for_segment(segment=segment, target=str(target)) for target in targets if str(target).strip()]
        return {
            "claim": _segment_detail_claim(segment, target_hits=target_hits),
            "confidence": 1.0,
            "input_artifacts": [current.video_path],
            "segment_id": segment.segment_id,
            "start_sec": float(segment.start_sec),
            "end_sec": float(segment.end_sec),
            "visual_caption": segment.low_fps_caption,
            "asr_summary": segment.asr_text,
            "raw_asr_excerpt": segment.asr_text,
            "ocr_text": segment.ocr_text,
            "entities": list(segment.entities),
            "keyframe_paths": list(segment.keyframe_paths),
            "target_hits": target_hits,
            "regions": [segment.to_dict()],
            "limitations": "Indexed segment detail only; call vision_read or caption_segment for fresh visual evidence.",
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

    @tool(name="commit_map_proposals", description="Apply pending map update proposals to the mutable VideoMap.")
    def commit_map_proposals(limit: int = 8, min_confidence: float = 0.0) -> Mapping[str, object]:
        if workspace is None:
            raise ValueError("commit_map_proposals requires an EvidenceWorkspace")
        pending = workspace.load_pending_proposals()
        applied = []
        skipped = []
        for proposal in pending[: max(0, int(limit))]:
            if proposal.confidence < float(min_confidence):
                skipped.append({"proposal_id": proposal.proposal_id, "reason": "below_min_confidence"})
                continue
            try:
                updated = _apply_map_update_proposal(video_map_store=video_map_store, proposal=proposal)
            except ValueError as exc:
                skipped.append({"proposal_id": proposal.proposal_id, "reason": str(exc)})
                continue
            committed = workspace.mark_proposal_committed(proposal.proposal_id)
            applied.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "segment_id": updated.segment_id,
                    "update_type": proposal.update_type,
                    "committed_at": committed.committed_at if committed else None,
                }
            )
        return {
            "claim": f"Committed {len(applied)} map proposal(s); skipped {len(skipped)}.",
            "confidence": 1.0,
            "input_artifacts": [],
            "regions": [{"applied": applied, "skipped": skipped}],
            "applied": applied,
            "skipped": skipped,
            "limitations": "Applies audited proposal payloads only; no model inference is performed.",
        }

    registry.register(video_ls)
    registry.register(search_segments)
    registry.register(target_coverage)
    registry.register(ground_question)
    registry.register(read_segment)
    registry.register(read_segment_detail)
    registry.register(expand_window)
    registry.register(zoom)
    registry.register(commit_map_proposals)
    return registry


def _apply_map_update_proposal(*, video_map_store: VideoMapStore, proposal: MapUpdateProposal) -> VideoMapSegment:
    payload = dict(proposal.payload)
    allowed = {
        "low_fps_caption": payload.get("low_fps_caption"),
        "asr_text": payload.get("asr_text"),
        "ocr_text": payload.get("ocr_text"),
        "entities": payload.get("entities"),
        "keyframe_paths": payload.get("keyframe_paths"),
        "embedding_refs": payload.get("embedding_refs"),
    }
    if all(value in (None, "", []) for value in allowed.values()):
        raise ValueError("proposal has no supported map update fields")
    return video_map_store.update_segment(
        proposal.target_segment_id,
        low_fps_caption=_optional_text(allowed["low_fps_caption"]),
        asr_text=_optional_text(allowed["asr_text"]),
        ocr_text=_optional_text(allowed["ocr_text"]),
        entities=_optional_text_list(allowed["entities"]),
        keyframe_paths=_optional_text_list(allowed["keyframe_paths"]),
        embedding_refs=_optional_text_list(allowed["embedding_refs"]),
    )


def _optional_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_text_list(value: object) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    return [text] if text else None


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


_GROUNDING_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "with",
    "without",
    "for",
    "from",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "his",
    "her",
    "their",
    "its",
    "how",
    "what",
    "which",
    "where",
    "who",
    "when",
    "why",
    "video",
    "question",
    "questions",
    "option",
    "options",
    "answer",
    "letter",
    "exactly",
    "one",
    "short",
    "reason",
    "outside",
    "knowledge",
    "unless",
    "directly",
    "supported",
    "evidence",
    "according",
    "use",
    "not",
    "multiple",
    "choice",
    "first",
}


def _normalize_grounding_query(query: str) -> str:
    text = re.sub(r"^\s*[A-H][.)]\s*", "", str(query)).strip()
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text):
        lowered = token.lower().strip("'")
        if len(lowered) < 3 or lowered in _GROUNDING_STOPWORDS:
            continue
        tokens.append(token)
    return " ".join(tokens)


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


def _grounding_candidate(result: object) -> Mapping[str, object]:
    segment = getattr(result, "segment")
    matches = getattr(result, "matches", []) or []
    modalities = []
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        modality = str(match.get("modality", "") or match.get("field", "")).strip()
        if modality and modality not in modalities:
            modalities.append(modality)
    matched_fields = [str(field) for field in getattr(result, "matched_fields", []) or []]
    reason = str(getattr(result, "relevance_reason", "") or "").strip()
    if not reason:
        reason = _relevance_reason(matched_fields)
    return {
        "segment_id": segment.segment_id,
        "start_sec": float(segment.start_sec),
        "end_sec": float(segment.end_sec),
        "reason": reason,
        "modality": ", ".join(modalities) or (matched_fields[0] if matched_fields else "timeline_anchor"),
        "confidence": float(getattr(result, "score", 0.0) or 0.0),
        "matched_fields": matched_fields,
        "matches": [dict(match) for match in matches if isinstance(match, Mapping)],
    }


def _coverage_candidate(result: object) -> Mapping[str, object]:
    segment = getattr(result, "segment")
    return {
        "segment_id": segment.segment_id,
        "start_sec": float(segment.start_sec),
        "end_sec": float(segment.end_sec),
        "score": float(getattr(result, "score", 0.0) or 0.0),
        "matched_fields": [str(field) for field in getattr(result, "matched_fields", []) or []],
        "matches": [dict(match) for match in getattr(result, "matches", []) or [] if isinstance(match, Mapping)],
        "summary": segment.compact_text(),
        "relevance_reason": str(getattr(result, "relevance_reason", "") or ""),
    }


def _relevance_reason(matched_fields: Sequence[str]) -> str:
    fields = [str(field) for field in matched_fields if str(field)]
    if not fields:
        return "fallback timeline anchor"
    return "matched indexed field(s): " + ", ".join(fields)


def _segment_claim(segment: VideoMapSegment) -> str:
    parts = [
        f"{segment.segment_id} covers {segment.start_sec:.1f}-{segment.end_sec:.1f}s.",
        f"caption: {segment.low_fps_caption}" if segment.low_fps_caption else "",
        f"ASR: {segment.asr_text}" if segment.asr_text else "",
        f"OCR: {segment.ocr_text}" if segment.ocr_text else "",
        f"entities: {', '.join(segment.entities)}" if segment.entities else "",
    ]
    return " ".join(part for part in parts if part)


def _segment_detail_claim(segment: VideoMapSegment, *, target_hits: Sequence[Mapping[str, object]]) -> str:
    hit_targets = [str(hit.get("target")) for hit in target_hits if bool(hit.get("matched"))]
    parts = [
        f"{segment.segment_id} detail covers {segment.start_sec:.1f}-{segment.end_sec:.1f}s.",
        "visual caption available" if segment.low_fps_caption else "",
        "ASR summary available" if segment.asr_text else "",
        "OCR available" if segment.ocr_text else "",
        f"entities: {', '.join(segment.entities)}" if segment.entities else "",
        f"target hits: {', '.join(hit_targets)}" if hit_targets else "target hits: none",
    ]
    return " ".join(part for part in parts if part)


def _target_hit_for_segment(*, segment: VideoMapSegment, target: str) -> Mapping[str, object]:
    target_text = str(target).strip()
    target_terms = _target_tokens(target_text)
    field_hits = []
    for field_name, field_value in _detail_search_fields(segment).items():
        value_terms = _target_tokens(field_value)
        overlap = target_terms.intersection(value_terms)
        if not overlap:
            continue
        field_hits.append(
            {
                "field": field_name,
                "modality": _detail_field_modality(field_name),
                "matched_terms": sorted(overlap),
                "evidence": _detail_evidence_snippet(field_value),
                "score": round(len(overlap) / max(len(target_terms), 1), 3),
            }
        )
    return {
        "target": target_text,
        "matched": bool(field_hits),
        "fields": [str(hit["field"]) for hit in field_hits],
        "matches": field_hits,
    }


def _target_tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", str(text or ""))}


def _detail_search_fields(segment: VideoMapSegment) -> Mapping[str, str]:
    return {
        "low_fps_caption": segment.low_fps_caption,
        "asr_text": segment.asr_text,
        "ocr_text": segment.ocr_text,
        "entities": " ".join(segment.entities),
    }


def _detail_field_modality(field_name: str) -> str:
    return {
        "low_fps_caption": "caption",
        "asr_text": "asr",
        "ocr_text": "ocr",
        "entities": "entities",
    }.get(field_name, field_name)


def _detail_evidence_snippet(text: str, max_chars: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
