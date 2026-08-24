from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from vcah.anchor_evidence import observed_evidence_channels
from vcah.caption_context import (
    CAPTION_CONTEXT_CONTRACT,
    expand_query_conditioned_context,
    ordered_source_timeline,
)
from vcah.caption_evidence_bundle import build_caption_evidence_bundle_set
from vcah.caption_lexical_index import normalize_caption_query
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1
from vcah.occurrence_ocr import enrich_caption_passages_with_ocr


RELATION_EVIDENCE_SEARCH_CONTRACT = "WP16-4-relation-guided-evidence-search-v1"
RELATION_DIRECTIONS = frozenset({"before", "after"})
TARGET_TEXT_SOURCES = frozenset({"caption", "ocr"})


def select_oracle_anchor_passage(
    passages: Sequence[CaptionPassageV1],
    anchor_intervals: Sequence[Sequence[float]],
) -> CaptionPassageV1 | None:
    """Select the passage with greatest overlap with a manual anchor label."""

    intervals = _intervals(anchor_intervals)
    candidates: list[tuple[float, float, CaptionPassageV1]] = []
    for passage in passages:
        for start, end in intervals:
            overlap = max(
                0.0,
                min(passage.virtual_end_sec, end)
                - max(passage.virtual_start_sec, start),
            )
            if overlap <= 0.0:
                continue
            passage_midpoint = (
                passage.virtual_start_sec + passage.virtual_end_sec
            ) / 2.0
            interval_midpoint = (start + end) / 2.0
            candidates.append(
                (overlap, abs(passage_midpoint - interval_midpoint), passage)
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            -row[0],
            row[1],
            row[2].virtual_end_sec - row[2].virtual_start_sec,
            row[2].virtual_start_sec,
            row[2].passage_id,
        ),
    )[2]


def build_oracle_fixed_window(
    passages: Sequence[CaptionPassageV1],
    *,
    anchor_intervals: Sequence[Sequence[float]],
    direction: str,
    evidence_channels: Sequence[str],
    ocr_rows: Sequence[Mapping[str, Any]] = (),
    distance: int = 20,
    index_digest: str,
    config_digest: str,
    source_video_id_by_segment: Mapping[str, str] | None = None,
    max_gap_sec: float = 600.0,
) -> dict[str, Any]:
    """Build a fixed directional window from an independently labeled anchor."""

    selected_direction = _direction(direction)
    enriched = enrich_caption_passages_with_ocr(passages, ocr_rows)
    anchor = select_oracle_anchor_passage(enriched, anchor_intervals)
    if anchor is None:
        return _empty_result(
            selected_direction,
            evidence_channels,
            status="anchor_not_found",
        )
    anchor_hit = _passage_hit(
        anchor,
        rank=1,
        index_digest=index_digest,
        config_digest=config_digest,
        metadata={"oracle_anchor": True},
    )
    hits = tuple(
        expand_query_conditioned_context(
            enriched,
            (anchor_hit,),
            distance=max(0, int(distance)),
            time_range=None,
            segment_ids=(),
            index_digest=index_digest,
            config_digest=config_digest,
            source_video_id_by_segment=source_video_id_by_segment,
            max_gap_sec=max(0.0, float(max_gap_sec)),
            direction=selected_direction,
        )
    )
    annotated = tuple(
        _annotate_hit(
            hit,
            anchor_passage_id=anchor.passage_id,
            direction=selected_direction,
            evidence_channels=evidence_channels,
            visited_count=max(0, len(hits) - 1),
            stop_match=(),
            variant="fixed_window",
        )
        for hit in hits
    )
    return {
        "contract": RELATION_EVIDENCE_SEARCH_CONTRACT,
        "variant": "fixed_window",
        "status": "complete",
        "direction": selected_direction,
        "requested_evidence_channels": list(evidence_channels),
        "anchor_hit": annotated[0],
        "stop_hit": None,
        "hits": annotated,
        "visited_passage_count": max(0, len(annotated) - 1),
        "stop_success": False,
        "stop_reason": "fixed_distance_complete",
        "matched_target_terms": [],
        "evidence_bundle_set": build_caption_evidence_bundle_set(annotated),
    }


def relation_guided_evidence_search(
    passages: Sequence[CaptionPassageV1],
    *,
    anchor_intervals: Sequence[Sequence[float]],
    direction: str,
    target_event_term_groups: Sequence[Sequence[str]],
    evidence_channels: Sequence[str],
    target_text_sources: Sequence[str] = ("caption",),
    ocr_rows: Sequence[Mapping[str, Any]] = (),
    max_passages: int = 80,
    max_elapsed_sec: float = 2400.0,
    max_gap_sec: float = 600.0,
    index_digest: str,
    config_digest: str,
    source_video_id_by_segment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Walk from a correct anchor and stop at the first target event."""

    selected_direction = _direction(direction)
    groups = _term_groups(target_event_term_groups)
    sources = _target_sources(target_text_sources)
    original = tuple(passages)
    enriched = enrich_caption_passages_with_ocr(original, ocr_rows)
    enriched_by_id = {passage.passage_id: passage for passage in enriched}
    anchor = select_oracle_anchor_passage(original, anchor_intervals)
    if anchor is None:
        return _empty_result(
            selected_direction,
            evidence_channels,
            status="anchor_not_found",
        )

    timeline = ordered_source_timeline(
        original,
        anchor,
        source_video_id_by_segment,
    )
    positions = {passage.passage_id: index for index, passage in enumerate(timeline)}
    anchor_index = positions.get(anchor.passage_id)
    if anchor_index is None:
        return _empty_result(
            selected_direction,
            evidence_channels,
            status="anchor_timeline_missing",
        )

    step_direction = 1 if selected_direction == "after" else -1
    visited: list[dict[str, Any]] = []
    stop: CaptionPassageV1 | None = None
    stop_match: tuple[tuple[str, ...], ...] = ()
    stop_reason = "timeline_exhausted"
    previous = anchor
    limit = max(1, int(max_passages))
    elapsed_limit = max(0.0, float(max_elapsed_sec))
    edge_limit = max(0.0, float(max_gap_sec))

    for step in range(1, limit + 1):
        index = anchor_index + step_direction * step
        if index < 0 or index >= len(timeline):
            break
        passage = timeline[index]
        edge_gap = _interval_gap(previous, passage)
        if edge_gap > edge_limit:
            stop_reason = "edge_gap_limit"
            break
        previous = passage
        elapsed = _elapsed_from_anchor(anchor, passage, selected_direction)
        if elapsed > elapsed_limit:
            stop_reason = "elapsed_limit"
            break
        enriched_passage = enriched_by_id.get(passage.passage_id, passage)
        matched = _matched_target_terms(
            passage,
            enriched_passage,
            groups,
            sources,
        )
        visited.append(
            {
                "passage_id": passage.passage_id,
                "time_range": [
                    passage.virtual_start_sec,
                    passage.virtual_end_sec,
                ],
                "offset": step_direction * step,
                "edge_gap_sec": round(edge_gap, 3),
                "elapsed_sec": round(elapsed, 3),
                "matched_target_terms": [list(value) for value in matched],
            }
        )
        if matched:
            stop = enriched_passage
            stop_match = matched
            stop_reason = "target_event_found"
            break
    else:
        stop_reason = "passage_limit"

    enriched_anchor = enriched_by_id.get(anchor.passage_id, anchor)
    anchor_hit = _passage_hit(
        enriched_anchor,
        rank=1,
        index_digest=index_digest,
        config_digest=config_digest,
        metadata={"oracle_anchor": True},
    )
    hits: tuple[CaptionHitV1, ...]
    stop_hit: CaptionHitV1 | None = None
    if stop is None:
        hits = (anchor_hit,)
    else:
        stop_hit = _passage_hit(
            stop,
            rank=2,
            index_digest=index_digest,
            config_digest=config_digest,
            metadata={"candidate_only": True},
        )
        hits = (anchor_hit, stop_hit)

    annotated = tuple(
        _annotate_hit(
            hit,
            anchor_passage_id=anchor.passage_id,
            direction=selected_direction,
            evidence_channels=evidence_channels,
            visited_count=len(visited),
            stop_match=stop_match if index > 0 else (),
            variant="bounded_search",
        )
        for index, hit in enumerate(hits)
    )
    return {
        "contract": RELATION_EVIDENCE_SEARCH_CONTRACT,
        "variant": "bounded_search",
        "status": "complete",
        "direction": selected_direction,
        "requested_evidence_channels": list(evidence_channels),
        "target_text_sources": list(sources),
        "anchor_hit": annotated[0],
        "stop_hit": annotated[1] if len(annotated) > 1 else None,
        "hits": annotated,
        "visited_passage_count": len(visited),
        "visited": visited,
        "stop_success": stop is not None,
        "stop_reason": stop_reason,
        "matched_target_terms": [list(value) for value in stop_match],
        "evidence_bundle_set": build_caption_evidence_bundle_set(annotated),
    }


def _annotate_hit(
    hit: CaptionHitV1,
    *,
    anchor_passage_id: str,
    direction: str,
    evidence_channels: Sequence[str],
    visited_count: int,
    stop_match: Sequence[Sequence[str]],
    variant: str,
) -> CaptionHitV1:
    metadata = dict(hit.metadata)
    metadata.update(
        {
            "relation_evidence_search_contract": (RELATION_EVIDENCE_SEARCH_CONTRACT),
            "relation_evidence_search_variant": variant,
            "relation_anchor_passage_id": anchor_passage_id,
            "relation_direction": direction,
            "relation_passages_visited": int(visited_count),
            "target_event_match": [list(value) for value in stop_match],
            "evidence_channels_requested": list(evidence_channels),
            "evidence_channels_observed": list(observed_evidence_channels(metadata)),
        }
    )
    if hit.passage_id != anchor_passage_id:
        metadata.update(
            {
                "context_expansion_contract": (CAPTION_CONTEXT_CONTRACT),
                "context_direction": direction,
                "context_seed_passage_ids": [anchor_passage_id],
                "neighbor_of": anchor_passage_id,
            }
        )
    return replace(hit, metadata=metadata)


def _passage_hit(
    passage: CaptionPassageV1,
    *,
    rank: int,
    index_digest: str,
    config_digest: str,
    metadata: Mapping[str, Any],
) -> CaptionHitV1:
    return CaptionHitV1(
        passage_id=passage.passage_id,
        caption_id=passage.caption_id,
        rank=rank,
        lexical_score=None,
        dense_score=None,
        fused_score=1.0 if rank == 1 else 0.0,
        virtual_start_sec=passage.virtual_start_sec,
        virtual_end_sec=passage.virtual_end_sec,
        wall_clock_begin=_optional_text(passage.metadata.get("wall_clock_begin")),
        wall_clock_end=_optional_text(passage.metadata.get("wall_clock_end")),
        text=passage.text,
        interval_precision=str(passage.metadata.get("interval_precision", "chunk")),
        source_pointer=f"caption://{config_digest}/{passage.passage_id}",
        metadata={
            **dict(passage.metadata),
            **dict(metadata),
            "index_digest": index_digest,
        },
    )


def _matched_target_terms(
    original: CaptionPassageV1,
    enriched: CaptionPassageV1,
    groups: Sequence[Sequence[str]],
    sources: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    pieces: list[str] = []
    if "caption" in sources:
        pieces.append(original.text)
    if "ocr" in sources:
        pieces.extend(
            str(value)
            for value in tuple(enriched.metadata.get("ocr_text", ()) or ())
            if str(value)
        )
    haystack = normalize_caption_query(" ".join(pieces))
    matched: list[tuple[str, ...]] = []
    for group in groups:
        values = tuple(
            term for term in group if normalize_caption_query(term) in haystack
        )
        if not values:
            return ()
        matched.append(values)
    return tuple(matched)


def _empty_result(
    direction: str,
    evidence_channels: Sequence[str],
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "contract": RELATION_EVIDENCE_SEARCH_CONTRACT,
        "variant": "bounded_search",
        "status": status,
        "direction": direction,
        "requested_evidence_channels": list(evidence_channels),
        "anchor_hit": None,
        "stop_hit": None,
        "hits": (),
        "visited_passage_count": 0,
        "visited": [],
        "stop_success": False,
        "stop_reason": status,
        "matched_target_terms": [],
        "evidence_bundle_set": build_caption_evidence_bundle_set(()),
    }


def _direction(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in RELATION_DIRECTIONS:
        raise ValueError(f"unsupported relation direction: {value}")
    return normalized


def _target_sources(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip().casefold() for value in values))
    if not normalized or any(value not in TARGET_TEXT_SOURCES for value in normalized):
        raise ValueError("target text sources must be caption and/or ocr")
    return normalized


def _term_groups(
    values: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    groups = tuple(
        tuple(dict.fromkeys(str(term).strip() for term in group if str(term).strip()))
        for group in values
    )
    if not groups or any(not group for group in groups):
        raise ValueError("target event term groups must be non-empty")
    return groups


def _intervals(
    values: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for value in values:
        if len(value) != 2:
            raise ValueError("interval must contain exactly two endpoints")
        start, end = sorted((float(value[0]), float(value[1])))
        if end <= start:
            raise ValueError("interval end must be greater than start")
        intervals.append((start, end))
    if not intervals:
        raise ValueError("at least one anchor interval is required")
    return tuple(intervals)


def _interval_gap(
    left: CaptionPassageV1,
    right: CaptionPassageV1,
) -> float:
    return max(
        0.0,
        max(left.virtual_start_sec, right.virtual_start_sec)
        - min(left.virtual_end_sec, right.virtual_end_sec),
    )


def _elapsed_from_anchor(
    anchor: CaptionPassageV1,
    passage: CaptionPassageV1,
    direction: str,
) -> float:
    if direction == "after":
        return max(0.0, passage.virtual_start_sec - anchor.virtual_end_sec)
    return max(0.0, anchor.virtual_start_sec - passage.virtual_end_sec)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
