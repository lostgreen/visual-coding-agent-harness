from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Mapping, Sequence

from vcah.caption_schema import CaptionHitV1, stable_digest


CAPTION_OCCURRENCE_SCHEMA_VERSION = "CaptionOccurrenceSetV1"
DEFAULT_OCCURRENCE_GAP_SEC = 120.0


@dataclass(frozen=True)
class CaptionOccurrenceCandidateV1:
    occurrence_id: str
    rank: int
    virtual_start_sec: float
    virtual_end_sec: float
    source_video_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    passage_ids: tuple[str, ...]
    hit_ranks: tuple[int, ...]
    query_matches: tuple[Mapping[str, Any], ...]
    max_score: float
    hit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "rank": self.rank,
            "time_range": [self.virtual_start_sec, self.virtual_end_sec],
            "source_video_ids": list(self.source_video_ids),
            "segment_ids": list(self.segment_ids),
            "passage_ids": list(self.passage_ids),
            "hit_ranks": list(self.hit_ranks),
            "query_matches": [dict(item) for item in self.query_matches],
            "max_score": self.max_score,
            "hit_count": self.hit_count,
            "evidence_role": "candidate",
            "status": "uninspected",
        }


@dataclass(frozen=True)
class _OccurrenceHit:
    passage_id: str
    caption_id: str
    rank: int
    start_sec: float
    end_sec: float
    score: float
    source_video_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    query_matches: tuple[Mapping[str, Any], ...]

    @property
    def scope_key(self) -> tuple[str, ...]:
        if self.source_video_ids:
            return ("source", *self.source_video_ids)
        if self.segment_ids:
            return ("segment", *self.segment_ids)
        if self.caption_id:
            return ("caption", self.caption_id)
        return ("unscoped",)


def build_caption_occurrence_set(
    hits: Sequence[CaptionHitV1 | Mapping[str, Any]],
    *,
    gap_sec: float = DEFAULT_OCCURRENCE_GAP_SEC,
) -> dict[str, Any]:
    """Cluster locator hits without deciding which occurrence is semantically correct."""

    candidates = cluster_caption_occurrences(hits, gap_sec=gap_sec)
    candidate_count = len(candidates)
    return {
        "schema_version": CAPTION_OCCURRENCE_SCHEMA_VERSION,
        "clustering_gap_sec": max(0.0, float(gap_sec)),
        "candidate_count": candidate_count,
        "occurrence_ambiguous": candidate_count > 1,
        "status": (
            "empty"
            if candidate_count == 0
            else "single_candidate"
            if candidate_count == 1
            else "competing_candidates"
        ),
        "selected_occurrence_id": None,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def cluster_caption_occurrences(
    hits: Sequence[CaptionHitV1 | Mapping[str, Any]],
    *,
    gap_sec: float = DEFAULT_OCCURRENCE_GAP_SEC,
) -> tuple[CaptionOccurrenceCandidateV1, ...]:
    normalized = tuple(_occurrence_hit(hit) for hit in hits)
    if not normalized:
        return ()

    tolerance = max(0.0, float(gap_sec))
    by_scope: dict[tuple[str, ...], list[_OccurrenceHit]] = {}
    for hit in normalized:
        by_scope.setdefault(hit.scope_key, []).append(hit)

    candidates: list[CaptionOccurrenceCandidateV1] = []
    for scoped_hits in by_scope.values():
        ordered = sorted(
            scoped_hits,
            key=lambda hit: (hit.start_sec, hit.end_sec, hit.rank, hit.passage_id),
        )
        cluster: list[_OccurrenceHit] = []
        cluster_end = 0.0
        for hit in ordered:
            if cluster and hit.start_sec > cluster_end + tolerance:
                candidates.append(_candidate(cluster))
                cluster = []
            cluster.append(hit)
            cluster_end = max(cluster_end, hit.end_sec) if len(cluster) > 1 else hit.end_sec
        if cluster:
            candidates.append(_candidate(cluster))

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            min(candidate.hit_ranks, default=candidate.rank),
            candidate.virtual_start_sec,
            candidate.occurrence_id,
        ),
    )
    return tuple(
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(ordered_candidates, start=1)
    )


def _candidate(hits: Sequence[_OccurrenceHit]) -> CaptionOccurrenceCandidateV1:
    ordered = tuple(sorted(hits, key=lambda hit: (hit.start_sec, hit.rank, hit.passage_id)))
    start_sec = round(min(hit.start_sec for hit in ordered), 3)
    end_sec = round(max(hit.end_sec for hit in ordered), 3)
    source_video_ids = tuple(sorted({value for hit in ordered for value in hit.source_video_ids}))
    segment_ids = tuple(sorted({value for hit in ordered for value in hit.segment_ids}))
    passage_ids = tuple(hit.passage_id for hit in ordered)
    identity = stable_digest(
        {
            "source_video_ids": source_video_ids,
            "segment_ids": segment_ids,
            "passage_ids": sorted(passage_ids),
            "time_range": [start_sec, end_sec],
        }
    )
    return CaptionOccurrenceCandidateV1(
        occurrence_id=f"occ_{identity[:20]}",
        rank=min(hit.rank for hit in ordered),
        virtual_start_sec=start_sec,
        virtual_end_sec=end_sec,
        source_video_ids=source_video_ids,
        segment_ids=segment_ids,
        passage_ids=passage_ids,
        hit_ranks=tuple(hit.rank for hit in ordered),
        query_matches=_deduplicated_query_matches(ordered),
        max_score=max(hit.score for hit in ordered),
        hit_count=len(ordered),
    )


def _occurrence_hit(hit: CaptionHitV1 | Mapping[str, Any]) -> _OccurrenceHit:
    if isinstance(hit, CaptionHitV1):
        metadata = dict(hit.metadata)
        passage_id = hit.passage_id
        caption_id = hit.caption_id
        rank = hit.rank
        start_sec = hit.virtual_start_sec
        end_sec = hit.virtual_end_sec
        score = hit.fused_score
    else:
        metadata = dict(hit.get("metadata") or {})
        raw_range = tuple(hit.get("range", ()) or ())
        passage_id = str(hit.get("passage_id", "") or "")
        caption_id = str(hit.get("caption_id", "") or "")
        rank = max(1, int(hit.get("rank", 1) or 1))
        start_sec = float(
            raw_range[0]
            if len(raw_range) == 2
            else hit.get("virtual_start_sec", 0.0) or 0.0
        )
        end_sec = float(
            raw_range[1]
            if len(raw_range) == 2
            else hit.get("virtual_end_sec", start_sec) or start_sec
        )
        score = float(hit.get("fused_score", hit.get("score", 0.0)) or 0.0)
    return _OccurrenceHit(
        passage_id=str(passage_id),
        caption_id=str(caption_id),
        rank=max(1, int(rank)),
        start_sec=round(min(float(start_sec), float(end_sec)), 3),
        end_sec=round(max(float(start_sec), float(end_sec)), 3),
        score=float(score),
        source_video_ids=_string_tuple(metadata.get("source_video_ids", ())),
        segment_ids=_string_tuple(metadata.get("source_segments", metadata.get("segment_ids", ()))),
        query_matches=_query_matches(metadata),
    )


def _query_matches(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_matches = metadata.get("query_matches") or metadata.get("matched_queries", ())
    if isinstance(raw_matches, (str, Mapping)):
        raw_matches = (raw_matches,)
    matches: list[Mapping[str, Any]] = []
    for value in tuple(raw_matches or ()):
        if isinstance(value, Mapping):
            query = str(value.get("query", "") or "").strip()
            if query:
                matches.append({"query": query, "rank": max(1, int(value.get("rank", 1) or 1))})
        else:
            query = str(value or "").strip()
            if query:
                matches.append({"query": query})
    return tuple(matches)


def _deduplicated_query_matches(
    hits: Sequence[_OccurrenceHit],
) -> tuple[Mapping[str, Any], ...]:
    matches: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        for match in hit.query_matches:
            key = json.dumps(dict(match), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                matches.append(dict(match))
    return tuple(matches)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Sequence):
        raw = tuple(value)
    else:
        raw = ()
    return tuple(sorted({str(item).strip() for item in raw if str(item).strip()}))
