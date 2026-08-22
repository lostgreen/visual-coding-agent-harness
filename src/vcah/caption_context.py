from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from vcah.caption_schema import CaptionHitV1, CaptionPassageV1, passage_in_segments


CAPTION_CONTEXT_CONTRACT = "query_conditioned_caption_context_v1"
DEFAULT_CONTEXT_MAX_GAP_SEC = 180.0


def expand_query_conditioned_context(
    passages: Sequence[CaptionPassageV1],
    seed_hits: Sequence[CaptionHitV1],
    *,
    distance: int,
    time_range: tuple[float, float] | None,
    segment_ids: Sequence[str] = (),
    index_digest: str,
    config_digest: str,
    source_video_id_by_segment: Mapping[str, str] | None = None,
    max_gap_sec: float = DEFAULT_CONTEXT_MAX_GAP_SEC,
) -> list[CaptionHitV1]:
    """Expand retrieved seeds on the same source timeline without merging events."""

    seeds = list(seed_hits)
    radius = max(0, int(distance))
    if radius == 0 or not seeds:
        return seeds

    source_map = {
        str(segment_id): str(source_id)
        for segment_id, source_id in dict(source_video_id_by_segment or {}).items()
        if str(segment_id) and str(source_id)
    }
    eligible = tuple(
        passage
        for passage in passages
        if _in_time_range(passage, time_range)
        and passage_in_segments(passage, segment_ids)
    )
    by_id = {passage.passage_id: passage for passage in eligible}
    max_gap = max(0.0, float(max_gap_sec))
    additions: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        source = by_id.get(seed.passage_id)
        if source is None:
            continue
        timeline = tuple(
            sorted(
                (
                    passage
                    for passage in eligible
                    if _same_source_timeline(source, passage, source_map)
                ),
                key=lambda passage: (
                    passage.virtual_start_sec,
                    passage.virtual_end_sec,
                    passage.caption_id,
                    passage.ordinal,
                    passage.passage_id,
                ),
            )
        )
        positions = {
            passage.passage_id: index for index, passage in enumerate(timeline)
        }
        source_index = positions.get(source.passage_id)
        if source_index is None:
            continue
        for direction in (-1, 1):
            previous = source
            for step in range(1, radius + 1):
                index = source_index + direction * step
                if index < 0 or index >= len(timeline):
                    break
                neighbor = timeline[index]
                edge_gap = _interval_gap(previous, neighbor)
                if edge_gap > max_gap:
                    break
                previous = neighbor
                if neighbor.passage_id == seed.passage_id:
                    continue
                link = {
                    "seed_passage_id": seed.passage_id,
                    "seed_rank": seed.rank,
                    "offset": direction * step,
                    "edge_gap_sec": round(edge_gap, 3),
                }
                row = additions.setdefault(
                    neighbor.passage_id,
                    {
                        "passage": neighbor,
                        "score": 0.0,
                        "links": [],
                    },
                )
                row["score"] = max(
                    float(row["score"]),
                    max(0.0, seed.fused_score * (0.5 ** step)),
                )
                row["links"].append(link)

    seed_ids = {hit.passage_id for hit in seeds}
    ordered_additions = sorted(
        (
            row
            for passage_id, row in additions.items()
            if passage_id not in seed_ids
        ),
        key=lambda row: (
            min(
                (
                    int(link["seed_rank"]),
                    abs(int(link["offset"])),
                    int(link["offset"]) > 0,
                )
                for link in row["links"]
            ),
            row["passage"].virtual_start_sec,
            row["passage"].passage_id,
        ),
    )
    expanded = list(seeds)
    for row in ordered_additions:
        passage = row["passage"]
        links = tuple(
            sorted(
                row["links"],
                key=lambda link: (
                    int(link["seed_rank"]),
                    abs(int(link["offset"])),
                    int(link["offset"]),
                    str(link["seed_passage_id"]),
                ),
            )
        )
        primary = links[0]
        expanded.append(
            CaptionHitV1(
                passage_id=passage.passage_id,
                caption_id=passage.caption_id,
                rank=len(expanded) + 1,
                lexical_score=None,
                dense_score=None,
                fused_score=float(row["score"]),
                virtual_start_sec=passage.virtual_start_sec,
                virtual_end_sec=passage.virtual_end_sec,
                wall_clock_begin=_optional_text(
                    passage.metadata.get("wall_clock_begin")
                ),
                wall_clock_end=_optional_text(
                    passage.metadata.get("wall_clock_end")
                ),
                text=passage.text,
                interval_precision=str(
                    passage.metadata.get("interval_precision", "chunk")
                ),
                source_pointer=f"caption://{config_digest}/{passage.passage_id}",
                metadata={
                    **dict(passage.metadata),
                    "index_digest": index_digest,
                    "candidate_only": True,
                    "context_expansion_contract": CAPTION_CONTEXT_CONTRACT,
                    "context_seed_passage_ids": list(
                        dict.fromkeys(
                            str(link["seed_passage_id"]) for link in links
                        )
                    ),
                    "context_links": [dict(link) for link in links],
                    "neighbor_of": str(primary["seed_passage_id"]),
                    "neighbor_offset": int(primary["offset"]),
                    "cross_caption": passage.caption_id
                    != by_id[str(primary["seed_passage_id"])].caption_id,
                },
            )
        )
    return [
        CaptionHitV1(**{**asdict(hit), "rank": rank})
        for rank, hit in enumerate(expanded, start=1)
    ]


def _same_source_timeline(
    left: CaptionPassageV1,
    right: CaptionPassageV1,
    source_video_id_by_segment: Mapping[str, str],
) -> bool:
    left_segments = _string_set(left.metadata.get("source_segments", ()))
    right_segments = _string_set(right.metadata.get("source_segments", ()))
    left_sources = {
        source_video_id_by_segment[segment_id]
        for segment_id in left_segments
        if segment_id in source_video_id_by_segment
    }
    right_sources = {
        source_video_id_by_segment[segment_id]
        for segment_id in right_segments
        if segment_id in source_video_id_by_segment
    }
    if left_sources and right_sources:
        return bool(left_sources & right_sources)
    if left_segments and right_segments:
        return bool(left_segments & right_segments)
    return left.caption_id == right.caption_id


def _in_time_range(
    passage: CaptionPassageV1,
    time_range: tuple[float, float] | None,
) -> bool:
    if time_range is None:
        return True
    start, end = sorted((float(time_range[0]), float(time_range[1])))
    return passage.virtual_end_sec > start and passage.virtual_start_sec < end


def _interval_gap(left: CaptionPassageV1, right: CaptionPassageV1) -> float:
    return max(
        0.0,
        max(left.virtual_start_sec, right.virtual_start_sec)
        - min(left.virtual_end_sec, right.virtual_end_sec),
    )


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        value = (value,)
    return {str(item) for item in tuple(value or ()) if str(item)}


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
