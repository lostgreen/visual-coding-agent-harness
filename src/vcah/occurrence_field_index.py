from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from vcah.caption_schema import CaptionPassageV1


OCCURRENCE_FIELDS = ("entity", "event", "state")
ORACLE_FIELD_INDEX_CONTRACT = "WP16-5-oracle-occurrence-field-index-v1"


def normalize_occurrence_fields(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, tuple[str, ...]]]:
    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    for field_name in OCCURRENCE_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, Mapping):
            raise ValueError(f"missing occurrence field: {field_name}")
        query_terms = _terms(value.get("query_terms"))
        document_terms = _terms(value.get("document_terms"))
        if not query_terms or not document_terms:
            raise ValueError(f"empty occurrence field: {field_name}")
        normalized[field_name] = {
            "query_terms": query_terms,
            "document_terms": document_terms,
        }
    return normalized


def select_oracle_occurrence_passage(
    passages: Sequence[CaptionPassageV1],
    anchor_intervals: Sequence[Sequence[float]],
) -> CaptionPassageV1 | None:
    intervals = _intervals(anchor_intervals)
    candidates: list[tuple[float, float, float, str, CaptionPassageV1]] = []
    for passage in passages:
        midpoint = (passage.virtual_start_sec + passage.virtual_end_sec) / 2.0
        for start, end in intervals:
            overlap = max(
                0.0,
                min(passage.virtual_end_sec, end)
                - max(passage.virtual_start_sec, start),
            )
            if overlap <= 0.0:
                continue
            interval_midpoint = (start + end) / 2.0
            candidates.append(
                (
                    -overlap,
                    abs(midpoint - interval_midpoint),
                    passage.virtual_start_sec,
                    passage.passage_id,
                    passage,
                )
            )
    return min(candidates)[-1] if candidates else None


def augment_oracle_occurrence_passage(
    passages: Sequence[CaptionPassageV1],
    *,
    oracle_passage_id: str,
    fields: Mapping[str, Mapping[str, Sequence[str]]],
    selected_fields: Sequence[str],
) -> tuple[CaptionPassageV1, ...]:
    selected = tuple(dict.fromkeys(str(value) for value in selected_fields))
    if not selected or not set(selected) <= set(OCCURRENCE_FIELDS):
        raise ValueError("selected occurrence fields are invalid")
    normalized = normalize_occurrence_fields(fields)
    matched = False
    augmented: list[CaptionPassageV1] = []
    for passage in passages:
        if passage.passage_id != str(oracle_passage_id):
            augmented.append(passage)
            continue
        matched = True
        blocks = [passage.text]
        for field_name in selected:
            terms = normalized[field_name]["document_terms"]
            blocks.append(f"{field_name}: {' ; '.join(terms)}")
        augmented.append(
            replace(
                passage,
                text="\n".join(value for value in blocks if value),
                metadata={
                    **dict(passage.metadata),
                    "oracle_field_contract": ORACLE_FIELD_INDEX_CONTRACT,
                    "oracle_fields": list(selected),
                },
            )
        )
    if not matched:
        raise ValueError("oracle occurrence passage is absent from the index")
    return tuple(augmented)


def oracle_field_passage(
    passage: CaptionPassageV1,
    *,
    field_name: str,
    fields: Mapping[str, Mapping[str, Sequence[str]]],
) -> CaptionPassageV1:
    normalized = normalize_occurrence_fields(fields)
    if field_name not in OCCURRENCE_FIELDS:
        raise ValueError(f"invalid occurrence field: {field_name}")
    terms = normalized[field_name]["document_terms"]
    return replace(
        passage,
        text=f"{field_name}: {' ; '.join(terms)}",
        metadata={
            **dict(passage.metadata),
            "oracle_field_contract": ORACLE_FIELD_INDEX_CONTRACT,
            "oracle_fields": [field_name],
            "field_only_index": True,
        },
    )


def occurrence_field_queries(
    fields: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, tuple[str, ...]]:
    normalized = normalize_occurrence_fields(fields)
    return {
        field_name: normalized[field_name]["query_terms"]
        for field_name in OCCURRENCE_FIELDS
    }


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    channel_weights: Mapping[str, float] | None = None,
    rrf_k0: int = 60,
) -> tuple[str, ...]:
    weights = dict(channel_weights or {})
    k0 = max(1, int(rrf_k0))
    scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    for channel, raw_ids in rankings.items():
        weight = float(weights.get(channel, 1.0))
        if weight < 0.0:
            raise ValueError("RRF channel weights cannot be negative")
        passage_ids = tuple(dict.fromkeys(str(value) for value in raw_ids if str(value)))
        for rank, passage_id in enumerate(passage_ids, start=1):
            scores[passage_id] = scores.get(passage_id, 0.0) + weight / (k0 + rank)
            best_ranks[passage_id] = min(best_ranks.get(passage_id, rank), rank)
    return tuple(
        sorted(
            scores,
            key=lambda passage_id: (
                -scores[passage_id],
                best_ranks[passage_id],
                passage_id,
            ),
        )
    )


def passage_rank(passage_ids: Sequence[str], passage_id: str) -> int | None:
    target = str(passage_id)
    for rank, value in enumerate(passage_ids, start=1):
        if str(value) == target:
            return rank
    return None


def _terms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(
        dict.fromkeys(str(item).strip() for item in value if str(item).strip())
    )


def _intervals(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for value in values:
        if len(value) != 2:
            continue
        start, end = sorted((float(value[0]), float(value[1])))
        if end > start:
            intervals.append((start, end))
    return tuple(intervals)
