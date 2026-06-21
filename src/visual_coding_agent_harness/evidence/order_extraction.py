"""Ordered-list evidence extraction and hypothesis matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from .order_hypotheses import OptionOrderHypothesis, OrderedSetSpec


@dataclass(frozen=True)
class ObservedOrder:
    ordered_set_id: str
    entity_order: tuple[str, ...]
    source: Literal["indexed_asr", "ocr", "caption", "visual_fact"]
    support_span: str
    cue_ids: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class OrderMatch:
    ordered_set_id: str
    status: Literal["full_match", "partial_match", "ambiguous", "no_match"]
    option_id: str = ""
    confidence: float = 0.0
    scores: Mapping[str, float] = field(default_factory=dict)


def extract_observed_order_from_text(
    text: str,
    ordered_set: OrderedSetSpec,
    *,
    max_window_chars: int = 1200,
    source: Literal["indexed_asr", "ocr", "caption", "visual_fact"] = "indexed_asr",
    cue_ids: Sequence[str] = (),
) -> ObservedOrder | None:
    raw = str(text or "")
    if not raw.strip():
        return None
    mentions: list[tuple[int, int, str]] = []
    for entity in ordered_set.entities:
        phrases = [entity.canonical_name, *list(entity.aliases)]
        first: tuple[int, int, str] | None = None
        for phrase in phrases:
            match = re.search(_phrase_pattern(phrase), raw, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = (match.start(), match.end(), entity.entity_id)
            if first is None or candidate[0] < first[0]:
                first = candidate
        if first is not None:
            mentions.append(first)
    mentions.sort(key=lambda item: item[0])
    order: list[str] = []
    seen: set[str] = set()
    for _start, _end, entity_id in mentions:
        if entity_id in seen:
            continue
        seen.add(entity_id)
        order.append(entity_id)
    if len(order) < 3:
        return None
    span_start = max(0, mentions[0][0])
    span_end = min(len(raw), mentions[-1][1])
    if span_end - span_start > max_window_chars:
        span_end = min(len(raw), span_start + max_window_chars)
    confidence = 0.9 if len(order) == len(tuple(ordered_set.entities)) else 0.65
    return ObservedOrder(
        ordered_set_id=ordered_set.set_id,
        entity_order=tuple(order),
        source=source,
        support_span=" ".join(raw[span_start:span_end].split()),
        cue_ids=tuple(str(cue_id) for cue_id in cue_ids if str(cue_id).strip()),
        confidence=confidence,
    )


def match_observed_order_to_hypotheses(
    observed: ObservedOrder,
    hypotheses: Sequence[OptionOrderHypothesis],
) -> OrderMatch:
    if observed is None:
        return OrderMatch(ordered_set_id="", status="no_match")
    observed_order = tuple(observed.entity_order)
    scores: dict[str, float] = {}
    for hypothesis in hypotheses:
        expected = tuple(hypothesis.ordered_entity_ids)
        if not expected:
            continue
        if observed_order == expected:
            return OrderMatch(
                ordered_set_id=observed.ordered_set_id,
                status="full_match",
                option_id=hypothesis.option_id,
                confidence=max(0.9, observed.confidence),
                scores={hypothesis.option_id: 1.0},
            )
        scores[hypothesis.option_id] = _pairwise_order_score(observed_order, expected)
    if not scores:
        return OrderMatch(ordered_set_id=observed.ordered_set_id, status="no_match", scores={})
    best_score = max(scores.values())
    best = [option for option, score in scores.items() if score == best_score]
    if len(best) != 1:
        return OrderMatch(ordered_set_id=observed.ordered_set_id, status="ambiguous", scores=scores)
    return OrderMatch(
        ordered_set_id=observed.ordered_set_id,
        status="partial_match",
        option_id=best[0],
        confidence=min(0.75, best_score),
        scores=scores,
    )


def _pairwise_order_score(observed: tuple[str, ...], expected: tuple[str, ...]) -> float:
    positions = {entity_id: index for index, entity_id in enumerate(expected)}
    comparable = [entity_id for entity_id in observed if entity_id in positions]
    total = 0
    correct = 0
    for left_index, left in enumerate(comparable):
        for right in comparable[left_index + 1 :]:
            total += 1
            if positions[left] < positions[right]:
                correct += 1
    return correct / total if total else 0.0


def _phrase_pattern(phrase: str) -> str:
    escaped = re.escape(str(phrase or "").strip())
    escaped = escaped.replace(r"\ ", r"[\s-]+")
    return rf"(?<!\w){escaped}(?!\w)"
