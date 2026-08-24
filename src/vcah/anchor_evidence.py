from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from vcah.caption_context import expand_query_conditioned_context
from vcah.caption_evidence_bundle import build_caption_evidence_bundle_set
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1
from vcah.occurrence_ocr import enrich_caption_passages_with_ocr


ANCHOR_EVIDENCE_CONTRACT = "WP16-3-anchor-conditioned-evidence-v1"


@dataclass(frozen=True)
class AnchorEvidenceRequest:
    eligible: bool
    direction: str | None
    relation: str | None
    evidence_channels: tuple[str, ...]
    reason: str


def infer_anchor_evidence_request(question: str) -> AnchorEvidenceRequest:
    """Map explicit temporal wording to a deterministic expansion request."""

    text = " ".join(str(question).casefold().split())
    direction: str | None = None
    relation: str | None = None
    if any(
        token in text
        for token in (
            " after ",
            "after ",
            "随后",
            "之后",
            "以后",
            "接着",
            "接下来",
        )
    ):
        direction = "after"
        relation = "after"
    elif any(
        token in text
        for token in (
            " before ",
            "before ",
            "prior to",
            "此前",
            "之前",
            "以前",
        )
    ):
        direction = "before"
        relation = "before"

    channels: list[str] = []
    if any(
        token in text
        for token in (
            " say",
            "said",
            "sing",
            "line",
            "dialogue",
            "subtitle",
            "说了",
            "唱",
            "台词",
            "字幕",
        )
    ):
        channels.append("subtitle")
    if any(
        token in text
        for token in (
            "item",
            "material",
            "reward",
            "obtain",
            "acquire",
            "name",
            "title",
            "spirit",
            "gourd",
            "armor",
            "equipment",
            "talent",
            "物品",
            "材料",
            "获得",
            "名字",
            "名称",
            "精魄",
            "葫芦",
            "装备",
        )
    ):
        channels.append("visible_ocr")
    if any(
        token in text
        for token in (
            "how many units",
            "what level",
            "how much",
            "多少单位",
            "数值",
            "等级",
        )
    ):
        channels = [value for value in channels if value != "visible_ocr"]
        channels.append("numeric_ocr")
    if (
        any(
            token in text
            for token in (
                "how many",
                "how many times",
                "how long",
                "number of",
                " in total",
                "多少",
                "几次",
                "多久",
            )
        )
        and "numeric_ocr" not in channels
    ):
        channels.append("numeric_or_aggregate")
    if any(
        token in text
        for token in (
            "look",
            "appearance",
            "transform",
            "color",
            "effect",
            "外观",
            "变身",
            "颜色",
            "效果",
        )
    ):
        channels.append("visual_caption")
    channels = list(dict.fromkeys(channels or ("caption",)))

    if direction is None:
        return AnchorEvidenceRequest(
            False,
            None,
            None,
            tuple(channels),
            "no_explicit_before_or_after_relation",
        )
    if "which of the following" in text or "下列" in text:
        return AnchorEvidenceRequest(
            False,
            direction,
            relation,
            tuple(channels),
            "multi_candidate_relation_requires_separate_anchors",
        )
    relation_prefix = text.split(",", 1)[0]
    if (
        direction == "after"
        and relation_prefix.startswith("after ")
        and "which" in relation_prefix
    ):
        return AnchorEvidenceRequest(
            False,
            direction,
            relation,
            tuple(channels),
            "anchor_is_unknown_answer_target",
        )
    if "respectively" in text or ("first boss" in text and "final boss" in text):
        return AnchorEvidenceRequest(
            False,
            direction,
            relation,
            tuple(channels),
            "multi_occurrence_selection_requires_stateful_evidence",
        )
    if "numeric_or_aggregate" in channels:
        return AnchorEvidenceRequest(
            False,
            direction,
            relation,
            tuple(channels),
            "aggregate_question_requires_stateful_evidence",
        )
    return AnchorEvidenceRequest(
        True,
        direction,
        relation,
        tuple(channels),
        "explicit_directional_relation",
    )


def expand_anchor_conditioned_evidence(
    passages: Sequence[CaptionPassageV1],
    seed_hits: Sequence[CaptionHitV1],
    *,
    question: str,
    ocr_rows: Sequence[Mapping[str, Any]] = (),
    distance: int,
    index_digest: str,
    config_digest: str,
    source_video_id_by_segment: Mapping[str, str] | None = None,
    max_gap_sec: float = 600.0,
) -> dict[str, Any]:
    """Expand frozen anchors directionally; evaluator labels never enter this path."""

    request = infer_anchor_evidence_request(question)
    seed_rows = tuple(seed_hits)
    if not request.eligible:
        return {
            "contract": ANCHOR_EVIDENCE_CONTRACT,
            "request": asdict(request),
            "seed_hits": seed_rows,
            "hits": seed_rows,
            "evidence_bundle_set": build_caption_evidence_bundle_set(seed_rows),
        }

    enriched_passages = enrich_caption_passages_with_ocr(passages, ocr_rows)
    expanded = expand_query_conditioned_context(
        enriched_passages,
        seed_rows,
        distance=max(0, int(distance)),
        time_range=None,
        segment_ids=(),
        index_digest=index_digest,
        config_digest=config_digest,
        source_video_id_by_segment=source_video_id_by_segment,
        max_gap_sec=max(0.0, float(max_gap_sec)),
        direction=str(request.direction),
    )
    annotated: list[CaptionHitV1] = []
    seed_ids = {hit.passage_id for hit in seed_rows}
    for hit in expanded:
        if hit.passage_id in seed_ids:
            annotated.append(hit)
            continue
        metadata = dict(hit.metadata)
        metadata.update(
            {
                "anchor_evidence_contract": ANCHOR_EVIDENCE_CONTRACT,
                "anchor_relation": request.relation,
                "evidence_channels_requested": list(request.evidence_channels),
                "evidence_channels_observed": list(
                    observed_evidence_channels(metadata)
                ),
                "packet_scope_bypassed_after_anchor": True,
            }
        )
        annotated.append(replace(hit, metadata=metadata))
    return {
        "contract": ANCHOR_EVIDENCE_CONTRACT,
        "request": asdict(request),
        "seed_hits": seed_rows,
        "hits": tuple(annotated),
        "evidence_bundle_set": build_caption_evidence_bundle_set(annotated),
    }


def observed_evidence_channels(
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    rows = tuple(metadata.get("ocr_rows", ()) or ())
    channels = ["caption"]
    if rows:
        channels.append("visible_ocr")
    if any(
        any(character.isdigit() for character in str(row.get("text", "") or ""))
        for row in rows
        if isinstance(row, Mapping)
    ):
        channels.append("numeric_ocr")
    if any(
        "subtitle" in {str(region) for region in tuple(row.get("regions", ()) or ())}
        for row in rows
        if isinstance(row, Mapping)
    ):
        channels.append("subtitle")
    return tuple(dict.fromkeys(channels))
