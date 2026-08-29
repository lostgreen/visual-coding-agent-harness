from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from vcah.caption_lexical_index import normalize_caption_query
from vcah.caption_schema import CaptionPassageV1


GLOBAL_ENTITY_SIDECAR_CONTRACT = "WP16-6-global-ocr-entity-sidecar-v1"
ENTITY_TYPES = frozenset(
    {
        "boss_name",
        "npc_name",
        "location",
        "item_name",
        "skill_name",
        "menu_title",
        "chapter_title",
        "other_named_entity",
    }
)
ENTITY_UI_REGIONS = frozenset(
    {
        "boss_name_bar",
        "npc_label",
        "item_popup",
        "location_title",
        "menu_title",
        "chapter_title",
        "skill_panel",
        "other",
    }
)
HIGH_VALUE_UI_REGIONS = frozenset(ENTITY_UI_REGIONS - {"other"})
ENTITY_CONFIDENCES = frozenset({"high", "medium", "low"})
DEFAULT_BLOCKED_ENTITY_TEXT = frozenset(
    normalize_caption_query(value)
    for value in ("攻击", "返回", "确认", "取消", "x3", "F", "E")
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_NUMERIC_ONLY_RE = re.compile(r"^[\d\s.,:/+\-xX×%]+$")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
MAX_GLOBAL_ENTITY_ROWS_PER_FRAME = 32


def global_entity_ocr_prompt(frame_labels: Sequence[str]) -> str:
    labels = tuple(
        dict.fromkeys(str(label).strip() for label in frame_labels if str(label).strip())
    )
    if not labels:
        raise ValueError("entity OCR prompt requires at least one frame label")
    return (
        "You are a strict entity-oriented OCR engine for chronological video-game frames.\n"
        "Copy only stable named-entity text visibly supported by pixels in each exact frame: "
        "boss or NPC names, locations, item or skill names, menu titles, and chapter titles. "
        "Do not report damage or health numbers, button prompts, generic HUD labels, isolated "
        "single characters, or ordinary dialogue. Do not translate, summarize, infer an event "
        "or state, guess an alias, use game knowledge, or complete occluded text. The Caption, "
        "question, answer, and official intervals are unavailable.\n"
        "Return JSON only. Emit every allowed frame_label exactly once with this shape:\n"
        '{"frames":[{"frame_label":"frame_01","entities":['
        '{"text":"exact pixels","entity_type":"boss_name",'
        '"ui_region":"boss_name_bar","confidence":"high"}]}]}\n'
        f"Allowed frame_label values: {json.dumps(labels, ensure_ascii=False)}\n"
        f"Allowed entity_type values: {json.dumps(sorted(ENTITY_TYPES))}\n"
        f"Allowed ui_region values: {json.dumps(sorted(ENTITY_UI_REGIONS))}\n"
        f"Allowed confidence values: {json.dumps(sorted(ENTITY_CONFIDENCES))}"
    )


def parse_global_entity_ocr_response_diagnostic(
    raw: str,
    *,
    allowed_frame_labels: Sequence[str],
    max_rows_per_frame: int = MAX_GLOBAL_ENTITY_ROWS_PER_FRAME,
) -> dict[str, Any]:
    labels = tuple(
        dict.fromkeys(
            str(label).strip() for label in allowed_frame_labels if str(label).strip()
        )
    )
    if not labels:
        raise ValueError("allowed_frame_labels cannot be empty")
    counts: Counter[str] = Counter()
    try:
        payload = json.loads(_FENCE_RE.sub("", str(raw).strip()))
    except (TypeError, ValueError, json.JSONDecodeError):
        return _parse_diagnostic(None, "invalid_json", counts)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        frames = payload
        counts["root_array_to_frames"] += 1
    elif isinstance(payload, Mapping):
        frames = payload.get("frames")
    else:
        return _parse_diagnostic(None, "root_not_object", counts)
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return _parse_diagnostic(None, "frames_not_array", counts)

    # Some JSON-mode responses repeat the requested root schema once. Unwrap only
    # the single, otherwise-empty wrapper so the repair cannot reorder evidence.
    if (
        len(frames) == 1
        and isinstance(frames[0], Mapping)
        and set(frames[0]) == {"frames"}
        and isinstance(frames[0]["frames"], Sequence)
        and not isinstance(frames[0]["frames"], (str, bytes))
    ):
        frames = frames[0]["frames"]
        counts["single_frames_wrapper_unwrapped"] += 1

    positional_labels: tuple[str, ...] | None = None
    if len(frames) == len(labels) and all(
        isinstance(frame, Mapping)
        and not str(frame.get("frame_label", "") or "").strip()
        for frame in frames
    ):
        positional_labels = labels
        counts["frame_labels_recovered_by_position"] += len(labels)

    allowed = set(labels)
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            return _parse_diagnostic(None, "frame_not_object", counts)
        label = (
            positional_labels[frame_index]
            if positional_labels is not None
            else str(frame.get("frame_label", "") or "").strip()
        )
        if label not in allowed:
            contained = tuple(value for value in labels if value in label)
            if len(contained) != 1:
                return _parse_diagnostic(None, "unknown_frame_label", counts)
            label = contained[0]
            counts["frame_label_alias"] += 1
        if label in seen:
            return _parse_diagnostic(None, "duplicate_frame_label", counts)
        seen.add(label)
        entities = frame.get("entities", ())
        if isinstance(entities, str):
            entities = ({"text": entities},)
            counts["entities_string"] += 1
        if not isinstance(entities, Sequence) or isinstance(entities, bytes):
            return _parse_diagnostic(None, "entities_not_array", counts)
        if len(entities) > max(1, int(max_rows_per_frame)):
            return _parse_diagnostic(None, "too_many_rows", counts)
        for raw_row in entities:
            if isinstance(raw_row, str):
                raw_row = {"text": raw_row}
                counts["string_row"] += 1
            if not isinstance(raw_row, Mapping):
                return _parse_diagnostic(None, "row_not_object", counts)
            text = normalize_entity_text(raw_row.get("text", ""))
            if not text:
                counts["blank_row_dropped"] += 1
                continue
            if len(text) > 160:
                return _parse_diagnostic(None, "text_too_long", counts)
            entity_type = str(
                raw_row.get("entity_type", raw_row.get("type", "other_named_entity"))
                or "other_named_entity"
            ).strip().casefold()
            if entity_type not in ENTITY_TYPES:
                entity_type = "other_named_entity"
                counts["unknown_type_to_other"] += 1
            ui_region = str(
                raw_row.get("ui_region", raw_row.get("region", "other")) or "other"
            ).strip().casefold()
            if ui_region not in ENTITY_UI_REGIONS:
                ui_region = "other"
                counts["unknown_region_to_other"] += 1
            confidence = _normalized_confidence(raw_row.get("confidence", "low"))
            parsed.append(
                {
                    "frame_label": label,
                    "text": text,
                    "normalized_text": normalize_caption_query(text),
                    "entity_type": entity_type,
                    "ui_region": ui_region,
                    "confidence": confidence,
                }
            )
    counts["implicit_empty_frame"] += len(allowed - seen)
    return _parse_diagnostic(tuple(parsed), "success", counts)


def admit_global_entity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    passage_id: str,
    frame_metadata: Mapping[str, Mapping[str, Any]],
    multi_frame_min_support: int = 2,
    high_value_regions: Sequence[str] = tuple(sorted(HIGH_VALUE_UI_REGIONS)),
    blocked_normalized_text: Sequence[str] = tuple(sorted(DEFAULT_BLOCKED_ENTITY_TEXT)),
    lexical_filter_enabled: bool = True,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    rejection_counts: Counter[str] = Counter()
    for raw in rows:
        text = normalize_entity_text(raw.get("text", ""))
        normalized = normalize_caption_query(text)
        if not normalized:
            rejection_counts["blank"] += 1
            continue
        grouped.setdefault(normalized, []).append({**dict(raw), "text": text})

    admitted: list[dict[str, Any]] = []
    high_value = {str(value).strip().casefold() for value in high_value_regions}
    blocked = {normalize_caption_query(value) for value in blocked_normalized_text}
    for normalized in sorted(grouped):
        candidates = grouped[normalized]
        text = str(candidates[0]["text"])
        lexical_reason = (
            _lexical_rejection_reason(text, normalized, blocked)
            if lexical_filter_enabled
            else None
        )
        if lexical_reason:
            rejection_counts[lexical_reason] += 1
            continue
        frame_labels = tuple(
            dict.fromkeys(
                str(row.get("frame_label", "") or "").strip()
                for row in candidates
                if str(row.get("frame_label", "") or "").strip()
            )
        )
        regions = tuple(
            sorted(
                {
                    str(row.get("ui_region", "other") or "other").strip().casefold()
                    for row in candidates
                }
            )
        )
        support_count = len(frame_labels)
        repeated = support_count >= max(1, int(multi_frame_min_support))
        high_value_single = bool(set(regions) & high_value)
        if not repeated and not high_value_single:
            rejection_counts["insufficient_support"] += 1
            continue
        metadata_rows = [frame_metadata.get(label, {}) for label in frame_labels]
        entity_type = _majority_value(candidates, "entity_type", "other_named_entity")
        confidence = max(
            (
                str(row.get("confidence", "low") or "low").strip().casefold()
                for row in candidates
            ),
            key=lambda value: (_CONFIDENCE_ORDER.get(value, -1), value),
            default="low",
        )
        admitted.append(
            {
                "contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
                "text": text,
                "normalized": normalized,
                "type": entity_type,
                "passage_id": str(passage_id),
                "frame_ids": [
                    str(row.get("frame_id", label) or label)
                    for label, row in zip(frame_labels, metadata_rows)
                ],
                "frame_labels": list(frame_labels),
                "timestamps_sec": sorted(
                    {
                        round(float(row["virtual_time_sec"]), 3)
                        for row in metadata_rows
                        if isinstance(row.get("virtual_time_sec"), (int, float))
                    }
                ),
                "segment_ids": sorted(
                    {
                        str(row.get("segment_id", "") or "")
                        for row in metadata_rows
                        if str(row.get("segment_id", "") or "")
                    }
                ),
                "source_video_ids": sorted(
                    {
                        str(row.get("source_video_id", "") or "")
                        for row in metadata_rows
                        if str(row.get("source_video_id", "") or "")
                    }
                ),
                "regions": list(regions),
                "support_count": support_count,
                "max_confidence": confidence,
                "admission_reason": (
                    "multi_frame_consensus" if repeated else "high_value_ui_region"
                ),
            }
        )
    return {
        "admitted_rows": tuple(admitted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "candidate_unique_text_count": len(grouped),
    }


def fixed3_passage_targets(passage: CaptionPassageV1) -> tuple[dict[str, Any], ...]:
    start = float(passage.virtual_start_sec)
    end = float(passage.virtual_end_sec)
    midpoint = (start + end) / 2.0
    end_inside = max(start, end - 0.001) if end > start else start
    grouped: dict[float, list[str]] = {}
    for position, value in (
        ("start", start),
        ("midpoint", midpoint),
        ("end_minus_1ms", end_inside),
    ):
        grouped.setdefault(round(value, 3), []).append(position)
    return tuple(
        {
            "virtual_time_sec": value,
            "sample_positions": tuple(grouped[value]),
        }
        for value in sorted(grouped)
    )


def select_hashed_passages(
    passages: Sequence[CaptionPassageV1],
    *,
    seed: str,
    count: int,
) -> tuple[CaptionPassageV1, ...]:
    limit = int(count)
    if limit < 1 or limit > len(passages):
        raise ValueError("hashed passage selection count is outside passage collection")
    return tuple(
        sorted(
            passages,
            key=lambda passage: (
                hashlib.sha256(
                    f"{str(seed)}:{passage.passage_id}".encode("utf-8")
                ).hexdigest(),
                passage.passage_id,
            ),
        )[:limit]
    )


def build_entity_sidecar_passages(
    passages: Sequence[CaptionPassageV1],
    entity_rows: Sequence[Mapping[str, Any]],
) -> tuple[CaptionPassageV1, ...]:
    by_passage: dict[str, list[str]] = {}
    for row in entity_rows:
        passage_id = str(row.get("passage_id", "") or "")
        text = normalize_entity_text(row.get("text", ""))
        if passage_id and text:
            values = by_passage.setdefault(passage_id, [])
            if normalize_caption_query(text) not in {
                normalize_caption_query(value) for value in values
            }:
                values.append(text)
    return tuple(
        replace(
            passage,
            text=" ; ".join(by_passage.get(passage.passage_id, ())),
            metadata={
                **dict(passage.metadata),
                "global_entity_sidecar_contract": GLOBAL_ENTITY_SIDECAR_CONTRACT,
                "entity_sidecar_only": True,
                "entity_count": len(by_passage.get(passage.passage_id, ())),
            },
        )
        for passage in passages
    )


def global_entity_duplicate_stats(
    entity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    passages_by_text: dict[str, set[str]] = {}
    for row in entity_rows:
        normalized = normalize_caption_query(row.get("normalized", row.get("text", "")))
        passage_id = str(row.get("passage_id", "") or "")
        if normalized and passage_id:
            passages_by_text.setdefault(normalized, set()).add(passage_id)
    duplicate_texts = {
        text: passages for text, passages in passages_by_text.items() if len(passages) > 1
    }
    duplicate_rows = sum(len(passages) for passages in duplicate_texts.values())
    total_rows = sum(len(passages) for passages in passages_by_text.values())
    return {
        "unique_entity_text_count": len(passages_by_text),
        "duplicate_entity_text_count": len(duplicate_texts),
        "duplicate_entity_rate": (
            len(duplicate_texts) / len(passages_by_text) if passages_by_text else 0.0
        ),
        "entity_passage_pair_count": total_rows,
        "duplicate_entity_passage_pair_count": duplicate_rows,
        "duplicate_entity_passage_pair_rate": (
            duplicate_rows / total_rows if total_rows else 0.0
        ),
    }


def admitted_entity_row_valid(row: Mapping[str, Any]) -> bool:
    text = normalize_entity_text(row.get("text", ""))
    normalized = normalize_caption_query(text)
    if _lexical_rejection_reason(text, normalized, set(DEFAULT_BLOCKED_ENTITY_TEXT)):
        return False
    support_count = int(row.get("support_count", 0) or 0)
    regions = {
        str(value).strip().casefold()
        for value in tuple(row.get("regions", ()) or ())
    }
    return support_count >= 2 or bool(regions & HIGH_VALUE_UI_REGIONS)


def normalize_entity_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).strip()


def _lexical_rejection_reason(
    text: str,
    normalized: str,
    blocked: set[str],
) -> str | None:
    if normalized in blocked:
        return "blocked_text"
    if _NUMERIC_ONLY_RE.fullmatch(text):
        return "numeric_only"
    cjk_count = len(_CJK_RE.findall(text))
    english_count = len(_ENGLISH_TOKEN_RE.findall(text))
    if cjk_count:
        return None if cjk_count >= 2 else "single_cjk_character"
    if english_count < 2:
        return "insufficient_english_tokens"
    return None


def _majority_value(
    rows: Sequence[Mapping[str, Any]], key: str, default: str
) -> str:
    counts = Counter(
        str(row.get(key, default) or default).strip().casefold() for row in rows
    )
    return min(counts, key=lambda value: (-counts[value], value)) if counts else default


def _normalized_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        number = float(value)
        return "high" if number >= 0.8 else "medium" if number >= 0.5 else "low"
    text = str(value or "low").strip().casefold()
    if text in ENTITY_CONFIDENCES:
        return text
    return {
        "very high": "high",
        "certain": "high",
        "moderate": "medium",
        "uncertain": "low",
    }.get(text, "low")


def _parse_diagnostic(
    rows: Sequence[Mapping[str, Any]] | None,
    status: str,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "status": str(status),
        "rows": tuple(dict(row) for row in rows) if rows is not None else None,
        "normalization_counts": {
            str(key): int(value) for key, value in sorted(counts.items()) if int(value)
        },
    }
