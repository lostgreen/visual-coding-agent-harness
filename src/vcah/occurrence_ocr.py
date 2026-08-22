from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from vcah.caption_lexical_index import normalize_caption_query, tokenize_caption_text
from vcah.caption_schema import CaptionHitV1, CaptionPassageV1


GEMINI_OCR_CONTRACT = "MMLifelongGeminiOCRV1"
OCR_PROMPT_VARIANTS = ("generic_v0", "ui_aware_v1")
OCR_REGIONS = frozenset(
    {
        "boss_name",
        "item_popup",
        "menu_title",
        "skill_name",
        "subtitle",
        "location_title",
        "victory_defeat",
        "hud_number",
        "other",
    }
)
OCR_CONFIDENCES = frozenset({"high", "medium", "low"})
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def gemini_ocr_prompt(frame_labels: Sequence[str], *, variant: str) -> str:
    selected = str(variant).strip()
    if selected not in OCR_PROMPT_VARIANTS:
        raise ValueError(f"unsupported OCR prompt variant: {variant}")
    labels = tuple(str(label).strip() for label in frame_labels if str(label).strip())
    if not labels:
        raise ValueError("OCR prompt requires at least one frame label")
    focus = (
        "Transcribe every clearly visible on-screen text string."
        if selected == "generic_v0"
        else (
            "Actively scan game UI regions that generic captioning often misses: boss HP-bar "
            "names, item or loot popups, menu and skill titles, subtitles, location titles, "
            "victory/defeat overlays, and HUD numbers."
        )
    )
    return (
        "You are a strict OCR engine for a chronological sequence of video-game frames.\n"
        f"{focus}\n"
        "Copy only text that is visibly supported by pixels in that exact frame. Preserve the "
        "original Chinese/English characters, capitalization, punctuation, and digits. Do not "
        "translate, summarize, infer a hidden entity name, complete a partly occluded word, or "
        "use game knowledge. If uncertain, use low confidence; if no text is readable, return an "
        "empty visible_text list.\n"
        "Return JSON only with this exact shape:\n"
        '{"frames":[{"frame_label":"frame_01","visible_text":'
        '[{"text":"exact pixels","region":"boss_name","confidence":"high"}]}]}\n'
        f"Allowed frame_label values: {json.dumps(labels, ensure_ascii=False)}\n"
        f"Allowed region values: {json.dumps(sorted(OCR_REGIONS))}\n"
        f"Allowed confidence values: {json.dumps(sorted(OCR_CONFIDENCES))}"
    )


def parse_gemini_ocr_response(
    raw: str,
    *,
    allowed_frame_labels: Sequence[str],
    max_rows_per_frame: int = 96,
) -> tuple[dict[str, Any], ...] | None:
    diagnostic = parse_gemini_ocr_response_diagnostic(
        raw,
        allowed_frame_labels=allowed_frame_labels,
        max_rows_per_frame=max_rows_per_frame,
    )
    rows = diagnostic.get("rows")
    return tuple(rows) if isinstance(rows, Sequence) else None


def parse_gemini_ocr_response_diagnostic(
    raw: str,
    *,
    allowed_frame_labels: Sequence[str],
    max_rows_per_frame: int = 96,
) -> dict[str, Any]:
    labels = tuple(
        dict.fromkeys(
            str(label).strip() for label in allowed_frame_labels if str(label).strip()
        )
    )
    if not labels:
        raise ValueError("allowed_frame_labels cannot be empty")
    normalization_counts: Counter[str] = Counter()
    try:
        payload = json.loads(_FENCE_RE.sub("", str(raw).strip()))
    except (TypeError, ValueError, json.JSONDecodeError):
        return _parse_diagnostic(None, "invalid_json", normalization_counts)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        frames = payload
        normalization_counts["root_array_to_frames"] += 1
    elif isinstance(payload, Mapping):
        frames = payload.get("frames")
    else:
        return _parse_diagnostic(None, "root_not_object", normalization_counts)
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return _parse_diagnostic(None, "frames_not_array", normalization_counts)
    allowed = set(labels)
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            return _parse_diagnostic(
                None, "frame_not_object", normalization_counts
            )
        label = str(frame.get("frame_label", "") or "").strip()
        if label not in allowed:
            contained = tuple(value for value in labels if value in label)
            if len(contained) == 1:
                label = contained[0]
                normalization_counts["frame_label_alias"] += 1
            else:
                return _parse_diagnostic(
                    None, "unknown_frame_label", normalization_counts
                )
        if label in seen:
            return _parse_diagnostic(
                None, "duplicate_frame_label", normalization_counts
            )
        seen.add(label)
        visible = frame.get("visible_text", ())
        if isinstance(visible, str):
            visible = (visible,)
            normalization_counts["visible_text_string"] += 1
        if not isinstance(visible, Sequence) or isinstance(visible, bytes):
            return _parse_diagnostic(
                None, "visible_text_not_array", normalization_counts
            )
        if len(visible) > max(1, int(max_rows_per_frame)):
            return _parse_diagnostic(None, "too_many_rows", normalization_counts)
        for row in visible:
            if isinstance(row, str):
                row = {"text": row, "region": "other", "confidence": "low"}
                normalization_counts["string_row"] += 1
            if not isinstance(row, Mapping):
                return _parse_diagnostic(None, "row_not_object", normalization_counts)
            text = _normalize_ocr_text(row.get("text", ""))
            if not text:
                normalization_counts["blank_row_dropped"] += 1
                continue
            if len(text) > 500:
                return _parse_diagnostic(None, "text_too_long", normalization_counts)
            region = str(row.get("region", "other") or "other").strip().casefold()
            if region not in OCR_REGIONS:
                region = "other"
                normalization_counts["unknown_region_to_other"] += 1
            confidence = _normalized_confidence(row.get("confidence", "low"))
            if confidence != str(row.get("confidence", "low") or "low").strip().casefold():
                normalization_counts["confidence_normalized"] += 1
            parsed.append(
                {
                    "frame_label": label,
                    "text": text,
                    "normalized_text": normalize_caption_query(text),
                    "region": region,
                    "confidence": confidence,
                }
            )
    normalization_counts["implicit_empty_frame"] += len(allowed - seen)
    return _parse_diagnostic(tuple(parsed), "success", normalization_counts)


def deduplicate_ocr_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    metadata = frame_metadata or {}
    grouped: dict[str, dict[str, Any]] = {}
    for raw in rows:
        text = _normalize_ocr_text(raw.get("text", ""))
        key = normalize_caption_query(text)
        label = str(raw.get("frame_label", "") or "").strip()
        region = str(raw.get("region", "other") or "other").strip().casefold()
        confidence = str(raw.get("confidence", "low") or "low").strip().casefold()
        if not key or not label:
            continue
        current = grouped.get(key)
        if current is None:
            current = {
                "text": text,
                "normalized_text": key,
                "regions": [],
                "max_confidence": confidence,
                "frame_labels": [],
                "virtual_times_sec": [],
                "segment_ids": [],
            }
            grouped[key] = current
        if region not in current["regions"]:
            current["regions"].append(region)
        if _CONFIDENCE_ORDER.get(confidence, -1) > _CONFIDENCE_ORDER.get(
            str(current["max_confidence"]), -1
        ):
            current["max_confidence"] = confidence
        if label not in current["frame_labels"]:
            current["frame_labels"].append(label)
        frame = metadata.get(label, {})
        virtual_time = frame.get("virtual_time_sec")
        if isinstance(virtual_time, (int, float)):
            value = round(float(virtual_time), 3)
            if value not in current["virtual_times_sec"]:
                current["virtual_times_sec"].append(value)
        segment_id = str(frame.get("segment_id", "") or "").strip()
        if segment_id and segment_id not in current["segment_ids"]:
            current["segment_ids"].append(segment_id)
    return tuple(
        grouped[key]
        for key in sorted(
            grouped,
            key=lambda value: (
                min(grouped[value]["virtual_times_sec"] or [float("inf")]),
                value,
            ),
        )
    )


def ocr_query_overlap(
    ocr_rows: Sequence[Mapping[str, Any]],
    query_texts: Sequence[str],
) -> dict[str, Any]:
    ocr_tokens = {
        token
        for row in ocr_rows
        for token in tokenize_caption_text(str(row.get("text", "") or ""))
        if _diagnostic_token(token)
    }
    query_tokens = {
        token
        for query in query_texts
        for token in tokenize_caption_text(str(query))
        if _diagnostic_token(token)
    }
    matched = sorted(ocr_tokens & query_tokens)
    return {
        "ocr_token_count": len(ocr_tokens),
        "query_token_count": len(query_tokens),
        "matched_token_count": len(matched),
        "query_token_recall": (
            len(matched) / len(query_tokens) if query_tokens else 0.0
        ),
        "matched_tokens": matched[:24],
    }


def ocr_text_has_query_evidence(text: str, query_texts: Sequence[str]) -> bool:
    return bool(
        ocr_query_overlap(({"text": str(text)},), query_texts)[
            "matched_token_count"
        ]
    )


def enrich_caption_passages_with_ocr(
    passages: Sequence[CaptionPassageV1],
    ocr_rows: Sequence[Mapping[str, Any]],
    *,
    nearest_tolerance_sec: float = 8.0,
) -> tuple[CaptionPassageV1, ...]:
    additions: dict[str, list[str]] = {}
    for row in ocr_rows:
        text = _normalize_ocr_text(row.get("text", ""))
        times = tuple(
            float(value)
            for value in tuple(row.get("virtual_times_sec", ()) or ())
            if isinstance(value, (int, float))
        )
        segment_ids = {
            str(value)
            for value in tuple(row.get("segment_ids", ()) or ())
            if str(value)
        }
        if not text or not times:
            continue
        candidates: list[tuple[float, CaptionPassageV1]] = []
        for passage in passages:
            passage_segments = {
                str(value)
                for value in tuple(passage.metadata.get("source_segments", ()) or ())
                if str(value)
            }
            if segment_ids and passage_segments and segment_ids.isdisjoint(passage_segments):
                continue
            distance = min(_point_interval_distance(value, passage) for value in times)
            if distance <= max(0.0, float(nearest_tolerance_sec)):
                candidates.append((distance, passage))
        if not candidates:
            continue
        best_distance = min(distance for distance, _ in candidates)
        for distance, passage in candidates:
            if distance > best_distance + 1e-6:
                continue
            values = additions.setdefault(passage.passage_id, [])
            if normalize_caption_query(text) not in {
                normalize_caption_query(value) for value in values
            }:
                values.append(text)
    enriched: list[CaptionPassageV1] = []
    for passage in passages:
        values = additions.get(passage.passage_id, [])
        if not values:
            enriched.append(passage)
            continue
        suffix = " ; ".join(values)
        enriched.append(
            replace(
                passage,
                text=f"{passage.text} [VISIBLE OCR] {suffix}",
                metadata={
                    **dict(passage.metadata),
                    "ocr_contract": GEMINI_OCR_CONTRACT,
                    "ocr_text": list(values),
                },
            )
        )
    return tuple(enriched)


def ocr_sidecar_passages(
    enriched_passages: Sequence[CaptionPassageV1],
) -> tuple[CaptionPassageV1, ...]:
    """Keep passage lineage while removing the original Caption text from the OCR ranker."""

    sidecar: list[CaptionPassageV1] = []
    for passage in enriched_passages:
        raw_values = passage.metadata.get("ocr_text", ())
        values = (
            tuple(str(value) for value in raw_values if str(value).strip())
            if isinstance(raw_values, Sequence) and not isinstance(raw_values, str)
            else ()
        )
        sidecar.append(
            replace(
                passage,
                text=" ; ".join(values),
                metadata={
                    **dict(passage.metadata),
                    "ocr_sidecar_only": True,
                },
            )
        )
    return tuple(sidecar)


def fuse_caption_hit_ranks(
    baseline_hits: Sequence[CaptionHitV1 | Mapping[str, Any]],
    ocr_hits: Sequence[CaptionHitV1 | Mapping[str, Any]],
    *,
    top_k: int,
    rrf_k0: int = 60,
    baseline_weight: float = 1.0,
    ocr_weight: float = 1.0,
) -> tuple[dict[str, Any], ...]:
    sources: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    components: dict[str, dict[str, int]] = {}
    for name, hits, weight in (
        ("baseline", baseline_hits, float(baseline_weight)),
        ("ocr", ocr_hits, float(ocr_weight)),
    ):
        for fallback_rank, raw in enumerate(hits, start=1):
            hit = asdict(raw) if isinstance(raw, CaptionHitV1) else dict(raw)
            passage_id = str(hit.get("passage_id", "") or "")
            if not passage_id:
                continue
            rank = max(1, int(hit.get("rank", fallback_rank) or fallback_rank))
            scores[passage_id] = scores.get(passage_id, 0.0) + weight / (
                max(1, int(rrf_k0)) + rank
            )
            components.setdefault(passage_id, {})[name] = rank
            sources.setdefault(passage_id, hit)
    ordered = sorted(
        scores,
        key=lambda passage_id: (
            -scores[passage_id],
            int(components[passage_id].get("baseline", 10**9)),
            int(components[passage_id].get("ocr", 10**9)),
            passage_id,
        ),
    )[: max(1, int(top_k))]
    fused: list[dict[str, Any]] = []
    for rank, passage_id in enumerate(ordered, start=1):
        hit = dict(sources[passage_id])
        metadata = hit.get("metadata", {})
        hit["rank"] = rank
        hit["fused_score"] = scores[passage_id]
        hit["metadata"] = {
            **(dict(metadata) if isinstance(metadata, Mapping) else {}),
            "ocr_fusion_contract": GEMINI_OCR_CONTRACT,
            "component_ranks": components[passage_id],
        }
        fused.append(hit)
    return tuple(fused)


def _normalize_ocr_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).strip()


def _point_interval_distance(value: float, passage: CaptionPassageV1) -> float:
    if passage.virtual_start_sec <= value <= passage.virtual_end_sec:
        return 0.0
    return min(
        abs(value - passage.virtual_start_sec),
        abs(value - passage.virtual_end_sec),
    )


def _diagnostic_token(token: str) -> bool:
    value = str(token).strip()
    if not value:
        return False
    if len(value) >= 2:
        return True
    return value.isdigit()


def _normalized_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        number = float(value)
        return "high" if number >= 0.8 else "medium" if number >= 0.5 else "low"
    text = str(value or "low").strip().casefold()
    if text in OCR_CONFIDENCES:
        return text
    aliases = {
        "very high": "high",
        "certain": "high",
        "moderate": "medium",
        "uncertain": "low",
    }
    return aliases.get(text, "low")


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
