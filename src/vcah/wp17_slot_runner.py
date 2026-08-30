"""Question-blind packets and prompts for WP17 slot-memory construction."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from vcah.wp17_slot_memory import (
    WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
    WP17_SLOT_NAMES,
    WP17_SLOT_OPERATIONS,
    WP17_SLOT_TRANSACTION_CONTRACT,
)


WP17_OCR_AGGREGATION_CONTRACT = "WP17-segment-ocr-surface-aggregation-v1"
WP17_EVIDENCE_ALIAS_CONTRACT = "WP17-local-evidence-alias-v1"
_SPACE_RE = re.compile(r"\s+")


def build_ocr_packet(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    segment_id: str,
    start_sec: float,
    end_sec: float,
) -> tuple[dict[str, Any], ...]:
    """Aggregate repeated OCR tracks by normalized surface inside one segment.

    The packet keeps disjoint local time ranges and aggregate lineage counts. The
    canonical aggregate ID is deterministic; model-facing aliases are assigned
    separately by :func:`alias_current_evidence`.
    """
    start = float(start_sec)
    end = float(end_sec)
    selected: list[dict[str, Any]] = []
    for raw in evidence_rows:
        row = dict(raw)
        row_start = float(row.get("start_sec", 0.0) or 0.0)
        row_end = float(row.get("end_sec", row_start) or row_start)
        if row_start >= end or row_end < start:
            continue
        surfaces = tuple(
            str(value.get("surface", "") if isinstance(value, Mapping) else value).strip()
            for value in tuple(row.get("surfaces", ()) or ())
        )
        surfaces = tuple(value for value in surfaces if value)
        canonical = str(row.get("surface", row.get("canonical_surface", "")) or "").strip()
        if canonical and canonical not in surfaces:
            surfaces = (canonical,) + surfaces
        normalized_surface = str(row.get("normalized_surface", "") or "").strip()
        if not normalized_surface:
            normalized_surface = canonical or (surfaces[0] if surfaces else "")
        normalized_surface = _SPACE_RE.sub(" ", normalized_surface).strip().casefold()
        if not normalized_surface:
            normalized_surface = f"__source__:{row['evidence_id']}"
        selected.append(
            {
                "source_evidence_id": str(row["evidence_id"]),
                "normalized_surface": normalized_surface,
                "surfaces": list(dict.fromkeys(surfaces)),
                "local_time_range_sec": [
                    round(max(0.0, row_start - start), 3),
                    round(min(end - start, row_end - start), 3),
                ],
                "entity_types": [str(value) for value in row.get("entity_types", ())],
                "ui_regions": [str(value) for value in row.get("ui_regions", ())],
                "support_frame_ids": [
                    str(value) for value in tuple(row.get("support_frame_ids", ()) or ())
                ],
                "confidence": str(row.get("max_confidence", "") or ""),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["normalized_surface"]), []).append(row)
    packet = []
    for normalized_surface, rows in sorted(grouped.items()):
        digest = hashlib.sha256(normalized_surface.encode("utf-8")).hexdigest()[:16]
        ranges = _merge_time_ranges(
            tuple(tuple(value["local_time_range_sec"]) for value in rows)
        )
        packet.append(
            {
                "contract": WP17_OCR_AGGREGATION_CONTRACT,
                "evidence_id": f"ocragg:{segment_id}:{digest}",
                "surfaces": list(
                    dict.fromkeys(
                        surface
                        for row in rows
                        for surface in tuple(row.get("surfaces", ()) or ())
                    )
                ),
                "local_time_range_sec": [ranges[0][0], ranges[-1][1]],
                "occurrence_ranges_sec": [list(value) for value in ranges],
                "entity_types": sorted(
                    {
                        value
                        for row in rows
                        for value in tuple(row.get("entity_types", ()) or ())
                    }
                ),
                "ui_regions": sorted(
                    {
                        value
                        for row in rows
                        for value in tuple(row.get("ui_regions", ()) or ())
                    }
                ),
                "source_evidence_count": len(rows),
                "support_frame_count": len(
                    {
                        value
                        for row in rows
                        for value in tuple(row.get("support_frame_ids", ()) or ())
                    }
                ),
                "confidence": max(str(row.get("confidence", "") or "") for row in rows),
            }
        )
    return tuple(
        sorted(
            packet,
            key=lambda row: (
                float(row["local_time_range_sec"][0]),
                str(row["evidence_id"]),
            ),
        )
    )


def build_asr_packet(
    cues: Sequence[Mapping[str, Any]],
    *,
    segment_id: str,
    start_sec: float,
    end_sec: float,
) -> tuple[dict[str, Any], ...]:
    start = float(start_sec)
    end = float(end_sec)
    selected = []
    for raw in cues:
        row = dict(raw)
        cue_start = _number(row, "start", "virtual_start_sec", default=0.0)
        cue_end = _number(row, "end", "virtual_end_sec", default=cue_start)
        text = str(row.get("text", "") or "").strip()
        if not text or cue_start >= end or cue_end <= start:
            continue
        selected.append(
            {
                "evidence_id": f"asr:{segment_id}:{len(selected):04d}",
                "local_time_range_sec": [
                    round(max(0.0, cue_start - start), 3),
                    round(min(end - start, cue_end - start), 3),
                ],
                "text": text,
            }
        )
    return tuple(selected)


def frame_evidence_ids(segment_id: str, count: int) -> tuple[str, ...]:
    return tuple(f"frame:{segment_id}:{index:04d}" for index in range(1, int(count) + 1))


def alias_current_evidence(
    frame_ids: Sequence[str],
    ocr_packet: Sequence[Mapping[str, Any]],
    asr_packet: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[str, ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, str],
]:
    """Assign compact packet-local IDs while retaining a canonical ID map."""
    alias_map: dict[str, str] = {}
    aliased_frames = []
    for index, canonical_id in enumerate(frame_ids, 1):
        alias = f"f{index:03d}"
        alias_map[alias] = str(canonical_id)
        aliased_frames.append(alias)

    def alias_rows(
        prefix: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        aliased = []
        for index, raw in enumerate(rows, 1):
            row = dict(raw)
            canonical_id = str(row.pop("evidence_id"))
            alias = f"{prefix}{index:03d}"
            if alias in alias_map:
                raise ValueError(f"duplicate WP17 evidence alias: {alias}")
            alias_map[alias] = canonical_id
            row["evidence_id"] = alias
            aliased.append(row)
        return tuple(aliased)

    return (
        tuple(aliased_frames),
        alias_rows("o", ocr_packet),
        alias_rows("a", asr_packet),
        alias_map,
    )


def packet_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def construction_prompt(
    *,
    arm: str,
    segment_duration_sec: float,
    frame_ids: Sequence[str],
    ocr_packet: Sequence[Mapping[str, Any]],
    asr_packet: Sequence[Mapping[str, Any]],
    history_context: str,
    history_token_count: int,
    history_token_limit: int,
    repair_error: str = "",
) -> str:
    normalized_arm = str(arm).strip().casefold()
    if normalized_arm not in {"e1c0", "e1c1", "e1c2"}:
        raise ValueError(f"unknown WP17 slot arm: {arm}")
    slot_instruction = (
        "Return slot_operations=[] for this non-slot arm."
        if normalized_arm != "e1c2"
        else (
            "Maintain the working slots using only the allowed operations. Every slot currently present "
            "in history.slots must receive exactly one operation in this segment. Use expected_version "
            "from history.versions (0 when absent). write/update/close require observation_ids from this "
            "segment; each slot_operations.observation_ids entry must exactly copy an observation_id "
            "from the observations array, never a frame/OCR/ASR evidence ID. retain cannot rewrite a "
            "value but may cite current observations to refresh provenance and last verification; an "
            "evidence-free retain leaves the slot unchanged. archive/evict cannot rewrite values or "
            "attach observations. active_participants value must be "
            '{"event_ref":"<active encounter event_id>","participants":[...]}. '
            "active_encounter value must contain event_id. Do not put merely visible entities into "
            "active_participants. The slot capsule is sparse cross-segment working memory, not a copy "
            "of the current event record: do not instantiate every allowed slot, keep segment-local "
            "details only in structured_event_record, and use terse names/IDs/states as slot values. "
            "The serialized capsule must remain within 600 protocol tokens; target at most 400 tokens "
            "before runtime overhead."
        )
    )
    repair = "" if not repair_error else (
        "\nThe previous response was rejected by the deterministic validator. Return a complete JSON "
        "object with every required field (not a patch), preserve all array field types, and repair this "
        "error: "
        + str(repair_error)[:280]
    )
    schema = {
        "contract": WP17_SLOT_TRANSACTION_CONTRACT,
        "observations": [
            {
                "observation_id": "obs-unique",
                "kind": "entity|event|state|relation|visible_text|activity|location",
                "fact": "directly supported fact",
                "evidence_ids": ["one or more current evidence IDs"],
                "participants": [],
            }
        ],
        "slot_operations": [
            {
                "operation": "write|update|retain|close|archive|evict",
                "slot": "one allowed slot",
                "expected_version": 0,
                "value": "only for write/update",
                "observation_ids": ["required for write/update/close"],
            }
        ],
        "structured_event_record": {
            "entities": [],
            "events": [],
            "state_changes": [],
            "relations": [],
            "occurrence_refs": [],
            "summary": "detailed chronological 100-180 word caption grounded in this segment when evidence permits",
        },
    }
    packet = {
        "segment_duration_sec": float(segment_duration_sec),
        "frame_evidence_ids": list(frame_ids),
        "ocr_evidence": [dict(row) for row in ocr_packet],
        "asr_evidence": [dict(row) for row in asr_packet],
        "history": {
            "kind": (
                "none"
                if normalized_arm == "e1c0"
                else "previous_caption_tail"
                if normalized_arm == "e1c1"
                else "slot_capsule"
            ),
            "token_count": int(history_token_count),
            "token_limit": int(history_token_limit),
            "content": str(history_context),
        },
    }
    return (
        "Construct question-blind long-video memory from the attached chronological frames and evidence. "
        "The image label immediately before each image is its frame evidence ID and local timestamp. "
        "Report only facts supported by a current frame/OCR/ASR evidence ID. History may help resolve "
        "continuity but cannot support a new observation by itself. Do not infer the hidden evaluation "
        "question, answer, case identity, or official interval. Cite only the exact short evidence IDs "
        "listed in this packet (for example f001, o001, or a001); never invent or expand an ID. Keep at "
        f"most 16 observations, target at most {WP17_TARGET_OBSERVATION_EVIDENCE_IDS} directly useful "
        "evidence IDs per observation without dropping valid additional support, use at most 12 "
        "participants per observation, keep each "
        "structured_event_record list at most 12 concise entries, and keep the full JSON under 10000 "
        "characters. Preserve key events and state changes rather than transcribing every OCR row.\n"
        + slot_instruction
        + "\nAllowed slots: "
        + ", ".join(WP17_SLOT_NAMES)
        + ". Allowed operations: "
        + ", ".join(WP17_SLOT_OPERATIONS)
        + ".\nReturn exactly one JSON object matching this schema:\n"
        + _canonical_json(schema)
        + "\nCurrent packet:\n"
        + _canonical_json(packet)
        + repair
    )


def _number(row: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if isinstance(row.get(key), (int, float)):
            return float(row[key])
    return float(default)


def _merge_time_ranges(
    ranges: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted((float(value[0]), float(value[1])) for value in ranges)
    merged: list[list[float]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return tuple((round(left, 3), round(right, 3)) for left, right in merged)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
