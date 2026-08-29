from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping, Sequence

from vcah.caption_lexical_index import normalize_caption_query
from vcah.occurrence_entity_sidecar import normalize_entity_text


WP17_OCR_TRACK_CONTRACT = "WP17-1-dense-ocr-track-v1"
WP17_EVIDENCE_STORE_CONTRACT = "WP17-1-construction-evidence-store-v1"
DEFAULT_TRACK_MAX_GAP_SEC = 3.0
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def build_ocr_tracks(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_metadata: Mapping[str, Mapping[str, Any]],
    max_gap_sec: float = DEFAULT_TRACK_MAX_GAP_SEC,
    default_reader_source: str = "gemini_ocr",
) -> dict[str, Any]:
    gap = float(max_gap_sec)
    if gap < 0.0:
        raise ValueError("max_gap_sec cannot be negative")
    reader_source = str(default_reader_source or "").strip()
    if not reader_source:
        raise ValueError("default_reader_source cannot be empty")

    observations = []
    blank_count = 0
    for input_index, raw in enumerate(rows):
        text = normalize_entity_text(raw.get("text", ""))
        if not text:
            blank_count += 1
            continue
        label = str(raw.get("frame_label", "") or "").strip()
        metadata = frame_metadata.get(label)
        if not label or not isinstance(metadata, Mapping):
            raise ValueError("every OCR observation requires frame lineage")
        for field in ("virtual_time_sec", "source_time_sec"):
            if not isinstance(metadata.get(field), (int, float)):
                raise ValueError(f"frame lineage is missing numeric {field}: {label}")
        segment_id = str(metadata.get("segment_id", "") or "").strip()
        source_video_id = str(metadata.get("source_video_id", "") or "").strip()
        if not segment_id or not source_video_id:
            raise ValueError(f"frame lineage is incomplete: {label}")
        normalized = normalize_caption_query(text) or text.casefold()
        ui_region = str(raw.get("ui_region", "other") or "other").strip().casefold()
        entity_type = str(
            raw.get("entity_type", raw.get("type", "other_named_entity"))
            or "other_named_entity"
        ).strip().casefold()
        confidence = str(raw.get("confidence", "low") or "low").strip().casefold()
        if confidence not in _CONFIDENCE_ORDER:
            confidence = "low"
        frame_id = str(metadata.get("frame_id", label) or label)
        observation_seed = {
            "input_index": input_index,
            "frame_label": label,
            "text": text,
            "entity_type": entity_type,
            "ui_region": ui_region,
        }
        observations.append(
            {
                "observation_id": "ocr_obs_"
                + _stable_hash(observation_seed, length=20),
                "input_index": input_index,
                "frame_label": label,
                "frame_id": frame_id,
                "segment_id": segment_id,
                "source_video_id": source_video_id,
                "virtual_time_sec": round(float(metadata["virtual_time_sec"]), 3),
                "source_time_sec": round(float(metadata["source_time_sec"]), 3),
                "surface": text,
                "normalized_surface": normalized,
                "entity_type": entity_type,
                "ui_region": ui_region,
                "confidence": confidence,
                "reader_source": str(
                    raw.get("reader_source", reader_source) or reader_source
                ).strip(),
                "bbox": _normalized_bbox(
                    raw.get("bbox", raw.get("bbox_norm", raw.get("box")))
                ),
            }
        )

    ordered = sorted(
        observations,
        key=lambda row: (
            float(row["virtual_time_sec"]),
            str(row["source_video_id"]),
            str(row["frame_label"]),
            str(row["normalized_surface"]),
            str(row["ui_region"]),
            str(row["observation_id"]),
        ),
    )
    observation_ids = [str(row["observation_id"]) for row in ordered]
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("OCR observations contain duplicate stable identities")

    active: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    clusters: list[list[dict[str, Any]]] = []
    for row in ordered:
        key = (
            str(row["source_video_id"]),
            str(row["normalized_surface"]),
            str(row["ui_region"]),
        )
        current = active.get(key)
        if current is None or (
            float(row["virtual_time_sec"])
            - float(current[-1]["virtual_time_sec"])
            > gap
        ):
            current = []
            active[key] = current
            clusters.append(current)
        current.append(row)

    tracks = tuple(
        sorted(
            (_finalize_track(cluster, max_gap_sec=gap) for cluster in clusters),
            key=lambda row: (
                float(row["start_sec"]),
                str(row["source_video_id"]),
                str(row["normalized_surface"]),
                str(row["track_id"]),
            ),
        )
    )
    assigned = [
        observation_id
        for track in tracks
        for observation_id in track["observation_ids"]
    ]
    gates = {
        "all_input_rows_accounted": len(rows) == len(assigned) + blank_count,
        "all_nonblank_rows_assigned_once": Counter(assigned) == Counter(observation_ids),
        "unique_observation_ids": len(set(observation_ids)) == len(observation_ids),
        "unique_track_ids": len({row["track_id"] for row in tracks}) == len(tracks),
        "complete_frame_lineage": all(row["lineage_complete"] for row in tracks),
        "ordered_track_ranges": all(
            float(row["end_sec"]) >= float(row["start_sec"]) for row in tracks
        ),
        "raw_surfaces_preserved": all(row["surfaces"] for row in tracks),
        "source_paths_not_persisted": "source_path"
        not in json.dumps(tracks, ensure_ascii=False),
    }
    gates["structural_gate_passed"] = all(gates.values())
    normalized_track_counts = Counter(row["normalized_surface"] for row in tracks)
    return {
        "schema_version": "MMLifelongWP17OCRTrackBuildV1",
        "contract": WP17_OCR_TRACK_CONTRACT,
        "tracks": tracks,
        "counts": {
            "input_rows": len(rows),
            "blank_rows": blank_count,
            "assigned_observations": len(assigned),
            "tracks": len(tracks),
            "multi_frame_tracks": sum(row["support_frame_count"] > 1 for row in tracks),
            "singleton_tracks": sum(row["support_frame_count"] == 1 for row in tracks),
            "normalized_surfaces": len(normalized_track_counts),
            "surfaces_with_multiple_tracks": sum(
                count > 1 for count in normalized_track_counts.values()
            ),
        },
        "gates": gates,
        "structural_gate_passed": gates["structural_gate_passed"],
        "model_calls": 0,
        "admission_filter_applied": False,
    }


def build_track_evidence(
    tracks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    evidence = []
    for raw in tracks:
        track = dict(raw)
        if track.get("contract") != WP17_OCR_TRACK_CONTRACT:
            raise ValueError("unexpected OCR track contract")
        track_id = str(track.get("track_id", "") or "")
        if not track_id or not track.get("lineage_complete"):
            raise ValueError("OCR track is missing validated lineage")
        evidence.append(
            {
                "schema_version": "MMLifelongWP17EvidenceV1",
                "contract": WP17_EVIDENCE_STORE_CONTRACT,
                "evidence_id": f"ocr:{track_id}",
                "kind": "ocr_track",
                "track_id": track_id,
                "start_sec": float(track["start_sec"]),
                "end_sec": float(track["end_sec"]),
                "source_video_id": str(track["source_video_id"]),
                "segment_ids": list(track["segment_ids"]),
                "surface": str(track["canonical_surface"]),
                "surfaces": [dict(value) for value in track["surfaces"]],
                "normalized_surface": str(track["normalized_surface"]),
                "entity_types": list(track["entity_types"]),
                "ui_regions": list(track["ui_regions"]),
                "support_frame_ids": list(track["support_frame_ids"]),
                "observation_ids": list(track["observation_ids"]),
                "reader_sources": list(track["reader_sources"]),
                "max_confidence": str(track["max_confidence"]),
                "lineage_complete": True,
                "source_contract": WP17_OCR_TRACK_CONTRACT,
            }
        )
    evidence_ids = [str(row["evidence_id"]) for row in evidence]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("duplicate WP17 evidence IDs")
    if "source_path" in json.dumps(evidence, ensure_ascii=False):
        raise ValueError("source paths cannot enter the WP17 evidence store")
    return tuple(evidence)


def _finalize_track(
    observations: Sequence[Mapping[str, Any]], *, max_gap_sec: float
) -> dict[str, Any]:
    if not observations:
        raise ValueError("cannot finalize an empty OCR track")
    rows = tuple(dict(row) for row in observations)
    surface_counts = Counter(str(row["surface"]) for row in rows)
    first_seen = {
        surface: min(
            index for index, row in enumerate(rows) if str(row["surface"]) == surface
        )
        for surface in surface_counts
    }
    confidence_by_surface = {
        surface: max(
            _CONFIDENCE_ORDER[str(row["confidence"])]
            for row in rows
            if str(row["surface"]) == surface
        )
        for surface in surface_counts
    }
    canonical = min(
        surface_counts,
        key=lambda surface: (
            -surface_counts[surface],
            -confidence_by_surface[surface],
            first_seen[surface],
            surface,
        ),
    )
    frame_ids = tuple(dict.fromkeys(str(row["frame_id"]) for row in rows))
    segment_ids = tuple(dict.fromkeys(str(row["segment_id"]) for row in rows))
    start = min(float(row["virtual_time_sec"]) for row in rows)
    end = max(float(row["virtual_time_sec"]) for row in rows)
    seed = {
        "source_video_id": str(rows[0]["source_video_id"]),
        "normalized_surface": str(rows[0]["normalized_surface"]),
        "ui_region": str(rows[0]["ui_region"]),
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "observation_ids": [str(row["observation_id"]) for row in rows],
    }
    bbox_series = [
        {
            "observation_id": str(row["observation_id"]),
            "frame_id": str(row["frame_id"]),
            "bbox": list(row["bbox"]),
        }
        for row in rows
        if row.get("bbox") is not None
    ]
    return {
        "schema_version": "MMLifelongWP17OCRTrackV1",
        "contract": WP17_OCR_TRACK_CONTRACT,
        "track_id": "ocr_track_" + _stable_hash(seed, length=20),
        "canonical_surface": canonical,
        "surfaces": [
            {"surface": surface, "count": surface_counts[surface]}
            for surface in sorted(surface_counts, key=lambda value: first_seen[value])
        ],
        "normalized_surface": str(rows[0]["normalized_surface"]),
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "source_start_sec": round(min(float(row["source_time_sec"]) for row in rows), 3),
        "source_end_sec": round(max(float(row["source_time_sec"]) for row in rows), 3),
        "source_video_id": str(rows[0]["source_video_id"]),
        "segment_ids": list(segment_ids),
        "entity_types": sorted({str(row["entity_type"]) for row in rows}),
        "ui_regions": sorted({str(row["ui_region"]) for row in rows}),
        "support_frame_ids": list(frame_ids),
        "support_frame_count": len(frame_ids),
        "observation_ids": [str(row["observation_id"]) for row in rows],
        "reader_sources": sorted({str(row["reader_source"]) for row in rows}),
        "max_confidence": max(
            (str(row["confidence"]) for row in rows),
            key=lambda value: (_CONFIDENCE_ORDER[value], value),
        ),
        "bbox_series": bbox_series,
        "lineage_complete": True,
        "track_max_gap_sec": float(max_gap_sec),
    }


def _normalized_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4 or not all(isinstance(item, (int, float)) for item in value):
        return None
    left, top, right, bottom = (float(item) for item in value)
    if not (0.0 <= left <= right <= 1.0 and 0.0 <= top <= bottom <= 1.0):
        return None
    return (left, top, right, bottom)


def _stable_hash(payload: Mapping[str, Any], *, length: int) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[: int(length)]
