from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np

from vcah.caption_lexical_index import normalize_caption_query
from vcah.occurrence_entity_sidecar import (
    admit_global_entity_rows,
    normalize_entity_text,
)
from vcah.virtual_video import VirtualVideoSegment


CHANGE_TRIGGERED_ENTITY_CONTRACT = "WP16-7-change-triggered-entity-occurrence-v1"
DEFAULT_TIER0_FPS = 1.0
DEFAULT_TIER0_WIDTH = 160
DEFAULT_TIER0_HEIGHT = 90
DEFAULT_COVERAGE_BIN_SEC = 300.0
DEFAULT_MIN_SPACING_SEC = 2.0
DEFAULT_OCCURRENCE_GAP_SEC = 60.0


def scan_segment_change_observations(
    segment: VirtualVideoSegment,
    *,
    fps: float = DEFAULT_TIER0_FPS,
    width: int = DEFAULT_TIER0_WIDTH,
    height: int = DEFAULT_TIER0_HEIGHT,
    ffmpeg_executable: str = "ffmpeg",
) -> tuple[dict[str, Any], ...]:
    """Stream low-resolution frames from ffmpeg without materializing them."""
    rate = float(fps)
    if rate <= 0.0:
        raise ValueError("fps must be positive")
    frame_width = int(width)
    frame_height = int(height)
    if frame_width < 16 or frame_height < 16:
        raise ValueError("Tier-0 dimensions must be at least 16x16")
    duration = float(segment.source_end_sec) - float(segment.source_start_sec)
    if duration <= 0.0:
        return ()
    command = [
        str(ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(segment.source_start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(segment.source_path),
        "-vf",
        (
            f"fps={rate:.8f},scale={frame_width}:{frame_height}:"
            "flags=fast_bilinear,format=gray"
        ),
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg executable is unavailable: {ffmpeg_executable}"
        ) from exc
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("ffmpeg Tier-0 stream was not captured")
    try:
        rows = _read_change_stream(
            process.stdout,
            segment=segment,
            fps=rate,
            width=frame_width,
            height=frame_height,
        )
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        tail = " | ".join(stderr.strip().splitlines()[-2:])
        raise RuntimeError(f"ffmpeg Tier-0 scan failed: {tail}")
    return rows


def _read_change_stream(
    stream: BinaryIO,
    *,
    segment: VirtualVideoSegment,
    fps: float,
    width: int,
    height: int,
) -> tuple[dict[str, Any], ...]:
    frame_bytes = int(width) * int(height)
    previous: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        payload = _read_exact(stream, frame_bytes)
        if not payload:
            break
        if len(payload) != frame_bytes:
            raise RuntimeError("truncated Tier-0 raw frame")
        frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
        global_score, ui_score = frame_change_scores(previous, frame)
        offset_sec = frame_index / float(fps)
        source_time = min(
            float(segment.source_end_sec) - 0.001,
            float(segment.source_start_sec) + offset_sec,
        )
        virtual_time = min(
            float(segment.virtual_end_sec) - 0.001,
            float(segment.virtual_start_sec) + offset_sec,
        )
        rows.append(
            {
                "schema_version": "MMLifelongTier0ChangeObservationV1",
                "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
                "segment_id": segment.segment_id,
                "source_video_id": segment.source_video_id,
                "source_path": segment.source_path,
                "tier0_frame_index": frame_index,
                "source_time_sec": round(source_time, 3),
                "virtual_time_sec": round(virtual_time, 3),
                "global_change_score": round(global_score, 8),
                "ui_change_score": round(ui_score, 8),
                "selection_score": round(0.65 * global_score + 0.35 * ui_score, 8),
            }
        )
        previous = frame.copy()
        frame_index += 1
    return tuple(rows)


def frame_change_scores(
    previous: np.ndarray | None,
    current: np.ndarray,
) -> tuple[float, float]:
    image = np.asarray(current, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("Tier-0 change frames must be grayscale")
    if previous is None:
        return 0.0, 0.0
    prior = np.asarray(previous, dtype=np.uint8)
    if prior.shape != image.shape:
        raise ValueError("Tier-0 frame shapes do not match")
    delta = np.abs(image.astype(np.int16) - prior.astype(np.int16)) / 255.0
    height, width = image.shape
    top_end = max(1, int(round(height * 0.28)))
    bottom_start = min(height - 1, int(round(height * 0.68)))
    side_width = max(1, int(round(width * 0.18)))
    ui_values = np.concatenate(
        (
            delta[:top_end, :].ravel(),
            delta[bottom_start:, :].ravel(),
            delta[top_end:bottom_start, :side_width].ravel(),
            delta[top_end:bottom_start, width - side_width :].ravel(),
        )
    )
    return float(np.mean(delta)), float(np.mean(ui_values))


def select_uniform_budget(
    observations: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> tuple[dict[str, Any], ...]:
    ordered = _ordered_observations(observations)
    limit = _validate_budget(budget, len(ordered))
    selected = []
    for index in range(limit):
        source_index = min(
            len(ordered) - 1,
            int(math.floor((index + 0.5) * len(ordered) / limit)),
        )
        selected.append(
            {
                **ordered[source_index],
                "selection_arm": "a1_uniform",
                "selection_reason": "budget_matched_uniform",
            }
        )
    if len({row_identity(row) for row in selected}) != limit:
        raise RuntimeError("uniform budget selection produced duplicate frames")
    return tuple(selected)


def select_change_budget(
    observations: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    coverage_bin_sec: float = DEFAULT_COVERAGE_BIN_SEC,
    min_spacing_sec: float = DEFAULT_MIN_SPACING_SEC,
) -> tuple[dict[str, Any], ...]:
    ordered = _ordered_observations(observations)
    limit = _validate_budget(budget, len(ordered))
    bin_size = float(coverage_bin_sec)
    spacing = max(0.0, float(min_spacing_sec))
    if bin_size <= 0.0:
        raise ValueError("coverage_bin_sec must be positive")
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for row in ordered:
        bin_index = int(math.floor(float(row["virtual_time_sec"]) / bin_size))
        by_bin.setdefault(bin_index, []).append(row)
    bin_winners = tuple(
        _ranked_change_rows(rows)[0] for _, rows in sorted(by_bin.items())
    )
    if len(bin_winners) > limit:
        bin_winners = select_uniform_budget(bin_winners, budget=limit)
    selected: list[dict[str, Any]] = [
        {
            **dict(row),
            "selection_arm": "a2_change",
            "selection_reason": "coverage_bin_peak",
        }
        for row in bin_winners
    ]
    selected_ids = {row_identity(row) for row in selected}
    selected_times: dict[str, list[float]] = {}
    for row in selected:
        selected_times.setdefault(str(row["segment_id"]), []).append(
            float(row["virtual_time_sec"])
        )
    ranked = _ranked_change_rows(ordered)
    for raw in ranked:
        if len(selected) >= limit:
            break
        identity = row_identity(raw)
        if identity in selected_ids:
            continue
        segment_times = selected_times.setdefault(str(raw["segment_id"]), [])
        if any(
            abs(float(raw["virtual_time_sec"]) - value) < spacing
            for value in segment_times
        ):
            continue
        selected.append(
            {
                **raw,
                "selection_arm": "a2_change",
                "selection_reason": "ranked_change_peak",
            }
        )
        selected_ids.add(identity)
        segment_times.append(float(raw["virtual_time_sec"]))
    for raw in ranked:
        if len(selected) >= limit:
            break
        identity = row_identity(raw)
        if identity in selected_ids:
            continue
        selected.append(
            {
                **raw,
                "selection_arm": "a2_change",
                "selection_reason": "exact_budget_spacing_fallback",
            }
        )
        selected_ids.add(identity)
    if len(selected) != limit:
        raise RuntimeError("change budget selection did not fill the exact budget")
    return tuple(
        sorted(
            selected,
            key=lambda row: (
                float(row["virtual_time_sec"]),
                str(row["segment_id"]),
                int(row["tier0_frame_index"]),
            ),
        )
    )


def admit_entity_occurrences(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_metadata: Mapping[str, Mapping[str, Any]],
    merge_gap_sec: float = DEFAULT_OCCURRENCE_GAP_SEC,
) -> dict[str, Any]:
    gap = float(merge_gap_sec)
    if gap < 0.0:
        raise ValueError("merge_gap_sec cannot be negative")
    grouped: dict[str, list[dict[str, Any]]] = {}
    rejection_counts: Counter[str] = Counter()
    for raw in rows:
        label = str(raw.get("frame_label", "") or "").strip()
        metadata = frame_metadata.get(label)
        if metadata is None or not isinstance(
            metadata.get("virtual_time_sec"), (int, float)
        ):
            rejection_counts["missing_frame_lineage"] += 1
            continue
        text = normalize_entity_text(raw.get("text", ""))
        normalized = normalize_caption_query(text)
        if not normalized:
            rejection_counts["blank"] += 1
            continue
        grouped.setdefault(normalized, []).append({**dict(raw), "text": text})

    occurrences: list[dict[str, Any]] = []
    for normalized, candidates in sorted(grouped.items()):
        ordered = sorted(
            candidates,
            key=lambda row: (
                float(frame_metadata[str(row["frame_label"])]["virtual_time_sec"]),
                str(row["frame_label"]),
            ),
        )
        clusters: list[list[dict[str, Any]]] = []
        for row in ordered:
            timestamp = float(
                frame_metadata[str(row["frame_label"])]["virtual_time_sec"]
            )
            if not clusters:
                clusters.append([row])
                continue
            prior_time = float(
                frame_metadata[str(clusters[-1][-1]["frame_label"])]["virtual_time_sec"]
            )
            if timestamp - prior_time > gap:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for cluster in clusters:
            times = tuple(
                float(frame_metadata[str(row["frame_label"])]["virtual_time_sec"])
                for row in cluster
            )
            occurrence_seed = {
                "normalized": normalized,
                "start": round(min(times), 3),
                "end": round(max(times), 3),
            }
            occurrence_id = (
                "entity_occ_"
                + hashlib.sha256(
                    json.dumps(occurrence_seed, sort_keys=True).encode("utf-8")
                ).hexdigest()[:16]
            )
            admission = admit_global_entity_rows(
                cluster,
                passage_id=occurrence_id,
                frame_metadata=frame_metadata,
            )
            rejection_counts.update(admission["rejection_counts"])
            for admitted in admission["admitted_rows"]:
                occurrences.append(
                    {
                        **dict(admitted),
                        "contract": CHANGE_TRIGGERED_ENTITY_CONTRACT,
                        "occurrence_id": occurrence_id,
                        "occurrence_start_sec": round(min(times), 3),
                        "occurrence_end_sec": round(max(times), 3),
                        "merge_gap_sec": gap,
                    }
                )
    return {
        "occurrences": tuple(
            sorted(
                occurrences,
                key=lambda row: (
                    float(row["occurrence_start_sec"]),
                    str(row["normalized"]),
                ),
            )
        ),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "candidate_unique_text_count": len(grouped),
    }


def row_identity(row: Mapping[str, Any]) -> str:
    return (
        f"{str(row.get('segment_id', ''))}:" f"{int(row.get('tier0_frame_index', -1))}"
    )


def _ranked_change_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        sorted(
            (dict(row) for row in rows),
            key=lambda row: (
                -float(row.get("selection_score", 0.0) or 0.0),
                float(row["virtual_time_sec"]),
                str(row["segment_id"]),
                int(row["tier0_frame_index"]),
            ),
        )
    )


def _ordered_observations(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    ordered = tuple(
        sorted(
            (dict(row) for row in rows),
            key=lambda row: (
                float(row["virtual_time_sec"]),
                str(row["segment_id"]),
                int(row["tier0_frame_index"]),
            ),
        )
    )
    identities = tuple(row_identity(row) for row in ordered)
    if len(set(identities)) != len(identities):
        raise ValueError("Tier-0 observations contain duplicate frame identities")
    return ordered


def _validate_budget(budget: int, available: int) -> int:
    limit = int(budget)
    if limit < 1 or limit > int(available):
        raise ValueError("selection budget is outside available Tier-0 frames")
    return limit


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < int(size):
        payload = stream.read(int(size) - len(chunks))
        if not payload:
            break
        chunks.extend(payload)
    return bytes(chunks)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
