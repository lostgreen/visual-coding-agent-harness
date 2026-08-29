from __future__ import annotations

import hashlib
import json
import math
import subprocess
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np

from vcah.virtual_video import VirtualVideoManifest, virtual_to_source_windows


WP17_LOCAL_TIMELINE_CONTRACT = "WP17-1-local-timeline-v1"
WP17_DENSE_OCR_CONTRACT = "WP17-1-paddleocr-dense-v1"


def build_local_timeline(
    spec: Mapping[str, Any],
    *,
    manifest: VirtualVideoManifest,
) -> dict[str, Any]:
    if spec.get("contract") != WP17_LOCAL_TIMELINE_CONTRACT:
        raise ValueError("WP17-1 timeline contract mismatch")
    if not bool(spec.get("protocol_frozen_before_dense_ocr_outcomes")):
        raise ValueError("WP17-1 timeline was not frozen before OCR outcomes")
    padding = float(spec.get("padding_sec", 0.0) or 0.0)
    fps = float(spec.get("sampling_fps", 0.0) or 0.0)
    if padding < 0.0 or fps <= 0.0:
        raise ValueError("WP17-1 padding/fps is invalid")
    duration = float(manifest.duration_sec)
    expanded = []
    for case_id, raw_case in sorted(dict(spec.get("cases", {}) or {}).items()):
        intervals = tuple(dict(raw_case).get("anchor_intervals", ()) or ())
        if not intervals:
            raise ValueError(f"WP17-1 case lacks anchor intervals: {case_id}")
        for value in intervals:
            if not isinstance(value, Sequence) or len(value) != 2:
                raise ValueError(f"invalid WP17-1 anchor interval: {case_id}")
            anchor_start, anchor_end = sorted((float(value[0]), float(value[1])))
            start = max(0.0, anchor_start - padding)
            end = min(duration, anchor_end + padding)
            if end <= start:
                raise ValueError(f"empty WP17-1 expanded interval: {case_id}")
            expanded.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "case_ids": [str(case_id)],
                }
            )
    windows = _merge_windows(expanded)
    timeline_rows = []
    covered_duration = 0.0
    for window_index, window in enumerate(windows):
        window_id = f"wp17_window_{window_index:04d}"
        mapped = virtual_to_source_windows(
            manifest,
            float(window["start_sec"]),
            float(window["end_sec"]),
        )
        for slice_index, source in enumerate(mapped):
            covered_duration += source.virtual_end_sec - source.virtual_start_sec
            timeline_rows.append(
                {
                    "schema_version": "MMLifelongWP17TimelineSliceV1",
                    "contract": WP17_LOCAL_TIMELINE_CONTRACT,
                    "window_id": window_id,
                    "slice_id": f"{window_id}:slice_{slice_index:04d}",
                    "case_ids": list(window["case_ids"]),
                    "segment_id": source.segment_id,
                    "source_video_id": source.source_video_id,
                    "virtual_start_sec": source.virtual_start_sec,
                    "virtual_end_sec": source.virtual_end_sec,
                    "source_start_sec": source.source_start_sec,
                    "source_end_sec": source.source_end_sec,
                }
            )
    scoped_duration = sum(
        float(row["end_sec"]) - float(row["start_sec"]) for row in windows
    )
    sample_points = sum(
        int(math.ceil((float(row["end_sec"]) - float(row["start_sec"])) * fps))
        for row in windows
    )
    serialized = json.dumps(timeline_rows, ensure_ascii=False, sort_keys=True)
    forbidden_keys = {"question", "options", "gold", "gold_answer", "source_path"}
    checks = {
        "development_only": spec.get("development_only") is True,
        "question_gold_blind": all(
            value is False
            for value in dict(spec.get("construction_inputs", {}) or {}).values()
        ),
        "scope_annotations_hidden_from_reader": spec.get(
            "scope_annotations_visible_to_reader"
        )
        is False,
        "nonempty_case_set": bool(spec.get("cases")),
        "nonempty_timeline": bool(timeline_rows),
        "complete_virtual_coverage": abs(covered_duration - scoped_duration) < 0.05,
        "source_paths_not_persisted": "source_path" not in serialized,
        "forbidden_fields_not_persisted": all(
            key not in serialized for key in forbidden_keys
        ),
    }
    checks["structural_gate_passed"] = all(checks.values())
    return {
        "schema_version": "MMLifelongWP17LocalTimelineV1",
        "contract": WP17_LOCAL_TIMELINE_CONTRACT,
        "windows": tuple(
            {
                "window_id": f"wp17_window_{index:04d}",
                "virtual_start_sec": round(float(row["start_sec"]), 3),
                "virtual_end_sec": round(float(row["end_sec"]), 3),
                "case_ids": list(row["case_ids"]),
            }
            for index, row in enumerate(windows)
        ),
        "timeline_slices": tuple(timeline_rows),
        "counts": {
            "cases": len(dict(spec.get("cases", {}) or {})),
            "expanded_anchor_windows": len(expanded),
            "merged_windows": len(windows),
            "timeline_slices": len(timeline_rows),
            "scoped_duration_sec": round(scoped_duration, 3),
            "expected_sample_points": sample_points,
        },
        "sampling_fps": fps,
        "frame_width": int(spec.get("frame_width", 1280)),
        "frame_height": int(spec.get("frame_height", 720)),
        "views": tuple(dict(row) for row in tuple(spec.get("views", ()) or ())),
        "gates": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
    }


def paddle_result_rows(
    result: Any,
    *,
    frame_label: str,
    view_id: str,
    ui_region: str,
    view_bbox_norm: Sequence[float],
    view_width: int,
    view_height: int,
    reader_source: str,
) -> tuple[dict[str, Any], ...]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("PaddleOCR result is not a mapping")
    body = payload.get("res", payload)
    if not isinstance(body, Mapping):
        raise ValueError("PaddleOCR result body is not a mapping")
    texts = _result_column(body, "rec_texts")
    scores = _result_column(body, "rec_scores")
    boxes = _result_column(body, "rec_boxes", fallback="rec_polys")
    if not (len(texts) == len(scores) == len(boxes)):
        raise ValueError("PaddleOCR result columns have different lengths")
    rows = []
    for index, (text, score, box) in enumerate(zip(texts, scores, boxes)):
        surface = str(text or "").strip()
        if not surface:
            continue
        numeric_score = float(score)
        rows.append(
            {
                "schema_version": "MMLifelongWP17OCRObservationV1",
                "contract": WP17_DENSE_OCR_CONTRACT,
                "frame_label": str(frame_label),
                "view_id": str(view_id),
                "text": surface,
                "entity_type": "screen_text",
                "ui_region": str(ui_region),
                "confidence": _confidence_label(numeric_score),
                "reader_score": round(numeric_score, 6),
                "reader_source": str(reader_source),
                "bbox": _map_view_box_to_frame(
                    box,
                    view_bbox_norm=view_bbox_norm,
                    view_width=int(view_width),
                    view_height=int(view_height),
                ),
                "reader_row_index": index,
            }
        )
    return tuple(rows)


def crop_normalized_view(
    frame: np.ndarray, bbox_norm: Sequence[float]
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("OCR frame must be an HxWx3 image")
    x1, y1, x2, y2 = _validated_normalized_box(bbox_norm)
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, int(math.floor(x1 * width))))
    top = max(0, min(height - 1, int(math.floor(y1 * height))))
    right = max(left + 1, min(width, int(math.ceil(x2 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 * height))))
    return np.ascontiguousarray(frame[top:bottom, left:right])


def iter_bgr_frames(
    *,
    source_path: str,
    source_start_sec: float,
    source_end_sec: float,
    fps: float,
    width: int,
    height: int,
    ffmpeg_executable: str = "ffmpeg",
) -> Iterator[np.ndarray]:
    duration = float(source_end_sec) - float(source_start_sec)
    if duration <= 0.0:
        return
    command = [
        str(ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(source_start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_path),
        "-vf",
        (
            f"fps={float(fps):.8f},"
            f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease,"
            f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2,format=bgr24"
        ),
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("ffmpeg OCR stream was not captured")
    frame_bytes = int(width) * int(height) * 3
    try:
        while True:
            payload = _read_exact(process.stdout, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError("truncated OCR raw frame")
            yield np.frombuffer(payload, dtype=np.uint8).reshape(
                (int(height), int(width), 3)
            )
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        tail = " | ".join(stderr.strip().splitlines()[-2:])
        raise RuntimeError(f"ffmpeg OCR scan failed: {tail}")


def stable_frame_label(slice_id: str, frame_index: int) -> str:
    digest = hashlib.sha256(
        f"{slice_id}:{int(frame_index)}".encode("utf-8")
    ).hexdigest()[:20]
    return f"wp17_frame_{digest}"


def _merge_windows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    for raw in sorted(rows, key=lambda row: (float(row["start_sec"]), float(row["end_sec"]))):
        row = {
            "start_sec": float(raw["start_sec"]),
            "end_sec": float(raw["end_sec"]),
            "case_ids": list(raw["case_ids"]),
        }
        if not merged or row["start_sec"] > merged[-1]["end_sec"]:
            merged.append(row)
            continue
        merged[-1]["end_sec"] = max(merged[-1]["end_sec"], row["end_sec"])
        merged[-1]["case_ids"] = sorted(
            set(merged[-1]["case_ids"]) | set(row["case_ids"])
        )
    return tuple(merged)


def _map_view_box_to_frame(
    box: Any,
    *,
    view_bbox_norm: Sequence[float],
    view_width: int,
    view_height: int,
) -> list[float]:
    values = np.asarray(box, dtype=float)
    if values.ndim == 2 and values.shape[1] == 2:
        local = [
            float(np.min(values[:, 0])) / float(view_width),
            float(np.min(values[:, 1])) / float(view_height),
            float(np.max(values[:, 0])) / float(view_width),
            float(np.max(values[:, 1])) / float(view_height),
        ]
    elif values.size == 4:
        x1, y1, x2, y2 = values.reshape(-1).tolist()
        local = [x1 / view_width, y1 / view_height, x2 / view_width, y2 / view_height]
    else:
        raise ValueError("unsupported PaddleOCR box shape")
    vx1, vy1, vx2, vy2 = _validated_normalized_box(view_bbox_norm)
    mapped = [
        vx1 + local[0] * (vx2 - vx1),
        vy1 + local[1] * (vy2 - vy1),
        vx1 + local[2] * (vx2 - vx1),
        vy1 + local[3] * (vy2 - vy1),
    ]
    return [round(max(0.0, min(1.0, value)), 6) for value in mapped]


def _validated_normalized_box(value: Sequence[float]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("normalized bbox must have four coordinates")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("normalized bbox is outside [0, 1]")
    return x1, y1, x2, y2


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _result_column(
    body: Mapping[str, Any], key: str, *, fallback: str | None = None
) -> tuple[Any, ...]:
    value = body.get(key)
    if value is None and fallback is not None:
        value = body.get(fallback)
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    return tuple(value)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)
