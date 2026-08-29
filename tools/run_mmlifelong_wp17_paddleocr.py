#!/usr/bin/env python3
"""Stream the frozen WP17 timeline through local PaddleOCR without frame files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.virtual_video import VirtualVideoWorkspace
from vcah.wp17_dense_ocr import (
    WP17_DENSE_OCR_CONTRACT,
    WP17_LOCAL_TIMELINE_CONTRACT,
    crop_normalized_view,
    iter_bgr_frames,
    paddle_result_rows,
    stable_frame_label,
)
from vcah.wp17_memory_construction import build_ocr_tracks, build_track_evidence


def run(args: argparse.Namespace) -> Path:
    timeline_path = Path(args.timeline_manifest)
    timeline = _read_json(timeline_path)
    if timeline.get("contract") != WP17_LOCAL_TIMELINE_CONTRACT:
        raise ValueError("WP17 timeline contract mismatch")
    if not bool(timeline.get("structural_gate_passed")):
        raise ValueError("WP17 timeline structural gate did not pass")
    if file_sha256(timeline_path) != str(args.expected_timeline_sha256):
        raise ValueError("WP17 timeline SHA mismatch")
    workspace = VirtualVideoWorkspace.load(Path(args.workspace_root))
    segments = {row.segment_id: row for row in workspace.manifest.segments}
    slices = _clip_slices(
        tuple(timeline.get("timeline_slices", ()) or ()),
        virtual_start_sec=args.virtual_start_sec,
        virtual_end_sec=args.virtual_end_sec,
    )
    if not slices:
        raise ValueError("WP17 OCR scope contains no timeline slices")
    unknown = sorted({str(row["segment_id"]) for row in slices} - set(segments))
    if unknown:
        raise ValueError(f"WP17 OCR timeline has unknown segments: {unknown}")

    try:
        import paddle
        import paddleocr
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("PaddleOCR runtime is unavailable") from exc

    reader = PaddleOCR(
        lang=str(args.lang),
        device=str(args.device),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    reader_source = f"paddleocr-{getattr(paddleocr, '__version__', 'unknown')}"
    out_root = Path(args.out_root)
    shard_root = out_root / "observation_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = out_root / "run_manifest.json"
    report_path = out_root / "wp17_dense_ocr_report.json"
    if report_path.exists():
        raise FileExistsError("WP17 dense OCR report already exists")
    if run_manifest_path.exists() and not args.resume:
        raise FileExistsError("WP17 dense OCR output already exists")
    run_manifest = {
        "schema_version": "MMLifelongWP17DenseOCRRunV1",
        "contract": WP17_DENSE_OCR_CONTRACT,
        "source_commit": str(args.source_commit),
        "timeline_sha256": file_sha256(timeline_path),
        "workspace_id": workspace.workspace_id,
        "reader_source": reader_source,
        "paddle_version": str(getattr(paddle, "__version__", "unknown")),
        "device": str(args.device),
        "sampling_fps": float(timeline["sampling_fps"]),
        "frame_width": int(timeline["frame_width"]),
        "frame_height": int(timeline["frame_height"]),
        "views": tuple(dict(row) for row in timeline.get("views", ())),
        "scope_slice_count": len(slices),
        "canary_scope_override": args.virtual_start_sec is not None
        or args.virtual_end_sec is not None
        or int(args.max_frames) > 0,
        "question_visible_to_reader": False,
        "options_visible_to_reader": False,
        "answer_visible_to_reader": False,
        "target_entity_aliases_visible_to_reader": False,
        "official_intervals_visible_to_reader": False,
        "caption_text_visible_to_reader": False,
        "raw_frames_persisted": False,
        "source_paths_persisted": False,
        "gemini_calls": 0,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    _write_json(run_manifest_path, run_manifest)

    frame_metadata: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    reader_calls = 0
    processed_frames = 0
    expected_frames = _expected_frames(
        slices,
        fps=float(timeline["sampling_fps"]),
        max_frames=int(args.max_frames),
    )
    for slice_index, raw_slice in enumerate(slices):
        if int(args.max_frames) > 0 and processed_frames >= int(args.max_frames):
            break
        row = dict(raw_slice)
        shard_key = _safe_name(str(row["slice_id"]))
        shard_path = shard_root / f"{shard_key}.jsonl"
        metadata_path = shard_root / f"{shard_key}.meta.json"
        if args.resume and shard_path.is_file() and metadata_path.is_file():
            shard_rows = _read_jsonl(shard_path)
            shard_metadata = _read_json(metadata_path)
            observations.extend(shard_rows)
            for frame in tuple(shard_metadata.get("frames", ()) or ()):
                frame_metadata[str(frame["frame_label"])] = dict(frame)
            processed_frames += int(shard_metadata["frame_count"])
            reader_calls += int(shard_metadata["reader_calls"])
            print(
                "WP17_OCR_SLICE_DONE "
                f"completed={slice_index + 1}/{len(slices)} "
                f"frames={processed_frames}/{expected_frames} reused=true",
                flush=True,
            )
            continue
        if shard_path.exists() or metadata_path.exists():
            raise RuntimeError(f"incomplete WP17 OCR shard exists: {shard_key}")
        segment = segments[str(row["segment_id"])]
        remaining = (
            int(args.max_frames) - processed_frames
            if int(args.max_frames) > 0
            else 0
        )
        slice_rows: list[dict[str, Any]] = []
        slice_frames: list[dict[str, Any]] = []
        slice_calls = 0
        for frame_index, frame in enumerate(
            iter_bgr_frames(
                source_path=segment.source_path,
                source_start_sec=float(row["source_start_sec"]),
                source_end_sec=float(row["source_end_sec"]),
                fps=float(timeline["sampling_fps"]),
                width=int(timeline["frame_width"]),
                height=int(timeline["frame_height"]),
                ffmpeg_executable=str(args.ffmpeg_executable),
            )
        ):
            if remaining > 0 and frame_index >= remaining:
                break
            label = stable_frame_label(str(row["slice_id"]), frame_index)
            offset = frame_index / float(timeline["sampling_fps"])
            metadata = {
                "frame_label": label,
                "frame_id": label,
                "segment_id": str(row["segment_id"]),
                "source_video_id": str(row["source_video_id"]),
                "virtual_time_sec": round(
                    float(row["virtual_start_sec"]) + offset, 3
                ),
                "source_time_sec": round(float(row["source_start_sec"]) + offset, 3),
            }
            frame_metadata[label] = metadata
            slice_frames.append(metadata)
            views = []
            view_specs = []
            for raw_view in tuple(timeline.get("views", ()) or ()):
                view = dict(raw_view)
                crop = crop_normalized_view(frame, view["bbox_norm"])
                views.append(crop)
                view_specs.append(view)
            results = tuple(reader.predict(input=views))
            if len(results) != len(views):
                raise RuntimeError("PaddleOCR did not return one result per view")
            for view, crop, result in zip(view_specs, views, results):
                rows = paddle_result_rows(
                    result,
                    frame_label=label,
                    view_id=str(view["view_id"]),
                    ui_region=str(view["ui_region"]),
                    view_bbox_norm=view["bbox_norm"],
                    view_width=int(crop.shape[1]),
                    view_height=int(crop.shape[0]),
                    reader_source=reader_source,
                )
                slice_rows.extend(rows)
                slice_calls += 1
        _write_jsonl(shard_path, slice_rows)
        _write_json(
            metadata_path,
            {
                "schema_version": "MMLifelongWP17OCRShardMetaV1",
                "contract": WP17_DENSE_OCR_CONTRACT,
                "slice_id": str(row["slice_id"]),
                "frame_count": len(slice_frames),
                "reader_calls": slice_calls,
                "observation_count": len(slice_rows),
                "frames": slice_frames,
                "source_path_persisted": False,
            },
        )
        observations.extend(slice_rows)
        processed_frames += len(slice_frames)
        reader_calls += slice_calls
        print(
            "WP17_OCR_SLICE_DONE "
            f"completed={slice_index + 1}/{len(slices)} "
            f"frames={processed_frames}/{expected_frames} reused=false",
            flush=True,
        )

    track_build = build_ocr_tracks(
        observations,
        frame_metadata=frame_metadata,
        max_gap_sec=float(args.track_max_gap_sec),
        default_reader_source=reader_source,
    )
    tracks = tuple(track_build["tracks"])
    evidence = build_track_evidence(tracks)
    _write_jsonl(out_root / "ocr_tracks.jsonl", tracks)
    _write_jsonl(out_root / "evidence_store.jsonl", evidence)
    serialized = json.dumps(
        {"observations": observations, "frames": frame_metadata, "tracks": tracks},
        ensure_ascii=False,
    )
    gates = {
        "timeline_structural_gate_passed": bool(timeline["structural_gate_passed"]),
        "expected_frame_count_exact": processed_frames == expected_frames,
        "frame_lineage_unique": len(frame_metadata) == processed_frames,
        "track_build_structural_gate_passed": bool(
            track_build["structural_gate_passed"]
        ),
        "evidence_count_equals_track_count": len(evidence) == len(tracks),
        "question_gold_blind": all(
            run_manifest[key] is False
            for key in (
                "question_visible_to_reader",
                "options_visible_to_reader",
                "answer_visible_to_reader",
                "target_entity_aliases_visible_to_reader",
                "official_intervals_visible_to_reader",
                "caption_text_visible_to_reader",
            )
        ),
        "zero_frame_persistence": run_manifest["raw_frames_persisted"] is False,
        "zero_source_path_persistence": "source_path" not in serialized,
        "zero_gemini_calls": run_manifest["gemini_calls"] == 0,
    }
    gates["structural_gate_passed"] = all(gates.values())
    report = {
        "schema_version": "MMLifelongWP17DenseOCRReportV1",
        "contract": WP17_DENSE_OCR_CONTRACT,
        "decision": (
            "WP17_1_DENSE_OCR_READY"
            if gates["structural_gate_passed"]
            else "STRUCTURAL_FAILURE"
        ),
        "counts": {
            "frames": processed_frames,
            "views_per_frame": len(tuple(timeline.get("views", ()) or ())),
            "reader_calls": reader_calls,
            "observations": len(observations),
            "tracks": len(tracks),
            "evidence_records": len(evidence),
        },
        "reader": {
            "source": reader_source,
            "paddle_version": run_manifest["paddle_version"],
            "device": str(args.device),
        },
        "track_counts": dict(track_build["counts"]),
        "gates": gates,
        "structural_gate_passed": gates["structural_gate_passed"],
        "canary_scope_override": run_manifest["canary_scope_override"],
        "model_calls": 0,
    }
    _write_json(report_path, report)
    (out_root / "wp17_dense_ocr_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(
        "WP17_DENSE_OCR_DONE "
        f"decision={report['decision']} frames={processed_frames} "
        f"observations={len(observations)} tracks={len(tracks)} gate="
        f"{str(gates['structural_gate_passed']).lower()}",
        flush=True,
    )
    return report_path


def _clip_slices(
    rows: Sequence[Mapping[str, Any]],
    *,
    virtual_start_sec: float | None,
    virtual_end_sec: float | None,
) -> tuple[dict[str, Any], ...]:
    lower = -math.inf if virtual_start_sec is None else float(virtual_start_sec)
    upper = math.inf if virtual_end_sec is None else float(virtual_end_sec)
    if upper <= lower:
        raise ValueError("WP17 OCR virtual scope is empty")
    clipped = []
    for raw in rows:
        row = dict(raw)
        start = max(float(row["virtual_start_sec"]), lower)
        end = min(float(row["virtual_end_sec"]), upper)
        if end <= start:
            continue
        source_offset = start - float(row["virtual_start_sec"])
        source_end_offset = end - float(row["virtual_start_sec"])
        row["virtual_start_sec"] = round(start, 3)
        row["virtual_end_sec"] = round(end, 3)
        row["source_start_sec"] = round(
            float(row["source_start_sec"]) + source_offset, 3
        )
        row["source_end_sec"] = round(
            float(row["source_start_sec"]) + source_end_offset - source_offset, 3
        )
        row["slice_id"] = str(row["slice_id"]) + ":clipped"
        clipped.append(row)
    return tuple(clipped)


def _expected_frames(
    rows: Sequence[Mapping[str, Any]], *, fps: float, max_frames: int
) -> int:
    total = sum(
        int(
            math.ceil(
                (float(row["virtual_end_sec"]) - float(row["virtual_start_sec"]))
                * float(fps)
            )
        )
        for row in rows
    )
    return min(total, int(max_frames)) if int(max_frames) > 0 else total


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = dict(report["counts"])
    return "\n".join(
        (
            "# MM-Lifelong WP17-1 Dense OCR",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Frames / reader calls: `{counts['frames']} / {counts['reader_calls']}`",
            f"- Observations / tracks: `{counts['observations']} / {counts['tracks']}`",
            f"- Reader: `{report['reader']['source']}` on `{report['reader']['device']}`",
            f"- Structural gate: `{str(report['structural_gate_passed']).lower()}`",
            "- Frames are streamed and never persisted; construction is question/gold/target-alias blind.",
            "",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-manifest", required=True)
    parser.add_argument("--expected-timeline-sha256", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--ffmpeg-executable", default="ffmpeg")
    parser.add_argument("--track-max-gap-sec", type=float, default=3.0)
    parser.add_argument("--virtual-start-sec", type=float)
    parser.add_argument("--virtual-end-sec", type=float)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
