#!/usr/bin/env python3
"""Build a compact per-segment video evidence pack for debugging agents.

The script extracts a few representative frames from each fixed-size segment,
clips subtitle entries to the same segment windows, and writes a small manifest
plus optional contact sheets. It is intentionally independent of the harness
runtime so it can be run on a remote KML box against dataset cache files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_SRT_TIMING_RE = re.compile(
    r"(?P<start>\d\d:\d\d:\d\d[,.]\d\d\d)\s*-->\s*"
    r"(?P<end>\d\d:\d\d:\d\d[,.]\d\d\d)"
)


@dataclass(frozen=True)
class SubtitleEntry:
    start: float
    end: float
    text: str


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def _seconds_from_srt_time(value: str) -> float:
    value = value.replace(",", ".")
    hh, mm, rest = value.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hh = int(seconds // 3600)
    mm = int(seconds % 3600 // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_srt(path: Path) -> list[SubtitleEntry]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text)
    entries: list[SubtitleEntry] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_idx = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_idx < 0:
            continue
        match = _SRT_TIMING_RE.search(lines[timing_idx])
        if not match:
            continue
        body = " ".join(lines[timing_idx + 1 :])
        body = _norm_text(re.sub(r"<[^>]+>", " ", body))
        if body:
            entries.append(
                SubtitleEntry(
                    start=_seconds_from_srt_time(match.group("start")),
                    end=_seconds_from_srt_time(match.group("end")),
                    text=body,
                )
            )
    return entries


def ffprobe_duration(video_path: Path) -> float:
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return float(out)


def segment_windows(duration: float, segment_sec: float) -> list[tuple[str, float, float]]:
    count = int(math.ceil(duration / segment_sec))
    windows = []
    for idx in range(count):
        start = idx * segment_sec
        end = min(duration, (idx + 1) * segment_sec)
        windows.append((f"seg_{idx + 1:04d}", start, end))
    return windows


def frame_times(start: float, end: float, count: int) -> list[float]:
    span = max(0.0, end - start)
    if count <= 1 or span <= 1:
        return [start + span / 2]
    return [start + span * (idx + 1) / (count + 1) for idx in range(count)]


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        check=True,
    )


def subtitle_entries_for_window(
    entries: Iterable[SubtitleEntry], start: float, end: float
) -> list[SubtitleEntry]:
    return [entry for entry in entries if entry.end >= start and entry.start <= end]


def target_hits(text: str, targets: Iterable[str]) -> list[str]:
    haystack = _norm_text(text).lower()
    hits = []
    for target in targets:
        normalized = _norm_text(target)
        if normalized and normalized.lower() in haystack:
            hits.append(normalized)
    return hits


def write_subtitle_files(
    segment_dir: Path, entries: list[SubtitleEntry], targets: list[str]
) -> tuple[str, list[str]]:
    text_lines = []
    jsonl_lines = []
    all_text = []
    for entry in entries:
        all_text.append(entry.text)
        text_lines.append(f"[{_fmt_time(entry.start)} - {_fmt_time(entry.end)}] {entry.text}")
        jsonl_lines.append(
            json.dumps(
                {"start_sec": entry.start, "end_sec": entry.end, "text": entry.text},
                ensure_ascii=False,
            )
        )
    (segment_dir / "subtitles.txt").write_text(
        "\n".join(text_lines) + ("\n" if text_lines else ""),
        encoding="utf-8",
    )
    (segment_dir / "subtitles.jsonl").write_text(
        "\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""),
        encoding="utf-8",
    )
    joined = _norm_text(" ".join(all_text))
    return joined[:360], target_hits(joined, targets)


def create_contact_sheet(
    frame_paths: list[Path],
    labels: list[str],
    output_path: Path,
    thumb_width: int,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    if not frame_paths:
        return False

    images = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / float(img.width)
        img = img.resize((thumb_width, max(1, int(img.height * ratio))))
        images.append(img)

    label_h = 28
    gap = 6
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images) + label_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    x = 0
    for img, label in zip(images, labels):
        sheet.paste(img, (x, 0))
        draw.rectangle((x, img.height, x + img.width, height), fill=(255, 255, 255))
        draw.text((x + 4, img.height + 8), label, fill=(0, 0, 0), font=font)
        x += img.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def create_all_segments_sheet(
    rows: list[tuple[str, list[Path], list[str]]],
    output_path: Path,
    thumb_width: int,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    rendered_rows = []
    font = ImageFont.load_default()
    for segment_id, frame_paths, labels in rows:
        images = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            ratio = thumb_width / float(img.width)
            img = img.resize((thumb_width, max(1, int(img.height * ratio))))
            images.append(img)
        if not images:
            continue
        gap = 6
        label_w = 92
        label_h = 28
        row_w = label_w + sum(img.width for img in images) + gap * len(images)
        row_h = max(img.height for img in images) + label_h
        row = Image.new("RGB", (row_w, row_h), "white")
        draw = ImageDraw.Draw(row)
        draw.text((8, 10), segment_id, fill=(0, 0, 0), font=font)
        x = label_w
        for img, label in zip(images, labels):
            row.paste(img, (x, 0))
            draw.text((x + 4, img.height + 8), label, fill=(0, 0, 0), font=font)
            x += img.width + gap
        rendered_rows.append(row)
    if not rendered_rows:
        return False
    gap = 8
    width = max(row.width for row in rendered_rows)
    height = sum(row.height for row in rendered_rows) + gap * (len(rendered_rows) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--srt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--segment-sec", type=float, default=300.0)
    parser.add_argument("--frames-per-segment", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=256)
    parser.add_argument("--targets", nargs="*", default=[])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_srt(args.srt)
    duration = ffprobe_duration(args.video)
    windows = segment_windows(duration, args.segment_sec)

    manifest = {
        "video": str(args.video),
        "srt": str(args.srt),
        "duration_sec": duration,
        "segment_sec": args.segment_sec,
        "frames_per_segment": args.frames_per_segment,
        "targets": args.targets,
        "segments": [],
    }
    summary_lines = [
        "# Segment Evidence Pack",
        "",
        f"- video: `{args.video}`",
        f"- srt: `{args.srt}`",
        f"- duration_sec: `{duration:.3f}`",
        "",
        "| segment | time | frames | subtitle excerpt | target hits |",
        "| --- | --- | --- | --- | --- |",
    ]
    all_sheet_rows: list[tuple[str, list[Path], list[str]]] = []

    for segment_id, start, end in windows:
        segment_dir = args.out_dir / segment_id
        frame_dir = segment_dir / "frames"
        segment_dir.mkdir(parents=True, exist_ok=True)
        times = frame_times(start, end, args.frames_per_segment)
        frame_paths = []
        labels = []
        for idx, timestamp in enumerate(times, 1):
            frame_path = frame_dir / f"{segment_id}_f{idx:02d}_{timestamp:.1f}s.jpg"
            extract_frame(args.video, timestamp, frame_path)
            frame_paths.append(frame_path)
            labels.append(f"{timestamp:.1f}s")

        segment_entries = subtitle_entries_for_window(entries, start, end)
        subtitle_excerpt, hits = write_subtitle_files(segment_dir, segment_entries, args.targets)
        sheet_path = segment_dir / f"{segment_id}_sheet.jpg"
        has_sheet = create_contact_sheet(frame_paths, labels, sheet_path, args.thumb_width)
        all_sheet_rows.append((segment_id, frame_paths, labels))

        rel_frames = [str(path.relative_to(args.out_dir)) for path in frame_paths]
        segment_record = {
            "segment_id": segment_id,
            "start_sec": start,
            "end_sec": end,
            "frame_times_sec": times,
            "frames": rel_frames,
            "contact_sheet": str(sheet_path.relative_to(args.out_dir)) if has_sheet else None,
            "subtitle_count": len(segment_entries),
            "subtitle_excerpt": subtitle_excerpt,
            "target_hits": hits,
        }
        manifest["segments"].append(segment_record)
        frames_cell = "<br>".join(rel_frames[:2] + (["..."] if len(rel_frames) > 2 else []))
        excerpt_cell = subtitle_excerpt.replace("|", "\\|")
        hits_cell = ", ".join(hits) if hits else ""
        summary_lines.append(
            f"| {segment_id} | {_fmt_time(start)}-{_fmt_time(end)} | "
            f"{frames_cell} | {excerpt_cell} | {hits_cell} |"
        )

    all_sheet_path = args.out_dir / "all_segments_sheet.jpg"
    manifest["all_segments_sheet"] = (
        str(all_sheet_path.relative_to(args.out_dir))
        if create_all_segments_sheet(all_sheet_rows, all_sheet_path, args.thumb_width)
        else None
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "segment_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "segments": len(windows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
