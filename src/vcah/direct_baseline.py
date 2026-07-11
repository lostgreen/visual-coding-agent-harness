from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps


def uniform_midpoint_times(duration_sec: float, frame_count: int) -> tuple[float, ...]:
    duration = float(duration_sec)
    count = int(frame_count)
    if duration <= 0:
        raise ValueError("duration_sec must be positive")
    if count <= 0:
        raise ValueError("frame_count must be positive")
    step = duration / count
    return tuple(round((index + 0.5) * step, 3) for index in range(count))


def materialize_uniform_frames(
    *,
    video_path: Path,
    duration_sec: float,
    out_dir: Path,
    frame_count: int = 512,
    max_image_edge: int = 512,
    rebuild: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], ...]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frame_manifest.jsonl"
    if not rebuild:
        cached = _load_valid_frame_manifest(manifest_path, expected_count=int(frame_count))
        if cached:
            return cached

    for path in output_dir.glob("frame_*.jpg"):
        path.unlink()
    times = uniform_midpoint_times(float(duration_sec), int(frame_count))
    step = float(duration_sec) / int(frame_count)
    fps = int(frame_count) / float(duration_sec)
    output_pattern = output_dir / "frame_%04d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{step / 2.0:.6f}",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:.12f},scale={int(max_image_edge)}:{int(max_image_edge)}:force_original_aspect_ratio=decrease",
        "-frames:v",
        str(int(frame_count)),
        "-q:v",
        "3",
        str(output_pattern),
    ]
    completed = runner(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg uniform sampling failed: {' | '.join(tail)}")
    frame_paths = tuple(sorted(output_dir.glob("frame_*.jpg")))
    if len(frame_paths) != int(frame_count):
        raise RuntimeError(f"Expected {int(frame_count)} sampled frames, found {len(frame_paths)}")
    rows = tuple(
        {
            "frame_index": index,
            "time_sec": float(times[index - 1]),
            "path": str(path),
        }
        for index, path in enumerate(frame_paths, start=1)
    )
    _write_jsonl(manifest_path, rows)
    return rows


def format_timestamped_asr(cues: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"[{_format_timestamp(float(cue.get('start', 0.0) or 0.0))}-{_format_timestamp(float(cue.get('end', 0.0) or 0.0))}] "
        f"{str(cue.get('text', '') or '').strip()}"
        for cue in cues
        if str(cue.get("text", "") or "").strip()
    )


def build_direct_prompt(
    *,
    question: str,
    options: Mapping[str, str],
    frame_rows: Sequence[Mapping[str, Any]],
    asr_text: str,
) -> str:
    frame_map = " ".join(
        f"F{int(row['frame_index']):04d}={_format_timestamp(float(row['time_sec']))}"
        for row in frame_rows
    )
    return (
        "Answer this multiple-choice question from the complete uniformly sampled video frames and timestamped ASR. "
        "The images are supplied in ascending frame-index order. Use both modalities and choose exactly one option. "
        "Return compact JSON only: {\"answer\":\"A\",\"rationale\":\"brief observable justification\","
        "\"evidence\":[{\"frame_index\":1,\"time_sec\":1.0,\"asr_quote\":\"optional short quote\"}]}. "
        "The rationale must be a concise evidence summary. Do not provide hidden chain-of-thought or a step-by-step internal monologue.\n"
        f"Question: {question}\n"
        f"Options: {json.dumps(dict(options), ensure_ascii=False)}\n"
        f"Frame timeline: {frame_map}\n"
        f"Timestamped ASR:\n{asr_text}"
    )


def parse_direct_response(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    answer_value = payload.get("answer", "")
    if isinstance(answer_value, Mapping):
        answer_value = (
            answer_value.get("option")
            or answer_value.get("label")
            or answer_value.get("choice")
            or answer_value.get("answer")
            or answer_value.get("text")
            or ""
        )
    answer_match = re.match(r"\s*([A-H])(?:\.|\)|:|\s|$)", str(answer_value or "").upper())
    answer = answer_match.group(1) if answer_match else str(answer_value or "").strip()
    evidence = tuple(dict(item) for item in (payload.get("evidence") or ())[:32] if isinstance(item, Mapping))
    return {
        "answer": answer,
        "rationale": str(payload.get("rationale", "") or "")[:2000],
        "evidence": evidence,
    }


def render_contact_sheets(
    frame_paths: Sequence[Path | str],
    out_dir: Path,
    *,
    rows: int = 4,
    cols: int = 4,
    cell_size: tuple[int, int] = (160, 90),
) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in frame_paths)
    per_sheet = int(rows) * int(cols)
    if per_sheet <= 0:
        raise ValueError("rows and cols must be positive")
    if not paths or len(paths) % per_sheet:
        raise ValueError(f"frame count must be a non-zero multiple of {per_sheet}")
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for sheet_index, offset in enumerate(range(0, len(paths), per_sheet), start=1):
        canvas = Image.new("RGB", (int(cols) * cell_size[0], int(rows) * cell_size[1]), color=(0, 0, 0))
        for local_index, path in enumerate(paths[offset : offset + per_sheet]):
            with Image.open(path) as source:
                cell = ImageOps.fit(source.convert("RGB"), cell_size, method=Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(cell)
            frame_index = offset + local_index + 1
            label = f"F{frame_index:04d}"
            draw.rectangle((2, 2, 48, 14), fill=(0, 0, 0))
            draw.text((4, 2), label, fill=(255, 255, 255))
            canvas.paste(cell, ((local_index % int(cols)) * cell_size[0], (local_index // int(cols)) * cell_size[1]))
        sheet_path = output_dir / f"sheet_{sheet_index:03d}.jpg"
        canvas.save(sheet_path, format="JPEG", quality=88)
        sheets.append(sheet_path)
    return tuple(sheets)


def request_direct_answer(
    *,
    api: Any,
    prompt: str,
    frame_paths: Sequence[Path | str],
    sheet_dir: Path,
    force_contact_sheets: bool = False,
    max_tokens: int = 900,
) -> tuple[dict[str, Any], str, str, tuple[str, ...]]:
    frames = tuple(str(path) for path in frame_paths)
    if not force_contact_sheets:
        try:
            raw = api.chat(prompt, image_paths=frames, max_tokens=max_tokens)
            return parse_direct_response(raw), raw, f"images_{len(frames)}", frames
        except RuntimeError as exc:
            if not _is_request_shape_error(exc):
                raise
    sheets = render_contact_sheets(frame_paths, sheet_dir)
    submitted = tuple(str(path) for path in sheets)
    raw = api.chat(prompt, image_paths=submitted, max_tokens=max_tokens)
    return parse_direct_response(raw), raw, f"contact_sheets_{len(sheets)}", submitted


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(item) for item in results)
    successful = tuple(row for row in rows if not str(row.get("error", "") or ""))
    modes = Counter(str(row.get("input_mode", "") or "") for row in successful if str(row.get("input_mode", "") or ""))
    latencies = [float(row.get("latency_sec", 0.0) or 0.0) for row in successful]
    return {
        "case_count": len(rows),
        "successful_cases": len(successful),
        "correct": sum(bool(row.get("correct")) for row in rows),
        "accuracy": sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else 0.0,
        "failures": len(rows) - len(successful),
        "mean_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
        "input_modes": dict(sorted(modes.items())),
        "cases": rows,
    }


def _load_valid_frame_manifest(path: Path, *, expected_count: int) -> tuple[dict[str, Any], ...]:
    if not Path(path).exists():
        return ()
    try:
        rows = tuple(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, json.JSONDecodeError):
        return ()
    if len(rows) != int(expected_count) or any(not Path(str(row.get("path", ""))).exists() for row in rows):
        return ()
    return tuple(dict(row) for row in rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _format_timestamp(value: float) -> str:
    total_ms = max(0, round(float(value) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _is_request_shape_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    patterns = (
        "http 400",
        "http 413",
        "request too large",
        "payload too large",
        "too many images",
        "image count",
        "maximum number of images",
    )
    return any(pattern in text for pattern in patterns)
