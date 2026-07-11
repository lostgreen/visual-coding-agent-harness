from __future__ import annotations

from math import ceil
from pathlib import Path
import subprocess
import threading
from typing import Sequence

from PIL import Image

from vcah.types import Frame


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_FFMPEG_SEMAPHORE = threading.BoundedSemaphore(4)
_FFMPEG_FRAME_TIMEOUT_SEC = 30.0
_FFMPEG_FRAME_ATTEMPTS = 2


def is_image_path(path: str | Path, *, must_exist: bool = False) -> bool:
    candidate = Path(str(path))
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    return candidate.exists() if must_exist else True


def probe_duration(video_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required when duration_sec is not provided") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-2:]
        raise RuntimeError(f"ffprobe failed: {' | '.join(tail)}")
    return float((completed.stdout or "0").strip() or 0.0)


def detect_frame_ranges_uniform(duration_sec: float, *, window: float = 15.0) -> tuple[tuple[float, float], ...]:
    duration = max(0.0, float(duration_sec))
    step = max(0.1, float(window))
    ranges = []
    start = 0.0
    while start < duration:
        end = min(duration, start + step)
        ranges.append((round(start, 3), round(end, 3)))
        start = end
    return tuple(ranges)

def sample_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int,
    out_dir: Path,
) -> tuple[Frame, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, time_sec in enumerate(_sample_times(start_sec, end_sec, n_frames), start=1):
        output_path = out_dir / f"frame_{index:03d}.jpg"
        _extract_frame(video_path, time_sec, output_path)
        frames.append(Frame(frame_id=f"fr{index:03d}", time_sec=time_sec, path=str(output_path)))
    return tuple(frames)


def frame_ranges_to_beats(
    frame_ranges: Sequence[tuple[float, float]],
    keyframes: Sequence[str],
    *,
    sim_threshold: float = 0.85,
    max_beat_sec: float = 60.0,
) -> tuple[tuple[int, ...], ...]:
    ranges = tuple((float(start), float(end)) for start, end in frame_ranges)
    if not ranges:
        return ()
    signatures = tuple(_image_signature(path) for path in keyframes)
    groups: list[tuple[int, ...]] = []
    current = [0]
    current_start = ranges[0][0]
    previous_signature = signatures[0] if signatures else None
    threshold = max(0.0, min(1.0, float(sim_threshold)))
    max_duration = max(0.1, float(max_beat_sec))
    for index in range(1, len(ranges)):
        start_sec, end_sec = ranges[index]
        signature = signatures[index] if index < len(signatures) else None
        similar = _signature_similarity(previous_signature, signature) >= threshold
        within_duration = float(end_sec) - float(current_start) <= max_duration
        if similar and within_duration:
            current.append(index)
        else:
            groups.append(tuple(current))
            current = [index]
            current_start = float(start_sec)
        previous_signature = signature
    groups.append(tuple(current))
    return tuple(groups)


def render_timeline_grid(frame_paths: Sequence[str], out_path: Path, *, cols: int = 6) -> Path:
    sources = [Path(path) for path in frame_paths if is_image_path(path, must_exist=True)]
    cell_size = (160, 90)
    cells = [_fit_image(path, cell_size=cell_size) for path in sources] or [_placeholder(cell_size)]
    cols = max(1, min(int(cols), len(cells)))
    rows = max(1, ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * cell_size[0], rows * cell_size[1]), color=(12, 12, 12))
    for index, cell in enumerate(cells):
        canvas.paste(cell, ((index % cols) * cell_size[0], (index // cols) * cell_size[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=88)
    return out_path


def _sample_times(start_sec: float, end_sec: float, n_frames: int) -> tuple[float, ...]:
    count = max(1, int(n_frames))
    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    if count == 1 or end <= start:
        return (round(start, 3),)
    span = end - start
    return tuple(round(start + span * (index + 0.5) / count, 3) for index in range(count))


def _extract_frame(video_path: str, time_sec: float, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(time_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    last_error = ""
    for _ in range(_FFMPEG_FRAME_ATTEMPTS):
        output_path.unlink(missing_ok=True)
        try:
            with _FFMPEG_SEMAPHORE:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=_FFMPEG_FRAME_TIMEOUT_SEC,
                )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to sample frames") from exc
        except subprocess.TimeoutExpired:
            last_error = f"ffmpeg timed out after {_FFMPEG_FRAME_TIMEOUT_SEC:.0f}s while sampling a frame"
            continue
        if completed.returncode == 0 and output_path.exists():
            return
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-2:]
        last_error = f"ffmpeg failed to sample a frame: {' | '.join(tail)}"
    raise RuntimeError(last_error or "ffmpeg failed to sample a frame")


def _fit_image(path: Path, *, cell_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(cell_size)
        canvas = Image.new("RGB", cell_size, color=(18, 18, 18))
        canvas.paste(image, ((cell_size[0] - image.width) // 2, (cell_size[1] - image.height) // 2))
        return canvas


def _placeholder(cell_size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", cell_size, color=(42, 48, 56))


def _image_signature(path: str) -> tuple[int, ...] | None:
    if not is_image_path(path, must_exist=True):
        return None
    try:
        with Image.open(path) as image:
            small = image.convert("RGB").resize((8, 8))
            values: list[int] = []
            for red, green, blue in small.getdata():
                values.extend((int(red) // 32, int(green) // 32, int(blue) // 32))
            return tuple(values)
    except OSError:
        return None


def _signature_similarity(left: tuple[int, ...] | None, right: tuple[int, ...] | None) -> float:
    if left is None or right is None or len(left) != len(right) or not left:
        return 0.0
    return float(sum(1 for lhs, rhs in zip(left, right) if lhs == rhs)) / float(len(left))
