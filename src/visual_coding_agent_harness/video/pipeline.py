"""Video indexing pipeline helpers for multi_v3."""

from __future__ import annotations

from math import ceil
from pathlib import Path
import re
import subprocess
from typing import Sequence

from PIL import Image

from .index import Frame


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def is_image_path(path: str | Path, *, must_exist: bool = False) -> bool:
    candidate = Path(str(path))
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    return candidate.exists() if must_exist else True


def compose_shot_grid(
    frames: Sequence[Frame],
    out_path: Path,
    *,
    cols: int = 3,
    cell_size: tuple[int, int] = (320, 180),
) -> Path:
    """Compose a low-resolution JPG grid from existing frame thumbnails."""

    sources = [Path(frame.thumb_path) for frame in frames if frame.thumb_path and is_image_path(frame.thumb_path, must_exist=True)]
    if not sources:
        raise ValueError("compose_shot_grid requires at least one readable image frame")
    cells = [_fit_image(source, cell_size=cell_size) for source in sources]
    return _write_grid(cells, out_path=out_path, cols=cols, cell_size=cell_size)


def _first_existing_image(paths: Sequence[str]) -> Path | None:
    for raw_path in paths:
        if is_image_path(raw_path, must_exist=True):
            return Path(raw_path)
    return None


def _fit_image(path: Path, *, cell_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(cell_size)
        canvas = Image.new("RGB", cell_size, color=(18, 18, 18))
        x = (cell_size[0] - image.width) // 2
        y = (cell_size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def _placeholder(cell_size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", cell_size, color=(42, 48, 56))


def _write_grid(
    cells: Sequence[Image.Image],
    *,
    out_path: Path,
    cols: int,
    cell_size: tuple[int, int],
) -> Path:
    cols = max(1, min(int(cols), len(cells)))
    rows = max(1, ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * cell_size[0], rows * cell_size[1]), color=(12, 12, 12))
    for index, cell in enumerate(cells):
        x = (index % cols) * cell_size[0]
        y = (index // cols) * cell_size[1]
        canvas.paste(cell.convert("RGB"), (x, y))
    return _save_jpeg(canvas, out_path)


def _save_jpeg(image: Image.Image, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, format="JPEG", quality=88)
    return out_path


def sample_shot_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    *,
    n_frames: int = 6,
    out_dir: Path,
    size: tuple[int, int] | None = (256, 144),
) -> Sequence[Frame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    times = _sample_times(start_sec, end_sec, n_frames)
    frames = []
    for index, time_sec in enumerate(times, start=1):
        output_path = out_dir / f"frame_{index:03d}.jpg"
        _extract_frame(video_path=video_path, time_sec=time_sec, output_path=output_path)
        if size is not None:
            _resize_in_place(output_path, size=size)
        frames.append(Frame(frame_id=f"fr{index:03d}", time_sec=time_sec, thumb_path=str(output_path)))
    return tuple(frames)


def shots_to_beats(
    shots: Sequence[tuple[float, float]],
    keyframes: Sequence[str],
    *,
    sim_threshold: float = 0.85,
    max_beat_sec: float = 60.0,
) -> tuple[tuple[int, ...], ...]:
    """Return adjacent shot-index groups using lightweight perceptual similarity."""

    ranges = tuple((float(start), float(end)) for start, end in shots)
    if not ranges:
        return ()
    signatures = tuple(_image_signature(path) for path in keyframes)
    groups: list[tuple[int, ...]] = []
    current: list[int] = [0]
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
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def detect_shots_ffmpeg(
    video_path: str,
    *,
    min_shot_sec: float = 2.0,
    scene_threshold: float = 0.3,
) -> Sequence[tuple[float, float]]:
    duration = _probe_duration(video_path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{float(scene_threshold)})',showinfo",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to detect shot boundaries") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg shot detection failed: {' | '.join(tail)}")
    cuts = sorted(
        {
            time_sec
            for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr or "")
            if (time_sec := float(match.group(1))) > 0.0
        }
    )
    return _ranges_from_cuts(cuts, duration_sec=duration, min_shot_sec=min_shot_sec)


def detect_shots_uniform(duration_sec: float, *, window: float = 15.0) -> Sequence[tuple[float, float]]:
    duration = max(0.0, float(duration_sec))
    if duration <= 0.0:
        return ()
    step = max(0.1, float(window))
    ranges = []
    start = 0.0
    while start < duration:
        end = min(duration, start + step)
        ranges.append((round(start, 3), round(end, 3)))
        start = end
    return tuple(ranges)


def _sample_times(start_sec: float, end_sec: float, n_frames: int) -> tuple[float, ...]:
    count = max(0, int(n_frames))
    if count <= 0:
        return ()
    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    if count == 1 or end <= start:
        return (round(start, 3),)
    span = end - start
    return tuple(round(start + span * (index + 0.5) / count, 3) for index in range(count))


def _extract_frame(*, video_path: str, time_sec: float, output_path: Path) -> None:
    command = [
        "ffmpeg",
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
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to sample shot frames") from exc
    if completed.returncode != 0 or not output_path.exists():
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed to sample frame: {' | '.join(tail)}")


def _resize_in_place(path: Path, *, size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(size)
        image.save(path, format="JPEG", quality=88)


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
    same = sum(1 for lhs, rhs in zip(left, right) if lhs == rhs)
    return float(same) / float(len(left))


def _probe_duration(video_path: str) -> float:
    if not Path(video_path).exists():
        raise RuntimeError(f"video does not exist: {video_path}")
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
        out = subprocess.check_output(command, text=True, timeout=20).strip()
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to detect shot boundaries") from exc
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"ffprobe failed for {video_path}") from exc
    return max(0.0, float(out))


def _ranges_from_cuts(
    cuts: Sequence[float],
    *,
    duration_sec: float,
    min_shot_sec: float,
) -> tuple[tuple[float, float], ...]:
    boundaries = [0.0]
    for cut in cuts:
        if cut - boundaries[-1] >= float(min_shot_sec):
            boundaries.append(float(cut))
    duration = max(0.0, float(duration_sec))
    if not boundaries or duration - boundaries[-1] >= 0.001:
        boundaries.append(duration)
    ranges = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end > start:
            ranges.append((round(start, 3), round(end, 3)))
    return tuple(ranges)
