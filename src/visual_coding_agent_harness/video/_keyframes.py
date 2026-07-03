"""Keyframe sampling helpers for multi_v3 video indexing."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from PIL import Image

from .index import Frame


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
