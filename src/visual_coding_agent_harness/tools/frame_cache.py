"""Precomputed low-fps frame cache for long-video tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Callable, Sequence


FRAME_CACHE_SCHEMA_VERSION = "frame_cache_v1"
FrameSampler = Callable[[str, float, float, int], Sequence[str]]


@dataclass(frozen=True)
class FrameSample:
    timestamp_sec: float
    path: str


@dataclass(frozen=True)
class FrameCache:
    video_path: str
    frame_dir: Path
    fps: float
    frames: Sequence[FrameSample]

    def sample(self, *, start_sec: float, end_sec: float, max_frames: int) -> tuple[FrameSample, ...]:
        if max_frames <= 0:
            return ()
        start = max(0.0, float(start_sec))
        end = max(start, float(end_sec))
        candidates = [frame for frame in self.frames if start <= float(frame.timestamp_sec) < end]
        if not candidates:
            candidates = _nearest_frames(self.frames, start_sec=start, end_sec=end)
        if len(candidates) <= max_frames:
            return tuple(candidates)
        return tuple(candidates[index] for index in _uniform_indices(len(candidates), max_frames))

    def sample_paths(self, video_path: str, start_sec: float, end_sec: float, max_frames: int) -> tuple[str, ...]:
        return tuple(frame.path for frame in self.sample(start_sec=start_sec, end_sec=end_sec, max_frames=max_frames))


def build_extract_frame_cache_command(*, video_path: str, output_pattern: str, fps: float) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={float(fps)}",
        "-q:v",
        "2",
        output_pattern,
    ]


def build_frame_cache_for_video(
    *,
    video_path: Path,
    frame_dir: Path,
    fps: float = 2.0,
    duration_sec: float | None = None,
) -> FrameCache:
    manifest_path = frame_dir / "frame_cache_manifest.json"
    cached = _load_manifest(manifest_path)
    if cached is not None and cached.video_path == str(video_path) and float(cached.fps) == float(fps):
        if cached.frames and all(Path(frame.path).exists() for frame in cached.frames[:3]):
            return cached

    frame_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(frame_dir / "frame_%09d.jpg")
    command = build_extract_frame_cache_command(video_path=str(video_path), output_pattern=output_pattern, fps=fps)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to build the frame cache") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed to build frame cache: {' | '.join(message)}") from exc

    frames = _frames_from_dir(frame_dir=frame_dir, fps=fps)
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frame-cache images under {frame_dir}")
    cache = FrameCache(video_path=str(video_path), frame_dir=frame_dir, fps=float(fps), frames=frames)
    _write_manifest(manifest_path, cache=cache, duration_sec=duration_sec)
    return cache


def _uniform_indices(length: int, max_items: int) -> tuple[int, ...]:
    if max_items <= 1:
        return (0,)
    last = length - 1
    indices = [round(index * last / (max_items - 1)) for index in range(max_items)]
    deduped: list[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    return tuple(deduped)


def _nearest_frames(frames: Sequence[FrameSample], *, start_sec: float, end_sec: float) -> list[FrameSample]:
    if not frames:
        return []
    midpoint = (float(start_sec) + float(end_sec)) / 2.0
    return [min(frames, key=lambda frame: abs(float(frame.timestamp_sec) - midpoint))]


def _frames_from_dir(*, frame_dir: Path, fps: float) -> tuple[FrameSample, ...]:
    frames = []
    for index, path in enumerate(sorted(frame_dir.glob("frame_*.jpg"))):
        frames.append(FrameSample(timestamp_sec=index / float(fps), path=str(path)))
    return tuple(frames)


def _load_manifest(path: Path) -> FrameCache | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != FRAME_CACHE_SCHEMA_VERSION:
        return None
    frames = tuple(
        FrameSample(timestamp_sec=float(item["timestamp_sec"]), path=str(item["path"]))
        for item in payload.get("frames") or ()
        if item.get("path") is not None
    )
    return FrameCache(
        video_path=str(payload.get("video_path") or ""),
        frame_dir=Path(str(payload.get("frame_dir") or path.parent)),
        fps=float(payload.get("fps") or 0.0),
        frames=frames,
    )


def _write_manifest(path: Path, *, cache: FrameCache, duration_sec: float | None) -> None:
    payload = {
        "schema_version": FRAME_CACHE_SCHEMA_VERSION,
        "video_path": cache.video_path,
        "frame_dir": str(cache.frame_dir),
        "fps": float(cache.fps),
        "duration_sec": None if duration_sec is None else float(duration_sec),
        "frames": [
            {"timestamp_sec": float(frame.timestamp_sec), "path": frame.path}
            for frame in cache.frames
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
