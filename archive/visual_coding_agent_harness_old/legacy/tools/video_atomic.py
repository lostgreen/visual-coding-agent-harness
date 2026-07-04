"""Traditional video tools backed by ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from visual_coding_agent_harness.legacy.core.registry import ToolError


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise ToolError("ffmpeg is required for video tools but was not found on PATH")
    return path


def build_sample_frames_command(
    video_path: str,
    output_pattern: str,
    fps: float = 1.0,
    start_time: float | None = None,
    duration: float | None = None,
) -> Sequence[str]:
    command = ["ffmpeg", "-y"]
    if start_time is not None:
        command.extend(["-ss", str(start_time)])
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend(["-i", video_path, "-vf", f"fps={fps}", output_pattern])
    return command


def build_extract_clip_command(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float,
) -> Sequence[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-c",
        "copy",
        output_path,
    ]


def sample_frames(
    video_path: str,
    output_pattern: str,
    fps: float = 1.0,
    start_time: float | None = None,
    duration: float | None = None,
) -> Mapping[str, object]:
    require_ffmpeg()
    _ensure_parent(output_pattern)
    command = list(
        build_sample_frames_command(
            video_path=video_path,
            output_pattern=output_pattern,
            fps=fps,
            start_time=start_time,
            duration=duration,
        )
    )
    subprocess.run(command, check=True, capture_output=True)
    return {
        "claim": f"Sampled frames from {video_path} at fps={fps}.",
        "confidence": 1.0,
        "input_artifacts": [video_path],
        "output_artifacts": [output_pattern],
        "time_range": _time_range(start_time, duration),
        "raw_command": command,
        "limitations": "Frame pattern is returned; caller should inspect concrete generated files if needed.",
    }


def extract_clip(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float,
) -> Mapping[str, object]:
    require_ffmpeg()
    _ensure_parent(output_path)
    command = list(
        build_extract_clip_command(
            video_path=video_path,
            output_path=output_path,
            start_time=start_time,
            duration=duration,
        )
    )
    subprocess.run(command, check=True, capture_output=True)
    return {
        "claim": f"Extracted clip {start_time:.2f}s-{start_time + duration:.2f}s.",
        "confidence": 1.0,
        "input_artifacts": [video_path],
        "output_artifacts": [output_path],
        "time_range": [start_time, start_time + duration],
        "raw_command": command,
        "limitations": "Uses stream copy; keyframe boundaries may slightly affect exact visual start.",
    }


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _time_range(start_time: float | None, duration: float | None) -> Sequence[float]:
    if start_time is None or duration is None:
        return []
    return [start_time, start_time + duration]
