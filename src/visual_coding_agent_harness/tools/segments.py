"""Segment-level VLM tools for long-video exploration."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Optional

from ..agents.contracts import resolve_nframes
from ..agents.open_questions import exploration_question
from ..agents.output_quality import confidence_signal_from_text
from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace
from .frame_cache import FrameSampler


ClipExtractor = Callable[[str, str, float, float], str]


def build_segment_vlm_registry(
    backend: VisionLanguageBackend,
    *,
    workspace: Optional[EvidenceWorkspace] = None,
    extract_clips: bool = False,
    clip_extractor: Optional[ClipExtractor] = None,
    frame_sampler: Optional[FrameSampler] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption a time-bounded video segment with the shared VLM backend.")
    def caption_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str = "Describe this video segment.",
        nframes: int | None = None,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
    ) -> Mapping[str, object]:
        return _run_segment_tool(
            backend=backend,
            task="caption_segment",
            video_path=video_path,
            segment_id=segment_id,
            start_sec=start_sec,
            end_sec=end_sec,
            question=question,
            nframes=nframes,
            max_pixels=max_pixels,
            fps=fps,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
            frame_sampler=frame_sampler,
        )

    @tool(name="qa_segment", description="Answer a question about a time-bounded video segment with the shared VLM backend.")
    def qa_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        nframes: int | None = None,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
    ) -> Mapping[str, object]:
        return _run_segment_tool(
            backend=backend,
            task="qa_segment",
            video_path=video_path,
            segment_id=segment_id,
            start_sec=start_sec,
            end_sec=end_sec,
            question=question,
            nframes=nframes,
            max_pixels=max_pixels,
            fps=fps,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
            frame_sampler=frame_sampler,
        )

    registry.register(caption_segment)
    registry.register(qa_segment)
    return registry


def _run_segment_tool(
    *,
    backend: VisionLanguageBackend,
    task: str,
    video_path: str,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    question: str,
    nframes: int | None,
    max_pixels: int,
    fps: float,
    workspace: Optional[EvidenceWorkspace] = None,
    extract_clips: bool = False,
    clip_extractor: Optional[ClipExtractor] = None,
    frame_sampler: Optional[FrameSampler] = None,
) -> Mapping[str, object]:
    resolved_nframes, _ = resolve_nframes(nframes)
    prompt_question = exploration_question(question)
    metadata = {
        "segment_id": segment_id,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "nframes": int(resolved_nframes),
        "max_pixels": int(max_pixels),
        "question": prompt_question,
    }
    if fps > 0:
        metadata["fps"] = float(fps)
    media_path: str | None = video_path
    media_type = "video"
    frame_paths: tuple[str, ...] = ()
    input_artifacts = [f"{video_path}#t={float(start_sec):.3f},{float(end_sec):.3f}"]
    limitations = "Segment VLM observation; backend may need physical clipping for strict temporal isolation."

    if frame_sampler is not None:
        frame_paths = tuple(
            frame_sampler(video_path, float(start_sec), float(end_sec), int(resolved_nframes))
        )
        if frame_paths:
            media_path = None
            media_type = "image"
            input_artifacts = list(frame_paths)
            metadata["source_video_path"] = video_path
            metadata["frame_cache_policy"] = "precomputed_2fps"
            metadata["frame_count"] = len(frame_paths)
            limitations = "Precomputed 2fps frame-cache observation; no per-call video clipping or decoding."
    if not frame_paths and extract_clips:
        if workspace is None:
            raise ValueError("extract_clips=True requires an EvidenceWorkspace")
        output_path = _clip_output_path(
            workspace=workspace,
            segment_id=segment_id,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
        )
        extractor = clip_extractor or _extract_clip_ffmpeg
        clip_path = extractor(video_path, str(output_path), float(start_sec), float(end_sec))
        media_path = clip_path
        input_artifacts = [clip_path]
        metadata["source_video_path"] = video_path
        metadata["clip_path"] = clip_path
        limitations = "Physical segment clip extracted before VLM call; fine-grained motion may still be limited by nframes."

    response = backend.generate(
        BackendRequest(
            task=task,
            prompt=_segment_prompt(
                task=task,
                segment_id=segment_id,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                question=prompt_question,
            ),
            media_path=media_path,
            media_type=media_type,
            frames=frame_paths,
            max_new_tokens=256,
            metadata=metadata,
        )
    )
    claim = response.text.strip()
    confidence_signal = confidence_signal_from_text(claim)
    result: dict[str, object] = {
        "claim": claim,
        "confidence": 0.66,
        "input_artifacts": input_artifacts,
        "regions": [metadata],
        "limitations": limitations,
        "raw_backend": dict(response.raw),
    }
    if confidence_signal:
        result["confidence_signal"] = confidence_signal
        result["grounding_quality"] = "inferred"
    return result


def _segment_prompt(*, task: str, segment_id: str, start_sec: float, end_sec: float, question: str) -> str:
    mode = "Caption task" if task == "caption_segment" else "QA task"
    return (
        f"{mode}: use only visible evidence from video segment {segment_id}.\n"
        f"Target time range: {start_sec:.3f}s to {end_sec:.3f}s.\n"
        "Do not invent details, identities, text, or temporal order that are not supported.\n"
        "Mention uncertainty when evidence is ambiguous or too low resolution.\n"
        "Ignore other parts of the video when possible.\n"
        f"Question: {question}"
    )


def _clip_output_path(*, workspace: EvidenceWorkspace, segment_id: str, start_sec: float, end_sec: float) -> Path:
    safe_segment_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment_id)
    start_ms = int(round(start_sec * 1000))
    end_ms = int(round(end_sec * 1000))
    return workspace.root / "artifacts" / "clips" / f"{safe_segment_id}_{start_ms}_{end_ms}.mp4"


def _extract_clip_ffmpeg(video_path: str, output_path: str, start_sec: float, end_sec: float) -> str:
    output = Path(output_path)
    if output.exists() and output.stat().st_size > 0:
        return str(output)
    duration = max(0.001, float(end_sec) - float(start_sec))
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{float(start_sec):.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to extract segment clips") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed to extract clip: {' | '.join(message)}") from exc
    return output_path
