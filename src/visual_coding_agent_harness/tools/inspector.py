"""Segment Inspector subagent boundary for long-video context control."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace
from .segments import ClipExtractor, _clip_output_path, _extract_clip_ffmpeg


def build_segment_inspector_registry(
    backend: VisionLanguageBackend,
    *,
    workspace: Optional[EvidenceWorkspace] = None,
    extract_clips: bool = False,
    clip_extractor: Optional[ClipExtractor] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="inspect_segment", description="Delegate localized visual inspection to an isolated Segment Inspector.")
    def inspect_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        candidate_options: Sequence[str] = (),
        nframes: int = 16,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
    ) -> Mapping[str, object]:
        return _run_inspector(
            backend=backend,
            video_path=video_path,
            segment_id=segment_id,
            start_sec=start_sec,
            end_sec=end_sec,
            question=question,
            candidate_options=candidate_options,
            nframes=nframes,
            max_pixels=max_pixels,
            fps=fps,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
        )

    registry.register(inspect_segment)
    return registry


def _run_inspector(
    *,
    backend: VisionLanguageBackend,
    video_path: str,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    question: str,
    candidate_options: Sequence[str],
    nframes: int,
    max_pixels: int,
    fps: float,
    workspace: Optional[EvidenceWorkspace],
    extract_clips: bool,
    clip_extractor: Optional[ClipExtractor],
) -> Mapping[str, object]:
    metadata = {
        "tool_role": "segment_inspector",
        "context_tier": "isolated_subagent",
        "segment_id": segment_id,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "question": question,
        "candidate_options": list(candidate_options),
        "nframes": int(nframes),
        "max_pixels": int(max_pixels),
    }
    if fps > 0:
        metadata["fps"] = float(fps)

    media_path = video_path
    input_artifacts = [f"{video_path}#t={float(start_sec):.3f},{float(end_sec):.3f}"]
    limitations = (
        "Inspector distilled one localized observation; intermediate visual reasoning stays outside the main context."
    )

    if extract_clips:
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
        limitations = (
            "Inspector used a physical segment clip and returned one distilled observation; "
            "fine-grained motion may still be limited by sampling."
        )

    response = backend.generate(
        BackendRequest(
            task="inspect_segment",
            prompt=_inspector_prompt(
                segment_id=segment_id,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                question=question,
                candidate_options=candidate_options,
            ),
            media_path=media_path,
            media_type="video",
            max_new_tokens=256,
            metadata=metadata,
        )
    )
    return {
        "claim": response.text.strip(),
        "confidence": 0.74,
        "input_artifacts": input_artifacts,
        "regions": [metadata],
        "limitations": limitations,
        "raw_backend": dict(response.raw),
    }


def _inspector_prompt(
    *,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    question: str,
    candidate_options: Sequence[str],
) -> str:
    options_text = "\n".join(str(option) for option in candidate_options) or "(none)"
    return (
        "You are a Segment Inspector subagent for long-video reasoning.\n"
        "Your context is intentionally isolated from the master planner.\n"
        "Inspect only the provided time window and do not rely on outside video context.\n"
        "Return one distilled observation: claim, supported option if any, confidence, and limitation.\n"
        "Do not include step-by-step reasoning or raw frame descriptions unless essential evidence.\n"
        f"Segment: {segment_id} [{start_sec:.3f}s, {end_sec:.3f}s]\n"
        f"Question: {question}\n"
        f"Candidate options:\n{options_text}"
    )
