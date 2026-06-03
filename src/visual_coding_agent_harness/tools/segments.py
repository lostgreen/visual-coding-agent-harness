"""Segment-level VLM tools for long-video exploration."""

from __future__ import annotations

from typing import Mapping

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool


def build_segment_vlm_registry(backend: VisionLanguageBackend) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_segment", description="Caption a time-bounded video segment with the shared VLM backend.")
    def caption_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str = "Describe this video segment.",
        nframes: int = 8,
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
        )

    @tool(name="qa_segment", description="Answer a question about a time-bounded video segment with the shared VLM backend.")
    def qa_segment(
        video_path: str,
        segment_id: str,
        start_sec: float,
        end_sec: float,
        question: str,
        nframes: int = 8,
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
    nframes: int,
) -> Mapping[str, object]:
    metadata = {
        "segment_id": segment_id,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "nframes": int(nframes),
    }
    response = backend.generate(
        BackendRequest(
            task=task,
            prompt=_segment_prompt(
                task=task,
                segment_id=segment_id,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                question=question,
            ),
            media_path=video_path,
            media_type="video",
            max_new_tokens=256,
            metadata=metadata,
        )
    )
    return {
        "claim": response.text.strip(),
        "confidence": 0.66,
        "input_artifacts": [f"{video_path}#t={float(start_sec):.3f},{float(end_sec):.3f}"],
        "regions": [metadata],
        "limitations": "Segment VLM observation; backend may need physical clipping for strict temporal isolation.",
        "raw_backend": dict(response.raw),
    }


def _segment_prompt(*, task: str, segment_id: str, start_sec: float, end_sec: float, question: str) -> str:
    instruction = "Describe the visual content" if task == "caption_segment" else "Answer the question"
    return (
        f"{instruction} for video segment {segment_id} only.\n"
        f"Target time range: {start_sec:.3f}s to {end_sec:.3f}s.\n"
        "Ignore other parts of the video when possible.\n"
        f"Question: {question}"
    )
