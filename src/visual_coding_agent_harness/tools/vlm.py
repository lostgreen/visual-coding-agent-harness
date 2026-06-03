"""VLM-backed visual tools.

These tools deliberately accept a backend instance instead of constructing a
model internally. That lets a main VLM agent and its tools share one loaded
foundation model during smoke tests and later benchmarks.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool


def build_vlm_registry(backend: VisionLanguageBackend) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_image", description="Caption an image using the shared VLM backend.")
    def caption_image(image_path: str, question: str = "Describe the image.") -> Mapping[str, object]:
        return _run_vlm_tool(
            backend=backend,
            task="caption_image",
            media_path=image_path,
            media_type="image",
            prompt=question,
        )

    @tool(name="qa_image", description="Answer an image question using the shared VLM backend.")
    def qa_image(image_path: str, question: str) -> Mapping[str, object]:
        return _run_vlm_tool(
            backend=backend,
            task="qa_image",
            media_path=image_path,
            media_type="image",
            prompt=question,
        )

    @tool(name="caption_video", description="Caption a video using the shared VLM backend.")
    def caption_video(video_path: str, question: str = "Describe the video.") -> Mapping[str, object]:
        return _run_vlm_tool(
            backend=backend,
            task="caption_video",
            media_path=video_path,
            media_type="video",
            prompt=question,
        )

    @tool(name="qa_video", description="Answer a video question using the shared VLM backend.")
    def qa_video(video_path: str, question: str) -> Mapping[str, object]:
        return _run_vlm_tool(
            backend=backend,
            task="qa_video",
            media_path=video_path,
            media_type="video",
            prompt=question,
        )

    registry.register(caption_image)
    registry.register(qa_image)
    registry.register(caption_video)
    registry.register(qa_video)
    return registry


def _run_vlm_tool(
    *,
    backend: VisionLanguageBackend,
    task: str,
    media_path: str,
    media_type: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    confidence: float = 0.65,
) -> Mapping[str, object]:
    response = backend.generate(
        BackendRequest(
            task=task,
            prompt=prompt,
            media_path=media_path,
            media_type=media_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    )
    return {
        "claim": response.text.strip(),
        "confidence": confidence,
        "input_artifacts": [media_path],
        "regions": [],
        "limitations": "VLM-generated observation; verify with atomic tools for high-stakes claims.",
        "raw_backend": dict(response.raw),
    }
