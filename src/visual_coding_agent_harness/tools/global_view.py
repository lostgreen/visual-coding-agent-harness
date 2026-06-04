"""Sparse whole-video view tool for preserving a direct-answer floor."""

from __future__ import annotations

import re
from typing import Mapping

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool


DEFAULT_GLOBAL_NFRAMES = 64
DEFAULT_MAX_PIXELS = 151200


def build_global_view_registry(backend: VisionLanguageBackend) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="global_gist", description="Inspect a sparse whole-video view before local decomposition.")
    def global_gist(
        video_path: str,
        question: str,
        duration_sec: float,
        nframes: int = DEFAULT_GLOBAL_NFRAMES,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> Mapping[str, object]:
        response = backend.generate(
            BackendRequest(
                task="global_gist",
                prompt=_global_gist_prompt(question=question, duration_sec=duration_sec),
                media_path=video_path,
                media_type="video",
                max_new_tokens=256,
                metadata={
                    "nframes": int(nframes),
                    "max_pixels": int(max_pixels),
                    "duration_sec": float(duration_sec),
                    "question": question,
                },
            )
        )
        answer_text = response.text.strip()
        supported_option = _extract_choice(answer_text)
        raw_fields = {
            "supported_option": supported_option,
            "grounding_quality": "global_sparse",
            "time_range": [0.0, float(duration_sec)],
            "nframes": int(nframes),
            "max_pixels": int(max_pixels),
            "raw_backend": dict(response.raw),
        }
        return {
            "claim": answer_text,
            "confidence": 0.76,
            "input_artifacts": [f"{video_path}#t=0.000,{float(duration_sec):.3f}"],
            "regions": [
                {
                    "tool_role": "global_view",
                    "start_sec": 0.0,
                    "end_sec": float(duration_sec),
                    "nframes": int(nframes),
                    "max_pixels": int(max_pixels),
                    "grounding_quality": "global_sparse",
                }
            ],
            "limitations": "Sparse whole-video sampling; fine local details may require follow-up inspection.",
            **raw_fields,
            "raw_output": raw_fields,
        }

    registry.register(global_gist)
    return registry


def _global_gist_prompt(*, question: str, duration_sec: float) -> str:
    return (
        "Answer from a sparse full-video view before any local decomposition.\n"
        "Use the sampled whole-video context as a direct baseline floor.\n"
        "Start multiple-choice answers with exactly one option letter when options are provided.\n"
        "Mention uncertainty if the sparse global view is insufficient for fine local details.\n"
        f"Video duration: {float(duration_sec):.1f} seconds.\n"
        f"Question:\n{question}"
    )


def _extract_choice(text: str) -> str:
    upper = (text or "").upper()
    patterns = [
        r"\b(?:ANSWER|CHOICE|OPTION|FINAL)\s*(?:IS|:)?\s*([A-H])\b",
        r"^\s*([A-H])\b",
        r"\b([A-H])\s*[\).:-]",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return ""
