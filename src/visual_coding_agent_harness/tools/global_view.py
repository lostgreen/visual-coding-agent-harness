"""Sparse whole-video view tool for whole-video topic observations."""

from __future__ import annotations

import re
from typing import Mapping

from ..agent_contracts import VISUAL_EVIDENCE_NFRAMES, resolve_nframes
from ..open_questions import exploration_question
from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from .frame_cache import FrameSampler, frame_cache_artifact_ref


DEFAULT_GLOBAL_NFRAMES = VISUAL_EVIDENCE_NFRAMES
DEFAULT_MAX_PIXELS = 151200


def build_global_view_registry(
    backend: VisionLanguageBackend,
    *,
    frame_sampler: FrameSampler | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="global_gist", description="Inspect a sparse whole-video view before local decomposition.")
    def global_gist(
        video_path: str,
        question: str,
        duration_sec: float,
        nframes: int | None = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        sample_offset_sec: float = 0.0,
    ) -> Mapping[str, object]:
        resolved_nframes, _ = resolve_nframes(nframes)
        prompt_question = exploration_question(question)
        metadata = {
            "nframes": int(resolved_nframes),
            "max_pixels": int(max_pixels),
            "duration_sec": float(duration_sec),
            "sample_offset_sec": float(sample_offset_sec),
            "question": prompt_question,
        }
        media_path: str | None = video_path
        media_type = "video"
        frame_paths: tuple[str, ...] = ()
        input_artifacts = [f"{video_path}#t=0.000,{float(duration_sec):.3f}"]
        if frame_sampler is not None:
            frame_paths = tuple(frame_sampler(video_path, 0.0, float(duration_sec), int(resolved_nframes)))
            if frame_paths:
                media_path = None
                media_type = "video"
                input_artifacts = [
                    frame_cache_artifact_ref(
                        video_path=video_path,
                        start_sec=0.0,
                        end_sec=float(duration_sec),
                        frame_count=len(frame_paths),
                    )
                ]
                metadata["source_video_path"] = video_path
                metadata["frame_cache_policy"] = "precomputed_2fps"
                metadata["frame_count"] = len(frame_paths)
        response = backend.generate(
            BackendRequest(
                task="global_gist",
                prompt=_global_gist_prompt(
                    question=prompt_question,
                    duration_sec=duration_sec,
                    sample_offset_sec=sample_offset_sec,
                ),
                media_path=media_path,
                media_type=media_type,
                frames=frame_paths,
                max_new_tokens=256,
                metadata=metadata,
            )
        )
        answer_text = response.text.strip()
        candidate_option_hint = _extract_choice(answer_text)
        raw_fields = {
            "candidate_option_hint": candidate_option_hint,
            "candidate_option_relations": [],
            "grounding_quality": "global_sparse",
            "time_range": [0.0, float(duration_sec)],
            "nframes": int(resolved_nframes),
            "max_pixels": int(max_pixels),
            "sample_offset_sec": float(sample_offset_sec),
            "raw_backend": dict(response.raw),
        }
        return {
            "claim": answer_text,
            "confidence": 0.76,
            "input_artifacts": input_artifacts,
            "regions": [
                {
                    "tool_role": "global_view",
                    "start_sec": 0.0,
                    "end_sec": float(duration_sec),
                    "nframes": int(resolved_nframes),
                    "max_pixels": int(max_pixels),
                    "sample_offset_sec": float(sample_offset_sec),
                    "grounding_quality": "global_sparse",
                }
            ],
            "limitations": "Sparse whole-video sampling; fine local details may require follow-up inspection.",
            **raw_fields,
            "raw_output": raw_fields,
        }

    registry.register(global_gist)
    return registry


def _global_gist_prompt(*, question: str, duration_sec: float, sample_offset_sec: float = 0.0) -> str:
    return (
        "Answer from a sparse full-video view before any local decomposition.\n"
        "Describe the apparent whole-video topic, coverage, and uncertainty.\n"
        "Do not choose an option or emit supported_option; any option-like text is only a candidate hint.\n"
        "Mention if the sparse global view is insufficient for fine local details.\n"
        f"Video duration: {float(duration_sec):.1f} seconds.\n"
        f"Sampling offset: {float(sample_offset_sec):.3f} seconds.\n"
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
