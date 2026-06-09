"""VLM-backed visual tools.

These tools deliberately accept a backend instance instead of constructing a
model internally. That lets a main VLM agent and its tools share one loaded
foundation model during smoke tests and later benchmarks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from ..agents.open_questions import exploration_question
from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from ..workspace import EvidenceWorkspace
from . import image_atomic


Cropper = Callable[[str, Sequence[int], str], Mapping[str, object]]


def build_vlm_registry(
    backend: VisionLanguageBackend,
    *,
    workspace: Optional[EvidenceWorkspace] = None,
    cropper: Optional[Cropper] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    @tool(name="caption_image", description="Caption an image using the shared VLM backend.")
    def caption_image(image_path: str, question: str = "Describe the image.") -> Mapping[str, object]:
        prompt_question = exploration_question(question)
        return _run_vlm_tool(
            backend=backend,
            task="caption_image",
            media_path=image_path,
            media_type="image",
            prompt=_caption_qa_prompt(task="caption_image", question=prompt_question),
            metadata={"task": "caption_image", "question": prompt_question},
        )

    @tool(name="qa_image", description="Answer an image question using the shared VLM backend.")
    def qa_image(image_path: str, question: str) -> Mapping[str, object]:
        prompt_question = exploration_question(question)
        return _run_vlm_tool(
            backend=backend,
            task="qa_image",
            media_path=image_path,
            media_type="image",
            prompt=_caption_qa_prompt(task="qa_image", question=prompt_question),
            metadata={"task": "qa_image", "question": prompt_question},
        )

    @tool(name="caption_region", description="Crop and caption one image region using the shared VLM backend.")
    def caption_region(
        image_path: str,
        bbox: Sequence[int],
        question: str = "Describe this image region.",
    ) -> Mapping[str, object]:
        prompt_question = exploration_question(question)
        crop_path = _materialize_region_crop(
            image_path=image_path,
            bbox=bbox,
            workspace=workspace,
            cropper=cropper,
        )
        metadata = {
            "task": "caption_region",
            "question": prompt_question,
            "source_image_path": image_path,
            "bbox": list(bbox),
            "crop_path": crop_path,
        }
        return _run_vlm_tool(
            backend=backend,
            task="caption_region",
            media_path=crop_path,
            media_type="image",
            prompt=_caption_qa_prompt(task="caption_region", question=prompt_question),
            metadata=metadata,
            input_artifacts=[crop_path],
        )

    @tool(name="qa_region", description="Crop and answer a question about one image region using the shared VLM backend.")
    def qa_region(image_path: str, bbox: Sequence[int], question: str) -> Mapping[str, object]:
        prompt_question = exploration_question(question)
        crop_path = _materialize_region_crop(
            image_path=image_path,
            bbox=bbox,
            workspace=workspace,
            cropper=cropper,
        )
        metadata = {
            "task": "qa_region",
            "question": prompt_question,
            "source_image_path": image_path,
            "bbox": list(bbox),
            "crop_path": crop_path,
        }
        return _run_vlm_tool(
            backend=backend,
            task="qa_region",
            media_path=crop_path,
            media_type="image",
            prompt=_caption_qa_prompt(task="qa_region", question=prompt_question),
            metadata=metadata,
            input_artifacts=[crop_path],
        )

    @tool(name="caption_video", description="Caption a video using the shared VLM backend.")
    def caption_video(
        video_path: str,
        question: str = "Describe the video.",
        nframes: int = 8,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
    ) -> Mapping[str, object]:
        prompt_question = exploration_question(question)
        return _run_vlm_tool(
            backend=backend,
            task="caption_video",
            media_path=video_path,
            media_type="video",
            prompt=_caption_qa_prompt(task="caption_video", question=prompt_question),
            metadata={
                "task": "caption_video",
                "question": prompt_question,
                **_video_metadata(nframes=nframes, max_pixels=max_pixels, fps=fps),
            },
        )

    @tool(name="qa_video", description="Answer a video question using the shared VLM backend.")
    def qa_video(
        video_path: str,
        question: str,
        nframes: int = 8,
        max_pixels: int = 360 * 420,
        fps: float = 0.0,
    ) -> Mapping[str, object]:
        prompt_question = exploration_question(question)
        return _run_vlm_tool(
            backend=backend,
            task="qa_video",
            media_path=video_path,
            media_type="video",
            prompt=_caption_qa_prompt(task="qa_video", question=prompt_question),
            metadata={
                "task": "qa_video",
                "question": prompt_question,
                **_video_metadata(nframes=nframes, max_pixels=max_pixels, fps=fps),
            },
        )

    registry.register(caption_image)
    registry.register(qa_image)
    registry.register(caption_region)
    registry.register(qa_region)
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
    metadata: Mapping[str, object] | None = None,
    input_artifacts: Sequence[str] | None = None,
) -> Mapping[str, object]:
    resolved_metadata = dict(metadata or {})
    response = backend.generate(
        BackendRequest(
            task=task,
            prompt=prompt,
            media_path=media_path,
            media_type=media_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            metadata=resolved_metadata,
        )
    )
    return {
        "claim": response.text.strip(),
        "confidence": confidence,
        "input_artifacts": list(input_artifacts or [media_path]),
        "regions": [resolved_metadata],
        "limitations": "VLM-generated observation; verify with atomic tools for high-stakes claims.",
        "raw_backend": dict(response.raw),
    }


def _video_metadata(*, nframes: int, max_pixels: int, fps: float) -> Mapping[str, object]:
    metadata: dict[str, object] = {"nframes": nframes, "max_pixels": max_pixels}
    if fps > 0:
        metadata["fps"] = fps
    return metadata


def _caption_qa_prompt(*, task: str, question: str) -> str:
    mode = "Caption task" if task.startswith("caption") else "QA task"
    media_hint = "video" if task.endswith("video") else "image"
    region_hint = " If this is a region tool, focus only on the cropped region." if task.endswith("region") else ""
    return (
        f"{mode}: use only visible evidence from the provided {media_hint}.\n"
        "Do not invent details, identities, text, or temporal order that are not supported.\n"
        "Mention uncertainty when evidence is ambiguous or too low resolution.\n"
        f"{region_hint}\n"
        f"Question: {question}"
    )


def _materialize_region_crop(
    *,
    image_path: str,
    bbox: Sequence[int],
    workspace: Optional[EvidenceWorkspace],
    cropper: Optional[Cropper],
) -> str:
    if workspace is None:
        raise ValueError("caption_region/qa_region require a workspace to store crop artifacts")
    bbox_values = [int(value) for value in bbox]
    if len(bbox_values) != 4:
        raise ValueError("bbox must contain four normalized coordinates")
    output_path = _region_crop_path(workspace=workspace, image_path=image_path, bbox=bbox_values)
    active_cropper = cropper or _default_cropper
    active_cropper(image_path, bbox_values, str(output_path))
    return str(output_path)


def _region_crop_path(*, workspace: EvidenceWorkspace, image_path: str, bbox: Sequence[int]) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_path).stem) or "image"
    bbox_text = "_".join(str(int(value)) for value in bbox)
    return workspace.root / "artifacts" / "crops" / f"{stem}_region_{bbox_text}.png"


def _default_cropper(image_path: str, bbox: Sequence[int], output_path: str) -> Mapping[str, object]:
    return image_atomic.crop_region(image_path=image_path, bbox=bbox, output_path=output_path)
