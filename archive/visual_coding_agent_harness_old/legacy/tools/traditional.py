"""Registry builder for deterministic traditional tools."""

from __future__ import annotations

from typing import Mapping, Sequence

from visual_coding_agent_harness.legacy.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.legacy.tools import image_atomic, video_atomic


@tool(name="crop_region", description="Crop an image region using normalized [0,1000] bbox.")
def crop_region(image_path: str, bbox: Sequence[int], output_path: str) -> Mapping[str, object]:
    return image_atomic.crop_region(image_path=image_path, bbox=bbox, output_path=output_path)


@tool(name="zoom_region", description="Crop and resize an image region using normalized [0,1000] bbox.")
def zoom_region(
    image_path: str,
    bbox: Sequence[int],
    output_path: str,
    scale: float = 2.0,
) -> Mapping[str, object]:
    return image_atomic.zoom_region(
        image_path=image_path,
        bbox=bbox,
        output_path=output_path,
        scale=scale,
    )


@tool(name="threshold_image", description="Apply deterministic grayscale thresholding.")
def threshold_image(image_path: str, output_path: str, threshold: int = 128) -> Mapping[str, object]:
    return image_atomic.threshold_image(
        image_path=image_path,
        output_path=output_path,
        threshold=threshold,
    )


@tool(name="enhance_image", description="Apply deterministic sharpness and contrast enhancement.")
def enhance_image(
    image_path: str,
    output_path: str,
    sharpness: float = 1.0,
    contrast: float = 1.0,
) -> Mapping[str, object]:
    return image_atomic.enhance_image(
        image_path=image_path,
        output_path=output_path,
        sharpness=sharpness,
        contrast=contrast,
    )


@tool(name="edge_detect", description="Apply deterministic image edge detection.")
def edge_detect(image_path: str, output_path: str) -> Mapping[str, object]:
    return image_atomic.edge_detect(image_path=image_path, output_path=output_path)


@tool(name="sample_frames", description="Sample video frames with ffmpeg.")
def sample_frames(
    video_path: str,
    output_pattern: str,
    fps: float = 1.0,
    start_time: float | None = None,
    duration: float | None = None,
) -> Mapping[str, object]:
    return video_atomic.sample_frames(
        video_path=video_path,
        output_pattern=output_pattern,
        fps=fps,
        start_time=start_time,
        duration=duration,
    )


@tool(name="extract_clip", description="Extract a video clip with ffmpeg.")
def extract_clip(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float,
) -> Mapping[str, object]:
    return video_atomic.extract_clip(
        video_path=video_path,
        output_path=output_path,
        start_time=start_time,
        duration=duration,
    )


def build_traditional_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for spec in [
        crop_region,
        zoom_region,
        threshold_image,
        enhance_image,
        edge_detect,
        sample_frames,
        extract_clip,
    ]:
        registry.register(spec)
    return registry
