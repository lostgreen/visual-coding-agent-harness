"""Combined tool registry for video exploration agents."""

from __future__ import annotations

from ..backends.base import VisionLanguageBackend
from ..registry import ToolRegistry
from ..video_map import VideoMap
from .navigation import build_video_navigation_registry
from .segments import build_segment_vlm_registry


def build_video_exploration_registry(*, video_map: VideoMap, backend: VisionLanguageBackend) -> ToolRegistry:
    registry = ToolRegistry()
    registry.extend(build_video_navigation_registry(video_map))
    registry.extend(build_segment_vlm_registry(backend))
    return registry
