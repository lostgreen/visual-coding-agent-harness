"""Combined tool registry for video exploration agents."""

from __future__ import annotations

from typing import Optional

from ..backends.base import VisionLanguageBackend
from ..registry import ToolRegistry
from ..video_map import VideoMap
from ..workspace import EvidenceWorkspace
from .navigation import build_video_navigation_registry
from .segments import ClipExtractor, build_segment_vlm_registry


def build_video_exploration_registry(
    *,
    video_map: VideoMap,
    backend: VisionLanguageBackend,
    workspace: Optional[EvidenceWorkspace] = None,
    extract_clips: bool = False,
    clip_extractor: Optional[ClipExtractor] = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.extend(build_video_navigation_registry(video_map))
    registry.extend(
        build_segment_vlm_registry(
            backend,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
        )
    )
    return registry
