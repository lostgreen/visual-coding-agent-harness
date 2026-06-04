"""Combined tool registry for video exploration agents."""

from __future__ import annotations

from typing import Optional

from ..backends.base import VisionLanguageBackend
from ..registry import ToolRegistry
from ..video_map import VideoMap, VideoMapStore
from ..workspace import EvidenceWorkspace
from .enrichment import build_video_enrichment_registry
from .inspector import build_segment_inspector_registry
from .navigation import build_video_navigation_registry
from .segments import ClipExtractor, build_segment_vlm_registry
from .verification import build_verification_registry


def build_video_exploration_registry(
    *,
    video_map: VideoMap | VideoMapStore,
    backend: VisionLanguageBackend,
    workspace: Optional[EvidenceWorkspace] = None,
    extract_clips: bool = False,
    clip_extractor: Optional[ClipExtractor] = None,
) -> ToolRegistry:
    video_map_store = video_map if isinstance(video_map, VideoMapStore) else VideoMapStore(video_map)
    registry = ToolRegistry()
    registry.extend(build_video_navigation_registry(video_map_store))
    registry.extend(build_video_enrichment_registry(video_map_store=video_map_store, backend=backend))
    registry.extend(build_verification_registry(workspace=workspace))
    registry.extend(
        build_segment_inspector_registry(
            backend,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
        )
    )
    registry.extend(
        build_segment_vlm_registry(
            backend,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
        )
    )
    return registry
