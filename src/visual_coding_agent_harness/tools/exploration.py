"""Combined tool registry for video exploration agents."""

from __future__ import annotations

from typing import Optional

from ..backends.base import VisionLanguageBackend
from ..registry import ToolRegistry
from ..video_map import IndexRefiner, VideoMap, VideoMapStore
from ..workspace import EvidenceWorkspace
from .asr_binding import build_asr_binding_registry
from .enrichment import build_video_enrichment_registry
from .global_view import build_global_view_registry
from .inspector import build_segment_inspector_registry
from .navigation import build_video_navigation_registry
from .query_context import build_query_context_registry
from .frame_cache import FrameSampler
from .runtime_specs import install_video_runtime_specs
from .segments import ClipExtractor, build_segment_vlm_registry
from .timeline import build_timeline_registry
from .verification import build_verification_registry
from .workspace_primitives import build_workspace_primitives_registry
from .workspace_v2 import build_workspace_v2_registry


def build_video_exploration_registry(
    *,
    video_map: VideoMap | VideoMapStore,
    backend: VisionLanguageBackend,
    workspace: Optional[EvidenceWorkspace] = None,
    extract_clips: bool = False,
    clip_extractor: Optional[ClipExtractor] = None,
    frame_sampler: Optional[FrameSampler] = None,
) -> ToolRegistry:
    video_map_store = video_map if isinstance(video_map, VideoMapStore) else VideoMapStore(video_map)
    registry = ToolRegistry()
    _extend_without(
        registry,
        build_video_navigation_registry(video_map_store, workspace=workspace),
        names={"read_segment"},
    )
    registry.extend(build_query_context_registry(video_map=video_map_store, backend=backend, frame_sampler=frame_sampler))
    registry.extend(build_global_view_registry(backend, frame_sampler=frame_sampler))
    registry.extend(
        build_video_enrichment_registry(
            video_map_store=video_map_store,
            backend=backend,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
            frame_sampler=frame_sampler,
        )
    )
    registry.extend(build_timeline_registry(workspace=workspace))
    registry.extend(build_workspace_primitives_registry(workspace=workspace, include=("all",)))
    registry.extend(
        build_workspace_v2_registry(
            video_map=video_map_store,
            backend=backend,
            workspace=workspace,
            index_refiner=IndexRefiner(backend=backend, frame_sampler=frame_sampler),
            frame_sampler=frame_sampler,
            include_workspace_primitives=False,
        )
    )
    registry.extend(build_verification_registry(workspace=workspace))
    registry.extend(build_asr_binding_registry(video_map_store=video_map_store, backend=backend, workspace=workspace))
    registry.extend(
        build_segment_inspector_registry(
            backend,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
            frame_sampler=frame_sampler,
        )
    )
    registry.extend(
        build_segment_vlm_registry(
            backend,
            workspace=workspace,
            extract_clips=extract_clips,
            clip_extractor=clip_extractor,
            frame_sampler=frame_sampler,
        )
    )
    return install_video_runtime_specs(registry, required=True)


def _extend_without(registry: ToolRegistry, other: ToolRegistry, *, names: set[str]) -> None:
    for runtime_spec in other.list_runtime_specs():
        if runtime_spec.tool_spec.name in names:
            continue
        registry.register(runtime_spec)
