"""Query-scoped global context tool for long-video exploration."""

from __future__ import annotations

from typing import Literal, Mapping, Optional

from ..agents.contracts import resolve_nframes
from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolRegistry, tool
from ..video_map import VideoMap, VideoMapStore


DEFAULT_MAX_PIXELS = 151200
QueryContextScope = Literal["full_video", "route_relevant"]


def build_query_context_registry(
    *,
    video_map: VideoMap | VideoMapStore,
    backend: VisionLanguageBackend,
) -> ToolRegistry:
    video_map_store = video_map if isinstance(video_map, VideoMapStore) else VideoMapStore(video_map)
    registry = ToolRegistry()

    @tool(name="query_context", description="Build a query-scoped global context capsule without option voting.")
    def query_context(
        video_path: str,
        query: str,
        duration_sec: Optional[float] = None,
        scope: QueryContextScope = "full_video",
        nframes: int | None = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> Mapping[str, object]:
        resolved_nframes, budget_reason = resolve_nframes(nframes)
        current = video_map_store.current
        resolved_duration = float(duration_sec if duration_sec is not None else current.duration_sec)
        regions = _regions_for_scope(video_map_store=video_map_store, query=query, scope=scope)
        response = backend.generate(
            BackendRequest(
                task="query_context",
                prompt=_query_context_prompt(query=query, scope=scope),
                media_path=video_path,
                media_type="video",
                max_new_tokens=256,
                metadata={
                    "query": query,
                    "scope": scope,
                    "nframes": int(resolved_nframes),
                    "budget_reason": budget_reason,
                    "duration_sec": resolved_duration,
                    "max_pixels": int(max_pixels),
                    "candidate_segments": [str(region.get("segment_id", "")) for region in regions],
                },
            )
        )
        capsule = response.text.strip()
        raw_fields = {
            "grounding_quality": "query_global_context",
            "time_range": [0.0, resolved_duration],
            "nframes": int(resolved_nframes),
            "budget_reason": budget_reason,
            "max_pixels": int(max_pixels),
            "scope": scope,
            "query": query,
            "raw_backend": dict(response.raw),
        }
        return {
            "claim": _format_capsule(capsule=capsule, query=query),
            "confidence": 0.62,
            "input_artifacts": [f"{video_path}#query_context={scope}"],
            "regions": regions
            or [
                {
                    "tool_role": "query_context",
                    "segment_id": None,
                    "start_sec": 0.0,
                    "end_sec": resolved_duration,
                    "nframes": int(resolved_nframes),
                    "max_pixels": int(max_pixels),
                    "grounding_quality": "query_global_context",
                }
            ],
            "limitations": "Global query context capsule; not sole support for final answers.",
            **raw_fields,
            "raw_output": raw_fields,
        }

    registry.register(query_context)
    return registry


def _regions_for_scope(
    *,
    video_map_store: VideoMapStore,
    query: str,
    scope: QueryContextScope,
) -> list[dict[str, object]]:
    current = video_map_store.current
    if scope == "route_relevant":
        results = current.search(query=query, top_k=5)
        regions = [
            {
                "tool_role": "query_context",
                "segment_id": result.segment.segment_id,
                "start_sec": float(result.segment.start_sec),
                "end_sec": float(result.segment.end_sec),
                "score": float(result.score),
                "grounding_quality": "query_global_context",
            }
            for result in results
        ]
        if regions:
            return regions
    return [
        {
            "tool_role": "query_context",
            "segment_id": None,
            "start_sec": 0.0,
            "end_sec": float(current.duration_sec),
            "grounding_quality": "query_global_context",
        }
    ]


def _query_context_prompt(*, query: str, scope: QueryContextScope) -> str:
    return (
        "Build a compact query-relevant context capsule from sampled video context.\n"
        "Do not choose an option. Do not emit supported_option, answer_option, or final_answer.\n"
        "Return only scene/context facts that may guide later grounded inspection.\n"
        "Mark uncertainty when the global capsule is insufficient.\n"
        f"Scope: {scope}\n"
        f"Query:\n{query}"
    )


def _format_capsule(*, capsule: str, query: str) -> str:
    text = " ".join(capsule.split())
    return f"Query context for '{query}': {text}" if text else f"Query context for '{query}' is empty."
