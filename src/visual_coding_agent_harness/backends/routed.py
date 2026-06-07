"""Backend router for text-only and vision-language requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BackendRequest, BackendResponse


TEXT_TASKS = {
    "replan",
    "answer_from_evidence",
    "verify_from_evidence",
    "summarize_subtitle_segment",
}

VISION_TASKS = {
    "caption_scene_segment",
    "caption_segment",
    "caption_video",
    "caption_frames",
    "direct_description",
    "global_gist",
    "inspect_segment",
    "qa_region",
    "qa_video",
    "vision_read",
}


@dataclass
class RoutedBackend:
    text_backend: Any
    vl_backend: Any

    def generate(self, request: BackendRequest) -> BackendResponse:
        route_name = self._route_name(request)
        backend = self.text_backend if route_name == "text" else self.vl_backend
        response = backend.generate(request)
        raw = dict(response.raw)
        raw["route_backend"] = route_name
        return BackendResponse(text=response.text, raw=raw)

    def _route_name(self, request: BackendRequest) -> str:
        if request.media_path or request.frames:
            return "vl"
        if request.task in TEXT_TASKS:
            return "text"
        if request.task in VISION_TASKS:
            return "vl"
        return "vl"
