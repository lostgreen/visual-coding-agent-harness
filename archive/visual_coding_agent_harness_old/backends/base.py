"""Common foundation-model backend protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class BackendRequest:
    task: str
    prompt: str
    system_prompt: str = ""
    media_path: Optional[str] = None
    media_type: Optional[str] = None
    frames: Sequence[str] = field(default_factory=list)
    max_new_tokens: int = 256
    temperature: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendResponse:
    text: str
    raw: Mapping[str, Any] = field(default_factory=dict)


class VisionLanguageBackend(Protocol):
    """A model adapter that can answer text, image, or video prompts."""

    def generate(self, request: BackendRequest) -> BackendResponse:
        """Generate text for a multimodal backend request."""
