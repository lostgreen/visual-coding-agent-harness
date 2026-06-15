"""Backend capability declarations used by prompt role-splitting work."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SystemMessageSupport(str, Enum):
    NATIVE = "native"
    PREPEND_TO_USER = "prepend"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BackendCapabilities:
    system_message: SystemMessageSupport
    prefix_cache: bool
    max_context_tokens: int


BACKEND_CAPABILITIES: dict[str, BackendCapabilities] = {
    "openai_chat": BackendCapabilities(
        system_message=SystemMessageSupport.NATIVE,
        prefix_cache=True,
        max_context_tokens=128_000,
    ),
    "qwen_text": BackendCapabilities(
        system_message=SystemMessageSupport.NATIVE,
        prefix_cache=False,
        max_context_tokens=32_768,
    ),
    "qwen_vl": BackendCapabilities(
        system_message=SystemMessageSupport.NATIVE,
        prefix_cache=False,
        max_context_tokens=32_768,
    ),
}
