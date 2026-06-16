"""Runtime capability helpers for planner prompt retention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptRetentionMode(str, Enum):
    STATELESS_FULL = "stateless_full"
    PREFIX_CACHED_FULL = "prefix_cached_full"
    STICKY_REFERENCE = "sticky_reference"


@dataclass(frozen=True)
class PlannerRuntimeCapabilities:
    prefix_cache: bool = False
    persistent_conversation: bool = False


def planner_prompt_retention_mode(caps: PlannerRuntimeCapabilities) -> PromptRetentionMode:
    if caps.persistent_conversation:
        return PromptRetentionMode.STICKY_REFERENCE
    if caps.prefix_cache:
        return PromptRetentionMode.PREFIX_CACHED_FULL
    return PromptRetentionMode.STATELESS_FULL
