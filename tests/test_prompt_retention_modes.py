from __future__ import annotations

from visual_coding_agent_harness.agents.runtime_capabilities import (
    PlannerRuntimeCapabilities,
    PromptRetentionMode,
    planner_prompt_retention_mode,
)


def test_planner_retention_mode_keeps_stateless_backends_full_prompt() -> None:
    caps = PlannerRuntimeCapabilities(prefix_cache=False, persistent_conversation=False)

    assert planner_prompt_retention_mode(caps) is PromptRetentionMode.STATELESS_FULL


def test_planner_retention_mode_does_not_treat_prefix_cache_as_sticky_memory() -> None:
    caps = PlannerRuntimeCapabilities(prefix_cache=True, persistent_conversation=False)

    assert planner_prompt_retention_mode(caps) is PromptRetentionMode.PREFIX_CACHED_FULL


def test_planner_retention_mode_uses_reference_only_for_persistent_conversation() -> None:
    caps = PlannerRuntimeCapabilities(prefix_cache=True, persistent_conversation=True)

    assert planner_prompt_retention_mode(caps) is PromptRetentionMode.STICKY_REFERENCE
