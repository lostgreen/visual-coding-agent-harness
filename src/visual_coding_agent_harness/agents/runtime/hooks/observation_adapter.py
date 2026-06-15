"""Observation adapter post-hook."""

from __future__ import annotations

from typing import Any

from ....protocol import ToolRequest, ToolResult
from ..lifecycle import PostToolEffects, RunContext


class ObservationAdapterHook:
    def __call__(self, ctx: RunContext, request: ToolRequest, result: ToolResult) -> PostToolEffects:
        adapter = ctx.registry.get_runtime_spec(request.tool).observation_adapter
        if adapter is None:
            return PostToolEffects()
        observation_ids = _coerce_ids(adapter(ctx, request, result))
        return PostToolEffects(observation_ids=observation_ids)


def _coerce_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)
