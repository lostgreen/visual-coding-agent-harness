"""Evidence promotion post-hook."""

from __future__ import annotations

from typing import Any, Mapping

from ....protocol import ToolRequest, ToolResult
from ..lifecycle import PostToolEffects, RunContext


class EvidencePromotionHook:
    def __call__(self, ctx: RunContext, request: ToolRequest, result: ToolResult) -> PostToolEffects:
        promoter = ctx.registry.get_runtime_spec(request.tool).evidence_promoter
        if promoter is None:
            return PostToolEffects()
        events = _coerce_events(promoter(ctx, request, result))
        if ctx.record_trace is not None:
            for event_type, payload in events:
                ctx.record_trace(event_type, payload)
        return PostToolEffects(trace_events=events)


def _coerce_events(value: Any) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    if value is None:
        return ()
    events: list[tuple[str, Mapping[str, Any]]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        event_type, payload = item
        if isinstance(payload, Mapping):
            events.append((str(event_type), dict(payload)))
    return tuple(events)
