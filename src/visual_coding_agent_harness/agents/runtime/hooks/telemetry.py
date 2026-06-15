"""Telemetry post-hook."""

from __future__ import annotations

from ....protocol import ToolRequest, ToolResult
from ..lifecycle import PostToolEffects, RunContext


class TelemetryHook:
    def __call__(self, ctx: RunContext, request: ToolRequest, result: ToolResult) -> PostToolEffects:
        payload = {
            "tool": request.tool,
            "args": dict(request.arguments),
            "claim": result.claim,
            "confidence": result.confidence,
        }
        if ctx.record_trace is not None:
            ctx.record_trace("tool_lifecycle_result", payload)
        return PostToolEffects(trace_events=(("tool_lifecycle_result", payload),))
