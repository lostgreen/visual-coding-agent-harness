"""Zero-yield post-hook."""

from __future__ import annotations

from ....protocol import ToolRequest, ToolResult
from ..lifecycle import PostToolEffects, RunContext


class ZeroYieldHook:
    def __call__(self, ctx: RunContext, request: ToolRequest, result: ToolResult) -> PostToolEffects:
        if result.claim.strip() or float(result.confidence) > 0.0:
            return PostToolEffects()
        payload = {
            "tool": request.tool,
            "args": dict(request.arguments),
            "limitations": result.limitations,
        }
        if ctx.record_trace is not None:
            ctx.record_trace("zero_yield_tool_result", payload)
        return PostToolEffects(trace_events=(("zero_yield_tool_result", payload),))
