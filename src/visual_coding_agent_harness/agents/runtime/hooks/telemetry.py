"""Telemetry post-hook."""

from __future__ import annotations

from ....protocol import ToolRequest, ToolResult
from ..lifecycle import PostToolEffects, RunContext


class TelemetryHook:
    def __call__(self, ctx: RunContext, request: ToolRequest, result: ToolResult) -> PostToolEffects:
        payload = {
            "tool": request.tool,
            "request_id": request.request_id,
            "claim_chars": len(result.claim),
            "confidence": result.confidence,
        }
        semantic_key = _semantic_key(ctx, request)
        if semantic_key:
            payload["semantic_key"] = semantic_key
        if ctx.record_trace is not None:
            ctx.record_trace("tool_lifecycle_result", payload)
        return PostToolEffects(trace_events=(("tool_lifecycle_result", payload),))


def _semantic_key(ctx: RunContext, request: ToolRequest) -> str:
    builder = ctx.registry.get_runtime_spec(request.tool).semantic_key_builder
    if builder is None:
        return ""
    return str(builder(ctx, request) or "").strip()
