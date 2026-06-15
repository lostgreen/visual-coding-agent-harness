"""Duplicate tool-call guard driven by semantic keys."""

from __future__ import annotations

from ....protocol import ToolRequest
from ....registry import DuplicateGuardPolicy
from ..lifecycle import PreToolDecision, RunContext


class DuplicateGuardHook:
    def __call__(self, ctx: RunContext, request: ToolRequest) -> PreToolDecision:
        runtime_spec = ctx.registry.get_runtime_spec(request.tool)
        if runtime_spec.duplicate_guard_policy is DuplicateGuardPolicy.OFF:
            return PreToolDecision.allow()
        builder = runtime_spec.semantic_key_builder
        if builder is None:
            return PreToolDecision.allow()
        key = str(builder(ctx, request) or "").strip()
        if not key:
            return PreToolDecision.allow()
        if key in ctx.seen_tool_semantic_keys:
            if runtime_spec.duplicate_guard_policy is DuplicateGuardPolicy.STRICT:
                return PreToolDecision.reject(
                    "duplicate_tool_call",
                    message=f"{request.tool} repeats semantic key {key}.",
                    payload={"tool": request.tool, "semantic_key": key},
                )
            return PreToolDecision.allow()
        ctx.seen_tool_semantic_keys.add(key)
        return PreToolDecision.allow()
