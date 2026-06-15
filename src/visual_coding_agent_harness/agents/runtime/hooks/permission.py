"""Permission pre-hook driven by tool runtime metadata."""

from __future__ import annotations

from ....protocol import ToolRequest
from ..lifecycle import PreToolDecision, RunContext


class PermissionHook:
    def __call__(self, ctx: RunContext, request: ToolRequest) -> PreToolDecision:
        policy = ctx.evidence_policy
        forbidden = set(getattr(policy, "forbidden_actions", ()) or ())
        if request.tool in forbidden:
            return PreToolDecision.reject(
                "permission_denied",
                message=f"{request.tool} is forbidden by the active evidence policy.",
                payload={"tool": request.tool},
            )
        runtime_spec = ctx.registry.get_runtime_spec(request.tool)
        predicate = runtime_spec.permission_predicate
        if predicate is not None and not bool(predicate(ctx, request)):
            return PreToolDecision.reject(
                "permission_denied",
                message=f"{request.tool} was denied by its runtime permission predicate.",
                payload={"tool": request.tool},
            )
        return PreToolDecision.allow()
