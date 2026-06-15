"""Permission pre-hook driven by tool runtime metadata."""

from __future__ import annotations

from ....protocol import ToolRequest
from ..lifecycle import PreToolDecision, RunContext


class PermissionHook:
    def __call__(self, ctx: RunContext, request: ToolRequest) -> PreToolDecision:
        effective_skill = getattr(ctx.skill_runtime, "effective_skill", None)
        allowed_actions = set(getattr(effective_skill, "allowed_actions", ()) or ())
        has_playbook = getattr(effective_skill, "playbook", None) is not None
        if not has_playbook and allowed_actions and request.tool not in allowed_actions:
            skill_name = str(getattr(effective_skill, "name", "") or "")
            return PreToolDecision.reject(
                "tool_not_allowed_by_active_skill",
                message=f"{request.tool} is not allowed by the active skill{f' {skill_name}' if skill_name else ''}.",
                payload={"tool": request.tool, "skill": skill_name, "allowed_actions": sorted(allowed_actions)},
            )
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
