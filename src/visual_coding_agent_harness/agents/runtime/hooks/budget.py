"""Round budget pre-hook."""

from __future__ import annotations

from ....protocol import ToolRequest
from ..lifecycle import PreToolDecision, RunContext


class BudgetHook:
    def __call__(self, ctx: RunContext, request: ToolRequest) -> PreToolDecision:
        del request
        max_calls = int(getattr(ctx.budget, "max_tool_calls_per_round", 1) or 1)
        if int(ctx.issued_tool_calls) >= max_calls:
            return PreToolDecision.reject(
                "round_tool_budget_exhausted",
                message=f"Round already issued {ctx.issued_tool_calls}/{max_calls} tool calls.",
                payload={"issued_tool_calls": ctx.issued_tool_calls, "max_tool_calls_per_round": max_calls},
            )
        return PreToolDecision.allow()
