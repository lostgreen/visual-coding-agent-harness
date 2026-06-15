"""Generic tool lifecycle context and hook result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from ...protocol import ToolRequest, ToolResult
from ...registry import ToolRegistry
from .state import RoundState, RunState


TraceWriter = Callable[[str, Mapping[str, Any]], None]
ObservationWriter = Callable[[Any], None]


@dataclass
class RunContext:
    workspace: Any
    scene_index: Any
    budget: Any
    run_state: RunState
    round_state: RoundState
    registry: ToolRegistry
    skill_runtime: Any | None = None
    evidence_policy: Any | None = None
    issued_tool_calls: int = 0
    seen_tool_semantic_keys: set[str] = field(default_factory=set)
    record_trace: TraceWriter | None = None
    record_observation: ObservationWriter | None = None


@dataclass(frozen=True)
class PreToolDecision:
    rejected: bool = False
    reason: str = ""
    message: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> "PreToolDecision":
        return cls()

    @classmethod
    def reject(cls, reason: str, *, message: str = "", payload: Mapping[str, Any] | None = None) -> "PreToolDecision":
        return cls(rejected=True, reason=reason, message=message, payload=dict(payload or {}))


@dataclass(frozen=True)
class PostToolEffects:
    observation_ids: tuple[str, ...] = ()
    trace_events: tuple[tuple[str, Mapping[str, Any]], ...] = ()


class PreToolHook(Protocol):
    def __call__(self, ctx: RunContext, request: ToolRequest) -> PreToolDecision: ...


class PostToolHook(Protocol):
    def __call__(self, ctx: RunContext, request: ToolRequest, result: ToolResult) -> PostToolEffects: ...


def evaluate_pre_tool_chain(
    hooks: tuple[PreToolHook, ...],
    ctx: RunContext,
    request: ToolRequest,
) -> PreToolDecision:
    for hook in hooks:
        decision = hook(ctx, request)
        if decision.rejected:
            return decision
    return PreToolDecision.allow()


def apply_post_tool_chain(
    hooks: tuple[PostToolHook, ...],
    ctx: RunContext,
    request: ToolRequest,
    result: ToolResult,
) -> tuple[PostToolEffects, ...]:
    return tuple(hook(ctx, request, result) for hook in hooks)


def mark_successful_tool_call(ctx: RunContext, request: ToolRequest) -> None:
    runtime_spec = ctx.registry.get_runtime_spec(request.tool)
    builder = runtime_spec.semantic_key_builder
    if builder is None:
        return
    key = str(builder(ctx, request) or "").strip()
    if key:
        ctx.seen_tool_semantic_keys.add(key)
