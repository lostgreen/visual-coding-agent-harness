"""Runtime host that normalizes and dispatches planner tool programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from ...interpreter import ProgramInterpreter, ProgramResult
from ...registry import ToolRegistry
from ...workspace import EvidenceWorkspace
from .hooks.budget import BudgetHook
from .hooks.duplicate_guard import DuplicateGuardHook
from .hooks.evidence_promotion import EvidencePromotionHook
from .hooks.observation_adapter import ObservationAdapterHook
from .hooks.permission import PermissionHook
from .hooks.telemetry import TelemetryHook
from .hooks.zero_yield import ZeroYieldHook
from .lifecycle import PostToolHook, PreToolHook, RunContext
from .program_normalizer import ProgramNormalizer


class ToolRuntimeHost:
    """Single entry point for runtime-spec aware tool execution."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        workspace: EvidenceWorkspace,
        program_normalizer: ProgramNormalizer | None = None,
        pre_tool_hooks: Sequence[PreToolHook] | None = None,
        post_tool_hooks: Sequence[PostToolHook] | None = None,
    ) -> None:
        self.registry = registry
        self.workspace = workspace
        self.program_normalizer = program_normalizer or ProgramNormalizer(registry)
        self.pre_tool_hooks = tuple(pre_tool_hooks or (PermissionHook(), DuplicateGuardHook(), BudgetHook()))
        self.post_tool_hooks = tuple(
            post_tool_hooks
            or (ObservationAdapterHook(), ZeroYieldHook(), EvidencePromotionHook(), TelemetryHook())
        )

    def run(
        self,
        program: Sequence[Mapping[str, Any]],
        *,
        ctx: RunContext,
        slots: Mapping[str, Any] | None = None,
        sufficiency_predicate: Callable[[EvidenceWorkspace, Mapping[str, str]], bool] | None = None,
    ) -> ProgramResult:
        normalized_program = self.normalize_program(program, ctx=ctx)
        return ProgramInterpreter(
            registry=self.registry,
            workspace=self.workspace,
            lifecycle_context=ctx,
            pre_tool_hooks=self.pre_tool_hooks,
            post_tool_hooks=self.post_tool_hooks,
        ).run(
            normalized_program,
            slots=slots,
            sufficiency_predicate=sufficiency_predicate,
        )

    def normalize_program(
        self,
        program: Sequence[Mapping[str, Any]],
        *,
        ctx: RunContext,
    ) -> list[dict[str, Any]]:
        requests = self.program_normalizer.normalize(list(program), ctx=ctx)
        normalized: list[dict[str, Any]] = []
        for request, step in zip(requests, program):
            next_step = dict(step)
            next_step.pop("op", None)
            next_step["tool"] = request.tool
            next_step["args"] = dict(request.arguments)
            normalized.append(next_step)
        return normalized
