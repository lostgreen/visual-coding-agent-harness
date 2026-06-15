"""Generic planner-program normalization using tool runtime specs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from ...protocol import ToolRequest
from ...registry import ToolError, ToolRegistry
from .lifecycle import RunContext


class ProgramNormalizer:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def normalize(self, program: Any, *, ctx: RunContext) -> tuple[ToolRequest, ...]:
        if not isinstance(program, list):
            raise ValueError("Planner action status=continue requires a list program")
        requests: list[ToolRequest] = []
        max_calls = int(getattr(ctx.budget, "max_tool_calls_per_round", len(program)) or len(program))
        for index, step in enumerate(program):
            if len(requests) >= max_calls:
                break
            if not isinstance(step, Mapping):
                raise ValueError("Planner program steps must be objects")
            tool_name = str(step.get("tool") or step.get("op") or "").strip()
            if not tool_name:
                raise ValueError("Planner program step is missing required 'tool'")
            args = step.get("args", {})
            if not isinstance(args, Mapping):
                raise ValueError(f"Planner program step args must be an object for {tool_name}")
            request = ToolRequest(tool=tool_name, arguments=dict(args), request_id=str(step.get("request_id", index)))
            try:
                runtime_spec = self.registry.get_runtime_spec(tool_name)
            except ToolError as exc:
                raise ValueError(str(exc)) from exc
            normalizer = runtime_spec.argument_normalizer
            if normalizer is not None:
                normalized_args = normalizer(ctx, request)
                if not isinstance(normalized_args, Mapping):
                    raise ValueError(f"Argument normalizer for {tool_name} must return a mapping")
                request = ToolRequest(
                    tool=request.tool,
                    arguments=dict(normalized_args),
                    request_id=request.request_id,
                    caller=request.caller,
                )
            requests.append(request)
        return tuple(requests)
