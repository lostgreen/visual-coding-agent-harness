"""Tool registry for visual harness modules.

This keeps the P0 runtime close to VisProg's module registry idea while using a
coding-agent style tool dispatcher.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .protocol import ToolRequest, ToolResult


class ToolError(Exception):
    """Raised when a tool cannot be executed safely."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Mapping[str, Any]]

    @property
    def parameters(self) -> Mapping[str, inspect.Parameter]:
        return inspect.signature(self.handler).parameters


class DuplicateGuardPolicy(str, Enum):
    STRICT = "strict"
    ADVISORY = "advisory"
    OFF = "off"


@dataclass(frozen=True)
class ToolRuntimeSpec:
    tool_spec: ToolSpec
    argument_normalizer: Any | None = None
    semantic_key_builder: Any | None = None
    duplicate_guard_policy: DuplicateGuardPolicy = DuplicateGuardPolicy.STRICT
    observation_adapter: Any | None = None
    evidence_promoter: Any | None = None
    permission_predicate: Any | None = None


def tool(name: str, description: str) -> Callable[[Callable[..., Mapping[str, Any]]], ToolSpec]:
    """Decorate a Python function as a harness tool."""

    def decorate(handler: Callable[..., Mapping[str, Any]]) -> ToolSpec:
        return ToolSpec(name=name, description=description, handler=handler)

    return decorate


class ToolRegistry:
    """Register and execute named visual tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolRuntimeSpec] = {}

    def register(self, spec: ToolSpec | ToolRuntimeSpec) -> None:
        runtime_spec = spec if isinstance(spec, ToolRuntimeSpec) else ToolRuntimeSpec(tool_spec=spec)
        tool_spec = runtime_spec.tool_spec
        if tool_spec.name in self._tools:
            raise ToolError(f"Tool already registered: {tool_spec.name}")
        self._tools[tool_spec.name] = runtime_spec

    def extend(self, other: "ToolRegistry") -> None:
        for runtime_spec in other.list_runtime_specs():
            self.register(runtime_spec)

    def replace_runtime_spec(self, name: str, **updates: Any) -> None:
        runtime_spec = self.get_runtime_spec(name)
        self._tools[name] = replace(runtime_spec, **updates)

    def list_specs(self) -> Sequence[ToolSpec]:
        return tuple(runtime_spec.tool_spec for runtime_spec in self._tools.values())

    def list_runtime_specs(self) -> Sequence[ToolRuntimeSpec]:
        return tuple(self._tools.values())

    def get(self, name: str) -> ToolSpec:
        return self.get_runtime_spec(name).tool_spec

    def get_runtime_spec(self, name: str) -> ToolRuntimeSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool: {name}") from exc

    def execute(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        spec = self.get(name)
        kwargs = dict(arguments or {})
        self._validate_arguments(spec, kwargs)
        result = spec.handler(**kwargs)
        if not isinstance(result, Mapping):
            raise ToolError(f"Tool {name} must return a mapping, got {type(result).__name__}")
        return result

    def execute_batch(self, requests: Sequence[ToolRequest]) -> Sequence[ToolResult]:
        results = []
        for request in requests:
            output = self.execute(request.tool, request.arguments)
            results.append(ToolResult.from_mapping(request=request, output=output))
        return results

    def _validate_arguments(self, spec: ToolSpec, arguments: Mapping[str, Any]) -> None:
        signature = inspect.signature(spec.handler)
        try:
            signature.bind(**arguments)
        except TypeError as exc:
            raise ToolError(f"Invalid arguments for {spec.name}: {exc}") from exc
