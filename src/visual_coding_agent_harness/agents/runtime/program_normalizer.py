"""Generic planner-program normalization using tool runtime specs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from ...protocol import ToolRequest
from ...registry import ToolError, ToolRegistry
from .lifecycle import RunContext

_PROGRAM_KEY_IGNORED_STEP_FIELDS = frozenset({"assign", "trace_id", "observation_id", "request_id", "caller"})
_PROGRAM_KEY_IGNORED_ARG_FIELDS = frozenset(
    {
        "video_path",
        "media_path",
        "image_path",
        "question",
        "raw_question",
        "vlm_safe_question",
        "prompt",
        "nframes",
        "fps",
        "temperature",
    }
)
_PROGRAM_KEY_ORDER_INSENSITIVE_FIELDS = frozenset(
    {"targets", "target_refs", "target_ids", "aliases", "candidate_options", "evidence_ids"}
)


@dataclass(frozen=True)
class ProgramKey:
    """Stable semantic key for detecting repeated planner actions."""

    value: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_program(cls, program: Sequence[Mapping[str, Any]]) -> "ProgramKey":
        return cls(tuple(_program_step_key(step) for step in program if isinstance(step, Mapping)))

    @property
    def fingerprint(self) -> str:
        return json.dumps(self.value, ensure_ascii=True, sort_keys=True, default=str)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(str(step.get("tool", "")) for step in self.value if step.get("tool"))


def program_key_fingerprint(program: Sequence[Mapping[str, Any]]) -> str:
    return ProgramKey.from_program(program).fingerprint


def _program_step_key(step: Mapping[str, Any]) -> Mapping[str, Any]:
    tool_name = str(step.get("tool") or step.get("op") or "").strip()
    args = step.get("args", {})
    return {
        "tool": tool_name,
        "args": _program_key_value(args, parent_key="args") if isinstance(args, Mapping) else {},
    }


def _program_key_value(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _program_key_value(item, parent_key=str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _PROGRAM_KEY_IGNORED_STEP_FIELDS
            and str(key) not in _PROGRAM_KEY_IGNORED_ARG_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [_program_key_value(item, parent_key=parent_key) for item in value]
        if parent_key in _PROGRAM_KEY_ORDER_INSENSITIVE_FIELDS:
            return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True, default=str))
        return normalized
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


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
            tool_name = self.registry.resolve_alias(tool_name)
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
