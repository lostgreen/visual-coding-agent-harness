"""Visual program interpreter.

The interpreter intentionally mirrors the useful part of VisProg: a program is
an ordered list of module calls. The difference is that every call also writes a
coding-agent-style trace and evidence ledger entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Callable, Dict, Mapping, Sequence

from .registry import ToolRegistry
from .workspace import EvidenceWorkspace


@dataclass(frozen=True)
class ProgramResult:
    observation_ids: Sequence[str]
    assignments: Mapping[str, str] = field(default_factory=dict)
    stopped_by_sufficiency: bool = False


class ProgramInterpreter:
    """Run a simple visual program against a tool registry and workspace."""

    def __init__(self, registry: ToolRegistry, workspace: EvidenceWorkspace) -> None:
        self.registry = registry
        self.workspace = workspace

    def run(
        self,
        program: Sequence[Mapping[str, Any]],
        *,
        slots: Mapping[str, Any] | None = None,
        sufficiency_predicate: Callable[[EvidenceWorkspace, Mapping[str, str]], bool] | None = None,
    ) -> ProgramResult:
        observation_ids = []
        assignments: Dict[str, str] = {}
        slot_values = dict(slots or {})
        stopped_by_sufficiency = False

        for index, step in enumerate(program, start=1):
            expanded_steps = _expand_foreach_step(step, slot_values)
            for expanded_step in expanded_steps:
                observation_id = self._run_step(
                    step_index=index,
                    step=expanded_step,
                    assignments=assignments,
                )
                observation_ids.append(observation_id)
                if sufficiency_predicate is None:
                    continue
                if sufficiency_predicate(self.workspace, assignments):
                    stopped_by_sufficiency = True
                    self.workspace.write_trace_event(
                        "sufficiency_stop",
                        {
                            "step": index,
                            "observation_id": observation_id,
                            "assignments": dict(assignments),
                        },
                    )
                    return ProgramResult(
                        observation_ids=observation_ids,
                        assignments=assignments,
                        stopped_by_sufficiency=stopped_by_sufficiency,
                    )

        return ProgramResult(
            observation_ids=observation_ids,
            assignments=assignments,
            stopped_by_sufficiency=stopped_by_sufficiency,
        )

    def _run_step(
        self,
        *,
        step_index: int,
        step: Mapping[str, Any],
        assignments: Dict[str, str],
    ) -> str:
        if "tool" not in step and "op" not in step:
            raise ValueError(f"Program step {step_index} is missing required 'tool'")
        tool_name = str(step.get("tool") or step.get("op"))
        arguments = dict(step.get("args", {}))
        self.workspace.write_trace_event(
            "tool_use",
            {"step": step_index, "tool": tool_name, "arguments": arguments},
        )

        raw_output = self.registry.execute(tool_name, arguments)
        observation = self.workspace.write_observation(
            tool_name=tool_name,
            input_artifacts=raw_output.get("input_artifacts", []),
            claim=str(raw_output.get("claim", "")),
            confidence=float(raw_output.get("confidence", 0.0)),
            regions=raw_output.get("regions", []),
            limitations=str(raw_output.get("limitations", "")),
            raw_output=raw_output,
        )
        self.workspace.write_trace_event(
            "tool_result",
            {
                "step": step_index,
                "tool": tool_name,
                "observation_id": observation.observation_id,
            },
        )
        self.workspace.write_ledger_entry(observation)

        if "assign" in step:
            assignments[str(step["assign"])] = observation.observation_id
        return observation.observation_id


def _expand_foreach_step(step: Mapping[str, Any], slots: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    foreach = step.get("foreach")
    if foreach is None:
        return [dict(step)]
    slot_name = str(foreach)
    values = slots.get(slot_name, [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"foreach slot '{slot_name}' must be a sequence")
    loop_name = str(step.get("as") or _singular_slot_name(slot_name))
    expanded = []
    for item in values:
        context = dict(slots)
        context[loop_name] = item
        context[f"{slot_name}_item"] = item
        expanded_step = {
            key: _format_value(value, context)
            for key, value in step.items()
            if key not in {"foreach", "as"}
        }
        expanded.append(expanded_step)
    return expanded


def _format_value(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _format_template(value, context)
    if isinstance(value, Mapping):
        return {key: _format_value(child, context) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_format_value(child, context) for child in value]
    return value


def _format_template(template: str, context: Mapping[str, Any]) -> str:
    needed = [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name]
    if not needed:
        return template
    values = {name: context.get(name, "{" + name + "}") for name in needed}
    return template.format(**values)


def _singular_slot_name(slot_name: str) -> str:
    if slot_name.endswith("ies") and len(slot_name) > 3:
        return f"{slot_name[:-3]}y"
    if slot_name.endswith("s") and len(slot_name) > 1:
        return slot_name[:-1]
    return "item"
