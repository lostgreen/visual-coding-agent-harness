"""Visual program interpreter.

The interpreter intentionally mirrors the useful part of VisProg: a program is
an ordered list of module calls. The difference is that every call also writes a
coding-agent-style trace and evidence ledger entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .registry import ToolRegistry
from .workspace import EvidenceWorkspace


@dataclass(frozen=True)
class ProgramResult:
    observation_ids: Sequence[str]
    assignments: Mapping[str, str] = field(default_factory=dict)


class ProgramInterpreter:
    """Run a simple visual program against a tool registry and workspace."""

    def __init__(self, registry: ToolRegistry, workspace: EvidenceWorkspace) -> None:
        self.registry = registry
        self.workspace = workspace

    def run(self, program: Sequence[Mapping[str, Any]]) -> ProgramResult:
        observation_ids = []
        assignments: Dict[str, str] = {}

        for index, step in enumerate(program, start=1):
            tool_name = str(step["tool"])
            arguments = dict(step.get("args", {}))
            self.workspace.write_trace_event(
                "tool_use",
                {"step": index, "tool": tool_name, "arguments": arguments},
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
                    "step": index,
                    "tool": tool_name,
                    "observation_id": observation.observation_id,
                },
            )
            self.workspace.write_ledger_entry(observation)

            observation_ids.append(observation.observation_id)
            if "assign" in step:
                assignments[str(step["assign"])] = observation.observation_id

        return ProgramResult(observation_ids=observation_ids, assignments=assignments)
