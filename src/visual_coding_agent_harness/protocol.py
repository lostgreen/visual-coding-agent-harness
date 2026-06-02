"""Structured tool-call protocol shared by agents and tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""
    caller: str = "main_agent"

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolRequest":
        return cls(
            tool=str(payload["tool"]),
            arguments=dict(payload.get("arguments", {})),
            request_id=str(payload.get("request_id", "")),
            caller=str(payload.get("caller", "main_agent")),
        )


@dataclass(frozen=True)
class ToolResult:
    tool: str
    request_id: str
    claim: str
    confidence: float
    input_artifacts: Sequence[str] = field(default_factory=list)
    output_artifacts: Sequence[str] = field(default_factory=list)
    regions: Sequence[Mapping[str, Any]] = field(default_factory=list)
    time_range: Sequence[float] = field(default_factory=list)
    limitations: str = ""
    raw_output: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, request: ToolRequest, output: Mapping[str, Any]) -> "ToolResult":
        return cls(
            tool=request.tool,
            request_id=request.request_id,
            claim=str(output.get("claim", "")),
            confidence=float(output.get("confidence", 0.0)),
            input_artifacts=list(output.get("input_artifacts", [])),
            output_artifacts=list(output.get("output_artifacts", [])),
            regions=list(output.get("regions", [])),
            time_range=list(output.get("time_range", [])),
            limitations=str(output.get("limitations", "")),
            raw_output=dict(output),
        )
