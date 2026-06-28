"""Small tool execution adapter for multi-agent investigators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...core.protocol import ToolRequest
from ...core.registry import ToolError, ToolRegistry
from ...workspace import EvidenceWorkspace
from ..workspace_agent import _caption_fact_writes, _mapping_items, _retrieval_candidate_writes, _structured_verify_writes


@dataclass
class _RunnerContext:
    workspace: EvidenceWorkspace
    registry: ToolRegistry
    round_number: int
    seen_tool_semantic_keys: set[str]
    issued_tool_calls: int = 0
    scene_index: Any | None = None
    budget: Any | None = None
    skill_runtime: Any | None = None
    evidence_policy: Any | None = None
    record_trace: Any | None = None

    def increment_tool_calls(self, count: int = 1) -> None:
        self.issued_tool_calls += int(count)


@dataclass(frozen=True)
class ToolRunOutcome:
    observation_id: str
    raw_output: Mapping[str, Any]
    memory_ids: tuple[str, ...] = ()


class MultiAgentToolRunner:
    """Execute registry tools and persist their observations for an Investigator."""

    def __init__(self, *, registry: ToolRegistry, workspace: EvidenceWorkspace) -> None:
        self.registry = registry
        self.workspace = workspace
        self.seen_tool_semantic_keys: set[str] = set()

    def run_tool(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        round_number: int,
        sub_goal_id: str,
    ) -> ToolRunOutcome:
        request = self._normalize(tool_name, args, round_number=round_number)
        self.workspace.write_trace_event(
            "investigator_tool_invoked",
            {
                "round": round_number,
                "sub_goal_id": sub_goal_id,
                "tool": request.tool,
                "args_summary": dict(request.arguments),
            },
        )
        self.workspace.write_trace_event(
            "tool_use",
            {"step": round_number, "tool": request.tool, "arguments": dict(request.arguments), "agent": "investigator"},
        )
        raw_output = dict(self.registry.execute(request.tool, request.arguments))
        observation = self.workspace.write_observation(
            tool_name=request.tool,
            input_artifacts=raw_output.get("input_artifacts", []),
            claim=str(raw_output.get("claim", "")),
            confidence=float(raw_output.get("confidence", 0.0) or 0.0),
            regions=raw_output.get("regions", []),
            limitations=str(raw_output.get("limitations", "")),
            confidence_signal=str(raw_output.get("confidence_signal", "")),
            raw_output=raw_output,
            frame_set_id=(None if raw_output.get("frame_set_id") is None else str(raw_output.get("frame_set_id"))),
        )
        self.workspace.write_trace_event(
            "tool_result",
            {
                "step": round_number,
                "tool": request.tool,
                "observation_id": observation.observation_id,
                "agent": "investigator",
            },
        )
        memory_ids = self._commit_known_observation(observation.observation_id, raw_output)
        return ToolRunOutcome(observation_id=observation.observation_id, raw_output=raw_output, memory_ids=memory_ids)

    def _normalize(self, tool_name: str, args: Mapping[str, Any], *, round_number: int) -> ToolRequest:
        canonical_name = self.registry.resolve_alias(str(tool_name).strip())
        request = ToolRequest(tool=canonical_name, arguments=dict(args), request_id=f"multi_{canonical_name}")
        runtime_spec = self.registry.get_runtime_spec(canonical_name)
        if runtime_spec.argument_normalizer is None:
            return request
        ctx = _RunnerContext(
            workspace=self.workspace,
            registry=self.registry,
            round_number=round_number,
            seen_tool_semantic_keys=self.seen_tool_semantic_keys,
            record_trace=self.workspace.write_trace_event,
        )
        normalized_args = runtime_spec.argument_normalizer(ctx, request)
        if not isinstance(normalized_args, Mapping):
            raise ValueError(f"Argument normalizer for {canonical_name} must return a mapping")
        return ToolRequest(tool=canonical_name, arguments=dict(normalized_args), request_id=request.request_id)

    def _commit_known_observation(self, observation_id: str, raw_output: Mapping[str, Any]) -> tuple[str, ...]:
        anchors = _mapping_items(raw_output.get("produced_anchors"))
        writes: Mapping[str, Any] = {}
        if str(raw_output.get("mode") or "") == "verify_window":
            writes = _structured_verify_writes(raw_output, anchors=anchors, reason="multi_agent_investigator_commit")
        if not writes:
            writes = _caption_fact_writes(raw_output, anchors=anchors, reason="multi_agent_investigator_commit")
        if not writes and anchors:
            writes = _retrieval_candidate_writes(raw_output, anchors=anchors, reason="multi_agent_investigator_commit")
        if not writes:
            return ()
        before = {entry.entry_id for entry in self.workspace.memory_entries()}
        try:
            self.workspace.commit_observation(observation_id, writes=writes)
        except (ToolError, ValueError) as exc:
            self.workspace.write_trace_event(
                "multi_agent_commit_failed",
                {"observation_id": observation_id, "error": str(exc)},
            )
            return ()
        after = [entry.entry_id for entry in self.workspace.memory_entries() if entry.entry_id not in before]
        self.workspace.write_trace_event(
            "multi_agent_observation_committed",
            {"observation_id": observation_id, "memory_ids": after},
        )
        return tuple(after)
