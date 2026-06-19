"""Workspace-first plan/act/commit agent skeleton."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..registry import ToolError, ToolRegistry
from ..workspace import EvidenceWorkspace
from .runtime.host import ToolRuntimeHost
from .runtime.lifecycle import RunContext
from .runtime.state import RoundState, RunState


DISPOSITION_TOOLS = {
    "commit_observation",
    "reject_observation",
    "defer_observation",
    "no_commit_needed",
}


@dataclass(frozen=True)
class WorkspaceRunResult:
    answer: str
    citations: tuple[str, ...] = ()
    confidence: str = ""
    rounds: int = 0
    metadata: Mapping[str, Any] | None = None


class WorkspaceVisualAgent:
    """Small v2 loop: plan, execute one action, commit if required, then repeat."""

    def __init__(
        self,
        *,
        backend: VisionLanguageBackend,
        registry: ToolRegistry,
        workspace: EvidenceWorkspace,
        max_rounds: int = 8,
        video_path: str = "",
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.workspace = workspace
        self.max_rounds = int(max_rounds)
        self.video_path = video_path
        self.runtime_host = ToolRuntimeHost(
            registry=registry,
            workspace=workspace,
            pre_tool_hooks=(),
            post_tool_hooks=(),
        )

    def run(self, question: str) -> WorkspaceRunResult:
        last_tool_result = ""
        for round_number in range(1, self.max_rounds + 1):
            plan_action = self._decide_plan(
                question=question,
                round_number=round_number,
                last_tool_result=last_tool_result,
            )
            if _tool_name(plan_action) == "answer":
                try:
                    return self._finalize_answer(plan_action, question=question, rounds=round_number)
                except (ToolError, ValueError) as exc:
                    last_tool_result = f"answer rejected: {exc}"
                    self.workspace.write_trace_event(
                        "workspace_answer_rejected",
                        {
                            "round": round_number,
                            "error": str(exc),
                            "action": {"tool": _tool_name(plan_action), "args": _action_args(plan_action)},
                        },
                    )
                    continue
            if _tool_name(plan_action) in DISPOSITION_TOOLS:
                raise ValueError("plan phase does not accept disposition tools")

            obs_ids = self._execute_plan_action(plan_action, question=question, round_number=round_number)
            observation_id = obs_ids[-1] if obs_ids else ""
            observation = self.workspace.get_observation(observation_id) if observation_id else None
            if observation is not None:
                last_tool_result = f"{observation.observation_id}: {observation.claim}"

            if observation_id and self._commit_required(_tool_name(plan_action), observation):
                self._run_commit_phase(
                    question=question,
                    observation_id=observation_id,
                    round_number=round_number,
                )
            elif observation_id and self.workspace.observation_status(observation_id) == "uncommitted":
                self.workspace.no_commit_needed(observation_id, reason="tool output did not require durable commit")

        return WorkspaceRunResult(
            answer="need_more_evidence",
            citations=(),
            confidence="low",
            rounds=self.max_rounds,
            metadata={"reason": "max_rounds_reached"},
        )

    def _runtime_context(self, *, question: str, round_number: int) -> RunContext:
        return RunContext(
            workspace=self.workspace,
            scene_index=None,
            budget=None,
            run_state=RunState(question=question, video_path=self.video_path),
            round_state=RoundState(round_number=round_number),
            registry=self.registry,
            record_trace=self.workspace.write_trace_event,
        )

    def _decide_plan(self, *, question: str, round_number: int, last_tool_result: str) -> Mapping[str, Any]:
        response = self.backend.generate(
            BackendRequest(
                task="workspace_plan",
                system_prompt=PLAN_SYSTEM_PROMPT,
                prompt=compose_plan_prompt(
                    question=question,
                    workspace=self.workspace,
                    last_tool_result=last_tool_result,
                ),
                metadata={"round": round_number, "phase": "plan"},
            )
        )
        self.workspace.write_trace_event(
            "workspace_plan_model_io",
            {"round": round_number, "response": response.text[:2000]},
        )
        return _parse_action(response.text)

    def _decide_commit(
        self,
        *,
        question: str,
        observation_id: str,
        round_number: int,
        validation_error: str = "",
        attempt: int = 1,
        prompt_mode: str = "full",
    ) -> Mapping[str, Any]:
        response = self.backend.generate(
            BackendRequest(
                task="workspace_commit",
                system_prompt=COMMIT_SYSTEM_PROMPT,
                prompt=compose_commit_prompt(
                    question=question,
                    workspace=self.workspace,
                    observation_id=observation_id,
                    validation_error=validation_error,
                    attempt=attempt,
                    prompt_mode=prompt_mode,
                ),
                metadata={
                    "round": round_number,
                    "phase": "commit",
                    "observation_id": observation_id,
                    "attempt": attempt,
                    "prompt_mode": prompt_mode,
                },
            )
        )
        self.workspace.write_trace_event(
            "workspace_commit_model_io",
            {"round": round_number, "observation_id": observation_id, "response": response.text[:2000]},
        )
        return _parse_action(response.text)

    def _run_commit_phase(self, *, question: str, observation_id: str, round_number: int) -> None:
        validation_error = ""
        for attempt in range(1, 4):
            prompt_mode = "minimal" if attempt == 3 else "full"
            commit_action = self._decide_commit(
                question=question,
                observation_id=observation_id,
                round_number=round_number,
                validation_error=validation_error,
                attempt=attempt,
                prompt_mode=prompt_mode,
            )
            if _tool_name(commit_action) not in DISPOSITION_TOOLS:
                validation_error = "commit phase only accepts disposition tools"
                self.workspace.write_trace_event(
                    "workspace_commit_validation_error",
                    {
                        "round": round_number,
                        "attempt": attempt,
                        "observation_id": observation_id,
                        "error": validation_error,
                    },
                )
                continue
            try:
                self.registry.execute(_tool_name(commit_action), _action_args(commit_action))
                return
            except (ToolError, ValueError) as exc:
                validation_error = str(exc)
                self.workspace.write_trace_event(
                    "workspace_commit_validation_error",
                    {
                        "round": round_number,
                        "attempt": attempt,
                        "observation_id": observation_id,
                        "error": validation_error,
                    },
                )
        observation = self.workspace.get_observation(observation_id)
        if observation is None:
            raise ValueError(f"observation_validation_failed: unknown observation_id={observation_id}")
        self._auto_pin_observation(
            observation,
            reason=f"commit_format_failure: {validation_error}",
        )

    def _auto_pin_observation(self, observation: Any, *, reason: str) -> None:
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        fact_text = _first_fact_text(raw_output.get("facts"))
        anchors = _mapping_items(raw_output.get("produced_anchors"))
        if not fact_text or not anchors:
            self.workspace.defer_observation(
                observation.observation_id,
                until="manual_review",
                reason=reason if fact_text else f"{reason}; no facts",
            )
            self.workspace.write_trace_event(
                "commit_auto_deferred",
                {"observation_id": observation.observation_id, "reason": reason},
            )
            return

        anchor = anchors[0]
        anchor_id = str(anchor.get("anchor_id") or "").strip()
        if not anchor_id:
            self.workspace.defer_observation(
                observation.observation_id,
                until="manual_review",
                reason=f"{reason}; no produced anchor id",
            )
            self.workspace.write_trace_event(
                "commit_auto_deferred",
                {"observation_id": observation.observation_id, "reason": reason},
            )
            return

        source_kind = str(anchor.get("source_kind") or "visual_fact")
        excerpt = fact_text[:500]
        self.workspace.commit_observation(
            observation.observation_id,
            writes={
                "pinned_anchors": [
                    {
                        "anchor_id": anchor_id,
                        "kind": str(anchor.get("modality") or source_kind),
                        "source_kind": source_kind,
                        "excerpt": excerpt,
                    }
                ],
                "memory": [
                    {
                        "kind": "unverified_capture",
                        "claim": fact_text,
                        "anchor_ids": [anchor_id],
                        "confidence": "low",
                        "metadata": {
                            "auto_pinned": True,
                            "auto_pin_reason": reason,
                        },
                    }
                ],
            },
        )
        self.workspace.write_trace_event(
            "commit_auto_pinned",
            {"observation_id": observation.observation_id, "reason": reason, "anchor_id": anchor_id},
        )

    def _execute_plan_action(
        self,
        action: Mapping[str, Any],
        *,
        question: str,
        round_number: int,
    ) -> tuple[str, ...]:
        ctx = self._runtime_context(question=question, round_number=round_number)
        result = self.runtime_host.run(
            [{"tool": _tool_name(action), "args": _action_args(action)}],
            ctx=ctx,
        )
        return tuple(str(item) for item in result.observation_ids)

    def _commit_required(self, tool_name: str, observation: Any | None = None) -> bool:
        canonical_name = self.registry.resolve_alias(tool_name)
        spec = self.registry.get_runtime_spec(canonical_name)
        if spec.commit_required:
            return True
        if spec.commit_required_predicate is None or observation is None:
            return False
        return bool(spec.commit_required_predicate(observation.raw_output or {}))

    def _finalize_answer(self, action: Mapping[str, Any], *, question: str, rounds: int) -> WorkspaceRunResult:
        args = _action_args(action)
        try:
            self.registry.get_runtime_spec("answer")
        except ToolError:
            return _answer_result(action, rounds=rounds)
        normalized = self.runtime_host.normalize_program(
            [{"tool": "answer", "args": args}],
            ctx=self._runtime_context(question=question, round_number=rounds),
        )
        normalized_args = dict(normalized[0].get("args", {})) if normalized else args
        output = self.registry.execute("answer", normalized_args)
        citations = output.get("citations", ())
        citation_values = citations if isinstance(citations, Sequence) and not isinstance(citations, str) else ()
        return WorkspaceRunResult(
            answer=str(output.get("answer") or output.get("claim") or normalized_args.get("text") or ""),
            citations=tuple(str(item) for item in citation_values if str(item)),
            confidence=str(output.get("answer_confidence") or normalized_args.get("confidence") or ""),
            rounds=rounds,
            metadata={"raw_output": dict(output)},
        )


PLAN_SYSTEM_PROMPT = """You are exploring a video through a durable workspace.
Output exactly one JSON object: {"tool":"...","args":{...}}.
Use exploration tools in plan phase. Final answers must use tool "answer" and cite committed memory.
"""


COMMIT_SYSTEM_PROMPT = """The previous tool produced an observation that requires disposition.
Output exactly one JSON object using only commit_observation, reject_observation, defer_observation, or no_commit_needed.
"""


def compose_plan_prompt(*, question: str, workspace: EvidenceWorkspace, last_tool_result: str = "") -> str:
    return "\n".join(
        [
            "# Question",
            question,
            "",
            workspace.render_plan_view(question=question),
            "",
            "# Last Tool Result",
            last_tool_result or "(none)",
        ]
    )


def compose_commit_prompt(
    *,
    question: str,
    workspace: EvidenceWorkspace,
    observation_id: str,
    validation_error: str = "",
    attempt: int = 1,
    prompt_mode: str = "full",
) -> str:
    sections = [
        "# Question",
        question,
        "",
        f"# Commit Phase (attempt {attempt})",
        workspace.render_commit_view(question=question, observation_id=observation_id),
    ]
    if validation_error:
        sections.extend(
            [
                "",
                "# Validation Error From Previous Attempt",
                validation_error,
                "",
                "Re-read the Pending Observation section and ensure pinned_anchors[].anchor_id values appear in the listed Candidate Anchors.",
            ]
        )
    if prompt_mode == "minimal":
        sections.extend(
            [
                "",
                "# Minimal Commit Mode",
                "Pin exactly one anchor from Candidate Anchors and write exactly one memory entry. Do not write entities, events, relations, or attributes in this attempt.",
            ]
        )
    return "\n".join(sections)


def _parse_action(text: str) -> Mapping[str, Any]:
    raw = str(text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("workspace_agent_parse_failed: no JSON object found")
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("workspace_agent_parse_failed: action must be a JSON object")
    if not _tool_name(payload):
        raise ValueError("workspace_agent_parse_failed: action missing tool")
    return dict(payload)


def _tool_name(action: Mapping[str, Any]) -> str:
    return str(action.get("tool") or action.get("op") or "").strip()


def _action_args(action: Mapping[str, Any]) -> dict[str, Any]:
    args = action.get("args", {})
    return dict(args) if isinstance(args, Mapping) else {}


def _first_fact_text(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, Mapping):
                text = str(item.get("text") or "").strip()
                if text:
                    return text
            else:
                text = str(item or "").strip()
                if text:
                    return text
    return ""


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _answer_result(action: Mapping[str, Any], *, rounds: int) -> WorkspaceRunResult:
    args = _action_args(action)
    citations = args.get("citations", ())
    if isinstance(citations, str):
        citation_values: Sequence[Any] = (citations,)
    elif isinstance(citations, Sequence):
        citation_values = citations
    else:
        citation_values = ()
    return WorkspaceRunResult(
        answer=str(args.get("text") or args.get("answer") or ""),
        citations=tuple(str(item) for item in citation_values if str(item)),
        confidence=str(args.get("confidence", "")),
        rounds=rounds,
        metadata={"raw_args": dict(args)},
    )
