"""Workspace-first plan/act/commit agent skeleton."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..core.protocol import ToolRequest, ToolResult
from ..core.registry import DuplicateGuardPolicy, ToolError, ToolRegistry
from ..workspace import EvidenceWorkspace


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


@dataclass
class _MvpRoundState:
    round_number: int
    issued_tool_calls: int = 0


@dataclass
class _MvpExecutionContext:
    workspace: EvidenceWorkspace
    registry: ToolRegistry
    round_state: _MvpRoundState
    seen_tool_semantic_keys: set[str]
    scene_index: Any | None = None
    budget: Any | None = None
    skill_runtime: Any | None = None
    evidence_policy: Any | None = None
    record_trace: Any | None = None

    @property
    def issued_tool_calls(self) -> int:
        return self.round_state.issued_tool_calls

    def increment_tool_calls(self, count: int = 1) -> None:
        self.round_state.issued_tool_calls += int(count)


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
        video_map: Any | None = None,
        log_root: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.workspace = workspace
        self.max_rounds = int(max_rounds)
        self.video_path = video_path
        self.video_map = video_map
        self.log_root = Path(log_root) if log_root is not None else workspace.root / "workspace_logs"

    def run(self, question: str) -> WorkspaceRunResult:
        last_tool_result = ""
        seen_tool_semantic_keys: set[str] = set()
        for round_number in range(1, self.max_rounds + 1):
            plan_action = self._decide_plan(
                question=question,
                round_number=round_number,
                last_tool_result=last_tool_result,
            )
            if _tool_name(plan_action) == "answer":
                try:
                    return self._finalize_answer(
                        plan_action,
                        question=question,
                        rounds=round_number,
                        seen_tool_semantic_keys=seen_tool_semantic_keys,
                    )
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

            try:
                obs_ids = self._execute_plan_action(
                    plan_action,
                    question=question,
                    round_number=round_number,
                    seen_tool_semantic_keys=seen_tool_semantic_keys,
                )
            except (ToolError, ValueError) as exc:
                last_tool_result = f"tool rejected: {exc}"
                self.workspace.write_trace_event(
                    "workspace_tool_rejected",
                    {
                        "round": round_number,
                        "tool": _tool_name(plan_action),
                        "args": _action_args(plan_action),
                        "error": str(exc),
                    },
                )
                continue
            observation_id = obs_ids[-1] if obs_ids else ""
            observation = self.workspace.get_observation(observation_id) if observation_id else None
            if observation is not None:
                last_tool_result = f"{observation.observation_id}: {observation.claim}"

            if observation_id and self._commit_required(_tool_name(plan_action), observation):
                commit_result = self._run_commit_phase(
                    question=question,
                    observation_id=observation_id,
                    round_number=round_number,
                )
                if commit_result:
                    last_tool_result = commit_result
            elif observation_id and self.workspace.observation_status(observation_id) == "uncommitted":
                self.workspace.no_commit_needed(observation_id, reason="tool output did not require durable commit")

        return self._force_final_answer(
            question=question,
            rounds=self.max_rounds,
            seen_tool_semantic_keys=seen_tool_semantic_keys,
        )

    def _execution_context(
        self,
        *,
        round_number: int,
        seen_tool_semantic_keys: set[str],
    ) -> _MvpExecutionContext:
        return _MvpExecutionContext(
            workspace=self.workspace,
            round_state=_MvpRoundState(round_number=round_number),
            registry=self.registry,
            seen_tool_semantic_keys=seen_tool_semantic_keys,
            record_trace=self.workspace.write_trace_event,
        )

    def _decide_plan(self, *, question: str, round_number: int, last_tool_result: str) -> Mapping[str, Any]:
        prompt = compose_plan_prompt(
            question=question,
            workspace=self.workspace,
            last_tool_result=last_tool_result,
            video_map=self.video_map,
        )
        response = self.backend.generate(
            BackendRequest(
                task="workspace_plan",
                system_prompt=PLAN_SYSTEM_PROMPT,
                prompt=prompt,
                metadata={"round": round_number, "phase": "plan"},
            )
        )
        planner_io = _write_model_io_artifacts(
            self.log_root,
            phase="plan",
            round_number=round_number,
            prompt=prompt,
            response=response.text,
        )
        self.workspace.write_trace_event(
            "workspace_plan_model_io",
            {
                "round": round_number,
                "response": response.text[:2000],
                **planner_io,
            },
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
        prompt = compose_commit_prompt(
            question=question,
            workspace=self.workspace,
            observation_id=observation_id,
            validation_error=validation_error,
            attempt=attempt,
            prompt_mode=prompt_mode,
        )
        response = self.backend.generate(
            BackendRequest(
                task="workspace_commit",
                system_prompt=COMMIT_SYSTEM_PROMPT,
                prompt=prompt,
                metadata={
                    "round": round_number,
                    "phase": "commit",
                    "observation_id": observation_id,
                    "attempt": attempt,
                    "prompt_mode": prompt_mode,
                },
            )
        )
        planner_io = _write_model_io_artifacts(
            self.log_root,
            phase="commit",
            round_number=round_number,
            prompt=prompt,
            response=response.text,
            attempt=attempt,
        )
        self.workspace.write_trace_event(
            "workspace_commit_model_io",
            {
                "round": round_number,
                "attempt": attempt,
                "prompt_mode": prompt_mode,
                "observation_id": observation_id,
                "response": response.text[:2000],
                **planner_io,
            },
        )
        return _parse_action(response.text)

    def _decide_final(self, *, question: str, round_number: int) -> Mapping[str, Any]:
        prompt = compose_final_prompt(question=question, workspace=self.workspace, video_map=self.video_map)
        response = self.backend.generate(
            BackendRequest(
                task="workspace_final",
                system_prompt=FINAL_SYSTEM_PROMPT,
                prompt=prompt,
                metadata={"round": round_number, "phase": "final", "forced": True},
            )
        )
        planner_io = _write_model_io_artifacts(
            self.log_root,
            phase="final",
            round_number=round_number,
            prompt=prompt,
            response=response.text,
        )
        self.workspace.write_trace_event(
            "workspace_final_model_io",
            {
                "round": round_number,
                "forced": True,
                "response": response.text[:2000],
                **planner_io,
            },
        )
        try:
            return _parse_action(response.text)
        except ValueError as exc:
            self.workspace.write_trace_event(
                "workspace_final_parse_error",
                {"round": round_number, "error": str(exc), "response": response.text[:2000]},
            )
            return {"tool": "answer", "args": {"text": str(response.text or "").strip(), "citations": [], "confidence": "low"}}

    def _run_commit_phase(self, *, question: str, observation_id: str, round_number: int) -> str:
        validation_error = ""
        for attempt in range(1, 4):
            prompt_mode = "minimal" if attempt == 3 else "full"
            try:
                commit_action = self._decide_commit(
                    question=question,
                    observation_id=observation_id,
                    round_number=round_number,
                    validation_error=validation_error,
                    attempt=attempt,
                    prompt_mode=prompt_mode,
                )
            except ValueError as exc:
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
                continue
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
                args = _normalized_disposition_args(
                    commit_action,
                    workspace=self.workspace,
                    observation_id=observation_id,
                )
                self.registry.execute(_tool_name(commit_action), args)
                return _latest_disposition_summary(self.workspace, observation_id)
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
        return _latest_disposition_summary(self.workspace, observation_id)

    def _auto_pin_observation(self, observation: Any, *, reason: str) -> None:
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        fact_text = _first_fact_text(raw_output.get("facts"))
        anchors = _mapping_items(raw_output.get("produced_anchors"))
        if not fact_text and anchors:
            retrieval_writes = _retrieval_candidate_writes(raw_output, anchors=anchors, reason=reason)
            if retrieval_writes:
                try:
                    self.workspace.commit_observation(observation.observation_id, writes=retrieval_writes)
                except (ToolError, ValueError) as exc:
                    self._defer_auto_pin_failure(observation.observation_id, reason=reason, error=str(exc))
                    return
                self.workspace.write_trace_event(
                    "commit_auto_pinned",
                    {
                        "observation_id": observation.observation_id,
                        "reason": reason,
                        "anchor_id": retrieval_writes["pinned_anchors"][0]["anchor_id"],
                        "kind": "retrieval_candidate",
                    },
                )
                return
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
        excerpt = str(anchor.get("excerpt") or "").strip() or fact_text[:500]
        try:
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
                        },
                    ],
                },
            )
        except (ToolError, ValueError) as exc:
            self._defer_auto_pin_failure(observation.observation_id, reason=reason, error=str(exc))
            return
        self.workspace.write_trace_event(
            "commit_auto_pinned",
            {"observation_id": observation.observation_id, "reason": reason, "anchor_id": anchor_id},
        )

    def _defer_auto_pin_failure(self, observation_id: str, *, reason: str, error: str) -> None:
        self.workspace.defer_observation(
            observation_id,
            until="manual_review",
            reason=f"{reason}; auto_pin_failed: {error}",
        )
        self.workspace.write_trace_event(
            "commit_auto_deferred",
            {
                "observation_id": observation_id,
                "reason": reason,
                "auto_pin_failed": True,
                "auto_pin_error": error,
            },
        )

    def _execute_plan_action(
        self,
        action: Mapping[str, Any],
        *,
        question: str,
        round_number: int,
        seen_tool_semantic_keys: set[str],
    ) -> tuple[str, ...]:
        del question
        ctx = self._execution_context(
            round_number=round_number,
            seen_tool_semantic_keys=seen_tool_semantic_keys,
        )
        request = self._normalize_tool_action(
            _tool_name(action),
            _action_args(action),
            ctx=ctx,
            request_id="1",
        )
        self.workspace.write_trace_event(
            "tool_use",
            {"step": 1, "tool": request.tool, "arguments": dict(request.arguments)},
        )
        semantic_key = self._duplicate_guard_key(request, ctx=ctx)
        if semantic_key and semantic_key in seen_tool_semantic_keys:
            rejection = {
                "step": 1,
                "tool": request.tool,
                "reason": "duplicate_tool_call",
                "message": f"{request.tool} repeats semantic key {semantic_key}.",
                "payload": {"tool": request.tool, "semantic_key": semantic_key},
            }
            self.workspace.write_trace_event("tool_call_rejected", rejection)
            raise ValueError(f"duplicate_tool_call: {request.tool} repeats semantic key {semantic_key}.")

        ctx.increment_tool_calls()
        raw_output = dict(self.registry.execute(request.tool, request.arguments))
        if semantic_key:
            seen_tool_semantic_keys.add(semantic_key)
        observation = self.workspace.write_observation(
            tool_name=request.tool,
            input_artifacts=raw_output.get("input_artifacts", []),
            claim=str(raw_output.get("claim", "")),
            confidence=float(raw_output.get("confidence", 0.0)),
            regions=raw_output.get("regions", []),
            limitations=str(raw_output.get("limitations", "")),
            confidence_signal=str(raw_output.get("confidence_signal", "")),
            raw_output=raw_output,
        )
        self.workspace.write_trace_event(
            "tool_result",
            {
                "step": 1,
                "tool": request.tool,
                "observation_id": observation.observation_id,
            },
        )
        observation_ids = [observation.observation_id]
        result = ToolResult.from_mapping(request=request, output=raw_output)
        for observation_id in self._adapted_observation_ids(request, result, ctx=ctx):
            if observation_id not in observation_ids:
                observation_ids.append(observation_id)
        return tuple(observation_ids)

    def _normalize_tool_action(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        ctx: _MvpExecutionContext,
        request_id: str,
    ) -> ToolRequest:
        canonical_name = self.registry.resolve_alias(str(tool_name).strip())
        if not canonical_name:
            raise ValueError("Planner tool action is missing required 'tool'")
        if not isinstance(args, Mapping):
            raise ValueError(f"Planner tool action args must be an object for {canonical_name}")
        request = ToolRequest(tool=canonical_name, arguments=dict(args), request_id=request_id)
        runtime_spec = self.registry.get_runtime_spec(canonical_name)
        normalizer = runtime_spec.argument_normalizer
        if normalizer is None:
            return request
        normalized_args = normalizer(ctx, request)
        if not isinstance(normalized_args, Mapping):
            raise ValueError(f"Argument normalizer for {canonical_name} must return a mapping")
        return ToolRequest(
            tool=request.tool,
            arguments=dict(normalized_args),
            request_id=request.request_id,
            caller=request.caller,
        )

    def _duplicate_guard_key(self, request: ToolRequest, *, ctx: _MvpExecutionContext) -> str:
        runtime_spec = self.registry.get_runtime_spec(request.tool)
        if runtime_spec.duplicate_guard_policy is DuplicateGuardPolicy.OFF:
            return ""
        builder = runtime_spec.semantic_key_builder
        if builder is None:
            return ""
        return str(builder(ctx, request) or "").strip()

    def _adapted_observation_ids(
        self,
        request: ToolRequest,
        result: ToolResult,
        *,
        ctx: _MvpExecutionContext,
    ) -> tuple[str, ...]:
        adapter = self.registry.get_runtime_spec(request.tool).observation_adapter
        if adapter is None:
            return ()
        return _coerce_observation_ids(adapter(ctx, request, result))

    def _commit_required(self, tool_name: str, observation: Any | None = None) -> bool:
        canonical_name = self.registry.resolve_alias(tool_name)
        spec = self.registry.get_runtime_spec(canonical_name)
        if spec.commit_required:
            return True
        if spec.commit_required_predicate is None or observation is None:
            return False
        return bool(spec.commit_required_predicate(observation.raw_output or {}))

    def _finalize_answer(
        self,
        action: Mapping[str, Any],
        *,
        question: str,
        rounds: int,
        seen_tool_semantic_keys: set[str],
    ) -> WorkspaceRunResult:
        args = _action_args(action)
        try:
            self.registry.get_runtime_spec("answer")
        except ToolError:
            return _answer_result(action, rounds=rounds)
        del question
        request = self._normalize_tool_action(
            "answer",
            args,
            ctx=self._execution_context(
                round_number=rounds,
                seen_tool_semantic_keys=seen_tool_semantic_keys,
            ),
            request_id="answer",
        )
        normalized_args = dict(request.arguments)
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

    def _force_final_answer(
        self,
        *,
        question: str,
        rounds: int,
        seen_tool_semantic_keys: set[str],
    ) -> WorkspaceRunResult:
        action = _coerce_answer_action(self._decide_final(question=question, round_number=rounds))
        try:
            result = self._finalize_answer(
                action,
                question=question,
                rounds=rounds,
                seen_tool_semantic_keys=seen_tool_semantic_keys,
            )
        except (ToolError, ValueError) as exc:
            attempted_args = _action_args(action)
            attempted_answer = str(attempted_args.get("text", ""))
            metadata: dict[str, Any] = {}
            metadata.update(
                {
                    "status": "low_confidence_final",
                    "forced_final": True,
                    "reason": "max_rounds_reached",
                    "validated": False,
                    "validation_error": str(exc),
                    "attempted_answer": attempted_answer,
                    "attempted_citations": _string_sequence(attempted_args.get("citations", []) or []),
                }
            )
            self.workspace.write_trace_event(
                "workspace_forced_answer_unvalidated",
                {
                    "round": rounds,
                    "error": str(exc),
                    "action": {"tool": _tool_name(action), "args": _action_args(action)},
                },
            )
            return WorkspaceRunResult(
                answer=attempted_answer,
                citations=(),
                confidence="low",
                rounds=rounds,
                metadata=metadata,
            )
        metadata = dict(result.metadata or {})
        metadata.update({"forced_final": True, "reason": "max_rounds_reached", "validated": True})
        return WorkspaceRunResult(
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            rounds=result.rounds,
            metadata=metadata,
        )


PLAN_SYSTEM_PROMPT = """You are exploring a video through a durable workspace.
Output exactly one JSON object: {"tool":"...","args":{...}}.
Available plan tools are explore, verify_window, read_workspace, synthesize_memory, and answer.
The Segment Cards are navigation-only summaries. Full Dense Video Caption beats are hidden from your default context.
Use explore to find candidate windows when the relevant local window is unknown. Explore results are navigation only.
Use verify_window to inspect a concrete candidate window and verify one or more factual checks using local video evidence.
Use read_workspace to inspect committed memory, pending candidates, and verification coverage.
Use synthesize_memory only after Committed Memory contains verified answer-support mem_* ids.
Use answer only after Committed Memory contains verified mem_* ids that directly support the answer.
For questions with multiple factual requirements, pass multiple checks to verify_window when they belong to the same local window.
Treat every verify_window result as scoped to its inspected time window. A local miss is not global absence.
For multiple-choice questions, answer text must be exactly one option letter: "A", "B", "C", or "D".
Every answer call must include {"text": "A|B|C|D", "citations": ["mem_*"], "confidence": "..."}.
If there is no committed memory, or if the last answer was rejected, choose an exploration tool instead of answer.
"""


COMMIT_SYSTEM_PROMPT = """The previous tool produced an observation that requires disposition.
Output exactly one JSON object using only commit_observation, reject_observation, defer_observation, or no_commit_needed.
The JSON object must include a "tool" field and an "args" object.
"""

FINAL_SYSTEM_PROMPT = """The exploration budget is exhausted.
Answer only from committed workspace memory. Do not invent citations.
For multiple-choice questions, the answer text must be exactly one option letter: "A", "B", "C", or "D".
Output exactly one JSON object using only {"tool":"answer","args":{"text":"<option_letter>","citations":["mem_*"],"confidence":"low|medium|high"}}.
If no committed memory directly supports an option, still output the most likely option letter with citations: [] and confidence: low; do not default to any specific letter.
"""


def compose_plan_prompt(
    *,
    question: str,
    workspace: EvidenceWorkspace,
    last_tool_result: str = "",
    video_map: Any | None = None,
) -> str:
    return "\n".join(
        [
            "# Question",
            question,
            "",
            "# Plan Protocol",
            "Return exactly one JSON object. Do not explain.",
            "Use Segment Cards as the starting navigation state. They are summaries only, not answer evidence.",
            'To find candidate windows, call {"tool":"explore","args":{"query":"what to localize","targets":[{"target_id":"target_1","claim":"fact to locate","aliases":[]}],"modalities":["index","asr","ocr","visual"],"top_k":8}}.',
            'For answer-grade local evidence, call {"tool":"verify_window","args":{"candidate_key":"obs_0001:cand_0001","evidence_mode":"multimodal","sampling":{"fps":2,"max_frames":128},"checks":[{"target_id":"target_1","claim":"fact to verify in this local window","polarity":"presence"}]}}.',
            "Explore results are candidate-only and cannot support final answers.",
            "Use verify_window to convert candidate windows into local evidence before committing answer-support memory.",
            'If Last Tool Result starts with "answer rejected", the next tool must be explore, verify_window, read_workspace, or synthesize_memory.',
            'If Last Tool Result starts with "observation rejected", change candidate, segment, sub_window, evidence_mode, or focus; do not repeat the same verify.',
            'If Last Tool Result starts with "tool rejected: duplicate_tool_call", change the tool scope/query/modality or inspect workspace state; do not repeat the same semantic request.',
            _synthesize_memory_availability(workspace),
            "Use answer only when citations are existing committed mem_* ids that directly support the option; do not invent memory ids.",
            "",
            workspace.render_plan_view(question=question, video_map=video_map),
            "",
            "# Last Tool Result",
            last_tool_result or "(none)",
        ]
    )


def _synthesize_memory_availability(workspace: EvidenceWorkspace) -> str:
    answer_supporting = {"answer_support", "synthesized_support", "answer_conflict_resolved"}
    if any(entry.kind in answer_supporting for entry in workspace.memory_entries()):
        return "synthesize_memory is available only for deriving from cited committed answer-support memory."
    return "synthesize_memory is unavailable until committed memory exists; first commit answer-support memory from verify_window."


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
        "# Commit Guidance",
        "Commit local factual evidence with its valid scope; do not convert local absence into full-video absence.",
        "Use memory kind answer_support when the fact supports an answer option or subclaim needed for that option; include supports_option when clear.",
        "Use memory kind local_negative when a fact only says something was not found, not mentioned, or not visible inside one local window; include metadata.scope and metadata.global_negation_allowed=false.",
        "A local_negative cannot support a final answer or global negation by itself; it only tells the planner where to search next or what a checked local window did not contain.",
        "Reject only when the observation is corrupt, off-topic, duplicate, or has no usable factual anchor.",
        "",
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
            "",
            "# Valid Commit Examples",
            "Full form:",
            """{
  "tool": "commit_observation",
  "args": {
    "writes": {
      "pinned_anchors": [
        {
          "anchor_id": "<candidate_anchor_id>",
          "kind": "retrieval_hit",
          "source_kind": "retrieval_hit",
          "excerpt": "<verbatim candidate excerpt>"
        }
      ],
      "memory": [
        {
          "kind": "retrieval_candidate",
          "claim": "<what this candidate may help verify>",
          "anchor_ids": ["<candidate_anchor_id>"],
          "confidence": "low",
          "metadata": {"requires_local_read": true}
        }
      ],
      "plan_update": "Next: verify_window the pinned candidate time range before final answer."
    }
  }
}""",
            "Shorthand form accepted by the harness:",
            """{
  "tool": "commit_observation",
  "args": {
    "anchor_id": "<candidate_anchor_id>",
    "claim": "<what this candidate may help verify>",
    "kind": "retrieval_candidate",
    "confidence": "low"
  }
}""",
        ]
    )
    return "\n".join(sections)


def compose_final_prompt(*, question: str, workspace: EvidenceWorkspace, video_map: Any | None = None) -> str:
    return "\n".join(
        [
            "# Question",
            question,
            "",
            "# Forced Final Protocol",
            "The maximum round count has been reached. You must answer now.",
            "Return exactly one JSON object using the answer tool.",
            "The answer args must include text, citations, and confidence.",
            "Citations must be existing committed mem_* ids; use [] when no valid citation supports the answer.",
            "For multiple-choice questions, text must be exactly one option letter: A, B, C, or D.",
            "Choose the best supported option from committed memory; do not default to any option letter.",
            "Prefer committed mem_* citations. If no valid citation supports the answer, choose the most likely option with citations: [] and confidence: low; the framework will mark the answer unvalidated.",
            "",
            workspace.render_plan_view(question=question, video_map=video_map),
        ]
    )


def _parse_action(text: str) -> Mapping[str, Any]:
    raw = str(text or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = _first_action_object(raw)
    if isinstance(payload, Mapping) and not (_tool_name(payload) or str(payload.get("disposition") or "").strip()):
        nested = _action_from_nested_value(payload)
        if nested is not None:
            payload = nested
    if not isinstance(payload, Mapping):
        raise ValueError("workspace_agent_parse_failed: action must be a JSON object")
    if not _tool_name(payload):
        disposition = str(payload.get("disposition") or "").strip()
        if disposition:
            payload = dict(payload)
            payload["tool"] = disposition
            payload.setdefault("args", {})
    if not _tool_name(payload):
        raise ValueError("workspace_agent_parse_failed: action missing tool")
    return dict(payload)


def _first_action_object(raw: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    first_object: Mapping[str, Any] | None = None
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        first_object = first_object or payload
        if _tool_name(payload) or str(payload.get("disposition") or "").strip():
            return payload
    if first_object is not None:
        return first_object
    raise ValueError("workspace_agent_parse_failed: no JSON object found")


def _action_from_nested_value(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if _tool_name(value) or str(value.get("disposition") or "").strip():
            return value
        for child in value.values():
            action = _action_from_nested_value(child)
            if action is not None:
                return action
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            action = _action_from_nested_value(child)
            if action is not None:
                return action
    return None


def _tool_name(action: Mapping[str, Any]) -> str:
    return str(action.get("tool") or action.get("op") or "").strip()


def _action_args(action: Mapping[str, Any]) -> dict[str, Any]:
    args = action.get("args", {})
    return dict(args) if isinstance(args, Mapping) else {}


def _string_sequence(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return []


def _coerce_answer_action(action: Mapping[str, Any]) -> Mapping[str, Any]:
    if _tool_name(action) == "answer":
        return action
    args = _action_args(action)
    text = (
        args.get("text")
        or args.get("answer")
        or args.get("claim")
        or args.get("choice")
        or action.get("text")
        or action.get("answer")
        or action.get("claim")
        or action.get("choice")
    )
    if text is None:
        text = json.dumps(action, ensure_ascii=True, sort_keys=True, default=str)
    citations = args.get("citations") or args.get("citation_ids") or args.get("memory_ids") or ()
    confidence = args.get("confidence") or action.get("confidence") or "low"
    return {
        "tool": "answer",
        "args": {
            "text": str(text),
            "citations": citations,
            "confidence": str(confidence),
        },
    }


def _write_model_io_artifacts(
    log_root: Path,
    *,
    phase: str,
    round_number: int,
    prompt: str,
    response: str,
    attempt: int | None = None,
) -> dict[str, Any]:
    if phase == "commit":
        stem = f"round_{int(round_number):03d}_commit_attempt_{int(attempt or 1):02d}"
    elif phase == "final":
        stem = f"round_{int(round_number):03d}_final"
    else:
        stem = f"round_{int(round_number):03d}_plan"
    prompt_meta = _write_log_text(log_root / f"{stem}_prompt.txt", prompt)
    response_meta = _write_log_text(log_root / f"{stem}_response.txt", response)
    return {
        "prompt_path": str(prompt_meta["path"]),
        "response_path": str(response_meta["path"]),
        "prompt_chars": int(prompt_meta["chars"]),
        "response_chars": int(response_meta["chars"]),
    }


def _write_log_text(path: Path, text: str) -> Mapping[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")
    return {"path": path.as_posix(), "chars": len(str(text or ""))}


def _normalized_disposition_args(
    action: Mapping[str, Any],
    *,
    workspace: EvidenceWorkspace,
    observation_id: str,
) -> dict[str, Any]:
    tool_name = _tool_name(action)
    args = _action_args(action)
    if "observation_id" not in args and args.get("obs_id"):
        args["observation_id"] = args["obs_id"]
    args.pop("obs_id", None)
    args.setdefault("observation_id", observation_id)
    if tool_name != "commit_observation":
        if tool_name in {"reject_observation", "no_commit_needed"}:
            args.setdefault("reason", "")
        if tool_name == "defer_observation":
            args.setdefault("until", "more_evidence")
            args.setdefault("reason", "")
        return args

    writes = args.get("writes")
    if "writes" in args and not isinstance(writes, Mapping):
        raise ValueError(_commit_writes_schema_error())
    if isinstance(writes, Mapping):
        return {"observation_id": str(args.get("observation_id") or observation_id), "writes": dict(writes)}

    legacy_writes = _legacy_commit_writes(args, workspace=workspace, observation_id=observation_id)
    if not legacy_writes:
        raise ValueError(_commit_writes_schema_error())
    return {"observation_id": str(args.get("observation_id") or observation_id), "writes": legacy_writes}


def _latest_disposition_summary(workspace: EvidenceWorkspace, observation_id: str) -> str:
    status = workspace.observation_status(observation_id)
    details = ""
    for item in reversed(workspace.observation_dispositions()):
        if str(item.get("observation_id") or "") != str(observation_id):
            continue
        reason = str(item.get("reason") or "").strip()
        until = str(item.get("until") or "").strip()
        if reason:
            details = f": {reason}"
        elif until:
            details = f": until={until}"
        break
    return f"observation {status}: {observation_id}{details}"


def _commit_writes_schema_error() -> str:
    return (
        "commit_observation writes must be an object with pinned_anchors and/or memory lists; "
        'for shorthand use {"tool":"commit_observation","args":{"anchor_id":"<candidate_anchor_id>",'
        '"claim":"<claim>","kind":"retrieval_candidate","confidence":"low"}}'
    )


def _legacy_commit_writes(
    args: Mapping[str, Any],
    *,
    workspace: EvidenceWorkspace,
    observation_id: str,
) -> dict[str, Any]:
    observation = workspace.get_observation(observation_id)
    explicit_claim = str(
        args.get("claim")
        or args.get("memory")
        or args.get("summary")
        or args.get("text")
        or ""
    ).strip()
    has_anchor_hint = any(key in args for key in ("pinned_anchors", "anchors", "anchor_ids", "anchor_id"))
    if not explicit_claim and not has_anchor_hint:
        return {}
    pinned_anchors = _legacy_anchor_payloads(args, workspace=workspace, observation_id=observation_id)
    claim = explicit_claim or str(observation.claim if observation is not None else "").strip()
    memory_kind = str(args.get("kind") or args.get("output_type") or "answer_support").strip()
    if memory_kind not in {
        "answer_support",
        "synthesized_support",
        "answer_conflict_resolved",
        "retrieval_candidate",
        "local_negative",
    }:
        memory_kind = "answer_support"

    writes: dict[str, Any] = {}
    if pinned_anchors:
        writes["pinned_anchors"] = pinned_anchors
    if claim:
        memory: dict[str, Any] = {
            "kind": memory_kind,
            "claim": claim,
            "confidence": str(args.get("confidence") or "medium"),
        }
        anchor_ids = [str(anchor.get("anchor_id") or "").strip() for anchor in pinned_anchors]
        anchor_ids = [anchor_id for anchor_id in anchor_ids if anchor_id]
        if anchor_ids:
            memory["anchor_ids"] = anchor_ids
        supports_option = str(args.get("supports_option") or args.get("option") or "").strip()
        if supports_option:
            memory["supports_option"] = supports_option
        if memory_kind == "local_negative":
            metadata: dict[str, Any] = {"global_negation_allowed": False}
            scope = args.get("scope")
            if isinstance(scope, Mapping):
                metadata["scope"] = dict(scope)
            memory["metadata"] = metadata
        if memory_kind == "retrieval_candidate":
            time_range = _first_anchor_time_range(pinned_anchors)
            memory["tags"] = ["retrieval_candidate", "requires_local_read"]
            memory["metadata"] = {
                "requires_local_read": True,
                "cannot_final_cite": True,
                "recommended_next_tool": "verify_window",
                "recommended_scope": {"time_range": time_range} if time_range is not None else {},
            }
            if anchor_ids:
                writes["plan_update"] = (
                    f"Next: verify_window candidate {anchor_ids[0]} at {_format_scope_for_plan(time_range)} before final answer."
                )
        writes["memory"] = [memory]
    return writes


def _first_anchor_time_range(anchors: Sequence[Mapping[str, Any]]) -> Any:
    for anchor in anchors:
        time_range = anchor.get("time_range")
        if time_range is not None:
            return time_range
        start_sec = anchor.get("start_sec")
        end_sec = anchor.get("end_sec")
        if start_sec is not None and end_sec is not None:
            return [start_sec, end_sec]
    return None


def _legacy_anchor_payloads(
    args: Mapping[str, Any],
    *,
    workspace: EvidenceWorkspace,
    observation_id: str,
) -> list[dict[str, Any]]:
    produced = {
        anchor.anchor_id: anchor.to_dict()
        for anchor in workspace.observation_anchors(observation_id)
        if anchor.anchor_id
    }
    raw_anchors = (
        args.get("pinned_anchors")
        or args.get("anchors")
        or args.get("anchor_ids")
        or args.get("anchor_id")
    )
    if raw_anchors is None and produced:
        raw_items: Sequence[Any] = (next(iter(produced)),)
    elif isinstance(raw_anchors, Mapping):
        raw_items = (raw_anchors,)
    elif isinstance(raw_anchors, Sequence) and not isinstance(raw_anchors, (str, bytes)):
        raw_items = raw_anchors
    elif raw_anchors is None:
        raw_items = ()
    else:
        raw_items = (raw_anchors,)

    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        payload = dict(item) if isinstance(item, Mapping) else {"anchor_id": str(item)}
        anchor_id = str(payload.get("anchor_id") or payload.get("candidate_anchor_id") or "").strip()
        if not anchor_id or anchor_id in seen:
            continue
        produced_payload = dict(produced.get(anchor_id, {}))
        merged = {**produced_payload, **payload, "anchor_id": anchor_id}
        anchors.append(merged)
        seen.add(anchor_id)
    return anchors


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


def _retrieval_candidate_writes(
    raw_output: Mapping[str, Any],
    *,
    anchors: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    results = _mapping_items(raw_output.get("results")) or _mapping_items(raw_output.get("candidate_windows"))
    if not results:
        return {}
    anchor = dict(anchors[0])
    anchor_id = str(anchor.get("anchor_id") or anchor.get("candidate_anchor_id") or "").strip()
    if not anchor_id:
        return {}
    result = next(
        (
            item
            for item in results
            if str(item.get("candidate_anchor_id") or item.get("anchor_id") or "").strip() == anchor_id
        ),
        results[0],
    )
    excerpt = str(anchor.get("excerpt") or result.get("excerpt") or result.get("rationale") or "").strip()
    time_range = result.get("time_range") or anchor.get("time_range") or [anchor.get("start_sec"), anchor.get("end_sec")]
    segment_id = str(result.get("segment_id") or anchor.get("segment_id") or "").strip()
    claim = (
        f"Candidate retrieval hit in {segment_id or 'unknown segment'} may help answer the question: {excerpt}"
        if excerpt
        else f"Candidate retrieval hit in {segment_id or 'unknown segment'} requires local read before answer."
    )
    pinned_anchor = {
        **anchor,
        "anchor_id": anchor_id,
        "kind": str(anchor.get("kind") or anchor.get("source_kind") or "retrieval_hit"),
        "source_kind": str(anchor.get("source_kind") or "retrieval_hit"),
        "excerpt": excerpt,
    }
    return {
        "pinned_anchors": [pinned_anchor],
        "memory": [
            {
                "kind": "retrieval_candidate",
                "claim": claim,
                "anchor_ids": [anchor_id],
                "confidence": "low",
                "tags": ["retrieval_candidate", "requires_local_read"],
                "metadata": {
                    "requires_local_read": True,
                    "cannot_final_cite": True,
                    "auto_pinned": True,
                    "auto_pin_reason": reason,
                    "recommended_next_tool": "verify_window",
                    "candidate_key": str(result.get("candidate_key") or ""),
                    "candidate_id": str(result.get("candidate_id") or ""),
                    "recommended_scope": {"time_range": time_range},
                },
            }
        ],
        "plan_update": f"Next: verify_window candidate {anchor_id} at {_format_scope_for_plan(time_range)} before final answer.",
    }


def _format_scope_for_plan(time_range: Any) -> str:
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
        return f"time_range=[{time_range[0]}, {time_range[1]}]"
    return "its candidate time range"


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _coerce_observation_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


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
