"""Workspace-first plan/act/commit agent skeleton."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..core.protocol import ToolRequest, ToolResult
from ..core.registry import DuplicateGuardPolicy, ToolError, ToolRegistry
from ..workspace import EvidenceWorkspace
from ..workspace.open_questions import extract_candidate_options


DISPOSITION_TOOLS = {
    "commit_observation",
    "reject_observation",
    "defer_observation",
    "no_commit_needed",
}

POSITIVE_SUPPORT_KINDS = frozenset(
    {
        "visual_support",
        "answer_support",
        "synthesized_support",
        "answer_conflict_resolved",
        "caption_support",
    }
)


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
                    self.workspace.write_trace_event(
                        "answer_gate_rejection",
                        {
                            "round": round_number,
                            "reason_code": str(exc),
                            "attempted_citations": _action_args(plan_action).get("citations", []),
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
                if _tool_name(plan_action) == "synthesize_memory":
                    self.workspace.write_trace_event(
                        "synthesis_gate_rejection",
                        {
                            "round": round_number,
                            "supports": _action_args(plan_action).get("supports", []),
                            "tags": _action_args(plan_action).get("tags", []),
                            "reason": str(exc),
                        },
                    )
                continue
            observation_id = obs_ids[-1] if obs_ids else ""
            observation = self.workspace.get_observation(observation_id) if observation_id else None
            if observation is not None:
                last_tool_result = f"{observation.observation_id}: {observation.claim}"

            commit_tool_name = str(getattr(observation, "tool_name", "") or _tool_name(plan_action))
            if observation_id and self._commit_required(commit_tool_name, observation):
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
                max_new_tokens=2048,
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
        if self._try_deterministic_commit(observation_id):
            return _latest_disposition_summary(self.workspace, observation_id)
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

    def _try_deterministic_commit(self, observation_id: str) -> bool:
        observation = self.workspace.get_observation(observation_id)
        if observation is None:
            return False
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        if str(raw_output.get("mode") or "") != "verify_window":
            return False
        if self.workspace.observation_status(observation_id) in {"committed", "rejected", "acknowledged", "auto_acknowledged"}:
            return False
        anchors = _mapping_items(raw_output.get("produced_anchors"))
        writes = _structured_verify_writes(raw_output, anchors=anchors, reason="deterministic_verify_commit")
        if not writes:
            return False
        try:
            self.workspace.commit_observation(observation_id, writes=writes)
        except (ToolError, ValueError) as exc:
            self.workspace.write_trace_event(
                "deterministic_verify_commit_failed",
                {"observation_id": observation_id, "error": str(exc)},
            )
            return False
        self.workspace.write_trace_event(
            "deterministic_verify_commit",
            {
                "observation_id": observation_id,
                "memory_count": len(writes.get("memory", [])),
            },
        )
        return True

    def _auto_pin_observation(self, observation: Any, *, reason: str) -> None:
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        fact_text = _first_fact_text(raw_output.get("facts"))
        anchors = _mapping_items(raw_output.get("produced_anchors"))
        caption_fact_writes = _caption_fact_writes(raw_output, anchors=anchors, reason=reason)
        if caption_fact_writes:
            try:
                self.workspace.commit_observation(observation.observation_id, writes=caption_fact_writes)
            except (ToolError, ValueError) as exc:
                self._defer_auto_pin_failure(observation.observation_id, reason=reason, error=str(exc))
                return
            self.workspace.write_trace_event(
                "commit_auto_pinned",
                {
                    "observation_id": observation.observation_id,
                    "reason": reason,
                    "kind": "caption_fact_results",
                    "memory_count": len(caption_fact_writes.get("memory", [])),
                },
            )
            return
        structured_verify_writes = _structured_verify_writes(raw_output, anchors=anchors, reason=reason)
        if structured_verify_writes:
            try:
                self.workspace.commit_observation(observation.observation_id, writes=structured_verify_writes)
            except (ToolError, ValueError) as exc:
                self._defer_auto_pin_failure(observation.observation_id, reason=reason, error=str(exc))
                return
            self.workspace.write_trace_event(
                "commit_auto_pinned",
                {
                    "observation_id": observation.observation_id,
                    "reason": reason,
                    "kind": "structured_verify_results",
                    "memory_count": len(structured_verify_writes.get("memory", [])),
                },
            )
            return
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
        ctx = self._execution_context(
            round_number=round_number,
            seen_tool_semantic_keys=seen_tool_semantic_keys,
        )
        tool_name = _tool_name(action)
        action_args = _action_args(action)
        if tool_name == "explore":
            action_args = _with_original_question_context(action_args, question=question)
        request = self._normalize_tool_action(
            tool_name,
            action_args,
            ctx=ctx,
            request_id="1",
        )
        if request.tool == "explore":
            auto_verified = self._maybe_auto_verify_on_saturation(
                question=question,
                round_number=round_number,
                seen_tool_semantic_keys=seen_tool_semantic_keys,
            )
            if auto_verified:
                self.workspace.write_trace_event(
                    "explore_replaced_by_auto_verify",
                    {"round": round_number, "reason": "pending_candidate_saturation"},
                )
                return auto_verified
        self.workspace.write_trace_event(
            "tool_use",
            {"step": 1, "tool": request.tool, "arguments": dict(request.arguments)},
        )
        semantic_key = self._duplicate_guard_key(request, ctx=ctx)
        if semantic_key and semantic_key in seen_tool_semantic_keys:
            if request.tool == "explore":
                auto_verified = self._maybe_auto_verify_pending_candidate(
                    question=question,
                    query=str(request.arguments.get("query") or ""),
                    round_number=round_number,
                    seen_tool_semantic_keys=seen_tool_semantic_keys,
                )
                if auto_verified:
                    return auto_verified
                raw_output = {
                    "mode": "planner_recovery_hint",
                    "support_status": "no_new_evidence",
                    "claim": "Similar explore already exists; inspect pending candidates instead.",
                    "confidence": 0.0,
                    "query": str(request.arguments.get("query") or ""),
                    "message": f"explore repeats semantic key {semantic_key}.",
                    "recommended_next_actions": _recommended_actions_from_search_ledger(self.workspace),
                    "cannot_final_cite": True,
                }
                observation = self.workspace.write_observation(
                    tool_name=request.tool,
                    claim=str(raw_output["claim"]),
                    confidence=0.0,
                    raw_output=raw_output,
                )
                self.workspace.write_trace_event(
                    "planner_recovery_hint_emitted",
                    {
                        "observation_id": observation.observation_id,
                        "recommended_next_actions": raw_output["recommended_next_actions"],
                    },
                )
                self.workspace.write_trace_event(
                    "tool_result",
                    {"step": 1, "tool": request.tool, "observation_id": observation.observation_id},
                )
                return (observation.observation_id,)
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

    def _maybe_auto_verify_pending_candidate(
        self,
        *,
        question: str,
        query: str,
        round_number: int,
        seen_tool_semantic_keys: set[str],
    ) -> tuple[str, ...]:
        if not _has_prior_recovery_hint(self.workspace, query=query):
            return ()
        candidate_key = ""
        for action in _recommended_actions_from_search_ledger(self.workspace):
            if str(action.get("tool") or "") == "verify_window" and str(action.get("candidate_key") or "").strip():
                candidate_key = str(action.get("candidate_key") or "").strip()
                break
        if not candidate_key:
            return ()
        return self._auto_verify_candidate_key(
            candidate_key=candidate_key,
            question=question,
            round_number=round_number,
            seen_tool_semantic_keys=seen_tool_semantic_keys,
            reason="pending_candidate_not_consumed",
            trace_payload={"query": query},
        )

    def _maybe_auto_verify_on_saturation(
        self,
        *,
        question: str,
        round_number: int,
        seen_tool_semantic_keys: set[str],
        min_pending: int = 3,
        max_drain: int = 3,
    ) -> tuple[str, ...]:
        drained: list[str] = []
        attempted: set[str] = set()
        for _ in range(max(1, int(max_drain))):
            snapshot = self.workspace.search_ledger_snapshot()
            raw_candidates = snapshot.get("candidates", [])
            if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
                break
            pending = [
                candidate
                for candidate in raw_candidates
                if isinstance(candidate, Mapping) and str(candidate.get("status") or "") == "pending"
            ]
            if len(pending) < min_pending:
                break
            if any(entry.kind in POSITIVE_SUPPORT_KINDS for entry in self.workspace.memory_entries()):
                break
            available = [
                candidate
                for candidate in pending
                if str(candidate.get("candidate_key") or candidate.get("event_id") or "").strip() not in attempted
            ]
            if not available:
                break
            selected = _highest_score_pending_candidate(available)
            candidate_key = str(selected.get("candidate_key") or selected.get("event_id") or "").strip()
            if not candidate_key:
                break
            attempted.add(candidate_key)
            observation_ids = self._auto_verify_candidate_key(
                candidate_key=candidate_key,
                question=question,
                round_number=round_number,
                seen_tool_semantic_keys=seen_tool_semantic_keys,
                reason="pending_candidate_saturation",
                trace_payload={"pending_count": len(pending), "selected_score": selected.get("score")},
            )
            if not observation_ids:
                break
            drained.extend(observation_ids)
        return tuple(drained)

    def _auto_verify_candidate_key(
        self,
        *,
        candidate_key: str,
        question: str,
        round_number: int,
        seen_tool_semantic_keys: set[str],
        reason: str,
        trace_payload: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...]:
        try:
            runtime_spec = self.registry.get_runtime_spec("verify_window")
        except ToolError:
            return ()
        ctx = self._execution_context(
            round_number=round_number,
            seen_tool_semantic_keys=seen_tool_semantic_keys,
        )
        checks = self._derive_checks_for_auto_verify(candidate_key=candidate_key, question=question)
        verify_args: dict[str, Any] = {"candidate_key": candidate_key, "focus": [question]}
        if "checks" in runtime_spec.tool_spec.parameters:
            verify_args["checks"] = checks
        request = self._normalize_tool_action(
            "verify_window",
            verify_args,
            ctx=ctx,
            request_id="auto_verify",
        )
        self.workspace.write_trace_event(
            "candidate_auto_verify_triggered",
            {
                "round": round_number,
                "reason": reason,
                "candidate_key": candidate_key,
                **dict(trace_payload or {}),
            },
        )
        self.workspace.write_trace_event(
            "tool_use",
            {"step": 1, "tool": request.tool, "arguments": dict(request.arguments), "auto": True},
        )
        ctx.increment_tool_calls()
        raw_output = dict(self.registry.execute(request.tool, request.arguments))
        semantic_key = self._duplicate_guard_key(request, ctx=ctx)
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
            {"step": 1, "tool": request.tool, "observation_id": observation.observation_id, "auto": True},
        )
        observation_ids = [observation.observation_id]
        result = ToolResult.from_mapping(request=request, output=raw_output)
        for observation_id in self._adapted_observation_ids(request, result, ctx=ctx):
            if observation_id not in observation_ids:
                observation_ids.append(observation_id)
        return tuple(observation_ids)

    def _derive_checks_for_auto_verify(self, *, candidate_key: str, question: str) -> list[dict[str, Any]]:
        source_observation_id = str(candidate_key or "").split(":", 1)[0] if ":" in str(candidate_key or "") else ""
        if source_observation_id:
            for observation in self.workspace.read_observations():
                if observation.observation_id != source_observation_id:
                    continue
                raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
                checks: list[dict[str, Any]] = []
                for target in _mapping_items(raw_output.get("targets")):
                    claim = str(target.get("claim") or target.get("question") or target.get("text") or "").strip()
                    if not claim:
                        continue
                    check: dict[str, Any] = {
                        "target_id": str(
                            target.get("target_id")
                            or target.get("id")
                            or target.get("target_ref")
                            or f"target_{len(checks) + 1}"
                        ),
                        "claim": claim,
                        "polarity": str(target.get("polarity") or target.get("kind") or "presence"),
                    }
                    option_id = str(target.get("option_id") or target.get("option") or "").strip().upper()[:1]
                    if option_id:
                        check["option_id"] = option_id
                    checks.append(check)
                if checks:
                    return checks
                break
        return [
            {
                "target_id": "auto_question_check",
                "claim": str(question or "").strip(),
                "polarity": "presence",
            }
        ]

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
        attempted_args = _action_args(action)
        attempted_citations = _string_sequence(attempted_args.get("citations", []) or [])
        attempted_answer = str(attempted_args.get("text", ""))
        if _answer_tool_available(self.registry) and not attempted_citations:
            autocited = self._try_forced_answer_autocite(
                action,
                question=question,
                rounds=rounds,
                seen_tool_semantic_keys=seen_tool_semantic_keys,
            )
            if autocited is not None:
                return autocited
            metadata: dict[str, Any] = {
                "status": "unvalidated_guess",
                "forced_final": True,
                "reason": "max_rounds_reached",
                "validated": False,
                "validation_error": "",
                "attempted_answer": attempted_answer,
                "attempted_citations": attempted_citations,
            }
            self.workspace.write_trace_event(
                "workspace_forced_unvalidated_guess",
                {
                    "round": rounds,
                    "reason": "no_valid_final_citation",
                    "action": {"tool": _tool_name(action), "args": attempted_args},
                },
            )
            return WorkspaceRunResult(
                answer=attempted_answer,
                citations=(),
                confidence=str(attempted_args.get("confidence") or "low"),
                rounds=rounds,
                metadata=metadata,
            )
        try:
            result = self._finalize_answer(
                action,
                question=question,
                rounds=rounds,
                seen_tool_semantic_keys=seen_tool_semantic_keys,
            )
        except (ToolError, ValueError) as exc:
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

    def _try_forced_answer_autocite(
        self,
        action: Mapping[str, Any],
        *,
        question: str,
        rounds: int,
        seen_tool_semantic_keys: set[str],
    ) -> WorkspaceRunResult | None:
        args = _action_args(action)
        choice = str(args.get("text") or args.get("answer") or "").strip().upper()[:1]
        if not choice:
            return None
        for memory_id in _support_memory_ids_for_choice(self.workspace, choice):
            patched_action = {"tool": "answer", "args": {**args, "text": choice, "citations": [memory_id]}}
            try:
                result = self._finalize_answer(
                    patched_action,
                    question=question,
                    rounds=rounds,
                    seen_tool_semantic_keys=seen_tool_semantic_keys,
                )
            except (ToolError, ValueError):
                continue
            metadata = dict(result.metadata or {})
            metadata.update(
                {
                    "forced_final": True,
                    "reason": "max_rounds_reached",
                    "validated": True,
                    "auto_cited": True,
                    "attempted_citations": [],
                }
            )
            self.workspace.write_trace_event(
                "workspace_forced_answer_autocited",
                {"round": rounds, "answer": choice, "citation": memory_id},
            )
            return WorkspaceRunResult(
                answer=result.answer,
                citations=result.citations,
                confidence=result.confidence,
                rounds=result.rounds,
                metadata=metadata,
            )
        return None


PLAN_SYSTEM_PROMPT = """You are exploring a video through a durable workspace.
Output exactly one JSON object: {"tool":"...","args":{...}}.
Available plan tools are explore, verify_window, read_workspace, synthesize_memory, and answer.
The Segment Cards are navigation-only summaries. Full Dense Video Caption beats are hidden from your default context.
Use explore to reason over dense captions, ASR, OCR, and index summaries. It may return caption_fact, mixed, or candidate_discovery.
caption_fact/mixed observations can provide caption-level facts for commit; candidate_discovery is navigation-only.
Use verify_window to inspect a concrete candidate window and verify one or more factual checks using local video evidence.
verify_window also accepts {segment_id, time_range} directly. Use this form to sweep unexplored regions of a segment when explore keeps proposing the same time window.
Use read_workspace to inspect committed memory, pending candidates, and verification coverage.
Use synthesize_memory only after Committed Memory contains verified or caption-supported mem_* ids.
Use answer only after Committed Memory contains mem_* ids that directly support the answer.
For questions with multiple factual requirements, pass multiple checks to verify_window when they belong to the same local window.
Treat every verify_window result as scoped to its inspected time window. A local miss is not global absence.
Query Framing Policy:
When calling explore, write the query as a verification question, not as a guessed answer.
The query should preserve the original question condition: when, after, before, during, first, last, shown as, how many, what object, which action, or what event.
The planner query is a retrieval hint. The original question defines what counts as evidence.
For multiple-choice questions:
1. Prefer a question-centered explore query first.
2. Do not begin by copying terms from only one answer option unless your explicit goal is to test that option.
3. If you test an option, phrase the target as: "Check whether option X answers the original question condition."
4. Evidence that matches an option but does not answer the original question condition is not answer evidence.
Task policy: counting, object presence, spatial relation, scoreboard/UI, fine visual action, and visual text questions require verify_window visual_support for final answers; caption_support alone is only eligible for narrative facts that match the original condition.
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
            "Query Framing Policy: for the first explore call, ask the question the user asked; do not merge in a candidate answer unless explicitly checking that option.",
            "HARD RULE: If Pending Candidate Windows count >= 3 and Committed Memory contains zero positive support entries (visual_support / answer_support / synthesized_support / answer_conflict_resolved / caption_support), your next tool MUST be verify_window using the highest-score candidate_key or segment_id + time_range. local_negative / answer_conflict / verification_uncertain do not disarm this rule. Do not call explore.",
            'HARD RULE: When testing an answer option, every check target_id must follow option_<letter>_check and carry option_id="<letter>"; generic target_1/target_2 names are only allowed for question-centered checks that are not testing a specific option.',
            'To inspect dense captions/indexes or find candidate windows, call {"tool":"explore","args":{"query":"question-centered condition to resolve","targets":[{"target_id":"target_1","question":"question condition to verify","verification_goal":"identify the fact that directly answers the original question condition"}],"modalities":["index","asr","ocr","visual"],"top_k":8}}.',
            'When testing an option, use a target like {"target_id":"option_B_check","claim":"option B claim","verification_goal":"Check whether option B answers the original question condition.","option_id":"B"}.',
            'For answer-grade local evidence, call {"tool":"verify_window","args":{"candidate_key":"obs_0001:cand_0001","evidence_mode":"multimodal","sampling":{"fps":2,"max_frames":128},"checks":[{"target_id":"target_1","claim":"fact to verify in this local window","polarity":"presence"}]}}.',
            'You may also call verify_window with an explicit time slice when pending candidates are exhausted: {"tool":"verify_window","args":{"segment_id":"seg_0001","time_range":[20.0,40.0],"evidence_mode":"multimodal","sampling":{"fps":2,"max_frames":64},"checks":[{"target_id":"target_1","claim":"fact to verify in this local window","polarity":"presence"}]}}.',
            "If every pending candidate window in the current segment has already been verified with not_found_in_window and the question still requires visual_support, do not call explore again with a paraphrased query; instead call verify_window with segment_id + a new time_range that covers a different part of the segment.",
            "Explore can return caption_fact, mixed, or candidate_discovery. Commit caption_fact/mixed caption facts; verify candidate-only windows before final.",
            "Inspect explore condition_match and answer_mapping: if condition_match is false or unknown, do not answer from that memory.",
            "For counting/object/spatial/scoreboard/fine-action/visual-text tasks, use caption/index evidence only for navigation; final citations need visual_support.",
            "If Pending Candidate Windows exist and no answer-support/caption-support memory exists, prefer verify_window over repeating similar explore.",
            "For MCQ, start with the question condition; map evidence to options only after evidence is retrieved.",
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
    answer_supporting = {"answer_support", "caption_support", "visual_support", "synthesized_support", "answer_conflict_resolved"}
    if any(entry.kind in answer_supporting for entry in workspace.memory_entries()):
        return "synthesize_memory is available only for deriving from cited committed support memory."
    return "synthesize_memory is unavailable until committed support memory exists; first commit caption facts from explore or visual facts from verify_window."


def _recommended_actions_from_search_ledger(workspace: EvidenceWorkspace) -> list[dict[str, object]]:
    snapshot = workspace.search_ledger_snapshot()
    candidates = snapshot.get("candidates", [])
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and candidate.get("status") == "pending":
                return [{"tool": "verify_window", "candidate_key": candidate.get("candidate_key") or candidate.get("event_id")}]
    return [{"tool": "read_workspace", "section": "pending_candidates"}]


def _has_prior_recovery_hint(workspace: EvidenceWorkspace, *, query: str) -> bool:
    query_norm = _norm_query(query)
    for observation in workspace.read_observations():
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        if str(raw_output.get("mode") or "") != "planner_recovery_hint":
            continue
        if query_norm and _norm_query(raw_output.get("query")) != query_norm:
            continue
        return True
    return False


def _norm_query(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _answer_tool_available(registry: ToolRegistry) -> bool:
    try:
        registry.get_runtime_spec("answer")
    except ToolError:
        return False
    return True


def _support_memory_ids_for_choice(workspace: EvidenceWorkspace, choice: str) -> list[str]:
    target = str(choice or "").strip().upper()[:1]
    if not target:
        return []
    priority = {
        "answer_conflict_resolved": 0,
        "answer_support": 1,
        "visual_support": 2,
        "caption_support": 3,
        "synthesized_support": 4,
    }
    rows: list[tuple[int, str]] = []
    for entry in workspace.memory_entries():
        option = str(getattr(entry, "supports_option", "") or "").strip().upper()[:1]
        if option != target:
            continue
        kind = str(getattr(entry, "kind", "") or "")
        if kind not in priority:
            continue
        rows.append((priority[kind], str(entry.entry_id)))
    return [entry_id for _, entry_id in sorted(rows)]


def _with_original_question_context(args: Mapping[str, Any], *, question: str) -> dict[str, Any]:
    enriched = dict(args)
    enriched.setdefault("original_question", str(question or ""))
    if "answer_options" not in enriched and "options" not in enriched:
        options: dict[str, str] = {}
        for option in extract_candidate_options(str(question or "")):
            text = str(option or "").strip()
            if len(text) >= 3 and text[0].upper() in "ABCDEFGH" and text[1] in {".", ")"}:
                options[text[0].upper()] = text[2:].strip()
        if options:
            enriched["answer_options"] = options
    return enriched


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
        "Use memory kind visual_support ONLY for verify_window facts whose verdict is 'supported'. The framework auto-assigns this kind; do not override.",
        "Use memory kind local_negative ONLY for verify_window facts whose verdict is 'not_found_in_window'. The framework auto-assigns this kind; do not override. It must carry metadata.scope and metadata.global_negation_allowed=false. A local_negative cannot support a final answer or global negation by itself; it only tells the planner where to search next or what a checked local window did not contain.",
        "Use memory kind answer_conflict ONLY for verify_window facts whose verdict is 'contradicted'.",
        "Use memory kind answer_support when verified or caption-supported facts directly support an answer option or subclaim needed for that option; include supports_option when clear.",
        "Preserve option_id from verification targets/results; if a supported target follows option_<letter>_check, write supports_option=<letter> for answer-grade memory.",
        "Use memory kind caption_support for caption/asr/ocr facts returned by explore caption_fact or mixed observations.",
        'Caption/index facts may be committed as caption_support only when the explore payload mode is "caption_fact" or "mixed" and condition_match.matches_original_question is true.',
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
        if '"tool"' in raw or "'tool'" in raw:
            raise ValueError("workspace_agent_parse_failed: truncated_or_invalid_action_json")
        return first_object
    if '"tool"' in raw or "'tool'" in raw:
        raise ValueError("workspace_agent_parse_failed: truncated_or_invalid_action_json")
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
        "caption_support",
        "visual_support",
        "synthesized_support",
        "answer_conflict_resolved",
        "retrieval_candidate",
        "local_negative",
        "verification_uncertain",
        "answer_conflict",
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
                text = str(item.get("text") or item.get("claim") or item.get("excerpt") or "").strip()
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


def _caption_fact_writes(
    raw_output: Mapping[str, Any],
    *,
    anchors: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    mode = str(raw_output.get("mode") or "").strip()
    support_status = str(raw_output.get("support_status") or "").strip()
    if mode not in {"caption_fact", "mixed"} or support_status not in {
        "caption_supported",
        "partial_caption_supported",
    }:
        return {}
    facts = _mapping_items(raw_output.get("facts"))
    if not facts:
        claim = str(raw_output.get("claim") or "").strip()
        if not claim:
            return {}
        facts = [_caption_fact_from_claim(claim, anchors=anchors, raw_output=raw_output)]

    normalized_anchors = _dedupe_anchor_payloads(_caption_fact_anchor_payloads(anchors, facts=facts))
    anchor_ids = [str(anchor.get("anchor_id") or "").strip() for anchor in normalized_anchors]
    anchor_ids = [anchor_id for anchor_id in anchor_ids if anchor_id]
    default_anchor_id = anchor_ids[0] if anchor_ids else ""
    answer_mapping = raw_output.get("answer_mapping") if isinstance(raw_output.get("answer_mapping"), Mapping) else {}
    condition_match = raw_output.get("condition_match") if isinstance(raw_output.get("condition_match"), Mapping) else {}
    query_analysis = raw_output.get("query_analysis") if isinstance(raw_output.get("query_analysis"), Mapping) else {}
    question_condition = raw_output.get("question_condition") if isinstance(raw_output.get("question_condition"), Mapping) else {}
    raw_claim_scope = str(raw_output.get("claim_scope") or "").strip()
    required_entities = [
        str(item).strip()
        for item in _sequence_items(raw_output.get("required_entities") or raw_output.get("ordering_required_entities"))
        if str(item).strip()
    ]
    memory: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        claim = str(fact.get("claim") or fact.get("text") or fact.get("excerpt") or "").strip()
        if not claim:
            continue
        fact_anchor_ids = [
            str(item).strip()
            for item in _sequence_items(fact.get("anchor_ids") or fact.get("anchor_id"))
            if str(item).strip()
        ]
        if not fact_anchor_ids:
            fact_anchor_ids = [anchor_ids[index]] if index < len(anchor_ids) else ([default_anchor_id] if default_anchor_id else [])
        supports_option = str(
            fact.get("supports_option")
            or fact.get("option")
            or (answer_mapping.get("supports_option") if isinstance(answer_mapping, Mapping) else "")
            or ""
        ).strip()
        claim_scope = str(fact.get("claim_scope") or raw_claim_scope or "").strip()
        if not claim_scope:
            claim_scope = _caption_claim_scope_for_commit(
                task_type=str(raw_output.get("task_type") or ""),
                condition_match=condition_match,
                cannot_final_cite=bool(raw_output.get("cannot_final_cite", False)),
            )
        covered_entities = [
            str(item).strip()
            for item in _sequence_items(fact.get("covered_entities") or fact.get("ordering_entities"))
            if str(item).strip()
        ]
        if required_entities and not covered_entities:
            claim_text = claim.lower()
            covered_entities = [entity for entity in required_entities if entity.lower() in claim_text]
        metadata: dict[str, Any] = {
            "auto_pinned": True,
            "auto_pin_reason": reason,
            "mode": mode,
            "support_status": support_status,
            "cannot_final_cite": bool(raw_output.get("cannot_final_cite", False)),
            "requires_visual_verify": bool(raw_output.get("needs_visual_verify") or fact.get("needs_visual_verify")),
            "task_type": str(raw_output.get("task_type") or ""),
            "condition_match": dict(condition_match),
            "question_condition_match": bool(condition_match.get("matches_original_question")) if condition_match else False,
            "condition_match_level": str(condition_match.get("match_level") or ""),
            "query_analysis": dict(query_analysis),
            "question_condition": dict(question_condition),
            "answer_mapping": dict(answer_mapping),
            "source_kind": str(fact.get("source_kind") or ""),
            "claim_scope": claim_scope,
        }
        if required_entities:
            metadata["required_entities"] = required_entities
        if covered_entities:
            metadata["covered_entities"] = covered_entities
        scope = _caption_fact_scope(fact)
        if scope:
            metadata["scope"] = scope
        item: dict[str, Any] = {
            "kind": "caption_support",
            "claim": claim,
            "anchor_ids": fact_anchor_ids,
            "confidence": _memory_confidence(fact.get("confidence", raw_output.get("confidence"))),
            "metadata": metadata,
        }
        if supports_option:
            item["supports_option"] = supports_option
        memory.append(item)
    if not memory:
        return {}
    return {"pinned_anchors": normalized_anchors, "memory": memory}


def _caption_claim_scope_for_commit(
    *,
    task_type: str,
    condition_match: Mapping[str, Any],
    cannot_final_cite: bool,
) -> str:
    if str(condition_match.get("match_level") or "") == "related_but_wrong_scope":
        return "wrong_scope"
    if task_type == "ordering" and cannot_final_cite:
        return "subclaim_support"
    if bool(condition_match.get("matches_original_question")) and str(condition_match.get("match_level") or "") == "direct":
        return "direct_answer"
    return "uncertain"


def _caption_fact_from_claim(
    claim: str,
    *,
    anchors: Sequence[Mapping[str, Any]],
    raw_output: Mapping[str, Any],
) -> dict[str, Any]:
    first_anchor = anchors[0] if anchors else {}
    fact: dict[str, Any] = {
        "claim": claim,
        "confidence": raw_output.get("confidence"),
        "source_kind": str(first_anchor.get("source_kind") or first_anchor.get("kind") or "dense_caption"),
    }
    segment_id = str(first_anchor.get("segment_id") or "").strip()
    if segment_id:
        fact["segment_id"] = segment_id
    time_range = first_anchor.get("time_range")
    if time_range is not None:
        fact["time_range"] = time_range
    elif first_anchor.get("start_sec") is not None and first_anchor.get("end_sec") is not None:
        fact["time_range"] = [first_anchor.get("start_sec"), first_anchor.get("end_sec")]
    excerpt = str(first_anchor.get("excerpt") or "").strip()
    if excerpt:
        fact["excerpt"] = excerpt
    answer_mapping = raw_output.get("answer_mapping")
    if isinstance(answer_mapping, Mapping):
        supports_option = str(answer_mapping.get("supports_option") or "").strip()
        if supports_option:
            fact["supports_option"] = supports_option
    return fact


def _caption_fact_anchor_payloads(
    anchors: Sequence[Mapping[str, Any]],
    *,
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        anchor_id = str(anchor.get("anchor_id") or "").strip()
        if not anchor_id:
            continue
        fact = facts[index] if index < len(facts) else {}
        excerpt = str(anchor.get("excerpt") or fact.get("excerpt") or fact.get("claim") or fact.get("text") or "").strip()
        payloads.append(
            {
                **dict(anchor),
                "anchor_id": anchor_id,
                "kind": str(anchor.get("kind") or anchor.get("modality") or anchor.get("source_kind") or "dense_caption"),
                "source_kind": str(anchor.get("source_kind") or fact.get("source_kind") or "dense_caption"),
                "excerpt": excerpt,
            }
        )
    return payloads


def _caption_fact_scope(fact: Mapping[str, Any]) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    segment_id = str(fact.get("segment_id") or "").strip()
    if segment_id:
        scope["segment_id"] = segment_id
    time_range = fact.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
        scope["time_range"] = [time_range[0], time_range[1]]
    elif fact.get("start_sec") is not None and fact.get("end_sec") is not None:
        scope["time_range"] = [fact.get("start_sec"), fact.get("end_sec")]
    return scope


def _structured_verify_writes(
    raw_output: Mapping[str, Any],
    *,
    anchors: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    if str(raw_output.get("mode") or "") != "verify_window":
        return {}
    results = _mapping_items(raw_output.get("verification_results"))
    if not results:
        return {}
    anchors_by_id = {str(anchor.get("anchor_id") or "").strip(): dict(anchor) for anchor in anchors if str(anchor.get("anchor_id") or "").strip()}
    pinned: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        verdict = str(result.get("verdict") or "uncertain").strip()
        kind = _memory_kind_for_verification_verdict(verdict)
        if kind == "verification_uncertain":
            continue
        supports_option = _verification_supports_option(result, verdict=verdict)
        option_truth_status = _option_truth_status(verdict)
        if supports_option and _verification_selects_false_option(result) and verdict in {
            "contradicted",
            "not_found_in_window",
        }:
            kind = "answer_conflict_resolved"
        anchor_ids = [str(item).strip() for item in _sequence_items(result.get("anchor_ids")) if str(item).strip()]
        if not anchor_ids and index < len(anchors):
            anchor_id = str(anchors[index].get("anchor_id") or "").strip()
            if anchor_id:
                anchor_ids = [anchor_id]
        for anchor_id in anchor_ids:
            anchor = dict(anchors_by_id.get(anchor_id, {}))
            if not anchor:
                continue
            pinned.append(
                {
                    **anchor,
                    "anchor_id": anchor_id,
                    "kind": str(anchor.get("modality") or anchor.get("source_kind") or "visual_fact"),
                    "source_kind": str(anchor.get("source_kind") or result.get("source_kind") or "visual_fact"),
                    "excerpt": str(anchor.get("excerpt") or result.get("rationale") or result.get("claim") or ""),
                }
            )
        claim = _verification_memory_claim(result)
        if not claim:
            continue
        item: dict[str, Any] = {
            "kind": kind,
            "claim": claim,
            "anchor_ids": anchor_ids,
            "confidence": _memory_confidence(result.get("confidence")),
            "target_id": str(result.get("target_id") or ""),
            "metadata": {
                "auto_pinned": True,
                "auto_pin_reason": reason,
                "verdict": verdict,
                "target_id": str(result.get("target_id") or ""),
                "scope": dict(result.get("scope", {}) or {}) if isinstance(result.get("scope"), Mapping) else {},
                "source_kind": str(result.get("source_kind") or ""),
                "claim_scope": "window_negative" if verdict == "not_found_in_window" else "direct_answer",
                "global_answer_support": verdict == "supported",
                "local_only": True,
                "event_id": str(result.get("event_id") or ""),
                "event_type": str(result.get("event_type") or ""),
                "answer_polarity": str(
                    result.get("answer_polarity")
                    or result.get("question_polarity")
                    or result.get("selection_polarity")
                    or ""
                ),
                "option_truth_status": option_truth_status,
            },
        }
        if supports_option and kind != "local_negative":
            item["supports_option"] = supports_option
        memory.append(item)
    if not memory:
        return {}
    return {"pinned_anchors": _dedupe_anchor_payloads(pinned), "memory": memory}


def _memory_kind_for_verification_verdict(verdict: str) -> str:
    normalized = str(verdict or "").strip().lower()
    if normalized == "supported":
        return "visual_support"
    if normalized == "not_found_in_window":
        return "local_negative"
    if normalized == "contradicted":
        return "answer_conflict"
    return "verification_uncertain"


def _highest_score_pending_candidate(pending: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    def sort_key(candidate: Mapping[str, Any]) -> tuple[float, str]:
        try:
            score = float(candidate.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        candidate_key = str(candidate.get("candidate_key") or candidate.get("event_id") or "")
        return (-score, candidate_key)

    return sorted(pending, key=sort_key)[0]


def _verification_supports_option(result: Mapping[str, Any], *, verdict: str) -> str:
    explicit = str(
        result.get("supports_option")
        or result.get("matched_option")
        or result.get("option_id")
        or result.get("option")
        or ""
    ).strip().upper()[:1]
    if explicit:
        return explicit
    target_id = str(result.get("target_id") or "").strip()
    option = _option_letter_from_target(target_id)
    if option:
        return option
    if str(verdict or "") == "supported":
        return _option_letter_from_target(target_id, allow_short_form=True)
    return ""


def _option_letter_from_target(target_id: str, *, allow_short_form: bool = False) -> str:
    target_text = str(target_id or "").strip()
    match = re.search(r"(?:^|[^A-Za-z])option[_\-\s]*([A-D])(?:[^A-Za-z]|$)", target_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if allow_short_form:
        match = re.search(
            r"(?:^|[^A-Za-z])([A-D])(?:[_\-\s]*check|[_\-\s]*option)(?:[^A-Za-z]|$)",
            target_text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
    return ""


def _verification_selects_false_option(result: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(result.get(key) or "")
        for key in (
            "answer_polarity",
            "question_polarity",
            "selection_polarity",
            "selection_mode",
            "question_type",
            "verification_goal",
        )
    ).lower()
    false_markers = (
        "select_false",
        "incorrect",
        "not correct",
        "false",
        "except",
        "not mentioned",
        "not discussed",
        "not used",
        "does not",
        "least likely",
    )
    return any(marker in text for marker in false_markers)


def _option_truth_status(verdict: str) -> str:
    if verdict == "supported":
        return "supported_true"
    if verdict == "contradicted":
        return "contradicted"
    if verdict == "not_found_in_window":
        return "not_found"
    return "unknown"


def _verification_memory_claim(result: Mapping[str, Any]) -> str:
    target_id = str(result.get("target_id") or "").strip()
    claim = str(result.get("claim") or "").strip()
    verdict = str(result.get("verdict") or "uncertain").strip()
    rationale = str(result.get("rationale") or "").strip()
    prefix = f"{target_id}: " if target_id else ""
    if verdict == "supported":
        return f"{prefix}{claim} is supported in the inspected window. {rationale}".strip()
    if verdict == "not_found_in_window":
        return f"{prefix}{claim} was not found in the inspected window. {rationale}".strip()
    if verdict == "contradicted":
        return f"{prefix}{claim} is contradicted in the inspected window. {rationale}".strip()
    return f"{prefix}{claim} is uncertain in the inspected window. {rationale}".strip()


def _memory_confidence(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "medium"
    if number >= 0.8:
        return "high"
    if number >= 0.45:
        return "medium"
    return "low"


def _sequence_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _dedupe_anchor_payloads(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id") or "").strip()
        if not anchor_id or anchor_id in seen:
            continue
        seen.add(anchor_id)
        deduped.append(dict(anchor))
    return deduped


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
