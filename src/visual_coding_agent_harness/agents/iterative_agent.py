"""Iterative visual agent for coarse-to-fine video exploration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..interpreter import ProgramInterpreter
from ..registry import ToolError, ToolRegistry
from ..video_index import SceneIndex, VideoSegment
from ..workspace import EvidenceWorkspace
from .answer_agent import AnswerAgent
from .prompt_stack import compose_replanning_prompt_blocks, render_prompt_blocks
from .question_policy import classify_question_route, extract_candidate_options, select_question_playbook
from .skills.predicates import (
    no_decisive_weak_grounding,
    no_unaddressed_conflict,
    selected_option_has_structured_support,
    temporal_order_consistent,
)
from .skills.specs import select_skill


_SEGMENT_MEDIA_TOOLS = {"caption_segment", "qa_segment", "inspect_segment", "vision_read"}
_GLOBAL_VIEW_TOOLS = {"global_gist"}
_CHEAP_TOOLS = {
    "video_ls",
    "search_segments",
    "ground_question",
    "read_segment",
    "expand_window",
    "zoom",
    "summarize_ledger_evidence",
}
_EXPENSIVE_TOOLS = {
    "global_gist",
    "inspect_segment",
    "vision_read",
    "caption_segment",
    "qa_segment",
    "caption_segments",
    "caption_region",
    "qa_region",
}
_VERIFIER_TOOLS = {"verify_ledger_answer"}
_TOOL_CLASSES = {
    **{tool_name: "cheap" for tool_name in _CHEAP_TOOLS},
    **{tool_name: "expensive" for tool_name in _EXPENSIVE_TOOLS},
    **{tool_name: "verifier" for tool_name in _VERIFIER_TOOLS},
}


@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = 8
    max_tool_calls_per_round: int = 2
    default_nframes: int = 8
    high_fps_nframes: int = 32
    planner_receives_media: bool = False
    reserve_final_round: bool = True
    cheap_tool_budget: int = 16
    expensive_tool_budget: int = 6
    verifier_tool_budget: int = 2
    answer_probe_rounds_before_final: int = 0
    free_exploration: bool = False
    persist_planner_io: bool = True
    planner_io_max_chars: int = 200_000
    max_repeated_programs: int = 3
    hard_skill_runtime: bool = False
    reflection_memory_max_items: int = 5

    @classmethod
    def free_explore(cls, *, max_rounds: int = 24, max_tool_calls_per_round: int = 4) -> "AgentBudget":
        """Disable policy budgets while keeping emergency safety caps."""
        return cls(
            max_rounds=max_rounds,
            max_tool_calls_per_round=max_tool_calls_per_round,
            reserve_final_round=False,
            cheap_tool_budget=0,
            expensive_tool_budget=0,
            verifier_tool_budget=0,
            answer_probe_rounds_before_final=0,
            free_exploration=True,
        )


@dataclass(frozen=True)
class IterativeRound:
    round_number: int
    status: str
    planner_text: str
    rationale: str = ""
    program: Sequence[Mapping[str, Any]] = field(default_factory=list)
    observation_ids: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "round_number": self.round_number,
            "status": self.status,
            "rationale": self.rationale,
            "program": list(self.program),
            "observation_ids": list(self.observation_ids),
            "planner_text": self.planner_text,
        }


@dataclass(frozen=True)
class IterativeRunResult:
    question: str
    video_path: str
    answer: str
    status: str
    citations: Sequence[str] = field(default_factory=list)
    confidence: float = 0.0
    rounds: Sequence[IterativeRound] = field(default_factory=list)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "input": {"question": self.question, "video_path": self.video_path},
            "output": {
                "answer": self.answer,
                "status": self.status,
                "citations": list(self.citations),
                "confidence": self.confidence,
            },
            "rounds": [round_result.to_dict() for round_result in self.rounds],
        }


class IterativeVisualAgent:
    """Let a VLM repeatedly plan tools, inspect evidence, and decide when to stop."""

    def __init__(
        self,
        *,
        backend: VisionLanguageBackend,
        registry: ToolRegistry,
        workspace: EvidenceWorkspace,
        scene_index: SceneIndex,
        budget: Optional[AgentBudget] = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.workspace = workspace
        self.scene_index = scene_index
        self.budget = budget or AgentBudget()

    def run(self, *, question: str, video_path: str) -> IterativeRunResult:
        if classify_question_route(question) == "gist_global" and self._has_tool("global_gist"):
            global_result = self._try_global_gist_route(question=question, video_path=video_path)
            if global_result is not None:
                return global_result
        if self.budget.hard_skill_runtime:
            skill_result = self._try_hard_skill_route(question=question, video_path=video_path)
            if skill_result is not None:
                return skill_result

        rounds: list[IterativeRound] = []
        citations: list[str] = []
        inspected_segment_ids: set[str] = set()
        tool_class_counts = {"cheap": 0, "expensive": 0, "verifier": 0}
        has_inspect_with_candidate_options = False
        answer_feedback: list[str] = []
        repeated_program_key = ""
        repeated_program_count = 0

        for round_number in range(1, self.budget.max_rounds + 1):
            ledger_text = self._read_ledger()
            final_round_reserved = self.budget.reserve_final_round and round_number == self.budget.max_rounds
            planner_prompt = _replanning_prompt(
                question=question,
                scene_index=self.scene_index,
                ledger_text=ledger_text,
                round_number=round_number,
                budget=self.budget,
                inspected_segment_ids=sorted(inspected_segment_ids),
                tool_class_counts=tool_class_counts,
                final_round_reserved=final_round_reserved,
                answer_feedback=answer_feedback,
                reflection_memory=self.workspace.reflection_memory(max_items=self.budget.reflection_memory_max_items),
            )
            self.workspace.write_trace_event(
                "iterative_round_start",
                {"round": round_number, "question": question, "evidence_count": len(citations)},
            )
            planner_response = self.backend.generate(
                BackendRequest(
                    task="replan",
                    prompt=planner_prompt,
                    media_path=video_path if self.budget.planner_receives_media else None,
                    media_type="video" if self.budget.planner_receives_media else None,
                    max_new_tokens=768,
                    metadata={
                        "round": round_number,
                        "segment_count": len(self.scene_index.segments),
                        "planner_input_mode": "video" if self.budget.planner_receives_media else "text-only",
                    },
                )
            )
            self._persist_planner_io(
                round_number=round_number,
                prompt=planner_prompt,
                response=planner_response.text,
                planner_input_mode="video" if self.budget.planner_receives_media else "text-only",
            )
            try:
                action = _parse_replan_action(planner_response.text)
            except (json.JSONDecodeError, ValueError) as exc:
                self.workspace.write_trace_event(
                    "planner_json_parse_error",
                    {
                        "round": round_number,
                        "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                        "response_excerpt": _compact_planner_response(planner_response.text),
                    },
                )
                self.workspace.write_reflection_memory(
                    route=classify_question_route(question),
                    failure_tag="planner_json_parse_error",
                    rule="return valid JSON matching the continue/final response contract before using tools",
                )
                action = {
                    "status": "continue",
                    "rationale": "planner_json_parse_error",
                    "program": self._fallback_inspector_program(
                        question=question,
                        inspected_segment_ids=inspected_segment_ids,
                    ),
                }
            status = str(action.get("status", "continue"))
            rationale = str(action.get("rationale", ""))
            planned_program: Any = action.get("program", [])

            if status == "final":
                blocked_reason = _blocked_final_reason(
                    question=question,
                    has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                    workspace=self.workspace,
                    answer=str(action.get("answer", "")),
                    citations=[str(item) for item in action.get("citations", [])],
                )
                if blocked_reason:
                    self.workspace.write_trace_event(
                        "iterative_final_blocked",
                        {
                            "round": round_number,
                            "reason": blocked_reason,
                            "answer": str(action.get("answer", "")),
                            "citations": [str(item) for item in action.get("citations", [])],
                        },
                    )
                    self.workspace.write_reflection_memory(
                        route=classify_question_route(question),
                        failure_tag=blocked_reason,
                        rule=_reflection_rule_for_failure(blocked_reason),
                    )
                    planned_program = []
                    if not final_round_reserved:
                        planned_program = self._fallback_inspector_program(
                            question=question,
                            inspected_segment_ids=inspected_segment_ids,
                        )
                    status = "continue"
                    rationale = blocked_reason
                else:
                    final_citations = [str(item) for item in action.get("citations", [])]
                    result_round = IterativeRound(
                        round_number=round_number,
                        status="final",
                        planner_text=planner_response.text,
                        rationale=rationale,
                    )
                    rounds.append(result_round)
                    self.workspace.write_trace_event(
                        "iterative_final",
                        {
                            "round": round_number,
                            "answer": str(action.get("answer", "")),
                            "citations": final_citations,
                        },
                    )
                    return IterativeRunResult(
                        question=question,
                        video_path=video_path,
                        answer=str(action.get("answer", "")),
                        status="final",
                        citations=final_citations,
                        confidence=float(action.get("confidence", 0.0)),
                        rounds=rounds,
                    )

            if status == "continue":
                program = self._normalize_program(
                    planned_program,
                    question=question,
                    video_path=video_path,
                    inspected_segment_ids=inspected_segment_ids,
                    tool_class_counts=tool_class_counts,
                    final_round_reserved=final_round_reserved,
                )
            else:
                program = []
            if final_round_reserved and not program:
                answer_result = AnswerAgent(self.backend).run(
                    question=question,
                    evidence_text=ledger_text,
                    evidence_table=self._answer_evidence_table(question),
                )
                self.workspace.write_trace_event(
                    "iterative_answer_agent",
                    {
                        "round": round_number,
                        "source": "reserved_final",
                        "status": answer_result.status,
                        "answer": answer_result.answer,
                        "citations": list(answer_result.citations),
                        "missing_evidence": list(answer_result.missing_evidence),
                    },
                )
                if answer_result.status == "final":
                    rounds.append(
                        IterativeRound(
                            round_number=round_number,
                            status="final",
                            planner_text=answer_result.raw_text,
                            rationale=answer_result.rationale,
                        )
                    )
                    self.workspace.write_trace_event(
                        "iterative_final",
                        {
                            "round": round_number,
                            "answer": answer_result.answer,
                            "citations": list(answer_result.citations),
                            "source": "answer_agent",
                        },
                    )
                    return IterativeRunResult(
                        question=question,
                        video_path=video_path,
                        answer=answer_result.answer,
                        status="final",
                        citations=list(answer_result.citations),
                        confidence=answer_result.confidence,
                        rounds=rounds,
                    )
            program_key = _program_signature(program)
            if program_key == repeated_program_key:
                repeated_program_count += 1
            else:
                repeated_program_key = program_key
                repeated_program_count = 1
            if (
                self.budget.max_repeated_programs > 0
                and repeated_program_count > self.budget.max_repeated_programs
            ):
                self.workspace.write_trace_event(
                    "iterative_no_progress_guard",
                    {
                        "round": round_number,
                        "reason": "repeated_program",
                        "repeat_count": repeated_program_count,
                        "max_repeated_programs": self.budget.max_repeated_programs,
                        "program_signature": program_key,
                    },
                )
                plural = "round" if len(rounds) == 1 else "rounds"
                partial_answer = _partial_answer_from_ledger(self._read_ledger())
                return IterativeRunResult(
                    question=question,
                    video_path=video_path,
                    answer=partial_answer
                    or f"Stopped after {len(rounds)} exploration {plural} because the planner repeated the same program.",
                    status="max_rounds_reached",
                    citations=citations,
                    rounds=rounds,
                )
            self.workspace.write_trace_event(
                "iterative_plan",
                {"round": round_number, "rationale": rationale, "program": program},
            )
            program_result = ProgramInterpreter(registry=self.registry, workspace=self.workspace).run(program)
            observation_ids = [str(observation_id) for observation_id in program_result.observation_ids]
            citations.extend(observation_ids)
            inspected_segment_ids.update(_segment_ids_from_program(program))
            _update_tool_class_counts(tool_class_counts, program)
            if _program_has_inspect_with_candidate_options(program):
                has_inspect_with_candidate_options = True
            if _should_run_answer_probe(
                budget=self.budget,
                question=question,
                round_number=round_number,
                has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                citations=citations,
            ):
                answer_result = AnswerAgent(self.backend).run(
                    question=question,
                    evidence_text=self._read_ledger(),
                    evidence_table=self._answer_evidence_table(question),
                )
                self.workspace.write_trace_event(
                    "iterative_answer_agent",
                    {
                        "round": round_number,
                        "source": "prefinal_probe",
                        "status": answer_result.status,
                        "answer": answer_result.answer,
                        "citations": list(answer_result.citations),
                        "missing_evidence": list(answer_result.missing_evidence),
                    },
                )
                if answer_result.status == "final":
                    rounds.append(
                        IterativeRound(
                            round_number=round_number,
                            status="final",
                            planner_text=answer_result.raw_text,
                            rationale=answer_result.rationale,
                        )
                    )
                    self.workspace.write_trace_event(
                        "iterative_final",
                        {
                            "round": round_number,
                            "answer": answer_result.answer,
                            "citations": list(answer_result.citations),
                            "source": "answer_agent_prefinal_probe",
                        },
                    )
                    return IterativeRunResult(
                        question=question,
                        video_path=video_path,
                        answer=answer_result.answer,
                        status="final",
                        citations=list(answer_result.citations),
                        confidence=answer_result.confidence,
                        rounds=rounds,
                    )
                answer_feedback = list(answer_result.missing_evidence)
            rounds.append(
                IterativeRound(
                    round_number=round_number,
                    status="continue",
                    planner_text=planner_response.text,
                    rationale=rationale,
                    program=program,
                    observation_ids=observation_ids,
                )
            )

        self.workspace.write_trace_event(
            "iterative_budget_exhausted",
            {"max_rounds": self.budget.max_rounds, "citations": citations},
        )
        plural = "round" if self.budget.max_rounds == 1 else "rounds"
        partial_answer = _partial_answer_from_ledger(self._read_ledger())
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=partial_answer
            or f"Stopped after {self.budget.max_rounds} exploration {plural} with partial evidence.",
            status="max_rounds_reached",
            citations=citations,
            rounds=rounds,
        )

    def _normalize_program(
        self,
        program: Any,
        *,
        question: str,
        video_path: str,
        inspected_segment_ids: set[str],
        tool_class_counts: Mapping[str, int],
        final_round_reserved: bool,
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(program, list):
            raise ValueError("Planner action status=continue requires a list program")

        normalized = []
        reserved_segment_ids = set(inspected_segment_ids)
        pending_tool_class_counts = {"cheap": 0, "expensive": 0, "verifier": 0}
        for step in program:
            if not isinstance(step, Mapping):
                raise ValueError("Planner program steps must be objects")
            if "tool" not in step:
                raise ValueError("Planner program step is missing required 'tool'")
            if len(normalized) >= self.budget.max_tool_calls_per_round:
                break

            tool_name = str(step["tool"])
            args = dict(step.get("args", {}))
            if final_round_reserved and tool_name not in _VERIFIER_TOOLS:
                self.workspace.write_trace_event(
                    "exploration_policy_adjustment",
                    {
                        "reason": "reserve_final_round",
                        "skipped_tool": tool_name,
                    },
                )
                continue

            segment_id = args.get("segment_id")
            if segment_id and tool_name == "read_segment":
                segment = _scene_segment_or_none(self.scene_index, str(segment_id))
                if segment is not None and not _segment_has_index_text(segment):
                    self.workspace.write_trace_event(
                        "exploration_policy_adjustment",
                        {
                            "reason": "upgrade_empty_read_segment_to_caption",
                            "requested_tool": "read_segment",
                            "resolved_tool": "caption_segment",
                            "segment_id": segment.segment_id,
                        },
                    )
                    tool_name = "caption_segment"

            if not _tool_budget_available(
                budget=self.budget,
                tool_name=tool_name,
                tool_class_counts=tool_class_counts,
                pending_tool_class_counts=pending_tool_class_counts,
            ):
                self.workspace.write_trace_event(
                    "exploration_policy_adjustment",
                    {
                        "reason": "tool_budget_exhausted",
                        "skipped_tool": tool_name,
                        "tool_class": _tool_class(tool_name),
                    },
                )
                continue

            if tool_name in _SEGMENT_MEDIA_TOOLS:
                segment = (
                    self._resolve_media_segment(
                        str(segment_id),
                        args=args,
                        reserved_segment_ids=reserved_segment_ids,
                    )
                    if segment_id
                    else self._resolve_missing_media_segment(
                        args=args,
                        reserved_segment_ids=reserved_segment_ids,
                    )
                )
                if segment is None:
                    continue
                if not segment_id:
                    self.workspace.write_trace_event(
                        "exploration_policy_adjustment",
                        {
                            "reason": "repair_missing_media_segment_id",
                            "tool": tool_name,
                            "resolved_segment_id": segment.segment_id,
                            "start_sec": segment.start_sec,
                            "end_sec": segment.end_sec,
                        },
                    )
                elif segment.segment_id != str(segment_id):
                    self.workspace.write_trace_event(
                        "exploration_policy_adjustment",
                        {
                            "reason": "avoid_repeated_segment",
                            "requested_segment_id": str(segment_id),
                            "resolved_segment_id": segment.segment_id,
                        },
                    )
                args["segment_id"] = segment.segment_id
                args["video_path"] = video_path
                args["start_sec"] = segment.start_sec
                args["end_sec"] = segment.end_sec
                if tool_name == "vision_read":
                    args.setdefault("ask_for", args.pop("question", question))
                else:
                    args.setdefault("question", question)
                candidate_options = list(extract_candidate_options(question))
                if candidate_options:
                    if tool_name == "vision_read":
                        args.setdefault("event_label", str(args.get("ask_for", "")))
                    else:
                        args["question"] = _append_candidate_options_to_tool_question(
                            str(args["question"]),
                            candidate_options=candidate_options,
                        )
                    if tool_name == "inspect_segment":
                        args["candidate_options"] = candidate_options
                args.setdefault("nframes", self.budget.default_nframes)
                reserved_segment_ids.add(segment.segment_id)

            normalized_step: dict[str, Any] = {"tool": tool_name, "args": args}
            if "assign" in step:
                normalized_step["assign"] = str(step["assign"])
            normalized.append(normalized_step)
            pending_tool_class_counts[_tool_class(tool_name)] += 1

        if not normalized and not final_round_reserved:
            fallback_segment_id = self._resolve_next_segment_id("", reserved_segment_ids)
            fallback_tool_name = self._fallback_visual_tool_name()
            if fallback_segment_id is not None and fallback_tool_name is not None and _tool_budget_available(
                budget=self.budget,
                tool_name=fallback_tool_name,
                tool_class_counts=tool_class_counts,
                pending_tool_class_counts=pending_tool_class_counts,
            ):
                segment = self.scene_index.get(fallback_segment_id)
                fallback_args: dict[str, Any] = {
                    "video_path": video_path,
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "nframes": self.budget.default_nframes,
                }
                if fallback_tool_name == "vision_read":
                    fallback_args["ask_for"] = question
                    fallback_args["event_label"] = question
                else:
                    fallback_args["question"] = question
                candidate_options = extract_candidate_options(question)
                if candidate_options and fallback_tool_name == "inspect_segment":
                    fallback_args["candidate_options"] = list(candidate_options)
                normalized.append(
                    {
                        "tool": fallback_tool_name,
                        "args": fallback_args,
                        "assign": f"auto_{segment.segment_id}",
                    }
                )
        return normalized

    def _fallback_inspector_program(
        self,
        *,
        question: str,
        inspected_segment_ids: set[str],
    ) -> Sequence[Mapping[str, Any]]:
        fallback_segment_id = self._resolve_next_segment_id("", inspected_segment_ids)
        if fallback_segment_id is None:
            return []
        tool_name = self._fallback_visual_tool_name()
        if tool_name is None:
            return []
        args: dict[str, Any] = {"segment_id": fallback_segment_id, "question": question}
        candidate_options = extract_candidate_options(question)
        if candidate_options and tool_name == "inspect_segment":
            args["candidate_options"] = list(candidate_options)
        return [{"tool": tool_name, "args": args, "assign": f"required_{fallback_segment_id}"}]

    def _fallback_visual_tool_name(self) -> str | None:
        for tool_name in ["inspect_segment", "vision_read", "caption_segment", "qa_segment"]:
            if self._has_tool(tool_name):
                return tool_name
        return None

    def _resolve_next_segment_id(self, requested_segment_id: str, reserved_segment_ids: set[str]) -> Optional[str]:
        if requested_segment_id and requested_segment_id not in reserved_segment_ids:
            self.scene_index.get(requested_segment_id)
            return requested_segment_id
        for segment in self.scene_index.segments:
            if segment.segment_id not in reserved_segment_ids:
                return segment.segment_id
        return None

    def _resolve_media_segment(
        self,
        requested_segment_id: str,
        *,
        args: Mapping[str, Any],
        reserved_segment_ids: set[str],
    ) -> Optional[VideoSegment]:
        scene_segment = _scene_segment_or_none(self.scene_index, requested_segment_id)
        if scene_segment is not None:
            resolved_segment_id = self._resolve_next_segment_id(requested_segment_id, reserved_segment_ids)
            if resolved_segment_id is None:
                return None
            return self.scene_index.get(resolved_segment_id)

        if requested_segment_id in reserved_segment_ids:
            return None
        dynamic_segment = _segment_from_dynamic_id(
            requested_segment_id,
            duration_sec=self.scene_index.duration_sec,
        )
        if dynamic_segment is not None:
            return dynamic_segment
        observed_segment = self.workspace.observed_segment_window(requested_segment_id)
        if observed_segment is not None:
            start_sec, end_sec = _normalize_dynamic_window(
                start_sec=float(observed_segment["start_sec"]),
                end_sec=float(observed_segment["end_sec"]),
                duration_sec=self.scene_index.duration_sec,
                label=f"for observed {requested_segment_id}",
            )
            return VideoSegment(
                segment_id=requested_segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                source=str(observed_segment.get("source") or "observed_tool_segment"),
            )
        if "start_sec" not in args or "end_sec" not in args:
            self.scene_index.get(requested_segment_id)
        start_sec, end_sec = _normalize_dynamic_window(
            start_sec=float(args["start_sec"]),
            end_sec=float(args["end_sec"]),
            duration_sec=self.scene_index.duration_sec,
            label=f"for {requested_segment_id}",
        )
        return VideoSegment(
            segment_id=requested_segment_id,
            start_sec=start_sec,
            end_sec=end_sec,
            source="dynamic_zoom",
        )

    def _resolve_missing_media_segment(
        self,
        *,
        args: Mapping[str, Any],
        reserved_segment_ids: set[str],
    ) -> Optional[VideoSegment]:
        if "start_sec" in args and "end_sec" in args:
            start_sec, end_sec = _normalize_dynamic_window(
                start_sec=float(args["start_sec"]),
                end_sec=float(args["end_sec"]),
                duration_sec=self.scene_index.duration_sec,
                label="without segment_id",
            )
            segment_id = _dynamic_segment_id(start_sec=start_sec, end_sec=end_sec)
            if segment_id in reserved_segment_ids:
                return None
            return VideoSegment(
                segment_id=segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                source="dynamic_window",
            )

        fallback_segment_id = self._resolve_next_segment_id("", reserved_segment_ids)
        if fallback_segment_id is None:
            return None
        return self.scene_index.get(fallback_segment_id)

    def _read_ledger(self) -> str:
        return self.workspace.compact_ledger_text()

    def _persist_planner_io(
        self,
        *,
        round_number: int,
        prompt: str,
        response: str,
        planner_input_mode: str,
    ) -> None:
        if not self.budget.persist_planner_io:
            return
        prefix = f"artifacts/planner_io/round_{round_number:04d}"
        prompt_meta = self.workspace.write_text_artifact(
            f"{prefix}_prompt.txt",
            prompt,
            max_chars=self.budget.planner_io_max_chars,
        )
        response_meta = self.workspace.write_text_artifact(
            f"{prefix}_response.txt",
            response,
            max_chars=self.budget.planner_io_max_chars,
        )
        self.workspace.write_trace_event(
            "planner_io",
            {
                "round": round_number,
                "planner_input_mode": planner_input_mode,
                "prompt": prompt_meta,
                "response": response_meta,
                "response_excerpt": _compact_planner_response(response),
            },
        )

    def _answer_evidence_table(self, question: str) -> Mapping[str, Any]:
        return self.workspace.evidence_table_v2(
            question=question,
            options=extract_candidate_options(question),
        )

    def _has_tool(self, tool_name: str) -> bool:
        try:
            self.registry.get(tool_name)
        except ToolError:
            return False
        return True

    def _try_global_gist_route(self, *, question: str, video_path: str) -> IterativeRunResult | None:
        program = [
            {
                "tool": "global_gist",
                "args": {
                    "video_path": video_path,
                    "question": question,
                    "duration_sec": self.scene_index.duration_sec,
                },
                "assign": "global_gist",
            }
        ]
        self.workspace.write_trace_event(
            "iterative_route",
            {"route": "gist_global", "tool": "global_gist"},
        )
        program_result = ProgramInterpreter(registry=self.registry, workspace=self.workspace).run(program)
        answer_result = AnswerAgent(self.backend).run(
            question=question,
            evidence_text=self._read_ledger(),
            evidence_table=self._answer_evidence_table(question),
        )
        self.workspace.write_trace_event(
            "iterative_answer_agent",
            {
                "round": 1,
                "source": "global_gist_route",
                "status": answer_result.status,
                "answer": answer_result.answer,
                "citations": list(answer_result.citations),
                "missing_evidence": list(answer_result.missing_evidence),
            },
        )
        if answer_result.status != "final":
            return None

        self.workspace.write_trace_event(
            "iterative_final",
            {
                "round": 1,
                "answer": answer_result.answer,
                "citations": list(answer_result.citations),
                "source": "global_gist_route",
            },
        )
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=answer_result.answer,
            status="final",
            citations=list(answer_result.citations),
            confidence=answer_result.confidence,
            rounds=[
                IterativeRound(
                    round_number=1,
                    status="final",
                    planner_text=answer_result.raw_text,
                    rationale=answer_result.rationale,
                    program=program,
                    observation_ids=list(program_result.observation_ids),
                )
            ],
        )

    def _try_hard_skill_route(self, *, question: str, video_path: str) -> IterativeRunResult | None:
        skill = select_skill(question)
        if skill.name not in {"grounded_factual_qa", "temporal_ordering"}:
            return None
        if not self._has_tool("ground_question") or not self._has_tool("vision_read"):
            return None

        route = classify_question_route(question)
        target_facts = _skill_target_facts(question=question, skill_name=skill.name)
        if not target_facts:
            return None

        self.workspace.write_trace_event(
            "iterative_route",
            {
                "route": route,
                "skill": f"{skill.name}@v{skill.version}",
                "source": "hard_skill_runtime",
            },
        )
        self.workspace.write_trace_event(
            "hard_skill_runtime",
            {
                "skill": f"{skill.name}@v{skill.version}",
                "target_facts": target_facts,
            },
        )

        program: list[dict[str, Any]] = []
        observation_ids: list[str] = []
        interpreter = ProgramInterpreter(registry=self.registry, workspace=self.workspace)

        for index, target_fact in enumerate(target_facts[: max(1, self.budget.max_tool_calls_per_round)], start=1):
            ground_program = [
                {
                    "tool": "ground_question",
                    "args": {"query": target_fact, "top_k": 3},
                    "assign": f"ground_{index}",
                }
            ]
            ground_result = interpreter.run(ground_program)
            program.extend(ground_program)
            observation_ids.extend(str(observation_id) for observation_id in ground_result.observation_ids)
            if not ground_result.observation_ids:
                continue
            candidates = self.workspace.grounding_candidates(str(ground_result.observation_ids[-1]), max_candidates=1)
            if not candidates:
                continue
            candidate = candidates[0]
            vision_program = [
                {
                    "tool": "vision_read",
                    "args": {
                        "video_path": video_path,
                        "segment_id": str(candidate["segment_id"]),
                        "start_sec": float(candidate["start_sec"]),
                        "end_sec": float(candidate["end_sec"]),
                        "ask_for": target_fact,
                        "event_label": target_fact,
                    },
                    "assign": f"fact_{index}",
                }
            ]
            vision_result = interpreter.run(vision_program)
            program.extend(vision_program)
            observation_ids.extend(str(observation_id) for observation_id in vision_result.observation_ids)

        if not observation_ids:
            return None

        table = self._answer_evidence_table(question)
        answer_result = AnswerAgent(self.backend).run(
            question=question,
            evidence_text=self._read_ledger(),
            evidence_table=table,
        )
        selected_option = _answer_option_letter(answer_result.answer)
        gate_reason = _hard_skill_gate_reason(
            skill_name=skill.name,
            question=question,
            table=table,
            selected_option=selected_option,
            citations=answer_result.citations,
        )
        if answer_result.status != "final" or gate_reason:
            failure_tag = gate_reason or "answer_agent_need_more_evidence"
            self.workspace.write_trace_event(
                "iterative_answer_agent",
                {
                    "round": 1,
                    "source": "hard_skill_runtime",
                    "status": "need_more_evidence",
                    "answer": answer_result.answer,
                    "citations": list(answer_result.citations),
                    "missing_evidence": list(answer_result.missing_evidence),
                    "failure_tag": failure_tag,
                },
            )
            self.workspace.write_reflection_memory(
                route=route,
                failure_tag=failure_tag,
                rule=_reflection_rule_for_failure(failure_tag),
            )
            return IterativeRunResult(
                question=question,
                video_path=video_path,
                answer="need_more_evidence",
                status="need_more_evidence",
                citations=list(answer_result.citations),
                confidence=0.0,
                rounds=[
                    IterativeRound(
                        round_number=1,
                        status="need_more_evidence",
                        planner_text=answer_result.raw_text,
                        rationale=answer_result.rationale or failure_tag,
                        program=program,
                        observation_ids=observation_ids,
                    )
                ],
            )

        self.workspace.write_trace_event(
            "iterative_answer_agent",
            {
                "round": 1,
                "source": "hard_skill_runtime",
                "status": answer_result.status,
                "answer": answer_result.answer,
                "citations": list(answer_result.citations),
                "missing_evidence": list(answer_result.missing_evidence),
            },
        )
        self.workspace.write_trace_event(
            "iterative_final",
            {
                "round": 1,
                "answer": answer_result.answer,
                "citations": list(answer_result.citations),
                "source": "hard_skill_runtime",
            },
        )
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=answer_result.answer,
            status="final",
            citations=list(answer_result.citations),
            confidence=answer_result.confidence,
            rounds=[
                IterativeRound(
                    round_number=1,
                    status="final",
                    planner_text=answer_result.raw_text,
                    rationale=answer_result.rationale,
                    program=program,
                    observation_ids=observation_ids,
                )
            ],
        )


def _replanning_prompt(
    *,
    question: str,
    scene_index: SceneIndex,
    ledger_text: str,
    round_number: int,
    budget: AgentBudget,
    inspected_segment_ids: Sequence[str] = (),
    tool_class_counts: Mapping[str, int] | None = None,
    final_round_reserved: bool = False,
    answer_feedback: Sequence[str] = (),
    reflection_memory: Sequence[str] = (),
) -> str:
    blocks = compose_replanning_prompt_blocks(
        question=question,
        scene_index=scene_index,
        ledger_text=ledger_text,
        round_number=round_number,
        budget=budget,
        inspected_segment_ids=inspected_segment_ids,
        tool_class_counts=tool_class_counts,
        final_round_reserved=final_round_reserved,
        answer_feedback=answer_feedback,
        reflection_memory=reflection_memory,
    )
    return render_prompt_blocks(blocks)


def _parse_replan_action(text: str) -> Mapping[str, Any]:
    payload = json.loads(_extract_json_object(text))
    if not isinstance(payload, Mapping):
        raise ValueError("Planner response must be a JSON object")
    if "status" not in payload:
        return {"status": "continue", "program": payload.get("program", []), "rationale": payload.get("rationale", "")}
    return payload


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]


def _compact_planner_response(text: str, limit: int = 480) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact[:limit] + ("..." if len(compact) > limit else "")


def _segment_ids_from_program(program: Sequence[Mapping[str, Any]]) -> Sequence[str]:
    segment_ids = []
    for step in program:
        if step.get("tool") not in _SEGMENT_MEDIA_TOOLS:
            continue
        args = step.get("args", {})
        if isinstance(args, Mapping) and args.get("segment_id"):
            segment_ids.append(str(args["segment_id"]))
    return segment_ids


def _program_has_inspect_with_candidate_options(program: Sequence[Mapping[str, Any]]) -> bool:
    for step in program:
        if step.get("tool") == "vision_read":
            return True
        if step.get("tool") != "inspect_segment":
            continue
        args = step.get("args", {})
        if isinstance(args, Mapping) and args.get("candidate_options"):
            return True
    return False


def _blocked_final_reason(
    *,
    question: str,
    has_inspect_with_candidate_options: bool,
    workspace: EvidenceWorkspace,
    answer: str,
    citations: Sequence[str],
) -> str:
    if extract_candidate_options(question) and not has_inspect_with_candidate_options:
        return "mcq_final_requires_local_visual_read"
    if not workspace.has_non_navigation_visual_citation(citations):
        return "final_requires_non_navigation_visual_evidence"
    return ""


def _hard_skill_gate_reason(
    *,
    skill_name: str,
    question: str,
    table: Mapping[str, Any],
    selected_option: str | None,
    citations: Sequence[str],
) -> str:
    if extract_candidate_options(question) and not selected_option:
        return "answer_agent_need_more_evidence"

    support = selected_option_has_structured_support(table, selected_option=selected_option)
    if not support.passed:
        return support.name

    weak = no_decisive_weak_grounding(table, selected_option=selected_option)
    if not weak.passed:
        return weak.name

    conflict = no_unaddressed_conflict(table, selected_option=selected_option, cited_obs_ids=citations)
    if not conflict.passed:
        return conflict.name

    if skill_name == "temporal_ordering":
        temporal = temporal_order_consistent(table, selected_option=selected_option)
        if not temporal.passed:
            return "temporal_order_requires_confirmed_event_timestamps"
    return ""


def _reflection_rule_for_failure(failure_tag: str) -> str:
    rules = {
        "planner_json_parse_error": "return valid JSON matching the continue/final response contract before using tools",
        "final_requires_non_navigation_visual_evidence": "cite a non-navigation visual observation before finalizing",
        "mcq_final_requires_local_visual_read": "localize a candidate and call vision_read or inspect_segment before finalizing MCQ answers",
        "answer_agent_need_more_evidence": "request targeted evidence when AnswerAgent abstains instead of forcing an option",
        "selected_option_has_structured_support": "map options only from structured visual facts with candidate_option_relations",
        "no_decisive_weak_grounding": "upgrade weak or inferred support to visually_confirmed evidence before finalizing",
        "no_unaddressed_conflict": "resolve stronger conflicting option support before finalizing",
        "temporal_order_requires_confirmed_event_timestamps": "confirm every event timestamp before comparing option sequence",
    }
    return rules.get(str(failure_tag), "request targeted evidence before finalizing")


def _skill_target_facts(*, question: str, skill_name: str) -> list[str]:
    if skill_name == "temporal_ordering":
        events = _temporal_events_from_question(question)
        if events:
            return events
    semantic = _semantic_question_text(question)
    return [semantic] if semantic else []


def _semantic_question_text(question: str) -> str:
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[A-H][.)]\s+\S+", stripped):
            continue
        if "answer with" in stripped.lower() and "option letter" in stripped.lower():
            continue
        lines.append(stripped)
    return " ".join(lines).strip() or question.strip()


def _temporal_events_from_question(question: str, *, max_events: int = 4) -> list[str]:
    options = extract_candidate_options(question)
    sources = options or [_semantic_question_text(question)]
    events = []
    seen = set()
    for source in sources:
        text = re.sub(r"^[A-H][.)]\s*", "", str(source).strip())
        chunks = re.split(r"\bthen\b|\bbefore\b|\bafter\b|,|;|->|/|\|", text, flags=re.IGNORECASE)
        for chunk in chunks:
            event = re.sub(r"\b(first|last|earlier|later|right|immediately)\b", "", chunk, flags=re.IGNORECASE)
            event = re.sub(r"\s+", " ", event).strip(" .:-")
            if len(event) < 2:
                continue
            key = event.lower()
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
            if len(events) >= max_events:
                return events
    return events


def _answer_option_letter(answer: str) -> str | None:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(answer), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _should_run_answer_probe(
    *,
    budget: AgentBudget,
    question: str,
    round_number: int,
    has_inspect_with_candidate_options: bool,
    citations: Sequence[str],
) -> bool:
    if not budget.reserve_final_round:
        return False
    if budget.answer_probe_rounds_before_final <= 0:
        return False
    target_round = budget.max_rounds - budget.answer_probe_rounds_before_final
    if target_round < 1 or round_number != target_round:
        return False
    if not extract_candidate_options(question):
        return False
    return has_inspect_with_candidate_options and bool(citations)


def _append_candidate_options_to_tool_question(question: str, *, candidate_options: Sequence[str]) -> str:
    if not candidate_options:
        return question
    if all(option in question for option in candidate_options):
        return question
    options_text = "\n".join(candidate_options)
    return f"{question}\n\nOptions:\n{options_text}"


def _update_tool_class_counts(tool_class_counts: dict[str, int], program: Sequence[Mapping[str, Any]]) -> None:
    for step in program:
        tool_class = _tool_class(str(step.get("tool", "")))
        tool_class_counts[tool_class] = tool_class_counts.get(tool_class, 0) + 1


def _program_signature(program: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(list(program), ensure_ascii=True, sort_keys=True, default=str)


def _tool_budget_available(
    *,
    budget: AgentBudget,
    tool_name: str,
    tool_class_counts: Mapping[str, int],
    pending_tool_class_counts: Mapping[str, int],
) -> bool:
    if budget.free_exploration:
        return True
    tool_class = _tool_class(tool_name)
    limit = _tool_budget_limit(budget=budget, tool_class=tool_class)
    used = int(tool_class_counts.get(tool_class, 0)) + int(pending_tool_class_counts.get(tool_class, 0))
    return used < limit


def _tool_budget_limit(*, budget: AgentBudget, tool_class: str) -> int:
    if tool_class == "expensive":
        return budget.expensive_tool_budget
    if tool_class == "verifier":
        return budget.verifier_tool_budget
    return budget.cheap_tool_budget


def _tool_class(tool_name: str) -> str:
    return _TOOL_CLASSES.get(tool_name, "cheap")


def _segment_has_index_text(segment: Any) -> bool:
    return bool(getattr(segment, "low_fps_caption", ""))


def _dynamic_segment_id(*, start_sec: float, end_sec: float) -> str:
    return f"window_{int(round(start_sec * 1000)):09d}_{int(round(end_sec * 1000)):09d}"


def _normalize_dynamic_window(
    *,
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    label: str,
    end_tolerance_sec: float = 1.0,
) -> tuple[float, float]:
    if start_sec > duration_sec and end_sec > duration_sec and end_sec / 1000.0 <= duration_sec + end_tolerance_sec:
        start_sec = start_sec / 1000.0
        end_sec = end_sec / 1000.0
    if end_sec > duration_sec and end_sec - duration_sec <= end_tolerance_sec:
        end_sec = duration_sec
    if start_sec < 0 or end_sec <= start_sec or end_sec > duration_sec:
        raise ValueError(f"Invalid dynamic segment window {label}: {start_sec}-{end_sec}")
    return start_sec, end_sec


def _segment_from_dynamic_id(segment_id: str, *, duration_sec: float) -> Optional[VideoSegment]:
    match = re.fullmatch(r"window_(\d{9})_(\d{9})", segment_id)
    if not match:
        return None
    start_sec, end_sec = _normalize_dynamic_window(
        start_sec=int(match.group(1)) / 1000.0,
        end_sec=int(match.group(2)) / 1000.0,
        duration_sec=duration_sec,
        label=f"for {segment_id}",
    )
    return VideoSegment(
        segment_id=segment_id,
        start_sec=start_sec,
        end_sec=end_sec,
        source="dynamic_window",
    )


def _scene_segment_or_none(scene_index: SceneIndex, segment_id: str) -> Optional[VideoSegment]:
    try:
        return scene_index.get(segment_id)
    except ValueError:
        return None


def _partial_answer_from_ledger(ledger_text: str, max_claims: int = 5) -> str:
    claims = []
    for line in ledger_text.splitlines():
        match = re.search(r"claim:\s*(.*?)\s*\|\s*limitations:", line)
        if match:
            claim = match.group(1).strip()
            if claim:
                claims.append(claim)
        if len(claims) >= max_claims:
            break
    if not claims:
        return ""
    return "Partial evidence summary (budget exhausted): " + " ".join(
        f"[{index}] {claim}" for index, claim in enumerate(claims, start=1)
    )


def _uninspected_segment_summary(*, scene_index: SceneIndex, inspected_segment_ids: Sequence[str], limit: int = 12) -> str:
    inspected = set(inspected_segment_ids)
    candidates = [segment.segment_id for segment in scene_index.segments if segment.segment_id not in inspected]
    if not candidates:
        return "(none)"
    visible = candidates[:limit]
    suffix = f", ... {len(candidates) - limit} more" if len(candidates) > limit else ""
    return ", ".join(visible) + suffix
