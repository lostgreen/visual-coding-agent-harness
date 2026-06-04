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
from .question_policy import classify_question_route, extract_candidate_options, select_question_playbook


_SEGMENT_MEDIA_TOOLS = {"caption_segment", "qa_segment", "inspect_segment", "vision_read"}
_GLOBAL_VIEW_TOOLS = {"global_gist"}
_CHEAP_TOOLS = {"video_ls", "search_segments", "read_segment", "expand_window", "zoom", "summarize_ledger_evidence"}
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

        rounds: list[IterativeRound] = []
        citations: list[str] = []
        inspected_segment_ids: set[str] = set()
        tool_class_counts = {"cheap": 0, "expensive": 0, "verifier": 0}
        has_inspect_with_candidate_options = False
        answer_feedback: list[str] = []

        for round_number in range(1, self.budget.max_rounds + 1):
            ledger_text = self._read_ledger()
            final_round_reserved = self.budget.reserve_final_round and round_number == self.budget.max_rounds
            self.workspace.write_trace_event(
                "iterative_round_start",
                {"round": round_number, "question": question, "evidence_count": len(citations)},
            )
            planner_response = self.backend.generate(
                BackendRequest(
                    task="replan",
                    prompt=_replanning_prompt(
                        question=question,
                        scene_index=self.scene_index,
                        ledger_text=ledger_text,
                        round_number=round_number,
                        budget=self.budget,
                        inspected_segment_ids=sorted(inspected_segment_ids),
                        tool_class_counts=tool_class_counts,
                        final_round_reserved=final_round_reserved,
                        answer_feedback=answer_feedback,
                    ),
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
            if fallback_segment_id is not None and _tool_budget_available(
                budget=self.budget,
                tool_name="caption_segment",
                tool_class_counts=tool_class_counts,
                pending_tool_class_counts=pending_tool_class_counts,
            ):
                segment = self.scene_index.get(fallback_segment_id)
                normalized.append(
                    {
                        "tool": "caption_segment",
                        "args": {
                            "video_path": video_path,
                            "segment_id": segment.segment_id,
                            "start_sec": segment.start_sec,
                            "end_sec": segment.end_sec,
                            "question": question,
                            "nframes": self.budget.default_nframes,
                        },
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
        args: dict[str, Any] = {"segment_id": fallback_segment_id, "question": question}
        candidate_options = extract_candidate_options(question)
        if candidate_options:
            args["candidate_options"] = list(candidate_options)
        return [{"tool": "inspect_segment", "args": args, "assign": f"required_{fallback_segment_id}"}]

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
) -> str:
    inspected_line = ", ".join(inspected_segment_ids) if inspected_segment_ids else "(none)"
    uninspected_line = _uninspected_segment_summary(scene_index=scene_index, inspected_segment_ids=inspected_segment_ids)
    playbook = select_question_playbook(question)
    counts = tool_class_counts or {}
    remaining_budget_line = (
        "free exploration mode; no per-class tool budget, only emergency round/tool-call caps"
        if budget.free_exploration
        else (
            f"cheap={max(0, budget.cheap_tool_budget - int(counts.get('cheap', 0)))}, "
            f"expensive={max(0, budget.expensive_tool_budget - int(counts.get('expensive', 0)))}, "
            f"verifier={max(0, budget.verifier_tool_budget - int(counts.get('verifier', 0)))}"
        )
    )
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    answer_feedback_line = (
        "Answer Agent says these evidence gaps must be resolved before final: "
        + "; ".join(str(item) for item in answer_feedback[:5])
        + "\n"
        if answer_feedback
        else ""
    )
    return (
        "You are an autonomous visual agent exploring a long video with tools.\n"
        "Planner input mode: text-only. Use the scene index and evidence ledger; tools inspect pixels/video.\n"
        "Use coarse-to-fine search: inspect promising segments, read the evidence ledger, then either continue or answer.\n"
        f"{playbook.to_prompt()}\n"
        "Available tools:\n"
        "- video_ls(query: str = '', max_segments: int = 16, top_k: int = 5)\n"
        "- search_segments(query: str, top_k: int = 5, modalities: list = [])\n"
        "- read_segment(segment_id: str)\n"
        "- expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0)\n"
        "- zoom(segment_id: str, target_granularity_sec: float = 60.0)\n"
        "- global_gist(video_path: str, question: str, duration_sec: float, nframes: int = 64, max_pixels: int = 151200)\n"
        "- caption_segments(segment_ids: list = [], question: str = 'Create a concise search caption for this segment.', nframes: int = 8, max_pixels: int = 151200, fps: float = 0.0, max_segments: int = 3)\n"
        "- ingest_segment_metadata(segment_id: str, low_fps_caption: str = '', asr_text: str = '', ocr_text: str = '', entities: list = [])\n"
        "- summarize_ledger_evidence(max_claims: int = 5)\n"
        "- verify_ledger_answer(answer: str, ledger_text: str = '', question: str = '', candidate_options: list = [], min_score: float = 0.6, required_citations: list = [])\n"
        "- vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = '', nframes: int = 16, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- inspect_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, candidate_options: list = [], nframes: int = 16, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 8, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- qa_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 8, max_pixels: int = 151200, fps: float = 0.0)\n"
        "Return only JSON with one of these schemas:\n"
        '{"status": "continue", "rationale": string, "program": [{"tool": string, "args": object, "assign": string}]}\n'
        '{"status": "final", "answer": string, "citations": [observation_id], "confidence": number}\n'
        "Rules:\n"
        "- Use video_ls first for open-ended description tasks or when the relevant segment is unclear.\n"
        "- For gist/global questions, use global_gist before local decomposition and cite it as the direct floor when sufficient.\n"
        "- caption_segments is offline VideoMap cache building; avoid it in online reasoning unless the cache/indexes are empty.\n"
        "- Use navigation output as a map, then delegate localized visual reading to vision_read or inspect_segment on one candidate segment.\n"
        "- Use zoom when a coarse segment is relevant but too long; then call vision_read or inspect_segment with the returned child segment_id and start_sec/end_sec.\n"
        "- Do not spend every round on navigation-only tools; gather visual evidence before finalizing.\n"
        "- Multiple-choice answers must use vision_read or inspect_segment on a localized candidate before finalizing; candidate options are only fact-finding hints.\n"
        "- Local workers must not choose options or emit supported_option; the AnswerAgent maps cited facts to options globally.\n"
        '- JSON safety: candidate_options in JSON should be option letters only, for example ["A", "B", "C", "D"]; the harness restores full option text.\n'
        "- Do not copy quoted option text into JSON string values; refer to option letters instead.\n"
        "- Final answers require at least one non-navigation visual observation from vision_read, inspect_segment, caption_segment, or qa_segment; navigation-only evidence is insufficient.\n"
        "- Prefer segment_id references; the harness binds video_path/start_sec/end_sec.\n"
        f"- Request at most {budget.max_tool_calls_per_round} new tool call(s) this round.\n"
        "- Cheap navigation tools and expensive VLM tools have separate budgets unless free exploration mode is active.\n"
        "- In free exploration mode, prioritize answer quality: keep using tools until evidence is sufficient, then finalize with citations.\n"
        f"- Remaining tool budgets: {remaining_budget_line}.\n"
        f"{final_round_line}"
        f"{answer_feedback_line}"
        "- Do not repeat already inspected segments unless the ledger says the prior observation was unusable.\n"
        "- Continue when evidence is missing, ambiguous, or too coarse.\n"
        "- Use verify_ledger_answer before finalizing when answer support is uncertain.\n"
        "- Final answers must cite observation ids from the ledger.\n"
        f"Round: {round_number}/{budget.max_rounds}\n"
        f"Question: {question}\n"
        f"Already inspected segments: {inspected_line}\n"
        f"Uninspected segment candidates: {uninspected_line}\n"
        "Scene index:\n"
        f"{scene_index.summary(max_segments=64)}\n"
        "Evidence ledger:\n"
        f"{ledger_text}"
    )


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


def _blocked_final_reason(*, question: str, has_inspect_with_candidate_options: bool) -> str:
    if extract_candidate_options(question) and not has_inspect_with_candidate_options:
        return "mcq_final_requires_local_visual_read"
    return ""


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
