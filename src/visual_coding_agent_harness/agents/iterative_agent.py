"""Iterative visual agent for coarse-to-fine video exploration."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..interpreter import ProgramInterpreter
from ..registry import ToolError, ToolRegistry
from ..video_index import SceneIndex, VideoSegment
from ..workspace import EvidenceWorkspace
from .answer_agent import AnswerAgent, AnswerAgentResult
from .contracts import VISUAL_EVIDENCE_NFRAMES
from .context_budget import default_context_budget_allocator
from .followup import FollowupBudget, FollowupRoute, FollowupScheduler, FollowupTarget
from .open_questions import exploration_question, rewrite_exploration_question_with_model
from .output_quality import is_unsupported_claim
from .prompt_stack import build_replanning_prompt, compose_replanning_prompt_blocks, render_prompt_blocks
from .question_policy import classify_question_route, extract_candidate_options, select_question_playbook
from .skills.predicates import (
    grounding_quality_floor,
    no_decisive_weak_grounding,
    no_unaddressed_conflict,
    selected_option_has_structured_support,
    temporal_order_consistent,
)
from .skills.specs import SkillSpec, builtin_skill_registry, select_skill


_SEGMENT_MEDIA_TOOLS = {"caption_segment", "qa_segment", "inspect_segment", "vision_read"}
_GLOBAL_VIEW_TOOLS = {"global_gist"}
_CHEAP_TOOLS = {
    "video_ls",
    "search_segments",
    "ground_question",
    "read_segment",
    "read_segment_detail",
    "target_coverage",
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
_ONE_SHOT_TOOLS: frozenset[str] = frozenset({"global_gist"})


def _exhausted_one_shot_tools(workspace: Any) -> frozenset[str]:
    return frozenset(
        tool_name for tool_name in _ONE_SHOT_TOOLS if workspace.observation_count(tool_name=tool_name) >= 1
    )


@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = 8
    max_tool_calls_per_round: int = 2
    default_nframes: int = VISUAL_EVIDENCE_NFRAMES
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
    context_budget_tokens: int = 12000
    context_budget_ratios: Mapping[str, float] | None = None
    max_repeated_programs: int = 3
    hard_skill_runtime: bool = False
    reflection_memory_max_items: int = 5
    disable_global_gist_route: bool = False
    rewrite_mcq_for_exploration: bool = False

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


@dataclass(frozen=True)
class NormalizationNote:
    tool: str
    reason: str
    original: Mapping[str, Any] = field(default_factory=dict)
    resolved: Mapping[str, Any] = field(default_factory=dict)
    next_action: str = ""


@dataclass(frozen=True)
class SkillTargetFact:
    fact: str
    mutex_group_id: str = ""


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
        self.context_allocator = default_context_budget_allocator(
            total_budget_tokens=self.budget.context_budget_tokens,
            slot_ratios=dict(self.budget.context_budget_ratios) if self.budget.context_budget_ratios else None,
        )

    def run(self, *, question: str, video_path: str) -> IterativeRunResult:
        raw_question = question
        exploration_question_text = self._question_for_exploration(raw_question)
        self.workspace.ensure_hypothesis(question)
        if not self.budget.rewrite_mcq_for_exploration:
            self._seed_scene_coverage_evidence(raw_question)
        if (
            not self.budget.disable_global_gist_route
            and classify_question_route(raw_question) == "gist_global"
            and self._has_tool("global_gist")
        ):
            global_result = self._try_global_gist_route(question=exploration_question_text, video_path=video_path)
            if global_result is not None:
                return global_result

        rounds: list[IterativeRound] = []
        citations: list[str] = []
        if self.budget.hard_skill_runtime:
            skill_result = self._try_hard_skill_route(
                question=raw_question,
                exploration_question=exploration_question_text,
                video_path=video_path,
            )
            if skill_result is not None:
                if skill_result.status == "final" or len(skill_result.rounds) >= self.budget.max_rounds:
                    return skill_result
                rounds.extend(skill_result.rounds)
                citations.extend(str(citation) for citation in skill_result.citations)
                self.workspace.write_trace_event(
                    "hard_skill_followup_handoff",
                    {
                        "status": skill_result.status,
                        "rounds": len(skill_result.rounds),
                        "citations": list(skill_result.citations),
                    },
                )

        inspected_segment_ids: set[str] = {
            str(segment_id)
            for round_item in rounds
            for segment_id in _segment_ids_from_program(round_item.program)
        }
        tool_class_counts = _tool_class_counts_for_rounds(rounds)
        has_inspect_with_candidate_options = any(
            _program_has_inspect_with_candidate_options(round_item.program) for round_item in rounds
        )
        answer_feedback: list[str] = []
        last_round_normalization_notes: list[NormalizationNote] = []
        repeated_program_key = ""
        repeated_program_count = 0
        no_evidence_growth_rounds = 0
        last_evidence_table_row_count = self.workspace.evidence_table_row_count()
        all_segments_answer_attempted = False

        for round_number in range(len(rounds) + 1, self.budget.max_rounds + 1):
            ledger_text = self._read_ledger()
            final_round_reserved = self.budget.reserve_final_round and round_number == self.budget.max_rounds
            evidence_status_summary = self.workspace.evidence_status_summary(
                question=raw_question,
                options=extract_candidate_options(raw_question),
            )
            exhausted_tools = _exhausted_one_shot_tools(self.workspace)
            planner_prompt, context_report = build_replanning_prompt(
                question=exploration_question_text,
                scene_index=self.scene_index,
                ledger_text=ledger_text,
                round_number=round_number,
                budget=self.budget,
                allocator=self.context_allocator,
                inspected_segment_ids=sorted(inspected_segment_ids),
                tool_class_counts=tool_class_counts,
                final_round_reserved=final_round_reserved,
                answer_feedback=answer_feedback,
                normalization_notes=last_round_normalization_notes,
                hypothesis_text=self.workspace.read_hypothesis_text(),
                reflection_memory=self.workspace.reflection_memory(max_items=self.budget.reflection_memory_max_items),
                evidence_status_summary=evidence_status_summary,
                exhausted_tools=exhausted_tools,
            )
            self.workspace.write_trace_event("context_budget_report", asdict(context_report))
            self.workspace.write_trace_event(
                "iterative_round_start",
                {
                    "round": round_number,
                    "question": exploration_question_text,
                    "raw_question": raw_question if exploration_question_text != raw_question else "",
                    "evidence_count": len(citations),
                },
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
                    route=classify_question_route(raw_question),
                    failure_tag="planner_json_parse_error",
                    rule="return valid JSON matching the continue/final response contract before using tools",
                )
                action = {
                    "status": "continue",
                    "rationale": "planner_json_parse_error",
                    "program": self._fallback_inspector_program(
                        question=exploration_question_text,
                        inspected_segment_ids=inspected_segment_ids,
                    ),
                }
            status = str(action.get("status", "continue"))
            rationale = str(action.get("rationale", ""))
            planned_program: Any = action.get("program", [])
            planner_skill, skill_status = _planner_selected_skill(action)
            if skill_status["status"] == "selected":
                self.workspace.write_trace_event(
                    "planner_skill_selection",
                    {
                        "round": round_number,
                        "requested_skill": skill_status["requested_skill"],
                        "resolved_skill": planner_skill.name,
                    },
                )
            elif skill_status["status"] == "missing":
                self.workspace.write_trace_event(
                    "planner_skill_missing",
                    {
                        "round": round_number,
                        "message": "Planner omitted skill; continuing with ordinary exploration policy.",
                    },
                )
            elif skill_status["status"] == "invalid":
                self.workspace.write_trace_event(
                    "planner_skill_invalid",
                    {
                        "round": round_number,
                        "requested_skill": skill_status["requested_skill"],
                        "message": "Planner requested an unknown skill; continuing with ordinary exploration policy.",
                    },
                )

            if status == "final":
                final_citations = [str(item) for item in action.get("citations", [])]
                final_source = "planner_final"
                if extract_candidate_options(raw_question):
                    self.workspace.write_trace_event(
                        "planner_final_answer_agent_takeover",
                        {
                            "round": round_number,
                            "planner_answer": str(action.get("answer", "")),
                            "planner_citations": final_citations,
                        },
                    )
                    takeover_result = self._try_answer_agent_final(
                        question=raw_question,
                        video_path=video_path,
                        rounds=rounds,
                        round_number=round_number,
                        source="planner_final_takeover",
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                    )
                    if takeover_result is not None:
                        return takeover_result
                    if not _has_minimum_non_navigation_visual_citations(
                        workspace=self.workspace,
                        question=raw_question,
                        citations=final_citations,
                        minimum=2,
                    ):
                        blocked_reason = "planner_final_requires_answer_agent"
                    else:
                        blocked_reason = _blocked_planner_final_reason(
                            question=raw_question,
                            has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                            workspace=self.workspace,
                            answer=str(action.get("answer", "")),
                            citations=final_citations,
                            planner_skill=planner_skill,
                        )
                    final_source = "planner_final_after_answer_agent_abstain"
                else:
                    blocked_reason = _blocked_planner_final_reason(
                        question=raw_question,
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                        workspace=self.workspace,
                        answer=str(action.get("answer", "")),
                        citations=final_citations,
                        planner_skill=planner_skill,
                    )
                if blocked_reason:
                    self.workspace.write_trace_event(
                        "iterative_final_blocked",
                        {
                            "round": round_number,
                            "reason": blocked_reason,
                            "answer": str(action.get("answer", "")),
                            "citations": final_citations,
                        },
                    )
                    self.workspace.write_reflection_memory(
                        route=classify_question_route(raw_question),
                        failure_tag=blocked_reason,
                        rule=_reflection_rule_for_failure(blocked_reason),
                    )
                    planned_program = []
                    if not final_round_reserved:
                        planned_program = self._fallback_inspector_program(
                            question=exploration_question_text,
                            inspected_segment_ids=inspected_segment_ids,
                        )
                    status = "continue"
                    rationale = blocked_reason
                else:
                    result_round = IterativeRound(
                        round_number=round_number,
                        status="final",
                        planner_text=planner_response.text,
                        rationale=rationale,
                    )
                    rounds.append(result_round)
                    self._write_final_trace(
                        round_number=round_number,
                        answer=str(action.get("answer", "")),
                        citations=final_citations,
                        source=final_source,
                    )
                    return IterativeRunResult(
                        question=raw_question,
                        video_path=video_path,
                        answer=str(action.get("answer", "")),
                        status="final",
                        citations=final_citations,
                        confidence=float(action.get("confidence", 0.0)),
                        rounds=rounds,
                    )

            if status == "continue":
                normalization_notes: list[NormalizationNote] = []
                program = self._normalize_program(
                    planned_program,
                    question=exploration_question_text,
                    raw_question=raw_question,
                    video_path=video_path,
                    inspected_segment_ids=inspected_segment_ids,
                    tool_class_counts=tool_class_counts,
                    final_round_reserved=final_round_reserved,
                    planner_skill=planner_skill,
                    notes_out=normalization_notes,
                )
                last_round_normalization_notes = normalization_notes
                if not program and not final_round_reserved and normalization_notes:
                    self.workspace.write_trace_event(
                        "iterative_normalization_empty",
                        {
                            "round": round_number,
                            "notes": [asdict(note) for note in normalization_notes],
                        },
                    )
                    reasons = []
                    for note in normalization_notes:
                        if note.reason not in reasons:
                            reasons.append(note.reason)
                        if len(reasons) >= 3:
                            break
                    if reasons:
                        answer_feedback = [
                            "All your tool calls were filtered. Last reasons: " + ", ".join(reasons) + "."
                        ]
                if (
                    no_evidence_growth_rounds >= 2
                    and not final_round_reserved
                    and program
                    and not _program_has_visual_evidence_tool(program)
                ):
                    forced_program = self._fallback_visual_evidence_program(
                        question=exploration_question_text,
                        video_path=video_path,
                        inspected_segment_ids=inspected_segment_ids,
                        tool_class_counts=tool_class_counts,
                        planner_skill=planner_skill,
                    )
                    if forced_program:
                        self.workspace.write_trace_event(
                            "exploration_policy_adjustment",
                            {
                                "reason": "force_visual_after_no_evidence_growth",
                                "round": round_number,
                                "no_growth_rounds": no_evidence_growth_rounds,
                                "skipped_tools": [str(step.get("tool", "")) for step in program],
                                "resolved_program": forced_program,
                            },
                        )
                        program = forced_program
                if (
                    program
                    and not final_round_reserved
                    and not _program_has_visual_evidence_tool(program)
                    and _all_scene_segments_inspected(self.scene_index, inspected_segment_ids)
                ):
                    forced_program = self._visual_evidence_from_navigation_program(
                        program=program,
                        question=exploration_question_text,
                        video_path=video_path,
                        tool_class_counts=tool_class_counts,
                        planner_skill=planner_skill,
                    )
                    if forced_program:
                        self.workspace.write_trace_event(
                            "exploration_policy_adjustment",
                            {
                                "round": round_number,
                                "reason": "force_visual_from_navigation_no_growth",
                                "original_tools": [str(step.get("tool", "")) for step in program],
                                "forced_tools": [str(step.get("tool", "")) for step in forced_program],
                            },
                        )
                        program = forced_program
                if (
                    round_number > 1
                    and extract_candidate_options(raw_question)
                    and not final_round_reserved
                    and program
                    and not _program_has_visual_evidence_tool(program)
                    and not _evidence_status_has_strong_option_support(evidence_status_summary)
                ):
                    forced_program = self._fallback_visual_evidence_program(
                        question=exploration_question_text,
                        video_path=video_path,
                        inspected_segment_ids=inspected_segment_ids,
                        tool_class_counts=tool_class_counts,
                        planner_skill=planner_skill,
                    )
                    if forced_program:
                        self.workspace.write_trace_event(
                            "exploration_policy_adjustment",
                            {
                                "reason": "force_uninspected_visual_without_option_support",
                                "round": round_number,
                                "skipped_tools": [str(step.get("tool", "")) for step in program],
                                "resolved_program": forced_program,
                                "evidence_status": evidence_status_summary,
                            },
                        )
                        program = forced_program
                if (
                    extract_candidate_options(raw_question)
                    and not final_round_reserved
                    and not all_segments_answer_attempted
                    and citations
                    and _all_scene_segments_inspected(self.scene_index, inspected_segment_ids)
                    and not _program_has_visual_evidence_tool(program)
                ):
                    all_segments_answer_attempted = True
                    sweep_final = self._try_answer_agent_final(
                        question=raw_question,
                        video_path=video_path,
                        rounds=rounds,
                        round_number=round_number,
                        source="all_segments_inspected",
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                        program=program,
                        observation_ids=[],
                    )
                    if sweep_final is not None:
                        return sweep_final
            else:
                program = []
                last_round_normalization_notes = []
            if final_round_reserved and not program:
                answer_result = AnswerAgent(self.backend).run(
                    question=raw_question,
                    evidence_text=ledger_text,
                    evidence_table=self._answer_evidence_table(raw_question),
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
                    blocked_reason = _blocked_final_reason(
                        question=raw_question,
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                        workspace=self.workspace,
                        answer=answer_result.answer,
                        citations=answer_result.citations,
                    )
                    if blocked_reason:
                        self.workspace.write_trace_event(
                            "iterative_final_blocked",
                            {
                                "round": round_number,
                                "source": "reserved_final",
                                "reason": blocked_reason,
                                "answer": answer_result.answer,
                                "citations": list(answer_result.citations),
                            },
                        )
                        answer_feedback = [blocked_reason]
                    else:
                        rounds.append(
                            IterativeRound(
                                round_number=round_number,
                                status="final",
                                planner_text=answer_result.raw_text,
                                rationale=answer_result.rationale,
                            )
                        )
                        self._write_final_trace(
                            round_number=round_number,
                            answer=answer_result.answer,
                            citations=answer_result.citations,
                            source="answer_agent",
                        )
                        return IterativeRunResult(
                            question=raw_question,
                            video_path=video_path,
                            answer=answer_result.answer,
                            status="final",
                            citations=list(answer_result.citations),
                            confidence=answer_result.confidence,
                            rounds=rounds,
                        )
                if answer_result.status == "final":
                    continue
                low_confidence_result = self._try_low_confidence_final(
                    answer_result=answer_result,
                    question=raw_question,
                    video_path=video_path,
                    rounds=rounds,
                    round_number=round_number,
                    source="reserved_final",
                    program=[],
                    observation_ids=[],
                )
                if low_confidence_result is not None:
                    return low_confidence_result
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
                guard_final = self._try_answer_agent_final(
                    question=raw_question,
                    video_path=video_path,
                    rounds=rounds,
                    round_number=round_number,
                    source="repeated_program_guard",
                    has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                    program=program,
                    observation_ids=[],
                )
                if guard_final is not None:
                    return guard_final
                plural = "round" if len(rounds) == 1 else "rounds"
                partial_answer = _partial_answer_from_ledger(self._read_ledger())
                return IterativeRunResult(
                    question=raw_question,
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
            current_evidence_table_row_count = self.workspace.evidence_table_row_count()
            if current_evidence_table_row_count <= last_evidence_table_row_count:
                no_evidence_growth_rounds += 1
            else:
                no_evidence_growth_rounds = 0
            last_evidence_table_row_count = current_evidence_table_row_count
            if no_evidence_growth_rounds >= 2 and not final_round_reserved:
                answer_result = AnswerAgent(self.backend).run(
                    question=raw_question,
                    evidence_text=self._read_ledger(),
                    evidence_table=self._answer_evidence_table(raw_question),
                )
                self.workspace.write_trace_event(
                    "iterative_no_progress_guard",
                    {
                        "round": round_number,
                        "reason": "evidence_table_no_growth",
                        "no_growth_rounds": no_evidence_growth_rounds,
                        "evidence_table_rows": current_evidence_table_row_count,
                        "answer_status": answer_result.status,
                    },
                )
                low_confidence_result = self._try_low_confidence_final(
                    answer_result=answer_result,
                    question=raw_question,
                    video_path=video_path,
                    rounds=rounds,
                    round_number=round_number,
                    source="evidence_table_no_growth",
                    program=program,
                    observation_ids=observation_ids,
                )
                if low_confidence_result is not None:
                    return low_confidence_result
            if _should_run_answer_probe(
                budget=self.budget,
                question=raw_question,
                round_number=round_number,
                has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                citations=citations,
            ):
                answer_result = AnswerAgent(self.backend).run(
                    question=raw_question,
                    evidence_text=self._read_ledger(),
                    evidence_table=self._answer_evidence_table(raw_question),
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
                    blocked_reason = _blocked_final_reason(
                        question=raw_question,
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                        workspace=self.workspace,
                        answer=answer_result.answer,
                        citations=answer_result.citations,
                    )
                    if blocked_reason:
                        self.workspace.write_trace_event(
                            "iterative_final_blocked",
                            {
                                "round": round_number,
                                "source": "prefinal_probe",
                                "reason": blocked_reason,
                                "answer": answer_result.answer,
                                "citations": list(answer_result.citations),
                            },
                        )
                        answer_feedback = [blocked_reason]
                        continue
                    rounds.append(
                        IterativeRound(
                            round_number=round_number,
                            status="final",
                            planner_text=answer_result.raw_text,
                            rationale=answer_result.rationale,
                        )
                    )
                    self._write_final_trace(
                        round_number=round_number,
                        answer=answer_result.answer,
                        citations=answer_result.citations,
                        source="answer_agent",
                    )
                    return IterativeRunResult(
                        question=raw_question,
                        video_path=video_path,
                        answer=answer_result.answer,
                        status="final",
                        citations=list(answer_result.citations),
                        confidence=answer_result.confidence,
                        rounds=rounds,
                    )
                if round_number >= self.budget.max_rounds:
                    low_confidence_result = self._try_low_confidence_final(
                        answer_result=answer_result,
                        question=raw_question,
                        video_path=video_path,
                        rounds=rounds,
                        round_number=round_number,
                        source="prefinal_probe_budget_exhausted",
                        program=program,
                        observation_ids=observation_ids,
                    )
                    if low_confidence_result is not None:
                        return low_confidence_result
                answer_feedback = (
                    _sanitize_option_blind_feedback(answer_result.missing_evidence, raw_question=raw_question)
                    if self.budget.rewrite_mcq_for_exploration
                    else list(answer_result.missing_evidence)
                )
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
        budget_final = self._try_answer_agent_final(
            question=raw_question,
            video_path=video_path,
            rounds=rounds,
            round_number=self.budget.max_rounds,
            source="budget_exhausted",
            has_inspect_with_candidate_options=has_inspect_with_candidate_options,
        )
        if budget_final is not None:
            return budget_final
        if extract_candidate_options(raw_question) and citations:
            answer_result = AnswerAgent(self.backend).run(
                question=raw_question,
                evidence_text=self._read_ledger(),
                evidence_table=self._answer_evidence_table(raw_question),
            )
            self.workspace.write_trace_event(
                "iterative_answer_agent",
                {
                    "round": self.budget.max_rounds,
                    "source": "budget_exhausted",
                    "status": answer_result.status,
                    "answer": answer_result.answer,
                    "citations": list(answer_result.citations),
                    "missing_evidence": list(answer_result.missing_evidence),
                },
            )
            low_confidence_result = self._try_low_confidence_final(
                answer_result=answer_result,
                question=raw_question,
                video_path=video_path,
                rounds=rounds,
                round_number=self.budget.max_rounds,
                source="budget_exhausted",
            )
            if low_confidence_result is not None:
                return low_confidence_result
        plural = "round" if self.budget.max_rounds == 1 else "rounds"
        partial_answer = _partial_answer_from_ledger(self._read_ledger())
        return IterativeRunResult(
            question=raw_question,
            video_path=video_path,
            answer=partial_answer
            or f"Stopped after {self.budget.max_rounds} exploration {plural} with partial evidence.",
            status="max_rounds_reached",
            citations=citations,
            rounds=rounds,
        )

    def _question_for_exploration(self, question: str) -> str:
        if not self.budget.rewrite_mcq_for_exploration or not extract_candidate_options(question):
            return question
        rewrite = rewrite_exploration_question_with_model(
            self.backend,
            question=question,
            route_hint=classify_question_route(question),
        )
        self.workspace.write_trace_event(
            "mcq_exploration_question_rewrite",
            {
                "used_model": rewrite.used_model,
                "fallback_reason": rewrite.fallback_reason,
                "exploration_question": rewrite.exploration_question,
                "focus_points": list(rewrite.focus_points),
                "target_entities": list(rewrite.target_entities),
            },
        )
        return rewrite.exploration_question or exploration_question(question)

    def _try_answer_agent_final(
        self,
        *,
        question: str,
        video_path: str,
        rounds: Sequence[IterativeRound],
        round_number: int,
        source: str,
        has_inspect_with_candidate_options: bool,
        program: Sequence[Mapping[str, Any]] = (),
        observation_ids: Sequence[str] = (),
    ) -> IterativeRunResult | None:
        if not extract_candidate_options(question):
            return None
        answer_result = AnswerAgent(self.backend).run(
            question=question,
            evidence_text=self._read_ledger(),
            evidence_table=self._answer_evidence_table(question),
        )
        self.workspace.write_trace_event(
            "iterative_answer_agent",
            {
                "round": round_number,
                "source": source,
                "status": answer_result.status,
                "answer": answer_result.answer,
                "citations": list(answer_result.citations),
                "missing_evidence": list(answer_result.missing_evidence),
            },
        )
        if answer_result.status != "final":
            return None
        blocked_reason = _blocked_final_reason(
            question=question,
            has_inspect_with_candidate_options=has_inspect_with_candidate_options,
            workspace=self.workspace,
            answer=answer_result.answer,
            citations=answer_result.citations,
        )
        if blocked_reason:
            self.workspace.write_trace_event(
                "iterative_final_blocked",
                {
                    "round": round_number,
                    "source": source,
                    "reason": blocked_reason,
                    "answer": answer_result.answer,
                    "citations": list(answer_result.citations),
                },
            )
            return None
        final_rounds = list(rounds)
        final_rounds.append(
            IterativeRound(
                round_number=round_number,
                status="final",
                planner_text=answer_result.raw_text,
                rationale=answer_result.rationale,
                program=program,
                observation_ids=observation_ids,
            )
        )
        self._write_final_trace(
            round_number=round_number,
            answer=answer_result.answer,
            citations=answer_result.citations,
            source=source,
        )
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=answer_result.answer,
            status="final",
            citations=list(answer_result.citations),
            confidence=answer_result.confidence,
            rounds=final_rounds,
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
        planner_skill: SkillSpec | None = None,
        notes_out: list[NormalizationNote] | None = None,
        raw_question: str = "",
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(program, list):
            raise ValueError("Planner action status=continue requires a list program")

        normalized = []
        reserved_segment_ids = set(inspected_segment_ids)
        pending_tool_class_counts = {"cheap": 0, "expensive": 0, "verifier": 0}
        active_skill = planner_skill
        blocked_route_violation = False
        pool_exhausted_logged = False
        pending_one_shot_tools: set[str] = set()
        for step in program:
            if not isinstance(step, Mapping):
                raise ValueError("Planner program steps must be objects")
            if "tool" not in step:
                raise ValueError("Planner program step is missing required 'tool'")
            if len(normalized) >= self.budget.max_tool_calls_per_round:
                break

            tool_name = str(step["tool"])
            args = dict(step.get("args", {}))
            alias_repair = self._repair_tool_alias(tool_name=tool_name, args=args)
            if alias_repair is not None:
                original_tool_name = tool_name
                original_args = dict(args)
                tool_name, args, repair_reason = alias_repair
                self.workspace.write_trace_event(
                    "route_tool_repaired",
                    {
                        "skill": active_skill.name if active_skill is not None else "",
                        "requested_tool": original_tool_name,
                        "resolved_tool": tool_name,
                        "reason": repair_reason,
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool=original_tool_name,
                    reason=repair_reason,
                    original={"tool": original_tool_name, "args": original_args},
                    resolved={"tool": tool_name, "args": args},
                )
            skill_name_reason = _skill_name_as_tool_reason(tool_name)
            if skill_name_reason:
                blocked_route_violation = True
                next_action = (
                    f"{tool_name} is a skill name, not an executable tool. Put it in the top-level "
                    "`skill` field and choose a concrete allowed action from the active skill."
                )
                self.workspace.write_trace_event(
                    "exploration_policy_adjustment",
                    {
                        "reason": "skill_name_as_tool",
                        "skipped_tool": tool_name,
                        "skill": active_skill.name if active_skill is not None else skill_name_reason,
                        "next_action": next_action,
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="skill_name_as_tool",
                    original={"tool": tool_name, "args": args},
                    next_action=next_action,
                )
                continue
            exhausted_tools = _exhausted_one_shot_tools(self.workspace) | frozenset(pending_one_shot_tools)
            if tool_name in exhausted_tools and not self.budget.free_exploration:
                blocked_route_violation = True
                next_action = (
                    f"{tool_name} is one-shot and has already executed. Read its claim from the ledger; "
                    "pick a remaining localized visual tool on an uninspected segment, or verify/finalize."
                )
                self.workspace.write_trace_event(
                    "exploration_policy_adjustment",
                    {
                        "reason": f"{tool_name}_one_shot_exhausted",
                        "skipped_tool": tool_name,
                        "skill": active_skill.name if active_skill is not None else "",
                        "next_action": next_action,
                    },
                )
                self.workspace.write_reflection_memory(
                    route=active_skill.trigger.route if active_skill is not None else classify_question_route(question),
                    failure_tag=f"{tool_name}_one_shot_exhausted",
                    rule=(
                        f"{tool_name} is one-shot and was already executed; read its claim from the ledger "
                        "and do not request it again."
                    ),
                )
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason=f"{tool_name}_one_shot_exhausted",
                    original={"tool": tool_name, "args": args},
                    next_action=next_action,
                )
                continue
            repair = self._repair_skill_route_tool(
                tool_name=tool_name,
                args=args,
                active_skill=active_skill,
                question=question,
                video_path=video_path,
            )
            if repair is not None:
                original_tool_name = tool_name
                original_args = dict(args)
                tool_name, args, repair_reason = repair
                self.workspace.write_trace_event(
                    "route_tool_repaired",
                    {
                        "skill": active_skill.name if active_skill is not None else "",
                        "requested_tool": original_tool_name,
                        "resolved_tool": tool_name,
                        "reason": repair_reason,
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool=original_tool_name,
                    reason=repair_reason,
                    original={"tool": original_tool_name, "args": original_args},
                    resolved={"tool": tool_name, "args": args},
                )
            segment_id = args.get("segment_id")
            if segment_id and tool_name == "read_segment":
                segment = _scene_segment_or_none(self.scene_index, str(segment_id))
                if segment is not None and not _segment_has_index_text(segment):
                    original_tool_name = tool_name
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
                    _append_normalization_note(
                        notes_out,
                        tool=original_tool_name,
                        reason="upgrade_empty_read_segment_to_caption",
                        original={"tool": original_tool_name, "segment_id": segment.segment_id},
                        resolved={"tool": tool_name, "segment_id": segment.segment_id},
                    )
            if _tool_denied_by_skill(tool_name=tool_name, active_skill=active_skill, free_exploration=self.budget.free_exploration):
                blocked_route_violation = True
                _record_skill_deny_list_violation(
                    workspace=self.workspace,
                    notes_out=notes_out,
                    tool_name=tool_name,
                    args=args,
                    active_skill=active_skill,
                )
                continue
            violation = _route_violation(tool_name=tool_name, active_skill=active_skill, free_exploration=self.budget.free_exploration)
            if violation is not None:
                blocked_route_violation = True
                self.workspace.write_trace_event(
                    "route_violation",
                    {
                        "tool": tool_name,
                        "error": violation,
                        "skill": active_skill.name if active_skill is not None else "",
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="route_violation",
                    original={"tool": tool_name, "args": args},
                )
                continue
            if self._tool_accepts_argument(tool_name, "video_path") and _is_video_path_placeholder(args.get("video_path")):
                original_args = dict(args)
                args["video_path"] = video_path
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="replace_video_path_placeholder",
                    original={"tool": tool_name, "args": original_args},
                    resolved={"tool": tool_name, "args": args},
                )
            if final_round_reserved and tool_name not in _VERIFIER_TOOLS:
                self.workspace.write_trace_event(
                    "exploration_policy_adjustment",
                    {
                        "reason": "reserve_final_round",
                        "skipped_tool": tool_name,
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="reserve_final_round",
                    original={"tool": tool_name, "args": args},
                )
                continue

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
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="tool_budget_exhausted",
                    original={"tool": tool_name, "args": args, "tool_class": _tool_class(tool_name)},
                )
                continue

            if tool_name in _SEGMENT_MEDIA_TOOLS:
                segment = (
                    self._resolve_media_segment(
                        str(segment_id),
                        args=args,
                        reserved_segment_ids=reserved_segment_ids,
                        tool_name=tool_name,
                        notes_out=notes_out,
                    )
                    if segment_id
                    else self._resolve_missing_media_segment(
                        args=args,
                        reserved_segment_ids=reserved_segment_ids,
                    )
                )
                if segment is None:
                    next_free = self._resolve_next_segment_id("", reserved_segment_ids)
                    if next_free is None:
                        blocked_route_violation = True
                        next_action = (
                            "All scene segments are already inspected. Stop requesting existing segment_ids. "
                            "Use zoom or expand_window on the most informative segment, or call "
                            "verify_ledger_answer to finalize."
                        )
                        self.workspace.write_trace_event(
                            "exploration_policy_adjustment",
                            {
                                "reason": "segment_pool_exhausted",
                                "skipped_tool": tool_name,
                                "segment_id": str(segment_id or ""),
                                "next_action": next_action,
                            },
                        )
                        _append_normalization_note(
                            notes_out,
                            tool=tool_name,
                            reason="segment_pool_exhausted",
                            original={"tool": tool_name, "segment_id": str(segment_id or "")},
                            next_action=next_action,
                        )
                        if not pool_exhausted_logged:
                            self.workspace.write_reflection_memory(
                                route=classify_question_route(question),
                                failure_tag="segment_pool_exhausted",
                                rule=(
                                    "Scene index segment pool is empty; pivot to zoom/expand_window on a key "
                                    "segment or call verify_ledger_answer + final."
                                ),
                            )
                            pool_exhausted_logged = True
                    else:
                        _append_normalization_note(
                            notes_out,
                            tool=tool_name,
                            reason="unresolved_media_segment",
                            original={"tool": tool_name, "segment_id": str(segment_id or "")},
                        )
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
                    _append_normalization_note(
                        notes_out,
                        tool=tool_name,
                        reason="repair_missing_media_segment_id",
                        original={"tool": tool_name},
                        resolved={
                            "tool": tool_name,
                            "segment_id": segment.segment_id,
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
                    args["ask_for"] = _tool_exploration_question(
                        str(args["ask_for"]),
                        route_hint=planner_skill.name if planner_skill else "",
                        question_context=question,
                        forbidden_question=raw_question,
                        option_blind=self.budget.rewrite_mcq_for_exploration,
                    )
                else:
                    args.setdefault("question", question)
                    args["question"] = _tool_exploration_question(
                        str(args["question"]),
                        route_hint=planner_skill.name if planner_skill else "",
                        question_context=question,
                        forbidden_question=raw_question,
                        option_blind=self.budget.rewrite_mcq_for_exploration,
                    )
                candidate_options = list(extract_candidate_options(question))
                if candidate_options:
                    if tool_name == "vision_read":
                        args.setdefault("event_label", str(args.get("ask_for", "")))
                    if tool_name == "inspect_segment" and not self.budget.rewrite_mcq_for_exploration:
                        args["candidate_options"] = candidate_options
                args.setdefault("nframes", self.budget.default_nframes)
                reserved_segment_ids.add(segment.segment_id)

            normalized_step: dict[str, Any] = {"tool": tool_name, "args": args}
            if "assign" in step:
                normalized_step["assign"] = str(step["assign"])
            normalized.append(normalized_step)
            pending_tool_class_counts[_tool_class(tool_name)] += 1
            if tool_name in _ONE_SHOT_TOOLS:
                pending_one_shot_tools.add(tool_name)

        if not normalized and not final_round_reserved and not blocked_route_violation:
            fallback_segment_id = self._resolve_next_segment_id("", reserved_segment_ids)
            fallback_tool_name = self._fallback_visual_tool_name_for_skill(active_skill)
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
                    target_question = _local_fact_question(question=question, planner_skill=active_skill)
                    fallback_args["ask_for"] = target_question
                    fallback_args["event_label"] = target_question
                else:
                    fallback_args["question"] = _local_fact_question(question=question, planner_skill=active_skill)
                candidate_options = extract_candidate_options(question)
                if (
                    candidate_options
                    and fallback_tool_name == "inspect_segment"
                    and not self.budget.rewrite_mcq_for_exploration
                ):
                    fallback_args["candidate_options"] = list(candidate_options)
                normalized.append(
                    {
                        "tool": fallback_tool_name,
                        "args": fallback_args,
                        "assign": f"auto_{segment.segment_id}",
                    }
                )
        return normalized

    def _repair_skill_route_tool(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        active_skill: SkillSpec | None,
        question: str,
        video_path: str,
    ) -> tuple[str, dict[str, Any], str] | None:
        if self.budget.free_exploration:
            return None
        is_main_idea_route = active_skill is not None and active_skill.name == "main_idea"
        if is_main_idea_route and tool_name == "vision_read" and self._has_tool("global_gist"):
            if self.workspace.observation_count(tool_name="global_gist") >= 1:
                return None
            repaired_args: dict[str, Any] = {
                "video_path": video_path,
                "question": question,
                "duration_sec": self.scene_index.duration_sec,
            }
            if self._tool_accepts_argument("global_gist", "seed"):
                repaired_args["seed"] = max(2, self.workspace.evidence_table_row_count() + 1)
            return "global_gist", repaired_args, "repair_main_idea_vision_read_to_global_gist"
        if (
            is_main_idea_route
            and tool_name == "global_gist"
            and self.workspace.observation_count(tool_name="global_gist") >= 1
            and self._has_tool("vision_read")
        ):
            segment_id = self._resolve_next_segment_id("", set())
            if segment_id is None:
                return None
            return (
                "vision_read",
                {
                    "segment_id": segment_id,
                    "ask_for": (
                        "Describe localized main-idea evidence for this segment. Report facts only. "
                        "Focus on visible entities, events, and whether this part shows rise, stability, decline, collapse, or causes."
                    ),
                    "event_label": "localized main-idea evidence",
                },
                "repair_repeated_main_idea_global_gist_to_vision_read",
            )
        if (
            active_skill is not None
            and active_skill.name == "mutex_fact_qa"
            and tool_name == "inspect_segment"
            and self._has_tool("vision_read")
        ):
            repaired_args = {
                key: value
                for key, value in dict(args).items()
                if key not in {"candidate_options", "question"}
            }
            repaired_args.setdefault("ask_for", str(args.get("question") or args.get("ask_for") or question))
            return "vision_read", repaired_args, "repair_mutex_inspect_segment_to_vision_read"
        if (
            active_skill is not None
            and active_skill.name == "timeline_ordering"
            and (
                tool_name == "caption_segments"
                or (tool_name == "caption_segment" and args.get("segment_ids") and not args.get("segment_id"))
            )
            and self._has_tool("caption_segment")
        ):
            segment_ids = [
                str(segment_id)
                for segment_id in args.get("segment_ids", [])
                if str(segment_id).strip()
            ]
            segment_id = segment_ids[0] if segment_ids else self._resolve_next_segment_id("", set())
            if segment_id is None:
                return None
            return (
                "caption_segment",
                {
                    "segment_id": segment_id,
                    "question": str(
                        args.get("question")
                        or (
                            "Openly describe this segment's actual visible artworks, objects, people, scene changes, "
                            "onscreen text, and narrated events in presentation order. Include timestamps if possible. "
                            "Focus on concrete observations rather than conclusions."
                        )
                    ),
                    "nframes": int(args.get("nframes", self.budget.default_nframes) or self.budget.default_nframes),
                },
                "repair_timeline_batch_caption_to_caption_segment",
            )
        return None

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

    def _fallback_visual_evidence_program(
        self,
        *,
        question: str,
        video_path: str,
        inspected_segment_ids: set[str],
        tool_class_counts: Mapping[str, int],
        planner_skill: SkillSpec | None,
    ) -> list[dict[str, Any]]:
        segment_id = self._resolve_next_segment_id("", inspected_segment_ids)
        if segment_id is None:
            return []
        tool_name = self._fallback_visual_tool_name_for_skill(planner_skill)
        if tool_name is None:
            return []
        pending_tool_class_counts = {"cheap": 0, "expensive": 0, "verifier": 0}
        if not _tool_budget_available(
            budget=self.budget,
            tool_name=tool_name,
            tool_class_counts=tool_class_counts,
            pending_tool_class_counts=pending_tool_class_counts,
        ):
            return []
        segment = self.scene_index.get(segment_id)
        args: dict[str, Any] = {
            "video_path": video_path,
            "segment_id": segment.segment_id,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "nframes": self.budget.default_nframes,
        }
        target_question = _local_fact_question(question=question, planner_skill=planner_skill)
        if tool_name == "vision_read":
            args["ask_for"] = target_question
            args["event_label"] = target_question
        else:
            args["question"] = target_question
        candidate_options = extract_candidate_options(question)
        if candidate_options and tool_name == "inspect_segment":
            args["candidate_options"] = list(candidate_options)
        return [{"tool": tool_name, "args": args, "assign": f"forced_visual_{segment.segment_id}"}]

    def _visual_evidence_from_navigation_program(
        self,
        *,
        program: Sequence[Mapping[str, Any]],
        question: str,
        video_path: str,
        tool_class_counts: Mapping[str, int],
        planner_skill: SkillSpec | None,
    ) -> list[dict[str, Any]]:
        tool_name = "vision_read" if self._has_tool("vision_read") else self._fallback_visual_tool_name_for_skill(planner_skill)
        if tool_name is None:
            return []
        pending_tool_class_counts = {"cheap": 0, "expensive": 0, "verifier": 0}
        forced: list[dict[str, Any]] = []
        for step in program:
            if len(forced) >= self.budget.max_tool_calls_per_round:
                break
            args = step.get("args", {})
            if not isinstance(args, Mapping):
                continue
            segment_id = str(args.get("segment_id", "")).strip()
            if not segment_id:
                continue
            segment = _scene_segment_or_none(self.scene_index, segment_id)
            if segment is None:
                continue
            if not _tool_budget_available(
                budget=self.budget,
                tool_name=tool_name,
                tool_class_counts=tool_class_counts,
                pending_tool_class_counts=pending_tool_class_counts,
            ):
                break
            media_args: dict[str, Any] = {
                "video_path": video_path,
                "segment_id": segment.segment_id,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "nframes": self.budget.default_nframes,
            }
            target_question = _local_fact_question(question=question, planner_skill=planner_skill)
            if tool_name == "vision_read":
                media_args["ask_for"] = target_question
                media_args["event_label"] = target_question
            else:
                media_args["question"] = target_question
            forced.append(
                {
                    "tool": tool_name,
                    "args": media_args,
                    "assign": f"forced_navigation_visual_{segment.segment_id}_{len(forced) + 1}",
                }
            )
            pending_tool_class_counts[_tool_class(tool_name)] += 1
        return forced

    def _fallback_visual_tool_name_for_skill(self, planner_skill: SkillSpec | None) -> str | None:
        if planner_skill is not None and not self.budget.free_exploration:
            if planner_skill.name == "timeline_ordering":
                preferences = ["caption_segment", "vision_read", "qa_segment", "inspect_segment"]
            elif planner_skill.name in {"mutex_fact_qa", "grounded_factual_qa"}:
                preferences = ["vision_read", "qa_segment", "caption_segment", "inspect_segment"]
            elif planner_skill.name == "main_idea":
                preferences = ["vision_read", "caption_segment", "qa_segment", "inspect_segment"]
            else:
                preferences = ["vision_read", "inspect_segment", "caption_segment", "qa_segment"]
            for tool_name in preferences:
                if self._has_tool(tool_name) and tool_name in planner_skill.allowed_actions:
                    return tool_name
            return None
        return self._fallback_visual_tool_name()

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
        tool_name: str,
        notes_out: list[NormalizationNote] | None = None,
    ) -> Optional[VideoSegment]:
        scene_segment = _scene_segment_or_none(self.scene_index, requested_segment_id)
        if scene_segment is not None:
            resolved_segment_id = self._resolve_next_segment_id(requested_segment_id, reserved_segment_ids)
            if resolved_segment_id is None:
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="avoid_repeated_segment",
                    original={"tool": tool_name, "segment_id": requested_segment_id},
                )
                return None
            if resolved_segment_id != requested_segment_id:
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="avoid_repeated_segment",
                    original={"tool": tool_name, "segment_id": requested_segment_id},
                    resolved={"tool": tool_name, "segment_id": resolved_segment_id},
                )
            return self.scene_index.get(resolved_segment_id)

        if requested_segment_id in reserved_segment_ids:
            _append_normalization_note(
                notes_out,
                tool=tool_name,
                reason="avoid_repeated_segment",
                original={"tool": tool_name, "segment_id": requested_segment_id},
            )
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

    def _write_final_trace(
        self,
        *,
        round_number: int,
        answer: str,
        citations: Sequence[str],
        source: str = "",
        status: str = "final",
    ) -> None:
        payload: dict[str, Any] = {
            "round": round_number,
            "answer": answer,
            "citations": list(citations),
        }
        if source:
            payload["source"] = source
        if status != "final":
            payload["status"] = status
        provenance = self._citation_provenance(citations)
        if provenance:
            payload["citation_provenance"] = provenance
        self.workspace.write_trace_event("iterative_final", payload)

    def _citation_provenance(self, citations: Sequence[str]) -> list[dict[str, Any]]:
        cited = [str(item) for item in citations if str(item)]
        if not cited:
            return []
        table = self.workspace.evidence_table_v2(
            question="",
            options=[],
            include_legacy_worker_votes=True,
        )
        rows = table.get("rows", [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return []
        provenance: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            obs_id = str(row.get("obs_id", ""))
            evidence_id = str(row.get("evidence_id", ""))
            matched_citations = [citation for citation in cited if citation in {obs_id, evidence_id}]
            if not matched_citations:
                continue
            for citation in matched_citations:
                key = (citation, evidence_id, obs_id)
                if key in seen:
                    continue
                seen.add(key)
                provenance.append(
                    _citation_provenance_from_evidence_row(
                        citation=citation,
                        row=row,
                        scene_index=self.scene_index,
                    )
                )
        return provenance

    def _seed_scene_coverage_evidence(self, question: str) -> None:
        route = classify_question_route(question)
        options = extract_candidate_options(question)
        if not options:
            return
        if route not in {"gist_global", "temporal_order"} and not _question_has_sequence_options(options):
            return
        if self.workspace.evidence_table_row_count() > 0:
            return

        rows = [
            *_scene_coverage_evidence_rows(scene_index=self.scene_index, question=question, options=options),
            *_scene_order_evidence_rows(scene_index=self.scene_index, question=question, options=options),
        ]
        for row in rows:
            self.workspace.write_evidence_row(row)
        if rows:
            self.workspace.write_trace_event(
                "scene_coverage_evidence_seeded",
                {
                    "row_count": len(rows),
                    "obs_ids": [str(row.get("obs_id", "")) for row in rows],
                },
            )

    def _has_tool(self, tool_name: str) -> bool:
        try:
            self.registry.get(tool_name)
        except ToolError:
            return False
        return True

    def _tool_accepts_argument(self, tool_name: str, argument_name: str) -> bool:
        try:
            return str(argument_name) in self.registry.get(tool_name).parameters
        except ToolError:
            return False

    def _repair_tool_alias(self, *, tool_name: str, args: Mapping[str, Any]) -> tuple[str, dict[str, Any], str] | None:
        aliases = {"verify": "verify_ledger_answer"}
        resolved_tool = aliases.get(tool_name)
        if resolved_tool is None or not self._has_tool(resolved_tool):
            return None
        return resolved_tool, dict(args), f"repair_tool_alias_{tool_name}_to_{resolved_tool}"

    def _try_low_confidence_final(
        self,
        *,
        answer_result: AnswerAgentResult,
        question: str,
        video_path: str,
        rounds: Sequence[IterativeRound],
        round_number: int,
        source: str,
        program: Sequence[Mapping[str, Any]] = (),
        observation_ids: Sequence[str] = (),
    ) -> IterativeRunResult | None:
        if answer_result.status != "need_more_evidence" or not answer_result.has_partial_support():
            return None
        low_confidence = answer_result.as_low_confidence_final()
        if low_confidence.status != "low_confidence_final":
            return None
        if not self.workspace.has_non_navigation_visual_citation(low_confidence.citations):
            self.workspace.write_trace_event(
                "low_confidence_final_blocked",
                {
                    "round": round_number,
                    "source": source,
                    "reason": "final_requires_non_navigation_visual_evidence",
                    "citations": list(low_confidence.citations),
                },
            )
            return None
        final_rounds = list(rounds)
        final_rounds.append(
            IterativeRound(
                round_number=round_number,
                status="low_confidence_final",
                planner_text=low_confidence.raw_text,
                rationale=low_confidence.rationale,
                program=program,
                observation_ids=observation_ids,
            )
        )
        self._write_final_trace(
            round_number=round_number,
            answer=low_confidence.answer,
            citations=low_confidence.citations,
            source=source,
            status="low_confidence_final",
        )
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=low_confidence.answer,
            status="low_confidence_final",
            citations=list(low_confidence.citations),
            confidence=low_confidence.confidence,
            rounds=final_rounds,
        )

    def _try_global_gist_route(self, *, question: str, video_path: str) -> IterativeRunResult | None:
        first_step = {
            "tool": "global_gist",
            "args": {
                "video_path": video_path,
                "question": question,
                "duration_sec": self.scene_index.duration_sec,
            },
            "assign": "global_gist_1",
        }
        if self._tool_accepts_argument("global_gist", "seed"):
            first_step["args"]["seed"] = 0
        self.workspace.write_trace_event(
            "iterative_route",
            {"route": "gist_global", "tool": "global_gist", "passes_required": 1, "mode": "topic_hint_seed"},
        )
        interpreter = ProgramInterpreter(registry=self.registry, workspace=self.workspace)
        result = interpreter.run([first_step])
        self.workspace.write_trace_event(
            "global_gist_topic_seeded",
            {"observation_ids": list(result.observation_ids), "source": "global_gist_route"},
        )
        return None

    def _try_hard_skill_route(
        self,
        *,
        question: str,
        exploration_question: str,
        video_path: str,
    ) -> IterativeRunResult | None:
        skill = select_skill(question)
        if skill.name not in {"grounded_factual_qa", "mutex_fact_qa", "timeline_ordering"}:
            return None
        if skill.name == "timeline_ordering":
            if not self._has_tool("caption_segment") or not self._has_tool("vision_read"):
                return None
        elif not self._has_tool("ground_question") or not self._has_tool("vision_read"):
            return None

        route = classify_question_route(question)
        target_specs = _skill_target_fact_specs(question=exploration_question, skill_name=skill.name)
        if not target_specs:
            target_specs = _skill_target_fact_specs(question=question, skill_name=skill.name)
        if not target_specs:
            return None
        target_facts = [spec.fact for spec in target_specs]

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
        if skill.name == "timeline_ordering":
            return self._try_timeline_ordering_route(
                question=question,
                video_path=video_path,
                target_facts=target_facts,
            )

        rounds: list[IterativeRound] = []
        all_observation_ids: list[str] = []
        interpreter = ProgramInterpreter(registry=self.registry, workspace=self.workspace)
        scheduler = FollowupScheduler(
            FollowupBudget(
                global_max_followups=max(1, self.budget.max_rounds * self.budget.max_tool_calls_per_round)
            )
        )
        scheduler.enqueue(
            [
                FollowupTarget(
                    target_id=f"fu_{self.workspace.run_id}_{index:04d}",
                    query=target_spec.fact,
                    event_label=target_spec.fact,
                    route=_followup_route_for_skill(skill.name),
                    reason="hard skill target fact still needs visual grounding",
                    priority=index,
                    attempt_count=0,
                    parent_missing_evidence_id=f"target_fact_{index:04d}",
                    mutex_group_id=target_spec.mutex_group_id,
                )
                for index, target_spec in enumerate(target_specs, start=1)
            ]
        )
        last_answer_result = None
        last_failure_tag = "answer_agent_need_more_evidence"

        for round_number in range(1, self.budget.max_rounds + 1):
            targets = _next_followup_chunk(
                scheduler=scheduler,
                chunk_size=max(1, self.budget.max_tool_calls_per_round),
            )
            if not targets:
                break

            program: list[dict[str, Any]] = []
            round_observation_ids: list[str] = []
            for target in targets:
                new_observation_ids = self._run_hard_skill_followup_target(
                    interpreter=interpreter,
                    target=target,
                    video_path=video_path,
                    assign_suffix=len(all_observation_ids) + len(round_observation_ids) + 1,
                )
                program.extend(new_observation_ids["program"])
                produced_ids = [str(observation_id) for observation_id in new_observation_ids["observation_ids"]]
                round_observation_ids.extend(produced_ids)
                new_frame_sets = set(produced_ids)
                scheduler.record_attempt(target, new_frame_sets)
                self.workspace.write_trace_event(
                    "followup_attempt",
                    {
                        "round": round_number,
                        "target_id": target.target_id,
                        "route": target.route,
                        "query": target.query,
                        "event_label": target.event_label,
                        "attempt_count": target.attempt_count,
                        "new_evidence_count": len(produced_ids),
                        "observation_ids": produced_ids,
                    },
                )
                scheduler.completed.append(target)

            all_observation_ids.extend(round_observation_ids)
            if not round_observation_ids:
                rounds.append(
                    IterativeRound(
                        round_number=round_number,
                        status="need_more_evidence",
                        planner_text="",
                        rationale="hard skill follow-up produced no observations",
                        program=program,
                        observation_ids=round_observation_ids,
                    )
                )
                continue

            timeline_decision = (
                _timeline_temporal_decision(question=question, timeline=self.workspace.read_timeline_sorted())
                if _is_timeline_skill(skill.name)
                else None
            )
            if timeline_decision is not None:
                answer = str(timeline_decision["answer"])
                citations = [str(obs_id) for obs_id in timeline_decision["citations"]]
                self.workspace.write_trace_event(
                    "iterative_timeline_temporal_decision",
                    {
                        "round": round_number,
                        "answer": answer,
                        "citations": citations,
                        "matched_events": list(timeline_decision["matched_events"]),
                    },
                )
                self._write_final_trace(
                    round_number=round_number,
                    answer=answer,
                    citations=citations,
                    source="timeline_temporal_order",
                )
                return IterativeRunResult(
                    question=question,
                    video_path=video_path,
                    answer=answer,
                    status="final",
                    citations=citations,
                    confidence=float(timeline_decision["confidence"]),
                    rounds=[
                        *rounds,
                        IterativeRound(
                            round_number=round_number,
                            status="final",
                            planner_text="",
                            rationale=str(timeline_decision["rationale"]),
                            program=program,
                            observation_ids=round_observation_ids,
                        ),
                    ],
                )

            table = self._answer_evidence_table(question)
            answer_result = AnswerAgent(self.backend).run(
                question=question,
                evidence_text=self._read_ledger(),
                evidence_table=table,
            )
            last_answer_result = answer_result
            if answer_result.status == "final" and answer_result.candidate_option_relations:
                self.workspace.annotate_candidate_option_relations(
                    observation_ids=answer_result.citations,
                    relations=answer_result.candidate_option_relations,
                    assigned_by="answer_agent",
                )
                table = self._answer_evidence_table(question)
            selected_option = _answer_option_letter(answer_result.answer)
            gate_reason = _hard_skill_gate_reason(
                workspace=self.workspace,
                skill_name=skill.name,
                question=question,
                table=table,
                selected_option=selected_option,
                citations=answer_result.citations,
            )
            if answer_result.status == "final" and not gate_reason:
                self.workspace.write_trace_event(
                    "iterative_answer_agent",
                    {
                        "round": round_number,
                        "source": "hard_skill_runtime",
                        "status": answer_result.status,
                        "answer": answer_result.answer,
                        "citations": list(answer_result.citations),
                        "missing_evidence": list(answer_result.missing_evidence),
                    },
                )
                self._write_final_trace(
                    round_number=round_number,
                    answer=answer_result.answer,
                    citations=answer_result.citations,
                    source="hard_skill_runtime",
                )
                return IterativeRunResult(
                    question=question,
                    video_path=video_path,
                    answer=answer_result.answer,
                    status="final",
                    citations=list(answer_result.citations),
                    confidence=answer_result.confidence,
                    rounds=[
                        *rounds,
                        IterativeRound(
                            round_number=round_number,
                            status="final",
                            planner_text=answer_result.raw_text,
                            rationale=answer_result.rationale,
                            program=program,
                            observation_ids=round_observation_ids,
                        ),
                    ],
                )

            last_failure_tag = gate_reason or "answer_agent_need_more_evidence"
            self.workspace.write_trace_event(
                "iterative_answer_agent",
                {
                    "round": round_number,
                    "source": "hard_skill_runtime",
                    "status": "need_more_evidence",
                    "answer": answer_result.answer,
                    "citations": list(answer_result.citations),
                    "missing_evidence": list(answer_result.missing_evidence),
                    "failure_tag": last_failure_tag,
                },
            )
            self.workspace.write_reflection_memory(
                route=route,
                failure_tag=last_failure_tag,
                rule=_reflection_rule_for_failure(last_failure_tag),
            )
            if round_number >= self.budget.max_rounds:
                low_confidence_result = self._try_low_confidence_final(
                    answer_result=answer_result,
                    question=question,
                    video_path=video_path,
                    rounds=rounds,
                    round_number=round_number,
                    source="hard_skill_budget_exhausted",
                    program=program,
                    observation_ids=round_observation_ids,
                )
                if low_confidence_result is not None:
                    return low_confidence_result
            rounds.append(
                IterativeRound(
                    round_number=round_number,
                    status="need_more_evidence",
                    planner_text=answer_result.raw_text,
                    rationale=answer_result.rationale or last_failure_tag,
                    program=program,
                    observation_ids=round_observation_ids,
                )
            )

        if not all_observation_ids:
            return None

        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer="need_more_evidence",
            status="need_more_evidence",
            citations=list(last_answer_result.citations) if last_answer_result is not None else [],
            confidence=0.0,
            rounds=rounds,
        )

    def _try_timeline_ordering_route(
        self,
        *,
        question: str,
        video_path: str,
        target_facts: Sequence[str],
    ) -> IterativeRunResult | None:
        segments = _timeline_caption_segments(self.scene_index.segments)
        if not segments:
            return None

        caption_program = [
            {
                "tool": "caption_segment",
                "args": {
                    "video_path": video_path,
                    "segment_id": segment.segment_id,
                    "start_sec": float(segment.start_sec),
                    "end_sec": float(segment.end_sec),
                    "question": (
                        "Openly describe this segment's actual visible artworks, objects, people, scene changes, "
                        "onscreen text, and narrated events in presentation order. Include timestamps if possible. "
                        "Focus on concrete observations rather than conclusions."
                    ),
                },
                "assign": f"timeline_caption_{index}",
            }
            for index, segment in enumerate(segments, start=1)
        ]
        interpreter = ProgramInterpreter(registry=self.registry, workspace=self.workspace)
        caption_result = interpreter.run(caption_program)
        caption_rows = _caption_rows_for_program(
            workspace=self.workspace,
            program=caption_program,
            observation_ids=caption_result.observation_ids,
        )

        missing_entities: list[str] = []
        matched_reads: list[dict[str, Any]] = []
        for entity in target_facts:
            match = _best_caption_row_for_entity(entity, caption_rows)
            if match is None:
                missing_entities.append(entity)
                continue
            matched_reads.append(
                {
                    "entity": entity,
                    "segment_id": str(match["segment_id"]),
                    "start_sec": float(match["start_sec"]),
                    "end_sec": float(match["end_sec"]),
                }
            )
        matched_reads.sort(key=lambda item: (float(item["start_sec"]), str(item["entity"])))
        vision_program = [
            {
                "tool": "vision_read",
                "args": {
                    "video_path": video_path,
                    "segment_id": str(match["segment_id"]),
                    "start_sec": float(match["start_sec"]),
                    "end_sec": float(match["end_sec"]),
                    "ask_for": f"At what timestamp (precise, in seconds) does '{match['entity']}' first appear?",
                    "event_label": str(match["entity"]),
                },
                "assign": f"timeline_fact_{index}",
            }
            for index, match in enumerate(matched_reads, start=1)
        ]
        vision_result = interpreter.run(vision_program) if vision_program else type(caption_result)([])
        program = [*caption_program, *vision_program]
        observation_ids = [
            *[str(observation_id) for observation_id in caption_result.observation_ids],
            *[str(observation_id) for observation_id in vision_result.observation_ids],
        ]
        if missing_entities:
            answer = "need_more_evidence: missing timestamp for " + ", ".join(missing_entities)
            self.workspace.write_trace_event(
                "timeline_ordering_missing_entity",
                {"missing_entities": missing_entities, "target_facts": list(target_facts)},
            )
            return IterativeRunResult(
                question=question,
                video_path=video_path,
                answer=answer,
                status="need_more_evidence",
                citations=[str(observation_id) for observation_id in vision_result.observation_ids],
                confidence=0.0,
                rounds=[
                    IterativeRound(
                        round_number=1,
                        status="need_more_evidence",
                        planner_text="",
                        rationale=answer,
                        program=program,
                        observation_ids=observation_ids,
                    )
                ],
            )

        timeline_decision = _timeline_temporal_decision(
            question=question,
            timeline=self.workspace.read_timeline_sorted(),
        )
        if timeline_decision is not None:
            answer = str(timeline_decision["answer"])
            citations = [str(obs_id) for obs_id in timeline_decision["citations"]]
            self.workspace.write_trace_event(
                "iterative_timeline_temporal_decision",
                {
                    "round": 1,
                    "answer": answer,
                    "citations": citations,
                    "matched_events": list(timeline_decision["matched_events"]),
                    "source": "timeline_ordering",
                },
            )
            self._write_final_trace(
                round_number=1,
                answer=answer,
                citations=citations,
                source="timeline_ordering",
            )
            return IterativeRunResult(
                question=question,
                video_path=video_path,
                answer=answer,
                status="final",
                citations=citations,
                confidence=float(timeline_decision["confidence"]),
                rounds=[
                    IterativeRound(
                        round_number=1,
                        status="final",
                        planner_text="",
                        rationale=str(timeline_decision["rationale"]),
                        program=program,
                        observation_ids=observation_ids,
                    )
                ],
            )

        missing_confirmed = _missing_confirmed_timeline_entities(
            target_facts=target_facts,
            timeline=self.workspace.read_timeline_sorted(),
        )
        detail = ", ".join(missing_confirmed) if missing_confirmed else "ambiguous confirmed order"
        answer = "need_more_evidence: " + detail
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=answer,
            status="need_more_evidence",
            citations=[str(observation_id) for observation_id in vision_result.observation_ids],
            confidence=0.0,
            rounds=[
                IterativeRound(
                    round_number=1,
                    status="need_more_evidence",
                    planner_text="",
                    rationale=answer,
                    program=program,
                    observation_ids=observation_ids,
                )
            ],
        )

    def _run_hard_skill_followup_target(
        self,
        *,
        interpreter: ProgramInterpreter,
        target: FollowupTarget,
        video_path: str,
        assign_suffix: int,
    ) -> dict[str, Any]:
        program: list[dict[str, Any]] = []
        observation_ids: list[str] = []
        ground_program = [
            {
                "tool": "ground_question",
                "args": {"query": target.query, "top_k": 3},
                "assign": f"ground_{assign_suffix}",
            }
        ]
        ground_result = interpreter.run(ground_program)
        program.extend(ground_program)
        observation_ids.extend(str(observation_id) for observation_id in ground_result.observation_ids)
        if not ground_result.observation_ids:
            return {"program": program, "observation_ids": observation_ids}

        candidates = self.workspace.grounding_candidates(str(ground_result.observation_ids[-1]), max_candidates=1)
        if not candidates:
            return {"program": program, "observation_ids": observation_ids}

        candidate = candidates[0]
        vision_args: dict[str, Any] = {
            "video_path": video_path,
            "segment_id": str(candidate["segment_id"]),
            "start_sec": float(candidate["start_sec"]),
            "end_sec": float(candidate["end_sec"]),
            "ask_for": target.query,
            "event_label": target.event_label or target.query,
        }
        if target.mutex_group_id and self._tool_accepts_argument("vision_read", "mutex_group_id"):
            vision_args["mutex_group_id"] = target.mutex_group_id
        vision_program = [{"tool": "vision_read", "args": vision_args, "assign": f"fact_{assign_suffix}"}]
        vision_result = interpreter.run(vision_program)
        program.extend(vision_program)
        observation_ids.extend(str(observation_id) for observation_id in vision_result.observation_ids)
        return {"program": program, "observation_ids": observation_ids}


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
    normalization_notes: Sequence[Any] = (),
    hypothesis_text: str = "",
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
        normalization_notes=normalization_notes,
        hypothesis_text=hypothesis_text,
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


def _planner_selected_skill(action: Mapping[str, Any]) -> tuple[SkillSpec | None, dict[str, str]]:
    requested = str(action.get("skill", "") or "").strip()
    if not requested:
        return None, {"status": "missing", "requested_skill": ""}
    name = requested.split("@", 1)[0].strip()
    registry = builtin_skill_registry()
    try:
        return registry.get(name), {"status": "selected", "requested_skill": requested}
    except KeyError:
        return None, {"status": "invalid", "requested_skill": requested}


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


def _tool_class_counts_for_rounds(rounds: Sequence[IterativeRound]) -> dict[str, int]:
    counts = {"cheap": 0, "expensive": 0, "verifier": 0}
    for round_item in rounds:
        for step in round_item.program:
            tool_class = _TOOL_CLASSES.get(str(step.get("tool", "")))
            if tool_class in counts:
                counts[tool_class] += 1
    return counts


def _followup_route_for_skill(skill_name: str) -> FollowupRoute:
    if _is_timeline_skill(skill_name):
        return "temporal_order"
    if skill_name in {"gist_qa", "main_idea"}:
        return "gist_global"
    return "needle_local"


def _is_timeline_skill(skill_name: str) -> bool:
    return str(skill_name) in {"timeline_ordering", "temporal_ordering"}


def _next_followup_chunk(*, scheduler: FollowupScheduler, chunk_size: int) -> list[FollowupTarget]:
    targets: list[FollowupTarget] = []
    for _ in range(max(1, chunk_size)):
        target = scheduler.next()
        if target is None:
            break
        try:
            scheduler.queue.remove(target)
        except ValueError:
            pass
        targets.append(target)
    return targets


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


def _program_has_visual_evidence_tool(program: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(step.get("tool", "")) in _SEGMENT_MEDIA_TOOLS or str(step.get("tool", "")) in _GLOBAL_VIEW_TOOLS for step in program)


def _all_scene_segments_inspected(scene_index: SceneIndex, inspected_segment_ids: set[str]) -> bool:
    segment_ids = {str(segment.segment_id) for segment in scene_index.segments if str(segment.segment_id)}
    return bool(segment_ids) and segment_ids.issubset({str(item) for item in inspected_segment_ids})


def _evidence_status_has_strong_option_support(summary: Mapping[str, Any]) -> bool:
    option_status = summary.get("option_status", {})
    if not isinstance(option_status, Mapping):
        return False
    for status in option_status.values():
        if not isinstance(status, Mapping):
            continue
        try:
            if int(status.get("strong_evidence_count", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _local_fact_question(*, question: str, planner_skill: SkillSpec | None) -> str:
    semantic = _semantic_question_text(question).strip() or question
    if planner_skill is None:
        return semantic
    fact_only_instruction = (
        "Do not choose an option."
        if extract_candidate_options(question)
        else "Report facts only."
    )
    if planner_skill.name == "timeline_ordering":
        return (
            "Openly describe this segment's actual visible artworks, objects, people, scene changes, onscreen text, "
            "and narrated events in presentation order. Include timestamps if possible. "
            "Focus on concrete observations rather than conclusions."
        )
    if planner_skill.name == "mutex_fact_qa":
        return (
            "Describe only relevant visible or narrated facts, including entities, attributes, background, "
            "class/status, life-stage changes, locations, and temporal order when visible or narrated. "
            f"{fact_only_instruction} Question: "
            + semantic
        )
    if planner_skill.name == "main_idea":
        return (
            "Describe localized main-idea evidence in this segment. Focus on entities, events, and narrative stage. "
            f"{fact_only_instruction} Question: "
            + semantic
        )
    return semantic


def _blocked_final_reason(
    *,
    question: str,
    has_inspect_with_candidate_options: bool,
    workspace: EvidenceWorkspace,
    answer: str,
    citations: Sequence[str],
) -> str:
    unsatisfied_slots = workspace.unsatisfied_hypothesis_slots()
    if unsatisfied_slots:
        return "hypothesis_slots_unsatisfied: " + ", ".join(unsatisfied_slots[:5])
    if (
        extract_candidate_options(question)
        and not has_inspect_with_candidate_options
        and not workspace.has_non_navigation_visual_citation(citations)
        and not _indexed_transcript_supports_answer(
            workspace=workspace,
            question=question,
            answer=answer,
        )
    ):
        return "mcq_final_requires_local_visual_read"
    if not workspace.has_non_navigation_visual_citation(citations):
        return "final_requires_non_navigation_visual_evidence"
    return ""


def _blocked_planner_final_reason(
    *,
    question: str,
    has_inspect_with_candidate_options: bool,
    workspace: EvidenceWorkspace,
    answer: str,
    citations: Sequence[str],
    planner_skill: SkillSpec | None = None,
) -> str:
    base_reason = _blocked_final_reason(
        question=question,
        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
        workspace=workspace,
        answer=answer,
        citations=citations,
    )
    if base_reason:
        return base_reason
    options = extract_candidate_options(question)
    if not options:
        return ""
    if planner_skill is None and classify_question_route(question) != "gist_global":
        return ""

    selected_option = _answer_option_letter(answer)
    table = workspace.evidence_table_v2(
        question=question,
        options=options,
        include_legacy_worker_votes=True,
    )
    return _hard_skill_gate_reason(
        workspace=workspace,
        skill_name=planner_skill.name if planner_skill is not None else "main_idea",
        question=question,
        table=table,
        selected_option=selected_option,
        citations=citations,
    )


def _has_minimum_non_navigation_visual_citations(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    citations: Sequence[str],
    minimum: int,
) -> bool:
    cited = {str(citation) for citation in citations if str(citation)}
    if not cited:
        return False
    matched_observation_ids: set[str] = set()
    table = workspace.evidence_table_v2(
        question=question,
        options=extract_candidate_options(question),
        include_legacy_worker_votes=True,
    )
    for row in table.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        row_ids = {str(row.get("obs_id", "")), str(row.get("evidence_id", ""))}
        if not (row_ids & cited):
            continue
        tool_name = str(row.get("tool", ""))
        if tool_name in workspace.ANSWER_EVIDENCE_TOOLS and tool_name not in workspace.NAVIGATION_TOOLS:
            canonical_id = str(row.get("obs_id", "") or row.get("evidence_id", ""))
            if canonical_id:
                matched_observation_ids.add(canonical_id)
            if len(matched_observation_ids) >= minimum:
                return True
    return False


def _main_idea_indexed_coverage_supports_answer(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
) -> bool:
    return _indexed_transcript_supports_answer(workspace=workspace, question=question, answer=answer)


def _indexed_transcript_supports_answer(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
) -> bool:
    options = extract_candidate_options(question)
    if not options:
        return False
    selected_option = _answer_option_letter(answer)
    if not selected_option:
        return False
    table = workspace.evidence_table_v2(question=question, options=options, include_legacy_worker_votes=True)
    support = selected_option_has_structured_support(table, selected_option=selected_option)
    if not support.passed:
        return False
    for row in table.get("groups", {}).get(selected_option, []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("tool", "")) == "timeline_asr_summary" and str(row.get("grounding_quality", "")) == "indexed_transcript":
            return True
    return False


def _scene_segment_provenance(segment: VideoSegment) -> dict[str, Any]:
    citation_provenance = dict(getattr(segment, "citation_provenance", {}) or {})
    if getattr(segment, "asr_summary_source", ""):
        citation_provenance.setdefault("asr_summary_source", str(segment.asr_summary_source))
    if getattr(segment, "visual_caption_source", ""):
        citation_provenance.setdefault("visual_caption_source", str(segment.visual_caption_source))
    return {
        "source_segment_id": str(getattr(segment, "source_segment_id", "") or segment.segment_id),
        "raw_asr_ref": getattr(segment, "raw_asr_ref", "") or "",
        "visual_caption_source": str(getattr(segment, "visual_caption_source", "") or ""),
        "citation_provenance": citation_provenance,
    }


def _citation_provenance_from_evidence_row(
    *,
    citation: str,
    row: Mapping[str, Any],
    scene_index: SceneIndex,
) -> dict[str, Any]:
    segment_id = str(row.get("segment_id", ""))
    segment: VideoSegment | None = None
    if segment_id:
        try:
            segment = scene_index.get(segment_id)
        except ValueError:
            segment = None

    segment_provenance = _scene_segment_provenance(segment) if segment is not None else {}
    citation_provenance = row.get("citation_provenance") or segment_provenance.get("citation_provenance") or {}
    if not isinstance(citation_provenance, Mapping):
        citation_provenance = {}

    payload: dict[str, Any] = {
        "citation": citation,
        "evidence_id": str(row.get("evidence_id", "")),
        "obs_id": str(row.get("obs_id", "")),
        "tool": str(row.get("tool", "")),
        "segment_id": segment_id,
    }
    time_range = row.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)):
        payload["time_range"] = list(time_range)
    source_segment_id = str(row.get("source_segment_id") or segment_provenance.get("source_segment_id") or "")
    if source_segment_id:
        payload["source_segment_id"] = source_segment_id
    raw_asr_ref = row.get("raw_asr_ref") or segment_provenance.get("raw_asr_ref", "")
    if raw_asr_ref:
        payload["raw_asr_ref"] = raw_asr_ref
    visual_caption_source = str(
        row.get("visual_caption_source") or segment_provenance.get("visual_caption_source") or ""
    )
    if visual_caption_source:
        payload["visual_caption_source"] = visual_caption_source
    if citation_provenance:
        payload["citation_provenance"] = dict(citation_provenance)
    return payload


_SCENE_RISE_TERMS = {
    "rise",
    "rises",
    "rose",
    "rising",
    "formation",
    "formed",
    "created",
    "creation",
    "growth",
    "prosperity",
    "industrial",
}
_SCENE_STABILITY_TERMS = {
    "stable",
    "stability",
    "governance",
    "government",
    "rights",
    "internal",
    "prosperity",
    "population",
}
_SCENE_FALL_TERMS = {
    "fall",
    "fell",
    "falls",
    "fallen",
    "collapse",
    "collapsed",
    "collapsing",
    "decline",
    "declining",
    "dissolution",
    "divided",
    "partition",
    "war",
    "independence",
}


def _scene_coverage_evidence_rows(
    *,
    scene_index: SceneIndex,
    question: str,
    options: Sequence[str],
) -> list[dict[str, Any]]:
    option_stage_targets = {
        _option_letter(option_text, index=index): _scene_option_stage_targets(option_text)
        for index, option_text in enumerate(options)
    }
    full_arc_options = {
        option
        for option, stages in option_stage_targets.items()
        if "rise" in stages and "fall" in stages
    }
    if not full_arc_options:
        return []

    rows: list[dict[str, Any]] = []
    seen_stage_by_option: dict[str, set[str]] = {option: set() for option in full_arc_options}
    for segment in scene_index.segments:
        text = _scene_segment_text(segment)
        if not text:
            continue
        stages = _scene_text_stages(text)
        if not stages:
            continue
        for option in sorted(full_arc_options):
            missing = stages.difference(seen_stage_by_option[option])
            if not missing:
                continue
            seen_stage_by_option[option].update(stages)
            rows.append(
                {
                    "evidence_id": f"ev_scene_coverage_{segment.segment_id}",
                    "obs_id": f"scene_coverage_{segment.segment_id}",
                    "tool": "timeline_asr_summary",
                    "segment_id": segment.segment_id,
                    "time_range": [float(segment.start_sec), float(segment.end_sec)],
                    "supported_option": option,
                    "event_label": "scene index coverage",
                    "claim": _scene_coverage_claim(segment_id=segment.segment_id, stages=stages, text=text),
                    "confidence": 0.84,
                    "grounding_quality": "indexed_transcript",
                    "candidate_option_relations": [
                        {
                            "option": option,
                            "relation": "support",
                            "strength": 0.84,
                            "assigned_by": "scene_index_coverage",
                        }
                    ],
                    "confidence_signal": "indexed transcript coverage",
                    "limitations": "Derived from indexed scene captions/subtitles; verify fine visual details locally when needed.",
                    "artifact": scene_index.video_path,
                    **_scene_segment_provenance(segment),
                }
            )
    return [row for row in rows if _scene_row_is_useful_for_full_arc(row, seen_stage_by_option)]


def _scene_order_evidence_rows(
    *,
    scene_index: SceneIndex,
    question: str,
    options: Sequence[str],
) -> list[dict[str, Any]]:
    if classify_question_route(question) != "temporal_order" and not _question_has_sequence_options(options):
        return []
    if not scene_index.segments:
        return []

    option_events = {
        _option_letter(option_text, index=index): _scene_option_event_sequence(option_text)
        for index, option_text in enumerate(options)
    }
    option_events = {
        option: events
        for option, events in option_events.items()
        if len(events) >= 2
    }
    if not option_events:
        return []

    segment_texts = [
        (segment, _normalize_scene_event_text(_scene_segment_text(segment)))
        for segment in scene_index.segments
        if _scene_segment_text(segment)
    ]
    matched_options: dict[str, list[tuple[VideoSegment, str]]] = {}
    for option, events in option_events.items():
        matches = _match_scene_events_in_order(events=events, segment_texts=segment_texts)
        if matches is not None:
            matched_options[option] = matches
    if not matched_options:
        return []
    max_events = max(len(matches) for matches in matched_options.values())
    matched_options = {
        option: matches
        for option, matches in matched_options.items()
        if len(matches) == max_events
    }
    if len(matched_options) != 1:
        return []

    option, matches = next(iter(matched_options.items()))
    rows = []
    for index, (segment, event) in enumerate(matches, start=1):
        rows.append(
            {
                "evidence_id": f"ev_scene_order_{segment.segment_id}_{index:02d}",
                "obs_id": f"scene_order_{segment.segment_id}",
                "tool": "timeline_asr_summary",
                "segment_id": segment.segment_id,
                "time_range": [float(segment.start_sec), float(segment.end_sec)],
                "supported_option": option,
                "event_label": event,
                "observed_at_sec": float(segment.start_sec),
                "claim": _scene_order_claim(
                    segment_id=segment.segment_id,
                    event=event,
                    position=index,
                    text=_scene_segment_text(segment),
                ),
                "confidence": 0.86,
                "grounding_quality": "indexed_transcript",
                "candidate_option_relations": [
                    {
                        "option": option,
                        "relation": "support",
                        "strength": 0.86,
                        "assigned_by": "scene_index_order",
                    }
                ],
                "confidence_signal": "indexed transcript order",
                "limitations": "Derived from indexed scene captions/subtitles; verify fine visual details locally when needed.",
                "artifact": scene_index.video_path,
                **_scene_segment_provenance(segment),
            }
        )
    return rows


def _question_has_sequence_options(options: Sequence[str]) -> bool:
    return any(len(_scene_option_event_sequence(option)) >= 2 for option in options)


def _scene_option_stage_targets(option_text: str) -> set[str]:
    lowered = str(option_text).lower()
    stages = set()
    if _scene_contains_any(lowered, _SCENE_RISE_TERMS):
        stages.add("rise")
    if _scene_contains_any(lowered, _SCENE_STABILITY_TERMS):
        stages.add("stability")
    if _scene_contains_any(lowered, _SCENE_FALL_TERMS):
        stages.add("fall")
    return stages


def _scene_segment_text(segment: VideoSegment) -> str:
    return " ".join(
        part
        for part in [
            str(getattr(segment, "asr_summary", "")),
            str(getattr(segment, "visual_caption", "")),
            str(getattr(segment, "low_fps_caption", "")),
            str(getattr(segment, "keyframe_path", "")) if not str(getattr(segment, "low_fps_caption", "")) else "",
        ]
        if part
    ).strip()


def _scene_text_stages(text: str) -> set[str]:
    lowered = str(text).lower()
    stages = set()
    if _scene_contains_any(lowered, _SCENE_RISE_TERMS):
        stages.add("rise")
    if _scene_contains_any(lowered, _SCENE_STABILITY_TERMS):
        stages.add("stability")
    if _scene_contains_any(lowered, _SCENE_FALL_TERMS):
        stages.add("fall")
    return stages


def _scene_contains_any(text: str, terms: set[str]) -> bool:
    tokens = set(re.findall(r"[A-Za-z0-9]+", str(text).lower()))
    return bool(tokens.intersection(terms))


def _scene_coverage_claim(*, segment_id: str, stages: set[str], text: str) -> str:
    stage_text = ", ".join(sorted(stages))
    compact_text = re.sub(r"\s+", " ", text).strip()
    compact_text = compact_text[:420] + ("..." if len(compact_text) > 420 else "")
    return f"{segment_id} indexed transcript/caption covers {stage_text}: {compact_text}"


def _scene_row_is_useful_for_full_arc(row: Mapping[str, Any], seen_stage_by_option: Mapping[str, set[str]]) -> bool:
    option = str(row.get("supported_option", ""))
    stages = seen_stage_by_option.get(option, set())
    return "rise" in stages and "fall" in stages


def _scene_option_event_sequence(option_text: str) -> list[str]:
    text = _strip_option_letter(option_text)
    quoted = [part.strip() for part in re.findall(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’', text) for part in part if part.strip()]
    if len(quoted) >= 2:
        return quoted
    normalized = (
        text.replace("->", ",")
        .replace(">", ",")
        .replace(" and ", ",")
        .replace(" then ", ",")
    )
    events = [part.strip(" .:-\"'") for part in re.split(r"[,;/]+", normalized) if part.strip(" .:-\"'")]
    return [event for event in events if len(_scene_event_tokens(event)) >= 2]


def _match_scene_events_in_order(
    *,
    events: Sequence[str],
    segment_texts: Sequence[tuple[VideoSegment, str]],
) -> list[tuple[VideoSegment, str]] | None:
    cursor = 0
    matches: list[tuple[VideoSegment, str]] = []
    for event in events:
        event_tokens = _scene_event_tokens(event)
        if not event_tokens:
            return None
        matched = None
        for index in range(cursor, len(segment_texts)):
            segment, normalized_text = segment_texts[index]
            if event_tokens.issubset(set(normalized_text.split())):
                matched = (index, segment)
                break
        if matched is None:
            return None
        cursor = matched[0] + 1
        matches.append((matched[1], event))
    return matches


def _scene_order_claim(*, segment_id: str, event: str, position: int, text: str) -> str:
    compact_text = re.sub(r"\s+", " ", text).strip()
    compact_text = compact_text[:360] + ("..." if len(compact_text) > 360 else "")
    return f"{segment_id} indexed transcript/caption places ordered item {position}: {event}. Context: {compact_text}"


def _strip_option_letter(option_text: str) -> str:
    return re.sub(r"^\s*[A-H](?:[\.)]\s*|\s+)", "", str(option_text), flags=re.IGNORECASE).strip()


def _normalize_scene_event_text(text: str) -> str:
    return " ".join(_scene_event_tokens(text))


def _scene_event_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "by",
        "created",
        "depicted",
        "does",
        "for",
        "four",
        "in",
        "is",
        "of",
        "order",
        "present",
        "presented",
        "scene",
        "shown",
        "single",
        "the",
        "then",
        "to",
        "video",
        "with",
    }
    return {
        _normalize_scene_event_token(token)
        for token in re.findall(r"[A-Za-z0-9]+", str(text).lower())
        if len(token) >= 3 and token not in stopwords
    }


def _normalize_scene_event_token(token: str) -> str:
    if token == "borned":
        return "born"
    return token


def _hard_skill_gate_reason(
    *,
    workspace: EvidenceWorkspace,
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

    if _is_timeline_skill(skill_name):
        temporal = temporal_order_consistent(table, selected_option=selected_option)
        if not temporal.passed:
            return "temporal_order_requires_confirmed_event_timestamps"

    grounding_reason = grounding_quality_floor(
        workspace.mapped_evidence_records(observation_ids=citations, selected_option=selected_option),
        workspace=workspace,
        require_visual=skill_name not in {"gist_qa", "main_idea"},
    )
    if grounding_reason and not _cited_table_rows_satisfy_grounding_floor(
        table=table,
        selected_option=selected_option,
        citations=citations,
        require_visual=skill_name not in {"gist_qa", "main_idea"},
    ):
        return "grounding_quality_floor"
    return ""


def _cited_table_rows_satisfy_grounding_floor(
    *,
    table: Mapping[str, Any],
    selected_option: str | None,
    citations: Sequence[str],
    require_visual: bool,
) -> bool:
    cited = {str(citation) for citation in citations if str(citation)}
    if not cited:
        return False
    for row in table.get("groups", {}).get(selected_option, []):
        if not isinstance(row, Mapping):
            continue
        row_ids = {str(row.get("obs_id", "")), str(row.get("evidence_id", ""))}
        if not cited.intersection(row_ids):
            continue
        quality = str(row.get("grounding_quality", ""))
        if require_visual and quality not in {"visually_confirmed", "indexed_transcript"}:
            continue
        if quality in {"weak", "inferred", "external_knowledge", "global_sparse"}:
            continue
        return True
    return False


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
        "grounding_quality_floor": "collect at least one visually_confirmed mapped evidence chain before finalizing",
    }
    return rules.get(str(failure_tag), "request targeted evidence before finalizing")


def _skill_target_facts(*, question: str, skill_name: str) -> list[str]:
    return [spec.fact for spec in _skill_target_fact_specs(question=question, skill_name=skill_name)]


def _skill_target_fact_specs(*, question: str, skill_name: str) -> list[SkillTargetFact]:
    options = extract_candidate_options(question)
    if _is_timeline_skill(skill_name):
        events = _temporal_events_from_question(question)
        if events:
            return [SkillTargetFact(fact=event) for event in events]
    option_targets = _option_fact_target_specs(options)
    if option_targets:
        return option_targets
    semantic = _semantic_question_text(question)
    return [SkillTargetFact(fact=semantic)] if semantic else []


def _semantic_question_text(question: str) -> str:
    match = re.search(r"\bQuestion:\s*(.*?)(?:\n\s*Options:|\Z)", question, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return " ".join(match.group(1).split()).strip()

    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[A-H][.)]\s+\S+", stripped):
            continue
        lowered = stripped.lower()
        if lowered.startswith("videomme multiple-choice question"):
            continue
        if "answer with" in stripped.lower() and "option letter" in stripped.lower():
            continue
        if "do not use outside knowledge" in lowered:
            continue
        lines.append(stripped)
    return " ".join(lines).strip() or question.strip()


def _temporal_events_from_question(question: str, *, max_events: int = 4) -> list[str]:
    options = extract_candidate_options(question)
    quoted_targets = _quoted_option_targets(options, max_targets=max_events)
    if quoted_targets:
        return quoted_targets

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


_TARGET_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "with",
    "without",
    "for",
    "from",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "then",
    "before",
    "after",
    "first",
    "last",
    "option",
    "answer",
}


def _option_fact_targets(options: Sequence[str], *, max_targets: int = 6) -> list[str]:
    return [spec.fact for spec in _option_fact_target_specs(options, max_targets=max_targets)]


def _option_fact_target_specs(options: Sequence[str], *, max_targets: int = 6) -> list[SkillTargetFact]:
    quoted_targets = _quoted_option_targets(options, max_targets=max_targets)
    if quoted_targets:
        return [
            SkillTargetFact(fact=target, mutex_group_id="option_fact_mutex")
            for target in quoted_targets
        ]

    targets: list[SkillTargetFact] = []
    seen: set[str] = set()
    for option in options:
        text = _strip_option_prefix(str(option))
        for chunk in _split_option_fact_text(text):
            target = _clean_target_fact(chunk)
            if not _informative_target_fact(target):
                continue
            key = _target_fact_key(target)
            if not key or key in seen:
                continue
            seen.add(key)
            targets.append(SkillTargetFact(fact=target, mutex_group_id="option_fact_mutex"))
            if len(targets) >= max_targets:
                return targets
    return targets


def _quoted_option_targets(options: Sequence[str], *, max_targets: int) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for option in options:
        for match in re.finditer(r"[\"“]([^\"”]+)[\"”]", str(option)):
            target = _clean_target_fact(match.group(1))
            if not _informative_target_fact(target):
                continue
            key = _target_fact_key(target)
            if not key or key in seen:
                continue
            seen.add(key)
            targets.append(target)
            if len(targets) >= max_targets:
                return targets
    return targets


def _strip_option_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-H][.)]\s*", "", text).strip()


def _split_option_fact_text(text: str) -> list[str]:
    primary_parts = re.split(r"\b(?:and then|then|before|after)\b|[,;]|->|/|\|", text, flags=re.IGNORECASE)
    parts: list[str] = []
    action_splitter = re.compile(
        r"\band\s+(?=(?:lived|entered|went|moved|became|was|were|is|are|appeared|appears|shown|shows|born|borned)\b)",
        flags=re.IGNORECASE,
    )
    for part in primary_parts:
        parts.extend(action_splitter.split(part))
    return parts


def _clean_target_fact(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip(" .:-\"'“”")
    cleaned = re.sub(r"\b(first|last|earlier|later|right|immediately)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-\"'“”")
    return cleaned


def _informative_target_fact(text: str) -> bool:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in _TARGET_STOPWORDS]
    return len(tokens) >= 1 and any(len(token) >= 3 for token in tokens)


def _target_fact_key(text: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in _TARGET_STOPWORDS]
    return " ".join(tokens)


def _timeline_caption_segments(segments: Sequence[VideoSegment], *, max_segments: int = 8) -> list[VideoSegment]:
    if len(segments) <= max_segments:
        return list(segments)
    if max_segments <= 1:
        return [segments[0]]
    selected: list[VideoSegment] = []
    seen: set[int] = set()
    for index in range(max_segments):
        source_index = round(index * (len(segments) - 1) / (max_segments - 1))
        if source_index in seen:
            continue
        seen.add(source_index)
        selected.append(segments[source_index])
    return selected


def _caption_rows_for_program(
    *,
    workspace: EvidenceWorkspace,
    program: Sequence[Mapping[str, Any]],
    observation_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, observation_id in zip(program, observation_ids):
        args = step.get("args", {})
        if not isinstance(args, Mapping):
            continue
        observation = workspace.get_observation(str(observation_id))
        rows.append(
            {
                "obs_id": str(observation_id),
                "claim": observation.claim if observation is not None else "",
                "segment_id": str(args.get("segment_id", "")),
                "start_sec": float(args.get("start_sec", 0.0) or 0.0),
                "end_sec": float(args.get("end_sec", 0.0) or 0.0),
            }
        )
    return rows


def _best_caption_row_for_entity(entity: str, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    entity_tokens = set(_target_fact_key(entity).split())
    if not entity_tokens:
        return None
    best: tuple[float, float, Mapping[str, Any]] | None = None
    for row in rows:
        claim = str(row.get("claim", ""))
        if is_unsupported_claim(claim):
            continue
        row_tokens = set(_target_fact_key(claim).split())
        if not row_tokens:
            continue
        if entity_tokens.issubset(row_tokens):
            score = 1.0
        else:
            overlap = len(entity_tokens.intersection(row_tokens))
            score = overlap / len(entity_tokens)
        if score < 0.5:
            continue
        start_sec = float(row.get("start_sec", 0.0) or 0.0)
        if best is None or score > best[0] or (score == best[0] and start_sec < -best[1]):
            best = (score, -start_sec, row)
    return best[2] if best is not None else None


def _missing_confirmed_timeline_entities(
    *,
    target_facts: Sequence[str],
    timeline: Sequence[Mapping[str, Any]],
) -> list[str]:
    confirmed = _confirmed_timeline_rows(timeline)
    missing = []
    for target in target_facts:
        if _match_timeline_row(target, confirmed, used_obs_ids=set()) is None:
            missing.append(target)
    return missing


def _answer_option_letter(answer: str) -> str | None:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(answer), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _timeline_temporal_decision(
    *,
    question: str,
    timeline: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    options = extract_candidate_options(question)
    confirmed = _confirmed_timeline_rows(timeline)
    if len(options) < 2 or len(confirmed) < 2:
        return None

    candidates = []
    for index, option_text in enumerate(options):
        option_letter = _option_letter(option_text, index=index)
        expected_events = _option_temporal_events(option_text)
        if len(expected_events) < 2:
            continue
        matched = []
        used_obs_ids = set()
        for event in expected_events:
            match = _match_timeline_row(event, confirmed, used_obs_ids=used_obs_ids)
            if match is None:
                break
            used_obs_ids.add(str(match.get("obs_id", "")))
            matched.append(
                {
                    "expected": event,
                    "observed": str(match.get("entity", "")),
                    "start_sec": float(match.get("observed_at_sec", 0.0) or 0.0),
                    "obs_id": str(match.get("obs_id", "")),
                }
            )
        if len(matched) != len(expected_events):
            continue
        times = [float(item["start_sec"]) for item in matched]
        if times == sorted(times):
            candidates.append(
                {
                    "answer": option_letter,
                    "citations": [str(item["obs_id"]) for item in matched],
                    "matched_events": matched,
                    "confidence": min(0.95, 0.75 + 0.05 * len(matched)),
                    "rationale": "confirmed timeline order uniquely matches option " + option_letter,
                }
            )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _confirmed_timeline_rows(timeline: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    confirmed = [
        row
        for row in timeline
        if isinstance(row, Mapping)
        and str(row.get("confidence_signal", "")).strip().lower() == "confirmed"
        and row.get("observed_at_sec") is not None
    ]
    return sorted(
        confirmed,
        key=lambda row: (float(row.get("observed_at_sec", 0.0) or 0.0), str(row.get("obs_id", ""))),
    )


def _option_temporal_events(option_text: str) -> list[str]:
    events = []
    for part in _split_option_fact_text(_strip_option_prefix(option_text)):
        event = _clean_target_fact(part)
        if event:
            events.append(event)
    return events


def _match_timeline_row(
    event: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    used_obs_ids: set[str],
) -> Mapping[str, Any] | None:
    event_tokens = set(_target_fact_key(event).split())
    if not event_tokens:
        return None
    best: tuple[float, Mapping[str, Any]] | None = None
    for row in rows:
        obs_id = str(row.get("obs_id", ""))
        if obs_id in used_obs_ids:
            continue
        row_tokens = set(_target_fact_key(str(row.get("entity", ""))).split())
        if not row_tokens:
            continue
        if event_tokens.issubset(row_tokens) or row_tokens.issubset(event_tokens):
            score = 1.0
        else:
            overlap = len(event_tokens.intersection(row_tokens))
            union = len(event_tokens.union(row_tokens))
            score = overlap / union if union else 0.0
        if score < 0.5:
            continue
        if best is None or score > best[0]:
            best = (score, row)
    return best[1] if best is not None else None


def _option_letter(option_text: str, *, index: int) -> str:
    match = re.match(r"\s*([A-H])(?:[.)]\s*|\s+|$)", str(option_text), flags=re.IGNORECASE)
    return match.group(1).upper() if match else chr(ord("A") + index)


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


def _tool_exploration_question(
    question: str,
    *,
    route_hint: str = "",
    question_context: str = "",
    forbidden_question: str = "",
    option_blind: bool = False,
) -> str:
    if not option_blind:
        return exploration_question(question, route_hint=route_hint)
    lowered_route = str(route_hint or "").lower()
    if "timeline" in lowered_route or "temporal" in lowered_route:
        return (
            "Openly describe this segment's actual visible artworks, objects, people, scene changes, "
            "onscreen text, and narrated events in presentation order. Include timestamps if possible. "
            "Focus on concrete observations rather than conclusions."
        )
    if "main_idea" in lowered_route or "gist" in lowered_route:
        return (
            "Openly describe this segment's actual visual content and narrated topic. Mention concrete "
            "entities, setting, stage of the story, and any visible text. Do not choose or compare options."
        )
    cleaned = " ".join(str(question or "").split()).strip()
    if forbidden_question and _text_has_option_surface(cleaned, raw_question=forbidden_question):
        fallback = " ".join(str(question_context or "").split()).strip()
        if fallback and not _text_has_option_surface(fallback, raw_question=forbidden_question):
            return fallback
        return "Gather factual evidence needed for final matching."
    if extract_candidate_options(question):
        return exploration_question(question, route_hint=route_hint)
    if extract_candidate_options(question_context) and "choose an option" not in cleaned.lower():
        return f"{cleaned} Do not choose an option.".strip()
    return cleaned


def _sanitize_option_blind_feedback(feedback: Sequence[str], *, raw_question: str) -> list[str]:
    sanitized: list[str] = []
    fallback = "Resolve the remaining evidence gap with factual observations."
    for item in feedback:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        if _text_has_option_surface(text, raw_question=raw_question):
            text = fallback
        if text not in sanitized:
            sanitized.append(text)
    return sanitized


def _text_has_option_surface(text: str, *, raw_question: str) -> bool:
    haystack = " ".join(str(text or "").split())
    if not haystack:
        return False
    for label, option_text in _candidate_option_surfaces(raw_question):
        if re.search(rf"\b(?:option|choice|answer)\s*{re.escape(label)}\b", haystack, flags=re.IGNORECASE):
            return True
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}[.)]\s+\S+", haystack, flags=re.IGNORECASE):
            return True
        lowered = haystack.lower()
        for surface in _option_text_surfaces(option_text):
            if surface.lower() in lowered:
                return True
    return False


def _candidate_option_surfaces(raw_question: str) -> list[tuple[str, str]]:
    surfaces: list[tuple[str, str]] = []
    for index, option in enumerate(extract_candidate_options(raw_question)):
        label = _option_letter(str(option), index=index)
        option_text = re.sub(r"^\s*[A-H][.)]\s*", "", str(option), flags=re.IGNORECASE).strip()
        if option_text:
            surfaces.append((label, option_text))
    return surfaces


def _option_text_surfaces(option_text: str) -> list[str]:
    cleaned = " ".join(str(option_text or "").split()).strip()
    trimmed = cleaned.strip(" .")
    surfaces = []
    for surface in (cleaned, trimmed):
        if len(surface) >= 6 and surface not in surfaces:
            surfaces.append(surface)
    return surfaces


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


def _tool_denied_by_skill(*, tool_name: str, active_skill: SkillSpec | None, free_exploration: bool) -> bool:
    return active_skill is not None and not free_exploration and tool_name not in active_skill.allowed_actions


def _skill_name_as_tool_reason(tool_name: str) -> str:
    normalized = str(tool_name).strip()
    if not normalized:
        return ""
    skill_name = normalized.split("@", 1)[0]
    for skill in builtin_skill_registry().list():
        if skill.name == skill_name:
            return skill.name
    return ""


def _record_skill_deny_list_violation(
    *,
    workspace: EvidenceWorkspace,
    notes_out: list[NormalizationNote] | None,
    tool_name: str,
    args: Mapping[str, Any],
    active_skill: SkillSpec | None,
) -> None:
    if active_skill is None:
        return
    allowed_actions = ", ".join(sorted(active_skill.allowed_actions)) or "(none)"
    next_action = (
        f"{tool_name} is not in the active skill ({active_skill.name}) allowed_actions. "
        f"Pick one of: {allowed_actions}."
    )
    workspace.write_trace_event(
        "route_violation",
        {
            "tool": tool_name,
            "error": "tool_not_in_allowed_actions",
            "skill": active_skill.name,
            "next_action": next_action,
        },
    )
    workspace.write_reflection_memory(
        route=active_skill.trigger.route,
        failure_tag="tool_not_in_allowed_actions",
        rule=(
            f"Skill {active_skill.name} only permits: {allowed_actions}. "
            f"{tool_name} is denied. Do not request it again."
        ),
    )
    _append_normalization_note(
        notes_out,
        tool=tool_name,
        reason="tool_not_in_allowed_actions",
        original={"tool": tool_name, "args": dict(args)},
        next_action=next_action,
    )


def _tool_class(tool_name: str) -> str:
    return _TOOL_CLASSES.get(tool_name, "cheap")


def _append_normalization_note(
    notes_out: list[NormalizationNote] | None,
    *,
    tool: str,
    reason: str,
    original: Mapping[str, Any],
    resolved: Mapping[str, Any] | None = None,
    next_action: str = "",
) -> None:
    if notes_out is None:
        return
    notes_out.append(
        NormalizationNote(
            tool=str(tool),
            reason=str(reason),
            original=dict(original),
            resolved=dict(resolved or {}),
            next_action=str(next_action),
        )
    )


def _global_gist_second_pass_offset(duration_sec: float) -> float:
    return max(0.001, min(float(duration_sec) / 256.0, 5.0))


def _global_gist_supported_options(table: Mapping[str, Any]) -> list[str]:
    groups = table.get("groups", {}) if isinstance(table.get("groups", {}), Mapping) else {}
    options = []
    for group_option, rows in groups.items():
        if group_option == "unassigned" or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("tool", "")) != "global_gist" and str(row.get("grounding_quality", "")) != "global_sparse":
                continue
            option = str(row.get("supported_option") or group_option).strip().upper()[:1]
            if option:
                options.append(option)
    return options


def _global_option_coverage_choice(*, question: str, ledger_text: str, min_margin: float = 0.15) -> Mapping[str, Any] | None:
    options = extract_candidate_options(question)
    if not options:
        return None
    ledger_tokens = set(_content_words(ledger_text))
    scored = []
    for index, option_text in enumerate(options):
        option_tokens = set(_content_words(_strip_option_prefix(option_text)))
        if not option_tokens:
            continue
        coverage = len(option_tokens.intersection(ledger_tokens)) / len(option_tokens)
        scored.append(
            {
                "option": _option_letter(option_text, index=index),
                "coverage": coverage,
                "tokens": sorted(option_tokens),
            }
        )
    if not scored:
        return None
    scored.sort(key=lambda item: (-float(item["coverage"]), str(item["option"])))
    top = scored[0]
    second = float(scored[1]["coverage"]) if len(scored) > 1 else 0.0
    margin = float(top["coverage"]) - second
    if margin <= min_margin:
        return None
    return {**top, "margin": margin}


def _content_words(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text).lower())
        if token not in _TARGET_STOPWORDS and len(token) >= 3
    ]


def _route_violation(*, tool_name: str, active_skill: SkillSpec | None, free_exploration: bool) -> str | None:
    if free_exploration or active_skill is None or not active_skill.allowed_actions:
        return None
    if tool_name in active_skill.allowed_actions:
        return None
    return f"action '{tool_name}' not in skill '{active_skill.name}' whitelist"


def _segment_has_index_text(segment: Any) -> bool:
    return bool(getattr(segment, "low_fps_caption", ""))


def _is_video_path_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text in {"", "video_path", "<video_path>", "$video_path", "${video_path}"}


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
