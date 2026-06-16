"""Iterative visual agent for coarse-to-fine video exploration."""

from __future__ import annotations

import json
import inspect
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..backends.base import BackendRequest, VisionLanguageBackend
from ..contracts import ClaimModality, ClaimRelation, OptionSpec, TargetRegistry, TargetSpec
from ..evidence.projection import ProjectionEvidence, ProjectionResult, project_option_support
from ..registry import ToolError, ToolRegistry
from ..task.spec import OptionSpec as ProjectionOptionSpec
from ..task.spec import TaskSpec, build_task_spec
from ..video_index import SceneIndex, VideoSegment
from ..workspace import EvidenceWorkspace
from .answer_agent import AnswerAgent, AnswerAgentResult
from .contracts import OptionEvaluation, VISUAL_EVIDENCE_NFRAMES
from .context_budget import default_context_budget_allocator
from .final_gate import evaluate_final_candidate
from .followup import FollowupBudget, FollowupRoute, FollowupScheduler, FollowupTarget
from .grounding import CompiledGroundingPlan, compile_fallback_plan, compile_grounding_plan, ground_question_with_model
from .open_questions import QuestionContext, build_question_context, exploration_question, rewrite_exploration_question_with_model
from .output_quality import is_unsupported_claim
from .prompt_stack import (
    build_replanning_prompt,
    compose_planner_prompts,
    compose_replanning_prompt_blocks,
    render_prompt_blocks,
)
from .question_policy import (
    classify_narration_subroute,
    classify_question_route,
    extract_candidate_options,
    extract_option_target_atom_map,
    extract_option_target_atoms,
    select_question_playbook,
)
from .runtime.host import ToolRuntimeHost
from .runtime.lifecycle import RunContext
from .runtime.state import AnswerSuggestionState, RoundState, RunState
from .skills.predicates import (
    grounding_quality_floor,
    no_decisive_weak_grounding,
    no_unaddressed_conflict,
    selected_option_has_structured_support,
    temporal_order_consistent,
)
from .skills.specs import PrefinalRepairKind, SkillSpec, builtin_skill_registry, select_skill
from .skill_runtime import (
    _initial_skill_runtime_state as _build_initial_skill_runtime_state,
    _planner_selected_skill as _select_planner_skill,
    _recommended_effective_skill,
    _skill_id as _runtime_skill_id,
    _skill_id_from_name as _runtime_skill_id_from_name,
    update_effective_skill_runtime as _update_effective_skill_runtime,
)


_SEGMENT_MEDIA_TOOLS = {"caption_segment", "qa_segment", "inspect_segment", "vision_read", "verify_segment_anchors"}
_SEGMENT_TEXT_TOOLS = {"locate_targets_in_segment", "read_segment_detail", "read_segment"}
_SEGMENT_ID_TOOLS = _SEGMENT_MEDIA_TOOLS | _SEGMENT_TEXT_TOOLS
_GLOBAL_VIEW_TOOLS = {"global_gist"}
_ONE_SHOT_TOOLS: frozenset[str] = frozenset({"global_gist"})
_TARGET_REF_RE = re.compile(r"^T[1-9]\d*$")
_ZERO_YIELD_EMPTY_RESULT_TOOLS = frozenset(
    {
        "bind_asr_claim",
        "read_segment_detail",
        "asr_cue_detail",
        "transcript_evidence_binder",
    }
)


def _exhausted_one_shot_tools(workspace: Any) -> frozenset[str]:
    return frozenset(
        tool_name for tool_name in _ONE_SHOT_TOOLS if workspace.observation_count(tool_name=tool_name) >= 1
    )


_ANSWER_AGENT_AUTO_FINAL_SOURCES = frozenset(
    {
        "planner_final_takeover",
        "reserved_final",
        "budget_exhausted",
        "prefinal_probe_budget_exhausted",
        "hard_skill_budget_exhausted",
        "repeated_program_guard",
    }
)


def _prefinal_repair_kind(skill: SkillSpec | None) -> PrefinalRepairKind:
    if skill is None:
        return PrefinalRepairKind.NONE
    return skill.behaviors.prefinal_repair


@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = 8
    max_tool_calls_per_round: int = 2
    default_nframes: int = VISUAL_EVIDENCE_NFRAMES
    high_fps_nframes: int = 32
    planner_receives_media: bool = False
    reserve_final_round: bool = True
    answer_probe_rounds_before_final: int = 0
    persist_planner_io: bool = True
    planner_io_max_chars: int = 200_000
    context_budget_tokens: int = 12000
    context_budget_ratios: Mapping[str, float] | None = None
    max_repeated_programs: int = 3
    max_repeated_invalid_programs: int = 3
    hard_skill_runtime: bool = False
    planner_owned_grounding: bool = False
    prompt_role_split_enabled: bool = False
    reflection_memory_max_items: int = 5
    disable_global_gist_route: bool = False
    rewrite_mcq_for_exploration: bool = False


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
    evidence_ids: Sequence[str] = field(default_factory=list)
    confidence: float = 0.0
    rounds: Sequence[IterativeRound] = field(default_factory=list)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "input": {"question": self.question, "video_path": self.video_path},
            "output": {
                "answer": self.answer,
                "status": self.status,
                "citations": list(self.citations),
                "evidence_ids": list(self.evidence_ids),
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
class FailureSignature:
    program_signature: str
    reason: str
    affected_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinalEvidenceBridgeResult:
    citations: list[str]
    evidence_ids: list[str]
    filled_evidence_ids: list[str] = field(default_factory=list)
    ambiguous: list[Mapping[str, Any]] = field(default_factory=list)
    changed: bool = False


@dataclass(frozen=True)
class SkillTargetFact:
    fact: str
    mutex_group_id: str = ""


@dataclass(frozen=True)
class GroundingBootstrapState:
    runtime: CompiledGroundingPlan | None = None
    failure: Mapping[str, Any] | None = None


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
        self._exploration_target_entities: tuple[str, ...] = ()
        self.runtime_host = ToolRuntimeHost(registry=self.registry, workspace=self.workspace)
        self.context_allocator = default_context_budget_allocator(
            total_budget_tokens=self.budget.context_budget_tokens,
            slot_ratios=dict(self.budget.context_budget_ratios) if self.budget.context_budget_ratios else None,
        )

    def run(self, *, question: str, video_path: str) -> IterativeRunResult:
        question_context = build_question_context(question)
        raw_question = question_context.raw_question
        vlm_safe_question = question_context.vlm_safe_question
        self.workspace.ensure_hypothesis(raw_question)
        grounding_bootstrap = self._initialize_planner_owned_grounding(question_context)
        grounding_runtime = grounding_bootstrap.runtime
        if grounding_bootstrap.failure is not None:
            return self._grounding_bootstrap_failed_result(
                question=raw_question,
                video_path=video_path,
                failure=grounding_bootstrap.failure,
            )
        effective_route = self._effective_route(raw_question)
        exploration_question_text = self._question_for_exploration(
            question_context,
            route_hint=effective_route,
        )
        self._seed_target_coverage(raw_question)
        if (
            not self.budget.disable_global_gist_route
            and effective_route == "gist_global"
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
                route=effective_route,
                recommended_skill_id=grounding_runtime.recommended_skill_id if grounding_runtime else "",
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
        run_state = RunState(
            question=raw_question,
            video_path=video_path,
            question_route=effective_route,
            raw_question=raw_question,
            vlm_safe_question=vlm_safe_question,
            effective_route=effective_route,
            inspected_segment_ids=inspected_segment_ids,
            grounding_runtime=grounding_runtime,
            bootstrap_failure=grounding_bootstrap.failure,
        )
        has_inspect_with_candidate_options = any(
            _program_has_inspect_with_candidate_options(round_item.program) for round_item in rounds
        )
        answer_feedback: list[str] = []
        pending_inferences: list[str] = []
        last_round_normalization_notes: list[NormalizationNote] = []
        repeated_program_key = ""
        repeated_program_count = 0
        invalid_failure_counts: dict[FailureSignature, int] = {}
        no_evidence_growth_rounds = 0
        supported_binding_no_growth_rounds = 0
        last_evidence_table_row_count = self.workspace.evidence_table_row_count()
        last_supported_binding_count = self._supported_evidence_binding_count()
        all_segments_answer_attempted = False
        planner_final_verifier_disagreed = False
        planner_final_auto_final_blocked = False
        last_projection_status: dict[str, Any] | None = None
        last_diagnostic_repair_hint = ""
        prefinal_evidence_repair_done = False
        prefinal_evidence_repair_failed = False
        post_repair_action_rounds_remaining = 0
        post_repair_action_start_round = 0
        evidence_repair_candidate_answer = ""
        evidence_repair_failure_reason = ""
        skill_runtime = (
            _build_initial_skill_runtime_state(
                raw_question,
                route=effective_route,
                recommended_skill_id=grounding_runtime.recommended_skill_id if grounding_runtime else "",
            )
            if self.budget.hard_skill_runtime
            else None
        )
        last_selected_skill_id: str | None = _runtime_skill_id(skill_runtime.effective_skill) if skill_runtime else None
        if skill_runtime is not None:
            self.workspace.write_trace_event(
                "skill_recommended",
                {
                    "round": len(rounds) + 1,
                    "recommended_skill": _runtime_skill_id(skill_runtime.recommended_skill),
                    "effective_skill": _runtime_skill_id(skill_runtime.effective_skill),
                    "compatible_skills": list(skill_runtime.compatible_skill_ids),
                    "recommendation_source": skill_runtime.recommendation_source,
                    "reason": "question_policy_recommended",
                },
            )

        for round_number in range(len(rounds) + 1, self.budget.max_rounds + 1):
            ledger_text = self._read_ledger()
            final_round_reserved = self.budget.reserve_final_round and round_number == self.budget.max_rounds
            run_state.final_round_reserved = final_round_reserved
            round_ctx = self._runtime_context(
                question=raw_question,
                video_path=video_path,
                route=effective_route,
                round_number=round_number,
                run_state=run_state,
                round_state=RoundState(round_number=round_number),
                skill_runtime=skill_runtime,
                evidence_policy=skill_runtime.effective_policy if skill_runtime is not None else None,
            )
            evidence_status_summary = self.workspace.evidence_status_summary(
                question=raw_question,
                options=extract_candidate_options(raw_question),
            )
            exhausted_tools = _exhausted_one_shot_tools(self.workspace)
            prompt_pair = compose_planner_prompts(
                prompt_role_split_enabled=self.budget.prompt_role_split_enabled,
                question=exploration_question_text,
                scene_index=self.scene_index,
                ledger_text=ledger_text,
                round_number=round_number,
                budget=self.budget,
                allocator=self.context_allocator,
                inspected_segment_ids=sorted(inspected_segment_ids),
                final_round_reserved=final_round_reserved,
                answer_feedback=answer_feedback,
                pending_inferences=pending_inferences,
                normalization_notes=last_round_normalization_notes,
                hypothesis_text=self.workspace.read_hypothesis_text(),
                reflection_memory=self.workspace.reflection_memory(max_items=self.budget.reflection_memory_max_items),
                evidence_status_summary=evidence_status_summary,
                recent_tool_outputs=self.workspace.recent_tool_outputs(limit=3),
                exhausted_tools=exhausted_tools,
                active_skill=last_selected_skill_id,
                route=effective_route,
                target_hints=self._exploration_target_entities,
                target_ref_descriptions=_registered_target_ref_descriptions(self.workspace),
                projection_status=last_projection_status,
                diagnostic_repair_hint=last_diagnostic_repair_hint,
            )
            planner_prompt = prompt_pair.user_prompt
            system_prompt = prompt_pair.system_prompt
            context_report = prompt_pair.context_report
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
                    system_prompt=system_prompt,
                    media_path=video_path if self.budget.planner_receives_media else None,
                    media_type="video" if self.budget.planner_receives_media else None,
                    max_new_tokens=4096,
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
                system_prompt=system_prompt,
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
                    route=effective_route,
                    failure_tag="planner_json_parse_error",
                    rule="return valid JSON matching the continue/final response contract before using tools",
                )
                recovery_program = self._safe_parse_error_recovery_program(run_state=run_state)
                if recovery_program:
                    action = {
                        "status": "continue",
                        "rationale": "planner_json_parse_error_safe_recovery",
                        "program": recovery_program,
                    }
                    self.workspace.write_trace_event(
                        "planner_parse_error_recovery_selected",
                        {"round": round_number, "program": recovery_program},
                    )
                else:
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
            planner_skill, skill_status = _select_planner_skill(action)
            if skill_status["status"] == "selected":
                self.workspace.write_trace_event(
                    "skill_selected",
                    {
                        "round": round_number,
                        "requested_skill": skill_status["requested_skill"],
                        "resolved_skill": planner_skill.name,
                    },
                )
                self.workspace.write_trace_event(
                    "planner_skill_selection",
                    {
                        "round": round_number,
                        "requested_skill": skill_status["requested_skill"],
                        "resolved_skill": planner_skill.name,
                    },
                )
                if skill_runtime is not None:
                    switch_record = _update_effective_skill_runtime(
                        skill_runtime,
                        requested_skill=planner_skill,
                        requested_skill_text=skill_status["requested_skill"],
                        round_number=round_number,
                        current_round_number=round_number,
                        rationale=rationale,
                        executed_rounds=len(rounds),
                        supported_binding_no_growth_rounds=supported_binding_no_growth_rounds,
                        no_evidence_growth_rounds=no_evidence_growth_rounds,
                        write_trace_event=self.workspace.write_trace_event,
                        recent_switches=tuple(run_state.skill_switch_history),
                    )
                    if switch_record is not None:
                        run_state.skill_switch_history.append(switch_record)
                else:
                    last_selected_skill_id = f"{planner_skill.name}@v{planner_skill.version}"
            elif skill_status["status"] == "missing":
                payload = {"round": round_number}
                if skill_runtime is not None:
                    payload["effective_skill"] = _runtime_skill_id(skill_runtime.effective_skill)
                    payload["message"] = "Planner omitted skill; using the run-level effective skill."
                else:
                    payload["message"] = "Planner omitted skill; continuing with ordinary exploration policy."
                self.workspace.write_trace_event("planner_skill_missing", payload)
            elif skill_status["status"] == "invalid":
                payload = {"round": round_number, "requested_skill": skill_status["requested_skill"]}
                if skill_runtime is not None:
                    payload["effective_skill"] = _runtime_skill_id(skill_runtime.effective_skill)
                    payload["message"] = "Planner requested an unknown skill; using the run-level effective skill."
                else:
                    payload["message"] = "Planner requested an unknown skill; continuing with ordinary exploration policy."
                self.workspace.write_trace_event("planner_skill_invalid", payload)
            active_skill = skill_runtime.effective_skill if skill_runtime is not None else planner_skill
            if skill_runtime is not None or planner_skill is not None:
                last_selected_skill_id = _runtime_skill_id(active_skill)
            round_ctx.skill_runtime = skill_runtime
            round_ctx.evidence_policy = skill_runtime.effective_policy if skill_runtime is not None else None

            if status == "final":
                final_citations = [str(item) for item in action.get("citations", [])]
                final_evidence_ids = [str(item) for item in action.get("evidence_ids", [])]
                final_answer = _planner_final_answer_with_option(
                    question=raw_question,
                    answer=str(action.get("answer", "")),
                )
                bridge_result = _bridge_final_evidence_refs(
                    workspace=self.workspace,
                    question=raw_question,
                    answer=final_answer,
                    citations=final_citations,
                    evidence_ids=final_evidence_ids,
                )
                if bridge_result.changed or bridge_result.ambiguous:
                    self.workspace.write_trace_event(
                        "planner_final_evidence_id_bridge",
                        {
                            "round": round_number,
                            "citations_before": final_citations,
                            "evidence_ids_before": final_evidence_ids,
                            "citations_after": bridge_result.citations,
                            "evidence_ids_after": bridge_result.evidence_ids,
                            "filled_evidence_ids": bridge_result.filled_evidence_ids,
                            "ambiguous": bridge_result.ambiguous,
                        },
                    )
                final_citations = bridge_result.citations
                final_evidence_ids = bridge_result.evidence_ids
                final_source = "planner_final"
                if extract_candidate_options(raw_question):
                    self.workspace.write_trace_event(
                        "planner_final_answer_verifier_requested",
                        {
                            "round": round_number,
                            "planner_answer": str(action.get("answer", "")),
                            "mapped_planner_answer": final_answer,
                            "planner_citations": final_citations,
                            "planner_evidence_ids": final_evidence_ids,
                        },
                    )
                    verifier_blocked_reason = self._verify_planner_final_with_answer_agent(
                        question=raw_question,
                        round_number=round_number,
                        answer=final_answer,
                        citations=final_citations,
                    )
                    if verifier_blocked_reason:
                        blocked_reason = verifier_blocked_reason
                    else:
                        blocked_reason = _blocked_planner_final_reason(
                            question=raw_question,
                            has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                            workspace=self.workspace,
                            answer=final_answer,
                            citations=final_citations,
                            evidence_ids=final_evidence_ids,
                            planner_skill=active_skill,
                        )
                    final_source = "planner_final_after_answer_agent_verifier"
                else:
                    blocked_reason = _blocked_planner_final_reason(
                        question=raw_question,
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                        workspace=self.workspace,
                        answer=final_answer,
                        citations=final_citations,
                        evidence_ids=final_evidence_ids,
                        planner_skill=active_skill,
                    )
                if (
                    blocked_reason == "planner_final_requires_supported_evidence_id"
                    and not prefinal_evidence_repair_done
                    and _prefinal_repair_kind(active_skill) is PrefinalRepairKind.NARRATION_TIMELINE
                ):
                    prefinal_evidence_repair_done = True
                    repair = self._try_narration_prefinal_evidence_repair(
                        question=raw_question,
                        answer=final_answer,
                        citations=final_citations,
                        runtime_context=round_ctx,
                    )
                    if repair is not None:
                        repair_observation_ids, repair_evidence_ids = repair
                        citations.extend(
                            obs_id for obs_id in repair_observation_ids if obs_id not in citations
                        )
                        final_citations = _unique_preserving_order([*final_citations, *repair_observation_ids])
                        final_evidence_ids = _unique_preserving_order([*final_evidence_ids, *repair_evidence_ids])
                        blocked_reason = _blocked_planner_final_reason(
                            question=raw_question,
                            has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                            workspace=self.workspace,
                            answer=final_answer,
                            citations=final_citations,
                            evidence_ids=final_evidence_ids,
                            planner_skill=active_skill,
                        )
                        if not blocked_reason:
                            final_source = "planner_final_after_prefinal_evidence_repair"
                    else:
                        prefinal_evidence_repair_failed = True
                        post_repair_action_rounds_remaining = 1
                        post_repair_action_start_round = round_number + 1
                        evidence_repair_candidate_answer = final_answer
                        evidence_repair_failure_reason = "prefinal_repair_failed"
                if blocked_reason:
                    if (
                        blocked_reason == "planner_final_requires_supported_evidence_id"
                        and prefinal_evidence_repair_done
                        and prefinal_evidence_repair_failed
                        and post_repair_action_rounds_remaining <= 0
                    ):
                        return self._evidence_repair_exhausted_result(
                            question=raw_question,
                            video_path=video_path,
                            rounds=rounds,
                            round_number=round_number,
                            planner_text=planner_response.text,
                            rationale=rationale or blocked_reason,
                            citations=citations,
                            candidate_answer=evidence_repair_candidate_answer or final_answer,
                            failure_reason=evidence_repair_failure_reason or blocked_reason,
                        )
                    if blocked_reason == "planner_final_verifier_disagrees":
                        planner_final_verifier_disagreed = True
                        planner_final_auto_final_blocked = True
                    if blocked_reason == "planner_final_requires_supported_evidence_id":
                        planner_final_auto_final_blocked = True
                    if blocked_reason.startswith("final_gate:"):
                        reason_code, repair_hint = _final_gate_feedback_parts(blocked_reason)
                        last_diagnostic_repair_hint = repair_hint
                        last_projection_status = {
                            "status": "rejected",
                            "candidate_option": _answer_option_letter(final_answer) or "",
                            "reason": reason_code,
                            "missing": repair_hint,
                        }
                    self.workspace.write_trace_event(
                        "iterative_final_blocked",
                        {
                            "round": round_number,
                            "reason": blocked_reason,
                            "answer": final_answer,
                            "citations": final_citations,
                            "evidence_ids": final_evidence_ids,
                            "diagnostic_repair_hint": last_diagnostic_repair_hint,
                        },
                    )
                    self.workspace.write_reflection_memory(
                        route=effective_route,
                        failure_tag=blocked_reason,
                        rule=_reflection_rule_for_failure(blocked_reason),
                    )
                    if final_round_reserved:
                        final_decision = f"final_rejected:{_final_rejection_reason_code(blocked_reason)}"
                        self.workspace.write_trace_event(
                            "iterative_final_rejected",
                            {
                                "round": round_number,
                                "final_decision": final_decision,
                                "reason": blocked_reason,
                                "answer": final_answer,
                                "citations": final_citations,
                                "evidence_ids": final_evidence_ids,
                            },
                        )
                        rounds.append(
                            IterativeRound(
                                round_number=round_number,
                                status="final_rejected",
                                planner_text=planner_response.text,
                                rationale=blocked_reason,
                            )
                        )
                        fallback_result = self._forced_mcq_fallback_result(
                            question=raw_question,
                            video_path=video_path,
                            rounds=rounds,
                            round_number=round_number,
                            citations=final_citations or citations,
                            source="final_rejected",
                            candidate_answer=final_answer,
                            reason=blocked_reason,
                        )
                        if fallback_result is not None:
                            return fallback_result
                        return IterativeRunResult(
                            question=raw_question,
                            video_path=video_path,
                            answer=final_decision,
                            status="final_rejected",
                            citations=final_citations,
                            evidence_ids=final_evidence_ids,
                            confidence=float(action.get("confidence", 0.0) or 0.0),
                            rounds=rounds,
                        )
                    planned_program = []
                    if prefinal_evidence_repair_failed and post_repair_action_rounds_remaining > 0:
                        answer_feedback = [
                            "Prefinal evidence repair did not create answer-grade evidence. You have one "
                            "post-repair action round: provide a different answer with explicit supported "
                            "evidence_ids, call one non-repeating transcript evidence tool, or abstain."
                        ]
                    elif bridge_result.ambiguous:
                        answer_feedback = [
                            "A cited observation maps to multiple supported evidence bindings; do not guess. "
                            f"Choose explicit evidence_ids from: {bridge_result.ambiguous[:3]}."
                        ]
                    if not final_round_reserved and not prefinal_evidence_repair_failed:
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
                        answer=final_answer,
                        citations=final_citations,
                        evidence_ids=final_evidence_ids,
                        source=final_source,
                    )
                    return IterativeRunResult(
                        question=raw_question,
                        video_path=video_path,
                        answer=final_answer,
                        status="final",
                        citations=final_citations,
                        evidence_ids=final_evidence_ids,
                        confidence=float(action.get("confidence", 0.0)),
                        rounds=rounds,
                    )

            if status == "continue":
                normalization_notes: list[NormalizationNote] = []
                program = self._normalize_program(
                    planned_program,
                    question=exploration_question_text,
                    vlm_safe_question=vlm_safe_question,
                    raw_question=raw_question,
                    video_path=video_path,
                    inspected_segment_ids=inspected_segment_ids,
                    final_round_reserved=final_round_reserved,
                    planner_skill=active_skill,
                    notes_out=normalization_notes,
                    run_state=run_state,
                )
                last_round_normalization_notes = normalization_notes
                if run_state.route_repair_exhausted is not None:
                    exhausted_payload = dict(run_state.route_repair_exhausted)
                    rounds.append(
                        IterativeRound(
                            round_number=round_number,
                            status="route_repair_exhausted",
                            planner_text=planner_response.text,
                            rationale=rationale or str(exhausted_payload.get("reason", "")),
                            program=[],
                            observation_ids=[],
                        )
                    )
                    partial_answer = _partial_answer_from_ledger(self._read_ledger())
                    return IterativeRunResult(
                        question=raw_question,
                        video_path=video_path,
                        answer=partial_answer
                        or "Stopped because the same route repair was requested three times without new supported evidence.",
                        status="route_repair_exhausted",
                        citations=citations,
                        rounds=rounds,
                    )
                invalid_failure_signature = _normalization_failure_signature(
                    planned_program=planned_program,
                    normalization_notes=normalization_notes,
                    active_skill=active_skill,
                )
                if invalid_failure_signature is not None and self.budget.max_repeated_invalid_programs > 0:
                    repeat_count = invalid_failure_counts.get(invalid_failure_signature, 0) + 1
                    invalid_failure_counts[invalid_failure_signature] = repeat_count
                    recovery = _structured_recovery_for_failure(
                        invalid_failure_signature,
                        normalization_notes=normalization_notes,
                    )
                    if repeat_count >= self.budget.max_repeated_invalid_programs:
                        self.workspace.write_trace_event(
                            "protocol_repair_exhausted",
                            {
                                "round": round_number,
                                "repeat_count": repeat_count,
                                "max_repeated_invalid_programs": self.budget.max_repeated_invalid_programs,
                                "failure_signature": asdict(invalid_failure_signature),
                                "recovery": recovery,
                            },
                        )
                        rounds.append(
                            IterativeRound(
                                round_number=round_number,
                                status="protocol_repair_exhausted",
                                planner_text=planner_response.text,
                                rationale=rationale or invalid_failure_signature.reason,
                                program=[],
                                observation_ids=[],
                            )
                        )
                        partial_answer = _partial_answer_from_ledger(self._read_ledger())
                        return IterativeRunResult(
                            question=raw_question,
                            video_path=video_path,
                            answer=partial_answer
                            or "Stopped because the same invalid tool protocol repeated three times.",
                            status="protocol_repair_exhausted",
                            citations=citations,
                            rounds=rounds,
                        )
                    if repeat_count == 2:
                        self.workspace.write_trace_event(
                            "repeated_invalid_program_blocked",
                            {
                                "round": round_number,
                                "repeat_count": repeat_count,
                                "failure_signature": asdict(invalid_failure_signature),
                                "recovery": recovery,
                            },
                        )
                        answer_feedback = [_protocol_recovery_feedback(recovery)]
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
                    skip_reason = self._generic_forced_visual_skip_reason(
                        question=raw_question,
                        planner_skill=active_skill,
                        run_state=run_state,
                    )
                    forced_program = [] if skip_reason else self._fallback_visual_evidence_program(
                        question=exploration_question_text,
                        video_path=video_path,
                        inspected_segment_ids=inspected_segment_ids,
                        planner_skill=active_skill,
                    )
                    if skip_reason:
                        self.workspace.write_trace_event(
                            "forced_visual_skipped_for_transcript_route",
                            {
                                "round": round_number,
                                "reason": skip_reason,
                                "trigger": "force_visual_after_no_evidence_growth",
                            },
                        )
                    if forced_program:
                        resolved_program = _append_program_steps(
                            program,
                            forced_program,
                            max_steps=self.budget.max_tool_calls_per_round,
                        )
                        self.workspace.write_trace_event(
                            "exploration_policy_adjustment",
                            {
                                "reason": "force_visual_after_no_evidence_growth",
                                "round": round_number,
                                "no_growth_rounds": no_evidence_growth_rounds,
                                "skipped_tools": [str(step.get("tool", "")) for step in program],
                                "resolved_program": forced_program,
                                "mode": "append_visual_followup",
                            },
                        )
                        program = resolved_program
                if (
                    program
                    and not final_round_reserved
                    and not _program_has_visual_evidence_tool(program)
                    and _all_scene_segments_inspected(self.scene_index, inspected_segment_ids)
                ):
                    skip_reason = self._generic_forced_visual_skip_reason(
                        question=raw_question,
                        planner_skill=active_skill,
                        run_state=run_state,
                    )
                    forced_program = [] if skip_reason else self._visual_evidence_from_navigation_program(
                        program=program,
                        question=exploration_question_text,
                        video_path=video_path,
                        planner_skill=active_skill,
                    )
                    if skip_reason:
                        self.workspace.write_trace_event(
                            "forced_visual_skipped_for_transcript_route",
                            {
                                "round": round_number,
                                "reason": skip_reason,
                                "trigger": "force_visual_from_navigation_no_growth",
                            },
                        )
                    if forced_program:
                        resolved_program = _append_program_steps(
                            program,
                            forced_program,
                            max_steps=self.budget.max_tool_calls_per_round,
                        )
                        self.workspace.write_trace_event(
                            "exploration_policy_adjustment",
                            {
                                "round": round_number,
                                "reason": "force_visual_from_navigation_no_growth",
                                "original_tools": [str(step.get("tool", "")) for step in program],
                                "forced_tools": [str(step.get("tool", "")) for step in forced_program],
                                "mode": "append_visual_followup",
                            },
                        )
                        program = resolved_program
                if (
                    round_number > 1
                    and extract_candidate_options(raw_question)
                    and not final_round_reserved
                    and program
                    and not _program_has_visual_evidence_tool(program)
                    and not _evidence_status_has_strong_option_support(evidence_status_summary)
                ):
                    skip_reason = self._generic_forced_visual_skip_reason(
                        question=raw_question,
                        planner_skill=active_skill,
                        run_state=run_state,
                    )
                    forced_program = [] if skip_reason else self._fallback_visual_evidence_program(
                        question=exploration_question_text,
                        video_path=video_path,
                        inspected_segment_ids=inspected_segment_ids,
                        planner_skill=active_skill,
                    )
                    if skip_reason:
                        self.workspace.write_trace_event(
                            "forced_visual_skipped_for_transcript_route",
                            {
                                "round": round_number,
                                "reason": skip_reason,
                                "trigger": "force_uninspected_visual_without_option_support",
                            },
                        )
                    if forced_program:
                        resolved_program = _append_program_steps(
                            program,
                            forced_program,
                            max_steps=self.budget.max_tool_calls_per_round,
                        )
                        self.workspace.write_trace_event(
                            "exploration_policy_adjustment",
                            {
                                "reason": "force_uninspected_visual_without_option_support",
                                "round": round_number,
                                "skipped_tools": [str(step.get("tool", "")) for step in program],
                                "resolved_program": forced_program,
                                "evidence_status": evidence_status_summary,
                                "mode": "append_visual_followup",
                            },
                        )
                        program = resolved_program
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
                        run_state=run_state,
                        program=program,
                        observation_ids=[],
                        pending_inferences_out=pending_inferences,
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
                    **_answer_agent_operator_kwargs(
                        workspace=self.workspace,
                        projection_status=last_projection_status,
                        diagnostic_repair_hint=last_diagnostic_repair_hint,
                    ),
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
                        **_answer_agent_trace_fields(answer_result),
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
                    runtime_context=round_ctx,
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
                    run_state=run_state,
                    program=program,
                    observation_ids=[],
                    pending_inferences_out=pending_inferences,
                )
                if guard_final is not None:
                    return guard_final
                guard_fallback = self._forced_mcq_fallback_result(
                    question=raw_question,
                    video_path=video_path,
                    rounds=rounds,
                    round_number=round_number,
                    citations=citations,
                    source="repeated_program_guard",
                    candidate_answer=_partial_answer_from_ledger(self._read_ledger()),
                    reason="repeated_program_guard",
                )
                if guard_fallback is not None:
                    return guard_fallback
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
            program_result = self._run_program(
                program,
                ctx=round_ctx,
                question=raw_question,
                video_path=video_path,
                route=effective_route,
                round_number=round_number,
                skill_runtime=skill_runtime,
                evidence_policy=skill_runtime.effective_policy if skill_runtime is not None else None,
            )
            observation_ids = [str(observation_id) for observation_id in program_result.observation_ids]
            self._record_zero_yield_tool_calls(
                program=program,
                observation_ids=observation_ids,
                run_state=run_state,
            )
            self._write_recovery_execution_traces(
                round_number=round_number,
                program=program,
                observation_ids=observation_ids,
                run_state=run_state,
            )
            citations.extend(observation_ids)
            inspected_segment_ids.update(_segment_ids_from_program(program))
            if _program_has_inspect_with_candidate_options(program):
                has_inspect_with_candidate_options = True
            timeline_decision = (
                _timeline_temporal_decision(question=raw_question, timeline=self.workspace.read_timeline_sorted())
                if effective_route == "temporal_order"
                else None
            )
            if timeline_decision is not None:
                timeline_citations = [str(obs_id) for obs_id in timeline_decision["citations"]]
                inference_hint = _timeline_decision_pending_inference(timeline_decision)
                pending_inferences = [inference_hint]
                self.workspace.write_trace_event(
                    "iterative_timeline_temporal_inference",
                    {
                        "round": round_number,
                        "answer": str(timeline_decision["answer"]),
                        "citations": timeline_citations,
                        "matched_events": list(timeline_decision["matched_events"]),
                        "source": "interactive_loop",
                        "planner_action": "hint_only",
                    },
                )
            current_evidence_table_row_count = self.workspace.evidence_table_row_count()
            if current_evidence_table_row_count <= last_evidence_table_row_count:
                no_evidence_growth_rounds += 1
            else:
                no_evidence_growth_rounds = 0
            last_evidence_table_row_count = current_evidence_table_row_count
            current_supported_binding_count = self._supported_evidence_binding_count()
            if current_supported_binding_count <= last_supported_binding_count:
                supported_binding_no_growth_rounds += 1
            else:
                supported_binding_no_growth_rounds = 0
                self._reset_route_repair_counts_for_supported_bindings(run_state=run_state)
            last_supported_binding_count = current_supported_binding_count
            if (
                prefinal_evidence_repair_failed
                and post_repair_action_rounds_remaining > 0
                and round_number >= post_repair_action_start_round
            ):
                post_repair_action_rounds_remaining -= 1
                if not _supported_evidence_ids_for_answer(
                    workspace=self.workspace,
                    question=raw_question,
                    answer=evidence_repair_candidate_answer,
                ):
                    return self._evidence_repair_exhausted_result(
                        question=raw_question,
                        video_path=video_path,
                        rounds=rounds,
                        round_number=round_number,
                        planner_text=planner_response.text,
                        rationale=rationale or "post_repair_action_round_exhausted",
                        citations=citations,
                        candidate_answer=evidence_repair_candidate_answer,
                        failure_reason=evidence_repair_failure_reason or "post_repair_action_round_exhausted",
                        program=program,
                        observation_ids=observation_ids,
                    )
            if supported_binding_no_growth_rounds >= 3 and not run_state.no_progress_warning_emitted:
                run_state.no_progress_warning_emitted = True
                promotion_candidates = _latest_asr_binding_candidates(
                    workspace=self.workspace,
                    target_refs=(),
                    failed_call_signatures=run_state.zero_yield_tool_signatures,
                    limit=2,
                )
                self.workspace.write_trace_event(
                    "iterative_no_progress_warning",
                    {
                        "round": round_number,
                        "reason": "supported_evidence_binding_no_growth",
                        "no_growth_rounds": supported_binding_no_growth_rounds,
                        "supported_binding_count": current_supported_binding_count,
                        "promotion_candidates": promotion_candidates,
                    },
                )
                answer_feedback = _supported_binding_no_growth_feedback(
                    candidates=promotion_candidates,
                    skill_locked=bool(skill_runtime is not None and skill_runtime.locked),
                )
            if no_evidence_growth_rounds >= 2 and not final_round_reserved:
                answer_result = AnswerAgent(self.backend).run(
                    question=raw_question,
                    evidence_text=self._read_ledger(),
                    evidence_table=self._answer_evidence_table(raw_question),
                    **_answer_agent_operator_kwargs(
                        workspace=self.workspace,
                        projection_status=last_projection_status,
                        diagnostic_repair_hint=last_diagnostic_repair_hint,
                    ),
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
                    remaining_rounds=self.budget.max_rounds - round_number,
                    supported_binding_no_growth_rounds=supported_binding_no_growth_rounds,
                    runtime_context=round_ctx,
                )
                if low_confidence_result is not None:
                    return low_confidence_result
                if answer_result.status == "final":
                    finalization_ready = self._finalize_answer_agent_result(
                        answer_result=answer_result,
                        question=raw_question,
                        video_path=video_path,
                        rounds=rounds,
                        round_number=round_number,
                        source="evidence_table_no_growth",
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                        program=program,
                        observation_ids=observation_ids,
                    )
                    if finalization_ready is not None:
                        return finalization_ready
                if answer_result.status == "final" or answer_result.has_partial_support():
                    pending_inferences = [
                        _answer_result_pending_inference(answer_result, source="evidence_table_no_growth")
                    ]
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
                    **_answer_agent_operator_kwargs(
                        workspace=self.workspace,
                        projection_status=last_projection_status,
                        diagnostic_repair_hint=last_diagnostic_repair_hint,
                    ),
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
                        **_answer_agent_trace_fields(answer_result),
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
                    stable_final = self._try_stable_answer_suggestion_final(
                        answer_result=answer_result,
                        question=raw_question,
                        video_path=video_path,
                        rounds=rounds,
                        round_number=round_number,
                        source="prefinal_probe",
                        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                        run_state=run_state,
                        program=program,
                        observation_ids=observation_ids,
                    )
                    if stable_final is not None:
                        return stable_final
                    pending_inferences = [_answer_result_pending_inference(answer_result, source="prefinal_probe")]
                    self.workspace.write_trace_event(
                        "iterative_answer_suggestion",
                        {
                            "round": round_number,
                            "source": "prefinal_probe",
                            "answer": answer_result.answer,
                            "citations": list(answer_result.citations),
                            "confidence": answer_result.confidence,
                            "recommended_to_planner": True,
                        },
                    )
                elif round_number >= self.budget.max_rounds and not prefinal_evidence_repair_failed:
                    low_confidence_result = self._try_low_confidence_final(
                        answer_result=answer_result,
                        question=raw_question,
                        video_path=video_path,
                        rounds=rounds,
                        round_number=round_number,
                        source="prefinal_probe_budget_exhausted",
                        program=program,
                        observation_ids=observation_ids,
                        runtime_context=round_ctx,
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
        if not planner_final_auto_final_blocked:
            budget_final = self._try_answer_agent_final(
                question=raw_question,
                video_path=video_path,
                rounds=rounds,
                round_number=self.budget.max_rounds,
                source="budget_exhausted",
                has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                run_state=run_state,
            )
            if budget_final is not None:
                return budget_final
        else:
            self.workspace.write_trace_event(
                "iterative_answer_agent_skipped",
                {
                    "round": self.budget.max_rounds,
                    "source": "budget_exhausted",
                    "reason": "planner_final_auto_final_blocked",
                },
            )
        if prefinal_evidence_repair_failed:
            return self._evidence_repair_exhausted_result(
                question=raw_question,
                video_path=video_path,
                rounds=rounds,
                round_number=self.budget.max_rounds,
                planner_text="",
                rationale="budget_exhausted_after_prefinal_evidence_repair_failed",
                citations=citations,
                candidate_answer=evidence_repair_candidate_answer,
                failure_reason=evidence_repair_failure_reason or "budget_exhausted_after_prefinal_evidence_repair_failed",
            )
        budget_answer_result: AnswerAgentResult | None = None
        if extract_candidate_options(raw_question) and citations and not planner_final_auto_final_blocked:
            answer_result = AnswerAgent(self.backend).run(
                question=raw_question,
                evidence_text=self._read_ledger(),
                evidence_table=self._answer_evidence_table(raw_question),
                **_answer_agent_operator_kwargs(
                    workspace=self.workspace,
                    projection_status=last_projection_status,
                    diagnostic_repair_hint=last_diagnostic_repair_hint,
                ),
            )
            budget_answer_result = answer_result
            self.workspace.write_trace_event(
                "iterative_answer_agent",
                {
                    "round": self.budget.max_rounds,
                    "source": "budget_exhausted",
                    "status": answer_result.status,
                    "answer": answer_result.answer,
                    "citations": list(answer_result.citations),
                    "missing_evidence": list(answer_result.missing_evidence),
                    **_answer_agent_trace_fields(answer_result),
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
        forced_mcq = self._forced_mcq_fallback_result(
            question=raw_question,
            video_path=video_path,
            rounds=rounds,
            round_number=self.budget.max_rounds,
            citations=citations,
            source="budget_exhausted",
            answer_result=budget_answer_result,
            candidate_answer=_partial_answer_from_ledger(self._read_ledger()),
            reason="budget_exhausted",
        )
        if forced_mcq is not None:
            return forced_mcq
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

    def _forced_mcq_fallback_result(
        self,
        *,
        question: str,
        video_path: str,
        rounds: Sequence[IterativeRound],
        round_number: int,
        citations: Sequence[str],
        source: str,
        answer_result: AnswerAgentResult | None = None,
        candidate_answer: str = "",
        reason: str = "",
    ) -> IterativeRunResult | None:
        options = extract_candidate_options(question)
        if not options:
            return None
        table = self._answer_evidence_table(question)
        projection_result = _projected_option_from_evidence_table(
            question=question,
            options=options,
            table=table,
            target_registry=getattr(self.workspace, "target_registry", None),
            answer_operator=str(getattr(getattr(self.workspace, "grounding_runtime", None), "answer_operator", "") or ""),
        )
        projection_answer = projection_result.option_label if projection_result.status == "supported" else ""
        low_confidence = answer_result.as_low_confidence_final() if answer_result is not None else None
        answer = projection_answer or (
            _answer_option_letter(low_confidence.answer) if low_confidence is not None and low_confidence.status == "low_confidence_final" else ""
        )
        fallback_source = (
            "answer_evidence_projection"
            if projection_answer
            else "answer_agent_partial_support"
            if low_confidence is not None and low_confidence.status == "low_confidence_final"
            else ""
        )
        if not answer and answer_result is not None:
            answer = _answer_option_letter(answer_result.answer) or ""
            fallback_source = "answer_agent_answer" if answer else fallback_source
        if not answer:
            answer = _best_effort_option_from_evidence_table(table)
            fallback_source = "answer_evidence_table_strong" if answer else fallback_source
        if not answer:
            answer = _top_option_from_evidence_table(table)
            fallback_source = "answer_evidence_table_top" if answer else fallback_source
        if not answer:
            answer = _answer_option_letter(candidate_answer) or ""
            fallback_source = "candidate_answer" if answer else fallback_source
        if not answer:
            answer = _latest_hypothesis_option(self.workspace)
            fallback_source = "latest_hypothesis" if answer else fallback_source
        if not answer:
            answer = _first_candidate_option_letter(options) or ""
            fallback_source = "fixed_first_option" if answer else fallback_source
        projection_citations = list(projection_result.supporting_evidence_ids) if projection_answer else []
        fallback_citations = (
            projection_citations
            or (list(low_confidence.citations) if low_confidence is not None and low_confidence.status == "low_confidence_final" else list(citations))
        )
        confidence = (
            min(0.8, float(projection_result.score or 0.0))
            if projection_answer
            else (low_confidence.confidence if low_confidence is not None and low_confidence.status == "low_confidence_final" else 0.0)
        )
        self.workspace.write_trace_event(
            "mcq_forced_fallback",
            {
                "round": round_number,
                "source": source,
                "answer": answer,
                "reason": reason or "mcq_terminal_non_empty_answer",
                "fallback_source": fallback_source or "fixed_first_option",
                "answer_agent_parse_failed": _answer_agent_parse_failed(answer_result),
                "projection_strategy": projection_result.strategy if projection_answer else "",
                "projection_status": projection_result.status,
                "projection_candidate": projection_result.option_label or "",
                "projection_reason": projection_result.reason,
                "projection_citations": projection_citations,
            },
        )
        final_rounds = list(rounds)
        final_rounds.append(
            IterativeRound(
                round_number=round_number,
                status="low_confidence_final",
                planner_text="",
                rationale="MCQ budget exhausted; returning best-effort non-empty option.",
                program=(),
                observation_ids=(),
            )
        )
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=answer,
            status="low_confidence_final",
            citations=fallback_citations,
            confidence=confidence,
            rounds=final_rounds,
        )

    def _evidence_repair_exhausted_result(
        self,
        *,
        question: str,
        video_path: str,
        rounds: list[IterativeRound],
        round_number: int,
        planner_text: str,
        rationale: str,
        citations: Sequence[str],
        candidate_answer: str,
        failure_reason: str,
        program: Sequence[Mapping[str, Any]] = (),
        observation_ids: Sequence[str] = (),
    ) -> IterativeRunResult:
        candidate_option = _answer_option_letter(candidate_answer)
        self.workspace.write_trace_event(
            "evidence_repair_exhausted",
            {
                "round": round_number,
                "final_decision": "evidence_repair_exhausted",
                "selected_option": "",
                "candidate_option": candidate_option,
                "candidate_answer": candidate_answer,
                "failure_reason": failure_reason,
                "citations": list(citations),
            },
        )
        rounds.append(
            IterativeRound(
                round_number=round_number,
                status="evidence_repair_exhausted",
                planner_text=planner_text,
                rationale=rationale,
                program=list(program),
                observation_ids=list(observation_ids),
            )
        )
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer=(
                "Stopped because narration evidence repair did not produce answer-grade supported evidence "
                f"for candidate option {candidate_option or '(unknown)'}."
            ),
            status="evidence_repair_exhausted",
            citations=list(citations),
            rounds=rounds,
        )

    def _question_for_exploration(self, question_context: QuestionContext, *, route_hint: str = "") -> str:
        question = question_context.raw_question
        registry_targets = _target_entities_from_registry(getattr(self.workspace, "target_registry", None))
        if registry_targets:
            self._exploration_target_entities = registry_targets
        else:
            self._exploration_target_entities = tuple(
                extract_option_target_atoms(question, max_targets=12, include_synonyms=False)
            )
        if not self.budget.rewrite_mcq_for_exploration or not question_context.options:
            return question_context.planner_question
        rewrite = rewrite_exploration_question_with_model(
            self.backend,
            question=question,
            route_hint=route_hint or classify_question_route(question),
        )
        rewrite_targets = tuple(rewrite.target_entities)
        if rewrite_targets and not registry_targets:
            self._exploration_target_entities = rewrite_targets
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
        return rewrite.exploration_question or question_context.vlm_safe_question

    def _initialize_planner_owned_grounding(self, question_context: QuestionContext) -> GroundingBootstrapState:
        if not self.budget.planner_owned_grounding:
            return GroundingBootstrapState()
        if getattr(self.workspace, "target_registry", None) is not None:
            return GroundingBootstrapState(runtime=getattr(self.workspace, "grounding_runtime", None))
        route_hint = classify_question_route(question_context.raw_question)
        self.workspace.write_trace_event(
            "grounding_requested",
            {
                "route_hint": route_hint,
                "option_count": len(question_context.options),
            },
        )
        result = ground_question_with_model(
            self.backend,
            question=question_context.raw_question,
            options=question_context.options,
            route_hint=route_hint,
        )
        raw_text_excerpt = _compact_planner_response(result.raw_text, limit=500)
        self.workspace.write_trace_event(
            "grounding_plan_received",
            {
                "attempts": result.attempts,
                "valid": result.validation.is_valid,
                "fallback_reason": result.fallback_reason,
                "raw_text_chars": len(result.raw_text or ""),
                "raw_text_excerpt": raw_text_excerpt if not result.validation.is_valid else "",
                "feedback": result.validation.feedback() if not result.validation.is_valid else "",
            },
        )
        findings = [finding.__dict__ for finding in result.validation.findings]
        plan = result.plan
        fallback_reason = ""
        if plan is None:
            fallback_reason = result.fallback_reason or "grounding_unavailable"
            try:
                plan = compile_fallback_plan(
                    question_context.raw_question,
                    question_context.options,
                    route_hint,
                )
            except Exception as exc:  # pragma: no cover - defensive guard for malformed caller state
                failure = {
                    "status": "grounding_bootstrap_failed",
                    "final_decision": "grounding_bootstrap_failed",
                    "reason_code": "grounding_bootstrap_failed",
                    "reason": fallback_reason,
                    "fallback_error": str(exc),
                    "attempts": result.attempts,
                    "findings": findings,
                    "feedback": result.validation.feedback() if findings else "",
                    "raw_text_chars": len(result.raw_text or ""),
                    "raw_text_excerpt": raw_text_excerpt,
                }
                self.workspace.write_trace_event("grounding_bootstrap_failed", failure)
                return GroundingBootstrapState(failure=failure)
            self.workspace.write_trace_event(
                "grounding_fallback_compiled",
                {
                    "reason": fallback_reason,
                    "attempts": result.attempts,
                    "findings": findings,
                    "feedback": result.validation.feedback(max_findings=5) if findings else "",
                    "raw_text_chars": len(result.raw_text or ""),
                    "raw_text_excerpt": raw_text_excerpt,
                    "option_count": len(question_context.options),
                    "route_hint": route_hint,
                },
            )
        raw_options = _raw_options_by_id(question_context.options)
        skill_ids = tuple(skill.name for skill in builtin_skill_registry().list())
        compiled = compile_grounding_plan(plan, raw_options=raw_options, skill_ids=skill_ids)
        self.workspace.target_registry = compiled.registry
        self.workspace.grounding_runtime = compiled
        self._exploration_target_entities = _target_entities_from_registry(compiled.registry)
        self.workspace.write_trace_event(
            "target_registry_compiled",
            {
                "source": "grounding_plan",
                "version": compiled.registry.version,
                "plan_hash": compiled.plan_hash,
                "route": compiled.route,
                "recommended_skill": _runtime_skill_id_from_name(compiled.recommended_skill_id),
                "answer_operator": compiled.answer_operator,
                "central_subjects": list(compiled.central_subjects),
                "acceptable_evidence_sources": list(compiled.acceptable_evidence_sources),
                "unresolved_ambiguities": list(compiled.unresolved_ambiguities),
                "target_key_to_id": dict(compiled.target_key_to_id),
                "relation_key_to_id": dict(compiled.relation_key_to_id),
            },
        )
        self.workspace.write_trace_event(
            "target_registry_frozen",
            {
                "version": compiled.registry.version,
                "target_refs": list(compiled.registry.targets_by_id),
                "option_count": len(compiled.registry.options_by_id),
            },
        )
        return GroundingBootstrapState(runtime=compiled)

    def _grounding_bootstrap_failed_result(
        self,
        *,
        question: str,
        video_path: str,
        failure: Mapping[str, Any],
    ) -> IterativeRunResult:
        self.workspace.write_trace_event(
            "iterative_final_rejected",
            {
                "status": "grounding_bootstrap_failed",
                "final_decision": "grounding_bootstrap_failed",
                "reason_code": "grounding_bootstrap_failed",
                "reason": failure.get("reason", "grounding_unavailable"),
                "attempts": failure.get("attempts", 0),
                "findings": list(failure.get("findings", ())),
            },
        )
        return IterativeRunResult(
            question=question,
            video_path=video_path,
            answer="grounding_bootstrap_failed",
            status="grounding_bootstrap_failed",
            citations=(),
            evidence_ids=(),
            confidence=0.0,
            rounds=(),
        )

    def _effective_route(self, raw_question: str) -> str:
        runtime = getattr(self.workspace, "grounding_runtime", None)
        route = str(getattr(runtime, "route", "") or "").strip()
        return route or classify_question_route(raw_question)

    def _verify_planner_final_with_answer_agent(
        self,
        *,
        question: str,
        round_number: int,
        answer: str,
        citations: Sequence[str],
    ) -> str:
        if not extract_candidate_options(question):
            return ""
        answer_result = AnswerAgent(self.backend).run(
            question=question,
            evidence_text=self._read_ledger(),
            evidence_table=self._answer_evidence_table(question),
            **_answer_agent_operator_kwargs(workspace=self.workspace),
        )
        planner_option = _answer_option_letter(answer)
        verifier_option = _answer_option_letter(answer_result.answer)
        verifier_disagrees = (
            answer_result.status == "final"
            and planner_option is not None
            and verifier_option is not None
            and verifier_option != planner_option
        )
        if answer_result.status == "final" and answer_result.candidate_option_relations and not verifier_disagrees:
            self.workspace.annotate_candidate_option_relations(
                observation_ids=answer_result.citations,
                relations=answer_result.candidate_option_relations,
                assigned_by="answer_agent_verifier",
            )
        self.workspace.write_trace_event(
            "planner_final_answer_verifier",
            {
                "round": round_number,
                "planner_answer": answer,
                "planner_citations": list(citations),
                "planner_option": planner_option or "",
                "verifier_status": answer_result.status,
                "verifier_answer": answer_result.answer,
                "verifier_option": verifier_option or "",
                "verifier_citations": list(answer_result.citations),
                "missing_evidence": list(answer_result.missing_evidence),
                "verifier_disagrees": verifier_disagrees,
                **_answer_agent_trace_fields(answer_result),
            },
        )
        self.workspace.write_trace_event(
            "iterative_answer_agent",
            {
                "round": round_number,
                "source": "planner_final_verifier",
                "status": answer_result.status,
                "answer": answer_result.answer,
                "citations": list(answer_result.citations),
                "missing_evidence": list(answer_result.missing_evidence),
                **_answer_agent_trace_fields(answer_result),
            },
        )
        if verifier_disagrees:
            return "planner_final_verifier_disagrees"
        return ""

    def _finalize_answer_agent_result(
        self,
        *,
        answer_result: AnswerAgentResult,
        question: str,
        video_path: str,
        rounds: Sequence[IterativeRound],
        round_number: int,
        source: str,
        has_inspect_with_candidate_options: bool,
        program: Sequence[Mapping[str, Any]] = (),
        observation_ids: Sequence[str] = (),
    ) -> IterativeRunResult | None:
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
        self.workspace.write_trace_event(
            "iterative_finalization_ready",
            {
                "round": round_number,
                "source": source,
                "answer": answer_result.answer,
                "citations": list(answer_result.citations),
                "confidence": answer_result.confidence,
            },
        )
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

    def _try_answer_agent_final(
        self,
        *,
        question: str,
        vlm_safe_question: str = "",
        video_path: str,
        rounds: Sequence[IterativeRound],
        round_number: int,
        source: str,
        has_inspect_with_candidate_options: bool,
        run_state: RunState,
        program: Sequence[Mapping[str, Any]] = (),
        observation_ids: Sequence[str] = (),
        pending_inferences_out: list[str] | None = None,
    ) -> IterativeRunResult | None:
        if not extract_candidate_options(question):
            return None
        answer_result = AnswerAgent(self.backend).run(
            question=question,
            evidence_text=self._read_ledger(),
            evidence_table=self._answer_evidence_table(question),
            **_answer_agent_operator_kwargs(workspace=self.workspace),
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
                **_answer_agent_trace_fields(answer_result),
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
        if source not in _ANSWER_AGENT_AUTO_FINAL_SOURCES:
            stable_final = self._try_stable_answer_suggestion_final(
                answer_result=answer_result,
                question=question,
                video_path=video_path,
                rounds=rounds,
                round_number=round_number,
                source=source,
                has_inspect_with_candidate_options=has_inspect_with_candidate_options,
                run_state=run_state,
                program=program,
                observation_ids=observation_ids,
            )
            if stable_final is not None:
                return stable_final
            if pending_inferences_out is not None:
                pending_inferences_out.append(_answer_result_pending_inference(answer_result, source=source))
            self.workspace.write_trace_event(
                "iterative_answer_suggestion",
                {
                    "round": round_number,
                    "source": source,
                    "answer": answer_result.answer,
                    "citations": list(answer_result.citations),
                    "confidence": answer_result.confidence,
                    "recommended_to_planner": True,
                },
            )
            return None
        return self._finalize_answer_agent_result(
            answer_result=answer_result,
            question=question,
            video_path=video_path,
            rounds=rounds,
            round_number=round_number,
            source=source,
            has_inspect_with_candidate_options=has_inspect_with_candidate_options,
            program=program,
            observation_ids=observation_ids,
        )

    def _try_stable_answer_suggestion_final(
        self,
        *,
        answer_result: AnswerAgentResult,
        question: str,
        video_path: str,
        rounds: Sequence[IterativeRound],
        round_number: int,
        source: str,
        has_inspect_with_candidate_options: bool,
        run_state: RunState,
        program: Sequence[Mapping[str, Any]] = (),
        observation_ids: Sequence[str] = (),
    ) -> IterativeRunResult | None:
        option = _answer_option_letter(answer_result.answer)
        citations = tuple(str(citation) for citation in answer_result.citations if str(citation))
        if answer_result.status != "final" or not option or not citations or answer_result.confidence < 0.9:
            run_state.answer_suggestion_state = AnswerSuggestionState()
            return None
        previous = run_state.answer_suggestion_state
        count = previous.count + 1 if previous.option == option and previous.citations == citations else 1
        run_state.answer_suggestion_state = AnswerSuggestionState(
            option=option,
            citations=citations,
            confidence=answer_result.confidence,
            count=count,
        )
        self.workspace.write_trace_event(
            "stable_answer_suggestion_observed",
            {
                "round": round_number,
                "source": source,
                "answer": answer_result.answer,
                "option": option,
                "citations": list(citations),
                "confidence": answer_result.confidence,
                "stable_count": count,
            },
        )
        if count < 2:
            return None
        self.workspace.write_trace_event(
            "stable_answer_suggestion_finalized",
            {
                "round": round_number,
                "source": source,
                "answer": answer_result.answer,
                "citations": list(citations),
                "confidence": answer_result.confidence,
            },
        )
        return self._finalize_answer_agent_result(
            answer_result=answer_result,
            question=question,
            video_path=video_path,
            rounds=rounds,
            round_number=round_number,
            source="stable_answer_suggestion",
            has_inspect_with_candidate_options=has_inspect_with_candidate_options,
            program=program,
            observation_ids=observation_ids,
        )

    def _record_zero_yield_tool_calls(
        self,
        *,
        program: Sequence[Mapping[str, Any]],
        observation_ids: Sequence[str],
        run_state: RunState,
    ) -> None:
        for step, observation_id in zip(program, observation_ids):
            if not isinstance(step, Mapping):
                continue
            tool_name = str(step.get("tool") or step.get("op") or "").strip()
            args = step.get("args", {})
            if not tool_name or not isinstance(args, Mapping):
                continue
            observation = self.workspace.get_observation(str(observation_id))
            if observation is None or not _observation_is_zero_yield_failure(
                tool_name=tool_name,
                raw_output=observation.raw_output,
            ):
                continue
            signature = _tool_call_signature(tool_name=tool_name, args=args)
            if not signature:
                continue
            run_state.zero_yield_tool_signatures.add(signature)
            self.workspace.write_trace_event(
                "zero_yield_tool_call_recorded",
                {
                    "tool": tool_name,
                    "args": dict(args),
                    "observation_id": str(observation_id),
                    "limitations": str(observation.limitations or observation.raw_output.get("limitations", "")),
                },
            )

    def _normalize_program(
        self,
        program: Any,
        *,
        question: str,
        video_path: str,
        inspected_segment_ids: set[str],
        final_round_reserved: bool,
        planner_skill: SkillSpec | None = None,
        notes_out: list[NormalizationNote] | None = None,
        raw_question: str = "",
        vlm_safe_question: str = "",
        run_state: RunState,
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(program, list):
            raise ValueError("Planner action status=continue requires a list program")

        normalized = []
        reserved_segment_ids = set(inspected_segment_ids)
        active_skill = planner_skill
        blocked_route_violation = False
        pool_exhausted_logged = False
        pending_one_shot_tools: set[str] = set()
        for step_index, step in enumerate(program):
            if not isinstance(step, Mapping):
                raise ValueError("Planner program steps must be objects")
            if "tool" not in step:
                raise ValueError("Planner program step is missing required 'tool'")
            if len(normalized) >= self.budget.max_tool_calls_per_round:
                remaining_steps = [
                    str(item.get("tool", "") or "")
                    for item in program[step_index:]
                    if isinstance(item, Mapping)
                ]
                _append_normalization_note(
                    notes_out,
                    tool="program",
                    reason="max_tool_calls_per_round_exceeded",
                    original={
                        "max_tool_calls_per_round": self.budget.max_tool_calls_per_round,
                        "dropped_tools": remaining_steps,
                    },
                    next_action=(
                        f"Your previous program exceeded max_tool_calls_per_round={self.budget.max_tool_calls_per_round}; "
                        "submit at most that many tool calls and prioritize the single highest-value evidence action."
                    ),
                )
                self.workspace.write_trace_event(
                    "tool_call_budget_exceeded",
                    {
                        "max_tool_calls_per_round": self.budget.max_tool_calls_per_round,
                        "kept_count": len(normalized),
                        "dropped_tools": remaining_steps,
                    },
                )
                break

            tool_name = str(step["tool"])
            args = dict(step.get("args", {}))
            route_kind = str(step.get("route_kind") or "")
            candidate_id = str(step.get("candidate_id") or "")
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
            if tool_name in exhausted_tools:
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
                route_kind = _route_kind_for_repair_reason(repair_reason)
                candidate_id = str(args.pop("_candidate_id", "") or candidate_id)
                repair_action = self._record_route_repair_attempt(
                    run_state=run_state,
                    reason=repair_reason,
                    original_tool_name=original_tool_name,
                    original_args=original_args,
                    repaired_tool_name=tool_name,
                    repaired_args=args,
                    active_skill=active_skill,
                    notes_out=notes_out,
                )
                if repair_action == "propose":
                    blocked_route_violation = True
                    continue
                if repair_action == "exhausted":
                    blocked_route_violation = True
                    continue
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
            if segment_id and _tool_is(tool_name, "read_segment"):
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
            if _tool_forbidden_by_skill(tool_name=tool_name, active_skill=active_skill):
                blocked_route_violation = True
                _record_skill_forbidden_action(
                    workspace=self.workspace,
                    notes_out=notes_out,
                    tool_name=tool_name,
                    args=args,
                    active_skill=active_skill,
                )
                continue
            if _tool_denied_by_skill(tool_name=tool_name, active_skill=active_skill):
                blocked_route_violation = True
                _record_skill_deny_list_violation(
                    workspace=self.workspace,
                    notes_out=notes_out,
                    tool_name=tool_name,
                    args=args,
                    active_skill=active_skill,
                )
                continue
            if _tool_non_suggested_by_playbook(tool_name=tool_name, active_skill=active_skill):
                _record_skill_action_advisory(
                    workspace=self.workspace,
                    notes_out=notes_out,
                    tool_name=tool_name,
                    args=args,
                    active_skill=active_skill,
                )
            violation = _route_violation(tool_name=tool_name, active_skill=active_skill)
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
            normalized_target_args = self._normalize_target_protocol_args(
                tool_name=tool_name,
                args=args,
                notes_out=notes_out,
            )
            if normalized_target_args is None:
                blocked_route_violation = True
                continue
            args = normalized_target_args
            args = self._strip_unsupported_tool_args(tool_name=tool_name, args=args, notes_out=notes_out)
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
            if _tool_is(tool_name, "verify_ledger_answer") and "ledger_text" in args:
                original_args = dict(args)
                args.pop("ledger_text", None)
                self.workspace.write_trace_event(
                    "exploration_policy_adjustment",
                    {
                        "reason": "strip_verifier_ledger_text",
                        "tool": tool_name,
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="strip_verifier_ledger_text",
                    original={"tool": tool_name, "args": original_args},
                    resolved={"tool": tool_name, "args": args},
                    next_action="verify_ledger_answer reads the workspace ledger automatically; do not pass ledger_text.",
                )
            if _tool_is(tool_name, "verify_ledger_answer"):
                verifier_question = raw_question or question
                if verifier_question and self._tool_accepts_argument(tool_name, "question") and not args.get("question"):
                    args["question"] = verifier_question
                candidate_options = extract_candidate_options(verifier_question)
                if (
                    candidate_options
                    and self._tool_accepts_argument(tool_name, "candidate_options")
                    and not args.get("candidate_options")
                ):
                    args["candidate_options"] = list(candidate_options)
            if tool_name in _SEGMENT_TEXT_TOOLS:
                args = self._normalize_text_segment_tool_args(
                    tool_name=tool_name,
                    args=args,
                    notes_out=notes_out,
                )
                if args is None:
                    blocked_route_violation = True
                    continue
            if _tool_is(tool_name, "read_segment_detail"):
                if (
                    self._exploration_target_entities
                    and self._tool_accepts_argument(tool_name, "targets")
                    and not args.get("target_refs")
                    and not args.get("targets")
                ):
                    args["targets"] = list(self._exploration_target_entities)
            if final_round_reserved and tool_name != "verify_ledger_answer":
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

            if _tool_is(tool_name, "verify_segment_anchors"):
                normalized_verify_args = self._normalize_verify_segment_anchors_args(
                    args=args,
                    question=question,
                    raw_question=raw_question,
                    video_path=video_path,
                    planner_skill=planner_skill,
                    notes_out=notes_out,
                )
                if normalized_verify_args is None:
                    blocked_route_violation = True
                    continue
                normalized_step = {"tool": tool_name, "args": normalized_verify_args}
                if "assign" in step:
                    normalized_step["assign"] = str(step["assign"])
                if route_kind:
                    normalized_step["route_kind"] = route_kind
                if candidate_id:
                    normalized_step["candidate_id"] = candidate_id
                normalized.append(normalized_step)
                continue

            if tool_name in _SEGMENT_MEDIA_TOOLS:
                for option_arg in (
                    "candidate_options",
                    "mutex_option_x_text",
                    "mutex_option_y_text",
                    "mutex_option_x",
                    "mutex_option_y",
                ):
                    args.pop(option_arg, None)
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
                preserve_focused_ordered_window = (
                    _tool_is(tool_name, "vision_read")
                    and (
                        route_kind == "focused_ordered_list_vision"
                        or _is_focused_ordered_list_vision_args(args)
                    )
                    and _has_explicit_subwindow(args)
                )
                args["segment_id"] = segment.segment_id
                args["video_path"] = video_path
                if not preserve_focused_ordered_window:
                    args["start_sec"] = segment.start_sec
                    args["end_sec"] = segment.end_sec
                if _tool_is(tool_name, "vision_read"):
                    args.setdefault("ask_for", args.pop("question", question))
                    if not preserve_focused_ordered_window:
                        args["ask_for"] = _tool_exploration_question(
                            str(args["ask_for"]),
                            route_hint=planner_skill.name if planner_skill else "",
                            question_context=question,
                            vlm_safe_question=vlm_safe_question,
                            forbidden_question=raw_question,
                            option_blind=self.budget.rewrite_mcq_for_exploration,
                            target_entities=self._exploration_target_entities,
                        )
                else:
                    args.setdefault("question", question)
                    args["question"] = _tool_exploration_question(
                        str(args["question"]),
                        route_hint=planner_skill.name if planner_skill else "",
                        question_context=question,
                        vlm_safe_question=vlm_safe_question,
                        forbidden_question=raw_question,
                        option_blind=self.budget.rewrite_mcq_for_exploration,
                        target_entities=self._exploration_target_entities,
                    )
                candidate_options = list(extract_candidate_options(question))
                if candidate_options:
                    if _tool_is(tool_name, "vision_read"):
                        args.setdefault("event_label", str(args.get("ask_for", "")))
                if preserve_focused_ordered_window:
                    args.setdefault("nframes", 128)
                else:
                    args.setdefault("nframes", self.budget.default_nframes)
                reserved_segment_ids.add(segment.segment_id)

            zero_yield_signature = _tool_call_signature(tool_name=tool_name, args=args)
            if zero_yield_signature in run_state.zero_yield_tool_signatures:
                next_action = (
                    f"The exact {tool_name} call already produced no answer-grade evidence. "
                    "Use a different segment, target_refs, or evidence tool instead of repeating it."
                )
                self.workspace.write_trace_event(
                    "zero_yield_tool_call_skipped",
                    {
                        "tool": tool_name,
                        "args": args,
                        "reason": "previous_zero_yield",
                    },
                )
                self.workspace.write_reflection_memory(
                    route=active_skill.trigger.route if active_skill is not None else classify_question_route(question),
                    failure_tag="previous_zero_yield_tool_call",
                    rule=next_action,
                )
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="previous_zero_yield_tool_call",
                    original={"tool": tool_name, "args": args},
                    next_action=next_action,
                )
                continue

            normalized_step: dict[str, Any] = {"tool": tool_name, "args": args}
            if "assign" in step:
                normalized_step["assign"] = str(step["assign"])
            if route_kind:
                normalized_step["route_kind"] = route_kind
            if candidate_id:
                normalized_step["candidate_id"] = candidate_id
            normalized.append(normalized_step)
            if tool_name in _ONE_SHOT_TOOLS:
                pending_one_shot_tools.add(tool_name)

        if not normalized and not final_round_reserved and not blocked_route_violation:
            fallback_segment_id = self._resolve_next_segment_id("", reserved_segment_ids)
            fallback_tool_name = self._fallback_visual_tool_name_for_skill(active_skill)
            if fallback_segment_id is not None and fallback_tool_name is not None:
                segment = self.scene_index.get(fallback_segment_id)
                fallback_args: dict[str, Any] = {
                    "video_path": video_path,
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "nframes": self.budget.default_nframes,
                }
                if _tool_is(fallback_tool_name, "vision_read"):
                    target_question = _local_fact_question(
                        question=question,
                        planner_skill=active_skill,
                        target_entities=self._exploration_target_entities,
                    )
                    fallback_args["ask_for"] = target_question
                    fallback_args["event_label"] = target_question
                else:
                    fallback_args["question"] = _local_fact_question(
                        question=question,
                        planner_skill=active_skill,
                        target_entities=self._exploration_target_entities,
                    )
                normalized.append(
                    {
                        "tool": fallback_tool_name,
                        "args": fallback_args,
                        "assign": f"auto_{segment.segment_id}",
                    }
                )
        return normalized

    def _strip_unsupported_tool_args(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        notes_out: list[NormalizationNote] | None,
    ) -> dict[str, Any]:
        try:
            parameters = self.registry.get(tool_name).parameters
        except ToolError:
            return dict(args)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return dict(args)
        unsupported = sorted(str(name) for name in args if str(name) not in parameters)
        if not unsupported:
            return dict(args)
        stripped = {str(name): value for name, value in args.items() if str(name) in parameters}
        self.workspace.write_trace_event(
            "exploration_policy_adjustment",
            {
                "reason": "strip_unsupported_tool_args",
                "tool": tool_name,
                "stripped_args": unsupported,
            },
        )
        _append_normalization_note(
            notes_out,
            tool=tool_name,
            reason="strip_unsupported_tool_args",
            original={"tool": tool_name, "args": dict(args)},
            resolved={"tool": tool_name, "args": stripped},
        )
        return stripped

    def _normalize_target_protocol_args(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        notes_out: list[NormalizationNote] | None,
    ) -> dict[str, Any] | None:
        normalized = dict(args)
        original_args = dict(args)
        additional_targets = (
            _coerce_target_arg_list(normalized.get("additional_targets"))
            if "additional_targets" in normalized
            else []
        )
        if additional_targets:
            if not _additional_targets_allowed(tool_name=tool_name, args=normalized):
                self._record_additional_targets_rejection(
                    tool_name=tool_name,
                    args=original_args,
                    notes_out=notes_out,
                )
                return None
            normalized["additional_targets"] = _unique_nonempty_strings(
                str(target).strip() for target in additional_targets
            )
            if _tool_is(tool_name, "search_segments"):
                normalized["query"] = _append_additional_targets_to_text(
                    normalized.get("query"),
                    normalized["additional_targets"],
                )
            elif tool_name in {"vision_read", "caption_segment"}:
                field_name = _tool_text_field(tool_name, default="question")
                normalized[field_name] = _append_additional_targets_to_text(
                    normalized.get(field_name),
                    normalized["additional_targets"],
                )
            normalized.pop("additional_targets", None)
        target_refs = _coerce_target_arg_list(normalized.get("target_refs")) if "target_refs" in normalized else []
        legacy_targets = _coerce_target_arg_list(normalized.get("targets")) if "targets" in normalized else []
        resolved_refs: list[str] = []
        legacy_refs: list[str] = []
        free_text_targets: list[Any] = []

        for ref in target_refs:
            ref_text = str(ref).strip()
            if not _is_target_ref_key(ref_text):
                reason = "coverage_query_id_not_callable" if _is_coverage_query_id(ref_text) else "free_text_target_ref"
                self._record_target_protocol_rejection(
                    tool_name=tool_name,
                    args=original_args,
                    reason=reason,
                    invalid_target=str(ref),
                    notes_out=notes_out,
                )
                return None
            if not _workspace_knows_target_ref(self.workspace, ref_text):
                self._record_target_protocol_rejection(
                    tool_name=tool_name,
                    args=original_args,
                    reason="unknown_target_ref",
                    invalid_target=ref_text,
                    notes_out=notes_out,
                )
                return None
            resolved_refs.append(ref_text)

        if resolved_refs:
            normalized["target_refs"] = _unique_preserving_order(resolved_refs)
            normalized["normalized_target_keys"] = normalized["target_refs"]
            if legacy_targets:
                normalized.pop("targets", None)
                self.workspace.write_trace_event(
                    "exploration_policy_adjustment",
                    {
                        "reason": "target_refs_precedence_over_legacy_targets",
                        "tool": tool_name,
                        "target_refs": normalized["target_refs"],
                        "legacy_target_count": len(legacy_targets),
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool=tool_name,
                    reason="target_refs_precedence_over_legacy_targets",
                    original={"tool": tool_name, "args": original_args},
                    resolved={"tool": tool_name, "args": normalized},
                    next_action="target_refs are source of truth; legacy targets were retained only as audit text.",
                )
            return normalized

        for target in legacy_targets:
            target_text = str(target).strip()
            if not target_text:
                continue
            if _is_target_ref_key(target_text) and not _workspace_knows_target_ref(self.workspace, target_text):
                self._record_target_protocol_rejection(
                    tool_name=tool_name,
                    args=original_args,
                    reason="unknown_legacy_target_ref",
                    invalid_target=target_text,
                    notes_out=notes_out,
                )
                return None
            upgraded_ref = _exact_registry_ref_for_legacy_target(self.workspace, target_text)
            if upgraded_ref:
                legacy_refs.append(upgraded_ref)
            else:
                free_text_targets.append(target)

        if legacy_refs:
            resolved_refs = _unique_preserving_order([*resolved_refs, *legacy_refs])
            normalized["target_refs"] = resolved_refs
            normalized["normalized_target_keys"] = resolved_refs
            if free_text_targets:
                normalized = self._apply_additional_targets(
                    tool_name=tool_name,
                    normalized=normalized,
                    original_args=original_args,
                    additional_targets=free_text_targets,
                    notes_out=notes_out,
                )
                if normalized is None:
                    return None
            else:
                normalized.pop("targets", None)
            self.workspace.write_trace_event(
                "exploration_policy_adjustment",
                {
                    "reason": "rewrite_legacy_targets_to_target_refs",
                    "tool": tool_name,
                    "target_refs": resolved_refs,
                },
            )
            _append_normalization_note(
                notes_out,
                tool=tool_name,
                reason="rewrite_legacy_targets_to_target_refs",
                original={"tool": tool_name, "args": original_args},
                resolved={"tool": tool_name, "args": normalized},
                next_action="Use target_refs for registry target ids; keep targets for natural-language text only.",
            )
        elif free_text_targets and _workspace_has_target_registry(self.workspace):
            normalized = self._apply_additional_targets(
                tool_name=tool_name,
                normalized=normalized,
                original_args=original_args,
                additional_targets=free_text_targets,
                notes_out=notes_out,
            )
            if normalized is None:
                return None

        return normalized

    def _apply_additional_targets(
        self,
        *,
        tool_name: str,
        normalized: dict[str, Any],
        original_args: Mapping[str, Any],
        additional_targets: Sequence[Any],
        notes_out: list[NormalizationNote] | None,
    ) -> dict[str, Any] | None:
        if not _additional_targets_allowed(tool_name=tool_name, args=normalized):
            self._record_additional_targets_rejection(
                tool_name=tool_name,
                args=original_args,
                notes_out=notes_out,
            )
            return None
        extras = _unique_nonempty_strings(additional_targets)
        if not extras:
            normalized.pop("targets", None)
            return normalized
        if _tool_is(tool_name, "search_segments"):
            normalized["query"] = _append_additional_targets_to_text(normalized.get("query"), extras)
        elif _tool_is(tool_name, "vision_read"):
            normalized["ask_for"] = _append_additional_targets_to_text(normalized.get("ask_for"), extras)
        elif _tool_is(tool_name, "caption_segment"):
            source_field = "question" if "question" in normalized else "ask_for"
            normalized["question"] = _append_additional_targets_to_text(normalized.get(source_field), extras)
        normalized.pop("targets", None)
        normalized.pop("additional_targets", None)
        return normalized

    def _record_additional_targets_rejection(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        notes_out: list[NormalizationNote] | None,
    ) -> None:
        next_action = (
            "additional_targets is allowed only for discovery-only calls: search_segments(query), "
            "vision_read(ask_for), and caption_segment(question). Use target_refs for bound-target tools."
        )
        self.workspace.write_trace_event(
            "exploration_policy_adjustment",
            {
                "error_code": "invalid_tool_args",
                "reason": "additional_targets_not_allowed",
                "reason_code": "additional_targets_not_allowed",
                "tool": tool_name,
                "next_action": next_action,
            },
        )
        _append_normalization_note(
            notes_out,
            tool=tool_name,
            reason="additional_targets_not_allowed",
            original={"tool": tool_name, "args": dict(args)},
            next_action=next_action,
        )

    def _record_target_protocol_rejection(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        reason: str,
        invalid_target: str,
        notes_out: list[NormalizationNote] | None,
    ) -> None:
        registered_refs = _registered_target_ref_descriptions(self.workspace)
        if registered_refs:
            registry_text = "; ".join(registered_refs)
            next_action = (
                "Coverage-row labels are local to the coverage observation and are not automatically valid "
                "target_refs. Only these exact IDs are registered target_refs for this run: "
                f"{registry_text}. Retry with registered target_refs, or use natural-language targets. "
                "Do not invent T<n> IDs."
            )
        else:
            next_action = (
                "No target_refs are registered for this run. Coverage-row labels are local to the coverage "
                "observation. Use natural-language targets and leave target_refs empty."
            )
        self.workspace.write_trace_event(
            "exploration_policy_adjustment",
            {
                "reason": reason,
                "tool": tool_name,
                "invalid_target": invalid_target,
                "registered_target_refs": registered_refs,
                "next_action": next_action,
            },
        )
        _append_normalization_note(
            notes_out,
            tool=tool_name,
            reason=reason,
            original={"tool": tool_name, "args": dict(args)},
            next_action=next_action,
        )

    def _record_route_repair_attempt(
        self,
        *,
        run_state: RunState,
        reason: str,
        original_tool_name: str,
        original_args: Mapping[str, Any],
        repaired_tool_name: str,
        repaired_args: Mapping[str, Any],
        active_skill: SkillSpec | None,
        notes_out: list[NormalizationNote] | None,
    ) -> str:
        key = _route_repair_key(reason=reason, args=original_args)
        count = run_state.route_repair_counts.get(key, 0) + 1
        run_state.route_repair_counts[key] = count
        key_payload = _route_repair_key_payload(key)

        if count == 1:
            self.workspace.write_trace_event(
                "route_repair_applied",
                {
                    **key_payload,
                    "count": count,
                    "skill": active_skill.name if active_skill is not None else "",
                    "requested_tool": original_tool_name,
                    "resolved_tool": repaired_tool_name,
                },
            )
            return "apply"

        recovery_program = _route_repair_recovery_program(
            reason=reason,
            original_args=original_args,
            repaired_tool_name=repaired_tool_name,
            repaired_args=repaired_args,
            active_skill=active_skill,
        )
        if count == 2:
            next_action = (
                "The same repaired route repeated. Do not execute the repaired program again; "
                "switch to the proposed recovery program or finalize from supported evidence."
            )
            self.workspace.write_trace_event(
                "route_repair_recovery_proposed",
                {
                    **key_payload,
                    "count": count,
                    "skill": active_skill.name if active_skill is not None else "",
                    "requested_tool": original_tool_name,
                    "resolved_tool": repaired_tool_name,
                    "recommended_program": recovery_program,
                    "next_action": next_action,
                },
            )
            _append_normalization_note(
                notes_out,
                tool=original_tool_name,
                reason="route_repair_recovery_proposed",
                original={"tool": original_tool_name, "args": dict(original_args)},
                resolved={"recommended_program": recovery_program},
                next_action=next_action,
            )
            return "propose"

        run_state.route_repair_exhausted = {
            **key_payload,
            "count": count,
            "skill": active_skill.name if active_skill is not None else "",
            "requested_tool": original_tool_name,
            "resolved_tool": repaired_tool_name,
            "recommended_program": recovery_program,
        }
        self.workspace.write_trace_event("route_repair_exhausted", dict(run_state.route_repair_exhausted))
        _append_normalization_note(
            notes_out,
            tool=original_tool_name,
            reason="route_repair_exhausted",
            original={"tool": original_tool_name, "args": dict(original_args)},
            resolved={"recommended_program": recovery_program},
            next_action="Stop this route; collect supported evidence through the recovery program before retrying.",
        )
        return "exhausted"

    def _supported_evidence_binding_count(self) -> int:
        return len(_supported_evidence_binding_rows(self.workspace))

    def _reset_route_repair_counts_for_supported_bindings(self, *, run_state: RunState) -> None:
        if not run_state.route_repair_counts:
            return
        rows = _supported_evidence_binding_rows(self.workspace)
        if not rows:
            return
        reset_keys = [
            key
            for key in run_state.route_repair_counts
            if any(_supported_binding_overlaps_route_repair_key(row, key) for row in rows)
        ]
        for key in reset_keys:
            run_state.route_repair_counts.pop(key, None)
        if reset_keys:
            self.workspace.write_trace_event(
                "route_repair_count_reset",
                {"keys": [_route_repair_key_payload(key) for key in reset_keys]},
            )

    def _normalize_text_segment_tool_args(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        notes_out: list[NormalizationNote] | None,
    ) -> dict[str, Any] | None:
        normalized = dict(args)
        requested_segment_id = str(normalized.get("segment_id", "") or "").strip()
        if not requested_segment_id or _scene_segment_or_none(self.scene_index, requested_segment_id) is None:
            self._record_invalid_segment_id(
                tool_name=tool_name,
                args=args,
                requested_segment_id=requested_segment_id,
                notes_out=notes_out,
            )
            return None
        normalized["segment_id"] = requested_segment_id
        return normalized

    def _record_invalid_segment_id(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        requested_segment_id: str,
        notes_out: list[NormalizationNote] | None,
    ) -> None:
        valid_ids = [segment.segment_id for segment in self.scene_index.segments]
        visible_ids = valid_ids[:16]
        suffix = f", ... {len(valid_ids) - len(visible_ids)} more" if len(valid_ids) > len(visible_ids) else ""
        next_action = (
            f"Unknown segment_id '{requested_segment_id or '(missing)'}'. Use one exact segment_id from "
            f"the scene index: {', '.join(visible_ids) or '(none)'}{suffix}. Do not invent segment ids."
        )
        self.workspace.write_trace_event(
            "invalid_segment_id",
            {
                "tool": tool_name,
                "requested_segment_id": requested_segment_id,
                "valid_segment_ids": visible_ids,
                "next_action": next_action,
            },
        )
        _append_normalization_note(
            notes_out,
            tool=tool_name,
            reason="invalid_segment_id",
            original={"tool": tool_name, "args": dict(args)},
            next_action=next_action,
        )

    def _try_narration_prefinal_evidence_repair(
        self,
        *,
        question: str,
        answer: str,
        citations: Sequence[str],
        runtime_context: RunContext,
    ) -> tuple[list[str], list[str]] | None:
        if not self._has_tool("read_segment_detail"):
            self.workspace.write_trace_event(
                "prefinal_evidence_repair_failed",
                {"reason": "read_segment_detail_unavailable"},
            )
            return None
        target_refs = _target_refs_for_answer(workspace=self.workspace, answer=answer)
        if not target_refs:
            self.workspace.write_trace_event(
                "prefinal_evidence_repair_failed",
                {"reason": "no_registered_target_refs", "answer": answer},
            )
            return None
        candidate_segment_ids = self._prefinal_repair_candidate_segment_ids(citations=citations)
        if not candidate_segment_ids:
            self.workspace.write_trace_event(
                "prefinal_evidence_repair_failed",
                {"reason": "no_candidate_segment", "answer": answer, "target_refs": target_refs},
            )
            return None
        before_ids = set(_supported_evidence_ids_for_answer(workspace=self.workspace, question=question, answer=answer))
        observation_ids: list[str] = []
        for segment_id in candidate_segment_ids[:2]:
            program = [
                {
                    "tool": "read_segment_detail",
                    "args": {
                        "segment_id": segment_id,
                        "target_refs": target_refs,
                        "promote_answer_evidence": True,
                    },
                    "assign": "prefinal_evidence_repair",
                }
            ]
            self.workspace.write_trace_event(
                "prefinal_evidence_repair_requested",
                {
                    "segment_id": segment_id,
                    "target_refs": target_refs,
                    "answer": answer,
                },
            )
            result = self._run_program(
                program,
                ctx=runtime_context,
                question=question,
                video_path=self.scene_index.video_path,
                route=classify_question_route(question),
            )
            observation_ids.extend(str(obs_id) for obs_id in result.observation_ids)
        after_ids = _supported_evidence_ids_for_answer(workspace=self.workspace, question=question, answer=answer)
        new_ids = [evidence_id for evidence_id in after_ids if evidence_id not in before_ids]
        if not new_ids:
            self.workspace.write_trace_event(
                "prefinal_evidence_repair_failed",
                {
                    "reason": "no_supported_binding_created",
                    "answer": answer,
                    "candidate_segments": candidate_segment_ids[:2],
                    "target_refs": target_refs,
                    "observation_ids": observation_ids,
                },
            )
            return None
        self.workspace.write_trace_event(
            "sequence_binding_created",
            {
                "source": "prefinal_evidence_repair",
                "answer": answer,
                "evidence_ids": new_ids,
                "observation_ids": observation_ids,
            },
        )
        return observation_ids, new_ids

    def _prefinal_repair_candidate_segment_ids(self, *, citations: Sequence[str]) -> list[str]:
        cited = {str(citation) for citation in citations if str(citation)}
        segment_ids: list[str] = []
        for observation in self.workspace.read_observations():
            if cited and observation.observation_id not in cited:
                continue
            segment_ids.extend(_segment_ids_from_observation_payload(observation.raw_output))
        for observation in reversed(self.workspace.read_observations(tool_name="target_coverage")):
            coverage = observation.raw_output.get("coverage", [])
            if not isinstance(coverage, Sequence) or isinstance(coverage, (str, bytes)):
                continue
            for row in coverage:
                if not isinstance(row, Mapping):
                    continue
                candidates = row.get("candidates", [])
                if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                    continue
                for candidate in candidates:
                    if isinstance(candidate, Mapping):
                        segment_ids.append(str(candidate.get("segment_id", "") or "").strip())
        segment_ids.extend(segment.segment_id for segment in self.scene_index.segments)
        valid = {segment.segment_id for segment in self.scene_index.segments}
        return _unique_preserving_order([segment_id for segment_id in segment_ids if segment_id in valid])

    def _repair_skill_route_tool(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        active_skill: SkillSpec | None,
        question: str,
        video_path: str,
    ) -> tuple[str, dict[str, Any], str] | None:
        is_main_idea_route = active_skill is not None and active_skill.name == "main_idea"
        if is_main_idea_route and _tool_is(tool_name, "vision_read") and self._has_tool("global_gist"):
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
            and _tool_is(tool_name, "global_gist")
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
                        "Focus on visible entities, events, and how this part contributes to the overall topic."
                    ),
                    "event_label": "localized main-idea evidence",
                },
                "repair_repeated_main_idea_global_gist_to_vision_read",
            )
        if (
            active_skill is not None
            and active_skill.name in {"timeline_ordering", "narration_timeline_qa", "visual_timeline_qa"}
            and _tool_is(tool_name, "locate_targets_in_segment")
        ):
            if active_skill.name in {"timeline_ordering", "visual_timeline_qa"} and self._has_tool("vision_read"):
                recommended_action = self._latest_locator_recommended_action(
                    segment_id=str(args.get("segment_id") or ""),
                    route_kind="focused_ordered_list_vision",
                )
                if recommended_action:
                    recovery_args = dict(recommended_action.get("args") or {})
                    candidate_id = str(recommended_action.get("candidate_id") or "")
                    if candidate_id:
                        recovery_args["_candidate_id"] = candidate_id
                    self.workspace.write_trace_event(
                        "route_recovery_selected",
                        {
                            "route_kind": "focused_ordered_list_vision",
                            "candidate_id": candidate_id,
                            "tool": "vision_read",
                            "args": dict(recommended_action.get("args") or {}),
                        },
                    )
                    return (
                        "vision_read",
                        recovery_args,
                        "repair_ordered_list_locator_to_focused_ordered_list_vision",
                    )
            if active_skill.name == "narration_timeline_qa" and self._has_tool("read_segment_detail"):
                promotion_args = self._narration_transcript_promotion_args(
                    segment_id=str(args.get("segment_id") or ""),
                    original_args=args,
                )
                if promotion_args:
                    candidate_id = f"narration_transcript_promotion:{promotion_args['segment_id']}"
                    promotion_args["_candidate_id"] = candidate_id
                    self.workspace.write_trace_event(
                        "route_recovery_selected",
                        {
                            "route_kind": "narration_transcript_promotion",
                            "candidate_id": candidate_id,
                            "tool": "read_segment_detail",
                            "args": {key: value for key, value in promotion_args.items() if not key.startswith("_")},
                        },
                    )
                    self.workspace.write_trace_event(
                        "narration_transcript_promotion_recommended",
                        {
                            "candidate_id": candidate_id,
                            "segment_id": promotion_args["segment_id"],
                            "target_refs": list(promotion_args.get("target_refs") or []),
                        },
                    )
                    return (
                        "read_segment_detail",
                        promotion_args,
                        "repair_narration_locator_to_transcript_promotion",
                    )
            if not self._has_tool("verify_segment_anchors"):
                return None
            verify_args = self._latest_locator_verify_args(segment_id=str(args.get("segment_id") or ""))
            if verify_args:
                return (
                    "verify_segment_anchors",
                    verify_args,
                    "repair_repeated_locator_to_verify_segment_anchors",
                )
        if (
            active_skill is not None
            and active_skill.name
            in {"timeline_ordering", "narration_timeline_qa", "visual_timeline_qa", "grounded_factual_qa", "mutex_fact_qa"}
            and tool_name in {"zoom", "expand_window"}
            and self._has_tool("locate_targets_in_segment")
        ):
            repaired_args = {
                "segment_id": args.get("segment_id"),
                "targets": self._inherited_locator_targets(),
            }
            return "locate_targets_in_segment", repaired_args, f"repair_{tool_name}_to_locate_targets_in_segment"
        if (
            active_skill is not None
            and active_skill.name
            in {"timeline_ordering", "narration_timeline_qa", "visual_timeline_qa", "grounded_factual_qa", "mutex_fact_qa"}
            and _tool_is(tool_name, "read_segment")
            and self._has_tool("read_segment_detail")
        ):
            repaired_args = dict(args)
            if (
                self._exploration_target_entities
                and self._tool_accepts_argument("read_segment_detail", "targets")
                and not repaired_args.get("targets")
            ):
                repaired_args["targets"] = list(self._exploration_target_entities)
            return "read_segment_detail", repaired_args, "repair_read_segment_to_read_segment_detail"
        if (
            active_skill is not None
            and active_skill.name == "mutex_fact_qa"
            and _tool_is(tool_name, "inspect_segment")
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
            and active_skill.name in {"timeline_ordering", "narration_timeline_qa", "visual_timeline_qa"}
            and (
                _tool_is(tool_name, "caption_segments")
                or (_tool_is(tool_name, "caption_segment") and args.get("segment_ids") and not args.get("segment_id"))
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

    def _inherited_locator_targets(self) -> list[str]:
        if self._exploration_target_entities:
            return list(self._exploration_target_entities)
        for observation in reversed(self.workspace.read_observations(tool_name="target_coverage")):
            coverage = observation.raw_output.get("coverage", [])
            if not isinstance(coverage, Sequence) or isinstance(coverage, (str, bytes)):
                continue
            targets = [
                str(row.get("target", "")).strip()
                for row in coverage
                if isinstance(row, Mapping) and str(row.get("target", "")).strip()
            ]
            if targets:
                return targets
        return []

    def _latest_locator_verify_args(self, *, segment_id: str = "") -> dict[str, Any]:
        requested = str(segment_id or "").strip()
        for observation in reversed(self.workspace.read_observations(tool_name="locate_targets_in_segment")):
            raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
            verify_args = raw_output.get("verify_call_args")
            if not isinstance(verify_args, Mapping) or not verify_args:
                continue
            located_segment = str(verify_args.get("segment_id") or raw_output.get("segment_id") or "").strip()
            if requested and located_segment and located_segment != requested:
                continue
            anchors = verify_args.get("anchors")
            if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)) or not anchors:
                continue
            return dict(verify_args)
        return {}

    def _latest_locator_recommended_action(self, *, segment_id: str = "", route_kind: str = "") -> dict[str, Any]:
        requested_segment = str(segment_id or "").strip()
        requested_route = str(route_kind or "").strip()
        for observation in reversed(self.workspace.read_observations(tool_name="locate_targets_in_segment")):
            raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
            actions = raw_output.get("recommended_next_actions")
            if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
                continue
            for action in actions:
                if not isinstance(action, Mapping):
                    continue
                if requested_route and str(action.get("route_kind") or "") != requested_route:
                    continue
                tool_name = str(action.get("tool") or "")
                if not tool_name or not self._has_tool(tool_name):
                    continue
                args = action.get("args")
                if not isinstance(args, Mapping):
                    continue
                located_segment = str(args.get("segment_id") or raw_output.get("segment_id") or "").strip()
                if requested_segment and located_segment and located_segment != requested_segment:
                    continue
                candidate_id = str(action.get("candidate_id") or observation.observation_id).strip()
                return {
                    "candidate_id": candidate_id,
                    "route_kind": str(action.get("route_kind") or ""),
                    "tool": tool_name,
                    "args": dict(args),
                    "target_refs": list(action.get("target_refs") or []),
                }
        return {}

    def _narration_transcript_promotion_args(
        self,
        *,
        segment_id: str,
        original_args: Mapping[str, Any],
    ) -> dict[str, Any]:
        resolved_segment = str(segment_id or original_args.get("segment_id") or "").strip()
        if not resolved_segment:
            return {}
        registry = getattr(self.workspace, "target_registry", None)
        if registry is None:
            return {}
        requested_refs = [
            str(ref).strip()
            for ref in _coerce_target_arg_list(original_args.get("target_refs"))
            if str(ref).strip()
        ]
        target_refs = [
            ref
            for ref in requested_refs
            if getattr(registry, "known_target_ref", lambda _ref: False)(ref)
        ]
        if not target_refs:
            target_refs = sorted(
                (str(ref) for ref in registry.targets_by_id),
                key=lambda ref: int(ref[1:]) if ref.startswith("T") and ref[1:].isdigit() else 10**9,
            )
        if not target_refs:
            return {}
        return {
            "segment_id": resolved_segment,
            "target_refs": target_refs,
            "promote_answer_evidence": True,
        }

    def _safe_parse_error_recovery_program(self, *, run_state: RunState) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for observation in self.workspace.read_observations():
            raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
            actions = raw_output.get("recommended_next_actions")
            if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
                continue
            for action in actions:
                if not isinstance(action, Mapping):
                    continue
                route_kind = str(action.get("route_kind") or "")
                if route_kind not in {"focused_ordered_list_vision", "narration_transcript_promotion"}:
                    continue
                tool_name = str(action.get("tool") or "")
                if not tool_name or not self._has_tool(tool_name):
                    continue
                args = action.get("args")
                if not isinstance(args, Mapping):
                    continue
                candidate_id = str(action.get("candidate_id") or observation.observation_id).strip()
                action_key = f"{route_kind}:{candidate_id}"
                if action_key in run_state.executed_recommended_action_ids:
                    continue
                candidates.append(
                    {
                        "tool": tool_name,
                        "args": dict(args),
                        "route_kind": route_kind,
                        "candidate_id": candidate_id,
                    }
                )
        if len(candidates) != 1:
            return []
        selected = candidates[0]
        self.workspace.write_trace_event(
            "pending_action_selected",
            {
                "route_kind": selected["route_kind"],
                "candidate_id": selected["candidate_id"],
                "tool": selected["tool"],
                "args": dict(selected.get("args") or {}),
            },
        )
        return [selected]

    def _write_recovery_execution_traces(
        self,
        *,
        round_number: int,
        program: Sequence[Mapping[str, Any]],
        observation_ids: Sequence[str],
        run_state: RunState,
    ) -> None:
        for step, observation_id in zip(program, observation_ids):
            if not isinstance(step, Mapping):
                continue
            route_kind = str(step.get("route_kind") or "")
            if not route_kind:
                continue
            candidate_id = str(step.get("candidate_id") or "")
            action_key = f"{route_kind}:{candidate_id}"
            if candidate_id:
                run_state.executed_recommended_action_ids.add(action_key)
            self.workspace.write_trace_event(
                "pending_action_execution_started",
                {
                    "round": round_number,
                    "route_kind": route_kind,
                    "candidate_id": candidate_id,
                    "tool": str(step.get("tool") or ""),
                },
            )
            self.workspace.write_trace_event(
                "recovery_executed",
                {
                    "round": round_number,
                    "route_kind": route_kind,
                    "candidate_id": candidate_id,
                    "tool": str(step.get("tool") or ""),
                    "observation_id": str(observation_id),
                },
            )
            self.workspace.write_trace_event(
                "pending_action_executed",
                {
                    "round": round_number,
                    "route_kind": route_kind,
                    "candidate_id": candidate_id,
                    "tool": str(step.get("tool") or ""),
                    "observation_id": str(observation_id),
                },
            )
            if route_kind == "focused_ordered_list_vision":
                self.workspace.write_trace_event(
                    "focused_ordered_list_vision_executed",
                    {
                        "round": round_number,
                        "candidate_id": candidate_id,
                        "observation_id": str(observation_id),
                        "args": dict(step.get("args") or {}),
                    },
                )
            elif route_kind == "narration_transcript_promotion":
                self.workspace.write_trace_event(
                    "narration_transcript_promotion_executed",
                    {
                        "round": round_number,
                        "candidate_id": candidate_id,
                        "observation_id": str(observation_id),
                        "args": dict(step.get("args") or {}),
                    },
                )

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
        return [{"tool": tool_name, "args": args, "assign": f"required_{fallback_segment_id}"}]

    def _fallback_visual_tool_name(self) -> str | None:
        for tool_name in ["inspect_segment", "vision_read", "caption_segment", "qa_segment"]:
            if self._has_tool(tool_name):
                return tool_name
        return None

    def _generic_forced_visual_skip_reason(
        self,
        *,
        question: str,
        planner_skill: SkillSpec | None,
        run_state: RunState,
    ) -> str:
        if planner_skill is not None and planner_skill.name == "narration_timeline_qa":
            return "narration_transcript_route"
        if self._has_pending_candidate_specific_action(run_state=run_state):
            return "pending_candidate_specific_action"
        return "silent_forced_visual_disabled"

    def _has_pending_candidate_specific_action(self, *, run_state: RunState) -> bool:
        for observation in self.workspace.read_observations():
            raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
            actions = raw_output.get("recommended_next_actions")
            if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
                continue
            for action in actions:
                if not isinstance(action, Mapping):
                    continue
                route_kind = str(action.get("route_kind") or "")
                if route_kind not in {
                    "ordered_list_transcript_complete",
                    "focused_ordered_list_vision",
                    "narration_transcript_promotion",
                }:
                    continue
                candidate_id = str(action.get("candidate_id") or observation.observation_id).strip()
                action_key = f"{route_kind}:{candidate_id}"
                if action_key not in run_state.executed_recommended_action_ids:
                    return True
        return False

    def _fallback_visual_evidence_program(
        self,
        *,
        question: str,
        video_path: str,
        inspected_segment_ids: set[str],
        planner_skill: SkillSpec | None,
    ) -> list[dict[str, Any]]:
        segment_id = self._resolve_next_segment_id("", inspected_segment_ids)
        if segment_id is None:
            return []
        tool_name = self._fallback_visual_tool_name_for_skill(planner_skill)
        if tool_name is None:
            return []
        segment = self.scene_index.get(segment_id)
        args: dict[str, Any] = {
            "video_path": video_path,
            "segment_id": segment.segment_id,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "nframes": self.budget.default_nframes,
        }
        target_question = _local_fact_question(
            question=question,
            planner_skill=planner_skill,
            target_entities=self._exploration_target_entities,
        )
        if _tool_is(tool_name, "vision_read"):
            args["ask_for"] = target_question
            args["event_label"] = target_question
        else:
            args["question"] = target_question
        return [{"tool": tool_name, "args": args, "assign": f"forced_visual_{segment.segment_id}"}]

    def _visual_evidence_from_navigation_program(
        self,
        *,
        program: Sequence[Mapping[str, Any]],
        question: str,
        video_path: str,
        planner_skill: SkillSpec | None,
    ) -> list[dict[str, Any]]:
        tool_name = "vision_read" if self._has_tool("vision_read") else self._fallback_visual_tool_name_for_skill(planner_skill)
        if tool_name is None:
            return []
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
            media_args: dict[str, Any] = {
                "video_path": video_path,
                "segment_id": segment.segment_id,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "nframes": self.budget.default_nframes,
            }
            target_question = _local_fact_question(
                question=question,
                planner_skill=planner_skill,
                target_entities=self._exploration_target_entities,
            )
            if _tool_is(tool_name, "vision_read"):
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
        return forced

    def _fallback_visual_tool_name_for_skill(self, planner_skill: SkillSpec | None) -> str | None:
        if planner_skill is not None:
            if planner_skill.name in {"timeline_ordering", "narration_timeline_qa", "visual_timeline_qa"}:
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
            focused_segment = _focused_window_for_scene_segment(
                scene_segment,
                args=args,
                duration_sec=self.scene_index.duration_sec,
            )
            if focused_segment is not None and tool_name in {"vision_read", "qa_segment", "inspect_segment", "verify_segment_anchors"}:
                return focused_segment
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
            self._record_invalid_segment_id(
                tool_name=tool_name,
                args=args,
                requested_segment_id=requested_segment_id,
                notes_out=notes_out,
            )
            return None
        try:
            start_sec, end_sec = _normalize_dynamic_window(
                start_sec=float(args["start_sec"]),
                end_sec=float(args["end_sec"]),
                duration_sec=self.scene_index.duration_sec,
                label=f"for {requested_segment_id}",
            )
        except (TypeError, ValueError):
            self._record_invalid_segment_id(
                tool_name=tool_name,
                args=args,
                requested_segment_id=requested_segment_id,
                notes_out=notes_out,
            )
            return None
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

    def _runtime_context(
        self,
        *,
        question: str = "",
        video_path: str = "",
        route: str = "",
        round_number: int = 0,
        run_state: RunState | None = None,
        round_state: RoundState | None = None,
        skill_runtime: Any | None = None,
        evidence_policy: Any | None = None,
    ) -> RunContext:
        resolved_route = route or classify_question_route(question)
        resolved_run_state = run_state or RunState(
            question=question,
            video_path=video_path or self.scene_index.video_path,
            question_route=resolved_route,
            raw_question=question,
            vlm_safe_question=question,
            effective_route=resolved_route,
        )
        return RunContext(
            workspace=self.workspace,
            scene_index=self.scene_index,
            budget=self.budget,
            run_state=resolved_run_state,
            round_state=round_state or RoundState(round_number=round_number),
            registry=self.registry,
            skill_runtime=skill_runtime,
            evidence_policy=evidence_policy,
            record_trace=self.workspace.write_trace_event,
            record_observation=None,
        )

    def _run_program(
        self,
        program: Sequence[Mapping[str, Any]],
        *,
        ctx: RunContext,
        question: str = "",
        video_path: str = "",
        route: str = "",
        round_number: int = 0,
        skill_runtime: Any | None = None,
        evidence_policy: Any | None = None,
        slots: Mapping[str, Any] | None = None,
    ):
        return self.runtime_host.run(
            program,
            ctx=ctx,
            slots=slots,
        )

    def _persist_planner_io(
        self,
        *,
        round_number: int,
        prompt: str,
        system_prompt: str = "",
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
        system_prompt_meta = None
        if system_prompt:
            system_prompt_meta = self.workspace.write_text_artifact(
                f"{prefix}_system_prompt.txt",
                system_prompt,
                max_chars=self.budget.planner_io_max_chars,
            )
        self.workspace.write_trace_event(
            "planner_io",
            {
                "round": round_number,
                "planner_input_mode": planner_input_mode,
                "prompt": prompt_meta,
                "system_prompt": system_prompt_meta,
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
        evidence_ids: Sequence[str] = (),
        source: str = "",
        status: str = "final",
        planner_answer: str = "",
        resolved_answer: str = "",
        conflict: bool | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "round": round_number,
            "answer": answer,
            "citations": list(citations),
        }
        if evidence_ids:
            payload["evidence_ids"] = list(evidence_ids)
        if source:
            payload["source"] = source
            payload["final_source"] = source
        if planner_answer:
            payload["planner_answer"] = planner_answer
        if resolved_answer:
            payload["resolved_answer"] = resolved_answer
        if conflict is not None:
            payload["conflict"] = bool(conflict)
        if status != "final":
            payload["status"] = status
        provenance = self._citation_provenance(_unique_preserving_order([*citations, *evidence_ids]))
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

    def _seed_target_coverage(self, question: str) -> None:
        if not extract_candidate_options(question):
            return
        if not self._has_tool("target_coverage"):
            return
        if self.workspace.observation_count(tool_name="target_coverage") > 0:
            return
        registry = getattr(self.workspace, "target_registry", None)
        if registry is None or not isinstance(getattr(registry, "targets_by_id", None), Mapping):
            return
        target_refs = sorted(str(ref) for ref in registry.targets_by_id)
        targets: list[str] = []
        if not target_refs and not targets:
            return
        coverage_args: dict[str, Any] = {"top_k": 3}
        if target_refs:
            coverage_args["target_refs"] = target_refs
            coverage_args["group_by_option"] = True
        else:
            coverage_args["targets"] = targets
        runtime_context = self._runtime_context(
            question=question,
            video_path=self.scene_index.video_path,
            route=classify_question_route(question),
            round_number=0,
        )
        try:
            result = self._run_program(
                [
                    {
                        "tool": "target_coverage",
                        "args": coverage_args,
                        "assign": "auto_target_coverage",
                    }
                ],
                ctx=runtime_context,
                question="",
                video_path=self.scene_index.video_path,
                route="",
            )
        except ToolError as exc:
            fallback_targets = [str(target).strip() for target in self._exploration_target_entities if str(target).strip()]
            if not fallback_targets:
                raise
            self.workspace.write_trace_event(
                "target_coverage_seed_fallback",
                {
                    "reason": "target_refs_unavailable_in_tool_workspace",
                    "error": str(exc),
                    "target_refs": target_refs,
                    "target_count": len(fallback_targets),
                },
            )
            coverage_args = {"targets": fallback_targets, "top_k": 3}
            fallback_context = self._runtime_context(
                question=question,
                video_path=self.scene_index.video_path,
                route=classify_question_route(question),
                round_number=0,
            )
            result = self._run_program(
                [
                    {
                        "tool": "target_coverage",
                        "args": coverage_args,
                        "assign": "auto_target_coverage",
                    }
                ],
                ctx=fallback_context,
                question="",
                video_path=self.scene_index.video_path,
                route="",
            )
        self.workspace.write_trace_event(
            "target_coverage_seeded",
            {
                "target_count": len(target_refs or targets),
                "target_refs": target_refs,
                "observation_ids": list(result.observation_ids),
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

    def _normalize_verify_segment_anchors_args(
        self,
        *,
        args: Mapping[str, Any],
        question: str,
        raw_question: str,
        video_path: str,
        planner_skill: SkillSpec | None,
        notes_out: list[NormalizationNote] | None,
    ) -> dict[str, Any] | None:
        normalized_args = dict(args)
        original_args = dict(args)
        anchor_segment_ids = _anchor_segment_ids(normalized_args.get("anchors", []))
        if len(anchor_segment_ids) > 1:
            next_action = (
                "verify_segment_anchors received anchors from multiple segments. "
                "Call it once per source segment using the exact verify_call_args from locate_targets_in_segment."
            )
            self.workspace.write_trace_event(
                "exploration_policy_adjustment",
                {
                    "reason": "invalid_verify_anchor_segment_mismatch",
                    "tool": "verify_segment_anchors",
                    "anchor_segment_ids": anchor_segment_ids,
                    "next_action": next_action,
                },
            )
            _append_normalization_note(
                notes_out,
                tool="verify_segment_anchors",
                reason="invalid_verify_anchor_segment_mismatch",
                original={"tool": "verify_segment_anchors", "args": original_args},
                next_action=next_action,
            )
            return None

        requested_segment_id = str(normalized_args.get("segment_id", "") or "").strip()
        if anchor_segment_ids:
            anchor_segment_id = anchor_segment_ids[0]
            if requested_segment_id and requested_segment_id != anchor_segment_id:
                self.workspace.write_trace_event(
                    "route_tool_repaired",
                    {
                        "skill": planner_skill.name if planner_skill is not None else "",
                        "requested_tool": "verify_segment_anchors",
                        "resolved_tool": "verify_segment_anchors",
                        "reason": "repair_verify_anchor_segment_id_from_anchor",
                        "requested_segment_id": requested_segment_id,
                        "anchor_segment_id": anchor_segment_id,
                    },
                )
                _append_normalization_note(
                    notes_out,
                    tool="verify_segment_anchors",
                    reason="repair_verify_anchor_segment_id_from_anchor",
                    original={"tool": "verify_segment_anchors", "args": original_args},
                    resolved={"tool": "verify_segment_anchors", "segment_id": anchor_segment_id},
                )
            normalized_args["segment_id"] = anchor_segment_id
        elif not requested_segment_id:
            next_action = (
                "verify_segment_anchors needs anchors with segment_id or an explicit segment_id. "
                "Use locate_targets_in_segment verify_call_args."
            )
            _append_normalization_note(
                notes_out,
                tool="verify_segment_anchors",
                reason="missing_verify_anchor_segment_id",
                original={"tool": "verify_segment_anchors", "args": original_args},
                next_action=next_action,
            )
            return None

        segment_id = str(normalized_args.get("segment_id", "") or "").strip()
        scene_segment = _scene_segment_or_none(self.scene_index, segment_id)
        start_sec, end_sec = _anchor_args_window(
            normalized_args.get("anchors", []),
            fallback_start=float(normalized_args.get("start_sec", 0.0) or 0.0),
            fallback_end=float(normalized_args.get("end_sec", 0.0) or 0.0),
        )
        if scene_segment is not None:
            normalized_args["start_sec"] = scene_segment.start_sec
            normalized_args["end_sec"] = scene_segment.end_sec
        elif start_sec or end_sec:
            normalized_args.setdefault("start_sec", start_sec)
            normalized_args.setdefault("end_sec", end_sec)
        if self._tool_accepts_argument("verify_segment_anchors", "video_path"):
            normalized_args["video_path"] = video_path
        if self._tool_accepts_argument("verify_segment_anchors", "question"):
            normalized_args.setdefault("question", question)
            normalized_args["question"] = _tool_exploration_question(
                str(normalized_args["question"]),
                route_hint=planner_skill.name if planner_skill else "",
                question_context=question,
                forbidden_question=raw_question,
                option_blind=self.budget.rewrite_mcq_for_exploration,
                target_entities=self._exploration_target_entities,
            )
        if self._tool_accepts_argument("verify_segment_anchors", "nframes"):
            normalized_args.setdefault("nframes", self.budget.default_nframes)
        return normalized_args

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
        remaining_rounds: int | None = None,
        supported_binding_no_growth_rounds: int = 0,
        runtime_context: RunContext | None = None,
    ) -> IterativeRunResult | None:
        if answer_result.status != "need_more_evidence" or not answer_result.has_partial_support():
            return None
        low_confidence = answer_result.as_low_confidence_final()
        if low_confidence.status != "low_confidence_final":
            return None
        auto_promotion_guard_active = (
            supported_binding_no_growth_rounds >= 5
            and remaining_rounds is not None
            and remaining_rounds <= 3
        )
        if auto_promotion_guard_active and self._try_auto_evidence_promotion(
            answer=low_confidence.answer,
            question=question,
            round_number=round_number,
            source=source,
            runtime_context=runtime_context,
        ):
            return None
        if source not in _ANSWER_AGENT_AUTO_FINAL_SOURCES and not auto_promotion_guard_active:
            self.workspace.write_trace_event(
                "iterative_answer_suggestion",
                {
                    "round": round_number,
                    "source": source,
                    "answer": low_confidence.answer,
                    "citations": list(low_confidence.citations),
                    "confidence": low_confidence.confidence,
                    "status": low_confidence.status,
                    "recommended_to_planner": True,
                },
            )
            return None
        if not auto_promotion_guard_active and not _has_answer_grade_citation(
            workspace=self.workspace,
            question=question,
            answer=low_confidence.answer,
            citations=low_confidence.citations,
        ):
            self.workspace.write_trace_event(
                "low_confidence_final_blocked",
                {
                    "round": round_number,
                    "source": source,
                    "reason": "final_requires_answer_grade_evidence",
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

    def _try_auto_evidence_promotion(
        self,
        *,
        answer: str,
        question: str,
        round_number: int,
        source: str,
        runtime_context: RunContext | None = None,
    ) -> bool:
        if not self._has_tool("bind_asr_claim"):
            self.workspace.write_trace_event(
                "auto_evidence_promotion_attempted",
                {"round": round_number, "source": source, "answer": answer, "succeeded": False, "reason": "tool_unavailable"},
            )
            return False
        target_refs = _target_refs_for_answer(workspace=self.workspace, answer=answer)
        if not target_refs:
            self.workspace.write_trace_event(
                "auto_evidence_promotion_attempted",
                {"round": round_number, "source": source, "answer": answer, "succeeded": False, "reason": "no_target_refs"},
            )
            return False
        runtime_context = runtime_context or self._runtime_context(
            question=question,
            video_path=self.scene_index.video_path,
            route=classify_question_route(question),
            round_number=round_number,
        )
        run_state = runtime_context.run_state
        candidate = _latest_asr_binding_candidates(
            workspace=self.workspace,
            target_refs=target_refs,
            failed_call_signatures=run_state.zero_yield_tool_signatures,
            limit=1,
        )
        if not candidate:
            self.workspace.write_trace_event(
                "auto_evidence_promotion_attempted",
                {
                    "round": round_number,
                    "source": source,
                    "answer": answer,
                    "target_refs": target_refs,
                    "succeeded": False,
                    "reason": "no_coverage_candidate",
                },
            )
            return False
        segment_id = str(candidate[0].get("segment_id", "") or "").strip()
        candidate_target_refs = [
            str(ref).strip()
            for ref in candidate[0].get("target_refs", target_refs)
            if str(ref).strip()
        ]
        key = (str(_answer_option_letter(answer) or answer), segment_id, tuple(candidate_target_refs))
        if key in run_state.auto_evidence_promotion_attempted_keys:
            self.workspace.write_trace_event(
                "auto_evidence_promotion_attempted",
                {
                    "round": round_number,
                    "source": source,
                    "answer": answer,
                    "segment_id": segment_id,
                    "target_refs": candidate_target_refs,
                    "succeeded": False,
                    "reason": "already_attempted",
                },
            )
            return False
        run_state.auto_evidence_promotion_attempted_keys.add(key)
        before = self._supported_evidence_binding_count()
        try:
            result = self._run_program(
                [
                    {
                        "tool": "bind_asr_claim",
                        "args": {"segment_id": segment_id, "target_refs": candidate_target_refs},
                    }
                ],
                ctx=runtime_context,
                question=question,
                video_path=self.scene_index.video_path,
                route=classify_question_route(question),
            )
        except Exception as exc:
            self.workspace.write_trace_event(
                "auto_evidence_promotion_attempted",
                {
                    "round": round_number,
                    "source": source,
                    "answer": answer,
                    "segment_id": segment_id,
                    "target_refs": candidate_target_refs,
                    "succeeded": False,
                    "reason": f"tool_error:{type(exc).__name__}",
                },
            )
            return False
        after = self._supported_evidence_binding_count()
        succeeded = after > before
        self.workspace.write_trace_event(
            "auto_evidence_promotion_attempted",
            {
                "round": round_number,
                "source": source,
                "answer": answer,
                "segment_id": segment_id,
                "target_refs": candidate_target_refs,
                "observation_ids": list(result.observation_ids),
                "succeeded": succeeded,
                "reason": "supported_binding_created" if succeeded else "no_supported_binding_created",
            },
        )
        return succeeded

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
        runtime_context = self._runtime_context(
            question=question,
            video_path=video_path,
            route="gist_global",
            round_number=0,
        )
        result = self._run_program(
            [first_step],
            ctx=runtime_context,
            question=question,
            video_path=video_path,
            route="gist_global",
        )
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
        route: str = "",
        recommended_skill_id: str = "",
    ) -> IterativeRunResult | None:
        skill = _recommended_effective_skill(
            question,
            route=route or None,
            recommended_skill_id=recommended_skill_id,
        )
        if skill.name in {"timeline_ordering", "visual_timeline_qa", "narration_timeline_qa"}:
            return None
        if skill.name not in {"grounded_factual_qa", "mutex_fact_qa"}:
            return None
        if not self._has_tool("ground_question") or not self._has_tool("vision_read"):
            return None

        target_facts = list(_target_entities_from_registry(getattr(self.workspace, "target_registry", None)))
        target_specs: list[SkillTargetFact] = []
        if not target_facts:
            target_specs = _skill_target_fact_specs(question=exploration_question, skill_name=skill.name)
            if not target_specs:
                target_specs = _skill_target_fact_specs(question=question, skill_name=skill.name)
            target_facts = [spec.fact for spec in target_specs]
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
        if skill.name == "timeline_ordering":
            return self._try_timeline_ordering_route(
                question=question,
                video_path=video_path,
                target_facts=target_facts,
            )

        rounds: list[IterativeRound] = []
        all_observation_ids: list[str] = []
        target_budget_per_round = max(1, self.budget.max_tool_calls_per_round // 2)
        scheduler = FollowupScheduler(
            FollowupBudget(
                global_max_followups=max(1, self.budget.max_rounds * target_budget_per_round)
            )
        )
        scheduler.enqueue(
            [
                FollowupTarget(
                    target_id=f"fu_{self.workspace.run_id}_{index:04d}",
                    query=target_fact,
                    event_label=target_fact,
                    route=_followup_route_for_skill(skill.name),
                    reason="hard skill target fact still needs visual grounding",
                    priority=index,
                    attempt_count=0,
                    parent_missing_evidence_id=f"target_fact_{index:04d}",
                    mutex_group_id=target_specs[index - 1].mutex_group_id if index - 1 < len(target_specs) else "",
                )
                for index, target_fact in enumerate(target_facts, start=1)
            ]
        )
        last_answer_result = None
        last_failure_tag = "answer_agent_need_more_evidence"

        for round_number in range(1, self.budget.max_rounds + 1):
            targets = _next_followup_chunk(
                scheduler=scheduler,
                chunk_size=target_budget_per_round,
            )
            if not targets:
                break

            runtime_context = self._runtime_context(
                question=question,
                video_path=video_path,
                route=route or _followup_route_for_skill(skill.name),
                round_number=round_number,
            )
            program: list[dict[str, Any]] = []
            round_observation_ids: list[str] = []
            for target in targets:
                new_observation_ids = self._run_hard_skill_followup_target(
                    runtime_context=runtime_context,
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
                citations = [str(obs_id) for obs_id in timeline_decision["citations"]]
                self.workspace.write_trace_event(
                    "iterative_timeline_temporal_inference",
                    {
                        "round": round_number,
                        "answer": str(timeline_decision["answer"]),
                        "citations": citations,
                        "matched_events": list(timeline_decision["matched_events"]),
                        "source": "hard_skill_followup",
                        "planner_action": "hint_only",
                    },
                )

            table = self._answer_evidence_table(question)
            answer_result = AnswerAgent(self.backend).run(
                question=question,
                evidence_text=self._read_ledger(),
                evidence_table=table,
                **_answer_agent_operator_kwargs(workspace=self.workspace),
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
            if answer_result.status == "final" and not gate_reason and skill.name == "mutex_fact_qa" and scheduler.queue:
                gate_reason = "mutex_pending_targets"
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
                        **_answer_agent_trace_fields(answer_result),
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
                    **_answer_agent_trace_fields(answer_result),
                },
            )
            self.workspace.write_reflection_memory(
                route=route,
                failure_tag=last_failure_tag,
                rule=_reflection_rule_for_failure(last_failure_tag),
            )
            if round_number >= self.budget.max_rounds and not gate_reason:
                low_confidence_result = self._try_low_confidence_final(
                    answer_result=answer_result,
                    question=question,
                    video_path=video_path,
                    rounds=rounds,
                    round_number=round_number,
                    source="hard_skill_budget_exhausted",
                    program=program,
                    observation_ids=round_observation_ids,
                    runtime_context=runtime_context,
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
        runtime_context = self._runtime_context(
            question=question,
            video_path=video_path,
            route="timeline_ordering",
            round_number=1,
        )
        caption_result = self._run_program(caption_program, ctx=runtime_context)
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
        vision_result = self._run_program(vision_program, ctx=runtime_context) if vision_program else type(caption_result)([])
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
            citations = [str(obs_id) for obs_id in timeline_decision["citations"]]
            self.workspace.write_trace_event(
                "iterative_timeline_temporal_inference",
                {
                    "round": 1,
                    "answer": str(timeline_decision["answer"]),
                    "citations": citations,
                    "matched_events": list(timeline_decision["matched_events"]),
                    "source": "timeline_ordering",
                    "planner_action": "hint_only",
                },
            )
            return IterativeRunResult(
                question=question,
                video_path=video_path,
                answer="need_more_evidence: timeline inference requires planner or AnswerAgent confirmation",
                status="need_more_evidence",
                citations=citations,
                confidence=0.0,
                rounds=[
                    IterativeRound(
                        round_number=1,
                        status="need_more_evidence",
                        planner_text="",
                        rationale=_timeline_decision_pending_inference(timeline_decision),
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
        runtime_context: RunContext,
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
        ground_result = self._run_program(ground_program, ctx=runtime_context)
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
        vision_result = self._run_program(vision_program, ctx=runtime_context)
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


def _target_entities_from_registry(registry: object) -> tuple[str, ...]:
    targets_by_id = getattr(registry, "targets_by_id", None)
    if not isinstance(targets_by_id, Mapping):
        return ()
    return tuple(
        str(target.canonical_text).strip()
        for target in targets_by_id.values()
        if str(getattr(target, "canonical_text", "")).strip()
    )


def _raw_options_by_id(options: Sequence[str]) -> dict[str, str]:
    raw_options: dict[str, str] = {}
    for option in options:
        match = re.match(r"\s*([A-H])[\).:-]\s*(.*)\s*$", str(option or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        raw_options[match.group(1).upper()] = " ".join(match.group(2).split()).strip()
    return raw_options


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
        if step.get("tool") not in _SEGMENT_ID_TOOLS:
            continue
        args = step.get("args", {})
        if isinstance(args, Mapping) and args.get("segment_id"):
            segment_ids.append(str(args["segment_id"]))
    return segment_ids


def _anchor_segment_ids(anchors: Any) -> list[str]:
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        return []
    segment_ids: list[str] = []
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        segment_id = str(anchor.get("segment_id", "") or "").strip()
        if segment_id and segment_id not in segment_ids:
            segment_ids.append(segment_id)
    return segment_ids


def _anchor_args_window(anchors: Any, *, fallback_start: float, fallback_end: float) -> tuple[float, float]:
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        return float(fallback_start), float(fallback_end)
    starts = []
    ends = []
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        if anchor.get("start_sec") is not None:
            starts.append(float(anchor.get("start_sec", 0.0) or 0.0))
        if anchor.get("end_sec") is not None:
            ends.append(float(anchor.get("end_sec", 0.0) or 0.0))
    start_sec = min(starts) if starts else float(fallback_start)
    end_sec = max(ends) if ends else float(fallback_end)
    if end_sec < start_sec:
        end_sec = start_sec
    return start_sec, end_sec


def _followup_route_for_skill(skill_name: str) -> FollowupRoute:
    if _is_timeline_skill(skill_name):
        return "temporal_order"
    if skill_name in {"gist_qa", "main_idea"}:
        return "gist_global"
    return "needle_local"


def _is_timeline_skill(skill_name: str) -> bool:
    return str(skill_name) in {
        "timeline_ordering",
        "temporal_ordering",
        "narration_timeline_qa",
        "visual_timeline_qa",
    }


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


def _append_program_steps(
    program: Sequence[Mapping[str, Any]],
    extra_steps: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
) -> list[Mapping[str, Any]]:
    merged = [dict(step) for step in program]
    seen = {_program_step_key(step) for step in merged}
    for step in extra_steps:
        if len(merged) >= max_steps:
            break
        key = _program_step_key(step)
        if key in seen:
            continue
        merged.append(dict(step))
        seen.add(key)
    return merged


def _program_step_key(step: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "tool": str(step.get("tool", "")),
            "args": step.get("args", {}),
        },
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )


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


def _local_fact_question(
    *,
    question: str,
    planner_skill: SkillSpec | None,
    target_entities: Sequence[str] = (),
) -> str:
    semantic = _semantic_question_text(question).strip() or question
    if planner_skill is None:
        return semantic
    fact_only_instruction = (
        "Do not choose an option."
        if extract_candidate_options(question)
        else "Report facts only."
    )
    if planner_skill.name in {"timeline_ordering", "narration_timeline_qa", "visual_timeline_qa"}:
        return _append_target_attention_block(
            "Openly describe this segment's actual visible artworks, objects, people, scene changes, onscreen text, "
            "and narrated events in presentation order. Include timestamps if possible. "
            "Focus on concrete observations rather than conclusions. Do not choose an option.",
            target_entities=target_entities,
            timeline=True,
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
    has_answer_grade_evidence = _has_answer_grade_citation(
        workspace=workspace,
        question=question,
        answer=answer,
        citations=citations,
    )
    if extract_candidate_options(question) and not has_inspect_with_candidate_options and not has_answer_grade_evidence:
        return "mcq_final_requires_local_visual_read"
    if not has_answer_grade_evidence:
        return "final_requires_answer_grade_evidence"
    single_scene_reason = _single_scene_subwindow_final_reason(
        question=question,
        workspace=workspace,
        citations=citations,
    )
    if single_scene_reason:
        return single_scene_reason
    return ""


def _single_scene_subwindow_final_reason(
    *,
    question: str,
    workspace: EvidenceWorkspace,
    citations: Sequence[str],
) -> str:
    if not re.search(r"\b(single scene|one scene|same scene|single shot|one shot)\b", question, flags=re.IGNORECASE):
        return ""
    targets = _temporal_events_from_question(question, max_events=8)
    if len(targets) < 2:
        return ""
    cited = {str(citation) for citation in citations if str(citation)}
    for observation in workspace.read_observations():
        if cited and observation.observation_id not in cited:
            continue
        if observation.tool not in {"verify_segment_anchors", "vision_read"}:
            continue
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        if not _observation_has_short_single_scene_window(raw_output):
            continue
        observed_targets = _observation_confirmed_targets(raw_output)
        if _targets_cover_expected(observed_targets, targets):
            return ""
    return "single_scene_subwindow_vision_read_missing"


def _observation_has_short_single_scene_window(raw_output: Mapping[str, Any], *, max_window_sec: float = 60.0) -> bool:
    verify_windows = raw_output.get("verify_windows", [])
    if isinstance(verify_windows, Sequence) and not isinstance(verify_windows, (str, bytes)):
        for window in verify_windows:
            if isinstance(window, Mapping) and _window_duration(window.get("start_sec"), window.get("end_sec")) < max_window_sec:
                return True
    return _window_duration(raw_output.get("start_sec"), raw_output.get("end_sec")) < max_window_sec


def _window_duration(start_value: Any, end_value: Any) -> float:
    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError):
        return float("inf")
    return max(0.0, end - start)


def _observation_confirmed_targets(raw_output: Mapping[str, Any]) -> list[str]:
    targets: list[str] = []
    for key in ("ordered_visible_in_window", "ordered_visible"):
        values = raw_output.get(key, [])
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            targets.extend(str(value).strip() for value in values if str(value).strip())
    confirmations = raw_output.get("confirmations", [])
    if isinstance(confirmations, Sequence) and not isinstance(confirmations, (str, bytes)):
        for confirmation in confirmations:
            if isinstance(confirmation, Mapping):
                target = str(confirmation.get("target", "")).strip()
                if target:
                    targets.append(target)
    for key in ("event_label", "entity"):
        value = str(raw_output.get(key, "")).strip()
        if value:
            targets.append(value)
    return targets


def _targets_cover_expected(observed_targets: Sequence[str], expected_targets: Sequence[str]) -> bool:
    observed_keys = [_target_fact_key(target) for target in observed_targets if _target_fact_key(target)]
    for expected in expected_targets:
        expected_key = _target_fact_key(expected)
        if not expected_key:
            continue
        expected_tokens = set(expected_key.split())
        if not any(expected_tokens.issubset(set(observed_key.split())) or set(observed_key.split()).issubset(expected_tokens) for observed_key in observed_keys):
            return False
    return True


def _blocked_planner_final_reason(
    *,
    question: str,
    has_inspect_with_candidate_options: bool,
    workspace: EvidenceWorkspace,
    answer: str,
    citations: Sequence[str],
    evidence_ids: Sequence[str] = (),
    planner_skill: SkillSpec | None = None,
) -> str:
    final_evidence_refs = _unique_preserving_order([*citations, *evidence_ids])
    if _prefinal_repair_kind(planner_skill) is PrefinalRepairKind.NARRATION_TIMELINE:
        if not _has_supported_evidence_binding_id(workspace, evidence_ids):
            return "planner_final_requires_supported_evidence_id"
    gate_decision = _structured_final_gate_decision(
        workspace=workspace,
        question=question,
        answer=answer,
        final_refs=final_evidence_refs,
        planner_skill=planner_skill,
    )
    if gate_decision is not None:
        workspace.write_trace_event(
            "structured_final_gate",
            {
                "proposed_option": gate_decision.proposed_option,
                "gate_status": gate_decision.gate_status,
                "reason_code": gate_decision.reason_code or "",
                "answer_operator": str(getattr(getattr(workspace, "grounding_runtime", None), "answer_operator", "") or "select_present"),
                "diagnostic_repair_hint": str(getattr(gate_decision, "diagnostic_repair_hint", "") or ""),
                "supporting_evidence_ids": list(gate_decision.supporting_evidence_ids),
                "missing_target_refs": list(gate_decision.missing_target_refs),
                "missing_relation_refs": list(gate_decision.missing_relation_refs),
            },
        )
        if gate_decision.accepted:
            return ""
        hint = str(getattr(gate_decision, "diagnostic_repair_hint", "") or "")
        suffix = f":{hint}" if hint else ""
        return "final_gate:" + str(gate_decision.reason_code or "verifier_failed") + suffix
    base_reason = _blocked_final_reason(
        question=question,
        has_inspect_with_candidate_options=has_inspect_with_candidate_options,
        workspace=workspace,
        answer=answer,
        citations=final_evidence_refs,
    )
    if base_reason:
        return base_reason
    options = extract_candidate_options(question)
    if not options:
        return ""
    if planner_skill is None and classify_question_route(question) != "gist_global":
        return ""
    if _prefinal_repair_kind(planner_skill) is PrefinalRepairKind.NARRATION_TIMELINE:
        relation_reason = _missing_required_relations_final_reason(
            workspace=workspace,
            question=question,
            answer=answer,
            final_refs=final_evidence_refs,
        )
        if relation_reason:
            return relation_reason
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
        citations=final_evidence_refs,
    )


def _structured_final_gate_decision(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
    final_refs: Sequence[str],
    planner_skill: SkillSpec | None,
) -> Any | None:
    registry = getattr(workspace, "target_registry", None)
    if registry is None:
        return None
    selected_option = _answer_option_letter(answer)
    options_by_id = getattr(registry, "options_by_id", {})
    if not selected_option or not isinstance(options_by_id, Mapping) or selected_option not in options_by_id:
        return None
    policy_name = _final_gate_policy_name(question=question, planner_skill=planner_skill)
    if not policy_name:
        return None
    table = workspace.evidence_table_v2(
        question=question,
        options=extract_candidate_options(question),
        include_legacy_worker_votes=True,
    )
    rows = _final_gate_rows_for_option(table=table, selected_option=selected_option, final_refs=final_refs)
    grounding_runtime = getattr(workspace, "grounding_runtime", None)
    try:
        return evaluate_final_candidate(
            selected_option=selected_option,
            registry=registry,
            evidence_bindings=_final_gate_evidence_bindings(rows=rows, selected_option=selected_option),
            relation_bindings=_final_gate_relation_bindings(rows=rows),
            skill_name=policy_name,
            option_evaluations=_final_gate_option_evaluations(table),
            central_subjects=tuple(getattr(grounding_runtime, "central_subjects", ()) or ()),
            answer_operator=str(getattr(grounding_runtime, "answer_operator", "") or "select_present"),
        )
    except KeyError:
        return None


def _final_gate_policy_name(*, question: str, planner_skill: SkillSpec | None) -> str:
    policy_names = {
        "main_idea",
        "visual_timeline_qa",
        "narration_timeline_qa",
        "mixed_asr_visual_qa",
        "grounded_factual_qa",
        "mutex_fact_qa",
    }
    skill_name = str(getattr(planner_skill, "name", "") or "").strip()
    if skill_name in policy_names:
        return skill_name
    if skill_name == "timeline_ordering":
        return "narration_timeline_qa" if classify_narration_subroute(question) == "narration_timeline" else "visual_timeline_qa"
    route = classify_question_route(question)
    if route == "gist_global":
        return "main_idea"
    if route == "mixed_asr_visual":
        return "mixed_asr_visual_qa"
    if route == "needle_local":
        return "grounded_factual_qa"
    if route == "temporal_order":
        return "narration_timeline_qa" if classify_narration_subroute(question) == "narration_timeline" else "visual_timeline_qa"
    return ""


def _final_gate_rows_for_option(
    *,
    table: Mapping[str, Any],
    selected_option: str,
    final_refs: Sequence[str],
) -> list[Mapping[str, Any]]:
    refs = {str(ref).strip() for ref in final_refs if str(ref).strip()}
    rows = table.get("rows", [])
    if refs and isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [row for row in rows if isinstance(row, Mapping) and _evidence_row_matches_any_ref(row, refs)]
    groups = table.get("groups", {})
    if isinstance(groups, Mapping):
        grouped = groups.get(selected_option, [])
        if isinstance(grouped, Sequence) and not isinstance(grouped, (str, bytes)):
            return [row for row in grouped if isinstance(row, Mapping)]
    return []


def _final_gate_evidence_bindings(
    *,
    rows: Sequence[Mapping[str, Any]],
    selected_option: str,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for row in rows:
        binding = row.get("evidence_binding")
        if not isinstance(binding, Mapping):
            continue
        evidence_id = str(row.get("evidence_id", "") or binding.get("evidence_id", "") or row.get("obs_id", "")).strip()
        start, end = _row_time_bounds(row=row, binding=binding)
        bindings.append(
            {
                "evidence_id": evidence_id,
                "target_ref": binding.get("target_ref") or binding.get("target_id"),
                "relation_ref": binding.get("relation_ref") or binding.get("relation_id"),
                "option_id": row.get("supported_option") or binding.get("option_id") or selected_option,
                "modality": binding.get("modality") or binding.get("claim_modality") or row.get("grounding_quality") or row.get("tool"),
                "source": binding.get("source") or row.get("source") or row.get("tool"),
                "timestamp_start": start,
                "timestamp_end": end,
                "support_status": binding.get("support_status") or binding.get("status"),
                "confidence": row.get("confidence"),
                "rationale": row.get("claim", ""),
            }
        )
    return bindings


def _final_gate_relation_bindings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    relation_bindings: list[dict[str, Any]] = []
    for row in rows:
        binding = row.get("evidence_binding")
        if not isinstance(binding, Mapping):
            continue
        relations = binding.get("relation_bindings", [])
        if not isinstance(relations, Sequence) or isinstance(relations, (str, bytes)):
            continue
        evidence_id = str(row.get("evidence_id", "") or binding.get("evidence_id", "") or row.get("obs_id", "")).strip()
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            relation_bindings.append(
                {
                    "relation_ref": relation.get("relation_ref") or relation.get("relation_id"),
                    "ordered_target_refs": relation.get("ordered_target_refs") or relation.get("ordered_targets") or (),
                    "evidence_ids": relation.get("evidence_ids") or ([evidence_id] if evidence_id else []),
                    "support_status": relation.get("support_status") or relation.get("status"),
                    "timestamp_order": relation.get("timestamp_order") or (),
                    "modality": relation.get("modality") or relation.get("claim_modality") or binding.get("claim_modality"),
                    "source": relation.get("source") or binding.get("source") or row.get("tool"),
                }
            )
    return relation_bindings


def _final_gate_option_evaluations(table: Mapping[str, Any]) -> list[OptionEvaluation]:
    groups = table.get("groups", {})
    if not isinstance(groups, Mapping):
        return []
    evaluations: list[OptionEvaluation] = []
    for option_id, rows in groups.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        supporting_ids: list[str] = []
        target_refs: set[str] = set()
        conflict = False
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            binding = row.get("evidence_binding")
            if not isinstance(binding, Mapping):
                continue
            status = str(binding.get("support_status") or binding.get("status") or "").strip().lower()
            if status == "conflicting":
                conflict = True
            if status != "supported":
                continue
            evidence_id = str(row.get("evidence_id", "") or binding.get("evidence_id", "") or row.get("obs_id", "")).strip()
            if evidence_id:
                supporting_ids.append(evidence_id)
            target_ref = str(binding.get("target_ref") or binding.get("target_id") or "").strip()
            if target_ref:
                target_refs.add(target_ref)
        binding_status = "conflicting" if conflict else ("supported" if supporting_ids else ("partial" if rows else "unsupported"))
        evaluations.append(
            OptionEvaluation(
                option_id=str(option_id),
                binding_status=binding_status,
                rejection_reason=None,
                coverage_breadth=len(target_refs),
                supporting_evidence_ids=_unique_preserving_order(supporting_ids),
            )
        )
    return evaluations


def _row_time_bounds(*, row: Mapping[str, Any], binding: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = _float_or_none(row.get("t_start"))
    end = _float_or_none(row.get("t_end"))
    if start is None or end is None:
        time_range = row.get("time_range")
        if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
            start = start if start is not None else _float_or_none(time_range[0])
            end = end if end is not None else _float_or_none(time_range[1])
    if start is None:
        start = _float_or_none(binding.get("timestamp_start") or binding.get("mention_timestamp_sec"))
    if end is None:
        end = _float_or_none(binding.get("timestamp_end")) or start
    return start, end


def _final_rejection_reason_code(blocked_reason: str) -> str:
    text = str(blocked_reason or "")
    if text.startswith("final_gate:"):
        return _final_gate_feedback_parts(text)[0] or "verifier_failed"
    if "missing_required_relations" in text or "missing_required_relations" in text:
        return "missing_relation_binding"
    if "requires_supported_evidence_id" in text or "answer_grade" in text:
        return "no_answer_grade_citation"
    if "verifier_disagrees" in text:
        return "verifier_failed"
    return "verifier_failed"


def _final_gate_feedback_parts(blocked_reason: str) -> tuple[str, str]:
    text = str(blocked_reason or "")
    if not text.startswith("final_gate:"):
        return "", ""
    payload = text.split(":", 1)[1]
    reason, sep, hint = payload.partition(":")
    return reason or "verifier_failed", hint if sep else ""


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _missing_required_relations_final_reason(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
    final_refs: Sequence[str],
) -> str:
    registry = getattr(workspace, "target_registry", None)
    option_id = _answer_option_letter(answer)
    options_by_id = getattr(registry, "options_by_id", {}) if registry is not None else {}
    if not option_id or not isinstance(options_by_id, Mapping) or option_id not in options_by_id:
        return ""
    option = options_by_id[option_id]
    required = {str(relation_id) for relation_id in getattr(option, "required_relations", ()) if str(relation_id)}
    if not required:
        return ""
    supported = _supported_relation_ids_for_refs(
        workspace=workspace,
        question=question,
        final_refs=final_refs,
        selected_option=option_id,
    )
    missing = sorted(required - supported)
    if missing:
        workspace.write_trace_event(
            "narration_relation_chain_missing",
            {
                "answer": option_id,
                "required_relations": sorted(required),
                "supported_relations": sorted(supported),
                "missing_relations": missing,
                "final_refs": list(final_refs),
            },
        )
        return "planner_final_missing_required_relations:" + ",".join(missing)
    workspace.write_trace_event(
        "narration_relation_chain_complete",
        {
            "answer": option_id,
            "required_relations": sorted(required),
            "supported_relations": sorted(supported),
            "final_refs": list(final_refs),
        },
    )
    return ""


def _supported_relation_ids_for_refs(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    final_refs: Sequence[str],
    selected_option: str,
) -> set[str]:
    refs = {str(ref) for ref in final_refs if str(ref)}
    table = workspace.evidence_table_v2(
        question=question,
        options=extract_candidate_options(question),
        include_legacy_worker_votes=True,
    )
    groups = table.get("groups", {})
    if refs:
        candidate_rows = table.get("rows", [])
    elif isinstance(groups, Mapping):
        candidate_rows = groups.get(selected_option, [])
    else:
        candidate_rows = []
    if not isinstance(candidate_rows, Sequence) or isinstance(candidate_rows, (str, bytes)):
        return set()
    supported: set[str] = set()
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            continue
        if refs and not _evidence_row_matches_any_ref(row, refs):
            continue
        binding = row.get("evidence_binding")
        if not isinstance(binding, Mapping):
            continue
        if str(binding.get("status", "")).strip().lower() != "supported":
            continue
        relation_bindings = binding.get("relation_bindings", [])
        if not isinstance(relation_bindings, Sequence) or isinstance(relation_bindings, (str, bytes)):
            continue
        for relation in relation_bindings:
            if not isinstance(relation, Mapping):
                continue
            if str(relation.get("status", "")).strip().lower() != "supported":
                continue
            relation_id = str(relation.get("relation_id", "")).strip()
            if relation_id:
                supported.add(relation_id)
    return supported


def _evidence_row_matches_any_ref(row: Mapping[str, Any], refs: set[str]) -> bool:
    binding = row.get("evidence_binding")
    binding_id = ""
    if isinstance(binding, Mapping):
        binding_id = str(binding.get("evidence_id", "")).strip()
    row_refs = {
        str(row.get("obs_id", "")).strip(),
        str(row.get("observation_id", "")).strip(),
        str(row.get("evidence_id", "")).strip(),
        binding_id,
    }
    return bool(refs.intersection(ref for ref in row_refs if ref))


def _main_idea_indexed_coverage_supports_answer(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
) -> bool:
    return _indexed_transcript_supports_answer(workspace=workspace, question=question, answer=answer)


def _has_answer_grade_citation(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
    citations: Sequence[str],
) -> bool:
    return workspace.has_non_navigation_visual_citation(citations) or _indexed_transcript_supports_answer(
        workspace=workspace,
        question=question,
        answer=answer,
    )


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
        if str(row.get("tool", "")) in {
            "timeline_asr_summary",
            "asr_cue_detail",
            "transcript_evidence_binder",
        } and str(row.get("grounding_quality", "")) == "indexed_transcript":
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
        "final_requires_non_navigation_visual_evidence": "cite answer-grade visual, ASR, OCR, or QA evidence before finalizing",
        "final_requires_answer_grade_evidence": "cite answer-grade visual, ASR, OCR, or QA evidence before finalizing",
        "mcq_final_requires_local_visual_read": "localize a candidate and call vision_read or inspect_segment before finalizing MCQ answers",
        "answer_agent_need_more_evidence": "request targeted evidence when AnswerAgent abstains instead of forcing an option",
        "selected_option_has_structured_support": "map options only from structured visual facts with candidate_option_relations",
        "no_decisive_weak_grounding": "upgrade weak or inferred support to visually_confirmed evidence before finalizing",
        "no_unaddressed_conflict": "resolve stronger conflicting option support before finalizing",
        "temporal_order_requires_confirmed_event_timestamps": "confirm every event timestamp before comparing option sequence",
        "single_scene_subwindow_vision_read_missing": "for single-scene order questions, run verify_segment_anchors or vision_read on one <60s window covering all target items before finalizing",
        "grounding_quality_floor": "collect at least one visually_confirmed or indexed_transcript mapped evidence chain before finalizing",
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
    option_atoms = extract_option_target_atoms(options, max_targets=max_targets, include_synonyms=False)
    if option_atoms:
        return [
            SkillTargetFact(fact=target, mutex_group_id="option_fact_mutex")
            for target in option_atoms
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


def _projected_option_from_evidence_table(
    *,
    question: str,
    options: Sequence[str],
    table: Mapping[str, Any],
    target_registry: TargetRegistry | None,
    answer_operator: str = "",
) -> ProjectionResult:
    if target_registry is None:
        return ProjectionResult("unsupported", None, "none", 0.0, reason="missing_target_registry")
    evidence = _projection_evidence_from_table(table)
    if not evidence:
        return ProjectionResult("unsupported", None, "none", 0.0, reason="missing_target_evidence")
    route = classify_question_route(question)
    task = _projection_task_spec(
        question=question,
        options=options,
        route=route,
        target_registry=target_registry,
        answer_operator=answer_operator,
    )
    return project_option_support(task, evidence)


def _projection_task_spec(
    *,
    question: str,
    options: Sequence[str],
    route: str,
    target_registry: TargetRegistry,
    answer_operator: str = "",
) -> TaskSpec:
    fallback_task = build_task_spec(
        task_id="mcq_terminal_projection",
        question=question,
        options=options,
        route=route,
        target_registry=target_registry,
        answer_operator=answer_operator,
    )
    fallback_by_label = {option.label: option for option in fallback_task.options}
    registry_options = getattr(target_registry, "options_by_id", {})
    if not isinstance(registry_options, Mapping):
        registry_options = {}

    compiled: list[ProjectionOptionSpec] = []
    for index, option_text in enumerate(options):
        label = _answer_option_letter(str(option_text)) or _option_label_for_index(index)
        registry_option = registry_options.get(label)
        target_refs = tuple(
            str(ref).strip()
            for ref in getattr(registry_option, "target_sequence", ())
            if str(ref or "").strip()
        )
        if target_refs:
            compiled.append(
                _projection_option_from_registry_option(
                    label=label,
                    option_text=str(getattr(registry_option, "raw_option_text", "") or option_text),
                    target_refs=target_refs,
                    route=route,
                    option_kind=str(getattr(registry_option, "option_kind", "") or ""),
                )
            )
            continue
        compiled.append(
            fallback_by_label.get(label)
            or ProjectionOptionSpec(label=label, text=str(option_text), required_targets=())
        )

    return TaskSpec(
        task_id="mcq_terminal_projection",
        question=question,
        answer_format="mcq",
        route=route,
        options=tuple(compiled),
        target_registry=target_registry,
        answer_operator=answer_operator,
    )


def _projection_option_from_registry_option(
    *,
    label: str,
    option_text: str,
    target_refs: Sequence[str],
    route: str,
    option_kind: str,
) -> ProjectionOptionSpec:
    kind_text = str(option_kind or "").casefold()
    route_text = str(route or "").casefold()
    if any(marker in route_text for marker in ("order", "sequence", "timeline", "temporal")) or "sequence" in kind_text:
        return ProjectionOptionSpec(
            label=label,
            text=option_text,
            required_targets=tuple(target_refs),
            target_sequence=tuple(target_refs),
        )
    if any(marker in route_text for marker in ("main_idea", "gist", "synopsis", "summary")) or kind_text in {
        "theme",
        "main_idea",
        "summary",
        "gist",
    }:
        return ProjectionOptionSpec(
            label=label,
            text=option_text,
            required_targets=tuple(target_refs),
            theme_targets=tuple(target_refs),
        )
    return ProjectionOptionSpec(
        label=label,
        text=option_text,
        required_targets=tuple(target_refs),
    )


def _projection_evidence_from_table(table: Mapping[str, Any]) -> tuple[ProjectionEvidence, ...]:
    rows = table.get("rows", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    evidence: list[ProjectionEvidence] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        target_refs = _projection_target_refs_from_row(row)
        if not target_refs:
            continue
        start, end = _projection_time_bounds(row)
        evidence_id = str(row.get("obs_id") or row.get("evidence_id") or row.get("observation_id") or "")
        if not evidence_id:
            evidence_id = f"row_{len(evidence) + 1}"
        for target_ref in target_refs:
            evidence.append(
                ProjectionEvidence(
                    evidence_id=evidence_id,
                    target_ref=target_ref,
                    timestamp_start=start,
                    timestamp_end=end,
                    confidence=_float_or_none(row.get("confidence")),
                    segment_id=str(row.get("segment_id") or row.get("source_segment_id") or ""),
                    support_status=_projection_support_status(row),
                    modality=str(row.get("modality") or row.get("claim_modality") or row.get("grounding_quality") or ""),
                    source=str(row.get("tool") or row.get("source") or _projection_binding_value(row, "source") or ""),
                )
            )
    return tuple(evidence)


def _projection_target_refs_from_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    refs: list[str] = []
    candidates: list[Any] = [
        row.get("target_ref"),
        row.get("target_id"),
        row.get("event_label"),
        row.get("entity"),
        row.get("ordered_target_refs"),
        row.get("ordered_targets"),
    ]
    binding = row.get("evidence_binding")
    if isinstance(binding, Mapping):
        candidates.extend(
            [
                binding.get("target_ref"),
                binding.get("target_id"),
                binding.get("ordered_target_refs"),
                binding.get("ordered_targets"),
            ]
        )
    relations = row.get("candidate_option_relations")
    if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)):
        for relation in relations:
            if isinstance(relation, Mapping):
                relation_kind = str(relation.get("relation", "")).strip().lower()
                if relation_kind not in {"support", "supports", "supported"}:
                    continue
                candidates.extend([relation.get("target_ref"), relation.get("target_id")])
    for candidate in candidates:
        _append_projection_target_refs(refs, candidate)
    return tuple(dict.fromkeys(refs))


def _projection_binding_value(row: Mapping[str, Any], key: str) -> Any:
    binding = row.get("evidence_binding")
    if isinstance(binding, Mapping):
        return binding.get(key)
    return None


def _append_projection_target_refs(refs: list[str], candidate: Any) -> None:
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        for item in candidate:
            _append_projection_target_refs(refs, item)
        return
    text = str(candidate or "").strip()
    if _TARGET_REF_RE.fullmatch(text):
        refs.append(text)


def _projection_time_bounds(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = _float_or_none(row.get("t_start"))
    end = _float_or_none(row.get("t_end"))
    time_range = row.get("time_range")
    if (
        start is None
        and isinstance(time_range, Sequence)
        and not isinstance(time_range, (str, bytes))
        and len(time_range) >= 2
    ):
        start = _float_or_none(time_range[0])
        end = _float_or_none(time_range[1])
    if start is None:
        start = _float_or_none(row.get("observed_at_sec") or row.get("mention_timestamp_sec"))
    return start, end


def _projection_support_status(row: Mapping[str, Any]) -> str:
    raw_status = str(row.get("support_status") or row.get("status") or "").strip().lower()
    if raw_status in {"unsupported", "contradict", "contradicted", "negative"}:
        return "unsupported"
    confidence_signal = str(row.get("confidence_signal") or "").strip().lower()
    grounding = str(row.get("grounding_quality") or "").strip().lower()
    if confidence_signal == "unsupported" or grounding in {
        "weak",
        "inferred",
        "navigation_only",
        "external_knowledge",
    }:
        return "unsupported"
    return "supported"


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _option_label_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= index < len(alphabet):
        return alphabet[index]
    return str(index + 1)


def _best_effort_option_from_evidence_table(table: Mapping[str, Any]) -> str:
    groups = table.get("groups", {})
    if not isinstance(groups, Mapping):
        return ""
    scored: list[tuple[float, int, str]] = []
    for option, raw_rows in groups.items():
        option_letter = _answer_option_letter(str(option))
        if not option_letter or option_letter == "U":
            continue
        rows = list(raw_rows) if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)) else []
        score = 0.0
        support_count = 0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            quality = str(row.get("grounding_quality", ""))
            signal = str(row.get("confidence_signal", ""))
            if quality in {"visually_confirmed", "indexed_transcript", "global_sparse"} or signal == "asr_claim_binding_supported":
                score += float(row.get("confidence", 0.0) or 0.0)
                support_count += 1
        if support_count:
            scored.append((score, support_count, option_letter))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return scored[0][2]


def _top_option_from_evidence_table(table: Mapping[str, Any]) -> str:
    groups = table.get("groups", {})
    if not isinstance(groups, Mapping):
        return ""
    scored: list[tuple[float, int, str]] = []
    for option, raw_rows in groups.items():
        option_letter = _answer_option_letter(str(option))
        if not option_letter or option_letter == "U":
            continue
        rows = list(raw_rows) if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)) else []
        score = 0.0
        support_count = 0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            confidence = float(row.get("confidence", 0.0) or 0.0)
            if confidence <= 0 and not row.get("candidate_option_relations"):
                continue
            support_count += 1
            score += max(confidence, 0.05)
        if support_count:
            scored.append((score, support_count, option_letter))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return scored[0][2]


def _first_candidate_option_letter(options: Sequence[str]) -> str:
    for index, option in enumerate(options):
        letter = _answer_option_letter(str(option)) or chr(ord("A") + index)
        if letter:
            return letter
    return ""


def _latest_hypothesis_option(workspace: Any) -> str:
    try:
        text = str(workspace.read_hypothesis_text())
    except (AttributeError, OSError):
        return ""
    patterns = [
        r"\b(?:hypothesis|candidate|answer|selected|option)\s*(?:option|answer)?\s*[:=]\s*([A-H])\b",
        r"\boption\s+([A-H])\b",
        r"^\s*([A-H])(?:[.)]\s+|\s*$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return str(matches[-1]).upper()
    return ""


def _answer_agent_parse_failed(answer_result: AnswerAgentResult | None) -> bool:
    if answer_result is None:
        return False
    haystack = " ".join(
        str(item)
        for item in [
            answer_result.answer,
            answer_result.rationale,
            *list(answer_result.missing_evidence),
        ]
    ).lower()
    return "answer_json_parse_failed" in haystack


def _planner_final_answer_with_option(*, question: str, answer: str) -> str:
    return str(answer)


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
        used_row_keys = set()
        for event in expected_events:
            match = _match_timeline_row(event, confirmed, used_row_keys=used_row_keys)
            if match is None:
                break
            used_row_keys.add(_timeline_row_match_key(match))
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


def _timeline_decision_pending_inference(decision: Mapping[str, Any]) -> str:
    answer = str(decision.get("answer", "")).strip()
    matched_events = decision.get("matched_events", [])
    event_bits = []
    if isinstance(matched_events, Sequence) and not isinstance(matched_events, (str, bytes)):
        for item in list(matched_events)[:6]:
            if not isinstance(item, Mapping):
                continue
            expected = str(item.get("expected") or item.get("observed") or "").strip()
            start_sec = item.get("start_sec")
            if expected and start_sec is not None:
                event_bits.append(f"{expected} @ {float(start_sec):.3f}s")
            elif expected:
                event_bits.append(expected)
    evidence_summary = " -> ".join(event_bits) if event_bits else "timeline rows match one option"
    return (
        f"Timeline heuristic finds option {answer} is consistent with confirmed timeline rows: "
        f"{evidence_summary}. This is a pending inference, not an automatic final; decide whether to "
        "finalize with answer-grade citations or gather more evidence."
    )


def _answer_agent_operator_kwargs(
    *,
    workspace: EvidenceWorkspace,
    projection_status: Mapping[str, Any] | None = None,
    diagnostic_repair_hint: str | None = None,
) -> dict[str, str]:
    runtime = getattr(workspace, "grounding_runtime", None)
    operator = str(getattr(runtime, "answer_operator", "") or "select_present")
    status = projection_status or {}
    final_reason = str(status.get("reason") or "")
    return {
        "answer_operator": operator,
        "projection_candidate": str(status.get("candidate_option") or ""),
        "projection_reason": str(status.get("reason") or ""),
        "missing_evidence_shape": str(status.get("missing") or ""),
        "final_gate_rejection_reason": final_reason if str(status.get("status") or "") == "rejected" else "",
        "diagnostic_repair_hint": str(diagnostic_repair_hint or ""),
    }


def _answer_agent_trace_fields(answer_result: AnswerAgentResult) -> dict[str, str]:
    return {
        "answer_operator": str(getattr(answer_result, "answer_operator", "") or "select_present"),
        "projection_candidate": str(getattr(answer_result, "projection_candidate", "") or ""),
        "projection_reason": str(getattr(answer_result, "projection_reason", "") or ""),
        "final_gate_rejection_reason": str(getattr(answer_result, "final_gate_rejection_reason", "") or ""),
        "diagnostic_repair_hint": str(getattr(answer_result, "diagnostic_repair_hint", "") or ""),
    }


def _answer_result_pending_inference(answer_result: AnswerAgentResult, *, source: str) -> str:
    answer = str(answer_result.answer or "unknown").strip()
    citations = ", ".join(str(citation) for citation in list(answer_result.citations)[:5] if str(citation))
    citation_text = f" with citations {citations}" if citations else ""
    if answer_result.status == "final":
        return (
            f"AnswerAgent suggestion from {source}: option/answer {answer}{citation_text} "
            f"(confidence {answer_result.confidence:.2f}). This is suggestion-only; planner must decide "
            "whether to finalize or gather more evidence."
        )
    if answer_result.has_partial_support():
        low_confidence = answer_result.as_low_confidence_final()
        partial_citations = ", ".join(str(citation) for citation in list(low_confidence.citations)[:5] if str(citation))
        citation_text = f" with partial citations {partial_citations}" if partial_citations else ""
        return (
            f"AnswerAgent partial-support suggestion from {source}: option/answer {low_confidence.answer}"
            f"{citation_text} (confidence {low_confidence.confidence:.2f}). This is not a final; "
            "resolve the missing evidence before finalizing."
        )
    missing = "; ".join(str(item) for item in list(answer_result.missing_evidence)[:3] if str(item))
    return f"AnswerAgent from {source} needs more evidence: {missing or answer_result.rationale or 'targeted follow-up needed'}."


def _confirmed_timeline_rows(timeline: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    confirmed = [
        row
        for row in timeline
        if isinstance(row, Mapping)
        and str(row.get("confidence_signal", "")).strip().lower() == "visually_confirmed"
        and row.get("observed_at_sec") is not None
    ]
    return sorted(
        confirmed,
        key=lambda row: (float(row.get("observed_at_sec", 0.0) or 0.0), str(row.get("obs_id", ""))),
    )


def _option_temporal_events(option_text: str) -> list[str]:
    quoted_events = [
        _clean_target_fact(match.group(1))
        for match in re.finditer(r"[\"“]([^\"”]+)[\"”]", str(option_text))
    ]
    quoted_events = [event for event in quoted_events if _informative_target_fact(event)]
    if len(quoted_events) >= 2:
        return quoted_events

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
    used_row_keys: set[str] | None = None,
    used_obs_ids: set[str] | None = None,
) -> Mapping[str, Any] | None:
    used_keys = used_row_keys or used_obs_ids or set()
    event_tokens = set(_target_fact_key(event).split())
    if not event_tokens:
        return None
    best: tuple[float, Mapping[str, Any]] | None = None
    for row in rows:
        row_key = _timeline_row_match_key(row)
        if row_key in used_keys:
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


def _timeline_row_match_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("obs_id", "")),
            str(row.get("entity", "")),
            str(row.get("observed_at_sec", "")),
        ]
    )


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
    return bool(citations)


def _tool_exploration_question(
    question: str,
    *,
    route_hint: str = "",
    question_context: str = "",
    vlm_safe_question: str = "",
    forbidden_question: str = "",
    option_blind: bool = False,
    target_entities: Sequence[str] = (),
) -> str:
    cleaned = " ".join(str(question or "").split()).strip()
    rewritten_context = " ".join(str(question_context or "").split()).strip()
    safe_context = " ".join(str(vlm_safe_question or "").split()).strip()
    if rewritten_context and (
        not forbidden_question or not _text_has_option_surface(rewritten_context, raw_question=forbidden_question)
    ):
        safe_context = rewritten_context
    if not safe_context:
        safe_context = exploration_question(question_context or forbidden_question or question, route_hint=route_hint)
    if forbidden_question and _text_has_option_surface(cleaned, raw_question=forbidden_question):
        if safe_context and not _text_has_option_surface(safe_context, raw_question=forbidden_question):
            return safe_context
        return "Gather factual evidence needed for final matching. Do not choose or compare options."
    if not option_blind:
        candidate = exploration_question(question, route_hint=route_hint)
        if forbidden_question and _text_has_option_surface(candidate, raw_question=forbidden_question):
            return safe_context
        return candidate
    lowered_route = str(route_hint or "").lower()
    if "timeline" in lowered_route or "temporal" in lowered_route:
        return _append_target_attention_block(
            "Openly describe this segment's actual visible artworks, objects, people, scene changes, "
            "onscreen text, and narrated events in presentation order. Include timestamps if possible. "
            "Focus on concrete observations rather than conclusions. Do not choose an option.",
            target_entities=target_entities,
            timeline=True,
        )
    if "main_idea" in lowered_route or "gist" in lowered_route:
        return (
            "Openly describe this segment's actual visual content and narrated topic. Mention concrete "
            "entities, setting, stage of the story, and any visible text. Do not choose or compare options."
        )
    if forbidden_question and _text_has_option_surface(cleaned, raw_question=forbidden_question):
        fallback = safe_context or " ".join(str(question_context or "").split()).strip()
        if fallback and not _text_has_option_surface(fallback, raw_question=forbidden_question):
            return fallback
        return "Gather factual evidence needed for final matching."
    if extract_candidate_options(question):
        return exploration_question(question, route_hint=route_hint)
    if extract_candidate_options(question_context) and "choose an option" not in cleaned.lower():
        return f"{cleaned} Do not choose an option.".strip()
    return cleaned


def _append_target_attention_block(
    prompt: str,
    *,
    target_entities: Sequence[str],
    timeline: bool = False,
) -> str:
    targets = []
    seen = set()
    for target in target_entities:
        text = " ".join(str(target or "").split()).strip()
        if not text or text in seen:
            continue
        targets.append(text)
        seen.add(text)
    if not targets:
        return prompt
    lines = [
        prompt.strip(),
        "",
        "Pay special attention to these unordered target artwork names or aliases if they appear:",
        *[f"- {target}" for target in targets],
        "",
        "For each target-like item, report whether it is directly shown, narrated, visible as onscreen text, or only visually similar.",
    ]
    if timeline:
        lines.append("Report local timestamp/order, exact text if present, and visible cues. Also report other artworks/transitions in order.")
    lines.append("Do not choose or compare options.")
    return "\n".join(lines)


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


def _program_signature(program: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(_signature_value(list(program)), ensure_ascii=True, sort_keys=True, default=str)


def _tool_call_signature(*, tool_name: str, args: Mapping[str, Any]) -> str:
    return json.dumps(
        _signature_value({"tool": str(tool_name), "args": dict(args)}),
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )


def _observation_is_zero_yield_failure(*, tool_name: str, raw_output: Mapping[str, Any]) -> bool:
    limitations = str(raw_output.get("limitations", "") or "").strip()
    answer_rows = raw_output.get("answer_evidence_rows", [])
    if isinstance(answer_rows, Sequence) and not isinstance(answer_rows, (str, bytes)) and len(answer_rows) > 0:
        return False
    bindings = raw_output.get("evidence_bindings") or raw_output.get("bindings") or []
    if _has_supported_binding(bindings):
        return False
    relations = raw_output.get("candidate_option_relations", [])
    if _has_support_relation(relations):
        return False
    if str(tool_name).strip() in _ZERO_YIELD_EMPTY_RESULT_TOOLS:
        return True
    if not limitations:
        return False
    return True


def _has_supported_binding(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    for item in value:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or item.get("support_status") or "").strip().lower()
        if status == "supported":
            return True
    return False


def _has_support_relation(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    for item in value:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("relation", "")).strip().lower() == "support":
            return True
    return False


_PROTOCOL_FAILURE_REASONS = frozenset(
    {
        "free_text_target_ref",
        "coverage_query_id_not_callable",
        "unknown_target_ref",
        "unknown_legacy_target_ref",
        "additional_targets_not_allowed",
        "invalid_segment_id",
        "unresolved_media_segment",
        "route_violation",
        "skill_name_as_tool",
        "tool_not_in_allowed_actions",
    }
)


def _normalization_failure_signature(
    *,
    planned_program: Any,
    normalization_notes: Sequence[NormalizationNote],
    active_skill: SkillSpec | None,
) -> FailureSignature | None:
    protocol_notes = [note for note in normalization_notes if note.reason in _PROTOCOL_FAILURE_REASONS]
    if not protocol_notes:
        return None
    reason = protocol_notes[0].reason
    affected_tools = tuple(_unique_preserving_order([note.tool for note in protocol_notes if note.tool]))
    signature_payload = {
        "effective_skill": _runtime_skill_id(active_skill),
        "planned_program": planned_program if isinstance(planned_program, Sequence) else [],
        "failures": [
            {
                "tool": note.tool,
                "reason": note.reason,
                "original": note.original,
            }
            for note in protocol_notes
        ],
    }
    return FailureSignature(
        program_signature=json.dumps(_signature_value(signature_payload), ensure_ascii=True, sort_keys=True, default=str),
        reason=reason,
        affected_tools=affected_tools,
    )


def _structured_recovery_for_failure(
    signature: FailureSignature,
    *,
    normalization_notes: Sequence[NormalizationNote],
) -> dict[str, Any]:
    reasons = {note.reason for note in normalization_notes if note.reason in _PROTOCOL_FAILURE_REASONS}
    recovery: dict[str, Any] = {
        "affected_tools": list(signature.affected_tools),
        "do_not_repeat_failed_program": True,
    }
    if reasons & {"free_text_target_ref", "unknown_target_ref", "unknown_legacy_target_ref"}:
        recovery.update(
            {
                "remove_unknown_target_refs": True,
                "preserve_natural_language_targets": True,
                "do_not_invent_registry_ids": True,
            }
        )
    if "additional_targets_not_allowed" in reasons:
        recovery.update(
            {
                "remove_additional_targets_from_bound_tools": True,
                "use_additional_targets_only_for_discovery": True,
            }
        )
    if reasons & {"invalid_segment_id", "unresolved_media_segment"}:
        recovery.update(
            {
                "use_exact_scene_segment_id": True,
                "do_not_invent_segment_ids": True,
                "do_not_substitute_unknown_segment": True,
            }
        )
    if reasons & {"route_violation", "skill_name_as_tool", "tool_not_in_allowed_actions"}:
        recovery.update(
            {
                "choose_allowed_action_from_active_skill": True,
                "do_not_call_skill_names_as_tools": True,
            }
        )
    next_actions = _unique_preserving_order(
        [note.next_action for note in normalization_notes if note.reason in _PROTOCOL_FAILURE_REASONS and note.next_action]
    )
    if next_actions:
        recovery["next_actions"] = next_actions[:3]
    return recovery


def _protocol_recovery_feedback(recovery: Mapping[str, Any]) -> str:
    return "Repeated invalid tool protocol blocked. Apply this structured recovery: " + json.dumps(
        dict(recovery),
        ensure_ascii=True,
        sort_keys=True,
    )


def _signature_value(value: Any) -> Any:
    generated_keys = {"assign", "trace_id", "observation_id"}
    if isinstance(value, Mapping):
        return {
            str(key): _signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in generated_keys
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_signature_value(item) for item in value]
    return value


def _tool_denied_by_skill(*, tool_name: str, active_skill: SkillSpec | None) -> bool:
    return (
        active_skill is not None
        and active_skill.playbook is None
        and active_skill.allowed_actions
        and tool_name not in active_skill.allowed_actions
    )


def _tool_forbidden_by_skill(*, tool_name: str, active_skill: SkillSpec | None) -> bool:
    if active_skill is None or active_skill.playbook is None:
        return False
    return tool_name in set(active_skill.playbook.forbidden_actions)


def _tool_non_suggested_by_playbook(*, tool_name: str, active_skill: SkillSpec | None) -> bool:
    if active_skill is None or active_skill.playbook is None:
        return False
    suggested = set(active_skill.playbook.suggested_actions or active_skill.allowed_actions)
    return bool(suggested and tool_name not in suggested)


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


def _record_skill_action_advisory(
    *,
    workspace: EvidenceWorkspace,
    notes_out: list[NormalizationNote] | None,
    tool_name: str,
    args: Mapping[str, Any],
    active_skill: SkillSpec | None,
) -> None:
    if active_skill is None:
        return
    suggested_actions = ", ".join(sorted(active_skill.playbook.suggested_actions or active_skill.allowed_actions)) or "(none)"
    workspace.write_trace_event(
        "skill_action_advisory",
        {
            "tool": tool_name,
            "skill": active_skill.name,
            "reason": "non_suggested_action",
            "suggested_actions": suggested_actions,
        },
    )
    _append_normalization_note(
        notes_out,
        tool=tool_name,
        reason="non_suggested_action_advisory",
        original={"tool": tool_name, "args": dict(args)},
        next_action="This valid tool is outside the active playbook's suggested actions; use it only if it directly repairs the evidence gap.",
    )


def _record_skill_forbidden_action(
    *,
    workspace: EvidenceWorkspace,
    notes_out: list[NormalizationNote] | None,
    tool_name: str,
    args: Mapping[str, Any],
    active_skill: SkillSpec | None,
) -> None:
    if active_skill is None:
        return
    next_action = f"{tool_name} is forbidden by the active playbook ({active_skill.name}); choose a different valid tool."
    workspace.write_trace_event(
        "route_violation",
        {
            "tool": tool_name,
            "error": "tool_forbidden_by_playbook",
            "skill": active_skill.name,
            "next_action": next_action,
        },
    )
    _append_normalization_note(
        notes_out,
        tool=tool_name,
        reason="tool_forbidden_by_playbook",
        original={"tool": tool_name, "args": dict(args)},
        next_action=next_action,
    )


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


def _route_repair_key(*, reason: str, args: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(reason),
        str(args.get("segment_id") or ""),
        tuple(_route_repair_target_keys(args)),
    )


def _route_kind_for_repair_reason(reason: str) -> str:
    if str(reason) == "repair_ordered_list_locator_to_focused_ordered_list_vision":
        return "focused_ordered_list_vision"
    if str(reason) == "repair_narration_locator_to_transcript_promotion":
        return "narration_transcript_promotion"
    return ""


def _is_focused_ordered_list_vision_args(args: Mapping[str, Any]) -> bool:
    event_label = str(args.get("event_label") or "")
    return event_label.startswith("focused_ordered_list_candidate_")


def _has_explicit_subwindow(args: Mapping[str, Any]) -> bool:
    try:
        start_sec = float(args.get("start_sec"))
        end_sec = float(args.get("end_sec"))
    except (TypeError, ValueError):
        return False
    return end_sec > start_sec


def _route_repair_target_keys(args: Mapping[str, Any]) -> list[str]:
    for field_name in ("normalized_target_keys", "target_refs", "targets"):
        if field_name not in args:
            continue
        values = [
            str(value).strip()
            for value in _coerce_target_arg_list(args.get(field_name))
            if str(value).strip()
        ]
        if values:
            return _unique_preserving_order(values)
    return []


def _route_repair_key_payload(key: tuple[str, str, tuple[str, ...]]) -> dict[str, Any]:
    reason, segment_id, target_keys = key
    return {
        "reason": reason,
        "segment_id": segment_id,
        "normalized_target_keys": list(target_keys),
    }


def _route_repair_recovery_program(
    *,
    reason: str,
    original_args: Mapping[str, Any],
    repaired_tool_name: str,
    repaired_args: Mapping[str, Any],
    active_skill: SkillSpec | None,
) -> list[dict[str, Any]]:
    segment_id = str(original_args.get("segment_id") or repaired_args.get("segment_id") or "").strip()
    if segment_id:
        recovery_args: dict[str, Any] = {"segment_id": segment_id, "promote_answer_evidence": True}
        if original_args.get("target_refs"):
            recovery_args["target_refs"] = _route_repair_target_keys({"target_refs": original_args.get("target_refs")})
        elif original_args.get("normalized_target_keys"):
            recovery_args["target_refs"] = _route_repair_target_keys(
                {"normalized_target_keys": original_args.get("normalized_target_keys")}
            )
        elif original_args.get("targets"):
            recovery_args["targets"] = _route_repair_target_keys({"targets": original_args.get("targets")})
        if active_skill is not None and active_skill.name in {"timeline_ordering", "narration_timeline_qa"}:
            recovery_args.setdefault("question_route", active_skill.name)
        return [{"tool": "read_segment_detail", "args": recovery_args}]
    if repaired_tool_name:
        return [{"tool": repaired_tool_name, "args": dict(repaired_args)}]
    return []


def _supported_evidence_binding_rows(workspace: EvidenceWorkspace) -> list[Mapping[str, Any]]:
    table = workspace.evidence_table_v2(
        question="",
        options=[],
        include_legacy_worker_votes=True,
    )
    rows = table.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    supported_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        binding = row.get("evidence_binding")
        if not isinstance(binding, Mapping):
            continue
        if str(binding.get("status", "")).lower() != "supported":
            continue
        supported_rows.append(row)
    return supported_rows


def _has_supported_evidence_binding_id(workspace: EvidenceWorkspace, evidence_ids: Sequence[str]) -> bool:
    explicit_ids = {str(evidence_id) for evidence_id in evidence_ids if str(evidence_id)}
    if not explicit_ids:
        return False
    for row in _supported_evidence_binding_rows(workspace):
        if str(row.get("evidence_id", "")) in explicit_ids:
            return True
    return False


def _supported_evidence_ids_for_answer(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
) -> list[str]:
    evidence_ids: list[str] = []
    for row in _supported_evidence_rows_for_answer(workspace=workspace, question=question, answer=answer):
        binding = row.get("evidence_binding")
        binding_id = binding.get("evidence_id", "") if isinstance(binding, Mapping) else ""
        evidence_id = str(row.get("evidence_id", "") or binding_id).strip()
        if evidence_id:
            evidence_ids.append(evidence_id)
    return _unique_preserving_order(evidence_ids)


def _supported_evidence_rows_for_answer(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
) -> list[Mapping[str, Any]]:
    option = _answer_option_letter(answer)
    if not option:
        return []
    table = workspace.evidence_table_v2(
        question=question,
        options=extract_candidate_options(question),
        include_legacy_worker_votes=True,
    )
    groups = table.get("groups", {})
    if not isinstance(groups, Mapping):
        return []
    rows = groups.get(option, [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    supported_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        binding = row.get("evidence_binding")
        if not isinstance(binding, Mapping):
            continue
        if str(binding.get("status", "")).lower() != "supported":
            continue
        supported_rows.append(row)
    return supported_rows


def _bridge_final_evidence_refs(
    *,
    workspace: EvidenceWorkspace,
    question: str,
    answer: str,
    citations: Sequence[str],
    evidence_ids: Sequence[str],
) -> FinalEvidenceBridgeResult:
    original_citations = _unique_preserving_order([str(citation) for citation in citations if str(citation)])
    original_evidence_ids = _unique_preserving_order([str(evidence_id) for evidence_id in evidence_ids if str(evidence_id)])
    citation_list = list(original_citations)
    evidence_id_list = list(original_evidence_ids)
    supported_rows = _supported_evidence_rows_for_answer(workspace=workspace, question=question, answer=answer)
    evidence_to_obs: dict[str, str] = {}
    obs_to_evidence: dict[str, list[str]] = {}
    for row in supported_rows:
        evidence_id = str(row.get("evidence_id", "") or "").strip()
        obs_id = str(row.get("obs_id", "") or row.get("observation_id", "") or "").strip()
        if not evidence_id:
            continue
        if obs_id:
            evidence_to_obs[evidence_id] = obs_id
            obs_to_evidence.setdefault(obs_id, []).append(evidence_id)

    moved_citations: list[str] = []
    for citation in list(citation_list):
        if citation in evidence_to_obs or citation.startswith("ev_"):
            citation_list.remove(citation)
            evidence_id_list.append(citation)
            moved_citations.append(citation)
    for evidence_id in list(evidence_id_list):
        if evidence_id.startswith("obs_"):
            evidence_id_list.remove(evidence_id)
            citation_list.append(evidence_id)

    filled: list[str] = []
    ambiguous: list[Mapping[str, Any]] = []
    existing_evidence = set(evidence_id_list)
    for citation in citation_list:
        candidates = _unique_preserving_order(obs_to_evidence.get(citation, []))
        if not candidates or existing_evidence.intersection(candidates):
            continue
        if len(candidates) == 1:
            evidence_id_list.append(candidates[0])
            existing_evidence.add(candidates[0])
            filled.append(candidates[0])
        else:
            ambiguous.append({"obs_id": citation, "candidate_evidence_ids": candidates})

    citation_list = _unique_preserving_order(citation_list)
    evidence_id_list = _unique_preserving_order(evidence_id_list)
    changed = citation_list != original_citations or evidence_id_list != original_evidence_ids
    return FinalEvidenceBridgeResult(
        citations=citation_list,
        evidence_ids=evidence_id_list,
        filled_evidence_ids=filled,
        ambiguous=ambiguous,
        changed=changed,
    )


def _target_refs_for_answer(*, workspace: EvidenceWorkspace, answer: str) -> list[str]:
    registry = getattr(workspace, "target_registry", None)
    if registry is None:
        return []
    option = _answer_option_letter(answer)
    options_by_id = getattr(registry, "options_by_id", {})
    if option and isinstance(options_by_id, Mapping) and option in options_by_id:
        return [
            str(target_id)
            for target_id in getattr(options_by_id[option], "target_sequence", ())
            if str(target_id).strip()
        ]
    targets_by_id = getattr(registry, "targets_by_id", {})
    if isinstance(targets_by_id, Mapping):
        return [str(target_id) for target_id in sorted(targets_by_id)]
    return []


def _latest_asr_binding_candidates(
    *,
    workspace: EvidenceWorkspace,
    target_refs: Sequence[str] = (),
    failed_call_signatures: set[str] | frozenset[str] = frozenset(),
    limit: int = 2,
) -> list[dict[str, Any]]:
    requested_refs = {str(ref).strip() for ref in target_refs if str(ref).strip()}
    candidates: list[dict[str, Any]] = []
    for observation in reversed(workspace.read_observations(tool_name="target_coverage")):
        raw_output = observation.raw_output if isinstance(observation.raw_output, Mapping) else {}
        coverage = raw_output.get("coverage", [])
        if not isinstance(coverage, Sequence) or isinstance(coverage, (str, bytes)):
            continue
        for row in coverage:
            if not isinstance(row, Mapping):
                continue
            row_ref = str(row.get("target_ref") or row.get("target_id") or "").strip()
            is_registry_ref = bool(_TARGET_REF_RE.fullmatch(row_ref))
            if requested_refs and row_ref not in requested_refs:
                continue
            row_candidates = row.get("candidates", [])
            if not isinstance(row_candidates, Sequence) or isinstance(row_candidates, (str, bytes)):
                continue
            for candidate in row_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                segment_id = str(candidate.get("segment_id", "") or "").strip()
                if not segment_id:
                    continue
                target_refs_for_candidate = [row_ref] if is_registry_ref else []
                call_signature = _tool_call_signature(
                    tool_name="bind_asr_claim",
                    args={"segment_id": segment_id, "target_refs": target_refs_for_candidate},
                )
                if call_signature in failed_call_signatures:
                    continue
                candidates.append(
                    {
                        "segment_id": segment_id,
                        "target_refs": target_refs_for_candidate,
                        "target": str(row.get("target", "") or "").strip(),
                        "score": float(candidate.get("score", 0.0) or 0.0),
                        "source": str(candidate.get("source", "") or "").strip(),
                        "snippet": str(candidate.get("snippet", "") or "").strip(),
                    }
                )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in sorted(candidates, key=lambda item: (-float(item.get("score", 0.0) or 0.0), str(item.get("segment_id", "")))):
        refs = tuple(str(ref) for ref in candidate.get("target_refs", []) if str(ref))
        key = (str(candidate.get("segment_id", "")), refs)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= max(1, int(limit or 1)):
            break
    return deduped


def _supported_binding_no_growth_feedback(
    *,
    candidates: Sequence[Mapping[str, Any]],
    skill_locked: bool,
) -> list[str]:
    base = (
        "No new supported evidence bindings appeared for three rounds; stay within the effective skill and "
        "promote answer-grade evidence with evidence_binding.status=supported."
        if skill_locked
        else "No new supported evidence bindings appeared for three rounds; promote answer-grade evidence with evidence_binding.status=supported."
    )
    lines = [base]
    actionable = [
        candidate
        for candidate in candidates
        if str(candidate.get("segment_id", "") or "").strip()
        and any(str(ref).strip() for ref in candidate.get("target_refs", []) if str(ref).strip())
    ]
    if actionable:
        suggestions = []
        for candidate in actionable[:2]:
            segment_id = str(candidate.get("segment_id", "") or "").strip()
            refs = [str(ref).strip() for ref in candidate.get("target_refs", []) if str(ref).strip()]
            suggestions.append(f"bind_asr_claim(segment_id='{segment_id}', target_refs={refs})")
        lines.append("Suggested next action: " + "; ".join(suggestions) + ".")
    elif candidates:
        compact = []
        for candidate in candidates[:2]:
            compact.append(
                f"{candidate.get('target', 'target')} -> {candidate.get('segment_id', 'segment')}"
            )
        lines.append(
            "Top coverage candidates were found but lack registry target_refs for automatic binding: "
            + "; ".join(compact)
            + "."
        )
    return lines


def _segment_ids_from_observation_payload(raw_output: Mapping[str, Any]) -> list[str]:
    segment_ids: list[str] = []
    direct = str(raw_output.get("segment_id", "") or "").strip()
    if direct:
        segment_ids.append(direct)
    for key in ("regions", "candidates"):
        values = raw_output.get(key, [])
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            if isinstance(value, Mapping):
                segment_id = str(value.get("segment_id", "") or "").strip()
                if segment_id:
                    segment_ids.append(segment_id)
    return _unique_preserving_order(segment_ids)


def _supported_binding_overlaps_route_repair_key(
    row: Mapping[str, Any],
    key: tuple[str, str, tuple[str, ...]],
) -> bool:
    _, key_segment_id, key_targets = key
    binding = row.get("evidence_binding")
    if not isinstance(binding, Mapping):
        return False
    binding_segment_id = str(binding.get("segment_id") or row.get("segment_id") or "").strip()
    if key_segment_id and binding_segment_id and key_segment_id != binding_segment_id:
        return False
    if key_targets:
        binding_target = str(
            binding.get("target_id")
            or binding.get("target_ref")
            or binding.get("target")
            or row.get("entity")
            or ""
        ).strip()
        return bool(binding_target and binding_target in set(key_targets))
    return bool(key_segment_id and binding_segment_id == key_segment_id) or not key_segment_id


def _coerce_target_arg_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _is_target_ref_key(value: str) -> bool:
    return bool(_TARGET_REF_RE.fullmatch(value.strip()))


def _workspace_knows_target_ref(workspace: Any, key: str) -> bool:
    registry = getattr(workspace, "target_registry", None)
    if registry is None:
        return False
    for method_name in ("known_target_ref", "has", "contains", "is_known", "knows"):
        method = getattr(registry, method_name, None)
        if callable(method):
            try:
                return bool(method(key))
            except TypeError:
                continue
    get = getattr(registry, "get", None)
    if callable(get):
        try:
            return get(key) is not None
        except (KeyError, TypeError):
            pass
    resolve = getattr(registry, "resolve_target_ref", None) or getattr(registry, "resolve", None)
    if callable(resolve):
        try:
            return resolve(key) is not None
        except (KeyError, TypeError):
            pass
    try:
        return key in registry
    except TypeError:
        return False


def _workspace_has_target_registry(workspace: Any) -> bool:
    registry = getattr(workspace, "target_registry", None)
    return isinstance(getattr(registry, "targets_by_id", None), Mapping)


def _tool_is(tool_name: str | None, expected: str) -> bool:
    return str(tool_name or "").strip() in {expected}


def _tool_text_field(tool_name: str, *, default: str) -> str:
    return {"vision_read": "ask_for"}.get(str(tool_name or "").strip(), default)


def _additional_targets_allowed(*, tool_name: str, args: Mapping[str, Any]) -> bool:
    if _tool_is(tool_name, "search_segments"):
        return "query" in args
    if _tool_is(tool_name, "vision_read"):
        return "ask_for" in args
    if _tool_is(tool_name, "caption_segment"):
        return "question" in args or "ask_for" in args
    return False


def _append_additional_targets_to_text(value: Any, additional_targets: Sequence[str]) -> str:
    base = str(value or "").strip()
    extras = _unique_nonempty_strings(additional_targets)
    if not extras:
        return base
    suffix = "Additional targets: " + "; ".join(extras)
    return f"{base}\n{suffix}" if base else suffix


def _exact_registry_ref_for_legacy_target(workspace: Any, target_text: str) -> str:
    registry = getattr(workspace, "target_registry", None)
    if registry is None:
        return ""
    text = str(target_text).strip()
    if not text:
        return ""
    if _is_target_ref_key(text) and _workspace_knows_target_ref(workspace, text):
        return text
    targets_by_id = getattr(registry, "targets_by_id", None)
    if isinstance(targets_by_id, Mapping):
        if text in targets_by_id:
            return text
        for target_id, target in targets_by_id.items():
            surfaces = [
                str(getattr(target, "canonical_text", "")).strip(),
                *[str(alias).strip() for alias in getattr(target, "aliases", ())],
            ]
            if any(text == surface for surface in surfaces if surface):
                return str(target_id)
    return ""


def _unique_nonempty_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _registered_target_ref_descriptions(workspace: Any) -> list[str]:
    registry = getattr(workspace, "target_registry", None)
    if registry is None:
        return []
    targets_by_id = getattr(registry, "targets_by_id", {})
    if isinstance(targets_by_id, Mapping):
        descriptions = []
        for target_id in sorted(str(key) for key in targets_by_id):
            target = targets_by_id.get(target_id)
            text = str(getattr(target, "canonical_text", "") or "").strip()
            descriptions.append(f"{target_id}: {text}" if text else target_id)
        return descriptions
    return []


def _is_coverage_query_id(value: str) -> bool:
    return bool(re.fullmatch(r"Q[1-9]\d*", str(value or "").strip()))


def _coverage_query_target_for_id(workspace: Any, query_id: str) -> str:
    query_id = str(query_id or "").strip()
    if not _is_coverage_query_id(query_id):
        return ""
    try:
        observations = workspace.read_observations(tool_name="target_coverage")
    except AttributeError:
        return ""
    for observation in reversed(observations):
        raw_output = getattr(observation, "raw_output", {})
        if not isinstance(raw_output, Mapping):
            continue
        rows = raw_output.get("coverage", [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("query_id") or row.get("target_id") or "").strip() == query_id:
                return str(row.get("target") or "").strip()
    return ""


def _same_target_text(left: str, right: str) -> bool:
    return " ".join(str(left or "").split()) == " ".join(str(right or "").split())


def _unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


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


def _route_violation(*, tool_name: str, active_skill: SkillSpec | None) -> str | None:
    if active_skill is None or not active_skill.allowed_actions:
        return None
    if active_skill.playbook is not None:
        if tool_name in set(active_skill.playbook.forbidden_actions):
            return f"action '{tool_name}' is forbidden by skill '{active_skill.name}' playbook"
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


def _focused_window_for_scene_segment(
    segment: VideoSegment,
    *,
    args: Mapping[str, Any],
    duration_sec: float,
) -> Optional[VideoSegment]:
    if "start_sec" not in args or "end_sec" not in args:
        return None
    try:
        start_sec, end_sec = _normalize_dynamic_window(
            start_sec=float(args["start_sec"]),
            end_sec=float(args["end_sec"]),
            duration_sec=duration_sec,
            label=f"for focused {segment.segment_id}",
        )
    except (TypeError, ValueError):
        return None
    segment_start = float(segment.start_sec)
    segment_end = float(segment.end_sec)
    if start_sec < segment_start - 0.001 or end_sec > segment_end + 0.001:
        return None
    if (end_sec - start_sec) >= max(0.0, (segment_end - segment_start) - 1.0):
        return None
    return VideoSegment(
        segment_id=segment.segment_id,
        start_sec=start_sec,
        end_sec=end_sec,
        source="focused_scene_window",
    )


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
