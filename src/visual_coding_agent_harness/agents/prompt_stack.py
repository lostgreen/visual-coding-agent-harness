"""Composable prompt stack for the long-video planner loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..video_index import SceneIndex
from .context_budget import ContextBudgetAllocator, ContextBudgetReport, SlotName
from .question_policy import QuestionPlaybook, select_question_playbook
from .skills.specs import allowed_actions_for_skill, skill_catalog_prompt


@dataclass(frozen=True)
class PromptBlock:
    name: str
    title: str
    body: str

    def render(self) -> str:
        return f"# {self.title}\n{self.body.strip()}\n"


def render_prompt_blocks(blocks: Sequence[PromptBlock]) -> str:
    return "\n".join(block.render() for block in blocks).strip()


def _route_playbook_body(playbook: QuestionPlaybook, *, option_blind: bool = False) -> str:
    if not option_blind:
        return playbook.to_prompt()
    if playbook.route == "gist_global":
        return QuestionPlaybook(
            name=playbook.name,
            route=playbook.route,
            instructions=[
                "Start with global_gist to get a sparse whole-video topic and coverage hint.",
                "Use local inspection or indexed transcript evidence to verify full-video coverage.",
                "Collect factual coverage across the full narrative arc before handing off to AnswerAgent.",
            ],
            sufficiency_rules=[
                "A global_gist observation is a topic hint, not final support.",
                "Record whether cited facts cover the main entity, time span, and major narrative stages.",
            ],
        ).to_prompt()
    if playbook.route == "temporal_order":
        return QuestionPlaybook(
            name=playbook.name,
            route=playbook.route,
            instructions=[
                "Use coarse captions to locate target event/entity segments before focused timestamp reads.",
                "Inspect the relevant earlier and later windows when order matters.",
                "Local workers should report facts and presentation order only.",
            ],
            sufficiency_rules=[
                "Citations must include timestamped answer-grade visual, ASR, OCR, or QA evidence for the ordered events.",
                "Evidence must not conflict with the claimed temporal relation.",
                "Record the observed order with segment or timestamp evidence before final handoff.",
            ],
        ).to_prompt()
    return QuestionPlaybook(
        name=playbook.name,
        route=playbook.route,
        instructions=[
            "Use query-conditioned navigation to localize likely evidence.",
            "Delegate visual reading to vision_read or inspect_segment once a candidate segment is localized.",
            "Local workers should report facts only.",
        ],
        sufficiency_rules=[
            "Final handoff needs cited answer-grade visual, ASR, OCR, or QA evidence.",
            "State uncertainty when evidence is incomplete or ambiguous.",
        ],
    ).to_prompt()


def build_replanning_prompt(
    *,
    question: str,
    scene_index: SceneIndex,
    ledger_text: str,
    round_number: int,
    budget: Any,
    allocator: ContextBudgetAllocator,
    inspected_segment_ids: Sequence[str] = (),
    tool_class_counts: Mapping[str, int] | None = None,
    final_round_reserved: bool = False,
    answer_feedback: Sequence[str] = (),
    pending_inferences: Sequence[str] = (),
    normalization_notes: Sequence[Any] = (),
    hypothesis_text: str = "",
    reflection_memory: Sequence[str] = (),
    evidence_status_summary: Mapping[str, Any] | None = None,
    recent_tool_outputs: Sequence[Mapping[str, Any]] = (),
    exhausted_tools: frozenset[str] | None = None,
    active_skill: str | None = None,
    route: str | None = None,
    target_hints: Sequence[str] = (),
    target_ref_descriptions: Sequence[str] = (),
) -> tuple[str, ContextBudgetReport]:
    slots = compose_replanning_prompt_slots(
        question=question,
        scene_index=scene_index,
        ledger_text=ledger_text,
        round_number=round_number,
        budget=budget,
        inspected_segment_ids=inspected_segment_ids,
        final_round_reserved=final_round_reserved,
        answer_feedback=answer_feedback,
        pending_inferences=pending_inferences,
        normalization_notes=normalization_notes,
        hypothesis_text=hypothesis_text,
        reflection_memory=reflection_memory,
        evidence_status_summary=evidence_status_summary,
        recent_tool_outputs=recent_tool_outputs,
        exhausted_tools=exhausted_tools,
        active_skill=active_skill,
        route=route,
        target_hints=target_hints,
        target_ref_descriptions=target_ref_descriptions,
    )
    allocated, report = allocator.allocate(
        slots,
        ctx={
            "round": round_number,
            "active_followup_target_query": str(answer_feedback[0]) if answer_feedback else "",
        },
    )
    return _join_slots(allocated), report


def compose_replanning_prompt_slots(
    *,
    question: str,
    scene_index: SceneIndex,
    ledger_text: str,
    round_number: int,
    budget: Any,
    inspected_segment_ids: Sequence[str] = (),
    tool_class_counts: Mapping[str, int] | None = None,
    final_round_reserved: bool = False,
    answer_feedback: Sequence[str] = (),
    pending_inferences: Sequence[str] = (),
    normalization_notes: Sequence[Any] = (),
    hypothesis_text: str = "",
    reflection_memory: Sequence[str] = (),
    evidence_status_summary: Mapping[str, Any] | None = None,
    recent_tool_outputs: Sequence[Mapping[str, Any]] = (),
    exhausted_tools: frozenset[str] | None = None,
    active_skill: str | None = None,
    route: str | None = None,
    target_hints: Sequence[str] = (),
    target_ref_descriptions: Sequence[str] = (),
) -> dict[SlotName, str]:
    playbook = select_question_playbook(question)
    resolved_route = route or playbook.route
    option_blind = bool(getattr(budget, "rewrite_mcq_for_exploration", False))
    task_blocks = [
        PromptBlock(
            name="base_identity",
            title="Base Identity",
            body=(
                "You are an autonomous visual agent exploring a long video with tools.\n"
                "Planner input mode: text-only. Use the scene index and evidence ledger; tools inspect pixels/video.\n"
                "Use a short ReAct shell: pick the next action, observe tool output, then decide whether evidence is sufficient.\n"
                "Allowed ReAct actions: ground_question, vision_read, answer_agent, verify.\n"
                "Do not include step-by-step private reasoning in the JSON response."
            ),
        ),
        PromptBlock(
            name="route_playbook",
            title="Route Playbook",
            body=_route_playbook_body(playbook, option_blind=option_blind),
        ),
        PromptBlock(
            name="skill_catalog",
            title="Skill Catalog",
            body=_skill_catalog_block(active_skill=active_skill, exhausted_tools=exhausted_tools),
        ),
    ]
    tooling_blocks = [
        PromptBlock(
            name="tool_schema",
            title="Tool Schema",
            body=_tool_schema_block(
                option_blind=option_blind,
                active_skill=active_skill,
                exhausted=exhausted_tools or frozenset(),
                target_ref_descriptions=target_ref_descriptions,
            ),
        ),
        PromptBlock(
            name="final_gate",
            title="Final Gate",
            body=_final_gate_block(
                final_round_reserved=final_round_reserved,
                option_blind=option_blind,
                route=resolved_route,
                include_target_refs=bool(target_ref_descriptions),
            ),
        ),
        PromptBlock(
            name="response_contract",
            title="Response Contract",
            body=(
                "Return only JSON with one of these schemas:\n"
                '{"status": "continue", "skill": string, "rationale": string, "program": [{"tool": string, "args": object, "assign": string}]}\n'
                '{"status": "final", "skill": string, "answer": string, "citations": [observation_id], "confidence": number}'
            ),
        ),
    ]
    evidence_status_text = _evidence_status_summary_text(evidence_status_summary)
    evidence_body = "# Evidence Snapshot\n"
    if evidence_status_text:
        evidence_body += evidence_status_text + "\n"
    recent_outputs_text = _recent_tool_outputs_block(recent_tool_outputs)
    if recent_outputs_text:
        evidence_body += recent_outputs_text + "\n"
    evidence_body += "Evidence ledger:\n" + (ledger_text or "(none)")
    return {
        "task": render_prompt_blocks(task_blocks),
        "trajectory": _trajectory_snapshot_block(
            round_number=round_number,
            budget=budget,
            inspected_segment_ids=inspected_segment_ids,
        ),
        "hypothesis": _hypothesis_slot(hypothesis_text),
        "evidence": evidence_body,
        "scene_index": _scene_index_snapshot_block(
            question=question,
            scene_index=scene_index,
            inspected_segment_ids=inspected_segment_ids,
            target_hints=target_hints,
        ),
        "feedback": _feedback_slot(
            answer_feedback=answer_feedback,
            pending_inferences=pending_inferences,
            normalization_notes=normalization_notes,
            reflection_memory=reflection_memory,
        ),
        "budget": _budget_snapshot_block(
            round_number=round_number,
            budget=budget,
            final_round_reserved=final_round_reserved,
        ),
        "tooling": render_prompt_blocks(tooling_blocks),
    }


def compose_replanning_prompt_blocks(
    *,
    question: str,
    scene_index: SceneIndex,
    ledger_text: str,
    round_number: int,
    budget: Any,
    inspected_segment_ids: Sequence[str] = (),
    tool_class_counts: Mapping[str, int] | None = None,
    final_round_reserved: bool = False,
    answer_feedback: Sequence[str] = (),
    normalization_notes: Sequence[Any] = (),
    hypothesis_text: str = "",
    reflection_memory: Sequence[str] = (),
    evidence_status_summary: Mapping[str, Any] | None = None,
    exhausted_tools: frozenset[str] | None = None,
    active_skill: str | None = None,
    route: str | None = None,
    target_hints: Sequence[str] = (),
    target_ref_descriptions: Sequence[str] = (),
) -> list[PromptBlock]:
    playbook = select_question_playbook(question)
    resolved_route = route or playbook.route
    option_blind = bool(getattr(budget, "rewrite_mcq_for_exploration", False))
    blocks = [
        PromptBlock(
            name="base_identity",
            title="Base Identity",
            body=(
                "You are an autonomous visual agent exploring a long video with tools.\n"
                "Planner input mode: text-only. Use the scene index and evidence ledger; tools inspect pixels/video.\n"
                "Use a short ReAct shell: pick the next action, observe tool output, then decide whether evidence is sufficient.\n"
                "Allowed ReAct actions: ground_question, vision_read, answer_agent, verify.\n"
                "Do not include step-by-step private reasoning in the JSON response."
            ),
        ),
        PromptBlock(
            name="route_playbook",
            title="Route Playbook",
            body=_route_playbook_body(playbook, option_blind=option_blind),
        ),
        PromptBlock(
            name="skill_catalog",
            title="Skill Catalog",
            body=_skill_catalog_block(active_skill=active_skill, exhausted_tools=exhausted_tools),
        ),
        PromptBlock(
            name="trajectory_snapshot",
            title="Trajectory Snapshot",
            body=_trajectory_snapshot_block(
                round_number=round_number,
                budget=budget,
                inspected_segment_ids=inspected_segment_ids,
            ),
        ),
        PromptBlock(
            name="evidence_snapshot",
            title="Evidence Snapshot",
            body=_evidence_only_snapshot_block(
                ledger_text=ledger_text,
                evidence_status_summary=evidence_status_summary,
            ),
        ),
        PromptBlock(
            name="hypothesis",
            title="Hypothesis",
            body=_hypothesis_slot(hypothesis_text),
        ),
        PromptBlock(
            name="scene_index_snapshot",
            title="Compact Scene Index",
            body=_scene_index_snapshot_block(
                question=question,
                scene_index=scene_index,
                inspected_segment_ids=inspected_segment_ids,
                target_hints=target_hints,
            ),
        ),
    ]
    if normalization_notes:
        blocks.append(
            PromptBlock(
                name="normalization_notes",
                title="Last Round Adjustments",
                body=_normalization_notes_body(normalization_notes),
            )
        )
    if answer_feedback:
        blocks.append(
            PromptBlock(
                name="answer_feedback",
                title="Answer Feedback",
                body=(
                    "Answer Agent says these evidence gaps must be resolved before final: "
                    + "; ".join(str(item) for item in answer_feedback[:5])
                ),
            )
        )
    if reflection_memory:
        blocks.append(
            PromptBlock(
                name="reflection_memory",
                title="Reflection Memory",
                body="\n".join(f"- {item}" for item in reflection_memory[:5]),
            )
        )
    blocks.extend(
        [
            PromptBlock(
                name="budget_snapshot",
                title="Current Budgets",
                body=_budget_snapshot_block(
                    round_number=round_number,
                    budget=budget,
                    final_round_reserved=final_round_reserved,
                ),
            ),
            PromptBlock(
                name="tool_schema",
                title="Tool Schema",
                body=_tool_schema_block(
                    option_blind=option_blind,
                active_skill=active_skill,
                exhausted=exhausted_tools or frozenset(),
                target_ref_descriptions=target_ref_descriptions,
            ),
        ),
            PromptBlock(
                name="final_gate",
                title="Final Gate",
                body=_final_gate_block(
                    final_round_reserved=final_round_reserved,
                    option_blind=option_blind,
                    route=resolved_route,
                    include_target_refs=bool(target_ref_descriptions),
                ),
            ),
            PromptBlock(
                name="response_contract",
                title="Response Contract",
                body=(
                    "Return only JSON with one of these schemas:\n"
                    '{"status": "continue", "skill": string, "rationale": string, "program": [{"tool": string, "args": object, "assign": string}]}\n'
                    '{"status": "final", "skill": string, "answer": string, "citations": [observation_id], "confidence": number}'
                ),
            ),
        ]
    )
    return blocks


def _tool_schema_block(
    *,
    option_blind: bool = False,
    active_skill: str | None = None,
    exhausted: frozenset[str] = frozenset(),
    target_ref_descriptions: Sequence[str] = (),
) -> str:
    has_registered_refs = bool([item for item in target_ref_descriptions if str(item).strip()])
    signatures = list(_tool_schema_signatures(option_blind=option_blind, include_target_refs=has_registered_refs))
    allowed = allowed_actions_for_skill(active_skill or "") if active_skill else frozenset()
    if allowed:
        signatures = [signature for signature in signatures if _tool_name_from_signature(signature) in allowed]
    rendered = [_maybe_mark_exhausted(signature, exhausted) for signature in signatures]
    return "\n".join(
        [
            _target_registry_contract_block(target_ref_descriptions),
            "Available tools:",
            *[f"- {signature}" for signature in rendered],
        ]
    )


def _tool_schema_signatures(*, option_blind: bool = False, include_target_refs: bool = False) -> tuple[str, ...]:
    inspect_schema = "inspect_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)"
    target_coverage_schema = (
        "target_coverage(targets: list = [], target_refs: list = [], top_k: int = 3, modalities: list = [], group_by_option: bool = False)"
        if include_target_refs
        else "target_coverage(targets: list = [], top_k: int = 3, modalities: list = [], group_by_option: bool = False)"
    )
    read_segment_detail_schema = (
        "read_segment_detail(segment_id: str, targets: list = [], target_refs: list = [], promote_answer_evidence: bool = False)"
        if include_target_refs
        else "read_segment_detail(segment_id: str, targets: list = [], promote_answer_evidence: bool = False)"
    )
    locate_targets_schema = (
        "locate_targets_in_segment(segment_id: str, targets: list = [], target_refs: list = [], top_k_per_target: int = 3)"
        if include_target_refs
        else "locate_targets_in_segment(segment_id: str, targets: list = [], top_k_per_target: int = 3)"
    )
    verify_anchors_schema = (
        "verify_segment_anchors(segment_id: str, anchors: list, question: str = '', targets: list = [], target_refs: list = [])"
        if include_target_refs
        else "verify_segment_anchors(segment_id: str, anchors: list, question: str = '', targets: list = [])"
    )
    verifier_schema = (
        "verify_ledger_answer(answer: str, question: str = '', min_score: float = 0.6, required_citations: list = [])"
        if option_blind
        else "verify_ledger_answer(answer: str, question: str = '', candidate_options: list = [], min_score: float = 0.6, required_citations: list = [])"
    )
    return (
        "ground_question(query: str, top_k: int = 5, modalities: list = [])",
        target_coverage_schema,
        "search_segments(query: str, top_k: int = 5, modalities: list = [])",
        "read_segment(segment_id: str)",
        read_segment_detail_schema,
        locate_targets_schema,
        verify_anchors_schema,
        "global_gist(video_path: str, question: str, duration_sec: float, nframes: int = 128, max_pixels: int = 151200, sample_offset_sec: float = 0.0)",
        "summarize_ledger_evidence(max_claims: int = 5)",
        verifier_schema,
        "view_observation(obs_id: str, line_range: tuple | None = None)",
        "grep_evidence(pattern: str, in_field: str = 'claim')",
        "query_evidence_table(filter: dict)",
        "read_timeline_sorted()",
        "read_hypothesis()",
        "update_hypothesis_slot(slot_name: str, status: str, evidence_obs_id: str = '')",
        "vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = '', nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)",
        inspect_schema,
        "caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)",
        "qa_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)",
    )


def _skill_catalog_block(*, active_skill: str | None, exhausted_tools: frozenset[str] | None) -> str:
    lines = [skill_catalog_prompt(exhausted_tools=exhausted_tools)]
    if active_skill:
        lines.extend(
            [
                "# Effective Skill State",
                f"recommended_skill: {active_skill}",
                f"effective_skill: {active_skill}",
                "skill_locked: true",
                "unlock_used: false",
                "The `skill` field in your response must match effective_skill.",
                "Changing it will not change the active gate.",
                "The tool schema below is filtered to the effective skill.",
            ]
        )
    else:
        lines.append(
            "Select the skill that best matches this case in every planner JSON as `skill`. "
            "Choose from the catalog yourself; the route playbook is guidance, not a skill assignment. "
            "If no catalog skill fits, omit `skill` and use ordinary tool exploration."
        )
    lines.append("Tools listed as `=exhausted` are one-shot and cannot be requested again.")
    return "\n".join(lines)


def _target_registry_contract_block(target_ref_descriptions: Sequence[str]) -> str:
    descriptions = [str(item).strip() for item in target_ref_descriptions if str(item).strip()]
    if not descriptions:
        return (
            "Target Registry:\n"
            "No target_refs are registered for this run. Use natural-language `targets` only. "
            "Coverage-local Q<n> labels are not callable target_refs."
        )
    return "\n".join(
        [
            "Target Registry:",
            "Registered target_refs:",
            *[f"- {description}" for description in descriptions[:24]],
            "Use these exact ids in `target_refs`. Coverage-local Q<n> labels are not callable target_refs.",
        ]
    )


def _tool_name_from_signature(signature: str) -> str:
    return signature.split("(", 1)[0]


def _maybe_mark_exhausted(signature: str, exhausted: frozenset[str]) -> str:
    tool_name = _tool_name_from_signature(signature)
    return f"{signature}  =exhausted" if tool_name in exhausted else signature


def _join_slots(slots: Mapping[SlotName, str]) -> str:
    titles = {
        "task": "Task",
        "trajectory": "Trajectory",
        "navigation": "Navigation",
        "hypothesis": "Hypothesis",
        "evidence": "Evidence",
        "scene_index": "Compact scene index",
        "feedback": "Feedback",
        "budget": "Current budgets",
        "tooling": "Tooling",
    }
    return "\n\n".join(
        f"## {titles[name]}\n{slots.get(name, '').strip()}"
        for name in ["task", "trajectory", "hypothesis", "evidence", "scene_index", "feedback", "budget", "tooling"]
        if name in slots
    ).strip()


def _hypothesis_slot(hypothesis_text: str) -> str:
    return (hypothesis_text or "# Hypothesis\n\n(no slots yet)").strip()


def _evidence_status_summary_text(summary: Mapping[str, Any] | None) -> str:
    if not summary:
        return ""
    lines = ["Evidence status summary:"]
    for key in ["option_coverage", "coverage_pct", "duplicate_observations", "total_evidence_rows"]:
        if key in summary:
            lines.append(f"{key}: {summary.get(key)}")
    option_status = summary.get("option_status", {})
    if isinstance(option_status, Mapping) and option_status:
        lines.append("options:")
        for option in sorted(str(key) for key in option_status):
            raw_status = option_status.get(option, {})
            status = raw_status if isinstance(raw_status, Mapping) else {}
            visual = "yes" if bool(status.get("has_visual_citation")) else "no"
            lines.append(
                f"- {option}: strong={int(status.get('strong_evidence_count', 0) or 0)} "
                f"weak={int(status.get('weak_evidence_count', 0) or 0)} visual={visual}"
            )
    gaps = summary.get("hypothesis_gaps", [])
    if isinstance(gaps, Sequence) and not isinstance(gaps, (str, bytes)) and gaps:
        lines.append("hypothesis_gaps: " + ", ".join(str(item) for item in gaps[:8]))
    else:
        lines.append("hypothesis_gaps: (none)")
    return "\n".join(lines)


def _navigation_snapshot_block(
    *,
    question: str,
    scene_index: SceneIndex,
    round_number: int,
    budget: Any,
    inspected_segment_ids: Sequence[str],
    tool_class_counts: Mapping[str, int] | None,
    final_round_reserved: bool,
) -> str:
    inspected_line = ", ".join(inspected_segment_ids) if inspected_segment_ids else "(none)"
    uninspected_line = _uninspected_segment_summary(scene_index=scene_index, inspected_segment_ids=inspected_segment_ids)
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    return (
        f"Round: {round_number}/{getattr(budget, 'max_rounds', '?')}\n"
        f"Question: {question}\n"
        f"Already inspected segments: {inspected_line}\n"
        f"Uninspected segment candidates: {uninspected_line}\n"
        f"Request at most {getattr(budget, 'max_tool_calls_per_round', 1)} new tool call(s) this round.\n"
        f"{final_round_line}"
        "Scene index:\n"
        f"{scene_index.summary(max_segments=64)}"
    )


def _trajectory_snapshot_block(
    *,
    round_number: int,
    budget: Any,
    inspected_segment_ids: Sequence[str],
) -> str:
    inspected_line = ", ".join(inspected_segment_ids) if inspected_segment_ids else "(none)"
    return (
        f"Round: {round_number}/{getattr(budget, 'max_rounds', '?')}\n"
        f"Already inspected segments: {inspected_line}"
    )


def _scene_index_snapshot_block(
    *,
    question: str,
    scene_index: SceneIndex,
    inspected_segment_ids: Sequence[str],
    target_hints: Sequence[str] = (),
) -> str:
    uninspected_line = _uninspected_segment_summary(scene_index=scene_index, inspected_segment_ids=inspected_segment_ids)
    return (
        f"Question: {question}\n"
        f"Uninspected segment candidates: {uninspected_line}\n"
        "Compact scene index:\n"
        f"{scene_index.summary(max_segments=64, target_hints=target_hints)}"
    )


def _budget_snapshot_block(
    *,
    round_number: int,
    budget: Any,
    final_round_reserved: bool,
) -> str:
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    return (
        f"Round: {round_number}/{getattr(budget, 'max_rounds', '?')}\n"
        f"Request at most {getattr(budget, 'max_tool_calls_per_round', 1)} new tool call(s) this round.\n"
        f"{final_round_line}"
    )


def _feedback_slot(
    *,
    answer_feedback: Sequence[str],
    pending_inferences: Sequence[str] = (),
    normalization_notes: Sequence[Any],
    reflection_memory: Sequence[str],
) -> str:
    blocks = []
    if pending_inferences:
        blocks.append(
            PromptBlock(
                name="pending_inference",
                title="Pending Inference",
                body="\n".join(f"- {item}" for item in pending_inferences[:3]),
            ).render()
        )
    if normalization_notes:
        blocks.append(
            PromptBlock(
                name="normalization_notes",
                title="Last Round Adjustments",
                body=_normalization_notes_body(normalization_notes),
            ).render()
        )
    if answer_feedback:
        blocks.append(
            PromptBlock(
                name="answer_feedback",
                title="Answer Feedback",
                body=(
                    "Answer Agent says these evidence gaps must be resolved before final: "
                    + "; ".join(str(item) for item in answer_feedback[:5])
                ),
            ).render()
        )
    if reflection_memory:
        blocks.append(
            PromptBlock(
                name="reflection_memory",
                title="Reflection Memory",
                body="\n".join(f"- {item}" for item in reflection_memory[:5]),
            ).render()
        )
    return "\n".join(blocks).strip() or "(none)"


def _recent_tool_outputs_block(outputs: Sequence[Mapping[str, Any]]) -> str:
    if not outputs:
        return ""
    lines = ["# Recent Tool Outputs"]
    for output in list(outputs)[-3:]:
        if not isinstance(output, Mapping):
            continue
        obs_id = str(output.get("observation_id", "")).strip()
        tool_name = str(output.get("tool", "")).strip()
        claim = str(output.get("claim", "")).strip()
        title = f"## {obs_id or '(unknown obs)'} | {tool_name or '(unknown tool)'}"
        lines.append(title)
        if claim:
            lines.append(f"claim: {claim}")
        raw_output = output.get("raw_output", {})
        if raw_output:
            lines.append("raw_output:")
            lines.append(json.dumps(raw_output, ensure_ascii=True, sort_keys=True, indent=2))
    return "\n".join(lines).strip()


def _normalization_notes_body(notes: Sequence[Any]) -> str:
    lines = ["Your previous program was modified by the harness. Treat each `DO NEXT` line as a hard constraint:"]
    for note in list(notes)[:8]:
        reason = _note_attr(note, "reason")
        tool = _note_attr(note, "tool")
        original = _note_mapping(note, "original")
        resolved = _note_mapping(note, "resolved")
        next_action = _note_attr(note, "next_action")
        original_tool = str(original.get("tool") or tool)
        resolved_tool = str(resolved.get("tool") or original_tool)
        original_segment = str(original.get("segment_id") or "")
        resolved_segment = str(resolved.get("segment_id") or "")
        left = " ".join(item for item in [original_tool, original_segment] if item).strip() or tool
        if resolved:
            right = " ".join(item for item in [resolved_tool, resolved_segment] if item).strip() or resolved_tool
            lines.append(f"- {left} -> {right} (reason: {reason})")
        else:
            lines.append(f"- {left} step dropped (reason: {reason})")
        if next_action:
            lines.append(f"  DO NEXT: {next_action}")
    return "\n".join(lines)


def _note_attr(note: Any, name: str) -> str:
    if isinstance(note, Mapping):
        return str(note.get(name, ""))
    return str(getattr(note, name, ""))


def _note_mapping(note: Any, name: str) -> Mapping[str, Any]:
    value = note.get(name, {}) if isinstance(note, Mapping) else getattr(note, name, {})
    return value if isinstance(value, Mapping) else {}


def _evidence_snapshot_block(
    *,
    question: str,
    scene_index: SceneIndex,
    ledger_text: str,
    round_number: int,
    budget: Any,
    inspected_segment_ids: Sequence[str],
    tool_class_counts: Mapping[str, int] | None,
    final_round_reserved: bool,
    evidence_status_summary: Mapping[str, Any] | None = None,
) -> str:
    inspected_line = ", ".join(inspected_segment_ids) if inspected_segment_ids else "(none)"
    uninspected_line = _uninspected_segment_summary(scene_index=scene_index, inspected_segment_ids=inspected_segment_ids)
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    evidence_status_text = _evidence_status_summary_text(evidence_status_summary)
    status_block = f"{evidence_status_text}\n" if evidence_status_text else ""
    return (
        f"Round: {round_number}/{getattr(budget, 'max_rounds', '?')}\n"
        f"Question: {question}\n"
        f"Already inspected segments: {inspected_line}\n"
        f"Uninspected segment candidates: {uninspected_line}\n"
        f"Request at most {getattr(budget, 'max_tool_calls_per_round', 1)} new tool call(s) this round.\n"
        f"{final_round_line}"
        "Scene index:\n"
        f"{scene_index.summary(max_segments=64)}\n"
        f"{status_block}"
        "Evidence ledger:\n"
        f"{ledger_text}"
    )


def _evidence_only_snapshot_block(
    *,
    ledger_text: str,
    evidence_status_summary: Mapping[str, Any] | None = None,
) -> str:
    evidence_status_text = _evidence_status_summary_text(evidence_status_summary)
    status_block = f"{evidence_status_text}\n" if evidence_status_text else ""
    return f"{status_block}Evidence ledger:\n{ledger_text}"


_ROUTE_AGNOSTIC_FINAL_RULES = (
    "- The compact scene index is the default map; do not call video_ls for short indexed videos.",
    "- Use target_coverage when MCQ/QA targets need a segment coverage matrix.",
    "- Use read_segment_detail to expand one selected segment before spending VLM calls.",
    "- Use navigation output as a map, then delegate localized visual reading to one focused evidence tool on one candidate segment.",
    "- Do not spend every round on navigation-only tools; gather evidence-grade visual observations before finalizing.",
    "- Prefer segment_id references; the harness binds video_path/start_sec/end_sec.",
    "- Do not repeat already inspected segments unless the ledger says the prior observation was unusable.",
    "- Continue when evidence is missing, ambiguous, or too coarse.",
    "- Use verify_ledger_answer before finalizing when answer support is uncertain.",
    "- Final answers must cite observation ids from the ledger.",
)

_TARGET_REF_FINAL_RULES = (
    "- target_refs accepts only known registry ids like T1; free text or unknown T<n> ids hard-reject the tool call.",
    "- targets is only for natural-language target text; acceptance requires 0 occurrences of T<n> in legacy targets.",
)

_NO_TARGET_REF_FINAL_RULES = (
    "- No target_refs are registered in this run; do not include target_refs.",
    "- Coverage-local Q<n> labels are not callable ids; keep natural-language text in targets.",
)

_ROUTE_SPECIFIC_FINAL_RULES: dict[str, tuple[str, ...]] = {
    "gist_global": (
        "- For gist/global questions, use global_gist before local decomposition as a sparse topic hint, not an option vote.",
        "- Main-idea answers must compare whole-video coverage; partial ending-only evidence cannot beat a full rise/stability/fall arc.",
    ),
    "temporal_order": (
        "- For order/sequence questions, use target_coverage or scene-index ASR hints to pick a candidate segment, then call locate_targets_in_segment(segment_id, targets=[...]).",
        "- After locate_targets_in_segment returns anchors_for_vlm, call verify_segment_anchors on those anchors before relying on them as evidence.",
        "- After one verify_segment_anchors observation you can call read_timeline_sorted to read the materialized event order.",
    ),
    "needle_local": (
        "- For needle questions, use target_coverage + read_segment_detail to localize the candidate segment.",
        "- When target locations inside a long segment are needed, call locate_targets_in_segment followed by verify_segment_anchors.",
        "- Distinguishing facts should come from one focused visual observation; do not fan out caption_segment over every segment.",
    ),
}

_OPTION_BLIND_FINAL_RULES = (
    "- MCQ choices were rewritten into an option-blind exploration task; do not pass option labels or candidate choice text to local tools.",
    "- Use target_coverage for a target-to-segment coverage matrix, then read_segment_detail / locate_targets_in_segment for selected segments.",
    "- Local VLM tools must openly describe visible/narrated segment content as concrete observations.",
    "- The AnswerAgent will compare cited open facts to the original choices later.",
)

_OPTION_LABELED_FINAL_RULES = (
    "- Multiple-choice answers may use original options for planning, search, target coverage, and evidence comparison.",
    "- Local VLM tools must receive neutral factual prompts; do not pass option labels or complete candidate option text to local workers.",
    "- Local workers must not choose options or emit supported_option; the AnswerAgent maps cited facts to options globally.",
    "- Do not copy quoted option text into JSON string values; refer to option letters instead.",
)


def _final_gate_block(
    *,
    final_round_reserved: bool,
    option_blind: bool = False,
    route: str | None = None,
    include_target_refs: bool = False,
) -> str:
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    lines = list(_ROUTE_AGNOSTIC_FINAL_RULES)
    lines.extend(_TARGET_REF_FINAL_RULES if include_target_refs else _NO_TARGET_REF_FINAL_RULES)
    lines.extend(_ROUTE_SPECIFIC_FINAL_RULES.get(str(route or ""), ()))
    lines.extend(_OPTION_BLIND_FINAL_RULES if option_blind else _OPTION_LABELED_FINAL_RULES)
    lines.append(
        "- Final answers require at least one answer-grade citation from visual tools, indexed ASR/OCR, or QA evidence; navigation-only evidence and locate candidates are insufficient."
    )
    if final_round_line:
        lines.append(final_round_line.strip())
    return "\n".join(lines) + "\n"


def _uninspected_segment_summary(*, scene_index: SceneIndex, inspected_segment_ids: Sequence[str]) -> str:
    inspected = set(inspected_segment_ids)
    candidates = [segment.segment_id for segment in scene_index.segments if segment.segment_id not in inspected]
    if not candidates:
        return "(none)"
    return ", ".join(candidates[:12])
