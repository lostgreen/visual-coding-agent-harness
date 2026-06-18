"""Composable prompt stack for the long-video planner loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..video_index import SceneIndex
from .context_budget import ContextBudgetAllocator, ContextBudgetReport, SlotName
from .prompt_frames import PromptFrame, PromptFrameLedger
from .question_policy import QuestionPlaybook, extract_candidate_options, select_question_playbook
from .skills.specs import (
    allowed_actions_for_skill,
    render_skill_playbook_for_prompt,
    skill_catalog_prompt,
    skill_has_playbook,
)


@dataclass(frozen=True)
class PromptBlock:
    name: str
    title: str
    body: str

    def render(self) -> str:
        return f"# {self.title}\n{self.body.strip()}\n"


@dataclass(frozen=True)
class PromptPair:
    system_prompt: str
    user_prompt: str
    context_report: ContextBudgetReport


def render_prompt_blocks(blocks: Sequence[PromptBlock]) -> str:
    return "\n".join(block.render() for block in blocks).strip()


SLOT_OF_BLOCK: Mapping[str, SlotName] = {
    "base_identity": "task",
    "route_playbook": "task",
    "skill_catalog": "task",
    "trajectory_snapshot": "trajectory",
    "hypothesis": "hypothesis",
    "memory_snapshot": "evidence",
    "uncommitted_observations": "evidence",
    "evidence_snapshot": "evidence",
    "scene_index_snapshot": "scene_index",
    "normalization_notes": "feedback",
    "answer_feedback": "feedback",
    "diagnostic_repair_hint": "feedback",
    "reflection_memory": "feedback",
    "pending_inference": "feedback",
    "budget_snapshot": "budget",
    "tool_schema": "tooling",
    "final_gate": "tooling",
    "response_contract": "tooling",
}


_RENDERED_SLOT_BLOCKS = frozenset(
    {
        "base_identity",
        "route_playbook",
        "skill_catalog",
        "memory_snapshot",
        "uncommitted_observations",
        "evidence_snapshot",
        "normalization_notes",
        "answer_feedback",
        "diagnostic_repair_hint",
        "reflection_memory",
        "pending_inference",
        "tool_schema",
        "final_gate",
        "response_contract",
    }
)


def _blocks_to_slots(blocks: Sequence[PromptBlock]) -> dict[SlotName, str]:
    grouped: dict[SlotName, list[str]] = {}
    for block in blocks:
        slot = SLOT_OF_BLOCK.get(block.name)
        if slot is None:
            continue
        rendered = block.render().strip() if block.name in _RENDERED_SLOT_BLOCKS else block.body.strip()
        grouped.setdefault(slot, []).append(rendered)
    slots: dict[SlotName, str] = {}
    for slot in ("task", "trajectory", "hypothesis", "evidence", "scene_index", "feedback", "budget", "tooling"):
        parts = grouped.get(slot, [])
        slots[slot] = "\n\n".join(part for part in parts if part).strip() if parts else "(none)"
    return slots


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
    memory_snapshot: str = "",
    uncommitted_observations: str = "",
    evidence_status_summary: Mapping[str, Any] | None = None,
    recent_tool_outputs: Sequence[Mapping[str, Any]] = (),
    exhausted_tools: frozenset[str] | None = None,
    active_skill: str | None = None,
    route: str | None = None,
    target_hints: Sequence[str] = (),
    target_ref_descriptions: Sequence[str] = (),
    projection_status: Mapping[str, Any] | None = None,
    diagnostic_repair_hint: str | None = None,
    requested_tool_names: Sequence[str] = (),
) -> tuple[str, ContextBudgetReport]:
    blocks = compose_replanning_prompt_blocks(
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
        memory_snapshot=memory_snapshot,
        uncommitted_observations=uncommitted_observations,
        evidence_status_summary=evidence_status_summary,
        recent_tool_outputs=recent_tool_outputs,
        exhausted_tools=exhausted_tools,
        active_skill=active_skill,
        route=route,
        target_hints=target_hints,
        target_ref_descriptions=target_ref_descriptions,
        projection_status=projection_status,
        diagnostic_repair_hint=diagnostic_repair_hint,
        requested_tool_names=requested_tool_names,
    )
    slots = _blocks_to_slots(blocks)
    allocated, report = allocator.allocate(
        slots,
        ctx={
            "round": round_number,
            "active_followup_target_query": str(answer_feedback[0]) if answer_feedback else "",
        },
    )
    return _join_slots(allocated), report


def compose_planner_prompts(
    *,
    prompt_role_split_enabled: bool,
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
    memory_snapshot: str = "",
    uncommitted_observations: str = "",
    evidence_status_summary: Mapping[str, Any] | None = None,
    recent_tool_outputs: Sequence[Mapping[str, Any]] = (),
    exhausted_tools: frozenset[str] | None = None,
    active_skill: str | None = None,
    route: str | None = None,
    target_hints: Sequence[str] = (),
    target_ref_descriptions: Sequence[str] = (),
    projection_status: Mapping[str, Any] | None = None,
    diagnostic_repair_hint: str | None = None,
    prompt_frame_ledger: PromptFrameLedger | None = None,
    requested_tool_names: Sequence[str] = (),
) -> PromptPair:
    if not prompt_role_split_enabled:
        prompt, report = build_replanning_prompt(
            question=question,
            scene_index=scene_index,
            ledger_text=ledger_text,
            round_number=round_number,
            budget=budget,
            allocator=allocator,
            inspected_segment_ids=inspected_segment_ids,
            tool_class_counts=tool_class_counts,
            final_round_reserved=final_round_reserved,
            answer_feedback=answer_feedback,
            pending_inferences=pending_inferences,
            normalization_notes=normalization_notes,
            hypothesis_text=hypothesis_text,
            reflection_memory=reflection_memory,
            memory_snapshot=memory_snapshot,
            uncommitted_observations=uncommitted_observations,
            evidence_status_summary=evidence_status_summary,
            recent_tool_outputs=recent_tool_outputs,
            exhausted_tools=exhausted_tools,
            active_skill=active_skill,
            route=route,
            target_hints=target_hints,
            target_ref_descriptions=target_ref_descriptions,
                projection_status=projection_status,
                diagnostic_repair_hint=diagnostic_repair_hint,
                requested_tool_names=requested_tool_names,
            )
        return PromptPair(system_prompt="", user_prompt=prompt, context_report=report)
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
        pending_inferences=pending_inferences,
        normalization_notes=normalization_notes,
        hypothesis_text=hypothesis_text,
        reflection_memory=reflection_memory,
        memory_snapshot=memory_snapshot,
        uncommitted_observations=uncommitted_observations,
        evidence_status_summary=evidence_status_summary,
        recent_tool_outputs=recent_tool_outputs,
        exhausted_tools=exhausted_tools,
        active_skill=active_skill,
        route=route,
        target_hints=target_hints,
        target_ref_descriptions=target_ref_descriptions,
        projection_status=projection_status,
        diagnostic_repair_hint=diagnostic_repair_hint,
        requested_tool_names=requested_tool_names,
    )
    slots = _blocks_to_slots(blocks)
    if prompt_frame_ledger is not None:
        slots = _apply_prompt_frame_ledger(slots, blocks, prompt_frame_ledger)
    allocated, report = allocator.allocate(
        slots,
        ctx={
            "round": round_number,
            "active_followup_target_query": str(answer_feedback[0]) if answer_feedback else "",
        },
    )
    legacy_prompt = _join_slots(allocated)
    split_marker = "\n\n## Trajectory\n"
    if split_marker not in legacy_prompt:
        return PromptPair(system_prompt=legacy_prompt, user_prompt="", context_report=report)
    system_prompt, user_tail = legacy_prompt.split(split_marker, 1)
    return PromptPair(system_prompt=system_prompt, user_prompt=f"## Trajectory\n{user_tail}", context_report=report)


def _apply_prompt_frame_ledger(
    slots: Mapping[SlotName, str],
    blocks: Sequence[PromptBlock],
    ledger: PromptFrameLedger,
) -> dict[SlotName, str]:
    framed_blocks = {_frame.frame_id: _frame for _frame in _prompt_frames_for_blocks(blocks)}
    if not framed_blocks:
        return dict(slots)
    rendered: dict[SlotName, str] = dict(slots)
    for slot, text in slots.items():
        updated = str(text)
        for frame_id, frame in framed_blocks.items():
            rendered_body = f"# {frame.title}\n{frame.body.strip()}"
            if rendered_body in updated:
                updated = updated.replace(rendered_body, ledger.take(frame).strip(), 1)
        rendered[slot] = updated
    return rendered


def _prompt_frames_for_blocks(blocks: Sequence[PromptBlock]) -> tuple[PromptFrame, ...]:
    frames: list[PromptFrame] = []
    for block in blocks:
        if block.name == "skill_catalog":
            frames.append(
                PromptFrame(
                    frame_id="skill_catalog",
                    title=block.title,
                    body=block.body,
                    version="v1",
                )
            )
        elif block.name == "tool_schema":
            frames.append(
                PromptFrame(
                    frame_id="tool_schema",
                    title=block.title,
                    body=block.body,
                    version="v1",
                )
            )
        elif block.name == "route_playbook" and len(block.body) > 1200:
            frames.append(
                PromptFrame(
                    frame_id="route_playbook",
                    title=block.title,
                    body=block.body,
                    version="v1",
                )
            )
    return tuple(frames)


_REQUEST_TOOL_RE = re.compile(r"\brequest_tool\s*:\s*([A-Za-z_][A-Za-z0-9_]*)")


def requested_tool_names_from_rationale(rationale: str) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _REQUEST_TOOL_RE.finditer(str(rationale or "")):
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


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
    memory_snapshot: str = "",
    uncommitted_observations: str = "",
    evidence_status_summary: Mapping[str, Any] | None = None,
    recent_tool_outputs: Sequence[Mapping[str, Any]] = (),
    exhausted_tools: frozenset[str] | None = None,
    active_skill: str | None = None,
    route: str | None = None,
    target_hints: Sequence[str] = (),
    target_ref_descriptions: Sequence[str] = (),
    projection_status: Mapping[str, Any] | None = None,
    diagnostic_repair_hint: str | None = None,
    requested_tool_names: Sequence[str] = (),
) -> dict[SlotName, str]:
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
        pending_inferences=pending_inferences,
        normalization_notes=normalization_notes,
        hypothesis_text=hypothesis_text,
        reflection_memory=reflection_memory,
        memory_snapshot=memory_snapshot,
        uncommitted_observations=uncommitted_observations,
        evidence_status_summary=evidence_status_summary,
        recent_tool_outputs=recent_tool_outputs,
        exhausted_tools=exhausted_tools,
        active_skill=active_skill,
        route=route,
        target_hints=target_hints,
        target_ref_descriptions=target_ref_descriptions,
        projection_status=projection_status,
        diagnostic_repair_hint=diagnostic_repair_hint,
        requested_tool_names=requested_tool_names,
    )
    return _blocks_to_slots(blocks)


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
    pending_inferences: Sequence[str] = (),
    normalization_notes: Sequence[Any] = (),
    hypothesis_text: str = "",
    reflection_memory: Sequence[str] = (),
    memory_snapshot: str = "",
    uncommitted_observations: str = "",
    evidence_status_summary: Mapping[str, Any] | None = None,
    recent_tool_outputs: Sequence[Mapping[str, Any]] = (),
    exhausted_tools: frozenset[str] | None = None,
    active_skill: str | None = None,
    route: str | None = None,
    target_hints: Sequence[str] = (),
    target_ref_descriptions: Sequence[str] = (),
    projection_status: Mapping[str, Any] | None = None,
    diagnostic_repair_hint: str | None = None,
    requested_tool_names: Sequence[str] = (),
) -> list[PromptBlock]:
    playbook = select_question_playbook(question)
    resolved_route = route or playbook.route
    if route and route != playbook.route:
        playbook = QuestionPlaybook(name=str(route), route=str(route))
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
                "Skill selection is your choice; read each skill description and when_to_use guidance to pick.\n"
                "Evidence policy is the harness's responsibility; it does not change automatically with your skill choice.\n"
                "Do not include step-by-step private reasoning in the JSON response."
            ),
        ),
        PromptBlock(
            name="route_playbook",
            title="Route Playbook",
            body=playbook.to_prompt(option_blind=option_blind),
        ),
        PromptBlock(
            name="skill_catalog",
            title="Skill Catalog",
            body=_skill_catalog_block(
                active_skill=active_skill,
                exhausted_tools=exhausted_tools,
                question=question,
                target_hints=target_hints,
                projection_status=projection_status,
                diagnostic_repair_hint=diagnostic_repair_hint,
            ),
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
            name="memory_snapshot",
            title="Memory Snapshot",
            body=memory_snapshot.strip() or "(none)",
        ),
        PromptBlock(
            name="uncommitted_observations",
            title="Uncommitted Observations",
            body=uncommitted_observations.strip() or "(none)",
        ),
        PromptBlock(
            name="evidence_snapshot",
            title="Evidence Snapshot",
            body=_evidence_only_snapshot_block(
                ledger_text=ledger_text,
                evidence_status_summary=evidence_status_summary,
                recent_tool_outputs=recent_tool_outputs,
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
    if pending_inferences:
        rendered_pending = _dedupe_pending_inferences(pending_inferences)
        blocks.append(
            PromptBlock(
                name="pending_inference",
                title="Pending Inference",
                body="\n".join(f"- {item}" for item in rendered_pending[:3]),
            )
        )
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
    if diagnostic_repair_hint:
        blocks.append(
            PromptBlock(
                name="diagnostic_repair_hint",
                title="Last-Round Repair Hint",
                body="Last-round repair hint:\n" + str(diagnostic_repair_hint),
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
                requested_tool_names=requested_tool_names,
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
                    '{"status": "continue", "skill": string, "rationale": string, '
                    '"hypothesis": {"option": "A|B|C|D", "why": string, "missing": [string]}, '
                    '"program": [{"tool": string, "args": object, "assign": string}]}\n'
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
    requested_tool_names: Sequence[str] = (),
) -> str:
    has_registered_refs = bool([item for item in target_ref_descriptions if str(item).strip()])
    all_signatures = list(_tool_schema_signatures(option_blind=option_blind, include_target_refs=has_registered_refs))
    signatures = list(all_signatures)
    allowed = allowed_actions_for_skill(active_skill or "") if active_skill else frozenset()
    if allowed:
        signatures = [
            signature
            for signature in signatures
            if _tool_name_from_signature(signature) in allowed
            or _tool_name_from_signature(signature) in _GLOBAL_PROMPT_TOOL_NAMES
        ]
    requested = _requested_tool_signatures(
        all_signatures=all_signatures,
        existing_signatures=signatures,
        requested_tool_names=requested_tool_names,
        exhausted=exhausted,
    )
    if requested:
        signatures.extend(requested)
    rendered = [_maybe_mark_exhausted(signature, exhausted) for signature in signatures]
    lines = [
        _target_registry_contract_block(target_ref_descriptions),
        "Available tools:",
    ]
    if any(_tool_name_from_signature(signature) == "bind_asr_claim" for signature in signatures):
        lines.append(
            "Use bind_asr_claim to promote indexed ASR cue_ids into supported evidence for registered target_refs."
        )
    lines.extend(f"- {signature}" for signature in rendered)
    if requested:
        lines.append("temporarily widened tools:")
        lines.extend(f"- {signature}" for signature in requested)
    lines.append("<more tools available; request_tool: <exact_tool_name> in rationale to widen>")
    return "\n".join(lines)


_GLOBAL_PROMPT_TOOL_NAMES = frozenset(
    {
        "view_observation",
        "read_observation_detail",
        "grep_evidence",
        "read_timeline_sorted",
        "write_memory",
    }
)


def _requested_tool_signatures(
    *,
    all_signatures: Sequence[str],
    existing_signatures: Sequence[str],
    requested_tool_names: Sequence[str],
    exhausted: frozenset[str],
) -> list[str]:
    existing_names = {_tool_name_from_signature(signature) for signature in existing_signatures}
    requested_names = [str(name).strip() for name in requested_tool_names if str(name).strip()]
    selected: list[str] = []
    for signature in all_signatures:
        tool_name = _tool_name_from_signature(signature)
        if tool_name in existing_names or tool_name in exhausted or tool_name not in requested_names:
            continue
        selected.append(signature)
        existing_names.add(tool_name)
    return selected


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
    bind_asr_claim_schema = "bind_asr_claim(segment_id: str, target_refs: list)" if include_target_refs else ""
    verifier_schema = (
        "verify_ledger_answer(answer: str, question: str = '', min_score: float = 0.6, required_citations: list = [])"
        if option_blind
        else "verify_ledger_answer(answer: str, question: str = '', candidate_options: list = [], min_score: float = 0.6, required_citations: list = [])"
    )
    return (
        "ground_question(query: str, top_k: int = 5, modalities: list = [])",
        target_coverage_schema,
        "search_segments(query: str, top_k: int = 5, modalities: list[caption|asr|ocr|entities] = [], additional_targets: list = [])",
        "read_segment(segment_id: str)",
        read_segment_detail_schema,
        locate_targets_schema,
        verify_anchors_schema,
        *([bind_asr_claim_schema] if bind_asr_claim_schema else []),
        "global_gist(video_path: str, question: str, duration_sec: float, nframes: int = 128, max_pixels: int = 151200, sample_offset_sec: float = 0.0)",
        "summarize_ledger_evidence(max_claims: int = 5)",
        verifier_schema,
        "view_observation(obs_id: str, line_range: tuple | None = None)",
        "read_observation_detail(obs_id: str, line_range: tuple | None = None)",
        "grep_evidence(pattern: str, in_field: str = 'claim')",
        "query_evidence_table(filter: dict)",
        "read_timeline_sorted()",
        "read_hypothesis()",
        "update_hypothesis_slot(slot_name: str, status: str, evidence_obs_id: str = '')",
        "write_memory(kind: str, claim: str, anchors: list, supports_option: str = '', confidence: str = 'medium', previous_memory_refs: list = [], tags: list = [])",
        "vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, additional_targets: list = [], event_label: str = '', nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)",
        inspect_schema,
        "caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, additional_targets: list = [], nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)",
        "qa_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)",
    )


def _skill_catalog_block(
    *,
    active_skill: str | None,
    exhausted_tools: frozenset[str] | None,
    question: str = "",
    target_hints: Sequence[str] = (),
    projection_status: Mapping[str, Any] | None = None,
    diagnostic_repair_hint: str | None = None,
) -> str:
    lines = [skill_catalog_prompt(exhausted_tools=exhausted_tools, active_skill_id=active_skill or "")]
    if active_skill:
        playbook_block = render_skill_playbook_for_prompt(
            active_skill,
            option_labels=extract_candidate_options(question),
            central_subjects=target_hints,
            projection_status=projection_status,
            diagnostic_repair_hint=diagnostic_repair_hint,
        )
        if playbook_block:
            lines.extend(["", playbook_block])
        lines.extend(
            [
                "# Active Skill",
                f"current_skill: {active_skill}",
                "last_round_rationale: (initial round)",
                (
                    'To switch skill: set "skill" field in your JSON to the new name and explain the switch '
                    'in "rationale". The harness will accept or reject the switch based on evidence-state compatibility.'
                ),
                'If no specialized skill fits, choose "general_exploration".',
            ]
        )
        if skill_has_playbook(active_skill):
            lines.append("Suggested actions are advisory; valid non-suggested tools may be used with a trace note.")
        else:
            lines.append("The tool schema below is filtered to the effective skill.")
    else:
        lines.append(
            "Select the skill that best matches this case in every planner JSON as `skill`. "
            "Choose from the catalog yourself; the route playbook is guidance, not a skill assignment. "
            'If no catalog skill fits, choose "general_exploration".'
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
        f"Request at most {getattr(budget, 'max_tool_calls_per_round', 1)} new tool call(s) this round.\n"
        f"{final_round_line}"
    )


def _feedback_slot(
    *,
    answer_feedback: Sequence[str],
    pending_inferences: Sequence[str] = (),
    normalization_notes: Sequence[Any],
    reflection_memory: Sequence[str],
    diagnostic_repair_hint: str | None = None,
) -> str:
    blocks = []
    if pending_inferences:
        rendered_pending = _dedupe_pending_inferences(pending_inferences)
        blocks.append(
            PromptBlock(
                name="pending_inference",
                title="Pending Inference",
                body="\n".join(f"- {item}" for item in rendered_pending[:3]),
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
    if diagnostic_repair_hint:
        blocks.append(
            PromptBlock(
                name="diagnostic_repair_hint",
                title="Last-Round Repair Hint",
                body="Last-round repair hint:\n" + str(diagnostic_repair_hint),
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


def _dedupe_pending_inferences(pending_inferences: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for item in pending_inferences:
        text = " ".join(str(item or "").split())
        if not text:
            continue
        if text not in counts:
            order.append(text)
            counts[text] = 0
        counts[text] += 1
    rendered = []
    for text in order:
        count = counts[text]
        rendered.append(f"{text} previous suggestion unchanged (x{count} rounds)" if count > 1 else text)
    return rendered


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
        if bool(output.get("in_evidence_table")):
            raw_output = output.get("raw_output", {})
            raw_map = raw_output if isinstance(raw_output, Mapping) else {}
            segment_id = str(output.get("segment_id") or raw_map.get("segment_id") or "-").strip() or "-"
            modality = str(output.get("modality") or raw_map.get("modality") or "-").strip() or "-"
            verdict = str(output.get("verdict") or raw_map.get("verdict") or "supported").strip() or "supported"
            lines.append(
                f"[obs:{obs_id or '?'}] segment={segment_id} modality={modality} verdict={verdict} "
                "-> see workspace/evidence_table.jsonl"
            )
            if claim:
                lines.append(f"claim: {claim}")
            continue
        if claim:
            lines.append(f"claim: {claim}")
        raw_output = output.get("raw_output", {})
        if raw_output:
            compact = json.dumps(raw_output, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            if len(compact) > 600:
                compact = compact[:597].rstrip() + "..."
            lines.append(f"raw_output: {compact}")
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


def _evidence_only_snapshot_block(
    *,
    ledger_text: str,
    evidence_status_summary: Mapping[str, Any] | None = None,
    recent_tool_outputs: Sequence[Mapping[str, Any]] = (),
) -> str:
    evidence_status_text = _evidence_status_summary_text(evidence_status_summary)
    lines: list[str] = []
    if evidence_status_text:
        lines.append(evidence_status_text)
    recent_outputs_text = _recent_tool_outputs_block(recent_tool_outputs)
    if recent_outputs_text:
        lines.append(recent_outputs_text)
    lines.append("Evidence ledger:\n" + (ledger_text or "(none)"))
    return "\n".join(lines)


_ROUTE_AGNOSTIC_FINAL_RULES = (
    "- Use navigation output as a map, then delegate localized visual reading to one focused evidence tool on one candidate segment.",
    "- Do not spend every round on navigation-only tools; gather evidence-grade visual observations before finalizing.",
    "- Prefer segment_id references; the harness binds video_path/start_sec/end_sec.",
    "- Do not repeat already inspected segments unless the ledger says the prior observation was unusable.",
    "- Continue when evidence is missing, ambiguous, or too coarse.",
    "- Use verify_ledger_answer before finalizing when answer support is uncertain.",
    "- Final answers must cite observation ids from the ledger.",
)

_TARGET_REF_FINAL_RULES = (
    "- target_refs accepts only exact known registry ids like T1; option letters, Q<n> rows, lowercase ids, canonical text, and free text hard-reject the tool call.",
    "- When both target_refs and targets are present, target_refs are source of truth and targets is audit-only free text.",
    "- additional_targets is allowed only on discovery calls: search_segments(query), vision_read(ask_for), and caption_segment(question).",
    "- additional_targets is banned on target_coverage, read_segment_detail, verify_ledger_answer, verify_segment_anchors, and locate_targets_in_segment.",
)

_NO_TARGET_REF_FINAL_RULES = (
    "- No target_refs are registered in this run; do not include target_refs.",
    "- Coverage-local Q<n> labels are not callable ids; keep natural-language text in targets.",
)

_ROUTE_SPECIFIC_FINAL_RULES: dict[str, tuple[str, ...]] = {
    "gist_global": (
        "- For gist/global questions, use global_gist before local decomposition as a sparse topic hint, not an option vote.",
        "- Main-idea answers must compare cited whole-video coverage; partial coverage cannot beat a fuller supported account.",
    ),
    "temporal_order": (
        "- For order/sequence questions, use target_coverage or scene-index ASR hints to pick a candidate segment, then call locate_targets_in_segment(segment_id, targets=[...]).",
        "- A complete contiguous ASR enumeration may be answer-grade order evidence; promote the transcript sequence when route_kind=ordered_list_transcript_complete.",
        "- Use focused vision only when the ASR list is partial, ambiguous, contradicted, or the question explicitly requires onscreen/visible order.",
        "- If locate_targets_in_segment returns recommended_next_actions with route_kind=focused_ordered_list_vision, execute that focused vision_read before anchor verification.",
        "- Use verify_segment_anchors only for separate individual-event anchors; do not use it as the main route for a single ordered-list scene.",
        "- For narrated biography/life-order claims, use read_segment_detail(promote_answer_evidence=true) instead of visual-verifying abstract narrated facts.",
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
        "- Use Memory as your working notebook: when tool output has useful ASR/OCR/visual/caption/retrieval content, call write_memory with real anchor ids."
    )
    lines.append(
        "- Final answers require at least one citation to a real memory id or observation id; prefer citing Memory entries backed by direct ASR/OCR/visual/caption anchors."
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
