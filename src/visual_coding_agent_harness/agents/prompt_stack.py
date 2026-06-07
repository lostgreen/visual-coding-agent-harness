"""Composable prompt stack for the long-video planner loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..video_index import SceneIndex
from .context_budget import ContextBudgetAllocator, ContextBudgetReport, SlotName
from .question_policy import select_question_playbook
from .skills.specs import skill_catalog_prompt


@dataclass(frozen=True)
class PromptBlock:
    name: str
    title: str
    body: str

    def render(self) -> str:
        return f"# {self.title}\n{self.body.strip()}\n"


def render_prompt_blocks(blocks: Sequence[PromptBlock]) -> str:
    return "\n".join(block.render() for block in blocks).strip()


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
    normalization_notes: Sequence[Any] = (),
    hypothesis_text: str = "",
    reflection_memory: Sequence[str] = (),
    evidence_status_summary: Mapping[str, Any] | None = None,
) -> tuple[str, ContextBudgetReport]:
    slots = compose_replanning_prompt_slots(
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
        evidence_status_summary=evidence_status_summary,
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
    normalization_notes: Sequence[Any] = (),
    hypothesis_text: str = "",
    reflection_memory: Sequence[str] = (),
    evidence_status_summary: Mapping[str, Any] | None = None,
) -> dict[SlotName, str]:
    playbook = select_question_playbook(question)
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
        PromptBlock(name="route_playbook", title="Route Playbook", body=playbook.to_prompt()),
        PromptBlock(
            name="skill_catalog",
            title="Skill Catalog",
            body=(
                f"{skill_catalog_prompt()}\n"
                "Select the skill that best matches this case in every planner JSON as `skill`. "
                "Choose from the catalog yourself; the route playbook is guidance, not a skill assignment. "
                "If no catalog skill fits, omit `skill` and use ordinary tool exploration. "
                "The harness validates tool calls and final evidence only against a skill you explicitly select."
            ),
        ),
    ]
    tooling_blocks = [
        PromptBlock(name="tool_schema", title="Tool Schema", body=_tool_schema_block()),
        PromptBlock(name="final_gate", title="Final Gate", body=_final_gate_block(final_round_reserved=final_round_reserved)),
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
        ),
        "feedback": _feedback_slot(
            answer_feedback=answer_feedback,
            normalization_notes=normalization_notes,
            reflection_memory=reflection_memory,
        ),
        "budget": _budget_snapshot_block(
            round_number=round_number,
            budget=budget,
            tool_class_counts=tool_class_counts,
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
) -> list[PromptBlock]:
    playbook = select_question_playbook(question)
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
            body=playbook.to_prompt(),
        ),
        PromptBlock(
            name="skill_catalog",
            title="Skill Catalog",
            body=(
                f"{skill_catalog_prompt()}\n"
                "Select the skill that best matches this case in every planner JSON as `skill`. "
                "Choose from the catalog yourself; the route playbook is guidance, not a skill assignment. "
                "If no catalog skill fits, omit `skill` and use ordinary tool exploration. "
                "The harness validates tool calls and final evidence only against a skill you explicitly select."
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
                    tool_class_counts=tool_class_counts,
                    final_round_reserved=final_round_reserved,
                ),
            ),
            PromptBlock(
                name="tool_schema",
                title="Tool Schema",
                body=_tool_schema_block(),
            ),
            PromptBlock(
                name="final_gate",
                title="Final Gate",
                body=_final_gate_block(final_round_reserved=final_round_reserved),
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


def _tool_schema_block() -> str:
    return (
        "Available tools:\n"
        "- ground_question(query: str, top_k: int = 5, modalities: list = [])\n"
        "- video_ls(query: str = '', max_segments: int = 16, top_k: int = 5)\n"
        "- search_segments(query: str, top_k: int = 5, modalities: list = [])\n"
        "- read_segment(segment_id: str)\n"
        "- expand_window(segment_id: str, before_sec: float = 30.0, after_sec: float = 30.0)\n"
        "- zoom(segment_id: str, target_granularity_sec: float = 60.0)\n"
        "- global_gist(video_path: str, question: str, duration_sec: float, nframes: int = 128, max_pixels: int = 151200, sample_offset_sec: float = 0.0)\n"
        "- caption_segments(segment_ids: list = [], question: str = 'Create a concise search caption for this segment.', nframes: int = 8, max_pixels: int = 151200, fps: float = 0.0, max_segments: int = 3)\n"
        "- ingest_segment_metadata(segment_id: str, low_fps_caption: str = '', asr_text: str = '', ocr_text: str = '', entities: list = [])\n"
        "- summarize_ledger_evidence(max_claims: int = 5)\n"
        "- verify_ledger_answer(answer: str, ledger_text: str = '', question: str = '', candidate_options: list = [], min_score: float = 0.6, required_citations: list = [])\n"
        "- view_observation(obs_id: str, line_range: tuple | None = None)\n"
        "- grep_evidence(pattern: str, in_field: str = 'claim')\n"
        "- query_evidence_table(filter: dict)\n"
        "- read_timeline_sorted()\n"
        "- read_hypothesis()\n"
        "- update_hypothesis_slot(slot_name: str, status: str, evidence_obs_id: str = '')\n"
        "- vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = '', nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- inspect_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, candidate_options: list = [], nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- qa_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 128, max_pixels: int = 151200, fps: float = 0.0)"
    )


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
    counts = tool_class_counts or {}
    remaining_budget_line = (
        "free exploration mode; no per-class tool budget, only emergency round/tool-call caps"
        if getattr(budget, "free_exploration", False)
        else (
            f"cheap={max(0, int(getattr(budget, 'cheap_tool_budget', 0)) - int(counts.get('cheap', 0)))}, "
            f"expensive={max(0, int(getattr(budget, 'expensive_tool_budget', 0)) - int(counts.get('expensive', 0)))}, "
            f"verifier={max(0, int(getattr(budget, 'verifier_tool_budget', 0)) - int(counts.get('verifier', 0)))}"
        )
    )
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
        "Cheap navigation tools and expensive VLM tools have separate budgets unless free exploration mode is active.\n"
        "In free exploration mode, prioritize answer quality: keep using tools until evidence is sufficient, then finalize with citations.\n"
        f"Remaining tool budgets: {remaining_budget_line}.\n"
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
) -> str:
    uninspected_line = _uninspected_segment_summary(scene_index=scene_index, inspected_segment_ids=inspected_segment_ids)
    return (
        f"Question: {question}\n"
        f"Uninspected segment candidates: {uninspected_line}\n"
        "Compact scene index:\n"
        f"{scene_index.summary(max_segments=64)}"
    )


def _budget_snapshot_block(
    *,
    round_number: int,
    budget: Any,
    tool_class_counts: Mapping[str, int] | None,
    final_round_reserved: bool,
) -> str:
    counts = tool_class_counts or {}
    remaining_budget_line = (
        "free exploration mode; no per-class tool budget, only emergency round/tool-call caps"
        if getattr(budget, "free_exploration", False)
        else (
            f"cheap={max(0, int(getattr(budget, 'cheap_tool_budget', 0)) - int(counts.get('cheap', 0)))}, "
            f"expensive={max(0, int(getattr(budget, 'expensive_tool_budget', 0)) - int(counts.get('expensive', 0)))}, "
            f"verifier={max(0, int(getattr(budget, 'verifier_tool_budget', 0)) - int(counts.get('verifier', 0)))}"
        )
    )
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    return (
        f"Round: {round_number}/{getattr(budget, 'max_rounds', '?')}\n"
        f"Request at most {getattr(budget, 'max_tool_calls_per_round', 1)} new tool call(s) this round.\n"
        "Cheap navigation tools and expensive VLM tools have separate budgets unless free exploration mode is active.\n"
        "In free exploration mode, prioritize answer quality: keep using tools until evidence is sufficient, then finalize with citations.\n"
        f"Remaining tool budgets: {remaining_budget_line}.\n"
        f"{final_round_line}"
    )


def _feedback_slot(
    *,
    answer_feedback: Sequence[str],
    normalization_notes: Sequence[Any],
    reflection_memory: Sequence[str],
) -> str:
    blocks = []
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


def _normalization_notes_body(notes: Sequence[Any]) -> str:
    lines = ["Your previous program was modified by the harness:"]
    for note in list(notes)[:8]:
        reason = _note_attr(note, "reason")
        tool = _note_attr(note, "tool")
        original = _note_mapping(note, "original")
        resolved = _note_mapping(note, "resolved")
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
    counts = tool_class_counts or {}
    remaining_budget_line = (
        "free exploration mode; no per-class tool budget, only emergency round/tool-call caps"
        if getattr(budget, "free_exploration", False)
        else (
            f"cheap={max(0, int(getattr(budget, 'cheap_tool_budget', 0)) - int(counts.get('cheap', 0)))}, "
            f"expensive={max(0, int(getattr(budget, 'expensive_tool_budget', 0)) - int(counts.get('expensive', 0)))}, "
            f"verifier={max(0, int(getattr(budget, 'verifier_tool_budget', 0)) - int(counts.get('verifier', 0)))}"
        )
    )
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
        "Cheap navigation tools and expensive VLM tools have separate budgets unless free exploration mode is active.\n"
        "In free exploration mode, prioritize answer quality: keep using tools until evidence is sufficient, then finalize with citations.\n"
        f"Remaining tool budgets: {remaining_budget_line}.\n"
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


def _final_gate_block(*, final_round_reserved: bool) -> str:
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    return (
        "- Use video_ls first for open-ended description tasks or when the relevant segment is unclear.\n"
        "- For gist/global questions, use global_gist before local decomposition as a sparse topic hint, not an option vote.\n"
        "- caption_segments is offline VideoMap cache building; avoid it in online reasoning unless the cache/indexes are empty.\n"
        "- Use navigation output as a map, then delegate localized visual reading to vision_read or inspect_segment on one candidate segment.\n"
        "- Use zoom when a coarse segment is relevant but too long; then call vision_read or inspect_segment with the returned child segment_id and start_sec/end_sec.\n"
        "- Do not spend every round on navigation-only tools; gather visual evidence before finalizing.\n"
        "- Multiple-choice answers must use vision_read or inspect_segment on a localized candidate before finalizing; candidate options are only fact-finding hints.\n"
        "- Local workers must not choose options or emit supported_option; the AnswerAgent maps cited facts to options globally.\n"
        "- Main-idea answers must compare whole-video coverage; partial ending-only evidence cannot beat a full rise/stability/fall arc.\n"
        '- JSON safety: candidate_options in JSON should be option letters only, for example ["A", "B", "C", "D"]; the harness restores full option text.\n'
        "- Do not copy quoted option text into JSON string values; refer to option letters instead.\n"
        "- Final answers require at least one non-navigation visual observation from vision_read, inspect_segment, caption_segment, or qa_segment; navigation-only evidence is insufficient.\n"
        "- Prefer segment_id references; the harness binds video_path/start_sec/end_sec.\n"
        "- Do not repeat already inspected segments unless the ledger says the prior observation was unusable.\n"
        "- Continue when evidence is missing, ambiguous, or too coarse.\n"
        "- Use verify_ledger_answer before finalizing when answer support is uncertain.\n"
        "- Final answers must cite observation ids from the ledger.\n"
        f"{final_round_line}"
    )


def _uninspected_segment_summary(*, scene_index: SceneIndex, inspected_segment_ids: Sequence[str]) -> str:
    inspected = set(inspected_segment_ids)
    candidates = [segment.segment_id for segment in scene_index.segments if segment.segment_id not in inspected]
    if not candidates:
        return "(none)"
    return ", ".join(candidates[:12])
