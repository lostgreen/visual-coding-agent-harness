"""Composable prompt stack for the long-video planner loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..video_index import SceneIndex
from .question_policy import select_question_playbook
from .skills.specs import select_skill


@dataclass(frozen=True)
class PromptBlock:
    name: str
    title: str
    body: str

    def render(self) -> str:
        return f"# {self.title}\n{self.body.strip()}\n"


def render_prompt_blocks(blocks: Sequence[PromptBlock]) -> str:
    return "\n".join(block.render() for block in blocks).strip()


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
    reflection_memory: Sequence[str] = (),
) -> list[PromptBlock]:
    playbook = select_question_playbook(question)
    skill = select_skill(question, route=playbook.route)
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
            name="active_skill",
            title="Active Skill",
            body=(
                f"{skill.prompt_context()}\n"
                "Treat this skill as the current workflow contract. Use planner freedom only to fill missing slots, "
                "recover from failed grounding, or request targeted evidence."
            ),
        ),
        PromptBlock(
            name="tool_schema",
            title="Tool Schema",
            body=_tool_schema_block(),
        ),
        PromptBlock(
            name="evidence_snapshot",
            title="Evidence Snapshot",
            body=_evidence_snapshot_block(
                question=question,
                scene_index=scene_index,
                ledger_text=ledger_text,
                round_number=round_number,
                budget=budget,
                inspected_segment_ids=inspected_segment_ids,
                tool_class_counts=tool_class_counts,
                final_round_reserved=final_round_reserved,
            ),
        ),
    ]
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
                name="final_gate",
                title="Final Gate",
                body=_final_gate_block(final_round_reserved=final_round_reserved),
            ),
            PromptBlock(
                name="response_contract",
                title="Response Contract",
                body=(
                    "Return only JSON with one of these schemas:\n"
                    '{"status": "continue", "rationale": string, "program": [{"tool": string, "args": object, "assign": string}]}\n'
                    '{"status": "final", "answer": string, "citations": [observation_id], "confidence": number}'
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
        "- global_gist(video_path: str, question: str, duration_sec: float, nframes: int = 64, max_pixels: int = 151200)\n"
        "- caption_segments(segment_ids: list = [], question: str = 'Create a concise search caption for this segment.', nframes: int = 8, max_pixels: int = 151200, fps: float = 0.0, max_segments: int = 3)\n"
        "- ingest_segment_metadata(segment_id: str, low_fps_caption: str = '', asr_text: str = '', ocr_text: str = '', entities: list = [])\n"
        "- summarize_ledger_evidence(max_claims: int = 5)\n"
        "- verify_ledger_answer(answer: str, ledger_text: str = '', question: str = '', candidate_options: list = [], min_score: float = 0.6, required_citations: list = [])\n"
        "- vision_read(video_path: str, segment_id: str, start_sec: float, end_sec: float, ask_for: str, event_label: str = '', nframes: int = 16, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- inspect_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, candidate_options: list = [], nframes: int = 16, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- caption_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 8, max_pixels: int = 151200, fps: float = 0.0)\n"
        "- qa_segment(video_path: str, segment_id: str, start_sec: float, end_sec: float, question: str, nframes: int = 8, max_pixels: int = 151200, fps: float = 0.0)"
    )


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
        f"{scene_index.summary(max_segments=64)}\n"
        "Evidence ledger:\n"
        f"{ledger_text}"
    )


def _final_gate_block(*, final_round_reserved: bool) -> str:
    final_round_line = (
        "Reserved final round is active: return final now, or call verify_ledger_answer only if essential.\n"
        if final_round_reserved
        else ""
    )
    return (
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
