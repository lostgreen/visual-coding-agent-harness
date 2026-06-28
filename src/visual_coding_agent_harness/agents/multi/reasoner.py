"""Reasoner agent for hypothesis and answer decisions."""

from __future__ import annotations

from typing import Any, Mapping

from ..workspace_agent import WorkspaceRunResult
from .mutator import WorkspaceMutator
from .protocol import SubGoalBudget, SubGoalConstraint, SubGoalSuccessCriteria

MAX_OPEN_SUB_GOALS = 3
ANSWER_GROUNDING_KINDS = frozenset({"visual_support"})


class ReasonerAgent:
    """Minimal Reasoner implementation for the first multi-agent runner slice."""

    def __init__(
        self,
        *,
        backend: Any,
        mutator: WorkspaceMutator,
        workspace: Any,
        video_map: Any,
        log_root: Any,
    ) -> None:
        self.backend = backend
        self.mutator = mutator
        self.workspace = workspace
        self.video_map = video_map
        self.log_root = log_root
        self.answer_result: WorkspaceRunResult | None = None

    def step(self, *, round_number: int, question: str, options: Mapping[str, str]) -> bool:
        """Schedule option checks, then answer only from option-bound positive evidence."""

        if self.answer_result is not None:
            return False
        option_ids = _option_ids(options)
        sub_goals = self.mutator.sub_goals()
        active = [goal for goal in sub_goals if goal.status in {"open", "in_progress"}]
        active_options = {
            str(goal.constraint.option_id or "").strip().upper()[:1]
            for goal in active
            if str(goal.constraint.option_id or "").strip()
        }
        tested_options = {
            str(goal.constraint.option_id or "").strip().upper()[:1]
            for goal in sub_goals
            if goal.status in {"done", "abandoned"} and str(goal.constraint.option_id or "").strip()
        }
        untested_options = [option_id for option_id in option_ids if option_id not in tested_options | active_options]
        if untested_options and len(active) < MAX_OPEN_SUB_GOALS:
            created = 0
            for option_id in untested_options[: MAX_OPEN_SUB_GOALS - len(active)]:
                self._create_option_sub_goal(
                    option_id=option_id,
                    option_text=options.get(option_id, ""),
                    question=question,
                    round_number=round_number,
                )
                created += 1
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "emit_sub_goals", "n_sub_goals": created},
            )
            return created > 0

        if active:
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "wait", "n_sub_goals": len(active)},
            )
            return False

        findings = self.mutator.findings()
        scored_options = _score_positive_findings(findings, self.workspace, options=options)
        if scored_options:
            choice, citations = scored_options[0]
            self.answer_result = WorkspaceRunResult(
                answer=choice,
                citations=citations,
                confidence="medium",
                rounds=round_number,
                metadata={"status": "final", "strategy": "multi_agent_v0"},
            )
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "answer", "n_sub_goals": 0},
            )
            return True

        if option_ids:
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "wait", "n_sub_goals": 0, "reason": "no_positive_evidence"},
            )
            return False

        option_id = ""
        option_text = options.get(option_id or "", "") if option_id else ""
        claim = option_text or "Find answer-relevant local video evidence."
        self._create_option_sub_goal(
            option_id=option_id,
            option_text=claim,
            question=question,
            round_number=round_number,
        )
        self.workspace.write_trace_event(
            "reasoner_action_emitted",
            {"round": round_number, "action": "emit_sub_goals", "n_sub_goals": 1},
        )
        return True

    def _create_option_sub_goal(
        self,
        *,
        option_id: str,
        option_text: str,
        question: str,
        round_number: int,
    ) -> None:
        claim = _option_claim(question=question, option_id=option_id, option_text=option_text)
        self.mutator.create_sub_goal(
            intent="verify",
            constraint=SubGoalConstraint(
                option_id=option_id or None,
                claim=claim,
                modality_hint=("visual",),
            ),
            budget=SubGoalBudget(max_explores=1, max_verifies=1, max_frames=64),
            success_criteria=SubGoalSuccessCriteria(needs_visual_support=True, needs_option_relation=True),
            parent_question=question,
            created_by="reasoner",
            created_round=round_number,
            rationale=(
                f"Verify option {option_id} before choosing an answer."
                if option_id
                else "Create a local verification target before answering."
            ),
        )


def _score_positive_findings(findings: Any, workspace: Any, *, options: Mapping[str, str]) -> list[tuple[str, tuple[str, ...]]]:
    option_ids = set(_option_ids(options))
    memory_by_id = {entry.entry_id: entry for entry in workspace.memory_entries()}
    scores: dict[str, int] = {}
    citations: dict[str, list[str]] = {}
    for finding in findings:
        if finding.status != "satisfied":
            continue
        for memory_id in finding.memory_ids:
            entry = memory_by_id.get(str(memory_id))
            if entry is None or entry.kind not in ANSWER_GROUNDING_KINDS:
                continue
            option_id = str(entry.supports_option or "").strip().upper()[:1]
            if not option_id or (option_ids and option_id not in option_ids):
                continue
            scores[option_id] = scores.get(option_id, 0) + _confidence_score(entry.confidence)
            citations.setdefault(option_id, []).append(entry.entry_id)
    option_order = {option_id: index for index, option_id in enumerate(_option_ids(options))}
    ranked = sorted(
        scores,
        key=lambda option_id: (-scores[option_id], option_order.get(option_id, 999), option_id),
    )
    return [(option_id, tuple(citations[option_id])) for option_id in ranked]


def _option_claim(*, question: str, option_id: str, option_text: str) -> str:
    question_text = _question_stem(question)
    option_body = " ".join(str(option_text or "").split())
    if option_id and option_body:
        return f"{question_text} Option {option_id}: {option_body}."
    return option_body or question_text or "Find answer-relevant local video evidence."


def _question_stem(question: str) -> str:
    text = " ".join(str(question or "").split())
    for marker in (" Options:", " options:", "\nOptions:", "\noptions:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
            break
    return text


def _confidence_score(confidence: Any) -> int:
    normalized = str(confidence or "").strip().lower()
    if normalized == "high":
        return 3
    if normalized == "low":
        return 1
    return 2


def _option_ids(options: Mapping[str, str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in sorted(str(item).strip().upper() for item in options if str(item).strip()):
        option_id = key[:1]
        if option_id and option_id not in seen:
            seen.add(option_id)
            ordered.append(option_id)
    return ordered
