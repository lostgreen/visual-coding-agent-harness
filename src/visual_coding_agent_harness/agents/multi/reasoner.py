"""Reasoner agent for hypothesis and answer decisions."""

from __future__ import annotations

from typing import Any, Mapping

from ..workspace_agent import WorkspaceRunResult
from .mutator import WorkspaceMutator
from .protocol import SubGoalBudget, SubGoalConstraint, SubGoalSuccessCriteria


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
        """Emit one scoped verify sub-goal, then answer only from satisfied findings."""

        if self.answer_result is not None:
            return False
        findings = self.mutator.findings()
        satisfied = [finding for finding in findings if finding.status == "satisfied" and finding.memory_ids]
        if satisfied:
            choice = _first_option_id(options) or "A"
            self.answer_result = WorkspaceRunResult(
                answer=choice,
                citations=satisfied[-1].memory_ids,
                confidence="medium",
                rounds=round_number,
                metadata={"status": "final", "strategy": "multi_agent_v0"},
            )
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "answer", "n_sub_goals": 0},
            )
            return True

        active = [goal for goal in self.mutator.sub_goals() if goal.status in {"open", "in_progress"}]
        if active:
            self.workspace.write_trace_event(
                "reasoner_action_emitted",
                {"round": round_number, "action": "wait", "n_sub_goals": len(active)},
            )
            return False

        option_id = _first_option_id(options)
        option_text = options.get(option_id or "", "") if option_id else ""
        claim = option_text or "Find answer-relevant local video evidence."
        self.mutator.create_sub_goal(
            intent="verify",
            constraint=SubGoalConstraint(
                option_id=option_id,
                claim=claim,
                modality_hint=("visual",),
            ),
            budget=SubGoalBudget(max_explores=1, max_verifies=1, max_frames=64),
            success_criteria=SubGoalSuccessCriteria(needs_visual_support=True, needs_option_relation=True),
            parent_question=question,
            created_by="reasoner",
            created_round=round_number,
            rationale="Create the first local verification target before answering.",
        )
        self.workspace.write_trace_event(
            "reasoner_action_emitted",
            {"round": round_number, "action": "emit_sub_goals", "n_sub_goals": 1},
        )
        return True


def _first_option_id(options: Mapping[str, str]) -> str:
    for key in sorted(str(item).strip().upper() for item in options if str(item).strip()):
        return key[:1]
    return ""
