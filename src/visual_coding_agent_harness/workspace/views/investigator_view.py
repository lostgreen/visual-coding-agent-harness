"""Compact investigator-facing workspace view."""

from __future__ import annotations

from ...agents.multi import SubGoal, WorkspaceMutator


class InvestigatorView:
    """Render only the assigned sub-goal and scoped progress."""

    def __init__(self, mutator: WorkspaceMutator) -> None:
        self.mutator = mutator

    def render(self, sub_goal: SubGoal) -> str:
        constraint = sub_goal.constraint
        lines = [
            "# Investigator View",
            "",
            "## Current Sub Goal",
            f"- id: {sub_goal.sub_goal_id}",
            f"- intent: {sub_goal.intent}",
            f"- status: {sub_goal.status}",
            f"- claim: {constraint.claim or '(none)'}",
            f"- parent_question: {sub_goal.parent_question}",
            f"- segment_id: {constraint.segment_id or '(any)'}",
            f"- time_range: {constraint.time_range or '(any)'}",
            f"- option_id: {constraint.option_id or '(none)'}",
            "",
            "## Remaining Budget",
            f"- max_explores: {sub_goal.budget.max_explores}",
            f"- max_verifies: {sub_goal.budget.max_verifies}",
            f"- max_frames: {sub_goal.budget.max_frames}",
            "",
            "## Matching Findings",
        ]
        matches = [finding for finding in self.mutator.findings() if finding.sub_goal_id == sub_goal.sub_goal_id]
        if not matches:
            lines.append("- none")
        for finding in matches:
            lines.append(f"- {finding.finding_id}: {finding.status}; {finding.notes_for_planner}")
        return "\n".join(lines)
