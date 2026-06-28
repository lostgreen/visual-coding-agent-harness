"""Compact reasoner-facing workspace view."""

from __future__ import annotations

from ...agents.multi import WorkspaceMutator


class ReasonerView:
    """Render only decision-level state for the Reasoner."""

    def __init__(self, mutator: WorkspaceMutator, *, max_findings: int = 5) -> None:
        self.mutator = mutator
        self.max_findings = int(max_findings)

    def render(self, *, question: str, options: dict[str, str]) -> str:
        lines = ["# Reasoner View", "", "## Question", question.strip(), "", "## Options"]
        for option_id, option_text in sorted(options.items()):
            lines.append(f"- {option_id}: {option_text}")
        lines.extend(["", "## Active Sub Goals"])
        active = [goal for goal in self.mutator.sub_goals() if goal.status in {"open", "in_progress"}]
        if not active:
            lines.append("- none")
        for goal in active[:3]:
            lines.append(
                f"- {goal.sub_goal_id} [{goal.status}] {goal.intent}: "
                f"{goal.constraint.claim or '(no claim)'}"
            )
        lines.extend(["", "## Recent Findings"])
        findings = self.mutator.findings()[-self.max_findings :]
        if not findings:
            lines.append("- none")
        for finding in findings:
            lines.append(
                f"- {finding.finding_id} for {finding.sub_goal_id}: "
                f"{finding.status}; memories={','.join(finding.memory_ids) or 'none'}; "
                f"{finding.notes_for_planner}"
            )
        return "\n".join(lines)
