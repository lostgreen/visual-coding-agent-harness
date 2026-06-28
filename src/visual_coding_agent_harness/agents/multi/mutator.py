"""Validated sidecar writes for multi-agent workspace state."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Mapping

from ...workspace import EvidenceWorkspace
from .protocol import (
    Finding,
    FindingStatus,
    SubGoal,
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalIntent,
    SubGoalStatus,
    SubGoalSuccessCriteria,
)


_ALLOWED_TRANSITIONS: dict[SubGoalStatus, set[SubGoalStatus]] = {
    "open": {"in_progress", "abandoned"},
    "in_progress": {"done", "abandoned"},
    "done": set(),
    "abandoned": set(),
}


class WorkspaceMutator:
    """Owns all multi-agent sidecar state changes for an EvidenceWorkspace."""

    def __init__(self, workspace: EvidenceWorkspace) -> None:
        self.workspace = workspace
        self.root = workspace.root / "multi_agent"
        self.root.mkdir(parents=True, exist_ok=True)

    def sub_goals(self) -> list[SubGoal]:
        latest: dict[str, SubGoal] = {}
        order: list[str] = []
        for row in self._read_jsonl("sub_goals.jsonl"):
            sub_goal = SubGoal.from_dict(row)
            if sub_goal.sub_goal_id not in latest:
                order.append(sub_goal.sub_goal_id)
            latest[sub_goal.sub_goal_id] = sub_goal
        return [latest[sub_goal_id] for sub_goal_id in order if sub_goal_id in latest]

    def findings(self) -> list[Finding]:
        return [Finding.from_dict(row) for row in self._read_jsonl("findings.jsonl")]

    def create_sub_goal(
        self,
        *,
        intent: SubGoalIntent,
        constraint: SubGoalConstraint,
        budget: SubGoalBudget,
        success_criteria: SubGoalSuccessCriteria,
        parent_question: str,
        created_by: str,
        created_round: int,
        rationale: str = "",
    ) -> SubGoal:
        sub_goal = SubGoal(
            sub_goal_id=self._next_id("sg", len(self.sub_goals()) + 1),
            intent=intent,
            constraint=constraint,
            budget=budget,
            success_criteria=success_criteria,
            parent_question=parent_question,
            created_by=created_by,
            created_round=created_round,
            status="open",
            rationale=rationale[:400],
            updated_round=created_round,
        )
        self._append_jsonl("sub_goals.jsonl", sub_goal.to_dict())
        self.workspace.write_trace_event(
            "sub_goal_created",
            {
                "sub_goal_id": sub_goal.sub_goal_id,
                "intent": sub_goal.intent,
                "constraint": sub_goal.constraint.__dict__,
                "parent_round": created_round,
            },
        )
        return sub_goal

    def claim_next_open_sub_goal(self, *, agent_id: str, round_number: int) -> SubGoal | None:
        for sub_goal in self.sub_goals():
            if sub_goal.status == "open":
                return self.transition_sub_goal(
                    sub_goal.sub_goal_id,
                    to_status="in_progress",
                    round_number=round_number,
                    assigned_to=agent_id,
                )
        return None

    def transition_sub_goal(
        self,
        sub_goal_id: str,
        *,
        to_status: SubGoalStatus,
        round_number: int,
        assigned_to: str | None = None,
    ) -> SubGoal:
        current = self._get_sub_goal(sub_goal_id)
        allowed = _ALLOWED_TRANSITIONS[current.status]
        if to_status not in allowed:
            raise ValueError(f"invalid sub_goal transition: {current.status} -> {to_status}")
        updated = replace(
            current,
            status=to_status,
            assigned_to=current.assigned_to if assigned_to is None else assigned_to,
            updated_round=round_number,
        )
        self._append_jsonl("sub_goals.jsonl", updated.to_dict())
        self.workspace.write_trace_event(
            "sub_goal_transitioned",
            {
                "sub_goal_id": sub_goal_id,
                "from": current.status,
                "to": to_status,
                "round": round_number,
            },
        )
        return updated

    def report_finding(
        self,
        *,
        sub_goal_id: str,
        status: FindingStatus,
        memory_ids: tuple[str, ...],
        coverage: tuple[float, float],
        notes_for_planner: str,
        cost: Mapping[str, int],
        created_round: int,
    ) -> Finding:
        current = self._get_sub_goal(sub_goal_id)
        if current.status != "in_progress":
            raise ValueError(f"cannot report finding for sub_goal status {current.status}")
        finding = Finding(
            finding_id=self._next_id("find", len(self.findings()) + 1),
            sub_goal_id=sub_goal_id,
            status=status,
            memory_ids=tuple(memory_ids),
            coverage=coverage,
            notes_for_planner=notes_for_planner[:800],
            cost=dict(cost),
            created_round=created_round,
        )
        self._append_jsonl("findings.jsonl", finding.to_dict())
        final_status: SubGoalStatus = "abandoned" if status == "infeasible" else "done"
        self.transition_sub_goal(sub_goal_id, to_status=final_status, round_number=created_round)
        self.workspace.write_trace_event(
            "finding_created",
            {
                "finding_id": finding.finding_id,
                "sub_goal_id": sub_goal_id,
                "status": status,
                "cost": dict(cost),
            },
        )
        return finding

    def _get_sub_goal(self, sub_goal_id: str) -> SubGoal:
        for sub_goal in reversed(self.sub_goals()):
            if sub_goal.sub_goal_id == sub_goal_id:
                return sub_goal
        raise ValueError(f"unknown sub_goal_id: {sub_goal_id}")

    def _read_jsonl(self, filename: str) -> list[Mapping[str, object]]:
        path = self.root / filename
        if not path.exists():
            return []
        rows: list[Mapping[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _append_jsonl(self, filename: str, payload: Mapping[str, object]) -> None:
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _next_id(prefix: str, index: int) -> str:
        return f"{prefix}_{index:04d}"
