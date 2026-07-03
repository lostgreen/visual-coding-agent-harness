"""Rule-based EvidenceVerifier for candidate window verification."""

from __future__ import annotations

from typing import Any, Mapping

from ..debug_hooks import maybe_break
from .mutator import WorkspaceMutator
from .protocol import Finding
from .tool_runner import MultiAgentToolRunner


class EvidenceVerifier:
    """Verify one Scout candidate and close its evidence need."""

    def __init__(self, *, registry: Any, mutator: WorkspaceMutator, workspace: Any) -> None:
        self.registry = registry
        self.mutator = mutator
        self.workspace = workspace
        self.tool_runner = MultiAgentToolRunner(registry=registry, workspace=workspace)

    def verify(
        self,
        *,
        sub_goal: Any,
        candidate: Mapping[str, object],
        round_number: int,
        explore_calls: int = 0,
    ) -> Finding:
        candidate_key = str(candidate.get("candidate_key") or "")
        verify_args = self._verify_args(sub_goal, candidate=candidate, candidate_key=candidate_key)
        maybe_break("verifier.before_verify", verifier=self, sub_goal=sub_goal, candidate=candidate, verify_args=verify_args)
        outcome = self.tool_runner.run_tool(
            "verify_window",
            verify_args,
            round_number=round_number,
            sub_goal_id=sub_goal.sub_goal_id,
        )
        status = "satisfied" if _has_decisive_memory(self.workspace, outcome.memory_ids) else "empty"
        cost = {
            "explore_calls": int(explore_calls),
            "verify_calls": 1,
            "tool_calls": int(explore_calls) + 1,
            "frames_read": _frames_read(outcome.raw_output),
            "tokens": 0,
        }
        maybe_break("verifier.after_verify", verifier=self, outcome=outcome, status=status, cost=cost)
        finding = self.mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status=status,  # type: ignore[arg-type]
            memory_ids=outcome.memory_ids,
            coverage=(0.0, 0.0),
            notes_for_planner=f"Verified candidate {candidate_key}." if candidate_key else "Verified evidence candidate.",
            cost=cost,
            created_round=round_number,
        )
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id:
            self.mutator.mark_candidate_consumed(candidate_id, finding_id=finding.finding_id)
        self.workspace.write_trace_event(
            "evidence_verifier_committed",
            {
                "need_id": sub_goal.sub_goal_id,
                "candidate_id": candidate_id,
                "verdict": _first_verdict(outcome.raw_output),
                "memory_ids": list(outcome.memory_ids),
            },
        )
        self.workspace.write_trace_event(
            "evidence_need_closed",
            {
                "need_id": sub_goal.sub_goal_id,
                "final_status": "done" if status != "infeasible" else "abandoned",
                "finding_id": finding.finding_id,
            },
        )
        return finding

    def _verify_args(self, sub_goal: Any, *, candidate: Mapping[str, object], candidate_key: str) -> dict[str, Any]:
        constraint = sub_goal.constraint
        check: dict[str, Any] = {
            "target_id": f"option_{constraint.option_id}_check" if constraint.option_id else "sub_goal_check",
            "claim": constraint.claim or sub_goal.parent_question,
            "polarity": "presence",
        }
        if constraint.option_id:
            check["option_id"] = constraint.option_id
        args: dict[str, Any] = {
            "candidate_key": candidate_key,
            "focus": [constraint.claim or sub_goal.parent_question],
            "checks": [check],
            "sampling": {"fps": 2, "max_frames": min(128, int(sub_goal.budget.max_frames or 128))},
        }
        answer_options = dict(candidate.get("answer_options") or {})
        if answer_options:
            args["answer_options"] = answer_options
        if not candidate_key:
            segment_id = str(candidate.get("segment_id") or constraint.segment_id or "")
            time_range = _candidate_time_range(candidate) or constraint.time_range
            if segment_id:
                args["segment_id"] = segment_id
            if time_range:
                args["time_range"] = [float(time_range[0]), float(time_range[1])]
        return args


def _has_decisive_memory(workspace: Any, memory_ids: tuple[str, ...]) -> bool:
    if not memory_ids:
        return False
    decisive_kinds = {"visual_support", "answer_support", "synthesized_support", "answer_conflict_resolved", "answer_conflict"}
    memory_by_id = {entry.entry_id: entry for entry in workspace.memory_entries()}
    return any((entry := memory_by_id.get(str(memory_id))) is not None and entry.kind in decisive_kinds for memory_id in memory_ids)


def _first_verdict(raw_output: Mapping[str, Any]) -> str:
    results = raw_output.get("verification_results")
    if isinstance(results, (list, tuple)):
        for item in results:
            if isinstance(item, Mapping):
                verdict = str(item.get("verdict") or "").strip()
                if verdict:
                    return verdict
    return ""


def _frames_read(raw_output: Mapping[str, Any]) -> int:
    for key in ("frames_read", "nframes"):
        value = raw_output.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    frame_refs = raw_output.get("frame_refs")
    if isinstance(frame_refs, (list, tuple)):
        return len(frame_refs)
    return 0


def _candidate_time_range(candidate: Mapping[str, object]) -> tuple[float, float] | None:
    value = candidate.get("time_range")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    if "start_sec" in candidate and "end_sec" in candidate:
        return float(candidate.get("start_sec") or 0.0), float(candidate.get("end_sec") or 0.0)
    return None
