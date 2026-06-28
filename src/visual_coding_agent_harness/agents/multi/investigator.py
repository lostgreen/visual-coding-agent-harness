"""Investigator agent for scoped evidence collection."""

from __future__ import annotations

from typing import Any

from .mutator import WorkspaceMutator
from .tool_runner import MultiAgentToolRunner


class InvestigatorAgent:
    """Minimal Investigator implementation for the first runner slice."""

    def __init__(
        self,
        *,
        backend: Any,
        registry: Any,
        mutator: WorkspaceMutator,
        workspace: Any,
        video_map: Any,
        log_root: Any,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.mutator = mutator
        self.workspace = workspace
        self.video_map = video_map
        self.log_root = log_root
        self.tool_runner = MultiAgentToolRunner(registry=registry, workspace=workspace)

    def step(self, *, round_number: int) -> bool:
        """Claim one open sub-goal and try to turn it into committed evidence."""

        sub_goal = self.mutator.claim_next_open_sub_goal(agent_id="investigator", round_number=round_number)
        if sub_goal is None:
            return False
        memory_ids: tuple[str, ...] = ()
        notes = ""
        try:
            candidate_key = self._select_candidate_key(sub_goal)
            if not candidate_key:
                explore = self.tool_runner.run_tool(
                    "explore",
                    self._explore_args(sub_goal),
                    round_number=round_number,
                    sub_goal_id=sub_goal.sub_goal_id,
                )
                candidate_key = self._select_candidate_key(sub_goal) or self._first_candidate_key(explore.raw_output)
            if candidate_key:
                verify = self.tool_runner.run_tool(
                    "verify_window",
                    self._verify_args(sub_goal, candidate_key=candidate_key),
                    round_number=round_number,
                    sub_goal_id=sub_goal.sub_goal_id,
                )
                memory_ids = verify.memory_ids
                notes = f"Verified candidate {candidate_key}."
            else:
                notes = "No candidate window was available for this sub-goal."
        except Exception as exc:  # noqa: BLE001 - report as finding, do not break driver
            self.workspace.write_trace_event(
                "investigator_tool_error",
                {"round": round_number, "sub_goal_id": sub_goal.sub_goal_id, "error": str(exc)},
            )
            notes = f"Investigation failed: {exc}"
        status = "satisfied" if memory_ids else "empty"
        self.mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status=status,
            memory_ids=memory_ids,
            coverage=(0.0, 0.0),
            notes_for_planner=notes,
            cost={"tool_calls": 1 if memory_ids else 0, "frames_read": 0, "tokens": 0},
            created_round=round_number,
        )
        return True

    def _explore_args(self, sub_goal: Any) -> dict[str, Any]:
        constraint = sub_goal.constraint
        scope: dict[str, Any] = {}
        if constraint.segment_id:
            scope["segment_ids"] = [constraint.segment_id]
        if constraint.time_range:
            scope["time_range"] = list(constraint.time_range)
        target: dict[str, Any] = {
            "target_id": f"option_{constraint.option_id}_check" if constraint.option_id else "sub_goal_check",
            "claim": constraint.claim,
            "verification_goal": "Find a local window that can verify this sub-goal.",
        }
        if constraint.option_id:
            target["option_id"] = constraint.option_id
        return {
            "query": constraint.claim or sub_goal.parent_question,
            "targets": [target],
            "scope": scope,
            "modalities": list(constraint.modality_hint or ("index", "asr", "ocr", "visual")),
            "top_k": 3,
            "original_question": sub_goal.parent_question,
        }

    def _verify_args(self, sub_goal: Any, *, candidate_key: str) -> dict[str, Any]:
        constraint = sub_goal.constraint
        check: dict[str, Any] = {
            "target_id": f"option_{constraint.option_id}_check" if constraint.option_id else "sub_goal_check",
            "claim": constraint.claim or sub_goal.parent_question,
            "polarity": "presence",
        }
        if constraint.option_id:
            check["option_id"] = constraint.option_id
        return {
            "candidate_key": candidate_key,
            "focus": [constraint.claim or sub_goal.parent_question],
            "checks": [check],
            "sampling": {"fps": 2, "max_frames": min(128, int(sub_goal.budget.max_frames or 128))},
        }

    def _select_candidate_key(self, sub_goal: Any) -> str:
        best_key = ""
        best_score = -1.0
        for observation in self.workspace.read_observations():
            raw_output = observation.raw_output if isinstance(observation.raw_output, dict) else {}
            for candidate in _mapping_items(raw_output.get("candidate_windows")):
                if not self._candidate_matches_sub_goal(candidate, sub_goal):
                    continue
                score = float(candidate.get("score", 0.0) or 0.0)
                key = str(candidate.get("candidate_key") or "").strip()
                if key and score >= best_score:
                    best_key = key
                    best_score = score
        return best_key

    def _candidate_matches_sub_goal(self, candidate: dict[str, Any], sub_goal: Any) -> bool:
        constraint = sub_goal.constraint
        if constraint.segment_id and str(candidate.get("segment_id") or "") != constraint.segment_id:
            return False
        if constraint.time_range:
            start, end = constraint.time_range
            cand_start = float(candidate.get("start_sec", candidate.get("time_range", [0.0, 0.0])[0]) or 0.0)
            cand_end = float(candidate.get("end_sec", candidate.get("time_range", [0.0, 0.0])[-1]) or 0.0)
            if cand_end < start or cand_start > end:
                return False
        return True

    @staticmethod
    def _first_candidate_key(raw_output: Any) -> str:
        for candidate in _mapping_items(raw_output.get("candidate_windows") if isinstance(raw_output, dict) else None):
            key = str(candidate.get("candidate_key") or "").strip()
            if key:
                return key
        return ""


def _mapping_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
