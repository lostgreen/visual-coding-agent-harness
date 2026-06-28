"""Investigator agent for scoped evidence collection."""

from __future__ import annotations

from typing import Any

from .evidence_scout import EvidenceScout
from .evidence_verifier import EvidenceVerifier
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
        self.scout = EvidenceScout(registry=registry, mutator=mutator, workspace=workspace)
        self.verifier = EvidenceVerifier(registry=registry, mutator=mutator, workspace=workspace)

    def step(self, *, round_number: int) -> bool:
        """Claim one open sub-goal and try to turn it into committed evidence."""

        sub_goal = self.mutator.claim_next_open_sub_goal(agent_id="investigator", round_number=round_number)
        if sub_goal is None:
            return False
        try:
            candidate = self.scout.propose_candidate(sub_goal, round_number=round_number)
            if candidate:
                self.verifier.verify(sub_goal=sub_goal, candidate=candidate, round_number=round_number, explore_calls=1)
                return True
            notes = "No candidate window was available for this sub-goal."
        except Exception as exc:  # noqa: BLE001 - report as finding, do not break driver
            self.workspace.write_trace_event(
                "investigator_tool_error",
                {"round": round_number, "sub_goal_id": sub_goal.sub_goal_id, "error": str(exc)},
            )
            notes = f"Investigation failed: {exc}"
        self.mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status="empty",
            memory_ids=(),
            coverage=(0.0, 0.0),
            notes_for_planner=notes,
            cost={"explore_calls": 1, "verify_calls": 0, "tool_calls": 1, "frames_read": 0, "tokens": 0},
            created_round=round_number,
        )
        return True

    def _has_positive_memory(self, memory_ids: tuple[str, ...]) -> bool:
        if not memory_ids:
            return False
        positive_kinds = {"visual_support", "answer_support", "synthesized_support", "answer_conflict_resolved"}
        memory_by_id = {entry.entry_id: entry for entry in self.workspace.memory_entries()}
        for memory_id in memory_ids:
            entry = memory_by_id.get(str(memory_id))
            if entry is not None and entry.kind in positive_kinds:
                return True
        return False

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
            if not self._observation_matches_sub_goal(raw_output, sub_goal):
                continue
            for candidate in _mapping_items(raw_output.get("candidate_windows")):
                if not self._candidate_matches_sub_goal(candidate, sub_goal):
                    continue
                score = float(candidate.get("score", 0.0) or 0.0)
                key = str(candidate.get("candidate_key") or "").strip()
                if key and score >= best_score:
                    best_key = key
                    best_score = score
        return best_key

    def _select_shared_candidate_key(self, sub_goal: Any) -> str:
        best_key = ""
        best_score = -1.0
        for observation in self.workspace.read_observations():
            raw_output = observation.raw_output if isinstance(observation.raw_output, dict) else {}
            for candidate in _mapping_items(raw_output.get("candidate_windows")):
                if not self._candidate_matches_sub_goal(candidate, sub_goal, enforce_option=False):
                    continue
                score = float(candidate.get("score", 0.0) or 0.0)
                key = str(candidate.get("candidate_key") or "").strip()
                if key and score >= best_score:
                    best_key = key
                    best_score = score
        return best_key

    def _observation_matches_sub_goal(self, raw_output: dict[str, Any], sub_goal: Any) -> bool:
        option_id = str(sub_goal.constraint.option_id or "").strip().upper()[:1]
        if not option_id:
            return True
        tagged_option = str(raw_output.get("multi_agent_option_id") or "").strip().upper()[:1]
        if tagged_option:
            return tagged_option == option_id
        return False

    def _candidate_matches_sub_goal(
        self,
        candidate: dict[str, Any],
        sub_goal: Any,
        *,
        enforce_option: bool = True,
    ) -> bool:
        constraint = sub_goal.constraint
        option_id = str(constraint.option_id or "").strip().upper()[:1]
        if enforce_option and option_id:
            candidate_option = str(candidate.get("option_id") or "").strip().upper()[:1]
            if candidate_option and candidate_option != option_id:
                return False
            target_id = str(candidate.get("target_id") or "").strip().upper()
            if target_id and f"OPTION_{option_id}" not in target_id and f"OPTION-{option_id}" not in target_id:
                return False
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
