"""Rule-based EvidenceScout for candidate window discovery."""

from __future__ import annotations

from typing import Any, Mapping

from ...evidence import EvidenceLedger
from .mutator import WorkspaceMutator
from .tool_runner import MultiAgentToolRunner


class EvidenceScout:
    """Translate an evidence need into candidate windows without committing memory."""

    def __init__(self, *, registry: Any, mutator: WorkspaceMutator, workspace: Any) -> None:
        self.registry = registry
        self.mutator = mutator
        self.workspace = workspace
        self.tool_runner = MultiAgentToolRunner(registry=registry, workspace=workspace)

    def propose_candidate(self, sub_goal: Any, *, round_number: int) -> dict[str, object] | None:
        """Explore candidate windows for one SubGoal and record the best fresh candidate."""

        if int(getattr(sub_goal.budget, "max_explores", 0) or 0) <= 0:
            existing = self._best_existing_candidate(sub_goal, enforce_option=True)
            if existing:
                return self._record_candidate(
                    sub_goal,
                    existing,
                    round_number=round_number,
                    source="scout_existing_candidate",
                )

        explore = self.tool_runner.run_tool(
            "explore",
            self._explore_args(sub_goal),
            round_number=round_number,
            sub_goal_id=sub_goal.sub_goal_id,
        )
        raw_output = dict(explore.raw_output)
        candidates = self._fresh_candidates(sub_goal, raw_output.get("candidate_windows"))
        if not candidates:
            shared = self._best_existing_candidate(sub_goal, enforce_option=False, allow_consumed=False)
            if shared:
                return self._record_candidate(sub_goal, shared, round_number=round_number, source="scout_shared_candidate")
        if not candidates:
            sweep = self._sweep_candidate(sub_goal)
            if sweep:
                return self._record_candidate(sub_goal, sweep, round_number=round_number, source="scout_segment_sweep")
        if not candidates:
            self.workspace.write_trace_event(
                "evidence_scout_candidates_proposed",
                {
                    "need_id": sub_goal.sub_goal_id,
                    "candidate_ids": [],
                    "source": "scout_explore_hit",
                },
            )
            return None
        return self._record_candidate(sub_goal, candidates[0], round_number=round_number, source="scout_explore_hit")

    def _record_candidate(
        self,
        sub_goal: Any,
        candidate: Mapping[str, object],
        *,
        round_number: int,
        source: str,
    ) -> dict[str, object] | None:
        row = dict(candidate)
        row["source"] = source
        recorded = self.mutator.record_candidates(
            need_id=sub_goal.sub_goal_id,
            option_id=str(sub_goal.constraint.option_id or ""),
            candidates=[row],
            round_number=round_number,
        )
        candidate_ids = [str(item.get("candidate_id") or "") for item in recorded]
        self.workspace.write_trace_event(
            "evidence_scout_candidates_proposed",
            {
                "need_id": sub_goal.sub_goal_id,
                "candidate_ids": candidate_ids,
                "source": str(recorded[0].get("source") or "scout_explore_hit") if recorded else "scout_explore_hit",
            },
        )
        return recorded[0] if recorded else None

    def _fresh_candidates(self, sub_goal: Any, candidate_windows: Any) -> list[dict[str, object]]:
        verified = EvidenceLedger(workspace=self.workspace, mutator=self.mutator).verified_windows_for_option(
            str(sub_goal.constraint.option_id or "")
        )
        fresh: list[dict[str, object]] = []
        for candidate in _mapping_items(candidate_windows):
            row = dict(candidate)
            if sub_goal.constraint.option_id:
                row.setdefault("option_id", str(sub_goal.constraint.option_id))
                row.setdefault("target_id", f"option_{sub_goal.constraint.option_id}_check")
            time_range = _candidate_time_range(row)
            segment_id = str(row.get("segment_id") or "")
            if segment_id and time_range and (segment_id, time_range[0], time_range[1]) in verified:
                self.workspace.write_trace_event(
                    "evidence_scout_window_excluded",
                    {
                        "need_id": sub_goal.sub_goal_id,
                        "candidate_key": str(row.get("candidate_key") or ""),
                        "reason": "already_verified_for_option",
                    },
                )
                continue
            row.setdefault("source", "scout_explore_hit")
            fresh.append(row)
        fresh.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return fresh

    def _best_existing_candidate(
        self,
        sub_goal: Any,
        *,
        enforce_option: bool,
        allow_consumed: bool = True,
    ) -> dict[str, object] | None:
        candidates: list[dict[str, object]] = []
        for observation in self.workspace.read_observations():
            raw_output = observation.raw_output if isinstance(observation.raw_output, dict) else {}
            if enforce_option and not self._observation_matches_sub_goal(raw_output, sub_goal):
                continue
            for candidate in _mapping_items(raw_output.get("candidate_windows")):
                if not allow_consumed and self._candidate_was_consumed(candidate):
                    continue
                if self._candidate_matches_sub_goal(candidate, sub_goal, enforce_option=enforce_option):
                    candidates.append(candidate)
        fresh = self._fresh_candidates(sub_goal, candidates)
        return fresh[0] if fresh else None

    def _candidate_was_consumed(self, candidate: Mapping[str, object]) -> bool:
        key = str(candidate.get("candidate_key") or "").strip()
        segment_id = str(candidate.get("segment_id") or "").strip()
        time_range = _candidate_time_range(candidate)
        for recorded in self.mutator.evidence_candidates():
            finding_id = str(recorded.get("consumed_by_finding_id") or "").strip()
            if not finding_id:
                continue
            if self._consumed_candidate_is_reusable(finding_id):
                continue
            recorded_key = str(recorded.get("candidate_key") or "").strip()
            if key and recorded_key == key:
                return True
            recorded_range = _candidate_time_range(recorded)
            if (
                segment_id
                and segment_id == str(recorded.get("segment_id") or "").strip()
                and time_range is not None
                and recorded_range is not None
                and time_range == recorded_range
            ):
                return True
        return False

    def _consumed_candidate_is_reusable(self, finding_id: str) -> bool:
        finding = next((item for item in self.mutator.findings() if item.finding_id == finding_id), None)
        if finding is None or finding.status != "empty":
            return False
        if not finding.memory_ids:
            return False
        memory_by_id = {entry.entry_id: entry for entry in self.workspace.memory_entries()}
        return all(
            (entry := memory_by_id.get(str(memory_id))) is not None and entry.kind == "local_negative"
            for memory_id in finding.memory_ids
        )

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
        candidate: dict[str, object],
        sub_goal: Any,
        *,
        enforce_option: bool,
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
            time_range = _candidate_time_range(candidate)
            if time_range is None:
                return False
            start, end = constraint.time_range
            if time_range[1] < start or time_range[0] > end:
                return False
        return True

    def _sweep_candidate(self, sub_goal: Any) -> dict[str, object] | None:
        constraint = sub_goal.constraint
        scope = self._sweep_scope(sub_goal)
        if scope is None:
            return None
        segment_id, scope_range = scope
        option_id = str(constraint.option_id or "").strip().upper()[:1]
        verified = EvidenceLedger(workspace=self.workspace, mutator=self.mutator).verified_windows_for_option(option_id)
        time_range = _first_unverified_window(
            segment_id=segment_id,
            scope=scope_range,
            verified=verified,
        )
        if time_range is None:
            return None
        return {
            "candidate_key": "",
            "segment_id": segment_id,
            "time_range": [time_range[0], time_range[1]],
            "score": 0.0,
            "modalities": list(constraint.modality_hint or ("visual",)),
            "source": "scout_segment_sweep",
        }

    def _sweep_scope(self, sub_goal: Any) -> tuple[str, tuple[float, float]] | None:
        constraint = sub_goal.constraint
        if constraint.segment_id and constraint.time_range:
            return str(constraint.segment_id), (float(constraint.time_range[0]), float(constraint.time_range[1]))
        candidates = self._historical_candidate_windows(sub_goal)
        if not candidates:
            return None
        best = candidates[0]
        segment_id = str(best.get("segment_id") or "").strip()
        time_range = _candidate_time_range(best)
        if not segment_id or time_range is None:
            return None
        width = max(20.0, time_range[1] - time_range[0])
        return segment_id, (time_range[0], time_range[1] + width)

    def _historical_candidate_windows(self, sub_goal: Any) -> list[dict[str, object]]:
        option_id = str(sub_goal.constraint.option_id or "").strip().upper()[:1]
        candidates: list[dict[str, object]] = []
        for recorded in self.mutator.evidence_candidates():
            row = dict(recorded)
            if self._candidate_matches_sub_goal(row, sub_goal, enforce_option=False) and not self._has_other_option_support(
                row, option_id=option_id
            ):
                candidates.append(dict(recorded))
        for observation in self.workspace.read_observations():
            raw_output = observation.raw_output if isinstance(observation.raw_output, dict) else {}
            for candidate in _mapping_items(raw_output.get("candidate_windows")):
                if self._candidate_matches_sub_goal(candidate, sub_goal, enforce_option=False) and not self._has_other_option_support(
                    candidate, option_id=option_id
                ):
                    candidates.append(candidate)
        candidates = [candidate for candidate in candidates if str(candidate.get("segment_id") or "").strip()]
        candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return candidates

    def _has_other_option_support(self, candidate: Mapping[str, object], *, option_id: str) -> bool:
        segment_id = str(candidate.get("segment_id") or "").strip()
        time_range = _candidate_time_range(candidate)
        if not segment_id or time_range is None:
            return False
        ledger = EvidenceLedger(workspace=self.workspace, mutator=self.mutator)
        for item in ledger.items():
            item_option = str(item.option_id or "").strip().upper()[:1]
            if not item_option or item_option == option_id or item.polarity != "supports":
                continue
            if item.segment_id == segment_id and item.time_range == time_range:
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
            "verification_goal": "Find a local window that can verify this evidence need.",
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


def _mapping_items(value: Any) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _candidate_time_range(candidate: Mapping[str, object]) -> tuple[float, float] | None:
    value = candidate.get("time_range")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    if "start_sec" in candidate and "end_sec" in candidate:
        return float(candidate.get("start_sec") or 0.0), float(candidate.get("end_sec") or 0.0)
    return None


def _first_unverified_window(
    *,
    segment_id: str,
    scope: tuple[float, float],
    verified: set[tuple[str, float, float]],
    window_sec: float = 20.0,
) -> tuple[float, float] | None:
    scope_start, scope_end = float(scope[0]), float(scope[1])
    if scope_end <= scope_start:
        return None
    intervals = sorted(
        (max(scope_start, float(start)), min(scope_end, float(end)))
        for verified_segment_id, start, end in verified
        if verified_segment_id == segment_id and float(end) > scope_start and float(start) < scope_end
    )
    cursor = scope_start
    for start, end in intervals:
        if end <= cursor:
            continue
        if start > cursor:
            return (cursor, min(start, cursor + window_sec))
        cursor = max(cursor, end)
    if cursor < scope_end:
        return (cursor, min(scope_end, cursor + window_sec))
    return None
