from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set


FollowupRoute = Literal["temporal_order", "needle_local", "gist_global"]


@dataclass
class FollowupTarget:
    target_id: str
    query: str
    event_label: Optional[str]
    route: FollowupRoute
    reason: str
    priority: int
    attempt_count: int
    parent_missing_evidence_id: str
    mutex_group_id: str = ""


@dataclass
class FollowupBudget:
    global_max_followups: int = 4
    per_gap_max_attempts: int = 3
    saturation_threshold: float = 0.1
    saturation_window: int = 2


def _refine_grounding_query(raw: Dict) -> str:
    question = str(raw.get("question") or "").strip()
    reason = str(raw.get("reason") or "").strip()
    if question and reason:
        return f"{question} {reason}".strip()
    return question or reason


def normalize_missing_evidence(raw: Dict, route: FollowupRoute, run_id: str, seq: int) -> FollowupTarget:
    query = str(raw.get("query") or raw.get("missing_description") or "").strip()
    if not query:
        query = _refine_grounding_query(raw)

    return FollowupTarget(
        target_id=f"fu_{run_id}_{seq:04d}",
        query=query,
        event_label=raw.get("event_label"),
        route=route,
        reason=raw.get("reason", "unspecified"),
        priority=raw.get("priority", 1),
        attempt_count=0,
        parent_missing_evidence_id=raw["id"],
        mutex_group_id=str(raw.get("mutex_group_id", "")),
    )


class FollowupScheduler:
    def __init__(self, budget: FollowupBudget):
        self.budget = budget
        self.queue: List[FollowupTarget] = []
        self.completed: List[FollowupTarget] = []
        self.global_attempts: int = 0
        self._frame_set_history: List[Set[str]] = []

    def enqueue(self, targets: List[FollowupTarget]) -> None:
        seen = {(target.route, target.query, target.event_label) for target in self.queue}
        seen.update((target.route, target.query, target.event_label) for target in self.completed)

        for target in targets:
            key = (target.route, target.query, target.event_label)
            if key in seen:
                continue
            self.queue.append(target)
            seen.add(key)

    def next(self) -> Optional[FollowupTarget]:
        if self.global_attempts >= self.budget.global_max_followups:
            return None
        if self._saturated():
            return None

        self.queue.sort(key=lambda target: (target.priority, target.attempt_count))
        while self.queue:
            target = self.queue[0]
            if target.attempt_count >= self.budget.per_gap_max_attempts:
                self.queue.pop(0)
                self.completed.append(target)
                continue
            return target
        return None

    def record_attempt(self, target: FollowupTarget, new_frame_sets: Set[str]) -> None:
        target.attempt_count += 1
        self.global_attempts += 1
        self._frame_set_history.append(new_frame_sets)

    def _saturated(self) -> bool:
        if len(self._frame_set_history) < self.budget.saturation_window:
            return False

        recent = self._frame_set_history[-self.budget.saturation_window :]
        all_recent = set().union(*recent)
        if not all_recent:
            return True

        prior = set().union(*self._frame_set_history[: -self.budget.saturation_window])
        new = all_recent - prior
        ratio = len(new) / max(len(all_recent), 1)
        return ratio < self.budget.saturation_threshold
