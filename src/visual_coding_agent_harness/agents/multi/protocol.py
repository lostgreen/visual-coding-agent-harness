"""Small protocol objects for the multi-agent video QA path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping


SubGoalIntent = Literal["locate", "verify", "disprove", "cover", "disambiguate"]
SubGoalStatus = Literal["open", "in_progress", "done", "abandoned"]
FindingStatus = Literal["satisfied", "partial", "empty", "infeasible"]
MessageKind = Literal["question", "clarify", "push_back"]


@dataclass(frozen=True)
class SubGoalConstraint:
    time_range: tuple[float, float] | None = None
    segment_id: str | None = None
    option_id: str | None = None
    claim: str = ""
    modality_hint: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubGoalBudget:
    max_explores: int = 2
    max_verifies: int = 3
    max_frames: int = 256


@dataclass(frozen=True)
class SubGoalSuccessCriteria:
    needs_visual_support: bool = True
    needs_coverage: bool = False
    coverage_threshold: float = 0.6
    needs_option_relation: bool = True


@dataclass(frozen=True)
class SubGoal:
    sub_goal_id: str
    intent: SubGoalIntent
    constraint: SubGoalConstraint
    budget: SubGoalBudget
    success_criteria: SubGoalSuccessCriteria
    parent_question: str
    created_by: str
    created_round: int
    status: SubGoalStatus = "open"
    rationale: str = ""
    assigned_to: str = ""
    updated_round: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["constraint"]["time_range"] = list(self.constraint.time_range) if self.constraint.time_range else None
        payload["constraint"]["modality_hint"] = list(self.constraint.modality_hint)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SubGoal":
        constraint_payload = _mapping(payload.get("constraint"))
        budget_payload = _mapping(payload.get("budget"))
        criteria_payload = _mapping(payload.get("success_criteria"))
        time_range = constraint_payload.get("time_range")
        return cls(
            sub_goal_id=str(payload.get("sub_goal_id", "")),
            intent=str(payload.get("intent", "verify")),  # type: ignore[arg-type]
            constraint=SubGoalConstraint(
                time_range=_time_range_tuple(time_range),
                segment_id=_optional_text(constraint_payload.get("segment_id")),
                option_id=_optional_text(constraint_payload.get("option_id")),
                claim=str(constraint_payload.get("claim", "")),
                modality_hint=tuple(str(item) for item in _sequence(constraint_payload.get("modality_hint"))),
            ),
            budget=SubGoalBudget(
                max_explores=int(budget_payload.get("max_explores", 2) or 2),
                max_verifies=int(budget_payload.get("max_verifies", 3) or 3),
                max_frames=int(budget_payload.get("max_frames", 256) or 256),
            ),
            success_criteria=SubGoalSuccessCriteria(
                needs_visual_support=bool(criteria_payload.get("needs_visual_support", True)),
                needs_coverage=bool(criteria_payload.get("needs_coverage", False)),
                coverage_threshold=float(criteria_payload.get("coverage_threshold", 0.6) or 0.6),
                needs_option_relation=bool(criteria_payload.get("needs_option_relation", True)),
            ),
            parent_question=str(payload.get("parent_question", "")),
            created_by=str(payload.get("created_by", "")),
            created_round=int(payload.get("created_round", 0) or 0),
            status=str(payload.get("status", "open")),  # type: ignore[arg-type]
            rationale=str(payload.get("rationale", "")),
            assigned_to=str(payload.get("assigned_to", "")),
            updated_round=int(payload.get("updated_round", 0) or 0),
        )


@dataclass(frozen=True)
class Finding:
    finding_id: str
    sub_goal_id: str
    status: FindingStatus
    memory_ids: tuple[str, ...]
    coverage: tuple[float, float]
    notes_for_planner: str
    cost: Mapping[str, int] = field(default_factory=dict)
    created_round: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["memory_ids"] = list(self.memory_ids)
        payload["coverage"] = list(self.coverage)
        payload["cost"] = dict(self.cost)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Finding":
        coverage = _sequence(payload.get("coverage"))
        return cls(
            finding_id=str(payload.get("finding_id", "")),
            sub_goal_id=str(payload.get("sub_goal_id", "")),
            status=str(payload.get("status", "empty")),  # type: ignore[arg-type]
            memory_ids=tuple(str(item) for item in _sequence(payload.get("memory_ids"))),
            coverage=(
                float(coverage[0]) if len(coverage) >= 1 else 0.0,
                float(coverage[1]) if len(coverage) >= 2 else 0.0,
            ),
            notes_for_planner=str(payload.get("notes_for_planner", "")),
            cost={str(key): int(value) for key, value in _mapping(payload.get("cost")).items()},
            created_round=int(payload.get("created_round", 0) or 0),
        )


@dataclass(frozen=True)
class InboxMessage:
    message_id: str
    from_agent: str
    to_agent: str
    kind: MessageKind
    payload: Mapping[str, object]
    created_round: int


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _time_range_tuple(value: object) -> tuple[float, float] | None:
    items = _sequence(value)
    if len(items) < 2:
        return None
    return float(items[0]), float(items[1])
