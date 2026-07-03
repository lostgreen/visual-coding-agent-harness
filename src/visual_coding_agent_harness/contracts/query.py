"""Planning contracts for the multi_v3 long-video loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence


GoalKind = Literal["locate", "count", "compare", "order", "identify", "temporal"]


@dataclass(frozen=True)
class VerifiableGoal:
    goal_id: str
    text: str
    linked_options: Sequence[str] = field(default_factory=tuple)
    kind: GoalKind = "identify"
    priority: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "linked_options", _text_tuple(self.linked_options))
        if not self.goal_id:
            raise ValueError("goal_id is required")
        if not self.text:
            raise ValueError("text is required")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["linked_options"] = list(self.linked_options)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifiableGoal":
        return cls(
            goal_id=str(value.get("goal_id") or ""),
            text=str(value.get("text") or ""),
            linked_options=_text_tuple(value.get("linked_options") or ()),
            kind=str(value.get("kind") or "identify"),  # type: ignore[arg-type]
            priority=float(value.get("priority", 0.5) or 0.0),
        )


@dataclass(frozen=True)
class QueryScope:
    scene_ids: Sequence[str]
    time_range: tuple[float, float] | None = None
    entity_hints: Sequence[str] = field(default_factory=tuple)
    modality_hint: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        scene_ids = _text_tuple(self.scene_ids)
        if not scene_ids:
            raise ValueError("scene_ids must contain at least one scene id")
        object.__setattr__(self, "scene_ids", scene_ids)
        object.__setattr__(self, "entity_hints", _text_tuple(self.entity_hints))
        object.__setattr__(self, "modality_hint", _text_tuple(self.modality_hint))
        if self.time_range is not None:
            start, end = self.time_range
            if float(end) < float(start):
                raise ValueError("time_range end must be greater than or equal to start")
            object.__setattr__(self, "time_range", (float(start), float(end)))

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_ids": list(self.scene_ids),
            "time_range": list(self.time_range) if self.time_range is not None else None,
            "entity_hints": list(self.entity_hints),
            "modality_hint": list(self.modality_hint),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryScope":
        return cls(
            scene_ids=_text_tuple(value.get("scene_ids") or ()),
            time_range=_float_pair(value.get("time_range")),
            entity_hints=_text_tuple(value.get("entity_hints") or ()),
            modality_hint=_text_tuple(value.get("modality_hint") or ()),
        )


@dataclass(frozen=True)
class QueryBudget:
    max_shots_to_verify: int = 3
    max_frames: int = 128

    def __post_init__(self) -> None:
        if self.max_shots_to_verify < 0:
            raise ValueError("max_shots_to_verify must be non-negative")
        if self.max_frames < 0:
            raise ValueError("max_frames must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {"max_shots_to_verify": int(self.max_shots_to_verify), "max_frames": int(self.max_frames)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "QueryBudget":
        value = value or {}
        return cls(
            max_shots_to_verify=int(value.get("max_shots_to_verify", 3) or 0),
            max_frames=int(value.get("max_frames", 128) or 0),
        )


@dataclass(frozen=True)
class ScopedQuery:
    query_id: str
    goal_id: str
    natural_query: str
    scope: QueryScope
    expected_evidence: str
    budget: QueryBudget = field(default_factory=QueryBudget)

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id is required")
        if not self.goal_id:
            raise ValueError("goal_id is required")
        if not self.natural_query:
            raise ValueError("natural_query is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "goal_id": self.goal_id,
            "natural_query": self.natural_query,
            "scope": self.scope.to_dict(),
            "expected_evidence": self.expected_evidence,
            "budget": self.budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScopedQuery":
        scope = value.get("scope") if isinstance(value.get("scope"), Mapping) else {}
        budget = value.get("budget") if isinstance(value.get("budget"), Mapping) else {}
        return cls(
            query_id=str(value.get("query_id") or ""),
            goal_id=str(value.get("goal_id") or ""),
            natural_query=str(value.get("natural_query") or ""),
            scope=QueryScope.from_dict(scope),
            expected_evidence=str(value.get("expected_evidence") or ""),
            budget=QueryBudget.from_dict(budget),
        )


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        values = () if value is None else (value,)
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = (value,)
    return tuple(text for item in values if (text := str(item).strip()))


def _float_pair(value: object) -> tuple[float, float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(items) < 2:
        return None
    return float(items[0]), float(items[1])
