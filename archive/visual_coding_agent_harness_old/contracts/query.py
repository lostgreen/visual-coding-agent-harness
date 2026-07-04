"""Planning contracts for the multi_v3 long-video loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from visual_coding_agent_harness.contracts.playbook import Playbook


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


@dataclass(frozen=True, init=False)
class QueryScope:
    chapter_ids: Sequence[str]
    time_range: tuple[float, float] | None = None
    entity_hints: Sequence[str] = field(default_factory=tuple)
    modality_hint: Sequence[str] = field(default_factory=tuple)

    def __init__(
        self,
        chapter_ids: Sequence[str] | None = None,
        *,
        scene_ids: Sequence[str] | None = None,
        time_range: tuple[float, float] | None = None,
        entity_hints: Sequence[str] = (),
        modality_hint: Sequence[str] = (),
    ) -> None:
        object.__setattr__(self, "chapter_ids", _text_tuple(chapter_ids if chapter_ids is not None else scene_ids))
        object.__setattr__(self, "time_range", time_range)
        object.__setattr__(self, "entity_hints", entity_hints)
        object.__setattr__(self, "modality_hint", modality_hint)
        self.__post_init__()

    def __post_init__(self) -> None:
        chapter_ids = _text_tuple(self.chapter_ids)
        if not chapter_ids:
            raise ValueError("chapter_ids/scene_ids must contain at least one chapter id")
        object.__setattr__(self, "chapter_ids", chapter_ids)
        object.__setattr__(self, "entity_hints", _text_tuple(self.entity_hints))
        object.__setattr__(self, "modality_hint", _text_tuple(self.modality_hint))
        if self.time_range is not None:
            start, end = self.time_range
            if float(end) < float(start):
                raise ValueError("time_range end must be greater than or equal to start")
            object.__setattr__(self, "time_range", (float(start), float(end)))

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(self.chapter_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "chapter_ids": list(self.chapter_ids),
            "time_range": list(self.time_range) if self.time_range is not None else None,
            "entity_hints": list(self.entity_hints),
            "modality_hint": list(self.modality_hint),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryScope":
        return cls(
            chapter_ids=_text_tuple(value.get("chapter_ids") or value.get("scene_ids") or ()),
            time_range=_float_pair(value.get("time_range")),
            entity_hints=_text_tuple(value.get("entity_hints") or ()),
            modality_hint=_text_tuple(value.get("modality_hint") or ()),
        )


@dataclass(frozen=True, init=False)
class QueryBudget:
    max_beats_to_verify: int = 3
    max_frames: int = 128

    def __init__(
        self,
        max_beats_to_verify: int | None = None,
        max_frames: int = 128,
        *,
        max_shots_to_verify: int | None = None,
    ) -> None:
        value = max_beats_to_verify if max_beats_to_verify is not None else max_shots_to_verify
        object.__setattr__(self, "max_beats_to_verify", 3 if value is None else int(value))
        object.__setattr__(self, "max_frames", int(max_frames))
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.max_beats_to_verify < 0:
            raise ValueError("max_beats_to_verify must be non-negative")
        if self.max_frames < 0:
            raise ValueError("max_frames must be non-negative")

    @property
    def max_shots_to_verify(self) -> int:
        return int(self.max_beats_to_verify)

    def to_dict(self) -> dict[str, object]:
        return {"max_beats_to_verify": int(self.max_beats_to_verify), "max_frames": int(self.max_frames)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "QueryBudget":
        value = value or {}
        return cls(
            max_beats_to_verify=int(value.get("max_beats_to_verify", value.get("max_shots_to_verify", 3)) or 0),
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
    playbook: Playbook = Playbook.IDENTIFY_VISUAL
    text_queries: Sequence[str] = field(default_factory=tuple)
    visual_queries: Sequence[str] = field(default_factory=tuple)
    scope_b: QueryScope | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "playbook", Playbook.parse(self.playbook))
        if not self.query_id:
            raise ValueError("query_id is required")
        if not self.goal_id:
            raise ValueError("goal_id is required")
        if not self.natural_query:
            raise ValueError("natural_query is required")
        object.__setattr__(self, "text_queries", _text_tuple(self.text_queries) or (self.natural_query,))
        object.__setattr__(self, "visual_queries", _text_tuple(self.visual_queries) or (self.natural_query,))

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "goal_id": self.goal_id,
            "playbook": self.playbook.value,
            "natural_query": self.natural_query,
            "text_queries": list(self.text_queries),
            "visual_queries": list(self.visual_queries),
            "scope": self.scope.to_dict(),
            "scope_b": self.scope_b.to_dict() if self.scope_b is not None else None,
            "expected_evidence": self.expected_evidence,
            "budget": self.budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScopedQuery":
        scope = value.get("scope") if isinstance(value.get("scope"), Mapping) else {}
        budget = value.get("budget") if isinstance(value.get("budget"), Mapping) else {}
        scope_b = value.get("scope_b") if isinstance(value.get("scope_b"), Mapping) else None
        return cls(
            query_id=str(value.get("query_id") or ""),
            goal_id=str(value.get("goal_id") or ""),
            natural_query=str(value.get("natural_query") or ""),
            scope=QueryScope.from_dict(scope),
            expected_evidence=str(value.get("expected_evidence") or ""),
            budget=QueryBudget.from_dict(budget),
            playbook=Playbook.parse(value.get("playbook")),
            text_queries=_text_tuple(value.get("text_queries") or ()),
            visual_queries=_text_tuple(value.get("visual_queries") or ()),
            scope_b=QueryScope.from_dict(scope_b) if scope_b is not None else None,
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
