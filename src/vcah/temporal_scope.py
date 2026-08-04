from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence


TemporalRelation = Literal["before", "after", "within", "between"]
TemporalSelection = Literal["first", "next", "last", "all", "unspecified"]

TEMPORAL_RELATIONS = {"before", "after", "within", "between"}
TEMPORAL_SELECTIONS = {"first", "next", "last", "all", "unspecified"}


@dataclass(frozen=True)
class TemporalScope:
    scope_id: str
    relation: TemporalRelation
    selection: TemporalSelection
    anchor_requirement_id: str
    target_requirement_id: str

    def __post_init__(self) -> None:
        scope_id = str(self.scope_id or "").strip()
        anchor_id = str(self.anchor_requirement_id or "").strip()
        target_id = str(self.target_requirement_id or "").strip()
        relation = str(self.relation or "").strip().casefold()
        selection = str(self.selection or "unspecified").strip().casefold()
        if not scope_id:
            raise ValueError("temporal_scope_requires_scope_id")
        if relation not in TEMPORAL_RELATIONS:
            raise ValueError(f"invalid_temporal_relation:{relation or 'missing'}")
        if selection not in TEMPORAL_SELECTIONS:
            raise ValueError(f"invalid_temporal_selection:{selection or 'missing'}")
        if not anchor_id or not target_id:
            raise ValueError(f"temporal_scope_requires_requirements:{scope_id}")
        if anchor_id == target_id:
            raise ValueError(f"temporal_scope_requirements_must_differ:{scope_id}")
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "anchor_requirement_id", anchor_id)
        object.__setattr__(self, "target_requirement_id", target_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TemporalScope":
        return cls(
            scope_id=str(value.get("scope_id", value.get("id", "")) or ""),
            relation=str(value.get("relation", "") or ""),  # type: ignore[arg-type]
            selection=str(value.get("selection", "unspecified") or "unspecified"),  # type: ignore[arg-type]
            anchor_requirement_id=str(value.get("anchor_requirement_id", "") or ""),
            target_requirement_id=str(value.get("target_requirement_id", "") or ""),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_temporal_scope(
    scope: TemporalScope,
    *,
    anchor_intervals: Sequence[Sequence[float]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Mechanically filter and order candidates without judging event identity."""

    anchors = tuple(
        normalized
        for value in anchor_intervals
        if (normalized := _interval(value)) is not None
    )
    normalized_candidates = tuple(
        {
            **dict(candidate),
            "time_range": list(interval),
        }
        for candidate in candidates
        if (interval := _interval(candidate.get("time_range", ()))) is not None
    )
    if not anchors:
        return _result(scope, "unresolved_anchor", (), ())

    eligible = tuple(
        candidate
        for candidate in normalized_candidates
        if _matches_relation(scope.relation, tuple(candidate["time_range"]), anchors)
    )
    ordered = tuple(
        sorted(
            eligible,
            key=lambda candidate: (
                float(candidate["time_range"][0]),
                float(candidate["time_range"][1]),
                str(candidate.get("occurrence_id", "")),
            ),
        )
    )
    if not ordered:
        return _result(scope, "unresolved_candidates", (), ())

    if scope.selection in {"first", "next"}:
        selected = ordered[:1]
    elif scope.selection == "last":
        selected = ordered[-1:]
    else:
        selected = ordered
    status = (
        "ambiguous"
        if scope.selection == "unspecified" and len(selected) > 1
        else "resolved"
    )
    return _result(scope, status, ordered, selected)


def _matches_relation(
    relation: str,
    candidate: tuple[float, float],
    anchors: Sequence[tuple[float, float]],
) -> bool:
    if relation == "after":
        return candidate[0] >= max(anchor[1] for anchor in anchors)
    if relation == "before":
        return candidate[1] <= min(anchor[0] for anchor in anchors)
    if relation == "within":
        return any(
            min(candidate[1], anchor[1]) > max(candidate[0], anchor[0])
            for anchor in anchors
        )
    if len(anchors) < 2:
        return False
    lower = min(anchor[1] for anchor in anchors)
    upper = max(anchor[0] for anchor in anchors)
    return lower <= candidate[0] and candidate[1] <= upper


def _result(
    scope: TemporalScope,
    status: str,
    eligible: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "TemporalScopeResolutionV1",
        "scope_id": scope.scope_id,
        "relation": scope.relation,
        "selection": scope.selection,
        "status": status,
        "resolved": status == "resolved",
        "eligible_candidate_count": len(eligible),
        "eligible_occurrence_ids": [
            str(candidate.get("occurrence_id", "")) for candidate in eligible
        ],
        "selected_occurrence_ids": [
            str(candidate.get("occurrence_id", "")) for candidate in selected
        ],
    }


def _interval(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        start, end = sorted((float(value[0]), float(value[1])))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None
