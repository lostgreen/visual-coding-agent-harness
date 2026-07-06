from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from vcah.types import CoverageSegment


@dataclass(frozen=True)
class EntityObservation:
    observation_id: str
    evidence_id: str
    seen_at_sec: float
    role_hint: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", str(self.observation_id or "").strip())
        object.__setattr__(self, "evidence_id", str(self.evidence_id or "").strip())
        object.__setattr__(self, "seen_at_sec", float(self.seen_at_sec))
        object.__setattr__(self, "attributes", dict(self.attributes))


@dataclass(frozen=True)
class IdentityRelation:
    left_observation_id: str
    right_observation_id: str
    relation: Literal["same_entity", "different_entity", "unknown"]
    confidence: float
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_observation_id", str(self.left_observation_id or "").strip())
        object.__setattr__(self, "right_observation_id", str(self.right_observation_id or "").strip())
        relation = self.relation if self.relation in {"same_entity", "different_entity", "unknown"} else "unknown"
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "evidence_ids", tuple(str(item) for item in self.evidence_ids if str(item).strip()))


@dataclass(frozen=True)
class EntityCountResult:
    lower_bound: int
    upper_bound: int | None
    confidence: float
    supporting_observation_ids: tuple[str, ...]
    coverage_manifest: tuple[CoverageSegment, ...]

    @property
    def exact_count(self) -> int | None:
        return self.lower_bound if self.upper_bound == self.lower_bound else None


def count_entity_bounds(
    observations: tuple[EntityObservation, ...],
    relations: tuple[IdentityRelation, ...] = (),
    coverage_manifest: tuple[CoverageSegment, ...] = (),
) -> EntityCountResult:
    ids = tuple(obs.observation_id for obs in observations)
    parent = {obs_id: obs_id for obs_id in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for relation in relations:
        if relation.relation != "same_entity":
            continue
        if relation.left_observation_id in parent and relation.right_observation_id in parent:
            parent[find(relation.right_observation_id)] = find(relation.left_observation_id)

    groups = {find(obs_id) for obs_id in ids}
    explicit_different = {
        tuple(sorted((relation.left_observation_id, relation.right_observation_id)))
        for relation in relations
        if relation.relation == "different_entity"
    }
    lower_bound = max(1 if observations else 0, _different_lower_bound(ids, explicit_different))
    upper_bound = len(groups) if ids else 0
    if any(relation.relation == "unknown" for relation in relations):
        upper_bound = len(ids)
    return EntityCountResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence=1.0 if lower_bound == upper_bound and coverage_manifest else 0.5,
        supporting_observation_ids=ids,
        coverage_manifest=coverage_manifest,
    )


def _different_lower_bound(ids: tuple[str, ...], pairs: set[tuple[str, str]]) -> int:
    if not ids:
        return 0
    best = 1
    for obs_id in ids:
        different = {other for pair in pairs if obs_id in pair for other in pair if other != obs_id}
        best = max(best, 1 + len(different))
    return best
