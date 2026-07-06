from __future__ import annotations

from vcah.entities import EntityObservation, IdentityRelation, count_entity_bounds
from vcah.types import CoverageSegment


def test_entity_count_bounds_without_reid_are_conservative() -> None:
    observations = (
        EntityObservation("obs1", "ev_0001", 10.0, role_hint="person"),
        EntityObservation("obs2", "ev_0002", 20.0, role_hint="person"),
    )

    result = count_entity_bounds(observations)

    assert result.lower_bound == 1
    assert result.upper_bound == 2
    assert result.exact_count is None


def test_different_entity_relation_raises_lower_bound() -> None:
    observations = (
        EntityObservation("obs1", "ev_0001", 10.0),
        EntityObservation("obs2", "ev_0002", 20.0),
    )
    result = count_entity_bounds(
        observations,
        (IdentityRelation("obs1", "obs2", "different_entity", 0.9, ("ev_rel",)),),
        (CoverageSegment("win_full", 0.0, 30.0, "visual", 1.0),),
    )

    assert result.lower_bound == 2
    assert result.upper_bound == 2
    assert result.exact_count == 2


def test_same_entity_relation_merges_upper_bound() -> None:
    observations = (
        EntityObservation("obs1", "ev_0001", 10.0),
        EntityObservation("obs2", "ev_0002", 20.0),
    )
    result = count_entity_bounds(
        observations,
        (IdentityRelation("obs1", "obs2", "same_entity", 0.9, ("ev_rel",)),),
        (CoverageSegment("win_full", 0.0, 30.0, "visual", 1.0),),
    )

    assert result.lower_bound == 1
    assert result.upper_bound == 1
    assert result.exact_count == 1
