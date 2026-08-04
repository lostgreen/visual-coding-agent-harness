from vcah.temporal_scope import TemporalScope, resolve_temporal_scope
from vcah.workspace import WorkingDocument


def test_after_first_scope_orders_candidates_mechanically() -> None:
    scope = TemporalScope(
        scope_id="scope_after",
        relation="after",
        selection="first",
        anchor_requirement_id="req_anchor",
        target_requirement_id="req_target",
    )

    result = resolve_temporal_scope(
        scope,
        anchor_intervals=((10.0, 12.0),),
        candidates=(
            {"occurrence_id": "occ_late", "time_range": [30.0, 31.0]},
            {"occurrence_id": "occ_before", "time_range": [5.0, 6.0]},
            {"occurrence_id": "occ_first", "time_range": [20.0, 21.0]},
        ),
    )

    assert result["resolved"]
    assert result["eligible_occurrence_ids"] == ["occ_first", "occ_late"]
    assert result["selected_occurrence_ids"] == ["occ_first"]


def test_scope_stays_unresolved_without_anchor_material() -> None:
    scope = TemporalScope(
        scope_id="scope_next",
        relation="after",
        selection="next",
        anchor_requirement_id="req_anchor",
        target_requirement_id="req_target",
    )

    result = resolve_temporal_scope(
        scope,
        anchor_intervals=(),
        candidates=({"occurrence_id": "occ_1", "time_range": [20.0, 21.0]},),
    )
    assert result["status"] == "unresolved_anchor"
    assert not result["resolved"]


def test_temporal_scope_requires_existing_obligations() -> None:
    document = WorkingDocument.with_question_premise("What happens next?")
    accepted = document.apply_ops(
        (
            {
                "op": "add_obligation",
                "requirement_id": "req_anchor",
                "observable_goal": "Observe the anchor.",
                "evidence_kind": "relation",
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_target",
                "observable_goal": "Observe the next target.",
                "evidence_kind": "relation",
                "depends_on": ["req_anchor"],
            },
            {
                "op": "add_temporal_scope",
                "scope_id": "scope_1",
                "relation": "after",
                "selection": "next",
                "anchor_requirement_id": "req_anchor",
                "target_requirement_id": "req_target",
            },
        ),
        observation_ids=(),
    )

    assert accepted.accepted
    assert document.temporal_scopes["scope_1"].selection == "next"
