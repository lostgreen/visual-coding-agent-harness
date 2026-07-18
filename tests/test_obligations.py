from __future__ import annotations

from vcah.obligations import compile_query_obligations, evaluate_query_obligations
from vcah.types import ClaimContract


def test_last_instance_cross_window_identity_escalates_to_full_video() -> None:
    contract = ClaimContract(
        required_scope="multi_window",
        quantifier="comparison",
        observation_target="entity",
        aggregation="compare",
    )
    obligations = compile_query_obligations(
        contract,
        {
            "requires_event_participant_link": True,
            "requires_temporal_sequence": True,
            "requires_temporal_extremum": True,
        },
    )

    assert obligations.contract_scope == "multi_window"
    assert obligations.effective_scope == "full_video"
    assert "req_temporal_max_episode" in obligations.scope_escalation_requirement_ids
    assert "req_cross_window_entity_binding" in {
        obligation.requirement_id for obligation in obligations.obligations
    }


def test_uncovered_effective_full_video_blocks_temporal_max_obligation() -> None:
    contract = ClaimContract(
        required_scope="multi_window",
        quantifier="comparison",
        observation_target="entity",
        aggregation="compare",
    )
    obligations = compile_query_obligations(
        contract,
        {"requires_temporal_extremum": True},
    )
    evaluations = {
        row["requirement_id"]: row
        for row in evaluate_query_obligations(
            obligations,
            {"qualified_events": [{"candidate_id": "event_1"}]},
            {"range_coverage_complete": False, "enumeration_complete": False},
        )
    }

    assert evaluations["req_temporal_extremum_coverage"]["status"] == "unknown"
    assert evaluations["req_temporal_max_episode"]["status"] == "unknown"
