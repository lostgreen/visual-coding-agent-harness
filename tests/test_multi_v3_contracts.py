from __future__ import annotations

import pytest

from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery, VerifiableGoal
from visual_coding_agent_harness.contracts.report import (
    CandidateShot,
    DigestItem,
    Finding,
    InvestigationReport,
    VerifyRequest,
)


def test_scoped_query_requires_explicit_scene_scope() -> None:
    with pytest.raises(ValueError, match="scene_ids"):
        QueryScope(scene_ids=())


def test_query_contracts_round_trip_with_budget_and_scope() -> None:
    goal = VerifiableGoal(
        goal_id="g1",
        text="Find whether the red car appears.",
        linked_options=("A", "C"),
        kind="locate",
        priority=0.8,
    )
    query = ScopedQuery(
        query_id="q1",
        goal_id=goal.goal_id,
        natural_query="Look for the red car.",
        scope=QueryScope(scene_ids=("sc01",), time_range=(0.0, 90.0), entity_hints=("car",), modality_hint=("visual",)),
        expected_evidence="A verified shot showing the red car.",
        budget=QueryBudget(max_shots_to_verify=2, max_frames=32),
    )

    decoded = ScopedQuery.from_dict(query.to_dict())

    assert decoded == query
    assert decoded.scope.scene_ids == ("sc01",)
    assert decoded.budget.max_shots_to_verify == 2


def test_scoped_query_carries_split_text_and_visual_queries() -> None:
    query = ScopedQuery(
        query_id="q_split",
        goal_id="g1",
        natural_query="Find the person by the door.",
        text_queries=("doorbell transcript phrase",),
        visual_queries=("person standing beside the door", "door close-up"),
        scope=QueryScope(scene_ids=("sc01",)),
        expected_evidence="A person is beside the door.",
    )

    decoded = ScopedQuery.from_dict(query.to_dict())
    fallback = ScopedQuery(
        query_id="q_fallback",
        goal_id="g1",
        natural_query="fallback query",
        scope=QueryScope(scene_ids=("sc01",)),
        expected_evidence="fallback evidence",
    )

    assert decoded.text_queries == ("doorbell transcript phrase",)
    assert decoded.visual_queries == ("person standing beside the door", "door close-up")
    assert fallback.text_queries == ("fallback query",)
    assert fallback.visual_queries == ("fallback query",)


def test_report_contracts_round_trip_and_digest_shape() -> None:
    request = VerifyRequest(
        shot_id="sc01_sh001",
        time_range=(4.0, 12.0),
        focus_claim="The red car is visible.",
        sampling={"fps": 2, "max_frames": 32, "resolution": "high"},
        checks=({"target_id": "g1", "claim": "red car visible", "polarity": "presence"},),
    )
    finding = Finding(
        finding_id="ev_0001",
        query_id="q1",
        shot_id=request.shot_id,
        summary="A red car appears beside the curb.",
        supports_options=("A",),
        refutes_options=("C",),
        citation_ids=("ev_0001",),
        confidence=0.91,
    )
    report = InvestigationReport(
        query_id="q1",
        status="satisfied",
        findings=(finding,),
        explored_shots=("sc01_sh001", "sc01_sh002"),
        verified_shots=("sc01_sh001",),
        unresolved=(),
        cost={"explore_calls": 1, "verify_calls": 1, "frames_read": 18},
    )
    digest = DigestItem.from_report(report, goal_id="g1")

    assert VerifyRequest.from_dict(request.to_dict()) == request
    assert CandidateShot.from_dict(CandidateShot("sc01_sh001", 0.9, "visible car").to_dict()).shot_id == "sc01_sh001"
    assert InvestigationReport.from_dict(report.to_dict()) == report
    assert digest.supports_options == ("A",)
    assert digest.refutes_options == ("C",)
    assert digest.citation_ids == ("ev_0001",)


def test_digest_item_truncates_long_summaries() -> None:
    finding = Finding(
        finding_id="ev_0001",
        query_id="q1",
        shot_id="sc01_sh001",
        summary="x" * 300,
        citation_ids=("ev_0001",),
    )
    report = InvestigationReport(
        query_id="q1",
        status="satisfied",
        findings=(finding,),
        explored_shots=("sc01_sh001",),
        verified_shots=("sc01_sh001",),
        unresolved=(),
        cost={},
    )

    digest = DigestItem.from_report(report, goal_id="g1")

    assert len(digest.summary) == 240
    assert digest.summary.endswith("...")
