from __future__ import annotations

import json
from pathlib import Path

from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, InvestigationReport
from visual_coding_agent_harness.workspace.digest import digest_reports
from visual_coding_agent_harness.workspace.evidence import EvidenceLedger
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace


def _query() -> ScopedQuery:
    return ScopedQuery(
        query_id="q1",
        goal_id="g1",
        natural_query="Find the red car.",
        scope=QueryScope(scene_ids=("sc01",), entity_hints=("car",), modality_hint=("visual",)),
        expected_evidence="A verified red car sighting.",
        budget=QueryBudget(max_shots_to_verify=1, max_frames=16),
    )


def test_investigator_workspace_writes_query_sidecars_and_coverage(tmp_path: Path) -> None:
    workspace = InvestigatorWorkspace(tmp_path)
    query = _query()
    candidate = CandidateShot(shot_id="sc01_sh001", score=0.88, reason="red object appears")
    finding = Finding(
        finding_id="ev_0001",
        query_id=query.query_id,
        shot_id=candidate.shot_id,
        summary="The red car is visible.",
        supports_options=("A",),
        refutes_options=("C",),
        citation_ids=("ev_0001",),
    )
    report = InvestigationReport(
        query_id=query.query_id,
        status="satisfied",
        findings=(finding,),
        explored_shots=(candidate.shot_id,),
        verified_shots=(candidate.shot_id,),
        unresolved=(),
        cost={"explore_calls": 1, "verify_calls": 1, "frames_read": 8},
    )

    workspace.record_request(query)
    workspace.record_explore(query.query_id, (candidate,))
    workspace.record_report(report)

    assert (tmp_path / "queries" / "q1" / "request.json").exists()
    assert json.loads((tmp_path / "queries" / "q1" / "explore.json").read_text(encoding="utf-8"))["candidates"][0][
        "shot_id"
    ] == "sc01_sh001"
    assert json.loads((tmp_path / "queries" / "q1" / "report.json").read_text(encoding="utf-8"))["status"] == "satisfied"
    assert json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8")) == {
        "explored_shots": ["sc01_sh001"],
        "verified_shots": ["sc01_sh001"],
    }
    assert "report_recorded" in (tmp_path / "trace.jsonl").read_text(encoding="utf-8")


def test_evidence_ledger_is_append_only_and_digest_is_compact(tmp_path: Path) -> None:
    ledger = EvidenceLedger(tmp_path / "evidence_ledger.jsonl")
    finding = Finding(
        finding_id="ev_0001",
        query_id="q1",
        shot_id="sc01_sh001",
        summary="The red car is visible.",
        supports_options=("A",),
        refutes_options=("C",),
        citation_ids=("ev_0001",),
    )
    report = InvestigationReport(
        query_id="q1",
        status="satisfied",
        findings=(finding,),
        explored_shots=("sc01_sh001",),
        verified_shots=("sc01_sh001",),
        unresolved=(),
        cost={"explore_calls": 1, "verify_calls": 1, "frames_read": 8},
    )

    ledger.append(finding)
    ledger.append(finding)
    digest = digest_reports((report,), query_goal_ids={"q1": "g1"})

    assert [item.finding_id for item in ledger.read_all()] == ["ev_0001", "ev_0001"]
    assert digest[0].query_id == "q1"
    assert digest[0].goal_id == "g1"
    assert digest[0].summary == "The red car is visible."
    assert digest[0].citation_ids == ("ev_0001",)
