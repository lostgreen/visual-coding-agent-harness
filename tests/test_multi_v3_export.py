from __future__ import annotations

import json
from pathlib import Path

from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.contracts.report import CandidateShot, Finding, InvestigationReport
from visual_coding_agent_harness.evals.videomme.outputs import (
    export_multi_v3_evidence_chains,
    export_multi_v3_exploration_records,
    export_multi_v3_training_trajectory,
    export_multi_v3_trajectory,
)
from visual_coding_agent_harness.workspace.investigator_ws import InvestigatorWorkspace


def test_multi_v3_exports_longvideoagent_trajectory_and_evidence_chains(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path / "run" / "multi_v3")
    trajectory_path = tmp_path / "run" / "artifacts" / "trajectories" / "longvideoagent_trajectory.json"
    chains_path = tmp_path / "run" / "artifacts" / "evidence_chains" / "evidence_chains.json"

    trajectory = export_multi_v3_trajectory(
        workspace,
        question="Question: Which object is visible?",
        video_path="/videos/demo.mp4",
        final={"answer": "A", "status": "final", "citations": ["ev_0001"]},
        reward_tags=("final", "has_citations"),
        output_path=trajectory_path,
    )
    chains = export_multi_v3_evidence_chains(workspace, output_path=chains_path)

    assert trajectory["schema_version"] == "LongVideoAgentTrajectoryV1"
    assert trajectory["state"]["workspace_root"] == workspace.root.as_posix()
    assert len(trajectory["actions"]) >= 2
    assert trajectory["observations"][0]["claim"] == "A red car is visible."
    assert trajectory_path.exists()
    assert chains["schema_version"] == "EvidenceChainsV1"
    assert chains["chain_count"] == 1
    assert chains_path.exists()


def test_multi_v3_exports_exploration_records_jsonl(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path / "run" / "multi_v3")
    output_path = tmp_path / "run" / "artifacts" / "exploration_records" / "exploration_records.jsonl"

    payload = export_multi_v3_exploration_records(workspace, output_path=output_path)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert payload["schema_version"] == "MultiV3ExplorationRecordsV1"
    assert payload["record_count"] == 1
    assert rows[0]["schema_version"] == "MultiV3ExplorationRecordV1"
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["request"]["goal_id"] == "g1"
    assert rows[0]["explore"]["candidate_count"] == 1
    assert rows[0]["verify"][0]["finding_count"] == 1
    assert rows[0]["report"]["status"] == "satisfied"
    assert rows[0]["artifacts"]["verify"] == ["queries/q1/verify_sc01_sh001.json"]


def test_multi_v3_exports_minimal_training_trajectory(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path / "run" / "multi_v3")
    output_path = tmp_path / "training.json"

    payload = export_multi_v3_training_trajectory(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        options=["A. red car", "B. blue car"],
        ground_truth="A",
        final_decision="final",
        selected_option="A",
        is_correct=True,
        output_path=output_path,
    )

    disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "TrainingTrajectoryV1"
    assert disk_payload["tool_calls"][0]["tool"] == "multi_v3_verify"
    assert disk_payload["tool_results"][0]["claim"] == "A red car is visible."
    assert disk_payload["evidence_chain_ids"] == [["ev_0001"]]


def _seed_workspace(root: Path) -> InvestigatorWorkspace:
    workspace = InvestigatorWorkspace(root)
    query = ScopedQuery(
        query_id="q1",
        goal_id="g1",
        natural_query="Find the red car.",
        scope=QueryScope(scene_ids=("sc01",), entity_hints=("car",), modality_hint=("visual",)),
        expected_evidence="A red car is visible.",
        budget=QueryBudget(max_shots_to_verify=1, max_frames=4),
    )
    finding = Finding(
        finding_id="ev_0001",
        query_id="q1",
        shot_id="sc01_sh001",
        summary="A red car is visible.",
        supports_options=("A",),
        citation_ids=("ev_0001",),
        confidence=0.9,
    )
    workspace.record_request(query)
    workspace.record_explore("q1", (CandidateShot("sc01_sh001", 0.95, "red car"),))
    workspace.record_verify("q1", "sc01_sh001", (finding,))
    workspace.record_report(
        InvestigationReport(
            query_id="q1",
            status="satisfied",
            findings=(finding,),
            explored_shots=("sc01_sh001",),
            verified_shots=("sc01_sh001",),
            unresolved=(),
            cost={"explore_calls": 1, "verify_calls": 1, "frames_read": 2},
        )
    )
    return workspace
