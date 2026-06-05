import json
from pathlib import Path

from runs.training_trajectory import TrainingTrajectory
from visual_coding_agent_harness.workspace import EvidenceRecord, EvidenceWorkspace


def test_from_workspace_collects_all_chains_and_trace_context(tmp_path):
    workspace = _workspace_with_chain(tmp_path)
    workspace.write_trace_event("tool_use", {"step": 1, "tool": "vision_read", "arguments": {"segment_id": "seg_0001"}})
    workspace.write_trace_event(
        "context_budget_report",
        {"used_tokens_per_slot": {"task": 10, "navigation": 5, "evidence": 8, "feedback": 1}, "overflow": False},
    )
    workspace.write_trace_event("hard_skill_followup_handoff", {"rounds": 1})

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        options=["A. blue car", "B. red car"],
        ground_truth="B",
        final_decision="final",
        selected_option="B",
        is_correct=True,
        output_path="artifacts/training/case_001.json",
    )

    assert trajectory.schema_version == "TrainingTrajectoryV1"
    assert trajectory.case_id == "case_001"
    assert trajectory.evidence_chain_ids == [[record.evidence_id for record in _chain_records(workspace)]]
    assert trajectory.tool_calls[0]["tool"] == "vision_read"
    assert trajectory.context_budget_reports[0]["overflow"] is False
    assert trajectory.followup_history[0]["type"] == "hard_skill_followup_handoff"
    assert Path(trajectory.trajectory_path).exists()


def test_training_trajectory_export_does_not_inline_raw_output(tmp_path):
    workspace = _workspace_with_chain(tmp_path)

    trajectory = TrainingTrajectory.from_workspace(
        workspace,
        case_id="case_001",
        question="Which object is visible?",
        final_decision="final",
        selected_option="B",
        output_path="artifacts/training/case_001.json",
    )

    payload = json.loads(Path(trajectory.trajectory_path).read_text(encoding="utf-8"))
    assert "raw_output" not in json.dumps(payload)


def _workspace_with_chain(tmp_path) -> EvidenceWorkspace:
    workspace = EvidenceWorkspace.create(tmp_path, run_id="trajectory_case")
    observation = workspace.write_observation(
        tool_name="vision_read",
        claim="The localized window shows a red car.",
        confidence=0.9,
        regions=[{"segment_id": "seg_0001", "start_sec": 0.0, "end_sec": 12.0}],
        raw_output={"secret_raw_output": "should not be exported"},
    )
    distilled, ledger, mapped = _chain_records(workspace, observation_id=observation.observation_id)
    for record in [distilled, ledger, mapped]:
        workspace.write_evidence(record)
    return workspace


def _chain_records(workspace: EvidenceWorkspace, observation_id: str = "obs_0001") -> list[EvidenceRecord]:
    distilled = EvidenceRecord(
        evidence_id=f"ev_distilled_{workspace.run_id}_00001",
        stage="distilled",
        parent_id=None,
        tool="vision_read",
        observation_id=observation_id,
        frame_set_id=None,
        content={"claim": "red car"},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    ledger = EvidenceRecord(
        evidence_id=f"ev_ledger_{workspace.run_id}_00002",
        stage="ledger",
        parent_id=distilled.evidence_id,
        tool="vision_read",
        observation_id=observation_id,
        frame_set_id=None,
        content={"claim": "red car"},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    mapped = EvidenceRecord(
        evidence_id=f"ev_mapped_{workspace.run_id}_00003",
        stage="mapped",
        parent_id=ledger.evidence_id,
        tool="vision_read",
        observation_id=observation_id,
        frame_set_id=None,
        content={"candidate_option_relation": {"option": "B", "relation": "support", "strength": 0.9}},
        grounding_quality="visually_confirmed",
        confidence=0.9,
        created_at=1.0,
    )
    return [distilled, ledger, mapped]
