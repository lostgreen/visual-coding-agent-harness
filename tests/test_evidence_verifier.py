from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.agents.multi import (
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalSuccessCriteria,
    WorkspaceMutator,
)
from visual_coding_agent_harness.agents.multi.evidence_verifier import EvidenceVerifier
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_evidence_verifier_commits_supported_result_and_real_cost(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()
    sub_goal = mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(option_id="B", claim="Option B is visible."),
        budget=SubGoalBudget(max_explores=1, max_verifies=1, max_frames=32),
        success_criteria=SubGoalSuccessCriteria(),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    claimed = mutator.claim_next_open_sub_goal(agent_id="evidence_scout", round_number=2)
    assert claimed is not None
    candidate = mutator.record_candidates(
        need_id=sub_goal.sub_goal_id,
        option_id="B",
        candidates=[
            {
                "candidate_key": "obs_0001:cand_0001",
                "segment_id": "seg_0002",
                "time_range": [20.0, 30.0],
                "score": 0.8,
            }
        ],
        round_number=2,
    )[0]

    @tool(name="verify_window", description="Fake verifier.")
    def verify_window(candidate_key: str = "", checks=(), focus=(), sampling=None):
        assert candidate_key == "obs_0001:cand_0001"
        return {
            "mode": "verify_window",
            "claim": "Option B appears.",
            "confidence": 0.9,
            "frames_read": 12,
            "verification_results": [
                {
                    "target_id": "option_B_check",
                    "claim": "Option B is visible.",
                    "verdict": "supported",
                    "evidence": "Option B appears in the window.",
                    "confidence": 0.9,
                    "option_id": "B",
                }
            ],
            "produced_anchors": [
                {
                    "anchor_id": "anch_b",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0002",
                    "start_sec": 20.0,
                    "end_sec": 30.0,
                    "excerpt": "Option B appears in the window.",
                }
            ],
        }

    registry.register(verify_window)
    verifier = EvidenceVerifier(registry=registry, mutator=mutator, workspace=workspace)

    finding = verifier.verify(sub_goal=claimed, candidate=candidate, round_number=3, explore_calls=1)

    assert finding.status == "satisfied"
    assert finding.memory_ids == ("mem_0001",)
    assert finding.cost["explore_calls"] == 1
    assert finding.cost["verify_calls"] == 1
    assert finding.cost["tool_calls"] == 2
    assert finding.cost["frames_read"] == 12
    assert mutator.evidence_candidates()[0]["consumed_by_finding_id"] == finding.finding_id
    assert workspace.memory_entries()[0].kind == "visual_support"
    event_types = [event["type"] for event in workspace._read_jsonl_dicts("trace.jsonl")]
    assert "evidence_verifier_committed" in event_types
    assert "evidence_need_closed" in event_types


def test_evidence_verifier_accepts_segment_time_range_candidate(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()
    sub_goal = mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(
            option_id="D",
            claim="Option D is visible.",
            segment_id="seg_0003",
            time_range=(40.0, 60.0),
        ),
        budget=SubGoalBudget(max_explores=1, max_verifies=1, max_frames=32),
        success_criteria=SubGoalSuccessCriteria(),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    claimed = mutator.claim_next_open_sub_goal(agent_id="evidence_scout", round_number=2)
    assert claimed is not None
    candidate = mutator.record_candidates(
        need_id=sub_goal.sub_goal_id,
        option_id="D",
        candidates=[
            {
                "candidate_key": "",
                "segment_id": "seg_0003",
                "time_range": [40.0, 60.0],
                "score": 0.0,
                "source": "scout_segment_sweep",
            }
        ],
        round_number=2,
    )[0]

    @tool(name="verify_window", description="Fake explicit-window verifier.")
    def verify_window(candidate_key: str = "", segment_id: str = "", time_range=None, checks=(), focus=(), sampling=None):
        assert candidate_key == ""
        assert segment_id == "seg_0003"
        assert time_range == [40.0, 60.0]
        return {
            "mode": "verify_window",
            "claim": "Option D is absent in this window.",
            "confidence": 0.7,
            "frames_read": 8,
            "verification_results": [
                {
                    "target_id": "option_D_check",
                    "claim": "Option D is visible.",
                    "verdict": "not_found_in_window",
                    "evidence": "No matching evidence in the swept window.",
                    "confidence": 0.7,
                    "option_id": "D",
                }
            ],
            "produced_anchors": [
                {
                    "anchor_id": "anch_d_sweep",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0003",
                    "start_sec": 40.0,
                    "end_sec": 60.0,
                    "excerpt": "No matching evidence in the swept window.",
                }
            ],
        }

    registry.register(verify_window)
    verifier = EvidenceVerifier(registry=registry, mutator=mutator, workspace=workspace)

    finding = verifier.verify(sub_goal=claimed, candidate=candidate, round_number=3, explore_calls=1)

    assert finding.status == "empty"
    assert finding.cost["frames_read"] == 8
    assert mutator.evidence_candidates()[0]["consumed_by_finding_id"] == finding.finding_id
