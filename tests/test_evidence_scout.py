from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.agents.multi import (
    SubGoal,
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalSuccessCriteria,
    WorkspaceMutator,
)
from visual_coding_agent_harness.agents.multi.evidence_scout import EvidenceScout
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.memory import SourceAnchor
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_evidence_scout_excludes_already_verified_window_for_same_option(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()
    _write_verified_memory(workspace, option_id="D", segment_id="seg_0001", start_sec=0.0, end_sec=10.0)

    @tool(name="explore", description="Fake explorer with one stale and one fresh candidate.")
    def explore(query: str = "", targets=(), scope=None, modalities=(), top_k: int = 3, original_question: str = ""):
        return {
            "mode": "candidate_discovery",
            "claim": "Explorer proposed two windows.",
            "confidence": 0.8,
            "candidate_windows": [
                {
                    "candidate_key": "obs_old:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 10.0],
                    "score": 0.95,
                    "modalities": ["visual"],
                },
                {
                    "candidate_key": "obs_new:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [20.0, 30.0],
                    "score": 0.75,
                    "modalities": ["visual"],
                },
            ],
        }

    registry.register(explore)
    sub_goal = _sub_goal(option_id="D")
    scout = EvidenceScout(registry=registry, mutator=mutator, workspace=workspace)

    candidate = scout.propose_candidate(sub_goal, round_number=2)

    assert candidate is not None
    assert candidate["candidate_key"] == "obs_new:cand_0001"
    assert candidate["candidate_id"] == "ec_0001"
    recorded = mutator.evidence_candidates()
    assert [row["candidate_key"] for row in recorded] == ["obs_new:cand_0001"]
    event_types = [event["type"] for event in workspace._read_jsonl_dicts("trace.jsonl")]
    assert "evidence_scout_window_excluded" in event_types
    assert "evidence_scout_candidates_proposed" in event_types


def test_evidence_scout_persists_explore_observation_for_fresh_candidates(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()

    @tool(name="explore", description="Fake explorer.")
    def explore(query: str = "", targets=(), scope=None, modalities=(), top_k: int = 3, original_question: str = ""):
        return {
            "mode": "candidate_discovery",
            "claim": "Explorer proposed one window.",
            "confidence": 0.8,
            "candidate_windows": [
                {
                    "candidate_key": "obs_new:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [20.0, 30.0],
                    "score": 0.75,
                    "modalities": ["visual"],
                }
            ],
        }

    registry.register(explore)
    scout = EvidenceScout(registry=registry, mutator=mutator, workspace=workspace)

    candidate = scout.propose_candidate(_sub_goal(option_id="C"), round_number=2)

    assert candidate is not None
    observations = workspace.read_observations()
    assert len(observations) == 1
    assert observations[0].tool == "explore"
    assert observations[0].raw_output["multi_agent_option_id"] == "C"


def test_evidence_scout_prefers_fresh_explore_when_budget_allows(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()
    workspace.write_observation(
        tool_name="explore",
        claim="Earlier candidate for option D.",
        confidence=0.8,
        regions=[],
        raw_output={
            "mode": "candidate_discovery",
            "multi_agent_option_id": "D",
            "candidate_windows": [
                {
                    "candidate_key": "obs_old:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 10.0],
                    "score": 0.95,
                }
            ],
        },
    )

    @tool(name="explore", description="Fresh option explorer.")
    def explore(query: str = "", targets=(), scope=None, modalities=(), top_k: int = 3, original_question: str = ""):
        return {
            "mode": "candidate_discovery",
            "claim": "Fresh candidate for option D.",
            "confidence": 0.8,
            "candidate_windows": [
                {
                    "candidate_key": "obs_fresh:cand_0001",
                    "segment_id": "seg_0002",
                    "time_range": [40.0, 50.0],
                    "score": 0.7,
                }
            ],
        }

    registry.register(explore)
    scout = EvidenceScout(registry=registry, mutator=mutator, workspace=workspace)

    candidate = scout.propose_candidate(_sub_goal(option_id="D"), round_number=2)

    assert candidate is not None
    assert candidate["candidate_key"] == "obs_fresh:cand_0001"


def test_evidence_scout_reuses_existing_candidate_when_explore_budget_is_zero(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()
    workspace.write_observation(
        tool_name="explore",
        claim="Earlier candidate for option B.",
        confidence=0.8,
        regions=[],
        raw_output={
            "mode": "candidate_discovery",
            "multi_agent_option_id": "B",
            "candidate_windows": [
                {
                    "candidate_key": "obs_existing:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 10.0],
                    "score": 0.95,
                }
            ],
        },
    )

    scout = EvidenceScout(registry=registry, mutator=mutator, workspace=workspace)
    sub_goal = _sub_goal(option_id="B", max_explores=0)

    candidate = scout.propose_candidate(sub_goal, round_number=2)

    assert candidate is not None
    assert candidate["candidate_key"] == "obs_existing:cand_0001"


def test_evidence_scout_does_not_share_consumed_positive_candidate_across_options(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()
    _write_verified_memory(workspace, option_id="C", segment_id="seg_0004", start_sec=1123.26, end_sec=1139.47)
    recorded = mutator.record_candidates(
        need_id="sg_0003",
        option_id="C",
        candidates=[
            {
                "candidate_key": "obs_0005:cand_0001",
                "segment_id": "seg_0004",
                "time_range": [1123.26, 1139.47],
                "score": 0.4,
                "source": "scout_explore_hit",
            }
        ],
        round_number=3,
    )[0]
    mutator.mark_candidate_consumed(str(recorded["candidate_id"]), finding_id="find_0003")
    workspace.write_observation(
        tool_name="explore",
        claim="Earlier candidate for option C.",
        confidence=0.8,
        regions=[],
        raw_output={
            "mode": "candidate_discovery",
            "multi_agent_option_id": "C",
            "candidate_windows": [
                {
                    "candidate_key": "obs_0005:cand_0001",
                    "segment_id": "seg_0004",
                    "time_range": [1123.26, 1139.47],
                    "score": 0.4,
                }
            ],
        },
    )

    @tool(name="explore", description="No fresh candidates.")
    def explore(query: str = "", targets=(), scope=None, modalities=(), top_k: int = 3, original_question: str = ""):
        return {
            "mode": "candidate_discovery",
            "claim": "No new candidates.",
            "confidence": 0.1,
            "candidate_windows": [],
        }

    registry.register(explore)
    scout = EvidenceScout(registry=registry, mutator=mutator, workspace=workspace)

    candidate = scout.propose_candidate(_sub_goal(option_id="D"), round_number=4)

    assert candidate is None


def _sub_goal(*, option_id: str, max_explores: int = 1) -> SubGoal:
    return SubGoal(
        sub_goal_id="sg_0001",
        intent="verify",
        constraint=SubGoalConstraint(option_id=option_id, claim=f"Check option {option_id}."),
        budget=SubGoalBudget(max_explores=max_explores, max_verifies=1),
        success_criteria=SubGoalSuccessCriteria(),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )


def _write_verified_memory(
    workspace: EvidenceWorkspace,
    *,
    option_id: str,
    segment_id: str,
    start_sec: float,
    end_sec: float,
) -> None:
    workspace.write_produced_anchors(
        [
            SourceAnchor(
                anchor_id="anch_verified",
                observation_id="obs_verified",
                source_kind="visual_fact",
                segment_id=segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                field_path="verification_results.0",
                excerpt="Verified evidence.",
                modality="visual",
            )
        ]
    )
    workspace.write_memory(
        kind="visual_support",
        claim="Verified evidence.",
        anchors=[{"anchor_id": "anch_verified"}],
        supports_option=option_id,
        confidence="high",
        metadata={"verdict": "supported"},
    )
