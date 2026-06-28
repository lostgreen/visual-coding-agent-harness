from pathlib import Path

import pytest

from visual_coding_agent_harness.agents.multi import (
    Finding,
    FindingStatus,
    MultiAgentDriver,
    SubGoal,
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalIntent,
    SubGoalSuccessCriteria,
    WorkspaceMutator,
)
from visual_coding_agent_harness.agents.workspace_agent import WorkspaceRunResult
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.workspace import EvidenceWorkspace
from visual_coding_agent_harness.workspace.views import InvestigatorView, ReasonerView


class StubReasoner:
    def __init__(self, mutator: WorkspaceMutator) -> None:
        self.mutator = mutator
        self.answer_result: WorkspaceRunResult | None = None

    def step(self, *, round_number: int, question: str, options: dict[str, str]) -> bool:
        if round_number == 1:
            self.mutator.create_sub_goal(
                intent="verify",
                constraint=SubGoalConstraint(
                    segment_id="seg_0001",
                    time_range=(0.0, 20.0),
                    option_id="B",
                    claim="Check whether option B is visible in the local window.",
                    modality_hint=("visual",),
                ),
                budget=SubGoalBudget(max_explores=1, max_verifies=1, max_frames=16),
                success_criteria=SubGoalSuccessCriteria(needs_visual_support=True),
                parent_question=question,
                created_by="reasoner",
                created_round=round_number,
                rationale="Need local visual evidence before answering.",
            )
            return True
        findings = self.mutator.findings()
        if findings:
            self.answer_result = WorkspaceRunResult(
                answer="B",
                citations=findings[-1].memory_ids,
                confidence="medium",
                rounds=round_number,
                metadata={"strategy": "multi_agent_v0"},
            )
            return True
        return False


class StubInvestigator:
    def __init__(self, mutator: WorkspaceMutator) -> None:
        self.mutator = mutator

    def step(self, *, round_number: int) -> bool:
        sub_goal = self.mutator.claim_next_open_sub_goal(agent_id="investigator", round_number=round_number)
        if sub_goal is None:
            return False
        self.mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status="satisfied",
            memory_ids=("mem_0001",),
            coverage=(20.0, 20.0),
            notes_for_planner="Verified option B in the requested local window.",
            cost={"tool_calls": 1, "frames_read": 16, "tokens": 120},
            created_round=round_number,
        )
        return True


def test_multi_agent_protocol_exports_are_focused() -> None:
    assert SubGoalIntent.__args__ == ("locate", "verify", "disprove", "cover", "disambiguate")
    assert FindingStatus.__args__ == ("satisfied", "partial", "empty", "infeasible")
    sub_goal = SubGoal(
        sub_goal_id="sg_0001",
        intent="verify",
        constraint=SubGoalConstraint(claim="Check a visible object."),
        budget=SubGoalBudget(),
        success_criteria=SubGoalSuccessCriteria(),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    with pytest.raises(Exception):
        sub_goal.status = "done"  # type: ignore[misc]


def test_workspace_mutator_persists_sub_goal_state_machine(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)

    sub_goal = mutator.create_sub_goal(
        intent="locate",
        constraint=SubGoalConstraint(segment_id="seg_0001", claim="Find the relevant window."),
        budget=SubGoalBudget(max_explores=2),
        success_criteria=SubGoalSuccessCriteria(needs_visual_support=False),
        parent_question="Where does the event happen?",
        created_by="reasoner",
        created_round=1,
        rationale="Locate before verifying.",
    )

    assert sub_goal.sub_goal_id == "sg_0001"
    assert mutator.sub_goals()[-1].status == "open"

    claimed = mutator.claim_next_open_sub_goal(agent_id="investigator", round_number=2)
    assert claimed is not None
    assert claimed.sub_goal_id == "sg_0001"
    assert mutator.sub_goals()[-1].status == "in_progress"

    finding = mutator.report_finding(
        sub_goal_id="sg_0001",
        status="empty",
        memory_ids=(),
        coverage=(0.0, 20.0),
        notes_for_planner="No matching candidate appeared in this scope.",
        cost={"tool_calls": 1},
        created_round=3,
    )

    assert finding.finding_id == "find_0001"
    assert mutator.findings() == [finding]
    assert mutator.sub_goals()[-1].status == "done"

    replayed = WorkspaceMutator(workspace)
    assert replayed.sub_goals()[-1].status == "done"
    assert replayed.findings()[0].status == "empty"


def test_workspace_mutator_rejects_invalid_transitions(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    sub_goal = mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(claim="Verify one clue."),
        budget=SubGoalBudget(),
        success_criteria=SubGoalSuccessCriteria(),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    mutator.transition_sub_goal(sub_goal.sub_goal_id, to_status="abandoned", round_number=2)

    with pytest.raises(ValueError, match="invalid sub_goal transition"):
        mutator.transition_sub_goal(sub_goal.sub_goal_id, to_status="in_progress", round_number=3)


def test_multi_agent_driver_runs_reasoner_and_investigator(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    reasoner = StubReasoner(mutator)
    investigator = StubInvestigator(mutator)
    driver = MultiAgentDriver(reasoner=reasoner, investigator=investigator, workspace=workspace, max_rounds=4)

    result = driver.run("Question: Which option is supported?\nA. no\nB. yes", options={"A": "no", "B": "yes"})

    assert result.answer == "B"
    assert result.citations == ("mem_0001",)
    assert result.metadata is not None
    assert result.metadata["strategy"] == "multi_agent_v0"
    assert [goal.status for goal in mutator.sub_goals()] == ["done"]
    assert mutator.findings()[0].status == "satisfied"


def test_workspace_views_are_small_and_separated(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    sub_goal = mutator.create_sub_goal(
        intent="disambiguate",
        constraint=SubGoalConstraint(option_id="C", claim="Distinguish C from D."),
        budget=SubGoalBudget(max_verifies=2),
        success_criteria=SubGoalSuccessCriteria(needs_option_relation=True),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )

    reasoner_view = ReasonerView(mutator).render(question="Question?", options={"C": "claim C", "D": "claim D"})
    investigator_view = InvestigatorView(mutator).render(sub_goal)

    assert "# Reasoner View" in reasoner_view
    assert "Distinguish C from D." in reasoner_view
    assert "raw observation" not in reasoner_view.lower()
    assert "# Investigator View" in investigator_view
    assert "Remaining Budget" in investigator_view
    assert "Question?" in investigator_view


def test_investigator_verifies_candidate_and_reports_satisfied_finding(tmp_path: Path) -> None:
    from visual_coding_agent_harness.agents.multi import InvestigatorAgent

    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()

    @tool(name="verify_window", description="Fake candidate verifier.")
    def verify_window(candidate_key: str = "", checks=(), focus=(), sampling=None):
        assert candidate_key == "obs_0001:cand_0001"
        return {
            "mode": "verify_window",
            "claim": "Option B is visible in the window.",
            "confidence": 0.9,
            "verification_results": [
                {
                    "target_id": "option_B_check",
                    "claim": "Option B is visible.",
                    "verdict": "supported",
                    "evidence": "The red car is visible.",
                    "confidence": 0.9,
                    "option_id": "B",
                }
            ],
            "produced_anchors": [
                {
                    "anchor_id": "clip_anch_1",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0001",
                    "start_sec": 0.0,
                    "end_sec": 10.0,
                    "excerpt": "The red car is visible.",
                }
            ],
        }

    registry.register(verify_window)
    workspace.write_observation(
        tool_name="explore",
        claim="Explore found one candidate.",
        confidence=0.8,
        regions=[],
        raw_output={
            "mode": "candidate_discovery",
            "candidate_windows": [
                {
                    "candidate_key": "obs_0001:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 10.0],
                    "score": 0.9,
                }
            ],
        },
    )
    mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(
            segment_id="seg_0001",
            time_range=(0.0, 10.0),
            option_id="B",
            claim="Option B is visible.",
        ),
        budget=SubGoalBudget(max_explores=0, max_verifies=1),
        success_criteria=SubGoalSuccessCriteria(needs_visual_support=True),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    investigator = InvestigatorAgent(
        backend=object(),
        registry=registry,
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert investigator.step(round_number=2) is True

    findings = mutator.findings()
    assert findings[-1].status == "satisfied"
    assert findings[-1].memory_ids == ("mem_0001",)
    assert workspace.memory_entries()[0].kind == "visual_support"
    assert workspace.memory_entries()[0].supports_option == "B"
