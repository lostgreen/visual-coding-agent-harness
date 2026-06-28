import json
from pathlib import Path

import pytest

from visual_coding_agent_harness.agents.multi import (
    Finding,
    FindingStatus,
    MultiAgentDriver,
    ReasonerAgent,
    SubGoal,
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalIntent,
    SubGoalSuccessCriteria,
    WorkspaceMutator,
)
from visual_coding_agent_harness.agents.workspace_agent import WorkspaceRunResult
from visual_coding_agent_harness.core.registry import ToolRegistry, tool
from visual_coding_agent_harness.memory import SourceAnchor
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


def test_sub_goal_budget_round_trip_preserves_zero_limits() -> None:
    sub_goal = SubGoal(
        sub_goal_id="sg_0001",
        intent="verify",
        constraint=SubGoalConstraint(claim="Check one candidate."),
        budget=SubGoalBudget(max_explores=0, max_verifies=0, max_frames=0),
        success_criteria=SubGoalSuccessCriteria(),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )

    decoded = SubGoal.from_dict(sub_goal.to_dict())

    assert decoded.budget.max_explores == 0
    assert decoded.budget.max_verifies == 0
    assert decoded.budget.max_frames == 0


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


def test_multi_agent_driver_allows_final_reasoner_pass_after_last_investigator_step(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    reasoner = StubReasoner(mutator)
    investigator = StubInvestigator(mutator)
    driver = MultiAgentDriver(reasoner=reasoner, investigator=investigator, workspace=workspace, max_rounds=1)

    result = driver.run("Question: Which option is supported?\nA. no\nB. yes", options={"A": "no", "B": "yes"})

    assert result.answer == "B"
    assert result.citations == ("mem_0001",)
    assert result.metadata is not None
    assert result.metadata["strategy"] == "multi_agent_v0"


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
            "multi_agent_option_id": "B",
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


def test_investigator_does_not_reuse_candidate_from_another_option(tmp_path: Path) -> None:
    from visual_coding_agent_harness.agents.multi import InvestigatorAgent

    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()

    @tool(name="explore", description="Fake option-specific explorer.")
    def explore(query: str = "", targets=(), scope=None, modalities=(), top_k: int = 3, original_question: str = ""):
        return {
            "mode": "candidate_discovery",
            "claim": "Explore found an option B candidate.",
            "confidence": 0.8,
            "candidate_windows": [
                {
                    "candidate_key": "obs_0002:cand_0001",
                    "segment_id": "seg_0002",
                    "time_range": [20.0, 30.0],
                    "score": 0.9,
                }
            ],
        }

    @tool(name="verify_window", description="Fake verifier.")
    def verify_window(candidate_key: str = "", checks=(), focus=(), sampling=None):
        assert candidate_key == "obs_0002:cand_0001"
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
                    "anchor_id": "clip_anch_b",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0002",
                    "start_sec": 20.0,
                    "end_sec": 30.0,
                    "excerpt": "The red car is visible.",
                }
            ],
        }

    registry.register(explore)
    registry.register(verify_window)
    workspace.write_observation(
        tool_name="explore",
        claim="Explore found an option A candidate.",
        confidence=0.8,
        regions=[],
        raw_output={
            "mode": "candidate_discovery",
            "multi_agent_option_id": "A",
            "candidate_windows": [
                {
                    "candidate_key": "obs_0001:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 10.0],
                    "score": 1.0,
                }
            ],
        },
    )
    mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(option_id="B", claim="Option B is visible."),
        budget=SubGoalBudget(max_explores=1, max_verifies=1),
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

    assert mutator.findings()[-1].status == "satisfied"
    assert workspace.memory_entries()[-1].supports_option == "B"


def test_investigator_falls_back_to_shared_candidate_when_explore_is_deduped(tmp_path: Path) -> None:
    from visual_coding_agent_harness.agents.multi import InvestigatorAgent

    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()

    @tool(name="explore", description="Fake deduped explorer.")
    def explore(query: str = "", targets=(), scope=None, modalities=(), top_k: int = 3, original_question: str = ""):
        return {
            "mode": "candidate_discovery",
            "claim": "Explore produced no new candidate windows; existing pending candidates already cover these regions.",
            "confidence": 0.2,
            "candidate_windows": [],
        }

    @tool(name="verify_window", description="Fake shared-window verifier.")
    def verify_window(candidate_key: str = "", checks=(), focus=(), sampling=None):
        assert candidate_key == "obs_0001:cand_0001"
        return {
            "mode": "verify_window",
            "claim": "Option B is supported in the shared window.",
            "confidence": 0.9,
            "verification_results": [
                {
                    "target_id": "option_B_check",
                    "claim": "Option B is supported.",
                    "verdict": "supported",
                    "evidence": "The shared window supports option B.",
                    "confidence": 0.9,
                    "option_id": "B",
                }
            ],
            "produced_anchors": [
                {
                    "anchor_id": "clip_anch_shared_b",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0001",
                    "start_sec": 0.0,
                    "end_sec": 10.0,
                    "excerpt": "The shared window supports option B.",
                }
            ],
        }

    registry.register(explore)
    registry.register(verify_window)
    workspace.write_observation(
        tool_name="explore",
        claim="Explore found a shared candidate while checking option A.",
        confidence=0.8,
        regions=[],
        raw_output={
            "mode": "candidate_discovery",
            "multi_agent_option_id": "A",
            "candidate_windows": [
                {
                    "candidate_key": "obs_0001:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 10.0],
                    "score": 1.0,
                }
            ],
        },
    )
    mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(option_id="B", claim="Question? Option B: shared answer."),
        budget=SubGoalBudget(max_explores=1, max_verifies=1),
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

    assert mutator.findings()[-1].status == "satisfied"
    assert workspace.memory_entries()[-1].supports_option == "B"


def test_investigator_reports_empty_when_verify_only_finds_local_negative(tmp_path: Path) -> None:
    from visual_coding_agent_harness.agents.multi import InvestigatorAgent

    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    registry = ToolRegistry()

    @tool(name="verify_window", description="Fake negative verifier.")
    def verify_window(candidate_key: str = "", checks=(), focus=(), sampling=None):
        return {
            "mode": "verify_window",
            "claim": "Option B was not found.",
            "confidence": 0.8,
            "verification_results": [
                {
                    "target_id": "option_B_check",
                    "claim": "Option B is visible.",
                    "verdict": "not_found_in_window",
                    "evidence": "The red car is not visible.",
                    "confidence": 0.8,
                    "option_id": "B",
                    "scope": {"segment_id": "seg_0002", "time_range": [20.0, 30.0]},
                }
            ],
            "produced_anchors": [
                {
                    "anchor_id": "clip_anch_b_neg",
                    "source_kind": "visual_fact",
                    "segment_id": "seg_0002",
                    "start_sec": 20.0,
                    "end_sec": 30.0,
                    "excerpt": "The red car is not visible.",
                }
            ],
        }

    registry.register(verify_window)
    workspace.write_observation(
        tool_name="explore",
        claim="Explore found an option B candidate.",
        confidence=0.8,
        regions=[],
        raw_output={
            "mode": "candidate_discovery",
            "multi_agent_option_id": "B",
            "candidate_windows": [
                {
                    "candidate_key": "obs_0001:cand_0001",
                    "segment_id": "seg_0002",
                    "time_range": [20.0, 30.0],
                    "score": 0.9,
                }
            ],
        },
    )
    mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(option_id="B", claim="Option B is visible."),
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

    assert workspace.memory_entries()[-1].kind == "local_negative"
    assert mutator.findings()[-1].status == "empty"


def test_reasoner_answers_from_option_bound_positive_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    memory = _write_test_memory(workspace, kind="visual_support", supports_option="B")
    for option_id, status, memory_ids in [
        ("A", "empty", ()),
        ("B", "satisfied", (memory.entry_id,)),
    ]:
        sub_goal = mutator.create_sub_goal(
            intent="verify",
            constraint=SubGoalConstraint(option_id=option_id, claim=f"Check option {option_id}."),
            budget=SubGoalBudget(max_explores=1, max_verifies=1),
            success_criteria=SubGoalSuccessCriteria(needs_visual_support=True),
            parent_question="Question?",
            created_by="reasoner",
            created_round=1,
        )
        mutator.transition_sub_goal(sub_goal.sub_goal_id, to_status="in_progress", round_number=2)
        mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status=status,  # type: ignore[arg-type]
            memory_ids=memory_ids,
            coverage=(0.0, 0.0),
            notes_for_planner=f"Option {option_id} check complete.",
            cost={"tool_calls": 1},
            created_round=2,
        )
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert reasoner.step(round_number=3, question="Question?", options={"A": "blue car", "B": "red car"}) is True

    assert reasoner.answer_result is not None
    assert reasoner.answer_result.answer == "B"
    assert reasoner.answer_result.citations == (memory.entry_id,)


def test_reasoner_emits_disambiguation_need_when_multiple_options_have_positive_memory(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    memories = {
        option_id: _write_test_memory(workspace, kind="visual_support", supports_option=option_id)
        for option_id in ("C", "D")
    }
    for option_id in ("A", "B", "C", "D"):
        sub_goal = mutator.create_sub_goal(
            intent="verify",
            constraint=SubGoalConstraint(option_id=option_id, claim=f"Check option {option_id}."),
            budget=SubGoalBudget(max_explores=1, max_verifies=1),
            success_criteria=SubGoalSuccessCriteria(needs_visual_support=True),
            parent_question="Question?",
            created_by="reasoner",
            created_round=1,
        )
        mutator.transition_sub_goal(sub_goal.sub_goal_id, to_status="in_progress", round_number=2)
        memory_ids = (memories[option_id].entry_id,) if option_id in memories else ()
        mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status="satisfied" if memory_ids else "empty",
            memory_ids=memory_ids,
            coverage=(0.0, 0.0),
            notes_for_planner=f"Option {option_id} check complete.",
            cost={"tool_calls": 1},
            created_round=2,
        )
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert reasoner.step(
        round_number=3,
        question="Question?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
    ) is True

    assert reasoner.answer_result is None
    open_goals = [goal for goal in mutator.sub_goals() if goal.status == "open"]
    assert [goal.intent for goal in open_goals] == ["disambiguate", "disambiguate"]
    assert [goal.constraint.option_id for goal in open_goals] == ["C", "D"]


def test_reasoner_answers_supported_option_when_competing_positive_is_refuted(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    visual_c = _write_test_memory(workspace, kind="visual_support", supports_option="C")
    conflict_c = _write_test_memory(workspace, kind="answer_conflict", supports_option="C")
    visual_d = _write_test_memory(workspace, kind="visual_support", supports_option="D")
    for index, (option_id, status, memory_ids) in enumerate(
        [
            ("A", "empty", ()),
            ("B", "empty", ()),
            ("C", "satisfied", (visual_c.entry_id, conflict_c.entry_id)),
            ("D", "satisfied", (visual_d.entry_id,)),
        ],
        start=1,
    ):
        sub_goal = mutator.create_sub_goal(
            intent="verify",
            constraint=SubGoalConstraint(option_id=option_id, claim=f"Check option {option_id}."),
            budget=SubGoalBudget(max_explores=1, max_verifies=1),
            success_criteria=SubGoalSuccessCriteria(needs_visual_support=True),
            parent_question="Question?",
            created_by="reasoner",
            created_round=index,
        )
        mutator.transition_sub_goal(sub_goal.sub_goal_id, to_status="in_progress", round_number=index + 1)
        mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status=status,  # type: ignore[arg-type]
            memory_ids=memory_ids,
            coverage=(0.0, 0.0),
            notes_for_planner=f"Option {option_id} check complete.",
            cost={"tool_calls": 1},
            created_round=index + 1,
        )
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert reasoner.step(
        round_number=6,
        question="Question?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
    ) is True

    assert reasoner.answer_result is not None
    assert reasoner.answer_result.answer == "D"
    assert reasoner.answer_result.citations == (visual_d.entry_id,)


def test_reasoner_ignores_negative_memory_and_schedules_untested_options(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    negative = _write_test_memory(workspace, kind="local_negative", supports_option="")
    sub_goal = mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(option_id="A", claim="Check option A."),
        budget=SubGoalBudget(max_explores=1, max_verifies=1),
        success_criteria=SubGoalSuccessCriteria(needs_visual_support=True),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    mutator.transition_sub_goal(sub_goal.sub_goal_id, to_status="in_progress", round_number=2)
    mutator.report_finding(
        sub_goal_id=sub_goal.sub_goal_id,
        status="satisfied",
        memory_ids=(negative.entry_id,),
        coverage=(0.0, 0.0),
        notes_for_planner="Option A was not found in the local window.",
        cost={"tool_calls": 1},
        created_round=2,
    )
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert reasoner.step(
        round_number=3,
        question="Question?",
        options={"A": "blue car", "B": "red car", "C": "green car", "D": "yellow car"},
    ) is True

    assert reasoner.answer_result is None
    open_options = [goal.constraint.option_id for goal in mutator.sub_goals() if goal.status == "open"]
    assert open_options == ["B", "C", "D"]


def test_reasoner_answers_by_elimination_when_all_other_options_are_contradicted(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    conflict_ids: list[str] = []
    for option_id in ("A", "B", "C"):
        memory = _write_test_memory(workspace, kind="answer_conflict", supports_option=option_id)
        conflict_ids.append(memory.entry_id)
    for option_id in ("A", "B", "C", "D"):
        sub_goal = mutator.create_sub_goal(
            intent="verify",
            constraint=SubGoalConstraint(option_id=option_id, claim=f"Check option {option_id}."),
            budget=SubGoalBudget(max_explores=1, max_verifies=1),
            success_criteria=SubGoalSuccessCriteria(needs_visual_support=True),
            parent_question="Question?",
            created_by="reasoner",
            created_round=1,
        )
        mutator.transition_sub_goal(sub_goal.sub_goal_id, to_status="in_progress", round_number=2)
        memory_ids = (conflict_ids[ord(option_id) - ord("A")],) if option_id in {"A", "B", "C"} else ()
        mutator.report_finding(
            sub_goal_id=sub_goal.sub_goal_id,
            status="empty",
            memory_ids=memory_ids,
            coverage=(0.0, 0.0),
            notes_for_planner=f"Option {option_id} check complete.",
            cost={"tool_calls": 1},
            created_round=2,
        )
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert reasoner.step(
        round_number=3,
        question="Question?",
        options={"A": "first", "B": "second", "C": "third", "D": "fourth"},
    ) is True

    assert reasoner.answer_result is not None
    assert reasoner.answer_result.answer == "D"
    assert reasoner.answer_result.citations == tuple(conflict_ids)
    assert reasoner.answer_result.metadata is not None
    assert reasoner.answer_result.metadata["reason"] == "elimination"


def test_reasoner_sub_goal_claim_includes_question_context(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert reasoner.step(
        round_number=1,
        question="Question: In what order are the four sculptures presented?",
        options={"A": "Persephone, Apollo, David, Aeneas", "B": "Aeneas, David, Persephone, Apollo"},
    ) is True

    claims = [goal.constraint.claim for goal in mutator.sub_goals()]
    assert "In what order" in claims[0]
    assert "Option A" in claims[0]
    assert "Persephone, Apollo, David, Aeneas" in claims[0]


def test_reasoner_sub_goal_claim_drops_full_options_block(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    reasoner.step(
        round_number=1,
        question="Question: In what order are the sculptures presented? Options: A. old order B. correct order",
        options={"A": "old order", "B": "correct order"},
    )

    claim = mutator.sub_goals()[0].constraint.claim
    assert "Options:" not in claim
    assert "Question: In what order are the sculptures presented?" in claim
    assert "Option A: old order" in claim


def test_reasoner_routes_biography_question_to_asr_first(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    reasoner = ReasonerAgent(
        backend=object(),
        mutator=mutator,
        workspace=workspace,
        video_map=None,
        log_root=tmp_path / "logs",
    )

    assert reasoner.step(
        round_number=1,
        question="How was his life journey according to the video?",
        options={"A": "upper class only", "B": "humble background, upper class, seclusion"},
    ) is True

    first_goal = mutator.sub_goals()[0]
    assert first_goal.constraint.modality_hint == ("asr", "index", "visual")
    events = [json.loads(line) for line in (workspace.root / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    route_events = [event for event in events if event["type"] == "modality_route_chosen"]
    assert route_events
    assert route_events[0]["payload"]["route"] == "asr_primary"


def _write_test_memory(workspace: EvidenceWorkspace, *, kind: str, supports_option: str):
    workspace.write_produced_anchors(
        [
            SourceAnchor(
                anchor_id=f"anch_{kind}_{supports_option or 'none'}",
                observation_id="obs_0001",
                source_kind="visual_fact",
                segment_id="seg_0001",
                field_path="verification_results.0",
                excerpt="A visible clue appears in the inspected window.",
            )
        ]
    )
    return workspace.write_memory(
        kind=kind,  # type: ignore[arg-type]
        claim="A visible clue appears in the inspected window.",
        anchors=[{"anchor_id": f"anch_{kind}_{supports_option or 'none'}"}],
        supports_option=supports_option,
        confidence="high",
    )
