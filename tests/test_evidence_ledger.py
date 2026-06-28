from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.agents.multi import (
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalSuccessCriteria,
    WorkspaceMutator,
)
from visual_coding_agent_harness.evidence import EvidenceLedger
from visual_coding_agent_harness.memory import SourceAnchor
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_evidence_ledger_groups_items_by_option_and_polarity(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    _write_memory(
        workspace,
        kind="visual_support",
        supports_option="B",
        anchor_id="anch_b",
        segment_id="seg_0001",
        start_sec=10.0,
        end_sec=20.0,
        metadata={"verdict": "supported", "source_need_id": "sg_0001"},
    )
    _write_memory(
        workspace,
        kind="answer_conflict",
        supports_option="C",
        anchor_id="anch_c",
        segment_id="seg_0002",
        start_sec=30.0,
        end_sec=45.0,
        metadata={"verdict": "contradicted", "source_need_id": "sg_0002"},
    )

    ledger = EvidenceLedger(workspace=workspace, mutator=mutator)

    assert [item.evidence_id for item in ledger.supports_by_option()["B"]] == ["mem_0001"]
    assert [item.evidence_id for item in ledger.refutes_by_option()["C"]] == ["mem_0002"]
    assert ledger.items_for_option("B")[0].polarity == "supports"


def test_evidence_ledger_derives_needs_and_open_needs(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    open_goal = mutator.create_sub_goal(
        intent="verify",
        constraint=SubGoalConstraint(option_id="A", claim="Check A."),
        budget=SubGoalBudget(max_explores=1, max_verifies=1),
        success_criteria=SubGoalSuccessCriteria(),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    mutator.create_sub_goal(
        intent="disambiguate",
        constraint=SubGoalConstraint(option_id="B", claim="Disambiguate B.", segment_id="seg_0003"),
        budget=SubGoalBudget(max_explores=2, max_verifies=2),
        success_criteria=SubGoalSuccessCriteria(needs_option_relation=True),
        parent_question="Question?",
        created_by="reasoner",
        created_round=1,
    )
    mutator.transition_sub_goal(open_goal.sub_goal_id, to_status="in_progress", round_number=2)

    ledger = EvidenceLedger(workspace=workspace, mutator=mutator)

    assert [need.need_id for need in ledger.needs()] == ["sg_0001", "sg_0002"]
    assert [need.need_id for need in ledger.open_needs()] == ["sg_0002"]
    assert ledger.needs()[1].polarity == "disambiguate"


def test_evidence_ledger_verified_windows_for_option_and_coverage(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace(tmp_path / "workspace")
    mutator = WorkspaceMutator(workspace)
    _write_memory(
        workspace,
        kind="visual_support",
        supports_option="D",
        anchor_id="anch_d_1",
        segment_id="seg_0001",
        start_sec=0.0,
        end_sec=10.0,
        metadata={"verdict": "supported"},
    )
    _write_memory(
        workspace,
        kind="local_negative",
        supports_option="D",
        anchor_id="anch_d_2",
        segment_id="seg_0001",
        start_sec=10.0,
        end_sec=25.0,
        metadata={"verdict": "not_found_in_window"},
    )
    _write_memory(
        workspace,
        kind="retrieval_candidate",
        supports_option="D",
        anchor_id="anch_nav",
        segment_id="seg_0001",
        start_sec=50.0,
        end_sec=60.0,
        metadata={},
    )

    ledger = EvidenceLedger(workspace=workspace, mutator=mutator)

    assert ledger.verified_windows_for_option("D") == {("seg_0001", 0.0, 10.0), ("seg_0001", 10.0, 25.0)}
    assert ledger.coverage_by_segment()["seg_0001"] == 25.0


def _write_memory(
    workspace: EvidenceWorkspace,
    *,
    kind: str,
    supports_option: str,
    anchor_id: str,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    metadata: dict[str, object],
):
    workspace.write_produced_anchors(
        [
            SourceAnchor(
                anchor_id=anchor_id,
                observation_id=f"obs_{anchor_id}",
                source_kind="visual_fact",
                segment_id=segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                field_path="verification_results.0",
                excerpt=f"{kind} evidence.",
                modality="visual",
            )
        ]
    )
    return workspace.write_memory(
        kind=kind,
        claim=f"{kind} evidence.",
        anchors=[{"anchor_id": anchor_id}],
        supports_option=supports_option,
        confidence="high",
        metadata=metadata,
    )
