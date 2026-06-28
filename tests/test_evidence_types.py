from __future__ import annotations

from visual_coding_agent_harness.agents.multi import (
    SubGoal,
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalSuccessCriteria,
)
from visual_coding_agent_harness.evidence import EvidenceItem, EvidenceNeed
from visual_coding_agent_harness.memory import MemoryEntry, SourceAnchor


def test_evidence_need_adapts_sub_goal_and_round_trips_kwargs() -> None:
    sub_goal = SubGoal(
        sub_goal_id="sg_0007",
        intent="disprove",
        constraint=SubGoalConstraint(
            option_id="B",
            claim="Check whether option B is contradicted locally.",
            segment_id="seg_0002",
            time_range=(10.0, 20.0),
            modality_hint=("asr", "visual"),
        ),
        budget=SubGoalBudget(max_explores=1, max_verifies=2, max_frames=32),
        success_criteria=SubGoalSuccessCriteria(needs_visual_support=False, needs_option_relation=True),
        parent_question="Question?",
        created_by="reasoner",
        created_round=3,
        status="open",
        rationale="Need a refutation check.",
    )

    need = EvidenceNeed.from_sub_goal(sub_goal)

    assert need.need_id == "sg_0007"
    assert need.option_id == "B"
    assert need.polarity == "seek_refutation"
    assert need.segment_id == "seg_0002"
    assert need.time_range == (10.0, 20.0)
    assert need.modality_hint == ("asr", "visual")
    assert need.budget_max_explores == 1
    assert need.budget_max_verifies == 2

    kwargs = need.to_sub_goal_kwargs(created_by="evidence_reasoner", created_round=4)
    assert kwargs["intent"] == "disprove"
    assert kwargs["constraint"].option_id == "B"
    assert kwargs["constraint"].time_range == (10.0, 20.0)
    assert kwargs["budget"].max_explores == 1
    assert kwargs["budget"].max_verifies == 2
    assert kwargs["parent_question"] == "Question?"


def test_evidence_item_derives_polarity_and_primary_anchor_fields() -> None:
    entry = MemoryEntry(
        entry_id="mem_0003",
        round_number=2,
        kind="answer_conflict",
        claim="Option C is contradicted by the observed order.",
        anchors=(
            SourceAnchor(
                anchor_id="anch_1",
                observation_id="obs_0002",
                source_kind="visual_fact",
                segment_id="seg_0004",
                start_sec=42.0,
                end_sec=55.0,
                modality="visual",
                excerpt="Observed order differs from option C.",
            ),
        ),
        supports_option="C",
        confidence="high",
        tags=("ordered_projection",),
        metadata={"verdict": "contradicted", "source_need_id": "sg_0003"},
    )

    item = EvidenceItem.from_memory_entry(entry)

    assert item.evidence_id == "mem_0003"
    assert item.option_id == "C"
    assert item.polarity == "refutes"
    assert item.segment_id == "seg_0004"
    assert item.time_range == (42.0, 55.0)
    assert item.modality == "visual"
    assert item.source_need_id == "sg_0003"
    assert item.tags == ("ordered_projection",)


def test_evidence_item_maps_local_negative_to_absent() -> None:
    entry = MemoryEntry(
        entry_id="mem_0004",
        round_number=2,
        kind="local_negative",
        claim="Option D was not found in this window.",
        anchors=(
            SourceAnchor(
                anchor_id="anch_2",
                observation_id="obs_0003",
                source_kind="audio_fact",
                segment_id="seg_0005",
                start_sec=80.0,
                end_sec=95.0,
                modality="asr",
            ),
        ),
        supports_option="D",
        metadata={"verdict": "not_found_in_window"},
    )

    item = EvidenceItem.from_memory_entry(entry)

    assert item.polarity == "absent"
    assert item.modality == "asr"
