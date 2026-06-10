from __future__ import annotations

from types import SimpleNamespace

from visual_coding_agent_harness.agents.contracts import (
    OptionEvaluation,
)
from visual_coding_agent_harness.agents.final_gate import evaluate_final_candidate
from visual_coding_agent_harness.contracts import (
    ClaimRelation,
    OptionSpec,
    TargetRegistry,
    TargetSpec,
)


def _target(target_id: str, text: str, *, subject: str = "Subject X") -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        canonical_text=text,
        aliases=(subject,),
        subject=subject,
        source="unit_test",
    )


def _evidence(
    evidence_id: str,
    target_ref: str,
    *,
    modality: str = "visual",
    status: str = "supported",
    start: float | None = 1.0,
    end: float | None = 2.0,
    option_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        evidence_id=evidence_id,
        target_ref=target_ref,
        relation_ref=None,
        option_id=option_id,
        modality=modality,
        source=modality,
        timestamp_start=start,
        timestamp_end=end,
        support_status=status,
        confidence=0.9,
        rationale="unit evidence",
    )


def _relation(
    relation_ref: str,
    ordered_target_refs: tuple[str, ...],
    *,
    evidence_ids: tuple[str, ...] = ("E1",),
    modality: str = "visual",
    status: str = "supported",
    timestamp_order: tuple[float, ...] = (1.0, 2.0),
) -> SimpleNamespace:
    return SimpleNamespace(
        relation_ref=relation_ref,
        ordered_target_refs=ordered_target_refs,
        evidence_ids=evidence_ids,
        support_status=status,
        timestamp_order=timestamp_order,
        modality=modality,
    )


def _sequence_registry() -> TargetRegistry:
    return TargetRegistry.from_specs(
        targets=[
            _target("T1", "first event"),
            _target("T2", "second event"),
            _target("T3", "third event"),
        ],
        relations=[
            ClaimRelation("R1", "before", "T1", "T2"),
            ClaimRelation("R2", "before", "T2", "T3"),
            ClaimRelation("R3", "before", "T3", "T2"),
        ],
        options=[
            OptionSpec("B", target_sequence=("T1", "T2", "T3"), required_relations=("R1", "R2"), option_kind="sequence"),
            OptionSpec("C", target_sequence=("T1", "T3", "T2"), required_relations=("R1", "R3"), option_kind="sequence"),
        ],
    )


def test_flat_target_support_cannot_satisfy_sequence_option() -> None:
    registry = _sequence_registry()

    decision = evaluate_final_candidate(
        selected_option="B",
        registry=registry,
        evidence_bindings=[
            _evidence("E1", "T1"),
            _evidence("E2", "T2"),
            _evidence("E3", "T3"),
        ],
        relation_bindings=[],
        skill_name="visual_timeline_qa",
    )

    assert not decision.accepted
    assert decision.reason_code == "missing_relation_binding"
    assert decision.missing_relation_refs == ("R1", "R2")
    assert decision.supporting_evidence_ids == ("E1", "E2", "E3")


def test_same_targets_different_relation_refs_are_distinguishable() -> None:
    registry = _sequence_registry()

    option_b = evaluate_final_candidate(
        selected_option="B",
        registry=registry,
        evidence_bindings=[
            _evidence("E1", "T1"),
            _evidence("E2", "T2"),
            _evidence("E3", "T3"),
        ],
        relation_bindings=[
            _relation("R1", ("T1", "T2"), evidence_ids=("E1", "E2")),
            _relation("R2", ("T2", "T3"), evidence_ids=("E2", "E3")),
        ],
        skill_name="visual_timeline_qa",
    )
    option_c = evaluate_final_candidate(
        selected_option="C",
        registry=registry,
        evidence_bindings=[
            _evidence("E1", "T1"),
            _evidence("E2", "T2"),
            _evidence("E3", "T3"),
        ],
        relation_bindings=[
            _relation("R1", ("T1", "T2"), evidence_ids=("E1", "E2")),
            _relation("R2", ("T2", "T3"), evidence_ids=("E2", "E3")),
        ],
        skill_name="visual_timeline_qa",
    )

    assert option_b.accepted
    assert not option_c.accepted
    assert option_c.reason_code == "missing_relation_binding"
    assert option_c.missing_relation_refs == ("R3",)


def test_narrated_timeline_accepts_asr_relation_support() -> None:
    registry = _sequence_registry()

    decision = evaluate_final_candidate(
        selected_option="B",
        registry=registry,
        evidence_bindings=[
            _evidence("E1", "T1", modality="asr"),
            _evidence("E2", "T2", modality="transcript"),
            _evidence("E3", "T3", modality="asr"),
        ],
        relation_bindings=[
            _relation("R1", ("T1", "T2"), evidence_ids=("E1", "E2"), modality="asr"),
            _relation("R2", ("T2", "T3"), evidence_ids=("E2", "E3"), modality="transcript"),
        ],
        skill_name="narration_timeline_qa",
    )

    assert decision.accepted
    assert decision.reason_code is None
    assert decision.supporting_evidence_ids == ("E1", "E2", "E3")


def test_visual_timeline_rejects_asr_only_support() -> None:
    registry = _sequence_registry()

    decision = evaluate_final_candidate(
        selected_option="B",
        registry=registry,
        evidence_bindings=[
            _evidence("E1", "T1", modality="asr"),
            _evidence("E2", "T2", modality="transcript"),
            _evidence("E3", "T3", modality="asr"),
        ],
        relation_bindings=[
            _relation("R1", ("T1", "T2"), evidence_ids=("E1", "E2"), modality="asr"),
            _relation("R2", ("T2", "T3"), evidence_ids=("E2", "E3"), modality="transcript"),
        ],
        skill_name="visual_timeline_qa",
    )

    assert not decision.accepted
    assert decision.reason_code == "unsupported_modality"
    assert decision.missing_target_refs == ("T1", "T2", "T3")


def test_main_idea_rejects_insufficient_breadth_with_closed_reason() -> None:
    registry = TargetRegistry.from_specs(
        targets=[
            _target("T1", "subject opening"),
            _target("T2", "subject resolution"),
        ],
        options=[
            OptionSpec("D", target_sequence=("T1", "T2"), option_kind="topic_arc"),
        ],
    )

    decision = evaluate_final_candidate(
        selected_option="D",
        registry=registry,
        evidence_bindings=[_evidence("E1", "T1", modality="visual", start=4.0, end=5.0)],
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=[
            OptionEvaluation(
                option_id="D",
                binding_status="partial",
                rejection_reason="insufficient_breadth",
                coverage_breadth=1,
                supporting_evidence_ids=("E1",),
            )
        ],
        central_subjects=("Subject X",),
    )

    assert not decision.accepted
    assert decision.reason_code == "insufficient_breadth"
    assert decision.missing_target_refs == ("T2",)
