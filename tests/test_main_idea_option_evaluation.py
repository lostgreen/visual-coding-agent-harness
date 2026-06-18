from __future__ import annotations

from types import SimpleNamespace

from visual_coding_agent_harness.agents.contracts import OptionEvaluation
from visual_coding_agent_harness.agents.final_gate_legacy import evaluate_final_candidate
from visual_coding_agent_harness.contracts import OptionSpec, TargetRegistry, TargetSpec


def _target(target_id: str, text: str, *, subject: str = "Central Topic") -> TargetSpec:
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
    source: str,
    modality: str = "visual",
    option_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        evidence_id=evidence_id,
        target_ref=target_ref,
        relation_ref=None,
        option_id=option_id,
        modality=modality,
        source=source,
        timestamp_start=None,
        timestamp_end=None,
        support_status="supported",
        confidence=0.9,
        rationale="unit evidence",
    )


def _evaluation(
    option_id: str,
    *,
    status: str = "supported",
    rejection_reason: str | None = None,
    breadth: int = 2,
    evidence_ids: tuple[str, ...] = (),
) -> OptionEvaluation:
    return OptionEvaluation(
        option_id=option_id,
        binding_status=status,
        rejection_reason=rejection_reason,
        coverage_breadth=breadth,
        supporting_evidence_ids=evidence_ids,
    )


def _registry() -> TargetRegistry:
    return TargetRegistry.from_specs(
        targets=[
            _target("T1", "topic opening"),
            _target("T2", "topic development"),
            _target("T3", "side detail"),
            _target("T4", "side consequence"),
            _target("T5", "alternate focus"),
            _target("T6", "alternate ending"),
            _target("T7", "full topic arc opening"),
            _target("T8", "full topic arc ending"),
        ],
        options=[
            OptionSpec("A", target_sequence=("T1", "T2"), option_kind="topic_focus"),
            OptionSpec("B", target_sequence=("T3", "T4"), option_kind="topic_focus"),
            OptionSpec("C", target_sequence=("T5", "T6"), option_kind="topic_focus"),
            OptionSpec("D", target_sequence=("T7", "T8"), option_kind="topic_arc"),
        ],
    )


def _all_evaluations() -> list[OptionEvaluation]:
    return [
        _evaluation("A", status="partial", rejection_reason="insufficient_breadth", breadth=1),
        _evaluation("B", status="partial", rejection_reason="narrower", breadth=2),
        _evaluation("C", status="unsupported", rejection_reason="wrong_subject", breadth=0),
        _evaluation("D", status="supported", breadth=2, evidence_ids=("E1", "E2")),
    ]


def test_main_idea_requires_all_option_evaluations_before_final() -> None:
    registry = _registry()

    decision = evaluate_final_candidate(
        selected_option="D",
        registry=registry,
        evidence_bindings=[
            _evidence("E1", "T7", source="seg_0001"),
            _evidence("E2", "T8", source="seg_0003"),
        ],
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=[_evaluation("D", breadth=2, evidence_ids=("E1", "E2"))],
        central_subjects=("Central Topic",),
    )

    assert not decision.accepted
    assert decision.reason_code == "no_per_option_coverage"
    assert decision.actionable_next_program == (
        {
            "tool": "target_coverage",
            "args": {
                "target_refs": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
                "group_by_option": True,
                "top_k": 3,
            },
        },
    )


def test_main_idea_rejects_single_option_chase() -> None:
    decision = evaluate_final_candidate(
        selected_option="B",
        registry=_registry(),
        evidence_bindings=[
            _evidence("E1", "T3", source="seg_0001"),
            _evidence("E2", "T4", source="seg_0002"),
        ],
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=[_evaluation("B", breadth=2, evidence_ids=("E1", "E2"))],
        central_subjects=("Central Topic",),
    )

    assert not decision.accepted
    assert decision.reason_code == "no_per_option_coverage"
    assert decision.actionable_next_program[0]["args"]["group_by_option"] is True


def test_main_idea_prefers_full_arc_over_narrow_cause() -> None:
    registry = _registry()
    evidence = [
        _evidence("E1", "T7", source="seg_0001"),
        _evidence("E2", "T8", source="seg_0003"),
        _evidence("E3", "T3", source="seg_0001"),
        _evidence("E4", "T4", source="seg_0002"),
    ]

    narrow = evaluate_final_candidate(
        selected_option="B",
        registry=registry,
        evidence_bindings=evidence,
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=_all_evaluations(),
        central_subjects=("Central Topic",),
    )
    full_arc = evaluate_final_candidate(
        selected_option="D",
        registry=registry,
        evidence_bindings=evidence,
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=_all_evaluations(),
        central_subjects=("Central Topic",),
    )

    assert not narrow.accepted
    assert narrow.reason_code == "insufficient_breadth"
    assert full_arc.accepted


def test_global_gist_is_context_hint_not_answer_grade_support() -> None:
    decision = evaluate_final_candidate(
        selected_option="D",
        registry=_registry(),
        evidence_bindings=[
            _evidence("E1", "T7", source="global_gist"),
            _evidence("E2", "T8", source="global_gist"),
        ],
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=_all_evaluations(),
        central_subjects=("Central Topic",),
    )

    assert not decision.accepted
    assert decision.reason_code == "insufficient_breadth"


def test_selected_option_requires_distinct_segment_breadth_and_subject_overlap() -> None:
    same_segment = evaluate_final_candidate(
        selected_option="D",
        registry=_registry(),
        evidence_bindings=[
            _evidence("E1", "T7", source="seg_0001"),
            _evidence("E2", "T8", source="seg_0001"),
        ],
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=_all_evaluations(),
        central_subjects=("Central Topic",),
    )
    wrong_subject = evaluate_final_candidate(
        selected_option="D",
        registry=_registry(),
        evidence_bindings=[
            _evidence("E1", "T7", source="seg_0001"),
            _evidence("E2", "T8", source="seg_0003"),
        ],
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=_all_evaluations(),
        central_subjects=("Different Topic",),
    )

    assert not same_segment.accepted
    assert same_segment.reason_code == "insufficient_breadth"
    assert not wrong_subject.accepted
    assert wrong_subject.reason_code == "wrong_subject"


def test_main_idea_recovery_injects_group_by_option_coverage() -> None:
    decision = evaluate_final_candidate(
        selected_option="D",
        registry=_registry(),
        evidence_bindings=[],
        relation_bindings=[],
        skill_name="main_idea",
        option_evaluations=[],
        central_subjects=("Central Topic",),
    )

    recovery = decision.actionable_next_program[0]
    assert decision.reason_code == "no_per_option_coverage"
    assert recovery["tool"] == "target_coverage"
    assert recovery["args"]["group_by_option"] is True
    assert recovery["args"]["top_k"] == 3
