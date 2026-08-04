from __future__ import annotations

from vcah.workspace import WorkingDocument


def _observation_row(*, role: str = "unclassified") -> dict[str, object]:
    return {
        "attempt_id": "attempt_visual",
        "interpretation_id": "interpretation_visual",
        "requested_range": [5.0, 6.0],
        "inspected_ranges": [[5.0, 6.0]],
        "modality": "visual",
        "evidence_role": role,
        "sampling_config": {
            "mode": "window",
            "evidence_kind": "generic",
            "requires_refinement": role == "candidate",
        },
        "interpretation_items": [
            {
                "item_id": "item_visible",
                "time_anchor": [5.0, 5.0],
                "text": "A cup is visible.",
                "item_kind": "observation",
            }
        ],
    }


def test_answer_validation_exposes_independent_closure_dimensions() -> None:
    row = _observation_row(role="candidate")
    document = WorkingDocument.with_question_premise("What is visible?")
    assert document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_cup",
                "text": "A cup is visible.",
                "source": "observation",
                "cites": ["attempt_visual"],
                "time_anchor": [5.0, 5.0],
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_anchor",
                "observable_goal": "Find the anchor.",
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_target",
                "observable_goal": "Identify the target.",
                "depends_on": ["req_anchor"],
            },
            {
                "op": "add_temporal_scope",
                "scope_id": "scope_next",
                "relation": "after",
                "selection": "next",
                "anchor_requirement_id": "req_anchor",
                "target_requirement_id": "req_target",
            },
        ),
        observation_ids=("attempt_visual",),
        observation_rows=(row,),
        require_item_provenance=False,
    ).accepted

    validation = document.validate_answer(
        ("claim_cup",),
        observation_ids=("attempt_visual",),
        supporting_observation_ids=(),
        require_obligation_coverage=True,
        observation_rows=(row,),
        require_item_provenance=True,
        temporal_scope_resolutions={},
        unconsumed_observation_ids=("attempt_visual",),
        require_evidence_kind_requirements=True,
    )

    assert not validation.passed
    assert validation.reference_integrity_ok
    assert not validation.material_support_ok
    assert not validation.provenance_binding_ok
    assert not validation.temporal_consistency_ok
    assert validation.occurrence_binding_ok
    assert not validation.obligation_coverage_ok
    assert not validation.observation_consumption_ok
    assert validation.evidence_kind_requirements_ok


def test_fully_bound_generic_answer_passes_all_closure_dimensions() -> None:
    row = _observation_row()
    document = WorkingDocument.with_question_premise("What is visible?")
    assert document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "claim_cup",
                "text": "A cup is visible.",
                "source": "observation",
                "cites": ["attempt_visual"],
                "interpretation_id": "interpretation_visual",
                "interpretation_item_id": "item_visible",
                "time_anchor": [5.0, 5.0],
                "confidence": "high",
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_target",
                "observable_goal": "Identify the visible object.",
            },
            {
                "op": "set_obligation_status",
                "requirement_id": "req_target",
                "status": "satisfied",
                "supporting_claim_ids": ["claim_cup"],
                "supporting_attempt_ids": ["attempt_visual"],
            },
        ),
        observation_ids=("attempt_visual",),
        observation_rows=(row,),
        require_item_provenance=True,
    ).accepted

    validation = document.validate_answer(
        ("claim_cup",),
        observation_ids=("attempt_visual",),
        supporting_observation_ids=("attempt_visual",),
        require_obligation_coverage=True,
        observation_rows=(row,),
        require_item_provenance=True,
        temporal_scope_resolutions={},
        require_evidence_kind_requirements=True,
    )

    assert validation.passed
    assert validation.reference_integrity_ok
    assert validation.material_support_ok
    assert validation.provenance_binding_ok
    assert validation.temporal_consistency_ok
    assert validation.occurrence_binding_ok
    assert validation.obligation_coverage_ok
    assert validation.observation_consumption_ok
    assert validation.evidence_kind_requirements_ok
