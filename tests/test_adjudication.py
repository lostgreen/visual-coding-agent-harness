from __future__ import annotations

from vcah.adjudication import (
    RevisionContext,
    build_all_option_audit_record,
    evaluate_hard_override_guard,
    evaluate_soft_audit_correction_guard,
    final_adjudicate,
)


OPTIONS = {"D": "raw answer", "H": "stale recommendation"}


def _context() -> RevisionContext:
    return RevisionContext.from_inputs(
        {"qualified_events": [], "revision_fixture": True},
        ({"evidence_id": "ev_1", "summary": "direct witness"},),
        {"quantifier": "none", "required_scope": "window"},
    )


def _fresh_table(context: RevisionContext, *, grounding: str = "sufficient") -> dict[str, object]:
    return {
        **context.to_dict(),
        "option_verdict_revision": context.canonical_snapshot_revision,
        "option_verdicts": {
            "D": {"predicate_verdict": "contradicted", "grounding_eligibility": "sufficient"},
            "H": {"predicate_verdict": "supported", "grounding_eligibility": grounding},
        },
    }


def _fresh_audit(context: RevisionContext) -> dict[str, object]:
    return build_all_option_audit_record(
        options=OPTIONS,
        supplied_verdicts={
            "D": {"predicate_verdict": "contradicted", "grounding_eligibility": "sufficient"},
            "H": {"predicate_verdict": "supported", "grounding_eligibility": "sufficient"},
        },
        audit_status="complete",
        audit_reason="All options were checked against the final snapshot.",
        revision_context=context,
        required=True,
    )


def test_stale_recommendation_cannot_mutate_raw_reasoner_answer() -> None:
    context = _context()
    stale_context = RevisionContext("stale", "old-evidence", "old-contract")
    result = final_adjudicate(
        options=OPTIONS,
        raw_reasoner_answer="D. raw answer",
        raw_citations=("ev_1",),
        raw_gate={"passed": False, "reason": "completion_not_ready"},
        completion_status={"completion_ready": False},
        qualification_result={"status": "incomplete", "requirement_graph": {"unknown": 1}},
        option_verdict_table=_fresh_table(stale_context),
        audit_record=_fresh_audit(stale_context),
        revision_context=context,
    )

    assert result.answer == "D. raw answer"
    assert result.answer_mode == "forced_choice"
    assert result.selection_source == "raw_reasoner"
    assert result.answer_mutation_events == ()
    assert "verdict_table_stale" in result.guard.blockers
    assert "audit_stale" in result.guard.blockers


def test_unknown_grounding_blocks_hard_override_even_when_predicate_is_unique() -> None:
    context = _context()
    guard = evaluate_hard_override_guard(
        {"completion_ready": True},
        {"status": "complete", "requirement_graph": {}},
        _fresh_table(context, grounding="unknown"),
        _fresh_audit(context),
        context,
    )

    assert guard.allowed is False
    assert "selected_option_grounding_unknown" in guard.blockers


def test_unknown_competing_option_blocks_hard_override() -> None:
    context = _context()
    table = _fresh_table(context)
    table["option_verdicts"]["D"]["predicate_verdict"] = "unknown"

    guard = evaluate_hard_override_guard(
        {"completion_ready": True},
        {"status": "complete", "requirement_graph": {}},
        table,
        _fresh_audit(context),
        context,
    )

    assert guard.allowed is False
    assert "option_predicate_unknown" in guard.blockers


def test_heuristic_only_provenance_blocks_hard_override() -> None:
    context = _context()
    table = _fresh_table(context)
    table["provenance_required"] = True
    table["option_verdicts"]["H"]["provenance"] = [
        {
            "kind": "heuristic",
            "source_fact_ids": ["event_1"],
            "source_evidence_ids": ["ev_1"],
            "derivation": "model_assertion",
            "producer": "model",
        }
    ]
    guard = evaluate_hard_override_guard(
        {"completion_ready": True},
        {"status": "complete", "requirement_graph": {}},
        table,
        _fresh_audit(context),
        context,
    )

    assert guard.allowed is False
    assert "provenance_insufficient" in guard.blockers


def test_fresh_complete_guard_allows_single_final_adjudicator_mutation() -> None:
    context = _context()
    result = final_adjudicate(
        options=OPTIONS,
        raw_reasoner_answer="D. raw answer",
        raw_citations=("ev_1",),
        raw_gate={"passed": False, "reason": "raw_predicate_contradicted"},
        completion_status={"completion_ready": True},
        qualification_result={"status": "complete", "requirement_graph": {}},
        option_verdict_table=_fresh_table(context),
        audit_record=_fresh_audit(context),
        revision_context=context,
    )

    assert result.answer == "H. stale recommendation"
    assert result.selection_source == "final_adjudicator_hard_override"
    assert len(result.answer_mutation_events) == 1
    assert result.answer_selection_event["answer_mutation_event_count"] == 1


def test_required_invalid_audit_cannot_preserve_raw_answer_as_grounded() -> None:
    context = _context()
    invalid_audit = build_all_option_audit_record(
        options=OPTIONS,
        supplied_verdicts={},
        audit_status="invalid",
        audit_reason="Audit parse failed.",
        revision_context=context,
        required=True,
    )
    result = final_adjudicate(
        options=OPTIONS,
        raw_reasoner_answer="D. raw answer",
        raw_citations=("ev_1",),
        raw_gate={"passed": True, "reason": "verified_window_evidence"},
        completion_status={"completion_ready": True},
        qualification_result={"status": "complete", "requirement_graph": {}},
        option_verdict_table=_fresh_table(context),
        audit_record=invalid_audit,
        revision_context=context,
        audit_required=True,
    )

    assert result.answer == "D. raw answer"
    assert result.verified is False
    assert result.answer_mode == "forced_choice"
    assert result.verification_reason == "answer_audit_incomplete"


def test_supplied_audit_without_source_revision_fails_closed() -> None:
    context = _context()
    audit = build_all_option_audit_record(
        options=OPTIONS,
        supplied_verdicts={
            "D": {"predicate_verdict": "contradicted"},
            "H": {"predicate_verdict": "supported"},
        },
        audit_status="complete",
        audit_reason="Source revision was omitted.",
        revision_context=context,
        required=True,
        source_revision_context={},
    )

    assert audit["audit_status"] == "invalid"
    assert "audit_source_revision_missing" in audit["invalidity_flags"]
    assert audit["audit_snapshot_revision"] == ""


def test_partial_supplied_audit_is_invalid_even_when_audit_is_optional() -> None:
    context = _context()
    audit = build_all_option_audit_record(
        options=OPTIONS,
        supplied_verdicts={"H": {"predicate_verdict": "supported"}},
        audit_status="complete",
        audit_reason="Only one option was checked.",
        revision_context=context,
        required=False,
    )

    assert audit["audit_status"] == "invalid"
    assert "all_option_verdicts_incomplete" in audit["invalidity_flags"]


def test_soft_audit_correction_is_forced_only_and_requires_admissible_predicate() -> None:
    context = _context()
    table = _fresh_table(context)
    table["option_verdicts"]["H"].update({
        "evidence_ids": ["ev_1"],
        "provenance": [{
            "kind": "direct_witness",
            "source_evidence_ids": ["ev_1"],
            "witness_ranges": [[1.0, 2.0]],
            "producer": "observation",
        }],
    })
    result = final_adjudicate(
        options=OPTIONS,
        raw_reasoner_answer="D. raw answer",
        raw_citations=("ev_1",),
        raw_gate={"passed": False, "reason": "completion_not_ready"},
        completion_status={"completion_ready": False},
        qualification_result={"status": "incomplete", "requirement_graph": {}},
        option_verdict_table=table,
        audit_record=_fresh_audit(context),
        revision_context=context,
    )

    assert result.answer == "H. stale recommendation"
    assert result.answer_mode == "forced_choice"
    assert result.verified is False
    assert result.selection_source == "soft_audit_correction"
    assert result.answer_mutation_events[0]["source"] == "soft_audit_correction"


def test_soft_audit_correction_fails_closed_for_heuristic_or_unresolved_episode() -> None:
    context = _context()
    table = _fresh_table(context)
    table["option_verdicts"]["H"].update({
        "evidence_ids": ["ev_1"],
        "provenance": [{
            "kind": "heuristic",
            "source_evidence_ids": ["ev_1"],
            "derivation": "model_assertion",
            "producer": "model",
        }],
    })
    guard = evaluate_soft_audit_correction_guard(
        options=OPTIONS,
        raw_reasoner_answer="D. raw answer",
        completion_status={
            "completion_ready": False,
            "temporal_max_selection": {"status": "incomplete"},
            "event_participant_link_ready": False,
        },
        qualification_result={"status": "incomplete", "requirement_graph": {}},
        option_verdict_table=table,
        audit_record=_fresh_audit(context),
        revision_context=context,
    )

    assert guard.allowed is False
    assert "alternative_provenance_insufficient" in guard.blockers
    assert "episode_binding_incomplete" in guard.blockers
