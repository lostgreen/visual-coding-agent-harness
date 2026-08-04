from __future__ import annotations

from vcah.runtime_metrics import agent_run_metrics
from vcah.workspace import WorkingDocument


def test_multi_hop_obligations_require_dependency_lineage() -> None:
    document = WorkingDocument.with_question_premise("What happened after the anchor?")
    attempt_id = "attempt_visual"

    accepted = document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "c_anchor",
                "text": "The anchor is directly visible.",
                "source": "observation",
                "cites": [attempt_id],
            },
            {
                "op": "add_claim",
                "claim_id": "c_target",
                "text": "The target follows the anchor.",
                "source": "derived",
                "derived_from": ["c_anchor"],
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_anchor",
                "observable_goal": "Observe the anchor.",
                "evidence_kind": "relation",
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_target",
                "observable_goal": "Observe the first target after the anchor.",
                "evidence_kind": "relation",
                "temporal_relation": "after:first",
                "depends_on": ["req_anchor"],
            },
            {
                "op": "set_obligation_status",
                "requirement_id": "req_anchor",
                "status": "satisfied",
                "supporting_claim_ids": ["c_anchor"],
                "supporting_attempt_ids": [attempt_id],
            },
            {
                "op": "set_obligation_status",
                "requirement_id": "req_target",
                "status": "satisfied",
                "supporting_claim_ids": ["c_target"],
                "supporting_attempt_ids": [attempt_id],
            },
        ),
        observation_ids=(attempt_id,),
    )

    assert accepted.accepted
    validation = document.validate_answer(
        ("c_target",),
        observation_ids=(attempt_id,),
        supporting_observation_ids=(attempt_id,),
        require_obligation_coverage=True,
    )
    assert validation.passed
    assert document.obligation_summary()["obligation_coverage_rate"] == 1.0


def test_open_answer_bearing_obligation_blocks_definitive_answer() -> None:
    document = WorkingDocument.with_question_premise("Which statements are true?")
    attempt_id = "attempt_one"
    operations = [
        {
            "op": "add_claim",
            "claim_id": "c1",
            "text": "Only statement one was observed.",
            "source": "observation",
            "cites": [attempt_id],
        }
    ]
    for index in range(1, 5):
        operations.append(
            {
                "op": "add_obligation",
                "requirement_id": f"req_{index}",
                "observable_goal": f"Observe statement {index}.",
                "evidence_kind": "persistent_state",
            }
        )
    operations.append(
        {
            "op": "set_obligation_status",
            "requirement_id": "req_1",
            "status": "satisfied",
            "supporting_claim_ids": ["c1"],
            "supporting_attempt_ids": [attempt_id],
        }
    )

    assert document.apply_ops(tuple(operations), observation_ids=(attempt_id,)).accepted
    validation = document.validate_answer(
        ("c1",),
        observation_ids=(attempt_id,),
        supporting_observation_ids=(attempt_id,),
        require_obligation_coverage=True,
    )

    assert not validation.passed
    assert "open_answer_bearing_obligation:req_2:open" in validation.errors
    assert document.obligation_summary()["open_obligation_count_at_answer"] == 3


def test_unresolved_obligation_requires_explicit_uncertainty() -> None:
    document = WorkingDocument.with_question_premise("What can be concluded?")
    attempt_id = "attempt_observed"
    assert document.apply_ops(
        (
            {
                "op": "add_claim",
                "claim_id": "c1",
                "text": "One answer-bearing fact is observed.",
                "source": "observation",
                "cites": [attempt_id],
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_observed",
                "observable_goal": "Observe the answer-bearing fact.",
            },
            {
                "op": "add_obligation",
                "requirement_id": "req_missing",
                "observable_goal": "Observe the missing fact.",
            },
            {
                "op": "set_obligation_status",
                "requirement_id": "req_observed",
                "status": "satisfied",
                "supporting_claim_ids": ["c1"],
                "supporting_attempt_ids": [attempt_id],
            },
            {
                "op": "set_obligation_status",
                "requirement_id": "req_missing",
                "status": "unresolved",
                "residual_uncertainty": "The bounded material did not resolve this requirement.",
            },
        ),
        observation_ids=(attempt_id,),
    ).accepted

    validation = document.validate_answer(
        ("c1",),
        observation_ids=(attempt_id,),
        supporting_observation_ids=(attempt_id,),
        require_obligation_coverage=True,
    )
    assert validation.passed
    assert document.obligation_summary()["unresolved_obligation_count_at_answer"] == 1


def test_search_miss_cannot_satisfy_observation_obligation() -> None:
    document = WorkingDocument.with_question_premise("Did the event occur?")
    result = document.apply_ops(
        (
            {
                "op": "add_obligation",
                "requirement_id": "req_event",
                "observable_goal": "Observe the event in a bounded scope.",
                "evidence_kind": "transient_event",
            },
            {
                "op": "set_obligation_status",
                "requirement_id": "req_event",
                "status": "satisfied",
            },
        ),
        observation_ids=("attempt_empty_caption_search",),
    )

    assert not result.accepted
    assert any("satisfied_obligation_requires_claim:req_event" in error for error in result.errors)
    assert any("satisfied_obligation_requires_attempt:req_event" in error for error in result.errors)


def test_obligation_metrics_come_from_final_ledger_snapshot() -> None:
    metrics = agent_run_metrics(
        (
            {
                "type": "answer_outcome",
                "obligation_summary": {
                    "answer_bearing_obligation_count": 4,
                    "satisfied_obligation_count": 3,
                    "open_obligation_count_at_answer": 0,
                    "unresolved_obligation_count_at_answer": 1,
                    "obligation_coverage_rate": 0.75,
                },
            },
        ),
        (),
        answer_present=True,
        reference_valid=True,
    )

    assert metrics["answer_bearing_obligation_count"] == 4
    assert metrics["satisfied_obligation_count"] == 3
    assert metrics["open_obligation_count_at_answer"] == 0
    assert metrics["unresolved_obligation_count_at_answer"] == 1
    assert metrics["obligation_coverage_rate"] == 0.75
