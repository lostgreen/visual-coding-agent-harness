from __future__ import annotations

import json

import pytest

from vcah.wp17_slot_memory import (
    WP17_CAPSULE_PROVENANCE_CONTRACT,
    WP17_SLOT_TRANSACTION_CONTRACT,
    SlotMemoryState,
    SlotTransactionError,
    budget_token_count,
    parse_transaction_response,
    tail_budget_text,
    validate_construction_output,
)


def _ser(summary: str = "The encounter starts.") -> dict:
    return {
        "entities": ["player", "boss"],
        "events": ["encounter-start"],
        "state_changes": [],
        "relations": [],
        "occurrence_refs": [],
        "summary": summary,
    }


def _transaction(*operations: dict) -> dict:
    return {
        "contract": WP17_SLOT_TRANSACTION_CONTRACT,
        "observations": [
            {
                "observation_id": "obs-1",
                "kind": "event",
                "fact": "The player and boss are fighting.",
                "evidence_ids": ["frame:1"],
                "participants": ["player", "boss"],
            }
        ],
        "slot_operations": list(operations),
        "structured_event_record": _ser(),
    }


def test_slot_transaction_validates_observations_before_ops_and_binds_participants() -> None:
    state = SlotMemoryState("e1c2", token_budget=600)
    result = state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "active_encounter",
                "expected_version": 0,
                "value": {"event_id": "enc-1", "label": "boss fight"},
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "write",
                "slot": "active_participants",
                "expected_version": 0,
                "value": {
                    "event_ref": "enc-1",
                    "participants": ["player", "boss"],
                },
                "observation_ids": ["obs-1"],
            },
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )

    assert result["capsule"]["within_budget"] is True
    assert state.records["active_encounter"]["version"] == 1
    assert state.records["active_participants"]["provenance"] == ["frame:1"]
    capsule_slot = next(
        row
        for row in result["capsule"]["slots"]
        if row["slot"] == "active_participants"
    )
    assert capsule_slot["provenance_count"] == 1
    assert len(capsule_slot["provenance_digest"]) == 64
    assert "provenance" not in capsule_slot
    assert (
        result["capsule"]["provenance_projection_contract"]
        == WP17_CAPSULE_PROVENANCE_CONTRACT
    )


def test_working_slots_require_explicit_lifecycle() -> None:
    state = SlotMemoryState("e1c2")
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "walking"},
                "observation_ids": ["obs-1"],
            }
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    with pytest.raises(SlotTransactionError, match="explicit lifecycle"):
        state.apply(
            _transaction(),
            segment_id="segment-2",
            allowed_evidence_ids=("frame:1",),
        )


def test_retain_can_refresh_provenance_without_rewriting_value() -> None:
    state = SlotMemoryState("e1c2")
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "walking"},
                "observation_ids": ["obs-1"],
            }
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    payload = _transaction(
        {
            "operation": "retain",
            "slot": "current_activity",
            "expected_version": 1,
            "observation_ids": ["obs-1"],
        }
    )
    payload["observations"][0]["evidence_ids"] = ["frame:2"]

    result = state.apply(
        payload,
        segment_id="segment-2",
        allowed_evidence_ids=("frame:2",),
    )

    record = state.records["current_activity"]
    assert record["value"] == {"activity": "walking"}
    assert record["version"] == 2
    assert record["last_verified_segment_id"] == "segment-2"
    assert record["provenance"] == ["frame:1", "frame:2"]
    assert result["lifecycle_events"][0]["operation"] == "retain"
    with pytest.raises(SlotTransactionError, match="cannot rewrite"):
        state.apply(
            _transaction(
                {
                    "operation": "retain",
                    "slot": "current_activity",
                    "expected_version": 2,
                    "value": {"activity": "running"},
                    "observation_ids": [],
                }
            ),
            segment_id="segment-2",
            allowed_evidence_ids=("frame:1",),
        )


def test_archive_and_evict_remove_working_context_but_preserve_ledger() -> None:
    state = SlotMemoryState("e1c2")
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "walking"},
                "observation_ids": ["obs-1"],
            }
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    state.apply(
        _transaction(
            {
                "operation": "close",
                "slot": "current_activity",
                "expected_version": 1,
                "observation_ids": ["obs-1"],
            }
        ),
        segment_id="segment-2",
        allowed_evidence_ids=("frame:1",),
    )
    state.apply(
        _transaction(
            {
                "operation": "archive",
                "slot": "current_activity",
                "expected_version": 2,
                "observation_ids": [],
            }
        ),
        segment_id="segment-3",
        allowed_evidence_ids=("frame:1",),
    )

    assert state.capsule()["slots"] == []
    assert state.records["current_activity"]["status"] == "archived"
    assert len(state.ledger) == 3
    restored = SlotMemoryState.from_snapshot(state.snapshot())
    assert restored.digest() == state.digest()


def test_active_capsule_over_budget_fails_without_silent_truncation() -> None:
    state = SlotMemoryState("e1c2", token_budget=10)
    digest_before = state.digest()
    with pytest.raises(SlotTransactionError, match="exceeds"):
        state.apply(
            _transaction(
                {
                    "operation": "write",
                    "slot": "current_activity",
                    "expected_version": 0,
                    "value": {"activity": "a very long active activity description"},
                    "observation_ids": ["obs-1"],
                }
            ),
            segment_id="segment-1",
            allowed_evidence_ids=("frame:1",),
        )
    assert state.digest() == digest_before


def test_capsule_summarizes_lineage_but_state_and_ledger_keep_full_provenance() -> None:
    evidence_ids = tuple(f"frame:segment-1:{index:04d}" for index in range(1, 7))
    payload = _transaction(
        {
            "operation": "write",
            "slot": "current_activity",
            "expected_version": 0,
            "value": {"activity": "walking"},
            "observation_ids": ["obs-1"],
        }
    )
    payload["observations"][0]["evidence_ids"] = list(evidence_ids)
    state = SlotMemoryState("e1c2", token_budget=600)

    result = state.apply(
        payload,
        segment_id="segment-1",
        allowed_evidence_ids=evidence_ids,
    )

    capsule_slot = result["capsule"]["slots"][0]
    assert capsule_slot["provenance_count"] == 6
    assert not any(value in result["capsule"]["context"] for value in evidence_ids)
    assert state.records["current_activity"]["provenance"] == list(evidence_ids)
    assert state.ledger[-1]["provenance"] == list(evidence_ids)


def test_budget_tokenizer_and_response_parser_are_deterministic() -> None:
    text = "older context 新状态 most recent"
    tail = tail_budget_text(text, max_tokens=3)
    assert budget_token_count(tail) == 3
    payload = _transaction()
    assert parse_transaction_response("```json\n" + json.dumps(payload) + "\n```") == payload


def test_local_evidence_aliases_canonicalize_before_persistence() -> None:
    payload = _transaction()
    payload["observations"][0]["evidence_ids"] = ["f001"]

    normalized = validate_construction_output(
        payload,
        arm="e1c0",
        segment_id="segment-1",
        allowed_evidence_ids=("f001",),
        evidence_id_map={"f001": "frame:segment-1:0001"},
    )

    assert normalized["observations"][0]["evidence_ids"] == [
        "frame:segment-1:0001"
    ]


def test_output_bounds_reject_unbounded_observation_transcription() -> None:
    payload = _transaction()
    payload["observations"] = [
        {
            "observation_id": f"obs-{index}",
            "kind": "visible_text",
            "fact": f"row {index}",
            "evidence_ids": ["frame:1"],
            "participants": [],
        }
        for index in range(17)
    ]

    with pytest.raises(SlotTransactionError, match="observations exceed"):
        validate_construction_output(
            payload,
            arm="e1c0",
            segment_id="segment-1",
            allowed_evidence_ids=("frame:1",),
        )


def test_observation_keeps_valid_support_beyond_six_ids_without_truncation() -> None:
    evidence_ids = tuple(f"frame:{index}" for index in range(1, 9))
    payload = _transaction()
    payload["observations"][0]["evidence_ids"] = list(evidence_ids)

    normalized = validate_construction_output(
        payload,
        arm="e1c0",
        segment_id="segment-1",
        allowed_evidence_ids=evidence_ids,
    )

    assert normalized["observations"][0]["evidence_ids"] == list(evidence_ids)


def test_ser_singletons_normalize_without_semantic_inference() -> None:
    payload = _transaction()
    payload["structured_event_record"]["entities"] = {"name": "boss"}
    payload["structured_event_record"]["relations"] = None

    normalized = validate_construction_output(
        payload,
        arm="e1c0",
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )

    assert normalized["structured_event_record"]["entities"] == [{"name": "boss"}]


def test_ser_missing_list_fields_normalize_to_empty_without_inference() -> None:
    payload = _transaction()
    del payload["structured_event_record"]["entities"]
    del payload["structured_event_record"]["relations"]

    normalized = validate_construction_output(
        payload,
        arm="e1c0",
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )

    assert normalized["structured_event_record"]["entities"] == []
    assert normalized["structured_event_record"]["relations"] == []
    assert normalized["structured_event_record"]["relations"] == []


def test_unknown_slot_observation_feedback_lists_valid_ids() -> None:
    state = SlotMemoryState("e1c2")
    payload = _transaction(
        {
            "operation": "write",
            "slot": "current_activity",
            "expected_version": 0,
            "value": {"activity": "fight"},
            "observation_ids": ["frame:1"],
        }
    )

    with pytest.raises(SlotTransactionError, match="valid observation IDs.*obs-1"):
        state.apply(
            payload,
            segment_id="segment-1",
            allowed_evidence_ids=("frame:1",),
        )
