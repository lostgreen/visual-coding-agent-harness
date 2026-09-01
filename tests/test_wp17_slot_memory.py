from __future__ import annotations

import hashlib
import json

import pytest

from vcah.wp17_slot_memory import (
    WP17_CAPSULE_PROVENANCE_CONTRACT,
    WP17_SLOT_LIFECYCLE_POLICY_V10,
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


def test_omitted_working_slots_are_implicitly_retained_without_version_change() -> None:
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
    result = state.apply(
        _transaction(),
        segment_id="segment-2",
        allowed_evidence_ids=("frame:1",),
    )

    assert state.records["current_activity"]["version"] == 1
    assert result["lifecycle_events"][0]["operation"] == "implicit_retain"
    assert result["lifecycle_events"][0]["from_version"] == 1
    assert result["lifecycle_events"][0]["to_version"] == 1


def test_closed_slot_can_archive_and_write_replacement_in_one_transaction() -> None:
    state = SlotMemoryState("e1c2")
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "active_encounter",
                "expected_version": 0,
                "value": {"event_id": "enc-1"},
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "close",
                "slot": "active_encounter",
                "expected_version": 1,
                "observation_ids": ["obs-1"],
            },
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )

    result = state.apply(
        _transaction(
            {
                "operation": "archive",
                "slot": "active_encounter",
                "expected_version": 2,
                "observation_ids": [],
            },
            {
                "operation": "write",
                "slot": "active_encounter",
                "expected_version": 3,
                "value": {"event_id": "enc-2"},
                "observation_ids": ["obs-1"],
            },
        ),
        segment_id="segment-2",
        allowed_evidence_ids=("frame:1",),
    )

    assert state.records["active_encounter"]["version"] == 4
    assert state.records["active_encounter"]["value"] == {"event_id": "enc-2"}
    assert [row["operation"] for row in result["lifecycle_events"]] == [
        "archive",
        "write",
    ]


def test_v10_write_repair_has_literal_stepwise_versions() -> None:
    state = SlotMemoryState(
        "e1c2",
        lifecycle_policy=WP17_SLOT_LIFECYCLE_POLICY_V10,
    )
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "first"},
                "observation_ids": ["obs-1"],
            }
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    with pytest.raises(SlotTransactionError) as captured:
        state.apply(
            _transaction(
                {
                    "operation": "write",
                    "slot": "current_activity",
                    "expected_version": 1,
                    "value": {"activity": "second"},
                    "observation_ids": ["obs-1"],
                }
            ),
            segment_id="segment-2",
            allowed_evidence_ids=("frame:1",),
        )

    repair = captured.value.repair_contract()["details"]["repair_operations"]
    assert [row["operation"] for row in repair] == ["close", "archive", "write"]
    assert [row["expected_version"] for row in repair] == [1, 2, 3]
    state.apply(
        _transaction(*repair),
        segment_id="segment-2-repair",
        allowed_evidence_ids=("frame:1",),
    )
    assert state.records["current_activity"]["version"] == 4
    assert state.records["current_activity"]["value"] == {"activity": "second"}


def test_v10_terminal_operations_are_idempotent_and_recorded() -> None:
    state = SlotMemoryState(
        "e1c2",
        lifecycle_policy=WP17_SLOT_LIFECYCLE_POLICY_V10,
    )
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "walking"},
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "close",
                "slot": "current_activity",
                "expected_version": 1,
                "observation_ids": ["obs-1"],
            },
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    redundant = state.apply(
        _transaction(
            {
                "operation": "close",
                "slot": "current_activity",
                "expected_version": 2,
                "observation_ids": ["obs-1"],
            }
        ),
        segment_id="segment-2",
        allowed_evidence_ids=("frame:1",),
    )
    assert redundant["lifecycle_events"][0]["operation"] == "redundant_close"
    assert state.records["current_activity"]["version"] == 2


def test_v10_sweeps_closed_slot_after_one_untouched_transaction() -> None:
    state = SlotMemoryState(
        "e1c2",
        lifecycle_policy=WP17_SLOT_LIFECYCLE_POLICY_V10,
    )
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "walking"},
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "close",
                "slot": "current_activity",
                "expected_version": 1,
                "observation_ids": ["obs-1"],
            },
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    assert state.records["current_activity"]["closed_at_transaction_index"] == 1
    result = state.apply(
        _transaction(),
        segment_id="segment-2",
        allowed_evidence_ids=("frame:1",),
    )

    assert state.records["current_activity"]["status"] == "evicted"
    assert state.records["current_activity"]["version"] == 4
    assert any(
        row["operation"] == "runtime_lifecycle_sweep"
        for row in result["lifecycle_events"]
    )


def test_v9_compatibility_keeps_untouched_closed_slot() -> None:
    state = SlotMemoryState("e1c2")
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "walking"},
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "close",
                "slot": "current_activity",
                "expected_version": 1,
                "observation_ids": ["obs-1"],
            },
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    state.apply(
        _transaction(),
        segment_id="segment-2",
        allowed_evidence_ids=("frame:1",),
    )
    assert state.records["current_activity"]["status"] == "closed"
    assert "closed_at_transaction_index" not in state.records["current_activity"]


def test_v9_serialization_and_digest_remain_byte_compatible() -> None:
    state = SlotMemoryState("e1c2")
    capsule = state.capsule()
    snapshot = state.snapshot()
    expected_digest_payload = {
        "arm": "e1c2",
        "token_budget": 600,
        "transaction_index": 0,
        "records": {},
        "ledger": [],
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            expected_digest_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert "lifecycle_policy" not in capsule
    assert "lifecycle_policy" not in snapshot
    assert "closed_sweep_after_untouched_transactions" not in snapshot
    assert state.digest() == expected_digest
    assert SlotMemoryState.from_snapshot(snapshot).digest() == expected_digest


def test_multi_operation_failure_is_atomic_and_returns_structured_repair() -> None:
    state = SlotMemoryState("e1c2")
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "first"},
                "observation_ids": ["obs-1"],
            }
        ),
        segment_id="segment-1",
        allowed_evidence_ids=("frame:1",),
    )
    digest_before = state.digest()

    with pytest.raises(SlotTransactionError) as captured:
        state.apply(
            _transaction(
                {
                    "operation": "close",
                    "slot": "current_activity",
                    "expected_version": 1,
                    "observation_ids": ["obs-1"],
                },
                {
                    "operation": "archive",
                    "slot": "current_activity",
                    "expected_version": 99,
                    "observation_ids": [],
                },
            ),
            segment_id="segment-2",
            allowed_evidence_ids=("frame:1",),
        )

    assert state.digest() == digest_before
    repair = captured.value.repair_contract()
    assert repair["error_code"] == "slot_version_mismatch"
    assert repair["details"]["actual_version"] == 2
    assert "value" not in json.dumps(repair)


def test_participant_invariant_rejects_encounter_close_without_state_mutation() -> None:
    state = SlotMemoryState("e1c2")
    state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "active_encounter",
                "expected_version": 0,
                "value": {"event_id": "enc-1"},
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
    digest_before = state.digest()

    with pytest.raises(
        SlotTransactionError, match="requires an active encounter"
    ) as captured:
        state.apply(
            _transaction(
                {
                    "operation": "close",
                    "slot": "active_encounter",
                    "expected_version": 1,
                    "observation_ids": ["obs-1"],
                }
            ),
            segment_id="segment-2",
            allowed_evidence_ids=("frame:1",),
        )

    assert state.digest() == digest_before
    assert captured.value.code == "participants_without_active_encounter"


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


def test_update_changed_value_replaces_instead_of_unions_provenance() -> None:
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
            "operation": "update",
            "slot": "current_activity",
            "expected_version": 1,
            "value": {"activity": "running"},
            "observation_ids": ["obs-1"],
        }
    )
    payload["observations"][0]["evidence_ids"] = ["frame:2"]

    state.apply(
        payload,
        segment_id="segment-2",
        allowed_evidence_ids=("frame:2",),
    )

    assert state.records["current_activity"]["provenance"] == ["frame:2"]


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
    text = "older context，新状态！ most recent"
    tail = tail_budget_text(text, max_tokens=3)
    assert budget_token_count(tail) == 3
    assert tail == "！ most recent"
    assert tail_budget_text(text, max_tokens=99) == text
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
