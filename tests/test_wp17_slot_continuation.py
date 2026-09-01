from __future__ import annotations

import pytest

from vcah.wp17_slot_continuation import (
    WP17_SLOT_CONTINUATION_PLAN_CONTRACT,
    build_continuation_entries,
    continuation_semantic_payload,
    index_continuation_entries,
)


def _segments() -> list[dict]:
    rows = []
    for window in range(2):
        for ordinal in range(4):
            rows.append(
                {
                    "segment_id": f"w{window}-s{ordinal}",
                    "window_id": f"w{window}",
                    "window_segment_ordinal": ordinal,
                }
            )
    return rows


def _parent_rows(segments: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (segment["segment_id"], arm): {"status": "success"}
        for segment in segments
        for arm in ("e1c0", "e1c1", "e1c2")
    }


def test_dependency_closure_is_arm_specific_and_window_bounded() -> None:
    segments = _segments()
    parent = _parent_rows(segments)
    parent[("w0-s1", "e1c0")] = {
        "status": "failed",
        "failure_code": "validation_retry_exhausted",
    }
    parent[("w0-s2", "e1c1")] = {
        "status": "failed",
        "failure_code": "validation_retry_exhausted",
    }
    parent[("w1-s1", "e1c2")] = {
        "status": "failed",
        "failure_code": "model_call_hard_cap_exhausted",
    }

    entries = {
        (row["segment_id"], row["arm"]): row
        for row in build_continuation_entries(segments, parent)
    }

    assert entries[("w0-s1", "e1c0")]["action"] == "rerun"
    assert entries[("w0-s2", "e1c0")]["action"] == "reuse"
    assert entries[("w0-s1", "e1c1")]["action"] == "reuse"
    assert entries[("w0-s2", "e1c1")]["action"] == "rerun"
    assert entries[("w0-s3", "e1c1")]["action"] == "rerun"
    assert entries[("w1-s0", "e1c1")]["action"] == "reuse"
    assert entries[("w1-s0", "e1c2")]["action"] == "reuse"
    assert entries[("w1-s1", "e1c2")]["action"] == "rerun"
    assert entries[("w1-s3", "e1c2")]["action"] == "rerun"


def test_parent_result_set_must_be_complete() -> None:
    segments = _segments()
    parent = _parent_rows(segments)
    parent.pop(("w0-s0", "e1c0"))

    with pytest.raises(ValueError, match="result set mismatch"):
        build_continuation_entries(segments, parent)


def test_continuation_plan_rejects_duplicate_keys() -> None:
    row = {
        "segment_id": "s1",
        "arm": "e1c0",
        "action": "reuse",
        "reasons": ["parent_success_dependency_intact"],
    }
    plan = {
        "contract": WP17_SLOT_CONTINUATION_PLAN_CONTRACT,
        "entries": [row, row],
    }

    with pytest.raises(ValueError, match="duplicate"):
        index_continuation_entries(plan)


def test_chained_reuse_compares_semantic_payload_not_hop_provenance() -> None:
    parent = {
        "segment_id": "s1",
        "arm": "e1c0",
        "status": "success",
        "structured_event_record": {"summary": "event"},
        "continuation_provenance": {
            "action": "rerun",
            "plan_sha256": "first-hop",
        },
    }
    child = {
        **parent,
        "continuation_provenance": {
            "action": "reuse",
            "plan_sha256": "second-hop",
        },
    }

    assert continuation_semantic_payload(child) == continuation_semantic_payload(parent)


def test_chained_reuse_still_detects_semantic_mutation() -> None:
    parent = {
        "status": "success",
        "structured_event_record": {"summary": "event"},
        "continuation_provenance": {"plan_sha256": "first-hop"},
    }
    child = {
        **parent,
        "structured_event_record": {"summary": "changed"},
        "continuation_provenance": {"plan_sha256": "second-hop"},
    }

    assert continuation_semantic_payload(child) != continuation_semantic_payload(parent)
