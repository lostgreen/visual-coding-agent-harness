from __future__ import annotations

import json

from _canonical import canonical_trace_diff, canonicalize_event


def test_canonicalize_event_is_stable_across_round_trips() -> None:
    event = {
        "timestamp": "2026-06-15T00:00:00Z",
        "created_at": "2026-06-15T00:00:01Z",
        "event": "tool_result",
        "score": 0.123456789,
        "target_entities": [{"id": "T2"}, {"id": "T1"}],
        "payload": {
            "run_id": "run-a",
            "values": [3.333333333, {"absolute_path": "/tmp/a", "kept": "yes"}],
        },
    }

    once = canonicalize_event(event)
    twice = canonicalize_event(json.loads(json.dumps(once, sort_keys=True)))

    assert once == twice
    assert json.dumps(once, sort_keys=True) == json.dumps(twice, sort_keys=True)
    assert "timestamp" not in once
    assert "created_at" not in once
    assert once["score"] == 0.123457
    assert once["target_entities"] == [{"id": "T1"}, {"id": "T2"}]
    assert once["payload"] == {"values": [3.333333, {"kept": "yes"}]}


def test_canonical_trace_diff_ignores_timestamp_only_changes(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    actual = tmp_path / "actual.jsonl"
    baseline.write_text(
        json.dumps({"event": "round", "timestamp": "old", "answer": "A"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    actual.write_text(
        json.dumps({"event": "round", "timestamp": "new", "answer": "A"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert canonical_trace_diff(actual, baseline) == []


def test_canonical_trace_diff_ignores_created_at_only_changes(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    actual = tmp_path / "actual.jsonl"
    baseline.write_text(
        json.dumps({"type": "round", "created_at": "old", "answer": "A"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    actual.write_text(
        json.dumps({"type": "round", "created_at": "new", "answer": "A"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert canonical_trace_diff(actual, baseline) == []
