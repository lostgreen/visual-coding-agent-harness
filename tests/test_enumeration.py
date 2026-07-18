from __future__ import annotations

from vcah.enumeration import build_enumeration_manifest


def test_enumeration_does_not_treat_repaired_or_oversized_window_as_complete() -> None:
    manifest = build_enumeration_manifest(
        target_segment_id="seg_1",
        required_range=(0.0, 30.0),
        windows=(
            {
                "range": [0.0, 30.0],
                "sampling_fps": 2.0,
                "parse_status": "repaired",
                "candidate_ids": ["event_1"],
            },
        ),
        candidate_ids=("event_1",),
        unresolved_candidate_ids=(),
        expected_event_dwell_sec=1.0,
    )

    assert manifest["enumeration_complete"] is False
    assert manifest["windows"][0]["status"] == "incomplete"
    assert manifest["unprocessed_ranges"] == [[0.0, 30.0]]


def test_enumeration_requires_complete_overlap_windows_and_reconciled_candidates() -> None:
    manifest = build_enumeration_manifest(
        target_segment_id="seg_1",
        required_range=(0.0, 20.0),
        windows=(
            {
                "range": [0.0, 12.0],
                "sampling_fps": 2.0,
                "parse_status": "ok",
                "candidate_ids": ["event_1"],
            },
            {
                "range": [10.0, 20.0],
                "sampling_fps": 2.0,
                "parse_status": "ok",
                "candidate_ids": ["event_2"],
            },
        ),
        candidate_ids=("event_1", "event_2"),
        unresolved_candidate_ids=(),
        expected_event_dwell_sec=1.0,
    )

    assert manifest["enumeration_complete"] is True
    assert manifest["unprocessed_ranges"] == []
    assert manifest["boundary_gaps"] == []
    assert manifest["candidate_reconciliation_status"] == "complete"
