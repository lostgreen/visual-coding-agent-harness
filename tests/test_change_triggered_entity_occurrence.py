from __future__ import annotations

from io import BytesIO

import numpy as np

from vcah.change_triggered_entity_occurrence import (
    _read_change_stream,
    admit_entity_occurrences,
    frame_change_scores,
    select_change_budget,
    select_interval_diagnostic,
    select_uniform_budget,
)
from vcah.virtual_video import VirtualVideoSegment


def _observations(count: int) -> tuple[dict, ...]:
    return tuple(
        {
            "segment_id": "seg-a",
            "source_video_id": "video-a",
            "source_path": "/video.mp4",
            "tier0_frame_index": index,
            "source_time_sec": float(index),
            "virtual_time_sec": float(index),
            "selection_score": float(index % 7),
        }
        for index in range(count)
    )


def test_frame_change_scores_weight_ui_regions_separately() -> None:
    prior = np.zeros((20, 20), dtype=np.uint8)
    current = prior.copy()
    current[:4, :] = 255
    global_score, ui_score = frame_change_scores(prior, current)
    assert 0.15 < global_score < 0.25
    assert ui_score > global_score
    assert frame_change_scores(None, current) == (0.0, 0.0)


def test_raw_stream_produces_lineage_without_materializing_frames() -> None:
    segment = VirtualVideoSegment(
        segment_id="seg-a",
        source_video_id="video-a",
        source_path="/video.mp4",
        source_start_sec=10.0,
        source_end_sec=13.0,
        virtual_start_sec=100.0,
        virtual_end_sec=103.0,
    )
    first = bytes([0] * 16)
    second = bytes([255] * 16)
    rows = _read_change_stream(
        BytesIO(first + second),
        segment=segment,
        fps=1.0,
        width=4,
        height=4,
    )
    assert len(rows) == 2
    assert rows[0]["source_time_sec"] == 10.0
    assert rows[1]["virtual_time_sec"] == 101.0
    assert rows[0]["selection_score"] == 0.0
    assert rows[1]["selection_score"] == 1.0


def test_uniform_and_change_selection_are_exact_and_budget_matched() -> None:
    observations = _observations(30)
    uniform = select_uniform_budget(observations, budget=8)
    changed = select_change_budget(
        observations,
        budget=8,
        coverage_bin_sec=10.0,
        min_spacing_sec=2.0,
    )
    assert len(uniform) == len(changed) == 8
    assert len({row["tier0_frame_index"] for row in uniform}) == 8
    assert len({row["tier0_frame_index"] for row in changed}) == 8
    assert {int(row["virtual_time_sec"] // 10) for row in changed} == {0, 1, 2}
    assert all(row["selection_arm"] == "a1_uniform" for row in uniform)
    assert all(row["selection_arm"] == "a2_change" for row in changed)


def test_change_selection_uses_deterministic_spacing_fallback() -> None:
    observations = tuple(
        {
            **row,
            "selection_score": 100.0 - index,
        }
        for index, row in enumerate(_observations(6))
    )
    selected = select_change_budget(
        observations,
        budget=6,
        coverage_bin_sec=300.0,
        min_spacing_sec=100.0,
    )
    assert len(selected) == 6
    assert any(
        row["selection_reason"] == "exact_budget_spacing_fallback" for row in selected
    )


def test_interval_diagnostic_selects_shared_frames_and_tracks_cases() -> None:
    selected = select_interval_diagnostic(
        _observations(8),
        intervals_by_case={
            "case-a": ((1.0, 3.0),),
            "case-b": ((3.0, 4.0),),
        },
    )
    assert [row["tier0_frame_index"] for row in selected] == [1, 2, 3, 4]
    assert selected[2]["diagnostic_case_ids"] == ["case-a", "case-b"]
    assert all(
        row["selection_arm"] == "a3_tier0_diagnostic" for row in selected
    )


def test_entity_rows_merge_into_occurrences_and_preserve_lineage() -> None:
    rows = (
        {
            "frame_label": "f1",
            "text": "赤髯龙",
            "entity_type": "boss_name",
            "ui_region": "boss_name_bar",
            "confidence": "high",
        },
        {
            "frame_label": "f2",
            "text": "赤髯龙",
            "entity_type": "boss_name",
            "ui_region": "boss_name_bar",
            "confidence": "medium",
        },
        {
            "frame_label": "f3",
            "text": "赤髯龙",
            "entity_type": "boss_name",
            "ui_region": "boss_name_bar",
            "confidence": "high",
        },
    )
    metadata = {
        "f1": {
            "frame_id": "frame-1",
            "virtual_time_sec": 10.0,
            "segment_id": "seg-a",
            "source_video_id": "video-a",
        },
        "f2": {
            "frame_id": "frame-2",
            "virtual_time_sec": 20.0,
            "segment_id": "seg-a",
            "source_video_id": "video-a",
        },
        "f3": {
            "frame_id": "frame-3",
            "virtual_time_sec": 100.0,
            "segment_id": "seg-a",
            "source_video_id": "video-a",
        },
    }
    result = admit_entity_occurrences(
        rows,
        frame_metadata=metadata,
        merge_gap_sec=60.0,
    )
    occurrences = result["occurrences"]
    assert len(occurrences) == 2
    assert occurrences[0]["support_count"] == 2
    assert occurrences[0]["occurrence_start_sec"] == 10.0
    assert occurrences[0]["occurrence_end_sec"] == 20.0
    assert occurrences[1]["admission_reason"] == "high_value_ui_region"
    assert occurrences[0]["occurrence_id"] != occurrences[1]["occurrence_id"]


def test_missing_frame_lineage_is_rejected_explicitly() -> None:
    result = admit_entity_occurrences(
        ({"frame_label": "missing", "text": "赤髯龙"},),
        frame_metadata={},
    )
    assert result["occurrences"] == ()
    assert result["rejection_counts"] == {"missing_frame_lineage": 1}
