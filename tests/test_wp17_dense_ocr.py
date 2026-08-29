from __future__ import annotations

import numpy as np

from vcah.virtual_video import VirtualVideoManifest, VirtualVideoSegment
from vcah.wp17_dense_ocr import (
    WP17_LOCAL_TIMELINE_CONTRACT,
    build_local_timeline,
    crop_normalized_view,
    paddle_result_rows,
    stable_frame_label,
)


def _manifest() -> VirtualVideoManifest:
    return VirtualVideoManifest(
        workspace_id="demo",
        duration_sec=100.0,
        segments=(
            VirtualVideoSegment(
                segment_id="seg-a",
                source_video_id="video-a",
                source_path="/private/a.mp4",
                source_start_sec=0.0,
                source_end_sec=50.0,
                virtual_start_sec=0.0,
                virtual_end_sec=50.0,
            ),
            VirtualVideoSegment(
                segment_id="seg-b",
                source_video_id="video-b",
                source_path="/private/b.mp4",
                source_start_sec=10.0,
                source_end_sec=60.0,
                virtual_start_sec=50.0,
                virtual_end_sec=100.0,
            ),
        ),
    )


def test_timeline_merges_overlaps_and_strips_paths() -> None:
    spec = {
        "contract": WP17_LOCAL_TIMELINE_CONTRACT,
        "protocol_frozen_before_dense_ocr_outcomes": True,
        "development_only": True,
        "padding_sec": 5.0,
        "sampling_fps": 1.0,
        "construction_inputs": {
            "question": False,
            "options": False,
            "gold_answer": False,
        },
        "scope_annotations_visible_to_reader": False,
        "cases": {
            "case-a": {"anchor_intervals": [[10.0, 20.0]]},
            "case-b": {"anchor_intervals": [[18.0, 30.0]]},
            "case-c": {"anchor_intervals": [[90.0, 98.0]]},
        },
        "views": [],
    }

    result = build_local_timeline(spec, manifest=_manifest())

    assert result["structural_gate_passed"] is True
    assert result["counts"] == {
        "cases": 3,
        "expanded_anchor_windows": 3,
        "merged_windows": 2,
        "timeline_slices": 2,
        "scoped_duration_sec": 45.0,
        "expected_sample_points": 45,
    }
    assert all("source_path" not in row for row in result["timeline_slices"])
    assert result["windows"][0]["case_ids"] == ["case-a", "case-b"]


def test_paddle_rows_map_roi_boxes_to_full_frame() -> None:
    result = {
        "res": {
            "rec_texts": ["寅虎", "F"],
            "rec_scores": [0.91, 0.55],
            "rec_boxes": [[10, 20, 110, 60], [0, 0, 20, 20]],
        }
    }

    rows = paddle_result_rows(
        result,
        frame_label="frame-a",
        view_id="name-bar",
        ui_region="boss_name_bar",
        view_bbox_norm=[0.1, 0.5, 0.9, 0.9],
        view_width=200,
        view_height=100,
        reader_source="paddleocr-3.7.0",
    )

    assert [row["text"] for row in rows] == ["寅虎", "F"]
    assert [row["confidence"] for row in rows] == ["high", "low"]
    assert rows[0]["bbox"] == [0.14, 0.58, 0.54, 0.74]


def test_paddle_rows_accept_polygon_boxes() -> None:
    result = {
        "rec_texts": ["赤髯龙"],
        "rec_scores": [0.7],
        "rec_polys": [[[10, 10], [90, 10], [90, 30], [10, 30]]],
    }

    rows = paddle_result_rows(
        result,
        frame_label="frame-b",
        view_id="full",
        ui_region="full_frame",
        view_bbox_norm=[0, 0, 1, 1],
        view_width=100,
        view_height=50,
        reader_source="paddleocr-3.7.0",
    )

    assert rows[0]["bbox"] == [0.1, 0.2, 0.9, 0.6]
    assert rows[0]["confidence"] == "medium"


def test_crop_and_frame_ids_are_stable() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_normalized_view(frame, [0.25, 0.2, 0.75, 0.8])

    assert crop.shape == (60, 100, 3)
    assert stable_frame_label("slice-a", 3) == stable_frame_label("slice-a", 3)
    assert stable_frame_label("slice-a", 3) != stable_frame_label("slice-a", 4)
