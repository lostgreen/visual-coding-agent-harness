from __future__ import annotations

import json

import pytest

from vcah.wp17_memory_construction import build_ocr_tracks, build_track_evidence


def _metadata(index: int, *, video: str = "video-a") -> dict:
    return {
        "frame_id": f"frame-{index}",
        "segment_id": "segment-a" if index < 10 else "segment-b",
        "source_video_id": video,
        "virtual_time_sec": float(index),
        "source_time_sec": float(index),
        "source_path": "/private/video.mp4",
    }


def _row(
    index: int,
    text: str,
    *,
    region: str = "boss_name_bar",
    bbox: list[float] | None = None,
) -> dict:
    row = {
        "frame_label": f"label-{index}",
        "text": text,
        "entity_type": "boss_name",
        "ui_region": region,
        "confidence": "high",
    }
    if bbox is not None:
        row["bbox"] = bbox
    return row


def test_tracks_preserve_rows_and_split_reappearances() -> None:
    rows = (
        _row(0, "寅虎", bbox=[0.2, 0.1, 0.6, 0.2]),
        _row(1, "寅虎"),
        _row(8, "寅虎"),
        _row(9, "F", region="other"),
    )
    metadata = {f"label-{index}": _metadata(index) for index in (0, 1, 8, 9)}

    result = build_ocr_tracks(rows, frame_metadata=metadata, max_gap_sec=3.0)
    tracks = result["tracks"]

    assert result["structural_gate_passed"] is True
    assert result["counts"]["assigned_observations"] == 4
    assert result["counts"]["tracks"] == 3
    tiger_tracks = [row for row in tracks if row["canonical_surface"] == "寅虎"]
    assert [row["support_frame_count"] for row in tiger_tracks] == [2, 1]
    assert tiger_tracks[0]["bbox_series"][0]["bbox"] == [0.2, 0.1, 0.6, 0.2]
    assert any(row["canonical_surface"] == "F" for row in tracks)
    assert "source_path" not in json.dumps(tracks)


def test_tracks_separate_ui_regions_and_source_videos() -> None:
    rows = (
        _row(0, "Yin Tiger", region="boss_name_bar"),
        _row(1, "Yin Tiger", region="menu_title"),
        _row(2, "Yin Tiger", region="boss_name_bar"),
    )
    metadata = {
        "label-0": _metadata(0, video="video-a"),
        "label-1": _metadata(1, video="video-a"),
        "label-2": _metadata(2, video="video-b"),
    }

    result = build_ocr_tracks(rows, frame_metadata=metadata, max_gap_sec=3.0)

    assert result["counts"]["tracks"] == 3
    assert {tuple(row["ui_regions"]) for row in result["tracks"]} == {
        ("boss_name_bar",),
        ("menu_title",),
    }


def test_track_evidence_has_stable_unique_handles_and_no_paths() -> None:
    rows = (_row(0, "赤髯龙"), _row(1, "赤髯龙"))
    metadata = {f"label-{index}": _metadata(index) for index in (0, 1)}
    first = build_ocr_tracks(rows, frame_metadata=metadata)
    second = build_ocr_tracks(rows, frame_metadata=metadata)

    evidence = build_track_evidence(first["tracks"])

    assert first["tracks"] == second["tracks"]
    assert evidence[0]["evidence_id"].startswith("ocr:ocr_track_")
    assert evidence[0]["surface"] == "赤髯龙"
    assert evidence[0]["lineage_complete"] is True
    assert "source_path" not in json.dumps(evidence)


def test_track_build_rejects_missing_lineage() -> None:
    with pytest.raises(ValueError, match="frame lineage"):
        build_ocr_tracks((_row(0, "寅虎"),), frame_metadata={})


def test_duplicate_rows_are_preserved_as_distinct_observations() -> None:
    row = _row(0, "寅虎")
    result = build_ocr_tracks(
        (row, dict(row)), frame_metadata={"label-0": _metadata(0)}
    )

    assert result["counts"]["assigned_observations"] == 2
    assert len(result["tracks"][0]["observation_ids"]) == 2
