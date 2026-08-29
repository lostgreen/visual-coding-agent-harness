from __future__ import annotations

from vcah.occurrence_entity_sidecar import admit_global_entity_rows
from vcah.wp17_preflight import (
    build_wp17_preflight_report,
    compact_surface,
    diagnostic_frame_label,
    select_surface_review_rows,
    surface_matches,
)


def _selection(case_id: str, index: int, time_sec: float) -> dict:
    return {
        "segment_id": "segment-a",
        "tier0_frame_index": index,
        "virtual_time_sec": time_sec,
        "source_time_sec": time_sec,
        "source_video_id": "video-a",
        "diagnostic_case_ids": [case_id],
    }


def _row(selection: dict, text: str, region: str = "other") -> dict:
    return {
        "frame_label": diagnostic_frame_label(selection),
        "text": text,
        "ui_region": region,
        "entity_type": "boss_name",
        "confidence": "high",
    }


def test_surface_matching_normalizes_case_and_wrappers() -> None:
    assert compact_surface("妖王·赤髯龙") == "妖王赤髯龙"
    assert surface_matches("妖王·赤髯龙", ("赤髯龙",))
    assert surface_matches("QINGZHANG DAO", ("Qingzhang Dao",))
    assert not surface_matches("妖王·赤鼻龙", ("赤髯龙",))


def test_surface_review_rows_cover_timeline_and_high_value_ocr() -> None:
    selection = [_selection("0016", index, float(index)) for index in range(12)]
    parsed = [
        {
            "frame_label": diagnostic_frame_label(selection[9]),
            "text": "Yin Tiger",
            "ui_region": "boss_name_bar",
        }
    ]
    selected = select_surface_review_rows(
        case_id="0016",
        selection_rows=selection,
        parsed_rows=parsed,
        max_frames=6,
    )

    indexes = {row["tier0_frame_index"] for row in selected}
    assert {0, 5, 9, 11} <= indexes
    assert len(selected) == 6
    assert next(row for row in selected if row["tier0_frame_index"] == 9)[
        "surface_review_score"
    ] == 4


def test_preflight_separates_admission_reader_and_pixel_failures() -> None:
    admitted_a = _selection("case-admitted", 1, 10.0)
    admitted_b = _selection("case-admitted", 2, 11.0)
    rejected = _selection("case-rejected", 3, 20.0)
    reader_a = _selection("case-reader", 4, 30.0)
    reader_b = _selection("case-reader", 5, 31.0)
    absent = _selection("case-absent", 6, 40.0)
    report = build_wp17_preflight_report(
        case_specs={
            "case-admitted": {
                "canonical_entity": "Red Loong",
                "expected_surfaces": ["赤髯龙"],
                "visual_audit": {"pixel_status": "visible"},
            },
            "case-rejected": {
                "canonical_entity": "Tiger",
                "expected_surfaces": ["Tiger"],
                "visual_audit": {"pixel_status": "visible"},
            },
            "case-reader": {
                "canonical_entity": "First Prince",
                "expected_surfaces": ["沙大郎"],
                "visual_audit": {"pixel_status": "visible"},
            },
            "case-absent": {
                "canonical_entity": "Qingzhang Dao",
                "expected_surfaces": ["青嶂道"],
                "visual_audit": {"pixel_status": "absent"},
            },
        },
        selection_rows=(
            admitted_a,
            admitted_b,
            rejected,
            reader_a,
            reader_b,
            absent,
        ),
        parsed_rows=(
            _row(admitted_a, "妖王·赤髯龙", "boss_name_bar"),
            _row(admitted_b, "妖王·赤髯龙", "boss_name_bar"),
            _row(rejected, "Tiger", "other"),
            _row(reader_a, "沙太尉", "boss_name_bar"),
            _row(reader_b, "沙太尉", "boss_name_bar"),
        ),
        merge_gap_sec=60.0,
        structural_checks={"runtime_valid": True},
    )
    rows = {row["case_id"]: row for row in report["case_level"]}
    assert rows["case-admitted"]["category"] == "target_surface_admitted"
    assert rows["case-rejected"]["category"] == "admission_rejection"
    assert rows["case-reader"]["category"] == "reader_near_miss"
    assert rows["case-absent"]["category"] == "pixel_absent"
    assert len(report["admission_grid"]) == 12
    assert report["structural_gate_passed"] is True
    assert report["decision"] == "WP17_0_SURFACE_AUDIT_READY"


def test_lexical_filter_can_be_disabled_only_for_ablation() -> None:
    rows = (
        {
            "frame_label": "f1",
            "text": "Tiger",
            "entity_type": "boss_name",
            "ui_region": "boss_name_bar",
            "confidence": "high",
        },
    )
    metadata = {"f1": {"virtual_time_sec": 1.0}}
    current = admit_global_entity_rows(
        rows,
        passage_id="p1",
        frame_metadata=metadata,
    )
    ablation = admit_global_entity_rows(
        rows,
        passage_id="p1",
        frame_metadata=metadata,
        lexical_filter_enabled=False,
    )
    assert current["admitted_rows"] == ()
    assert current["rejection_counts"] == {"insufficient_english_tokens": 1}
    assert [row["text"] for row in ablation["admitted_rows"]] == ["Tiger"]
