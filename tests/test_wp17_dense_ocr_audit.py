from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "tools" / "audit_mmlifelong_wp17_dense_ocr.py"
)
SPEC = importlib.util.spec_from_file_location("wp17_dense_ocr_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _track(surface: str, start: float, end: float, frames: int = 1) -> dict:
    return {
        "start_sec": start,
        "end_sec": end,
        "surfaces": [{"surface": surface, "count": frames}],
        "support_frame_count": frames,
    }


def test_dense_audit_excludes_pixel_absent_and_reports_complementarity() -> None:
    surface_spec = {
        "evaluation_only": True,
        "visible_to_ocr_model": False,
        "cases": {
            "case-strict": {
                "canonical_entity": "沙大郎",
                "expected_surfaces": ["沙大郎"],
                "diagnostic_alias_surfaces": [],
                "visual_audit": {"pixel_status": "visible"},
            },
            "case-near": {
                "canonical_entity": "赤髯龙",
                "expected_surfaces": ["赤髯龙"],
                "diagnostic_alias_surfaces": [],
                "visual_audit": {"pixel_status": "visible"},
            },
            "case-absent": {
                "canonical_entity": "土地庙",
                "expected_surfaces": ["土地庙"],
                "diagnostic_alias_surfaces": [],
                "visual_audit": {"pixel_status": "absent"},
            },
        },
    }
    timeline = {
        "structural_gate_passed": True,
        "windows": [
            {
                "virtual_start_sec": 0.0,
                "virtual_end_sec": 10.0,
                "case_ids": ["case-strict"],
            },
            {
                "virtual_start_sec": 20.0,
                "virtual_end_sec": 30.0,
                "case_ids": ["case-near"],
            },
            {
                "virtual_start_sec": 40.0,
                "virtual_end_sec": 50.0,
                "case_ids": ["case-absent"],
            },
        ],
    }
    preflight = {
        "structural_gate_passed": True,
        "case_level": [
            {
                "case_id": "case-strict",
                "pre_admission_target_match_count": 0,
                "category": "reader_miss",
            },
            {
                "case_id": "case-near",
                "pre_admission_target_match_count": 1,
                "category": "target_surface_admitted",
            },
            {
                "case_id": "case-absent",
                "pre_admission_target_match_count": 0,
                "category": "pixel_absent",
            },
        ],
    }
    report = MODULE.build_dense_ocr_audit(
        tracks=(
            _track("沙大郎", 1.0, 2.0, 2),
            _track("赤髯能", 21.0, 22.0),
        ),
        dense_report={
            "structural_gate_passed": True,
            "counts": {
                "frames": 20,
                "views_per_frame": 4,
                "reader_calls": 80,
                "tracks": 2,
                "evidence_records": 2,
            },
            "track_counts": {
                "normalized_surfaces": 2,
                "surfaces_with_multiple_tracks": 0,
            },
            "model_calls": 0,
        },
        dense_manifest={
            "source_commit": "dense-commit",
            "day_test140_accessed": False,
            "week_accessed": False,
        },
        timeline=timeline,
        surface_spec=surface_spec,
        preflight_report=preflight,
        a3_runtime_report={
            "counts": {"selected_frames": 3, "model_calls": 1},
            "gates": {"structural_gate_passed": True},
        },
        track_line_count=2,
        evidence_line_count=2,
        expected_dense_source_commit="dense-commit",
        audit_source_commit="audit-commit",
    )

    assert report["structural_and_promotion_gate_passed"] is True
    assert report["counts"]["pixel_visible_cases"] == 2
    assert report["counts"]["pixel_absent_cases"] == 1
    assert report["counts"]["paddle_strict_hits"] == 1
    assert report["counts"]["paddle_alias_aware_hits"] == 2
    assert report["representation"]["strict_complementarity"] == {
        "a3_only": 1,
        "paddle_only": 1,
    }
    rows = {row["case_id"]: row for row in report["case_level"]}
    assert rows["case-near"]["paddle_outcome"] == "near_miss"
    assert rows["case-absent"]["paddle_outcome"] == "miss"


def test_near_match_reuses_frozen_wp17_threshold() -> None:
    assert MODULE._near_surface_match("沙太尉", ["沙大郎"]) is True
    assert MODULE._near_surface_match("unrelated", ["沙大郎"]) is False
