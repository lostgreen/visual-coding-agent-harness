from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mger_phase3_cases10_74f012d"
REQUIRED_CASE_SECTIONS = {
    "prediction",
    "runtime_summary",
    "evaluation_summary",
    "reasoner_decisions",
    "workspace_rejection_events",
    "task_requests",
    "task_outcomes",
    "caption_query_lists",
    "occurrence_candidates",
    "sampling_manifests",
    "frame_sampling_manifest",
    "observation_interpretations",
    "final_validation",
}


def _case_rows() -> tuple[Mapping[str, Any], ...]:
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((FIXTURE_ROOT / "cases").glob("*.json"))
    )


def test_phase3_fixture_preserves_required_diagnostics() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    rows = _case_rows()

    assert len(rows) == 10
    assert [row["case_id"] for row in rows] == manifest["case_ids"]
    assert manifest["source_bundle_sha256"] == (
        "aadfa8d2dfa86242cee37d715ed0e6ba1d06ea2da023d65a1dd5eb0b7c1b639f"
    )
    for row in rows:
        assert REQUIRED_CASE_SECTIONS <= row.keys()
        assert row["reasoner_decisions"]
        assert row["task_requests"]
        assert row["task_outcomes"]
        assert row["caption_query_lists"]
        assert row["sampling_manifests"]
        assert row["observation_interpretations"]
        assert row["final_validation"]


def test_phase3_fixture_recomputes_ten_case_baseline() -> None:
    rows = _case_rows()
    correct = [row["evaluation_summary"]["answer"]["score"] == 1.0 for row in rows]
    answer_present = [bool(row["runtime_summary"]["answer_present"]) for row in rows]
    reference_valid = [bool(row["runtime_summary"]["reference_valid"]) for row in rows]
    frame_counts = [
        int(row["runtime_summary"]["runtime_metrics"]["visual_frames_inspected"])
        for row in rows
    ]
    occurrence_counts = [
        len(
            {
                str(candidate["occurrence_id"])
                for candidate in row["occurrence_candidates"]
                if candidate.get("occurrence_id")
            }
        )
        for row in rows
    ]

    assert sum(correct) == 4
    assert sum(answer_present) == 9
    assert sum(reference_valid) == 5
    assert sum(not is_correct and is_valid for is_correct, is_valid in zip(correct, reference_valid)) == 3
    assert sum(is_correct and not is_valid for is_correct, is_valid in zip(correct, reference_valid)) == 2
    assert sum(is_correct and is_valid for is_correct, is_valid in zip(correct, reference_valid)) == 2
    assert sum(frame_counts) / len(frame_counts) == pytest.approx(55.1)
    assert sum(occurrence_counts) == 62
    assert {
        row["evaluation_summary"]["answer"]["official_judge_model_match"]
        for row in rows
    } == {False}
