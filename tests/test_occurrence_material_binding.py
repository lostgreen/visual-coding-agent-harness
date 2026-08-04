from __future__ import annotations

import json
from pathlib import Path

from evaluate.mmlifelong.metrics import (
    candidate_clue_recall,
    occurrence_candidate_recall,
)
from vcah.interactive_agents import _candidate_binding_for_task
from vcah.multiround import InvestigationTask, validate_occurrence_material_binding
from vcah.runtime_metrics import agent_run_metrics


def _locator_row() -> dict[str, object]:
    return {
        "attempt_id": "attempt_locator",
        "sampling_config": {
            "mode": "search_caption",
            "hits": [
                {
                    "passage_id": "passage_1",
                    "range": [100.0, 110.0],
                }
            ],
            "occurrence_set": {
                "candidates": [
                    {
                        "occurrence_id": "occ_1",
                        "time_range": [100.0, 110.0],
                        "source_video_ids": ["video_1"],
                        "segment_ids": ["seg_1"],
                        "passage_ids": ["passage_1"],
                    }
                ]
            },
        },
    }


def _bound_task(**overrides: object) -> InvestigationTask:
    values = {
        "query_id": "inspect_occurrence",
        "goal": "Inspect the selected occurrence.",
        "segment_id": "seg_1",
        "time_range": (99.0, 112.0),
        "source_video_ids": ("video_1",),
        "inspection_mode": "window",
        "locator_attempt_id": "attempt_locator",
        "occurrence_id": "occ_1",
        "temporal_scope_id": "scope_1",
        "evidence_kind": "transient_event",
    }
    values.update(overrides)
    return InvestigationTask(**values)


def test_occurrence_binding_validates_locator_source_and_range() -> None:
    assert validate_occurrence_material_binding(
        _bound_task(),
        (_locator_row(),),
        temporal_scope_ids=("scope_1",),
    ) is None

    missing = validate_occurrence_material_binding(
        _bound_task(occurrence_id="occ_missing"),
        (_locator_row(),),
        temporal_scope_ids=("scope_1",),
    )
    assert missing is not None
    assert missing["code"] == "occurrence_not_in_locator"

    source_mismatch = validate_occurrence_material_binding(
        _bound_task(source_video_ids=("video_2",)),
        (_locator_row(),),
        temporal_scope_ids=("scope_1",),
    )
    assert source_mismatch is not None
    assert source_mismatch["code"] == "occurrence_source_mismatch"

    wrong_temporal_selection = validate_occurrence_material_binding(
        _bound_task(),
        (_locator_row(),),
        temporal_scope_ids=("scope_1",),
        temporal_scope_resolutions={
            "scope_1": {
                "resolved": True,
                "selected_occurrence_ids": ["occ_other"],
            }
        },
    )
    assert wrong_temporal_selection is not None
    assert wrong_temporal_selection["code"] == "occurrence_outside_temporal_selection"


def test_visual_attempt_persists_candidate_binding(tmp_path: Path) -> None:
    path = tmp_path / "observation_log.jsonl"
    path.write_text(json.dumps(_locator_row()) + "\n", encoding="utf-8")

    binding = _candidate_binding_for_task(_bound_task(), path)

    assert binding == {
        "locator_attempt_id": "attempt_locator",
        "occurrence_id": "occ_1",
        "candidate_range": [100.0, 110.0],
        "passage_ids": ["passage_1"],
        "source_video_ids": ["video_1"],
        "segment_ids": ["seg_1"],
    }


def test_candidate_recall_metrics_remain_evaluation_side() -> None:
    rows = (_locator_row(),)

    assert candidate_clue_recall(rows, ((104.0, 106.0),)) == 1.0
    assert occurrence_candidate_recall(rows, ((104.0, 106.0),)) == 1.0
    assert candidate_clue_recall(rows, ((500.0, 501.0),)) == 0.0
    assert occurrence_candidate_recall(rows, ((500.0, 501.0),)) == 0.0


def test_occurrence_and_temporal_runtime_metrics_use_logged_bindings() -> None:
    trace = (
        {
            "type": "task_request",
            "ledger_id": "ledger_1",
            "task": {"occurrence_id": "occ_1"},
        },
        {
            "type": "task_outcome",
            "ledger_id": "ledger_1",
            "status": "executed",
        },
        {
            "type": "answer_outcome",
            "temporal_scope_summary": {
                "temporal_scope_count": 1,
                "resolved_temporal_scope_count": 1,
                "temporal_scope_resolved_rate": 1.0,
            },
        },
    )
    observation_rows = (
        {
            "attempt_id": "attempt_visual",
            "modality": "visual",
            "sampling_config": {
                "candidate_binding": {
                    "locator_attempt_id": "attempt_locator",
                    "occurrence_id": "occ_1",
                }
            },
        },
    )

    metrics = agent_run_metrics(
        trace,
        observation_rows,
        answer_present=True,
        reference_valid=True,
    )
    assert metrics["occurrence_binding_rate"] == 1.0
    assert metrics["temporal_scope_resolved_rate"] == 1.0
