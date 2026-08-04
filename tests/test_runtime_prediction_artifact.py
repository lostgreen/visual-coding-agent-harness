from __future__ import annotations

import json

import pytest

from benchmarks.mmlifelong.runner import assert_runtime_artifact, prediction_artifact
from benchmarks.schema import RuntimeQuestion


def test_prediction_artifact_contains_runtime_data_only() -> None:
    question = RuntimeQuestion(
        case_id="mmlifelong-game-test-0031",
        question="What text appeared?",
        runtime_metadata={"source_subset": "day", "source_index": 31},
    )

    payload = prediction_artifact(
        question,
        answer="Old Rattle-Drum",
        selected_option="",
        supporting_intervals=((10.0, 12.0),),
        supporting_attempt_ids=("attempt-1",),
        answer_present=True,
        duration_sec=100.0,
    )
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "RuntimePredictionV1"
    assert payload["supporting_attempt_ids"] == ["attempt-1"]
    assert payload["runtime_metadata"]["duration_sec"] == 100.0
    for forbidden in (
        "gold",
        "reference_answer",
        "clue_intervals",
        "target_virtual_interval",
        "accuracy_score",
    ):
        assert forbidden not in encoded


def test_runtime_question_rejects_nested_evaluator_fields() -> None:
    with pytest.raises(ValueError, match="evaluator-only"):
        RuntimeQuestion(
            case_id="case",
            question="question",
            runtime_metadata={"nested": {"gold_answer": "leak"}},
        )


def test_runtime_prediction_rejects_nested_target_fields() -> None:
    with pytest.raises(ValueError, match="evaluator-only"):
        assert_runtime_artifact(
            {"runtime_metadata": {"nested": {"target_virtual_interval": [1, 2]}}}
        )
