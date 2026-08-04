from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_summary_module() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "evaluate"
        / "mmlifelong"
        / "summarize.py"
    )
    spec = importlib.util.spec_from_file_location("mmlifelong_summary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_summary_module()


def _load_runner_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "run_mmlifelong_interactive.py"
    spec = importlib.util.spec_from_file_location("mmlifelong_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def test_aggregate_report_groups_by_reproducible_config() -> None:
    rows = (
        {
            "evaluation": {
                "config_digest": "cfg-a",
                "accuracy_score": 1.0,
                "ref": {"Ref@60": 100.0, "Ref@300": 100.0, "Ref@600": 100.0},
                "retrieval": {"ClueRecall@5": 1.0, "AllCluesRecall@5": 1.0},
                "agent": {
                    "answer_rate": 1.0,
                    "reference_valid_rate": 1.0,
                    "rounds": 3,
                    "caption_searches": 1,
                    "caption_material_attempts": 1,
                    "caption_result_novelty_rate": 1.0,
                    "caption_result_set_reuse_count": 0,
                    "caption_occurrence_candidate_count": 2,
                    "unique_visual_material_attempts": 1,
                    "visual_interpretation_count": 2,
                    "visual_confirmations": 1,
                    "visual_frames_inspected": 4,
                },
            },
            "config": {"caption_index_mode": "hybrid", "answer_policy": "benchmark_best_effort"},
        },
        {
            "evaluation": {
                "config_digest": "cfg-a",
                "accuracy_score": 0.5,
                "ref": {"Ref@60": 0.0, "Ref@300": 100.0, "Ref@600": 100.0},
                "retrieval": {"ClueRecall@5": 1.0, "AllCluesRecall@5": 0.0},
                "agent": {
                    "answer_rate": 1.0,
                    "reference_valid_rate": 0.0,
                    "rounds": 5,
                    "caption_searches": 2,
                    "caption_material_attempts": 1,
                    "caption_result_novelty_rate": 0.5,
                    "caption_result_set_reuse_count": 1,
                    "caption_occurrence_candidate_count": 3,
                    "unique_visual_material_attempts": 2,
                    "visual_interpretation_count": 3,
                    "visual_confirmations": 1,
                    "visual_frames_inspected": 8,
                },
            },
            "config": {"caption_index_mode": "hybrid", "answer_policy": "benchmark_best_effort"},
        },
    )

    aggregates = SUMMARY.aggregate_evaluations(rows)
    markdown = SUMMARY.render_markdown(aggregates)

    assert len(aggregates) == 1
    assert aggregates[0]["case_count"] == 2
    assert aggregates[0]["Acc"] == 0.75
    assert aggregates[0]["Ref@60"] == 50.0
    assert aggregates[0]["reference_valid_rate"] == 0.5
    assert aggregates[0]["caption_result_novelty_rate"] == 0.75
    assert aggregates[0]["avg_occurrence_candidates"] == 2.5
    assert aggregates[0]["avg_unique_visual_material_attempts"] == 1.5
    assert aggregates[0]["avg_visual_interpretations"] == 2.5
    assert "caption_index_mode" in markdown
    assert "caption_result_novelty_rate" in markdown
    assert "hybrid" in markdown


def test_implementation_digest_is_content_bound_and_stable() -> None:
    first = RUNNER._implementation_digest()
    second = RUNNER._implementation_digest()

    assert first == second
    assert len(first) == 64


def test_implementation_digest_includes_caption_occurrence_logic(monkeypatch: Any) -> None:
    hashed_paths: list[Path] = []
    monkeypatch.setattr(
        RUNNER,
        "_file_sha256",
        lambda path: hashed_paths.append(path) or "fixture-digest",
    )

    RUNNER._implementation_digest()

    assert any(path.name == "caption_occurrence.py" for path in hashed_paths)


def test_runtime_runner_has_no_correctness_or_judge_helper() -> None:
    assert not hasattr(RUNNER, "_correctness_outcome")
    assert not hasattr(RUNNER, "judge_free_form_answer")
