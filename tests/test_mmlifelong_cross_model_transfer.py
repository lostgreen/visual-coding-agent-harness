from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_module() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "analyze_mmlifelong_cross_model_transfer.py"
    )
    spec = importlib.util.spec_from_file_location("mmlifelong_cross_transfer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_module()


def _row(
    stack: str,
    arm: str,
    case_id: str,
    score: float,
    clue_count: int,
) -> dict[str, Any]:
    models = (
        {"reasoner": "base-model", "investigator": "base-model"}
        if stack == "base"
        else {"reasoner": "alt-reasoner", "investigator": "base-model"}
    )
    return {
        "stack": stack,
        "arm": arm,
        "case_id": case_id,
        "question_type": "Counting" if clue_count > 1 else "Event Recognition",
        "clue_count": clue_count,
        "clue_duration_sec": 10.0,
        "score": score,
        "parse_status": "parsed",
        "judge_model": "gpt-5.5",
        "frozen_config": {
            "controller_mode": "frozen_baseline",
            "models": models,
        },
        "audit": {
            "caption_config_digest": "caption",
            "intervention_digest": f"intervention-{case_id}",
            "natural_candidate_count": 12,
            "natural_clue_recall": 0.0,
            "final_clue_recall": 1.0,
            "final_candidate_count": 12,
            "candidate_passage_ids": [f"p{index}" for index in range(12)],
            "candidate_intervals": [[index, index + 1] for index in range(12)],
            "shuffle_seed_digest": f"shuffle-{case_id}",
        },
        "selected_candidate_request_recall": 1.0,
        "selected_candidate_inspection_recall": 1.0,
        "anchor_request_recall": 1.0,
        "anchor_inspection_recall": 1.0,
        "clue_center_visual_recall": 1.0,
        "visual_frames": 8,
        "vlm_calls": 2,
        "occurrence_candidate_count": 3,
    }


def test_cross_model_report_computes_paired_deltas_and_dod() -> None:
    manifest = {
        "selection_is_outcome_independent": True,
        "cases": [{"case_id": "single"}, {"case_id": "multi"}],
    }
    scores = {
        "base": {
            "o1": (0.0, 0.0),
            "o1.5": (0.5, 0.0),
            "o1.75": (1.0, 0.0),
        },
        "r1_i0": {
            "o1": (0.0, 0.0),
            "o1.5": (1.0, 0.5),
            "o1.75": (1.0, 1.0),
        },
    }
    rows = []
    for stack in ("base", "r1_i0"):
        for arm in ANALYSIS.ARMS:
            rows.extend(
                (
                    _row(stack, arm, "single", scores[stack][arm][0], 1),
                    _row(stack, arm, "multi", scores[stack][arm][1], 2),
                )
            )

    report = ANALYSIS.build_report(
        rows,
        manifest=manifest,
        bootstrap_samples=100,
        seed=7,
    )

    deltas = {
        (row["stack"], row["effect"]): row["mean_score_delta"]
        for row in report["paired_deltas"]
    }
    dod = {
        (row["stack"], row["effect"]): row["mean_score_delta"]
        for row in report["difference_of_differences"]
    }
    assert report["runtime_gate_passed"] is True
    assert report["gate_passed"] is True
    assert report["model_factor_checks"] == {"r1_i0": True}
    assert deltas[("base", "selection")] == 0.25
    assert deltas[("r1_i0", "selection")] == 0.75
    assert deltas[("base", "anchor")] == 0.25
    assert deltas[("r1_i0", "anchor")] == 0.25
    assert dod[("r1_i0", "selection_dod_vs_base")] == 0.5
    assert dod[("r1_i0", "anchor_dod_vs_base")] == 0.0
    assert report["acceptance"]["r1_i0"]["passed"] is True


def test_cross_model_gate_rejects_nonisolated_model_change() -> None:
    manifest = {
        "selection_is_outcome_independent": True,
        "cases": [{"case_id": "single"}],
    }
    rows = [
        _row(stack, arm, "single", 0.0, 1)
        for stack in ("base", "r1_i0")
        for arm in ANALYSIS.ARMS
    ]
    for row in rows:
        if row["stack"] == "r1_i0":
            row["frozen_config"]["models"]["investigator"] = "also-changed"

    report = ANALYSIS.build_report(rows, manifest=manifest, bootstrap_samples=10)

    assert report["runtime_gate_passed"] is False
    assert report["runtime_gate_checks"]["model_factor_isolation_valid"] is False
