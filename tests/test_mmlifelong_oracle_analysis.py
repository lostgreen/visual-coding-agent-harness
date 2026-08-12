from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _load_module() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "analyze_mmlifelong_oracle_ladder.py"
    )
    spec = importlib.util.spec_from_file_location("oracle_analysis", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_module()


def _row(arm: str, case_id: str, score: float) -> dict[str, Any]:
    audit = None
    if arm != "o0":
        audit = {
            "applied": True,
            "caption_config_digest": "caption-digest",
            "intervention_digest": f"intervention-{case_id}",
            "natural_clue_recall": 0.0,
            "final_clue_recall": (
                1.0 if arm in {"o1", "o1.5", "o1.75", "o2"} else 0.0
            ),
            "natural_candidate_count": 12,
            "final_candidate_count": 12 if arm != "o2" else 1,
            "injected_candidate_count": (
                1 if arm in {"o1", "o1.5", "o1.75"} else 0
            ),
            "exact_locator_count": 1 if arm == "o2" else 0,
            "candidate_passage_ids": [f"p{index}" for index in range(12)],
            "candidate_intervals": [[index, index + 1] for index in range(12)],
            "shuffle_seed_digest": "same-shuffle",
        }
        if arm in {"o1.5", "o1.75"}:
            audit.update(
                {
                    "guidance_type": (
                        "selected_coarse_candidates"
                        if arm == "o1.5"
                        else "selected_coarse_candidates_with_point_anchors"
                    ),
                    "exact_boundaries_visible": False,
                    "selected_candidate_count": 1,
                    "selected_candidate_ranks": [3],
                    "selected_candidate_passage_ids": ["p2"],
                    "selected_candidate_intervals": [[2, 3]],
                    "selected_candidate_clue_recall": 1.0,
                    "anchor_count": 0 if arm == "o1.5" else 1,
                    "point_anchor_candidate_ranks": [] if arm == "o1.5" else [3],
                    "point_anchor_candidate_passage_ids": (
                        [] if arm == "o1.5" else ["p2"]
                    ),
                }
            )
    return {
        "arm": arm,
        "case_id": case_id,
        "question_type": "Counting",
        "clue_count": 1,
        "clue_duration_sec": 10.0,
        "score": score,
        "parse_status": "parsed",
        "judge_model": "gpt-5.5",
        "official_judge_model_match": False,
        "ref_300": 1.0,
        "answer_rate": 1.0,
        "reference_valid_rate": 1.0,
        "visual_frames": 10,
        "audit": audit,
        "frozen_config": {"controller_mode": "frozen_baseline"},
    }


def test_report_preserves_causal_control_and_paired_deltas() -> None:
    scores = {
        "o0": (0.0, 0.0),
        "c0": (0.0, 0.5),
        "o1": (0.5, 0.5),
        "o1.5": (0.5, 1.0),
        "o1.75": (1.0, 1.0),
        "o2": (1.0, 1.0),
    }
    rows = tuple(
        _row(arm, f"case-{index}", score)
        for arm in ANALYSIS.ARMS
        for index, score in enumerate(scores[arm])
    )

    report = ANALYSIS.build_report(
        rows,
        expected_cases=2,
        bootstrap_samples=100,
        seed=7,
    )

    comparisons = {
        row["comparison"]: row for row in report["paired_comparisons"]
    }
    assert report["gate_passed"] is True
    assert report["runtime_gate_passed"] is True
    assert report["diagnostic_only"] is True
    assert comparisons["o1-c0"]["mean_score_delta"] == 0.25
    assert comparisons["o1.5-o1"]["mean_score_delta"] == 0.25
    assert comparisons["o1.75-o1.5"]["mean_score_delta"] == 0.25
    assert comparisons["o2-o1.75"]["mean_score_delta"] == 0.0
    assert comparisons["o2-o1"]["mean_score_delta"] == 0.5
    ogr = {row["arm"]: row for row in report["oracle_gap_recovery"]}
    assert ogr["o1.5"]["mean_score_ogr"] == 0.5
    assert ogr["o1.75"]["mean_score_ogr"] == 1.0
    assert len(report["case_matrix"]) == 2


def test_gate_rejects_incomplete_oracle_recall() -> None:
    rows = [
        _row(arm, "case-0", 0.0)
        for arm in ANALYSIS.ARMS
    ]
    rows[-1]["audit"]["final_clue_recall"] = 0.0

    report = ANALYSIS.build_report(rows, expected_cases=1, bootstrap_samples=10)

    assert report["gate_passed"] is False
    assert (
        report["gate_checks"]["candidate_oracle_arms_full_clue_recall"]
        is False
    )


def test_runtime_gate_rejects_o1_family_candidate_pool_drift() -> None:
    rows = [_row(arm, "case-0", 0.0) for arm in ANALYSIS.ARMS]
    rows[ANALYSIS.ARMS.index("o1.75")]["audit"]["candidate_passage_ids"][0] = "drift"

    report = ANALYSIS.build_report(rows, expected_cases=1, bootstrap_samples=10)

    assert report["runtime_gate_passed"] is False
    assert (
        report["runtime_gate_checks"]["o1_family_candidate_pools_identical"]
        is False
    )


def test_runtime_gate_rejects_cross_arm_natural_retrieval_drift() -> None:
    rows = [_row(arm, "case-0", 0.0) for arm in ANALYSIS.ARMS]
    rows[-1]["audit"]["natural_candidate_count"] = 11

    report = ANALYSIS.build_report(rows, expected_cases=1, bootstrap_samples=10)

    assert report["runtime_gate_passed"] is False
    assert (
        report["runtime_gate_checks"]["natural_caption_retrieval_aligned"]
        is False
    )


def test_runtime_gate_rejects_frozen_config_drift() -> None:
    rows = [_row(arm, "case-0", 0.0) for arm in ANALYSIS.ARMS]
    rows[-1]["frozen_config"] = {"controller_mode": "mger"}

    report = ANALYSIS.build_report(rows, expected_cases=1, bootstrap_samples=10)

    assert report["runtime_gate_passed"] is False
    assert report["runtime_gate_checks"]["frozen_configs_aligned"] is False


def test_trajectory_metrics_measure_guidance_consumption_without_model_text(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "modality": "caption_search",
            "sampling_config": {
                "mode": "search_caption",
                "oracle_guidance": {
                    "selected_candidates": [
                        {"inspection_range": [100.0, 120.0]}
                    ],
                    "anchor_timestamps_sec": [110.0],
                },
            },
        },
        {
            "modality": "visual",
            "sampling_config": {"mode": "window"},
            "requested_range": [108.0, 112.0],
            "inspected_ranges": [[108.0, 112.0]],
        },
        {
            "modality": "visual",
            "sampling_config": {"mode": "window"},
            "requested_range": [200.0, 240.0],
            "inspected_ranges": [[200.0, 240.0]],
        },
    ]
    (tmp_path / "observation_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics = ANALYSIS._trajectory_metrics(
        tmp_path,
        ((109.0, 111.0), (300.0, 310.0)),
    )

    assert metrics["visual_window_count"] == 2
    assert metrics["mean_visual_window_sec"] == 22.0
    assert metrics["clue_visual_recall"] == 0.5
    assert metrics["clue_center_visual_recall"] == 0.5
    assert metrics["selected_candidate_inspection_recall"] == 1.0
    assert metrics["selected_candidate_request_recall"] == 1.0
    assert metrics["anchor_inspection_recall"] == 1.0
    assert metrics["anchor_request_recall"] == 1.0
    assert metrics["mean_inspected_candidate_window_sec"] == 4.0
    assert metrics["mean_inspected_anchor_window_sec"] == 4.0
