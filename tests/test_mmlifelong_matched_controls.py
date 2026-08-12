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
        / "analyze_mmlifelong_matched_controls.py"
    )
    spec = importlib.util.spec_from_file_location("matched_controls", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_module()


def _row(arm: str, case_id: str, score: float) -> dict[str, Any]:
    forced = arm == "o1.75-forced"
    o175 = arm.startswith("o1.75")
    exact = arm.startswith("o2")
    policy = "force_if_requested" if forced else "agent_controlled"
    audit = {
        "applied": True,
        "caption_config_digest": "caption-digest",
        "intervention_digest": f"intervention-{case_id}",
        "natural_candidate_count": 12,
        "natural_clue_recall": 0.25,
        "final_clue_recall": 1.0,
        "candidate_passage_ids": (["p2"] if exact else [f"p{i}" for i in range(12)]),
        "candidate_intervals": ([[20.0, 22.0]] if exact else [[i, i + 1] for i in range(12)]),
        "shuffle_seed_digest": "same-shuffle",
        "guidance_type": (
            "selected_coarse_candidates_with_point_anchors"
            if o175
            else "exact_locators_with_point_anchors"
            if arm == "o2-center"
            else ""
        ),
        "exact_boundaries_visible": exact,
        "exact_locator_count": 1 if exact else 0,
        "selected_candidate_ranks": [1],
        "selected_candidate_passage_ids": ["p2"],
        "selected_candidate_intervals": [[20.0, 22.0]],
        "point_anchor_candidate_ranks": [1] if arm != "o2" else [],
        "point_anchor_candidate_passage_ids": ["p2"] if arm != "o2" else [],
        "anchor_timestamps_sec": [21.0] if arm != "o2" else [],
        "anchor_count": 1 if arm != "o2" else 0,
        "anchor_execution_policy": policy if arm != "o2" else "",
    }
    return {
        "arm": arm,
        "case_id": case_id,
        "question_type": "Event Tracking",
        "clue_count": 1,
        "score": score,
        "parse_status": "parsed",
        "judge_model": "gpt-5.5",
        "audit": audit,
        "frozen_config": {
            "controller_mode": "frozen_baseline",
            "anchor_execution_policy": policy,
        },
        "clue_visual_recall": 1.0,
        "clue_center_visual_recall": 1.0,
        "anchor_request_recall": None if arm == "o2" else 1.0,
        "anchor_inspection_recall": None if arm == "o2" else 1.0,
        "anchor_frame_recall": None if arm == "o2" else 1.0,
        "anchor_timestamps_sec": [21.0] if arm != "o2" else [],
        "anchor_requested_count": 0 if arm == "o2" else 1,
        "anchor_inspected_count": 0 if arm == "o2" else 1,
        "anchor_exact_frame_count": 0 if arm == "o2" else 1,
        "forced_anchor_count": 1 if forced else 0,
        "anchor_attachment_failure_count": 0,
        "visual_frames": 8,
        "visual_window_count": 1,
    }


def test_matched_control_report_uses_absolute_paired_deltas() -> None:
    scores = {
        "o1.75": (0.0, 1.0),
        "o1.75-forced": (1.0, 1.0),
        "o2": (0.0, 0.0),
        "o2-center": (0.0, 1.0),
    }
    rows = [
        _row(arm, f"case-{index}", score)
        for arm in ANALYSIS.ARMS
        for index, score in enumerate(scores[arm])
    ]

    report = ANALYSIS.build_report(
        rows,
        expected_cases=2,
        bootstrap_samples=100,
        seed=7,
    )

    comparisons = {row["comparison"]: row for row in report["paired_comparisons"]}
    assert report["runtime_gate_passed"] is True
    assert report["gate_passed"] is True
    assert comparisons["o1.75-forced-o1.75"]["mean_score_delta"] == 0.5
    assert comparisons["o2-center-o2"]["mean_score_delta"] == 0.5
    assert "oracle_gap_recovery" not in report
    forced = next(row for row in report["arms"] if row["arm"] == "o1.75-forced")
    assert forced["execution_fidelity"] == 1.0
    assert forced["exact_frame_execution_fidelity"] == 1.0


def test_matched_control_gate_rejects_missing_forced_anchor_frame() -> None:
    rows = [_row(arm, "case-0", 0.0) for arm in ANALYSIS.ARMS]
    forced = next(row for row in rows if row["arm"] == "o1.75-forced")
    forced["anchor_exact_frame_count"] = 0

    report = ANALYSIS.build_report(rows, expected_cases=1, bootstrap_samples=10)

    assert report["runtime_gate_passed"] is False
    assert report["runtime_gate_checks"]["forced_requested_anchors_executed"] is False


def test_exact_anchor_execution_metrics_use_structured_frame_times(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "modality": "caption_search",
            "sampling_config": {
                "oracle_guidance": {
                    "anchor_timestamps_sec": [10.0, 20.0],
                    "selected_candidates": [],
                }
            },
        },
        {
            "modality": "visual",
            "requested_range": [5.0, 12.0],
            "inspected_ranges": [[9.5, 10.5]],
            "frame_times": [10.0],
            "sampling_config": {"forced_anchor_timestamps_sec": [10.0]},
        },
    ]
    (tmp_path / "observation_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics = ANALYSIS.LADDER._trajectory_metrics(tmp_path, ((9.0, 11.0),))

    assert metrics["execution_fidelity"] == 1.0
    assert metrics["exact_frame_execution_fidelity"] == 1.0
    assert metrics["anchor_requested_count"] == 1
    assert metrics["anchor_exact_frame_count"] == 1


def test_exact_anchor_execution_metrics_accept_legacy_frame_time_field(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "modality": "caption_search",
            "sampling_config": {
                "oracle_guidance": {
                    "anchor_timestamps_sec": [10.0],
                    "selected_candidates": [],
                }
            },
        },
        {
            "modality": "visual",
            "requested_range": [5.0, 12.0],
            "inspected_ranges": [[9.5, 10.5]],
            "attached_frame_times": [10.0],
            "sampling_config": {"forced_anchor_timestamps_sec": [10.0]},
        },
    ]
    (tmp_path / "observation_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics = ANALYSIS.LADDER._trajectory_metrics(tmp_path, ((9.0, 11.0),))

    assert metrics["anchor_exact_frame_count"] == 1


def test_matched_control_cli_accepts_case_filters() -> None:
    parser_source = Path(ANALYSIS.__file__).read_text(encoding="utf-8")

    assert 'parser.add_argument("--case-id", action="append")' in parser_source
    assert "case_ids=(frozenset(args.case_id) if args.case_id else None)" in parser_source


def test_o2_center_gate_uses_anchor_rank_after_candidate_shuffle() -> None:
    rows = [_row(arm, "case-0", 0.0) for arm in ANALYSIS.ARMS]
    center = next(row for row in rows if row["arm"] == "o2-center")
    center["clue_count"] = 2
    center["audit"]["candidate_passage_ids"] = ["second", "first"]
    center["audit"]["candidate_intervals"] = [[30.0, 34.0], [10.0, 12.0]]
    center["audit"]["exact_locator_count"] = 2
    center["audit"]["anchor_count"] = 2
    center["audit"]["anchor_timestamps_sec"] = [11.0, 32.0]
    center["audit"]["point_anchor_candidate_ranks"] = [2, 1]
    center["audit"]["point_anchor_candidate_passage_ids"] = ["first", "second"]

    assert ANALYSIS._o2_center_guidance_valid(
        {arm: {"case-0": next(row for row in rows if row["arm"] == arm)} for arm in ANALYSIS.ARMS},
        {"case-0"},
    ) is True
