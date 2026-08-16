from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analyze_mmlifelong_occurrence_repeated_controls.py"
)
SPEC = importlib.util.spec_from_file_location("repeated_controls", MODULE_PATH)
assert SPEC and SPEC.loader
REPEATED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPEATED)


def _row(case_id: str, *, exact: bool, complete: bool) -> dict:
    return {
        "case_id": case_id,
        "score": 1.0 if exact else 0.0,
        "exact_correct": exact,
        "verified_correct": exact,
        "grounded_correct_ref300": exact,
        "grounded_correct_bound_visual": exact,
        "candidate_recall_resolved_set": True,
        "osa_eligible": True,
        "osa_strict": exact,
        "osa_any": exact,
        "final_resolution": "selected",
        "selected_locator_usage_rate": 1.0 if exact else 0.0,
        "bound_visual_clue_recall": 1.0 if exact else 0.0,
        "visual_frames": 10 if exact else 5,
        "vlm_calls": 2 if exact else 1,
        "semantic_rounds_used": 4,
        "frozen_replay_full_consumption": complete,
    }


def test_repeated_controls_use_common_complete_intersection() -> None:
    runs = {
        "wp6_a3": (
            _row("c1", exact=True, complete=True),
            _row("c2", exact=True, complete=False),
        ),
        "wp8_a3": (
            _row("c1", exact=False, complete=True),
            _row("c2", exact=True, complete=True),
        ),
        "wp6_a2": (
            _row("c1", exact=False, complete=True),
            _row("c2", exact=False, complete=True),
        ),
    }

    report = REPEATED.build_repeated_control_report(
        runs,
        repeat_labels=("wp6_a3", "wp8_a3"),
        effect_pair=("wp6_a3", "wp6_a2"),
    )

    assert report["aligned_case_count"] == 2
    assert report["common_complete_case_count"] == 1
    assert report["excluded_from_common_complete"]["wp6_a3"] == ["c2"]
    assert report["metrics_by_run"]["wp6_a3"]["exact_correct_rate"] == 1.0
    assert report["metrics_by_run"]["wp8_a3"]["exact_correct_rate"] == 0.0
    exact = report["effect_to_variation"]["exact_correct_rate"]
    assert exact["delta"] == 1.0
    assert exact["repeat_range"] == 1.0
    assert exact["absolute_effect_to_variation_ratio"] == 1.0
