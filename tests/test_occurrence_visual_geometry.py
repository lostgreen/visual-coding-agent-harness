from __future__ import annotations

from vcah.occurrence_visual_geometry import build_visual_geometry_report
from vcah.occurrence_visual_probe import VISUAL_PROBE_CONTRACT


def _fixture() -> tuple[dict, list[dict], dict[str, dict]]:
    cases = []
    results = []
    evaluations = {}
    verdict_pairs = (
        ("supported", "unknown", True),
        ("supported", "supported", True),
        ("unknown", "supported", False),
    )
    for index, (matched_verdict, mismatched_verdict, covered) in enumerate(
        verdict_pairs, start=1
    ):
        case_id = f"case-{index}"
        group_id = f"group-{index}"
        constraint_id = f"constraint-{index}"
        windows = []
        items = []
        for pair_kind, timestamp in (
            ("matched", 12.0 if covered else 18.0),
            ("mismatched", 50.0),
            ("null", 80.0),
        ):
            observation_id = f"obs-{index}-{pair_kind}"
            windows.append(
                {
                    "visual_observation_id": observation_id,
                    "pair_kind": pair_kind,
                    "frames": [
                        {
                            "frame_id": f"frame-{index}-{pair_kind}",
                            "virtual_time_sec": timestamp,
                        }
                    ],
                }
            )
            item_id = f"item-{index}-{pair_kind}"
            items.append(
                {
                    "item_id": item_id,
                    "pair_group_id": group_id,
                    "case_id": case_id,
                    "pair_kind": pair_kind,
                    "constraint_id": constraint_id,
                    "constraint_type": "identity",
                    "visual_observation_id": observation_id,
                }
            )
            verdict = (
                matched_verdict
                if pair_kind == "matched"
                else mismatched_verdict
                if pair_kind == "mismatched"
                else "unknown"
            )
            results.append(
                {"item_id": item_id, "status": "success", "verdict": verdict}
            )
        cases.append(
            {
                "case_id": case_id,
                "eligible": True,
                "windows": windows,
                "items": items,
            }
        )
        evaluations[case_id] = {
            "case_id": case_id,
            "clue_intervals": [[10.0, 14.0]],
        }
    return {"contract": VISUAL_PROBE_CONTRACT, "cases": cases}, results, evaluations


def test_visual_geometry_separates_pairs_joint_scores_and_coverage() -> None:
    manifest, results, evaluations = _fixture()
    report = build_visual_geometry_report(
        manifest,
        results,
        evaluations,
        expected_cases=3,
        expected_items=9,
        joint_min_wins=1,
        joint_max_losses=1,
        coverage_min_cases_per_stratum=1,
        coverage_min_support_gap=0.3,
    )

    assert report["structural_gate_passed"] is True
    counts = report["paired_support_geometry"]["counts"]
    assert counts == {
        "matched_only_supported": 1,
        "both_supported": 1,
        "mismatched_only_supported": 1,
        "neither_supported": 0,
    }
    joint = report["case_level_joint_evidence"]
    assert (joint["wins"], joint["ties"], joint["losses"]) == (1, 1, 1)
    coverage = report["matched_frame_coverage"]
    assert coverage["covered"]["case_count"] == 2
    assert coverage["uncovered"]["case_count"] == 1
    assert coverage["covered_minus_uncovered_supported_rate"] == 1.0
    assert report["decision"] == "PROCEED_TO_BLIND_COMPARATIVE_VISUAL_PROBE"


def test_visual_geometry_stops_when_input_pair_is_incomplete() -> None:
    manifest, results, evaluations = _fixture()
    results.pop()
    report = build_visual_geometry_report(
        manifest,
        results,
        evaluations,
        expected_cases=3,
        expected_items=9,
    )
    assert report["structural_gate_passed"] is False
    assert report["decision"] == "STOP_INVALID_WP15_0_INPUT"


def test_visual_geometry_audits_missing_clues_without_crashing() -> None:
    manifest, results, evaluations = _fixture()
    evaluations["case-1"]["clue_intervals"] = []
    report = build_visual_geometry_report(
        manifest,
        results,
        evaluations,
        expected_cases=3,
        expected_items=9,
    )
    assert report["structural_gate_passed"] is False
    assert report["structural_checks"]["matched_frame_records_complete"] is False
    assert report["decision"] == "STOP_INVALID_WP15_0_INPUT"
