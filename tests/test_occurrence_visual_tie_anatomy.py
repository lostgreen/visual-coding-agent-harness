from __future__ import annotations

from vcah.occurrence_visual_geometry import VISUAL_GEOMETRY_CONTRACT
from vcah.occurrence_visual_tie_anatomy import (
    build_visual_tie_anatomy_report,
    classify_tie,
)


def _row(case_id: str, index: int, geometry: str, kind: str = "identity") -> dict:
    return {
        "pair_group_id": f"{case_id}-group-{index}",
        "case_id": case_id,
        "constraint_id": f"{case_id}-constraint-{index}",
        "constraint_type": kind,
        "support_geometry": geometry,
    }


def _geometry_report(rows: list[dict]) -> dict:
    outcomes = {"wins": 0, "ties": 0, "losses": 0}
    case_ids = sorted({row["case_id"] for row in rows})
    for case_id in case_ids:
        selected = [row for row in rows if row["case_id"] == case_id]
        matched = sum(
            row["support_geometry"]
            in {"matched_only_supported", "both_supported"}
            for row in selected
        )
        mismatched = sum(
            row["support_geometry"]
            in {"mismatched_only_supported", "both_supported"}
            for row in selected
        )
        key = "wins" if matched > mismatched else "losses" if matched < mismatched else "ties"
        outcomes[key] += 1
    return {
        "contract": VISUAL_GEOMETRY_CONTRACT,
        "structural_gate_passed": True,
        "paired_support_geometry": {"by_constraint": rows},
        "case_level_joint_evidence": outcomes,
    }


def test_tie_classifier_uses_strict_mutually_exclusive_dominance() -> None:
    assert classify_tie({"both_supported": 2}) == "both_supported_dominated"
    assert classify_tie({"neither_supported": 2}) == "neither_supported_dominated"
    assert (
        classify_tie(
            {"matched_only_supported": 1, "mismatched_only_supported": 1}
        )
        == "mixed_cancellation"
    )
    assert (
        classify_tie({"both_supported": 1, "neither_supported": 1})
        == "balanced_other"
    )


def test_tie_anatomy_proposes_comparative_only_for_dominant_shared_ties() -> None:
    rows = []
    for case_id in ("tie-1", "tie-2", "tie-3"):
        rows.extend(
            [
                _row(case_id, 1, "both_supported"),
                _row(case_id, 2, "both_supported", "event"),
                _row(case_id, 3, "neither_supported"),
            ]
        )
    rows.extend(
        [
            _row("tie-4", 1, "matched_only_supported"),
            _row("tie-4", 2, "mismatched_only_supported"),
            _row("loss-1", 1, "mismatched_only_supported"),
            _row("win-1", 1, "matched_only_supported"),
        ]
    )
    report = build_visual_tie_anatomy_report(
        _geometry_report(rows),
        expected_cases=6,
        expected_pairs=len(rows),
    )
    assert report["structural_gate_passed"] is True
    assert report["residual_anatomy"]["tie_class_counts"] == {
        "both_supported_dominated": 3,
        "neither_supported_dominated": 0,
        "mixed_cancellation": 1,
        "balanced_other": 0,
    }
    assert report["decision"] == "PROPOSE_WP16_A_COMPARATIVE_DISCRIMINATIVE_PROBE"
    identity = next(
        row
        for row in report["constraint_type_discriminativeness"]
        if row["constraint_type"] == "identity"
    )
    assert identity["matched_only_rate"] is not None
    assert report["branch_thresholds"]["type_weights_learned"] is False


def test_tie_anatomy_stops_on_mixed_ties() -> None:
    rows = [
        _row("tie-1", 1, "both_supported"),
        _row("tie-2", 1, "neither_supported"),
        _row("tie-3", 1, "matched_only_supported"),
        _row("tie-3", 2, "mismatched_only_supported"),
        _row("tie-4", 1, "both_supported"),
        _row("tie-4", 2, "neither_supported"),
    ]
    report = build_visual_tie_anatomy_report(
        _geometry_report(rows),
        expected_cases=4,
        expected_pairs=len(rows),
    )
    assert report["structural_gate_passed"] is True
    assert report["decision"] == "STOP_CURRENT_VISUAL_INTEGRATION_LINE"
