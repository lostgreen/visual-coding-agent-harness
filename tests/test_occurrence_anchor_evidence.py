from __future__ import annotations

from vcah.occurrence_anchor_evidence import build_anchor_evidence_report


def _bundle(*, with_context: bool) -> dict:
    members = [
        {
            "passage_id": "anchor",
            "time_range": [10.0, 20.0],
            "role": "seed",
            "context_links": [],
            "evidence_channels_observed": [],
        }
    ]
    if with_context:
        members.append(
            {
                "passage_id": "evidence",
                "time_range": [30.0, 40.0],
                "role": "context",
                "context_links": [
                    {
                        "offset": 1,
                        "same_source_timeline": True,
                    }
                ],
                "evidence_channels_observed": ["caption", "visible_ocr"],
            }
        )
    return {"bundles": [{"rank": 1, "member_passages": members}]}


def test_report_separates_anchor_evidence_and_bound_recall() -> None:
    case = {
        "case_id": "case-a",
        "request": {
            "eligible": True,
            "direction": "after",
            "relation": "after",
            "evidence_channels": ["visible_ocr"],
        },
        "anchor_intervals": [[10.0, 20.0]],
        "evidence_intervals": [[30.0, 40.0]],
        "packets": [
            {
                "variants": {
                    "baseline": {
                        "seed_hit_ids": ["anchor"],
                        "context_hit_count": 0,
                        "bundle_set": _bundle(with_context=False),
                    },
                    "distance_1": {
                        "seed_hit_ids": ["anchor"],
                        "context_hit_count": 1,
                        "bundle_set": _bundle(with_context=True),
                    },
                },
                "scope_diagnostic": {
                    "scope_blocked_semantic_evidence": False,
                    "global_overlap": {"weak_numeric_only": False},
                },
            }
        ],
    }
    report = build_anchor_evidence_report(
        (case,), expected_cases=1, variant_order=("distance_1",)
    )
    assert report["structural_gate_passed"] is True
    assert report["decision"] == "DIAGNOSTIC_MECHANISM_SIGNAL"
    assert report["variants"]["baseline"]["anchor_seed_at_5"]["count"] == 1
    assert report["variants"]["baseline"]["bound_evidence_at_5"]["count"] == 0
    assert report["variants"]["distance_1"]["bound_evidence_at_5"]["count"] == 1


def test_endpoint_miss_is_not_a_structural_failure() -> None:
    case = {
        "case_id": "case-a",
        "request": {
            "eligible": True,
            "direction": "after",
            "relation": "after",
            "evidence_channels": ["visible_ocr"],
        },
        "anchor_intervals": [],
        "evidence_intervals": [[30.0, 40.0]],
        "packets": [
            {
                "variants": {
                    name: {
                        "seed_hit_ids": ["anchor"],
                        "context_hit_count": 0,
                        "bundle_set": _bundle(with_context=False),
                    }
                    for name in ("baseline", "distance_1")
                }
            }
        ],
    }
    report = build_anchor_evidence_report(
        (case,), expected_cases=1, variant_order=("distance_1",)
    )
    assert report["structural_gate_passed"] is True
    assert report["decision"] == "DIAGNOSTIC_NO_SIGNAL"
    assert report["variants"]["distance_1"]["evidence_at_5"]["count"] == 0
