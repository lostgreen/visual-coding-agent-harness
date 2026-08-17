from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "diagnose_mmlifelong_occurrence_sufficiency.py"
)
SPEC = importlib.util.spec_from_file_location("sufficiency_diagnosis", MODULE_PATH)
assert SPEC and SPEC.loader
DIAGNOSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSIS)


def _constraint(
    constraint_type: str,
    statuses: dict[str, str],
    *,
    implicit: tuple[str, ...] = (),
) -> dict:
    return {
        "constraint_id": f"{constraint_type}-1",
        "constraint_type": constraint_type,
        "support": [
            {"occurrence_id": occurrence_id, "status": status}
            for occurrence_id, status in statuses.items()
        ],
        "implicit_unknown_occurrence_ids": list(implicit),
    }


def _event(
    set_id: str,
    constraints: list[dict],
    *,
    verdict: str = "insufficient",
) -> dict:
    return {
        "type": "occurrence_sufficiency_decision",
        "round": 3,
        "set_id": set_id,
        "verdict": verdict,
        "constraints_checked": constraints,
    }


def _case(
    case_id: str,
    set_id: str,
    event: dict,
    *,
    gold_first: bool = True,
) -> dict:
    candidates = [
        {
            "occurrence_id": f"{case_id}-gold",
            "time_range": [10, 20],
            "rank": 1 if gold_first else 2,
        },
        {
            "occurrence_id": f"{case_id}-other",
            "time_range": [40, 50],
            "rank": 2 if gold_first else 1,
        },
    ]
    return {
        "case_id": case_id,
        "events": (event,),
        "candidate_sets": {set_id: tuple(candidates)},
        "clues": ((12.0, 18.0),),
    }


def test_blocker_decomposition_distinguishes_implicit_and_mixed() -> None:
    event = _event(
        "set-1",
        [
            _constraint(
                "identity",
                {"gold": "unknown", "other": "contradicted"},
                implicit=("gold",),
            ),
            _constraint(
                "action",
                {"gold": "supported", "other": "partial"},
            ),
        ],
    )

    classified = DIAGNOSIS.classify_candidate_blockers(event)
    assert classified == {"gold": "implicit_unknown", "other": "mixed"}
    diagnosis = DIAGNOSIS.build_blocker_diagnosis(
        ({"case_id": "c1", "events": (event,)},)
    )
    assert diagnosis["candidate_class_counts"]["implicit_unknown"] == 1
    assert diagnosis["candidate_class_counts"]["mixed"] == 1
    assert diagnosis["decision_class_counts"]["mixed"] == 1
    assert diagnosis["candidate_implicit_unknown_primary_rate"] == 0.5


def test_signed_evidence_declaration_reconstructs_gold_and_non_gold_rows() -> None:
    case_id = "signed"
    set_id = "set-signed"
    gold_id = f"{case_id}-gold"
    other_id = f"{case_id}-other"
    candidates = (
        {"occurrence_id": gold_id, "time_range": [10, 20], "rank": 1},
        {"occurrence_id": other_id, "time_range": [40, 50], "rank": 2},
    )
    trace = (
        {
            "type": "occurrence_evidence_declaration",
            "round": 2,
            "set_id": set_id,
            "scope_occurrence_ids": [gold_id, other_id],
            "constraints": [
                {
                    "constraint_id": "identity",
                    "constraint_type": "identity",
                    "description": "target identity",
                    "supported_candidates": [
                        {
                            "occurrence_id": gold_id,
                            "evidence_passage_ids": ["p1"],
                        }
                    ],
                    "contradicted_candidates": [
                        {
                            "occurrence_id": other_id,
                            "evidence_passage_ids": ["p2"],
                        }
                    ],
                }
            ],
        },
    )

    events = DIAGNOSIS._expanded_sufficiency_events(trace, {set_id: candidates})
    report = DIAGNOSIS.build_signed_evidence_diagnostic(
        (
            {
                "case_id": case_id,
                "events": events,
                "candidate_sets": {set_id: candidates},
                "clues": ((12.0, 18.0),),
            },
        ),
        bootstrap_samples=100,
        seed=7,
    )

    assert events[0]["constraints_checked"][0]["support"] == [
        {
            "occurrence_id": gold_id,
            "evidence_passage_ids": ["p1"],
            "status": "supported",
        },
        {
            "occurrence_id": other_id,
            "evidence_passage_ids": ["p2"],
            "status": "contradicted",
        },
    ]
    assert report["overall"]["gold"]["contradicted_rate"] == 0.0
    assert report["overall"]["non_gold"]["contradicted_rate"] == 1.0


def test_expanded_events_reconstruct_and_validate_implicit_rows() -> None:
    compact = {
        "type": "occurrence_sufficiency_decision",
        "round": 2,
        "occurrence_op_index": 0,
        "set_id": "set-1",
        "verdict": "sufficient",
        "constraints_checked": ["identity-1"],
        "constraint_types": ["identity"],
        "sufficient_occurrence_ids": ["gold"],
        "implicit_unknown_support_count": 1,
    }
    trace = (
        compact,
        {
            "type": "reasoner_decision",
            "round": 2,
            "occurrence_ops_accepted": True,
            "occurrence_ops": [
                {
                    "op": "assess_sufficiency",
                    "set_id": "set-1",
                    "constraints_checked": [
                        _constraint("identity", {"gold": "supported"})
                    ],
                }
            ],
        },
    )
    candidate_sets = {
        "set-1": (
            {"occurrence_id": "gold", "time_range": [10, 20]},
            {"occurrence_id": "other", "time_range": [40, 50]},
        )
    }

    events = DIAGNOSIS._expanded_sufficiency_events(trace, candidate_sets)

    assert len(events) == 1
    constraint = events[0]["constraints_checked"][0]
    assert constraint["implicit_unknown_occurrence_ids"] == ["other"]
    assert constraint["support"][-1] == {
        "occurrence_id": "other",
        "status": "unknown",
        "evidence_passage_ids": [],
    }


def test_expanded_events_fail_closed_on_reconstruction_mismatch() -> None:
    trace = (
        {
            "type": "occurrence_sufficiency_decision",
            "round": 2,
            "occurrence_op_index": 0,
            "set_id": "set-1",
            "verdict": "insufficient",
            "constraints_checked": ["identity-1"],
            "constraint_types": ["identity"],
            "sufficient_occurrence_ids": [],
            "implicit_unknown_support_count": 0,
        },
        {
            "type": "reasoner_decision",
            "round": 2,
            "occurrence_ops_accepted": True,
            "occurrence_ops": [
                {
                    "op": "assess_sufficiency",
                    "set_id": "set-1",
                    "constraints_checked": [
                        _constraint("identity", {"gold": "supported"})
                    ],
                }
            ],
        },
    )
    candidate_sets = {
        "set-1": (
            {"occurrence_id": "gold", "time_range": [10, 20]},
            {"occurrence_id": "other", "time_range": [40, 50]},
        )
    }

    try:
        DIAGNOSIS._expanded_sufficiency_events(trace, candidate_sets)
    except ValueError as error:
        assert "reconstruction mismatch" in str(error)
    else:
        raise AssertionError("expected reconstruction mismatch")


def test_expanded_events_follow_recorded_comparative_aggregation_rule() -> None:
    compact = {
        "type": "occurrence_sufficiency_decision",
        "round": 2,
        "occurrence_op_index": 0,
        "set_id": "set-1",
        "verdict": "sufficient",
        "constraints_checked": ["identity-1", "action-1"],
        "constraint_types": ["identity", "action"],
        "sufficient_occurrence_ids": ["gold"],
        "implicit_unknown_support_count": 3,
        "aggregation_rule": "unique_supported_count_margin",
        "minimum_support_margin": 1,
    }
    trace = (
        compact,
        {
            "type": "reasoner_decision",
            "round": 2,
            "occurrence_ops_accepted": True,
            "occurrence_ops": [
                {
                    "op": "assess_sufficiency",
                    "set_id": "set-1",
                    "constraints_checked": [
                        _constraint("identity", {"gold": "supported"}),
                        _constraint("action", {}),
                    ],
                }
            ],
        },
    )
    candidate_sets = {
        "set-1": (
            {"occurrence_id": "gold", "time_range": [10, 20]},
            {"occurrence_id": "other", "time_range": [40, 50]},
        )
    }

    events = DIAGNOSIS._expanded_sufficiency_events(trace, candidate_sets)

    assert events[0]["sufficient_occurrence_ids"] == ["gold"]


def test_support_discrimination_uses_gold_labels_and_case_bootstrap() -> None:
    c1_gold = "c1-gold"
    c1_other = "c1-other"
    c2_gold = "c2-gold"
    c2_other = "c2-other"
    c1 = _case(
        "c1",
        "set-1",
        _event(
            "set-1",
            [
                _constraint(
                    "identity",
                    {c1_gold: "supported", c1_other: "contradicted"},
                ),
                _constraint(
                    "action",
                    {c1_gold: "unknown", c1_other: "partial"},
                    implicit=(c1_gold,),
                ),
            ],
        ),
    )
    c2 = _case(
        "c2",
        "set-2",
        _event(
            "set-2",
            [
                _constraint(
                    "identity",
                    {c2_gold: "supported", c2_other: "unknown"},
                ),
                _constraint(
                    "action",
                    {c2_gold: "supported", c2_other: "unknown"},
                    implicit=(c2_other,),
                ),
            ],
            verdict="sufficient",
        ),
    )

    result = DIAGNOSIS.build_support_discrimination(
        (c1, c2), bootstrap_samples=200, seed=7
    )

    assert result["candidate_present_event_count"] == 2
    assert result["overall"]["gold"]["supported_rate"] == 0.75
    assert result["overall"]["non_gold"]["supported_rate"] == 0.0
    assert result["overall"]["support_gap"] == 0.75
    assert (
        result["by_constraint_type"]["action"]["gold"]["counts"]["implicit_unknown"]
        == 1
    )
    assert (
        result["by_constraint_type"]["identity"]["non_gold"]["counts"][
            "declared_unknown"
        ]
        == 1
    )
    assert (
        result["by_constraint_type"]["identity"]["semantic_group"]
        == "referent_identifying"
    )
    assert result["case_cluster_bootstrap"]["positive_probability"] == 1.0
    assert result["strong_gold_non_gold_discrimination"] is True

    signed = DIAGNOSIS.build_signed_evidence_diagnostic(
        (c1, c2), bootstrap_samples=200, seed=7
    )
    assert signed["overall"]["gold"]["contradicted_rate"] == 0.0
    assert signed["overall"]["non_gold"]["contradicted_rate"] == 0.25
    assert signed["overall"]["non_gold_minus_gold_gap"] == 0.25
    assert signed["case_cluster_bootstrap"]["positive_probability"] > 0.7
    assert signed["diagnostic_only"] is True
    assert signed["runtime_scoring_changed"] is False


def test_gold_at_k_reports_conditional_retention_and_candidate_counts() -> None:
    event1 = _event("set-1", [_constraint("identity", {"c1-gold": "supported"})])
    event2 = _event("set-2", [_constraint("identity", {"c2-gold": "supported"})])
    c1 = _case("c1", "set-1", event1, gold_first=True)
    c2 = _case("c2", "set-2", event2, gold_first=False)

    result = DIAGNOSIS.build_gold_at_k((c1, c2))

    assert result["analyzed_set_count"] == 2
    assert result["gold_at_k"]["1"]["unconditional_rate"] == 0.5
    assert result["gold_at_k"]["3"]["conditional_retention_rate"] == 1.0
    assert result["candidate_count_distribution"]["median"] == 2.0
    assert result["recommended_top_k"] == 3


def test_selection_metrics_compare_observed_with_always_abstain() -> None:
    rows = (
        {
            "arm": "a3",
            "candidate_recall_resolved_set": True,
            "final_resolution": "selected",
            "osa_strict": True,
        },
        {
            "arm": "a3",
            "candidate_recall_resolved_set": False,
            "final_resolution": "selected",
            "osa_strict": None,
        },
        {
            "arm": "a4",
            "candidate_recall_resolved_set": True,
            "final_resolution": "no_match",
            "osa_strict": False,
        },
        {
            "arm": "a4",
            "candidate_recall_resolved_set": False,
            "final_resolution": "no_match",
            "osa_strict": None,
        },
    )

    result = DIAGNOSIS.build_selection_diagnostics(rows)

    assert result["a3"]["observed"]["precision"] == 0.5
    assert result["a3"]["observed"]["recall"] == 1.0
    assert result["a3"]["observed"]["false_commit_rate"] == 1.0
    assert result["a4"]["observed"]["balanced_accuracy"] == 0.5
    assert result["a4"]["always_abstain"]["f1"] == 0.0
    assert result["a4"]["always_abstain"]["no_match_accuracy"] == 1.0


def test_selection_metrics_separate_gate_recall_from_resolver_accuracy() -> None:
    rows = (
        {
            "arm": "a4",
            "candidate_recall_resolved_set": True,
            "final_resolution": "selected",
            "osa_strict": False,
        },
        {
            "arm": "a4",
            "candidate_recall_resolved_set": True,
            "final_resolution": "no_match",
            "osa_strict": False,
        },
        {
            "arm": "a4",
            "candidate_recall_resolved_set": False,
            "final_resolution": "selected",
            "osa_strict": False,
        },
        {
            "arm": "a4",
            "candidate_recall_resolved_set": False,
            "final_resolution": "no_match",
            "osa_strict": False,
        },
    )

    metrics = DIAGNOSIS.build_selection_diagnostics(rows)["a4"]["observed"]

    assert metrics["recall"] == 0.5
    assert metrics["osa_given_commit"] == 0.0
    assert metrics["wrong_occurrence_commit_count"] == 1


def test_r5_error_geometry_separates_weak_tie_and_zero_cases() -> None:
    weak = _case(
        "weak",
        "set-weak",
        _event(
            "set-weak",
            [
                _constraint(
                    "identity",
                    {"weak-gold": "supported", "weak-other": "unknown"},
                )
            ],
        ),
    )
    weak["clues"] = ((80.0, 90.0),)
    tied = _case(
        "tied",
        "set-tied",
        _event(
            "set-tied",
            [
                _constraint(
                    "identity",
                    {"tied-gold": "supported", "tied-other": "supported"},
                )
            ],
        ),
    )
    zero = _case(
        "zero",
        "set-zero",
        _event(
            "set-zero",
            [
                _constraint(
                    "identity",
                    {"zero-gold": "unknown", "zero-other": "unknown"},
                )
            ],
        ),
    )
    correct = _case(
        "correct",
        "set-correct",
        _event(
            "set-correct",
            [
                _constraint(
                    "identity",
                    {"correct-gold": "supported", "correct-other": "unknown"},
                )
            ],
        ),
    )
    selection_rows = (
        {
            "arm": "a4",
            "case_id": "weak",
            "final_resolution": "selected",
            "selected_occurrence_ids": ["weak-gold"],
        },
        {
            "arm": "a4",
            "case_id": "tied",
            "final_resolution": "no_match",
            "selected_occurrence_ids": [],
        },
        {
            "arm": "a4",
            "case_id": "zero",
            "final_resolution": "no_match",
            "selected_occurrence_ids": [],
        },
        {
            "arm": "a4",
            "case_id": "correct",
            "final_resolution": "selected",
            "selected_occurrence_ids": ["correct-gold"],
        },
    )

    result = DIAGNOSIS.build_r5_error_geometry(
        (weak, tied, zero, correct), selection_rows=selection_rows
    )

    assert result["outcome_counts"] == {
        "correct_commit": 1,
        "false_abstention": 2,
        "false_commit": 1,
    }
    assert result["geometry_by_outcome"]["false_commit"] == {"weak_unique_leader": 1}
    assert result["geometry_by_outcome"]["false_abstention"] == {
        "all_zero": 1,
        "positive_tie": 1,
    }


def test_winner_guard_potential_uses_frozen_qualification_thresholds() -> None:
    false_rows = [
        {
            "case_id": f"negative-{index:02d}",
            "outcome": "false_commit",
            "selected_occurrence_id": f"negative-{index:02d}-winner",
            "winner_contradicted": index < 5,
            "winner_contradiction_constraint_types": (
                ["identity"] if index < 5 else []
            ),
            "winner_contradiction_passage_ids": (
                [f"passage-{index:02d}"] if index < 5 else []
            ),
        }
        for index in range(12)
    ]
    correct_rows = [
        {
            "case_id": f"positive-{index:02d}",
            "outcome": "correct_commit",
            "selected_occurrence_id": f"positive-{index:02d}-winner",
            "winner_contradicted": False,
            "winner_contradiction_constraint_types": [],
            "winner_contradiction_passage_ids": [],
        }
        for index in range(8)
    ]

    result = DIAGNOSIS.build_winner_guard_potential(
        {
            "error_rows": false_rows,
            "correct_commit_rows": correct_rows,
        }
    )

    assert result["false_winner_contradicted_count"] == 5
    assert result["false_winner_contradiction_coverage"] == 5 / 12
    assert result["correct_winner_contradicted_count"] == 0
    assert result["correct_winner_contradiction_rate"] == 0.0
    assert result["frozen_qualification"]["required_false_blocks"] == 5
    assert result["frozen_qualification"]["allowed_correct_blocks"] == 0
    assert result["hard_veto_qualified_on_this_repeat"] is True


def test_diagnosis_can_freeze_winner_rows_separately_from_signed_decisions() -> None:
    case = _case(
        "frozen",
        "set-frozen",
        _event(
            "set-frozen",
            [
                _constraint(
                    "identity",
                    {
                        "frozen-gold": "contradicted",
                        "frozen-other": "unknown",
                    },
                )
            ],
        ),
    )
    case["clues"] = ((80.0, 90.0),)
    signed_rows = (
        {
            "arm": "a4",
            "case_id": "frozen",
            "final_resolution": "no_match",
            "selected_occurrence_ids": [],
        },
    )
    frozen_rows = (
        {
            "arm": "a4",
            "case_id": "frozen",
            "final_resolution": "selected",
            "selected_occurrence_ids": ["frozen-gold"],
        },
    )

    result = DIAGNOSIS.build_diagnosis(
        (case,),
        selection_rows=signed_rows,
        winner_selection_rows=frozen_rows,
        expected_cases=1,
        bootstrap_samples=10,
    )

    d7 = result["d7_winner_guard_potential"]
    assert d7["winner_source"] == "external_frozen_control"
    assert d7["false_commit_winner_count"] == 1
    assert d7["false_winner_contradicted_count"] == 1


def test_aggregation_rule_sweep_replays_r0_and_finds_referent_working_point() -> None:
    present_gold = "present-gold"
    present_other = "present-other"
    present = _case(
        "present",
        "set-present",
        _event(
            "set-present",
            [
                _constraint(
                    "identity",
                    {present_gold: "supported", present_other: "unknown"},
                ),
                _constraint(
                    "outcome",
                    {present_gold: "unknown", present_other: "unknown"},
                ),
            ],
        ),
    )
    absent_gold = "absent-gold"
    absent_other = "absent-other"
    absent = _case(
        "absent",
        "set-absent",
        _event(
            "set-absent",
            [
                _constraint(
                    "identity",
                    {absent_gold: "unknown", absent_other: "unknown"},
                ),
                _constraint(
                    "outcome",
                    {absent_gold: "unknown", absent_other: "unknown"},
                ),
            ],
        ),
    )
    absent["clues"] = ((80.0, 90.0),)
    observed = (
        {
            "arm": "a4",
            "candidate_recall_resolved_set": True,
            "final_resolution": "no_match",
            "osa_strict": False,
        },
        {
            "arm": "a4",
            "candidate_recall_resolved_set": False,
            "final_resolution": "no_match",
            "osa_strict": False,
        },
    )

    result = DIAGNOSIS.build_aggregation_rule_sweep(
        (present, absent), observed_selection_rows=observed
    )

    variants = {row["variant_id"]: row for row in result["variants"]}
    assert result["r0_observed_parity"]["passed"] is True
    assert variants["R0:all_supported"]["metrics"]["tp"] == 0
    assert variants["R2:referent_all_supported"]["metrics"]["tp"] == 1
    assert variants["R2:referent_all_supported"]["metrics"]["tn"] == 1
    assert variants["R2:referent_all_supported"]["target_met"] is True
    assert any(row["rule_id"] == "R3" for row in result["variants"])
    assert any(row["rule_id"] == "R4" for row in result["variants"])
    assert any(row["rule_id"] == "R5" for row in result["variants"])
