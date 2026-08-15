from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analyze_mmlifelong_occurrence_agent.py"
)
SPEC = importlib.util.spec_from_file_location("occurrence_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)

AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mmlifelong_occurrence_canary.py"
)
AUDIT_SPEC = importlib.util.spec_from_file_location("occurrence_audit", AUDIT_PATH)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def _row(
    arm: str,
    case_id: str,
    *,
    score: float,
    signature: list[dict],
    digest: str = "digest",
    retrieval_digest: str = "retrieval",
) -> dict:
    return {
        "arm": arm,
        "case_id": case_id,
        "score": score,
        "exact_correct": score == 1.0,
        "raw_exact": score == 1.0,
        "verified_correct": False,
        "correct_and_ref_300": score == 1.0,
        "grounded_correct_ref300": score == 1.0,
        "grounded_correct_bound_visual": False,
        "parse_status": "parsed",
        "judge_model": "judge",
        "candidate_recall": True,
        "candidate_recall_trajectory": True,
        "candidate_recall_active_set": True,
        "candidate_recall_resolved_set": True,
        "candidate_clue_recall": 1.0,
        "resolved_set_id": "set-1",
        "final_resolution": "selected" if arm == "a2" else "unresolved",
        "osa_eligible": arm == "a2",
        "osa_correct": arm == "a2",
        "osa_any": True if arm == "a2" else None,
        "osa_strict": True if arm == "a2" else None,
        "osa_precision": 1.0 if arm == "a2" else None,
        "selected_occurrence_count": 1 if arm == "a2" else 0,
        "selected_clue_recall": 1.0 if arm == "a2" else None,
        "abstention_eligible": False,
        "abstention_correct": None,
        "no_match_correct": None,
        "false_commit": None,
        "false_abstention": False if arm == "a2" else None,
        "deferred_occurrence_set": False,
        "occurrence_handle_usage_rate": 1.0,
        "selected_locator_usage_rate": None,
        "locator_scope_single_set_passed": True,
        "bound_visual_clue_recall": None,
        "arbitration_activation_round": None,
        "resolution_activation_round": None,
        "resolution_activation_threshold_valid": True,
        "arbitration_activation_threshold_valid": True,
        "premature_occurrence_commit": False,
        "visual_frames": 3,
        "visual_windows": 1,
        "vlm_calls": 1,
        "semantic_rounds_used": 4,
        "forced_finalize_round": 4,
        "extra_rounds_granted": 0,
        "pre_treatment_signature": signature,
        "pre_treatment_prompt_digests": ["prompt"],
        "pre_activation_state_exposure": False,
        "treatment_exposure": {
            "visible_excerpt_digest": digest,
            "visible_text_digest": digest,
            "visible_excerpt_count": 2,
        },
        "treatment_retrieval_identity_digest": retrieval_digest,
        "same_packet_text_budget_parity_passed": (
            True if arm in {"a1", "a1-flat"} else None
        ),
        "no_oracle_gate_passed": True,
        "occurrence_replay_mode": "live",
        "occurrence_replay_fixture_digest": None,
        "occurrence_replay_complete": True,
        "frozen_replay_full_consumption": True,
        "occurrence_replay_prefix_valid": True,
        "occurrence_replay_identity_digests": [],
        "occurrence_replay_prime_configured": False,
        "occurrence_replay_prime_requested": False,
        "occurrence_replay_prime_consumed": False,
        "occurrence_replay_prime_event_count": 0,
        "occurrence_replay_prime_event_completed": False,
        "occurrence_replay_prime_event_pre_reasoner": False,
        "occurrence_replay_post_fixture_reuse_count": 0,
        "retired_locator_count": 0,
        "frozen_config": {
            "controller_mode": "frozen_baseline",
            "max_rounds": 4,
            "semantic_round_budget": 4,
        },
    }


def test_report_separates_pre_treatment_divergence_and_a2_osa() -> None:
    same = [{"action": "investigate", "tasks": []}]
    different = [{"action": "update_workspace", "tasks": []}]
    rows = (
        _row("a0", "c1", score=0.0, signature=same),
        _row("a0", "c2", score=0.0, signature=same),
        _row("a1", "c1", score=1.0, signature=same),
        _row("a1", "c2", score=1.0, signature=different),
        _row("a1-flat", "c1", score=0.0, signature=same),
        _row("a1-flat", "c2", score=0.0, signature=different),
        _row("a2", "c1", score=1.0, signature=same),
        _row("a2", "c2", score=1.0, signature=same),
    )

    report = ANALYSIS.build_report(
        rows, expected_cases=2, bootstrap_samples=100, seed=7
    )

    assert report["comparisons"]["a1-a0"][
        "pre_treatment_divergence_rate"
    ] == 0.5
    assert report["comparisons"]["a1-a0"][
        "matched_pre_treatment_subset"
    ]["paired_n"] == 1
    assert report["arms"]["a1"]["occurrence_selection_accuracy"] is None
    assert report["arms"]["a2"]["occurrence_selection_accuracy"] == 1.0
    assert report["text_budget_parity"]["comparable_n"] == 2
    assert report["text_budget_parity"]["passed"] is True
    assert report["structural_gate_passed"] is True


def test_candidate_recall_is_scoped_to_the_resolved_set() -> None:
    observations = (
        {
            "attempt_id": "set-gold",
            "sampling_config": {
                "occurrence_set": {
                    "candidates": [
                        {"occurrence_id": "gold", "time_range": [10, 20]}
                    ]
                }
            },
        },
        {
            "attempt_id": "set-wrong",
            "sampling_config": {
                "occurrence_set": {
                    "candidates": [
                        {"occurrence_id": "wrong", "time_range": [100, 110]}
                    ]
                }
            },
        },
    )
    state = {
        "active_set_id": "set-wrong",
        "active_resolution": "selected",
        "sets": [
            {
                "set_id": "set-gold",
                "lifecycle": "retired",
                "resolution": "deferred",
                "candidates": [
                    {"occurrence_id": "gold", "time_range": [10, 20]}
                ],
                "selected_occurrence_ids": [],
            },
            {
                "set_id": "set-wrong",
                "lifecycle": "active",
                "resolution": "selected",
                "candidates": [
                    {"occurrence_id": "wrong", "time_range": [100, 110]}
                ],
                "selected_occurrence_ids": ["wrong"],
            },
        ],
    }
    trace = (
        {
            "type": "reasoner_decision",
            "occurrence_ops_accepted": True,
            "occurrence_ops": [
                {
                    "op": "select",
                    "set_id": "set-wrong",
                    "occurrence_id": "wrong",
                }
            ],
        },
    )

    metrics = ANALYSIS._occurrence_resolution_metrics(
        arm="a3",
        state=state,
        trace=trace,
        observations=observations,
        clues=((10.0, 20.0),),
    )

    assert metrics["candidate_recall_trajectory"] is True
    assert metrics["candidate_recall_active_set"] is False
    assert metrics["candidate_recall_resolved_set"] is False
    assert metrics["resolved_set_id"] == "set-wrong"


def test_final_resolution_reports_all_four_terminal_states() -> None:
    expected = ("selected", "no_match", "deferred", "unresolved")
    for resolution in expected:
        state = {
            "active_set_id": "set-1",
            "active_resolution": resolution,
            "sets": [
                {
                    "set_id": "set-1",
                    "lifecycle": "active",
                    "resolution": resolution,
                    "candidates": [
                        {"occurrence_id": "occ-1", "time_range": [10, 20]}
                    ],
                    "selected_occurrence_ids": (
                        ["occ-1"] if resolution == "selected" else []
                    ),
                }
            ],
        }
        metrics = ANALYSIS._occurrence_resolution_metrics(
            arm="a2-clean",
            state=state,
            trace=(),
            observations=(),
            clues=((10.0, 20.0),),
        )
        assert metrics["final_resolution"] == resolution


def test_strict_osa_rejects_multi_selection_credit() -> None:
    state = {
        "active_set_id": "set-1",
        "active_resolution": "selected",
        "sets": [
            {
                "set_id": "set-1",
                "lifecycle": "active",
                "resolution": "selected",
                "candidates": [
                    {"occurrence_id": "correct", "time_range": [10, 20]},
                    {"occurrence_id": "wrong", "time_range": [100, 110]},
                ],
                "selected_occurrence_ids": ["correct", "wrong"],
            }
        ],
    }

    metrics = ANALYSIS._occurrence_resolution_metrics(
        arm="a3",
        state=state,
        trace=(),
        observations=(),
        clues=((10.0, 20.0),),
    )

    assert metrics["osa_any"] is True
    assert metrics["osa_strict"] is False
    assert metrics["osa_precision"] == 0.5


def test_false_commit_uses_final_resolved_set_state() -> None:
    state = {
        "active_set_id": "set-wrong",
        "active_resolution": "selected",
        "sets": [
            {
                "set_id": "set-wrong",
                "lifecycle": "active",
                "resolution": "selected",
                "candidates": [
                    {"occurrence_id": "wrong", "time_range": [100, 110]}
                ],
                "selected_occurrence_ids": ["wrong"],
            }
        ],
    }

    metrics = ANALYSIS._occurrence_resolution_metrics(
        arm="a2-clean",
        state=state,
        trace=(),
        observations=(),
        clues=((10.0, 20.0),),
    )

    assert metrics["candidate_recall_resolved_set"] is False
    assert metrics["false_commit"] is True
    assert metrics["no_match_correct"] is False
    assert metrics["false_abstention"] is None


def test_decomposition_writes_tables_and_automatic_a_b_decision(
    tmp_path: Path,
) -> None:
    signature = [{"action": "investigate", "tasks": []}]
    rows = []
    for case_id in ("loss", "absent"):
        a0 = _row(
            "a0",
            case_id,
            score=1.0 if case_id == "loss" else 0.0,
            signature=signature,
        )
        a2_clean = _row(
            "a2-clean", case_id, score=0.0, signature=signature
        )
        a3 = _row("a3", case_id, score=0.0, signature=signature)
        if case_id == "loss":
            a0["bound_visual_clue_recall"] = 0.0
            a3.update(
                {
                    "osa_strict": False,
                    "selected_locator_usage_rate": 1.0,
                    "bound_visual_clue_recall": 0.0,
                }
            )
        else:
            for row in (a0, a2_clean, a3):
                row["candidate_recall_resolved_set"] = False
            a2_clean.update(
                {
                    "final_resolution": "no_match",
                    "no_match_correct": True,
                }
            )
            a3.update(
                {
                    "final_resolution": "selected",
                    "false_commit": True,
                }
            )
        rows.extend((a0, a2_clean, a3))

    report = ANALYSIS.build_decomposition(
        tuple(rows), trajectory_provenance="fixture"
    )
    primary = report["slices"]["frozen_complete"]

    assert primary["n"] == 2
    assert primary["table_a"]["classification"] == "A"
    assert primary["table_a"]["loss_count"] == 1
    assert any(
        row["arm"] == "a3"
        and row["final_resolution"] == "selected"
        and row["n"] == 1
        for row in primary["table_b"]["rows"]
    )
    assert any(
        row["row_type"] == "paired_score_outcome"
        and row["arm"] == "a3-a0"
        for row in primary["table_c"]["rows"]
    )

    paths = ANALYSIS.write_decomposition_outputs(report, tmp_path)
    assert set(paths) == {
        "table_a_losses.csv",
        "table_b_candidate_absent.csv",
        "table_c_candidate_present_funnel.csv",
        "decomposition_summary.md",
    }
    assert all(Path(path).is_file() for path in paths.values())
    assert "Classification: **A**" in Path(
        paths["decomposition_summary.md"]
    ).read_text(encoding="utf-8")

    b_decision = ANALYSIS._table_a_decision(
        (
            {
                "a0_bound_visual_clue_recall": 1.0,
                "a3_osa_strict": True,
                "a3_bound_visual_clue_recall": 1.0,
            },
        )
    )
    assert b_decision["classification"] == "B"


def test_a1_flat_text_mismatch_fails_structural_gate() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    rows = (
        _row("a1", "c1", score=0.0, signature=signature, digest="left"),
        _row(
            "a1-flat", "c1", score=0.0, signature=signature, digest="right"
        ),
    )

    report = ANALYSIS.build_report(
        rows, expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["text_budget_parity"]["passed"] is False
    assert report["structural_gate_passed"] is False


def test_frozen_complete_is_primary_and_comparisons_report_sign_test() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    rows = []
    for case_id, baseline_score, treatment_score in (
        ("c1", 0.0, 1.0),
        ("c2", 1.0, 0.0),
    ):
        rows.append(
            _row("a0", case_id, score=baseline_score, signature=signature)
        )
        treatment = _row(
            "a1", case_id, score=treatment_score, signature=signature
        )
        if case_id == "c2":
            treatment["frozen_replay_full_consumption"] = False
        rows.append(treatment)

    report = ANALYSIS.build_report(
        tuple(rows), expected_cases=2, bootstrap_samples=20, seed=9
    )

    assert report["all_cases"]["n"] == 2
    assert report["frozen_complete"]["n"] == 1
    assert report["case_count"] == 1
    comparison = report["comparisons"]["a1-a0"]
    assert comparison["paired_n"] == 1
    assert comparison["sign_test_p"] == 1.0
    assert comparison["underpowered"] is True


def test_budget_symmetry_treats_realized_rounds_as_endpoint() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    left = _row("a2-clean", "c1", score=0.0, signature=signature)
    right = _row("a3", "c1", score=0.0, signature=signature)
    right["semantic_rounds_used"] = 5

    report = ANALYSIS.build_report(
        (left, right), expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["budget_symmetry"]["max_minus_min"] == 1.0
    assert report["budget_symmetry"]["configured_max_minus_min"] == 0
    assert report["budget_symmetry"][
        "observed_realized_rounds_endpoint_only"
    ] is True
    assert report["structural_checks"]["budget_symmetry_passed"] is True
    assert report["structural_gate_passed"] is True


def test_budget_symmetry_fails_when_configured_budgets_differ() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    left = _row("a2-clean", "c1", score=0.0, signature=signature)
    right = _row("a3", "c1", score=0.0, signature=signature)
    right["frozen_config"] = {
        **right["frozen_config"],
        "max_rounds": 5,
        "semantic_round_budget": 5,
    }

    report = ANALYSIS.build_report(
        (left, right), expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["budget_symmetry"]["configured_max_minus_min"] == 1
    assert report["structural_checks"]["budget_symmetry_passed"] is False
    assert report["structural_gate_passed"] is False


def test_locator_scope_must_resolve_to_one_active_set() -> None:
    row = _row(
        "a3",
        "c1",
        score=0.0,
        signature=[{"action": "investigate", "tasks": []}],
    )
    row["locator_scope_single_set_passed"] = False

    report = ANALYSIS.build_report(
        (row,), expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["structural_checks"][
        "locator_scope_single_set_passed"
    ] is False
    assert report["structural_gate_passed"] is False

    malformed_state = {
        "active_set_id": "new",
        "retired_set_ids": ["old"],
        "active_locators": [],
        "retired_locators": [],
        "sets": [
            {
                "set_id": "old",
                "lifecycle": "retired",
                "selected_occurrence_ids": ["old_1"],
            },
            {
                "set_id": "new",
                "lifecycle": "active",
                "selected_occurrence_ids": [],
            },
        ],
    }
    assert ANALYSIS._locator_scope_single_set_passed(malformed_state) is False


def test_a1_flat_text_parity_is_not_comparable_after_trajectory_divergence() -> None:
    rows = (
        _row(
            "a1",
            "c1",
            score=0.0,
            signature=[{"action": "investigate", "tasks": []}],
            digest="left",
        ),
        _row(
            "a1-flat",
            "c1",
            score=0.0,
            signature=[{"action": "answer", "tasks": []}],
            digest="right",
        ),
    )

    report = ANALYSIS.build_report(
        rows, expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["text_budget_parity"]["passed"] is None
    assert report["text_budget_parity"]["status"] == "not_comparable"
    assert report["structural_gate_passed"] is True


def test_a1_flat_text_parity_is_not_comparable_after_retrieval_divergence() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    rows = (
        _row(
            "a1",
            "c1",
            score=0.0,
            signature=signature,
            digest="left",
            retrieval_digest="retrieval-left",
        ),
        _row(
            "a1-flat",
            "c1",
            score=0.0,
            signature=signature,
            digest="right",
            retrieval_digest="retrieval-right",
        ),
    )

    report = ANALYSIS.build_report(
        rows, expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["text_budget_parity"]["passed"] is None
    assert report["text_budget_parity"]["status"] == "not_comparable"
    assert report["structural_gate_passed"] is True


def test_first_exposed_retrieval_identity_skips_empty_card_packets() -> None:
    audit = {
        "candidate_card_counts": [0, 3],
        "retrieval_identity_digests": ["empty", "exposed"],
    }

    assert ANALYSIS._first_exposed_retrieval_identity(audit) == "exposed"


def test_canary_audit_checks_structured_protocol_artifacts(tmp_path) -> None:
    bindings = {}
    for arm in ("a0", "a1-flat", "a1", "a2"):
        root = tmp_path / arm
        case = root / "cases" / "case-1"
        case.mkdir(parents=True)
        (case / "prediction.json").write_text(
            json.dumps({"case_id": "case-1"}), encoding="utf-8"
        )
        (case / "run_config.json").write_text(
            json.dumps(
                {
                    "occurrence_method_arm": arm,
                    "models": {
                        "reasoner": "pa/gmn-2.5-pr",
                        "investigator": "pa/gmn-2.5-pr",
                    },
                }
            ),
            encoding="utf-8",
        )
        trace = [
            {
                "type": "occurrence_treatment_eligible",
                "round": 2,
                "visible_occurrence_count": 2 if arm == "a2" else 1,
            },
            {
                "type": "reasoner_decision",
                "action": "update_workspace" if arm == "a2" else "investigate",
                "occurrence_ops": (
                    [{"op": "select", "occurrence_id": "occ_1"}]
                    if arm == "a2"
                    else []
                ),
                "occurrence_ops_accepted": True,
            },
        ]
        if arm == "a2":
            trace.insert(
                1,
                {
                    "type": "reasoner_decision",
                    "action": "update_workspace",
                    "occurrence_ops": [
                        {"op": "select", "occurrence_id": "stale_occ"}
                    ],
                    "occurrence_ops_accepted": False,
                },
            )
            trace.insert(
                2,
                {
                    "type": "decision_schema_error",
                    "errors": [
                        {"code": "occurrence_id_not_currently_visible"}
                    ],
                },
            )
            trace.insert(
                3,
                {
                    "type": "decision_schema_error",
                    "errors": [
                        {"code": "occurrence_selection_required"}
                    ],
                },
            )
            trace.append(
                {
                    "type": "decision_schema_error",
                    "errors": [
                        {
                            "code": "occurrence_answer_required_after_selection"
                        }
                    ],
                }
            )
            trace.append(
                {
                    "type": "reasoner_decision",
                    "action": "answer",
                    "occurrence_ops": [],
                    "occurrence_ops_accepted": True,
                }
            )
        if arm != "a0":
            trace.append({"type": "occurrence_treatment_exposed", "round": 2})
        (case / "runtime_summary.json").write_text(
            json.dumps(
                {
                    "no_oracle_runtime_gate": {
                        "no_oracle_runtime_gate_passed": True,
                        "text_budget_parity_passed": True,
                    },
                    "trace": trace,
                }
            ),
            encoding="utf-8",
        )
        if arm == "a2":
            (case / "occurrence_resolution_state.json").write_text(
                "{}", encoding="utf-8"
            )
        bindings[arm] = root

    report = AUDIT.audit_roots(bindings, expected_cases=1)

    assert report["structural_gate_passed"] is True
    assert report["per_arm"]["a2"]["occurrence_op_count"] == 2
    assert report["per_arm"]["a2"]["selection_case_count"] == 1
    assert report["per_arm"]["a2"]["selection_required_case_count"] == 1
    assert report["per_arm"]["a2"]["selection_missing_case_count"] == 0
    assert report["per_arm"]["a2"]["selection_not_prior_case_count"] == 0
    assert report["per_arm"]["a2"]["answer_missing_after_selection_case_count"] == 0
    assert report["per_arm"]["a2"]["selection_required_retry_count"] == 1
    assert report["per_arm"]["a2"]["answer_required_retry_count"] == 1
    assert report["per_arm"]["a2"]["occurrence_validation_error_count"] == 1
    assert report["per_arm"]["a2"]["rejected_occurrence_op_attempt_count"] == 1
    assert report["per_arm"]["a2"][
        "recovered_occurrence_op_rejection_case_count"
    ] == 1
    assert report["per_arm"]["a2"][
        "unrecovered_occurrence_op_rejection_case_count"
    ] == 0
    assert report["per_arm"]["a2"]["state_file_count"] == 1

    a1_runtime = bindings["a1"] / "cases" / "case-1" / "runtime_summary.json"
    a1_payload = json.loads(a1_runtime.read_text(encoding="utf-8"))
    a1_payload["trace"] = [
        row
        for row in a1_payload["trace"]
        if row.get("type")
        not in {"occurrence_treatment_eligible", "occurrence_treatment_exposed"}
    ]
    a1_runtime.write_text(json.dumps(a1_payload), encoding="utf-8")

    ineligible_report = AUDIT.audit_roots(bindings, expected_cases=1)
    assert ineligible_report["structural_gate_passed"] is True
    assert ineligible_report["per_arm"]["a1"]["eligible_event_count"] == 0
    assert ineligible_report["per_arm"]["a1"]["exposure_event_count"] == 0


def test_clean_and_actionable_arms_report_full_mechanism_chain() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    rows = []
    for case_id in ("c1", "c2"):
        rows.append(_row("a1", case_id, score=0.0, signature=signature))
        clean = _row(
            "a2-clean", case_id, score=1.0, signature=signature
        )
        clean.update(
            {
                "osa_eligible": case_id == "c1",
                "osa_correct": True if case_id == "c1" else None,
                "osa_any": True if case_id == "c1" else None,
                "osa_strict": True if case_id == "c1" else None,
                "osa_precision": 1.0 if case_id == "c1" else None,
                "candidate_recall": case_id == "c1",
                "candidate_recall_trajectory": case_id == "c1",
                "candidate_recall_active_set": case_id == "c1",
                "candidate_recall_resolved_set": case_id == "c1",
                "candidate_clue_recall": 1.0 if case_id == "c1" else 0.0,
                "selected_occurrence_count": 1 if case_id == "c1" else 0,
                "selected_clue_recall": 1.0 if case_id == "c1" else None,
                "abstention_eligible": case_id == "c2",
                "abstention_correct": True if case_id == "c2" else None,
                "no_match_correct": True if case_id == "c2" else None,
                "false_commit": False if case_id == "c2" else None,
                "false_abstention": False if case_id == "c1" else None,
                "final_resolution": "selected" if case_id == "c1" else "no_match",
                "arbitration_activation_round": 2,
            }
        )
        rows.append(clean)
        actionable = dict(clean)
        actionable.update(
            {
                "arm": "a3",
                "selected_locator_usage_rate": (
                    1.0 if case_id == "c1" else None
                ),
                "bound_visual_clue_recall": (
                    1.0 if case_id == "c1" else 0.0
                ),
            }
        )
        rows.append(actionable)

    report = ANALYSIS.build_report(
        tuple(rows), expected_cases=2, bootstrap_samples=50, seed=11
    )

    assert report["comparisons"]["a2-clean-a1"]["paired_n"] == 2
    assert report["comparisons"]["a3-a2-clean"][
        "pre_treatment_prompt_divergence_rate"
    ] == 0.0
    assert report["arms"]["a2-clean"]["occurrence_selection_accuracy"] == 1.0
    assert report["arms"]["a2-clean"]["abstention_accuracy"] == 1.0
    assert report["arms"]["a3"]["selected_locator_usage_rate"] == 1.0
    assert report["arms"]["a3"]["bound_visual_clue_recall"] == 0.5
    assert report["structural_gate_passed"] is True


def test_post_selection_only_divergence_requires_paired_resolution_identity() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    clean = _row("a2-clean", "c1", score=0.0, signature=signature)
    actionable = _row("a3", "c1", score=0.0, signature=signature)
    for row in (clean, actionable):
        row.update(
            {
                "resolved_set_id": "set_1",
                "final_resolution": "selected",
                "selected_locators_accounted": True,
                "selected_locator_silent_drop_count": 0,
                "selected_locator_accounting_conflict_count": 0,
            }
        )
    clean["selected_occurrence_ids"] = ["occ_1"]
    actionable["selected_occurrence_ids"] = ["occ_2"]

    report = ANALYSIS.build_report(
        (clean, actionable),
        expected_cases=1,
        bootstrap_samples=10,
        seed=3,
    )

    assert report["structural_checks"]["post_selection_only_divergence"] is False
    assert report["post_selection_only_divergence"]["mismatch_case_ids"] == [
        "c1"
    ]


def test_matched_pre_treatment_response_gate_requires_exact_role_counts() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    clean = _row("a2-clean", "c1", score=0.0, signature=signature)
    actionable = _row("a3", "c1", score=0.0, signature=signature)
    common = {
        "resolved_set_id": "set_1",
        "final_resolution": "selected",
        "selected_occurrence_ids": ["occ_1"],
        "selected_locators_accounted": True,
        "selected_locator_silent_drop_count": 0,
        "selected_locator_accounting_conflict_count": 0,
    }
    clean.update(common)
    actionable.update(common)
    clean["matched_response_control"] = {
        "mode": "record",
        "active": False,
        "deactivation_reason": "scoped_occurrence_resolution_persisted",
        "recorded": {"investigator": 2, "reasoner": 3},
        "mismatch_count": 0,
    }
    actionable["matched_response_control"] = {
        "mode": "replay",
        "active": False,
        "deactivation_reason": "scoped_occurrence_resolution_persisted",
        "replayed": {"investigator": 2, "reasoner": 3},
        "mismatch_count": 0,
    }

    report = ANALYSIS.build_report(
        (clean, actionable),
        expected_cases=1,
        bootstrap_samples=10,
        seed=3,
    )

    assert report["structural_checks"]["matched_pre_treatment_responses"] is True
    audit_gate = AUDIT._matched_pre_treatment_response_gate(
        {
            "a2-clean": {"c1": clean["matched_response_control"]},
            "a3": {"c1": actionable["matched_response_control"]},
        }
    )
    assert audit_gate["passed"] is True
    actionable["matched_response_control"]["replayed"]["reasoner"] = 2
    failed = ANALYSIS.build_report(
        (clean, actionable),
        expected_cases=1,
        bootstrap_samples=10,
        seed=3,
    )
    assert failed["structural_checks"]["matched_pre_treatment_responses"] is False


def test_matched_control_keeps_declared_cohort_when_replay_stops_post_treatment() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    clean = _row("a2-clean", "c1", score=0.0, signature=signature)
    actionable = _row("a3", "c1", score=1.0, signature=signature)
    common = {
        "resolved_set_id": "set_1",
        "final_resolution": "selected",
        "selected_occurrence_ids": ["occ_1"],
        "selected_locators_accounted": True,
        "selected_locator_silent_drop_count": 0,
        "selected_locator_accounting_conflict_count": 0,
        "frozen_replay_full_consumption": False,
    }
    clean.update(common)
    actionable.update(common)
    clean["matched_response_control"] = {
        "mode": "record",
        "active": False,
        "deactivation_reason": "scoped_occurrence_resolution_persisted",
        "recorded": {"investigator": 2, "reasoner": 3},
        "mismatch_count": 0,
    }
    actionable["matched_response_control"] = {
        "mode": "replay",
        "active": False,
        "deactivation_reason": "scoped_occurrence_resolution_persisted",
        "replayed": {"investigator": 2, "reasoner": 3},
        "mismatch_count": 0,
    }

    report = ANALYSIS.build_report(
        (clean, actionable),
        expected_cases=1,
        bootstrap_samples=10,
        seed=3,
    )

    assert report["frozen_complete"]["n"] == 0
    assert report["primary_analysis_set"] == "matched_aligned"
    assert report["case_count"] == 1
    assert report["arms"]["a2-clean"]["n"] == 1
    assert report["comparisons"]["a3-a2-clean"]["paired_n"] == 1


def test_scoped_canary_audit_requires_actionable_locator_accounting(
    tmp_path: Path,
) -> None:
    bindings = {}
    for arm in ("a1", "a2-clean", "a3"):
        root = tmp_path / arm
        case = root / "cases" / "case-1"
        case.mkdir(parents=True)
        (case / "prediction.json").write_text(
            json.dumps({"case_id": "case-1"}), encoding="utf-8"
        )
        (case / "run_config.json").write_text(
            json.dumps(
                {
                    "occurrence_method_arm": arm,
                    "models": {
                        "reasoner": "pa/gmn-2.5-pr",
                        "investigator": "pa/gmn-2.5-pr",
                    },
                }
            ),
            encoding="utf-8",
        )
        trace = [
            {
                "type": "occurrence_treatment_eligible",
                "round": 2,
                "visible_occurrence_count": 2,
            },
            {"type": "occurrence_treatment_exposed", "round": 2},
        ]
        if arm in {"a2-clean", "a3"}:
            trace.extend(
                [
                    {
                        "type": "reasoner_decision",
                        "round": 1,
                        "action": "investigate",
                        "occurrence_ops": [],
                        "occurrence_resolution_state_exposed": False,
                    },
                    {
                        "type": "occurrence_resolution_activated",
                        "round": 2,
                        "active_set_id": "locator_1",
                        "candidate_count": 2,
                        "arbitration_required": True,
                    },
                    {
                        "type": "occurrence_arbitration_activated",
                        "round": 2,
                        "active_set_id": "locator_1",
                        "candidate_count": 2,
                    },
                    {
                        "type": "reasoner_decision",
                        "round": 2,
                        "action": "update_workspace",
                        "occurrence_ops": [
                            {
                                "op": "select",
                                "set_id": "locator_1",
                                "occurrence_id": "occ_1",
                            }
                        ],
                        "occurrence_ops_accepted": True,
                        "occurrence_resolution_state_exposed": True,
                    },
                ]
            )
            if arm == "a3":
                trace.append(
                    {
                        "type": "occurrence_locator_released_unexecuted",
                        "round": 3,
                        "locator_attempt_id": "locator_1",
                        "occurrence_id": "occ_1",
                        "outcome": "released_at_budget_exhaustion",
                        "reason": "budget_exhausted_at_finalize",
                    }
                )
            trace.append(
                {
                    "type": "reasoner_decision",
                    "round": 4,
                    "action": "answer",
                    "occurrence_ops": [],
                    "occurrence_ops_accepted": True,
                }
            )
            (case / "occurrence_resolution_state.json").write_text(
                json.dumps(
                    {
                        "active_resolution": "selected",
                        "active_set_id": "locator_1",
                        "retired_set_ids": [],
                        "active_locators": [
                            {
                                "set_id": "locator_1",
                                "locator_attempt_id": "locator_1",
                                "occurrence_id": "occ_1",
                                "status": "selected_for_active_set",
                            }
                        ],
                        "retired_locators": [],
                        "sets": [
                            {
                                "set_id": "locator_1",
                                "resolution": "selected",
                                "lifecycle": "active",
                                "selected_occurrence_ids": ["occ_1"],
                                "candidates": [
                                    {"occurrence_id": "occ_1"},
                                    {"occurrence_id": "occ_2"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        (case / "runtime_summary.json").write_text(
            json.dumps(
                {
                    "no_oracle_runtime_gate": {
                        "no_oracle_runtime_gate_passed": True,
                        "text_budget_parity_passed": True,
                    },
                    "trace": trace,
                }
            ),
            encoding="utf-8",
        )
        bindings[arm] = root

    report = AUDIT.audit_roots(bindings, expected_cases=1)

    assert report["structural_gate_passed"] is True
    assert report["checks"]["no_pre_activation_state_exposure"] is True
    assert report["checks"]["scoped_set_integrity"] is True
    assert report["checks"]["a3_selected_locators_accounted"] is True
    assert report["per_arm"]["a3"]["selected_locator_count"] == 1
    assert report["per_arm"]["a3"][
        "selected_locator_inspection_failure_case_count"
    ] == 1
    assert report["per_arm"]["a3"]["selected_locator_release_counts"] == {
        "released_at_budget_exhaustion": 1,
        "released_on_set_retirement": 0,
        "released_by_revision": 0,
    }


def test_scoped_audit_does_not_count_no_match_as_missing_answer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "a2-clean"
    case = root / "cases" / "case-1"
    case.mkdir(parents=True)
    (case / "prediction.json").write_text(
        json.dumps({"case_id": "case-1"}), encoding="utf-8"
    )
    (case / "run_config.json").write_text(
        json.dumps(
            {
                "occurrence_method_arm": "a2-clean",
                "models": {
                    "reasoner": "pa/gmn-2.5-pr",
                    "investigator": "pa/gmn-2.5-pr",
                },
            }
        ),
        encoding="utf-8",
    )
    trace = [
        {
            "type": "occurrence_treatment_eligible",
            "round": 1,
            "visible_occurrence_count": 1,
        },
        {"type": "occurrence_treatment_exposed", "round": 1},
        {
            "type": "occurrence_resolution_activated",
            "round": 1,
            "active_set_id": "set_1",
            "candidate_count": 1,
            "arbitration_required": False,
        },
        {
            "type": "reasoner_decision",
            "round": 1,
            "action": "update_workspace",
            "occurrence_ops": [{"op": "no_match", "set_id": "set_1"}],
            "occurrence_ops_accepted": True,
            "occurrence_resolution_state_exposed": True,
        },
        {
            "type": "reasoner_decision",
            "round": 2,
            "action": "answer",
            "occurrence_ops": [],
            "occurrence_ops_accepted": True,
        },
    ]
    (case / "runtime_summary.json").write_text(
        json.dumps(
            {
                "no_oracle_runtime_gate": {
                    "no_oracle_runtime_gate_passed": True,
                    "text_budget_parity_passed": True,
                },
                "trace": trace,
            }
        ),
        encoding="utf-8",
    )
    (case / "occurrence_resolution_state.json").write_text(
        json.dumps(
            {
                "active_resolution": "no_match",
                "active_set_id": "set_1",
                "retired_set_ids": [],
                "active_locators": [],
                "retired_locators": [],
                "sets": [
                    {
                        "set_id": "set_1",
                        "resolution": "no_match",
                        "lifecycle": "active",
                        "selected_occurrence_ids": [],
                        "candidates": [{"occurrence_id": "occ_1"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = AUDIT.audit_roots({"a2-clean": root}, expected_cases=1)

    assert report["per_arm"]["a2-clean"][
        "answer_missing_after_selection_case_count"
    ] == 0
    assert report["per_arm"]["a2-clean"][
        "answer_missing_after_resolution_case_count"
    ] == 0
    assert report["checks"]["scoped_resolution_complete"] is True

    (case / "run_config.json").write_text(
        json.dumps(
            {
                "occurrence_method_arm": "a3",
                "models": {
                    "reasoner": "pa/gmn-2.5-pr",
                    "investigator": "pa/gmn-2.5-pr",
                },
            }
        ),
        encoding="utf-8",
    )
    a3_report = AUDIT.audit_roots({"a3": root}, expected_cases=1)

    assert a3_report["checks"]["a3_selected_locators_accounted"] is True
    assert a3_report["check_applicability"][
        "a3_selected_locators_accounted"
    ] is False


def test_frozen_replay_accepts_valid_consumed_prefixes() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    recorder = _row("a0", "c1", score=0.0, signature=signature)
    recorder.update(
        {
            "occurrence_replay_mode": "record",
            "occurrence_replay_fixture_digest": "fixture",
            "occurrence_replay_identity_digests": ["packet-1", "packet-2"],
        }
    )
    treatment = _row("a2-clean", "c1", score=1.0, signature=signature)
    treatment.update(
        {
            "occurrence_replay_mode": "replay",
            "occurrence_replay_fixture_digest": "fixture",
            "occurrence_replay_identity_digests": ["packet-1"],
            "occurrence_replay_complete": False,
            "occurrence_replay_prefix_valid": True,
        }
    )

    report = ANALYSIS.build_report(
        (recorder, treatment),
        expected_cases=1,
        bootstrap_samples=10,
        seed=5,
    )

    assert report["frozen_occurrence_replay"]["passed"] is True
    assert report["structural_gate_passed"] is True

    treatment["occurrence_replay_prefix_valid"] = None
    legacy_report = ANALYSIS.build_report(
        (recorder, treatment),
        expected_cases=1,
        bootstrap_samples=10,
        seed=5,
    )
    assert legacy_report["frozen_occurrence_replay"]["passed"] is True

    treatment["occurrence_replay_identity_digests"] = ["different"]
    mismatched = ANALYSIS.build_report(
        (recorder, treatment),
        expected_cases=1,
        bootstrap_samples=10,
        seed=5,
    )
    assert mismatched["frozen_occurrence_replay"]["passed"] is False
    assert mismatched["structural_gate_passed"] is False
    assert AUDIT._replay_parity(
        {
            "a0": {
                "c1": {
                    "mode": "record",
                    "fixture_digest": "fixture",
                    "complete": True,
                    "prefix_valid": True,
                    "identity_digests": ("packet-1", "packet-2"),
                }
            },
            "a3": {
                "c1": {
                    "mode": "replay",
                    "fixture_digest": "fixture",
                    "complete": False,
                    "identity_digests": ("packet-1",),
                }
            },
        }
    ) is True


def test_frozen_replay_prime_requires_every_replay_arm_to_consume_seed() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    rows = []
    for arm in ("a0", "a3"):
        row = _row(arm, "c1", score=0.0, signature=signature)
        row.update(
            {
                "occurrence_replay_mode": "replay",
                "occurrence_replay_fixture_digest": "fixture",
                "occurrence_replay_identity_digests": ["packet-1"],
                "occurrence_replay_prime_configured": True,
                "occurrence_replay_prime_requested": True,
                "occurrence_replay_prime_consumed": True,
                "occurrence_replay_prime_event_count": 1,
                "occurrence_replay_prime_event_completed": True,
                "occurrence_replay_prime_event_pre_reasoner": True,
            }
        )
        rows.append(row)

    report = ANALYSIS.build_report(
        tuple(rows), expected_cases=1, bootstrap_samples=10, seed=7
    )
    assert report["frozen_occurrence_replay"]["prime_passed"] is True
    assert report["structural_gate_passed"] is True

    rows[-1]["occurrence_replay_prime_consumed"] = False
    failed = ANALYSIS.build_report(
        tuple(rows), expected_cases=1, bootstrap_samples=10, seed=7
    )
    assert failed["frozen_occurrence_replay"]["prime_passed"] is False
    assert failed["structural_gate_passed"] is False
    assert AUDIT._replay_prime_gate(
        {
            "a0": {
                "c1": {
                    "mode": "replay",
                    "prime_configured": True,
                    "prime_requested": True,
                    "prime_consumed": True,
                    "prime_event_count": 1,
                    "prime_event_completed": True,
                    "prime_event_pre_reasoner": True,
                }
            },
            "a3": {
                "c1": {
                    "mode": "replay",
                    "prime_configured": True,
                    "prime_requested": True,
                    "prime_consumed": False,
                    "prime_event_count": 1,
                    "prime_event_completed": True,
                    "prime_event_pre_reasoner": True,
                }
            },
        }
    ) is False

    rows[-1]["occurrence_replay_prime_consumed"] = True
    rows[-1]["occurrence_replay_prime_event_pre_reasoner"] = False
    late = ANALYSIS.build_report(
        tuple(rows), expected_cases=1, bootstrap_samples=10, seed=7
    )
    assert late["frozen_occurrence_replay"]["prime_passed"] is False
    assert late["structural_gate_passed"] is False
