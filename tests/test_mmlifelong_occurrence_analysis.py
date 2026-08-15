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
        "verified_correct": False,
        "correct_and_ref_300": score == 1.0,
        "parse_status": "parsed",
        "judge_model": "judge",
        "candidate_recall": True,
        "candidate_clue_recall": 1.0,
        "osa_eligible": arm == "a2",
        "osa_correct": arm == "a2",
        "selected_occurrence_count": 1 if arm == "a2" else 0,
        "selected_clue_recall": 1.0 if arm == "a2" else None,
        "abstention_eligible": False,
        "abstention_correct": None,
        "false_abstention": False,
        "deferred_occurrence_set": False,
        "occurrence_handle_usage_rate": 1.0,
        "selected_locator_usage_rate": None,
        "bound_visual_clue_recall": None,
        "arbitration_activation_round": None,
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
        "occurrence_replay_prefix_valid": True,
        "occurrence_replay_identity_digests": [],
        "occurrence_replay_prime_configured": False,
        "occurrence_replay_prime_requested": False,
        "occurrence_replay_prime_consumed": False,
        "occurrence_replay_prime_event_count": 0,
        "occurrence_replay_prime_event_completed": False,
        "occurrence_replay_prime_event_pre_reasoner": False,
        "occurrence_replay_post_fixture_reuse_count": 0,
        "frozen_config": {"controller_mode": "frozen_baseline"},
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


def test_budget_symmetry_fails_when_arm_mean_gap_exceeds_threshold() -> None:
    signature = [{"action": "investigate", "tasks": []}]
    left = _row("a2-clean", "c1", score=0.0, signature=signature)
    right = _row("a3", "c1", score=0.0, signature=signature)
    right["semantic_rounds_used"] = 5

    report = ANALYSIS.build_report(
        (left, right), expected_cases=1, bootstrap_samples=10, seed=3
    )

    assert report["budget_symmetry"]["max_minus_min"] == 1.0
    assert report["structural_checks"]["budget_symmetry_passed"] is False
    assert report["structural_gate_passed"] is False


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
                "candidate_recall": case_id == "c1",
                "candidate_clue_recall": 1.0 if case_id == "c1" else 0.0,
                "selected_occurrence_count": 1 if case_id == "c1" else 0,
                "selected_clue_recall": 1.0 if case_id == "c1" else None,
                "abstention_eligible": case_id == "c2",
                "abstention_correct": True if case_id == "c2" else None,
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


def test_scoped_canary_audit_requires_actionable_locator_execution(
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
                        "type": "occurrence_arbitration_activated",
                        "round": 2,
                        "active_set_id": "locator_1",
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
                        "type": "reasoner_decision",
                        "round": 3,
                        "action": "investigate",
                        "tasks": [
                            {
                                "inspection_mode": "window",
                                "locator_attempt_id": "locator_1",
                                "occurrence_id": "occ_1",
                            }
                        ],
                        "occurrence_ops": [],
                        "occurrence_ops_accepted": True,
                    }
                )
                (case / "observation_log.jsonl").write_text(
                    json.dumps(
                        {
                            "sampling_config": {
                                "candidate_binding": {
                                    "locator_attempt_id": "locator_1",
                                    "occurrence_id": "occ_1",
                                }
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
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
                        "sets": [
                            {
                                "set_id": "locator_1",
                                "resolution": "selected",
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
    assert report["checks"]["a3_selected_locators_inspected"] is True
    assert report["per_arm"]["a3"]["selected_locator_count"] == 1


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
