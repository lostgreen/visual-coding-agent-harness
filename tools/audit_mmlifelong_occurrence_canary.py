#!/usr/bin/env python3
"""Compact structural audit for no-oracle occurrence-method runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_ARMS = {"a0", "a1-flat", "a1", "a2", "a2-clean", "a3"}
SCOPED_ARMS = {"a2-clean", "a3"}
EXPECTED_LIFECYCLE_RETRY_CODES = {
    "occurrence_selection_required",
    "occurrence_answer_required_after_selection",
    "occurrence_resolution_required",
    "occurrence_search_required",
    "occurrence_no_match_required_at_finalization",
    "occurrence_answer_required_after_resolution",
    "occurrence_locator_inspection_required",
    "occurrence_locator_binding_required",
    "occurrence_locator_unbound_window_forbidden",
}


def audit_roots(
    bindings: Mapping[str, Path], *, expected_cases: int
) -> dict[str, Any]:
    per_arm: dict[str, dict[str, Any]] = {}
    case_sets: dict[str, set[str]] = {}
    replay_cases: dict[str, dict[str, dict[str, Any]]] = {}
    resolution_signatures: dict[str, dict[str, tuple[Any, ...]]] = {}
    matched_response_cases: dict[str, dict[str, dict[str, Any]]] = {}
    for declared_arm, root in bindings.items():
        cases = []
        for prediction_path in sorted(Path(root).glob("cases/*/prediction.json")):
            run_dir = prediction_path.parent
            config = _read_json(run_dir / "run_config.json")
            runtime = _read_json(run_dir / "runtime_summary.json")
            raw_no_oracle_audit = runtime.get("no_oracle_runtime_gate", {})
            no_oracle_audit = (
                raw_no_oracle_audit
                if isinstance(raw_no_oracle_audit, Mapping)
                else {}
            )
            raw_replay = no_oracle_audit.get("occurrence_replay", {})
            replay = raw_replay if isinstance(raw_replay, Mapping) else {}
            replay_mode = str(replay.get("mode", "live") or "live")
            trace = tuple(
                row
                for row in tuple(runtime.get("trace", ()) or ())
                if isinstance(row, Mapping)
            )
            decisions = tuple(
                row for row in trace if row.get("type") == "reasoner_decision"
            )
            replay_prime_events = tuple(
                (index, row)
                for index, row in enumerate(trace)
                if row.get("type") == "occurrence_replay_primed"
            )
            first_reasoner_index = next(
                (
                    index
                    for index, row in enumerate(trace)
                    if row.get("type") == "reasoner_decision"
                ),
                len(trace),
            )
            replay_prime_event_completed = bool(
                len(replay_prime_events) == 1
                and replay_prime_events[0][1].get("completed") is True
                and int(replay_prime_events[0][1].get("round", -1)) == 0
            )
            replay_prime_event_pre_reasoner = bool(
                len(replay_prime_events) == 1
                and replay_prime_events[0][0] < first_reasoner_index
            )
            arm = str(config.get("occurrence_method_arm", "none") or "none")
            state = _read_json(run_dir / "occurrence_resolution_state.json")
            matched_response = _read_json(
                run_dir / "matched_response_cache.json"
            )
            observations = _read_jsonl(run_dir / "observation_log.jsonl")
            occurrence_errors = tuple(
                error
                for row in trace
                if row.get("type") == "decision_schema_error"
                for error in tuple(row.get("errors", ()) or ())
                if isinstance(error, Mapping)
                and str(error.get("code", "")).startswith("occurrence_")
            )
            occurrence_validation_errors = tuple(
                error
                for error in occurrence_errors
                if str(error.get("code", ""))
                not in EXPECTED_LIFECYCLE_RETRY_CODES
            )
            terminal_occurrence_failures = tuple(
                error
                for row in trace
                if row.get("type") == "decision_control_exhausted"
                for error in tuple(row.get("errors", ()) or ())
                if isinstance(error, Mapping)
                and str(error.get("code", "")).startswith("occurrence_")
            )
            selection_ops = sum(
                str(operation.get("op", operation.get("type", "")) or "")
                .casefold()
                == "select"
                for row in decisions
                for operation in tuple(row.get("occurrence_ops", ()) or ())
                if isinstance(operation, Mapping)
            )
            accepted_ops = tuple(
                (index, dict(operation))
                for index, row in enumerate(decisions)
                if row.get("occurrence_ops_accepted") is not False
                for operation in tuple(row.get("occurrence_ops", ()) or ())
                if isinstance(operation, Mapping)
            )
            prior_selection_indices = tuple(
                index
                for index, row in enumerate(decisions)
                if row.get("action") != "answer"
                and row.get("occurrence_ops_accepted") is not False
                and any(
                    str(
                        operation.get("op", operation.get("type", "")) or ""
                    ).casefold()
                    == "select"
                    for operation in tuple(row.get("occurrence_ops", ()) or ())
                    if isinstance(operation, Mapping)
                )
            )
            selection_before_answer = bool(prior_selection_indices)
            answer_after_selection = any(
                row.get("action") == "answer"
                and index > prior_selection_indices[0]
                for index, row in enumerate(decisions)
            ) if prior_selection_indices else False
            activation_events = tuple(
                row
                for row in trace
                if row.get("type") == "occurrence_arbitration_activated"
            )
            resolution_activation_events = tuple(
                row
                for row in trace
                if row.get("type") == "occurrence_resolution_activated"
            )
            activation_round = (
                min(
                    int(row.get("round", 0) or 0)
                    for row in resolution_activation_events
                )
                if resolution_activation_events
                else None
            )
            state_exposure_before_activation = sum(
                bool(row.get("occurrence_resolution_state_exposed"))
                and (
                    activation_round is None
                    or int(row.get("round", 0) or 0) < activation_round
                )
                for row in decisions
            )
            selection_required = (
                bool(resolution_activation_events)
                if arm in SCOPED_ARMS
                else any(
                    row.get("type") == "occurrence_treatment_eligible"
                    and int(row.get("visible_occurrence_count", 0) or 0) > 1
                    for row in trace
                )
            )
            resolution_indices = tuple(
                index
                for index, operation in accepted_ops
                if _operation_name(operation) in {"select", "no_match"}
            )
            answer_indices = tuple(
                index
                for index, row in enumerate(decisions)
                if row.get("action") == "answer"
            )
            resolution_before_answer = bool(
                resolution_indices
                and any(index > resolution_indices[0] for index in answer_indices)
            )
            final_active_resolution = str(
                state.get("active_resolution", "") or ""
            )
            lifecycle_complete = bool(
                not selection_required
                or (
                    resolution_before_answer
                    and (
                        arm not in SCOPED_ARMS
                        or final_active_resolution in {"selected", "no_match"}
                    )
                )
            )
            scoped_sets = _scoped_state_sets(state)
            locator_scope_single_set_passed = _locator_scope_single_set_passed(
                state
            )
            scoped_ops_have_set_id = all(
                bool(_operation_set_id(operation))
                for _, operation in accepted_ops
            )
            scoped_candidate_integrity = all(
                _scoped_operation_valid(operation, scoped_sets)
                for _, operation in accepted_ops
            )
            selected_locator_pairs = _selected_locator_pairs(trace)
            task_pairs_before_answer = {
                (
                    str(task.get("locator_attempt_id", "") or ""),
                    str(task.get("occurrence_id", "") or ""),
                )
                for index, row in enumerate(decisions)
                if not answer_indices or index < answer_indices[-1]
                for task in tuple(row.get("tasks", ()) or ())
                if isinstance(task, Mapping)
                and task.get("locator_attempt_id")
                and task.get("occurrence_id")
            }
            executed_binding_pairs = _executed_binding_pairs(observations)
            locator_accounting = _locator_accounting(
                selected_locator_pairs,
                executed_binding_pairs,
                trace,
            )
            selected_locators_inspected = bool(
                selected_locator_pairs.issubset(task_pairs_before_answer)
                and selected_locator_pairs.issubset(executed_binding_pairs)
            )
            rejected_occurrence_op_attempts = sum(
                row.get("occurrence_ops_accepted") is False for row in decisions
            )
            case_id = str(_read_json(prediction_path).get("case_id", run_dir.name))
            cases.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "models": config.get("models"),
                    "no_oracle": bool(
                        no_oracle_audit.get(
                            "no_oracle_runtime_gate_passed", False
                        )
                    ),
                    "text_budget_parity": (
                        bool(no_oracle_audit.get("text_budget_parity_passed"))
                        if arm != "a0"
                        else None
                    ),
                    "replay_mode": replay_mode,
                    "replay_fixture_digest": str(
                        replay.get("fixture_digest", "") or ""
                    ),
                    "replay_complete": (
                        replay.get("consumption_complete") is True
                        if replay_mode == "replay"
                        else True
                    ),
                    "replay_prefix_valid": (
                        replay.get("consumed_prefix_valid")
                        if replay_mode == "replay"
                        else True
                    ),
                    "replay_prime_configured": bool(
                        config.get("occurrence_replay_prime", False)
                    ),
                    "replay_prime_requested": bool(
                        replay.get("prime_requested", False)
                    ),
                    "replay_prime_consumed": bool(
                        replay.get("prime_consumed", False)
                    ),
                    "replay_prime_event_count": len(replay_prime_events),
                    "replay_prime_event_completed": (
                        replay_prime_event_completed
                    ),
                    "replay_prime_event_pre_reasoner": (
                        replay_prime_event_pre_reasoner
                    ),
                    "replay_post_fixture_reuse_count": int(
                        replay.get("post_fixture_reuse_count", 0) or 0
                    ),
                    "replay_identity_digests": tuple(
                        replay.get("consumed_identity_digests", ())
                        if replay_mode == "replay"
                        else no_oracle_audit.get(
                            "retrieval_identity_digests", ()
                        )
                    ),
                    "eligible_events": sum(
                        row.get("type") == "occurrence_treatment_eligible"
                        for row in trace
                    ),
                    "exposure_events": sum(
                        row.get("type") == "occurrence_treatment_exposed"
                        for row in trace
                    ),
                    "occurrence_ops": sum(
                        len(tuple(row.get("occurrence_ops", ()) or ()))
                        for row in decisions
                    ),
                    "selection_ops": selection_ops,
                    "selection_required": selection_required,
                    "selection_before_answer": selection_before_answer,
                    "answer_after_selection": answer_after_selection,
                    "resolution_before_answer": resolution_before_answer,
                    "lifecycle_complete": lifecycle_complete,
                    "arbitration_activation_events": len(activation_events),
                    "resolution_activation_events": len(
                        resolution_activation_events
                    ),
                    "resolution_activation_threshold_valid": all(
                        int(row.get("candidate_count", 0) or 0) >= 1
                        for row in resolution_activation_events
                    ),
                    "arbitration_activation_threshold_valid": all(
                        int(row.get("candidate_count", 0) or 0) >= 2
                        for row in activation_events
                    ),
                    "state_exposure_before_activation": (
                        state_exposure_before_activation
                    ),
                    "scoped_ops_have_set_id": scoped_ops_have_set_id,
                    "scoped_candidate_integrity": scoped_candidate_integrity,
                    "locator_scope_single_set_passed": (
                        locator_scope_single_set_passed
                    ),
                    "defer_ops": sum(
                        _operation_name(operation) == "defer"
                        for _, operation in accepted_ops
                    ),
                    "no_match_ops": sum(
                        _operation_name(operation) == "no_match"
                        for _, operation in accepted_ops
                    ),
                    "multi_selection_sets": sum(
                        len(value["selected_occurrence_ids"]) > 1
                        for value in scoped_sets.values()
                    ),
                    "selected_locator_count": len(selected_locator_pairs),
                    "selected_locator_inspected_count": len(
                        selected_locator_pairs & executed_binding_pairs
                    ),
                    "selected_locators_inspected": selected_locators_inspected,
                    "selected_locators_accounted": locator_accounting[
                        "accounted"
                    ],
                    "selected_locator_silent_drop_count": locator_accounting[
                        "silent_drop_count"
                    ],
                    "selected_locator_accounting_conflict_count": (
                        locator_accounting["conflict_count"]
                    ),
                    "selected_locator_release_counts": locator_accounting[
                        "release_counts"
                    ],
                    "contradictory_gate_state_count": sum(
                        row.get("type") == "contradictory_gate_state"
                        for row in trace
                    ),
                    "resolution_signature": _resolution_signature(
                        state,
                        accepted_ops,
                    ),
                    "ops_accepted": all(
                        row.get("occurrence_ops_accepted") is not False
                        for row in decisions
                    ),
                    "rejected_occurrence_op_attempts": (
                        rejected_occurrence_op_attempts
                    ),
                    "occurrence_schema_errors": len(occurrence_errors),
                    "selection_required_retries": sum(
                        str(error.get("code", ""))
                        == "occurrence_selection_required"
                        for error in occurrence_errors
                    ),
                    "answer_required_retries": sum(
                        str(error.get("code", ""))
                        == "occurrence_answer_required_after_selection"
                        for error in occurrence_errors
                    ),
                    "occurrence_validation_errors": len(
                        occurrence_validation_errors
                    ),
                    "terminal_occurrence_failures": len(
                        terminal_occurrence_failures
                    ),
                    "state_file": (
                        run_dir / "occurrence_resolution_state.json"
                    ).is_file(),
                    "premature_commits": sum(
                        bool(row.get("premature_occurrence_commit"))
                        for row in decisions
                    ),
                    "matched_response": matched_response,
                }
            )
        case_sets[declared_arm] = {row["case_id"] for row in cases}
        replay_cases[declared_arm] = {
            row["case_id"]: {
                "mode": row["replay_mode"],
                "fixture_digest": row["replay_fixture_digest"],
                "complete": row["replay_complete"],
                "prefix_valid": row["replay_prefix_valid"],
                "identity_digests": row["replay_identity_digests"],
                "prime_configured": row["replay_prime_configured"],
                "prime_requested": row["replay_prime_requested"],
                "prime_consumed": row["replay_prime_consumed"],
                "prime_event_count": row["replay_prime_event_count"],
                "prime_event_completed": row[
                    "replay_prime_event_completed"
                ],
                "prime_event_pre_reasoner": row[
                    "replay_prime_event_pre_reasoner"
                ],
            }
            for row in cases
        }
        resolution_signatures[declared_arm] = {
            row["case_id"]: row["resolution_signature"] for row in cases
        }
        matched_response_cases[declared_arm] = {
            row["case_id"]: dict(row["matched_response"])
            for row in cases
            if row["matched_response"]
        }
        per_arm[declared_arm] = {
            "n": len(cases),
            "declared_arm_match": all(
                row["arm"] == declared_arm for row in cases
            ),
            "models_ok": all(
                row["models"]
                == {
                    "reasoner": "pa/gmn-2.5-pr",
                    "investigator": "pa/gmn-2.5-pr",
                }
                for row in cases
            ),
            "no_oracle_gate_passed": all(row["no_oracle"] for row in cases),
            "text_budget_parity_passed": all(
                row["text_budget_parity"] is True
                for row in cases
                if row["arm"] != "a0"
            ),
            "eligible_event_count": sum(
                row["eligible_events"] for row in cases
            ),
            "exposure_event_count": sum(
                row["exposure_events"] for row in cases
            ),
            "duplicate_eligibility_event_case_count": sum(
                row["eligible_events"] > 1 for row in cases
            ),
            "duplicate_exposure_event_case_count": sum(
                row["exposure_events"] > 1 for row in cases
            ),
            "arbitration_activation_case_count": sum(
                row["arbitration_activation_events"] > 0 for row in cases
            ),
            "resolution_activation_case_count": sum(
                row["resolution_activation_events"] > 0 for row in cases
            ),
            "duplicate_arbitration_activation_case_count": sum(
                row["arbitration_activation_events"] > 1 for row in cases
            ),
            "duplicate_resolution_activation_case_count": sum(
                row["resolution_activation_events"] > 1 for row in cases
            ),
            "activation_threshold_failure_case_count": sum(
                not row["resolution_activation_threshold_valid"]
                or not row["arbitration_activation_threshold_valid"]
                for row in cases
            ),
            "pre_activation_state_exposure_count": sum(
                row["state_exposure_before_activation"] for row in cases
            ),
            "eligible_without_exposure_case_count": sum(
                row["eligible_events"] > 0 and row["exposure_events"] == 0
                for row in cases
            ),
            "exposure_without_eligibility_case_count": sum(
                row["exposure_events"] > 0 and row["eligible_events"] == 0
                for row in cases
            ),
            "occurrence_op_count": sum(row["occurrence_ops"] for row in cases),
            "selection_case_count": sum(
                row["selection_ops"] > 0 for row in cases
            ),
            "selection_required_case_count": sum(
                row["selection_required"] for row in cases
            ),
            "selection_missing_case_count": sum(
                row["selection_required"] and row["selection_ops"] == 0
                for row in cases
            ),
            "selection_not_prior_case_count": sum(
                row["selection_required"]
                and not row["selection_before_answer"]
                for row in cases
            ),
            "answer_missing_after_selection_case_count": sum(
                row["selection_required"]
                and not row["resolution_before_answer"]
                for row in cases
            ),
            "answer_missing_after_resolution_case_count": sum(
                row["selection_required"]
                and not row["resolution_before_answer"]
                for row in cases
            ),
            "resolution_missing_before_answer_case_count": sum(
                row["selection_required"]
                and not row["resolution_before_answer"]
                for row in cases
            ),
            "scoped_set_id_failure_case_count": sum(
                not row["scoped_ops_have_set_id"] for row in cases
            ),
            "scoped_candidate_integrity_failure_case_count": sum(
                not row["scoped_candidate_integrity"] for row in cases
            ),
            "locator_scope_single_set_failure_case_count": sum(
                not row["locator_scope_single_set_passed"] for row in cases
            ),
            "defer_op_count": sum(row["defer_ops"] for row in cases),
            "no_match_op_count": sum(row["no_match_ops"] for row in cases),
            "multi_selection_set_count": sum(
                row["multi_selection_sets"] for row in cases
            ),
            "selected_locator_count": sum(
                row["selected_locator_count"] for row in cases
            ),
            "selected_locator_inspected_count": sum(
                row["selected_locator_inspected_count"] for row in cases
            ),
            "selected_locator_inspection_failure_case_count": sum(
                row["selected_locator_count"] > 0
                and not row["selected_locators_inspected"]
                for row in cases
            ),
            "selected_locator_accounting_failure_case_count": sum(
                row["selected_locator_count"] > 0
                and not row["selected_locators_accounted"]
                for row in cases
            ),
            "selected_locator_silent_drop_count": sum(
                row["selected_locator_silent_drop_count"] for row in cases
            ),
            "selected_locator_accounting_conflict_count": sum(
                row["selected_locator_accounting_conflict_count"]
                for row in cases
            ),
            "selected_locator_release_counts": {
                outcome: sum(
                    row["selected_locator_release_counts"].get(outcome, 0)
                    for row in cases
                )
                for outcome in (
                    "released_at_budget_exhaustion",
                    "released_on_set_retirement",
                    "released_by_revision",
                )
            },
            "contradictory_gate_state_count": sum(
                row["contradictory_gate_state_count"] for row in cases
            ),
            "all_occurrence_ops_accepted": all(
                row["ops_accepted"] for row in cases
            ),
            "rejected_occurrence_op_attempt_count": sum(
                row["rejected_occurrence_op_attempts"] for row in cases
            ),
            "recovered_occurrence_op_rejection_case_count": sum(
                row["rejected_occurrence_op_attempts"] > 0
                and row["lifecycle_complete"]
                for row in cases
            ),
            "unrecovered_occurrence_op_rejection_case_count": sum(
                row["rejected_occurrence_op_attempts"] > 0
                and not row["lifecycle_complete"]
                for row in cases
            ),
            "occurrence_schema_error_count": sum(
                row["occurrence_schema_errors"] for row in cases
            ),
            "selection_required_retry_count": sum(
                row["selection_required_retries"] for row in cases
            ),
            "answer_required_retry_count": sum(
                row["answer_required_retries"] for row in cases
            ),
            "occurrence_validation_error_count": sum(
                row["occurrence_validation_errors"] for row in cases
            ),
            "unrecovered_occurrence_validation_error_case_count": sum(
                row["occurrence_validation_errors"] > 0
                and not row["lifecycle_complete"]
                for row in cases
            ),
            "terminal_occurrence_failure_count": sum(
                row["terminal_occurrence_failures"] for row in cases
            ),
            "state_file_count": sum(row["state_file"] for row in cases),
            "premature_commit_count": sum(
                row["premature_commits"] for row in cases
            ),
            "occurrence_replay_modes": sorted(
                {row["replay_mode"] for row in cases}
            ),
            "occurrence_replay_complete": all(
                row["replay_complete"] for row in cases
            ),
            "occurrence_replay_prime_configured": all(
                row["replay_prime_configured"] for row in cases
            ),
            "occurrence_replay_prime_consumed": all(
                row["replay_prime_consumed"] for row in cases
            ),
            "occurrence_replay_prime_event_completed": all(
                row["replay_prime_event_completed"] for row in cases
            ),
            "occurrence_replay_prime_event_pre_reasoner": all(
                row["replay_prime_event_pre_reasoner"] for row in cases
            ),
            "occurrence_replay_post_fixture_reuse_count": sum(
                row["replay_post_fixture_reuse_count"] for row in cases
            ),
            "matched_response_modes": sorted(
                {
                    str(row["matched_response"].get("mode", ""))
                    for row in cases
                    if row["matched_response"]
                }
            ),
            "matched_response_recorded_count": sum(
                int(row["matched_response"].get("recorded_count", 0) or 0)
                for row in cases
            ),
            "matched_response_replayed_count": sum(
                int(row["matched_response"].get("replayed_count", 0) or 0)
                for row in cases
            ),
            "matched_response_mismatch_count": sum(
                int(row["matched_response"].get("mismatch_count", 0) or 0)
                for row in cases
            ),
        }
    common = set.intersection(*case_sets.values()) if case_sets else set()
    treatment_arms = tuple(arm for arm in bindings if arm != "a0")
    scoped_arms = tuple(arm for arm in bindings if arm in SCOPED_ARMS)
    post_selection_balance = _post_selection_only_divergence(
        resolution_signatures
    )
    matched_response_gate = _matched_pre_treatment_response_gate(
        matched_response_cases
    )
    checks = {
        "expected_arms_present": bool(bindings)
        and set(bindings).issubset(SUPPORTED_ARMS),
        "case_sets_aligned": bool(case_sets)
        and all(case_set == common for case_set in case_sets.values()),
        "expected_case_count": len(common) == int(expected_cases),
        "declared_arms_match": all(
            row["declared_arm_match"] for row in per_arm.values()
        ),
        "actual_models_match": all(row["models_ok"] for row in per_arm.values()),
        "no_oracle_gate_passed": all(
            row["no_oracle_gate_passed"] for row in per_arm.values()
        ),
        "eligibility_event_integrity": all(
            row["duplicate_eligibility_event_case_count"] == 0
            for row in per_arm.values()
        ),
        "treatment_exposure_integrity": all(
            per_arm[arm]["eligible_without_exposure_case_count"] == 0
            and per_arm[arm]["exposure_without_eligibility_case_count"] == 0
            and per_arm[arm]["duplicate_exposure_event_case_count"] == 0
            for arm in treatment_arms
        ),
        "a1_flat_same_packet_text_budget_parity": all(
            per_arm[arm]["text_budget_parity_passed"]
            for arm in ("a1-flat", "a1")
            if arm in per_arm
        ),
        "a0_has_no_treatment_exposure": (
            "a0" not in per_arm
            or per_arm["a0"].get("exposure_event_count") == 0
        ),
        "a2_state_files_complete": (
            "a2" not in per_arm
            or per_arm["a2"].get("state_file_count") == int(expected_cases)
        ),
        "a2_selection_complete": (
            "a2" not in per_arm
            or (
                per_arm["a2"].get("selection_required_case_count", 0) > 0
                and per_arm["a2"].get("selection_missing_case_count") == 0
                and per_arm["a2"].get("selection_not_prior_case_count") == 0
                and per_arm["a2"].get(
                    "answer_missing_after_selection_case_count"
                )
                == 0
            )
        ),
        "scoped_resolution_activated": all(
            per_arm[arm]["resolution_activation_case_count"] > 0
            for arm in scoped_arms
        ),
        "scoped_activation_unique": all(
            per_arm[arm]["duplicate_resolution_activation_case_count"] == 0
            and per_arm[arm]["duplicate_arbitration_activation_case_count"] == 0
            for arm in scoped_arms
        ),
        "scoped_activation_thresholds_valid": all(
            per_arm[arm]["activation_threshold_failure_case_count"] == 0
            for arm in scoped_arms
        ),
        "no_pre_activation_state_exposure": all(
            per_arm[arm]["pre_activation_state_exposure_count"] == 0
            for arm in scoped_arms
        ),
        "scoped_set_integrity": all(
            per_arm[arm]["scoped_set_id_failure_case_count"] == 0
            and per_arm[arm]["scoped_candidate_integrity_failure_case_count"]
            == 0
            for arm in scoped_arms
        ),
        "locator_scope_single_set_passed": all(
            per_arm[arm]["locator_scope_single_set_failure_case_count"] == 0
            for arm in scoped_arms
        ),
        "scoped_resolution_complete": all(
            per_arm[arm]["resolution_missing_before_answer_case_count"] == 0
            for arm in scoped_arms
        ),
        "a3_selected_locators_accounted": (
            "a3" not in per_arm
            or (
                per_arm["a3"]["selected_locator_count"] > 0
                and per_arm["a3"][
                    "selected_locator_accounting_failure_case_count"
                ]
                == 0
                and per_arm["a3"]["selected_locator_silent_drop_count"] == 0
                and per_arm["a3"][
                    "selected_locator_accounting_conflict_count"
                ]
                == 0
            )
        ),
        "post_selection_only_divergence": (
            post_selection_balance["passed"]
            if post_selection_balance["applicable"]
            else True
        ),
        "matched_pre_treatment_responses": (
            matched_response_gate["passed"]
            if matched_response_gate["applicable"]
            else True
        ),
        "no_contradictory_gate_states": all(
            row["contradictory_gate_state_count"] == 0
            for row in per_arm.values()
        ),
        "no_premature_occurrence_commits": all(
            row["premature_commit_count"] == 0 for row in per_arm.values()
        ),
        "frozen_occurrence_replay_parity": _replay_parity(replay_cases),
        "frozen_occurrence_replay_prime": _replay_prime_gate(replay_cases),
        "no_unrecovered_occurrence_validation_errors": sum(
            row["unrecovered_occurrence_validation_error_case_count"]
            for row in per_arm.values()
        )
        == 0,
        "no_terminal_occurrence_failures": sum(
            row["terminal_occurrence_failure_count"]
            for row in per_arm.values()
        )
        == 0,
        "no_unrecovered_occurrence_op_rejections": sum(
            row["unrecovered_occurrence_op_rejection_case_count"]
            for row in per_arm.values()
        )
        == 0,
    }
    return {
        "schema_version": "MMLifelongOccurrenceCanaryAuditV5",
        "per_arm": per_arm,
        "post_selection_only_divergence": post_selection_balance,
        "matched_pre_treatment_responses": matched_response_gate,
        "checks": checks,
        "structural_gate_passed": all(checks.values()),
    }


def _operation_name(operation: Mapping[str, Any]) -> str:
    return str(operation.get("op", operation.get("type", "")) or "").casefold()


def _operation_set_id(operation: Mapping[str, Any]) -> str:
    return str(
        operation.get("set_id", operation.get("locator_attempt_id", "")) or ""
    )


def _scoped_state_sets(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for raw_set in tuple(state.get("sets", ()) or ()):
        if not isinstance(raw_set, Mapping):
            continue
        set_id = str(raw_set.get("set_id", "") or "")
        if not set_id:
            continue
        candidates = {
            str(candidate.get("occurrence_id", "") or "")
            for candidate in tuple(raw_set.get("candidates", ()) or ())
            if isinstance(candidate, Mapping)
            and str(candidate.get("occurrence_id", "") or "")
        }
        selected = tuple(
            str(value)
            for value in tuple(raw_set.get("selected_occurrence_ids", ()) or ())
            if str(value)
        )
        values[set_id] = {
            "candidate_ids": candidates,
            "selected_occurrence_ids": selected,
            "resolution": str(raw_set.get("resolution", "") or ""),
            "lifecycle": str(raw_set.get("lifecycle", "") or ""),
        }
    return values


def _scoped_operation_valid(
    operation: Mapping[str, Any],
    scoped_sets: Mapping[str, Mapping[str, Any]],
) -> bool:
    set_id = _operation_set_id(operation)
    if not set_id or set_id not in scoped_sets:
        return False
    op = _operation_name(operation)
    occurrence_id = str(operation.get("occurrence_id", "") or "")
    if op in {"defer", "no_match"}:
        return not occurrence_id
    return bool(
        occurrence_id
        and occurrence_id in scoped_sets[set_id].get("candidate_ids", set())
    )


def _selected_locator_pairs(
    trace: tuple[Mapping[str, Any], ...],
) -> set[tuple[str, str]]:
    return {
        (_operation_set_id(operation), str(operation.get("occurrence_id", "") or ""))
        for row in trace
        if row.get("type") == "reasoner_decision"
        and row.get("occurrence_ops_accepted") is not False
        for operation in tuple(row.get("occurrence_ops", ()) or ())
        if isinstance(operation, Mapping)
        and _operation_name(operation) == "select"
        and _operation_set_id(operation)
        and str(operation.get("occurrence_id", "") or "")
    }


def _locator_accounting(
    selected_pairs: set[tuple[str, str]],
    executed_pairs: set[tuple[str, str]],
    trace: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    release_outcomes: dict[tuple[str, str], set[str]] = {}
    explicit_conflicts: set[tuple[str, str]] = set()
    for row in trace:
        event_type = str(row.get("type", "") or "")
        pair = (
            str(row.get("locator_attempt_id", "") or ""),
            str(row.get("occurrence_id", "") or ""),
        )
        if not all(pair):
            continue
        if event_type == "occurrence_locator_released_unexecuted":
            outcome = str(row.get("outcome", "") or "")
            if outcome:
                release_outcomes.setdefault(pair, set()).add(outcome)
        elif event_type == "occurrence_locator_accounting_conflict":
            explicit_conflicts.add(pair)

    outcomes_by_pair: dict[tuple[str, str], set[str]] = {}
    for pair in selected_pairs:
        outcomes = set(release_outcomes.get(pair, set()))
        if pair in executed_pairs:
            outcomes.add("inspected")
        outcomes_by_pair[pair] = outcomes
    silent_pairs = {
        pair for pair, outcomes in outcomes_by_pair.items() if not outcomes
    }
    conflict_pairs = {
        pair for pair, outcomes in outcomes_by_pair.items() if len(outcomes) > 1
    } | (explicit_conflicts & selected_pairs)
    release_counts = {
        outcome: sum(
            outcomes == {outcome} for outcomes in outcomes_by_pair.values()
        )
        for outcome in (
            "released_at_budget_exhaustion",
            "released_on_set_retirement",
            "released_by_revision",
        )
    }
    return {
        "accounted": not silent_pairs and not conflict_pairs,
        "silent_drop_count": len(silent_pairs),
        "conflict_count": len(conflict_pairs),
        "release_counts": release_counts,
    }


def _resolution_signature(
    state: Mapping[str, Any],
    accepted_ops: tuple[tuple[int, dict[str, Any]], ...],
) -> tuple[Any, ...]:
    resolution_ops = tuple(
        operation
        for _, operation in accepted_ops
        if _operation_name(operation) in {"select", "no_match"}
    )
    active_set_id = str(state.get("active_set_id", "") or "")
    resolved_set_id = (
        _operation_set_id(resolution_ops[-1]) if resolution_ops else active_set_id
    )
    scoped_sets = _scoped_state_sets(state)
    resolved = scoped_sets.get(resolved_set_id, {})
    resolution = str(resolved.get("resolution", "") or "")
    if not resolution and resolved_set_id == active_set_id:
        resolution = str(state.get("active_resolution", "") or "")
    selected = tuple(resolved.get("selected_occurrence_ids", ()) or ())
    return (resolved_set_id, resolution, selected)


def _post_selection_only_divergence(
    signatures: Mapping[str, Mapping[str, tuple[Any, ...]]],
) -> dict[str, Any]:
    if "a2-clean" not in signatures or "a3" not in signatures:
        return {
            "applicable": False,
            "passed": None,
            "paired_case_count": 0,
            "mismatch_case_ids": [],
        }
    clean = signatures["a2-clean"]
    actionable = signatures["a3"]
    paired = sorted(set(clean) & set(actionable))
    mismatches = [
        case_id for case_id in paired if clean[case_id] != actionable[case_id]
    ]
    return {
        "applicable": True,
        "passed": set(clean) == set(actionable) and not mismatches,
        "paired_case_count": len(paired),
        "mismatch_case_ids": mismatches,
    }


def _matched_pre_treatment_response_gate(
    cases: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    clean = cases.get("a2-clean", {})
    actionable = cases.get("a3", {})
    applicable = bool(clean or actionable)
    if not applicable:
        return {
            "applicable": False,
            "passed": None,
            "paired_case_count": 0,
            "mismatch_case_ids": [],
        }
    paired = sorted(set(clean) & set(actionable))
    mismatches = []
    for case_id in paired:
        recorded = clean[case_id]
        replayed = actionable[case_id]
        if not all(
            (
                recorded.get("mode") == "record",
                replayed.get("mode") == "replay",
                dict(recorded.get("recorded", {}) or {})
                == dict(replayed.get("replayed", {}) or {}),
                int(recorded.get("mismatch_count", 0) or 0) == 0,
                int(replayed.get("mismatch_count", 0) or 0) == 0,
                recorded.get("active") is False,
                replayed.get("active") is False,
                recorded.get("deactivation_reason")
                == "scoped_occurrence_resolution_persisted",
                replayed.get("deactivation_reason")
                == "scoped_occurrence_resolution_persisted",
            )
        ):
            mismatches.append(case_id)
    return {
        "applicable": True,
        "passed": bool(paired)
        and set(clean) == set(actionable)
        and not mismatches,
        "paired_case_count": len(paired),
        "mismatch_case_ids": mismatches,
    }


def _locator_scope_single_set_passed(state: Mapping[str, Any]) -> bool:
    scoped_sets = _scoped_state_sets(state)
    if not scoped_sets:
        return not state.get("active_set_id")
    active_set_id = str(state.get("active_set_id", "") or "")
    active_set_ids = {
        set_id
        for set_id, value in scoped_sets.items()
        if value.get("lifecycle") == "active"
    }
    retired_set_ids = {
        set_id
        for set_id, value in scoped_sets.items()
        if value.get("lifecycle") == "retired"
    }
    serialized_retired = {
        str(value)
        for value in tuple(state.get("retired_set_ids", ()) or ())
        if str(value)
    }
    active_locators = tuple(state.get("active_locators", ()) or ())
    retired_locators = tuple(state.get("retired_locators", ()) or ())
    expected_active_pairs = {
        (active_set_id, occurrence_id)
        for set_id, value in scoped_sets.items()
        if set_id == active_set_id
        for occurrence_id in value["selected_occurrence_ids"]
    }
    expected_retired_pairs = {
        (set_id, occurrence_id)
        for set_id, value in scoped_sets.items()
        if set_id in retired_set_ids
        for occurrence_id in value["selected_occurrence_ids"]
    }
    actual_active_pairs = {
        (
            str(locator.get("set_id", "") or ""),
            str(locator.get("occurrence_id", "") or ""),
        )
        for locator in active_locators
        if isinstance(locator, Mapping)
    }
    actual_retired_pairs = {
        (
            str(locator.get("set_id", "") or ""),
            str(locator.get("occurrence_id", "") or ""),
        )
        for locator in retired_locators
        if isinstance(locator, Mapping)
    }
    return bool(active_set_id) and all(
        (
            active_set_ids == {active_set_id},
            retired_set_ids == serialized_retired,
            actual_active_pairs == expected_active_pairs,
            actual_retired_pairs == expected_retired_pairs,
            all(
                isinstance(locator, Mapping)
                and str(locator.get("set_id", "") or "") == active_set_id
                and locator.get("status") == "selected_for_active_set"
                for locator in active_locators
            ),
            all(
                isinstance(locator, Mapping)
                and str(locator.get("set_id", "") or "") in retired_set_ids
                and locator.get("status") == "retired_history"
                for locator in retired_locators
            ),
        )
    )


def _executed_binding_pairs(
    observations: tuple[dict[str, Any], ...],
) -> set[tuple[str, str]]:
    pairs = set()
    for row in observations:
        config = row.get("sampling_config")
        binding = config.get("candidate_binding") if isinstance(config, Mapping) else None
        if not isinstance(binding, Mapping):
            continue
        locator_attempt_id = str(binding.get("locator_attempt_id", "") or "")
        occurrence_id = str(binding.get("occurrence_id", "") or "")
        if locator_attempt_id and occurrence_id:
            pairs.add((locator_attempt_id, occurrence_id))
    return pairs


def _replay_parity(
    replay_cases: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> bool:
    replay_arms = tuple(
        arm
        for arm, cases in replay_cases.items()
        if any(
            row.get("mode") in {"record", "replay"}
            for row in cases.values()
        )
    )
    if not replay_arms:
        return True
    case_sets = [set(replay_cases[arm]) for arm in replay_arms]
    case_ids = set.intersection(*case_sets) if case_sets else set()
    if not case_ids:
        return False
    for case_id in case_ids:
        rows = [replay_cases[arm][case_id] for arm in replay_arms]
        digests = {str(row.get("fixture_digest", "") or "") for row in rows}
        sequences = [
            tuple(row.get("identity_digests", ()) or ()) for row in rows
        ]
        reference = max(sequences, key=len, default=())
        if (
            not reference
            or not all(
                sequence
                and reference[: len(sequence)] == sequence
                and row.get("prefix_valid") is not False
                for row, sequence in zip(rows, sequences)
            )
            or len(digests) != 1
            or "" in digests
        ):
            return False
    return True


def _replay_prime_gate(
    replay_cases: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> bool:
    replay_rows = [
        row
        for cases in replay_cases.values()
        for row in cases.values()
        if row.get("mode") == "replay"
    ]
    if not any(row.get("prime_configured") for row in replay_rows):
        return True
    return bool(replay_rows) and all(
        row.get("prime_configured") is True
        and row.get("prime_requested") is True
        and row.get("prime_consumed") is True
        and row.get("prime_event_count") == 1
        and row.get("prime_event_completed") is True
        and row.get("prime_event_pre_reasoner") is True
        for row in replay_rows
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not Path(path).is_file():
        return ()
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-root", action="append", nargs=2, metavar=("ARM", "ROOT"), required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    bindings = {arm: Path(root) for arm, root in args.arm_root}
    report = audit_roots(bindings, expected_cases=args.expected_cases)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "structural_gate_passed": report["structural_gate_passed"],
                "checks": report["checks"],
                "per_arm": report["per_arm"],
                "output_json": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
