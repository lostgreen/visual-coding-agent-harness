#!/usr/bin/env python3
"""Analyze no-oracle occurrence-method arms without feeding gold into runtime."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from math import comb
from pathlib import Path
import random
from statistics import mean
import sys
from typing import Any, Iterable, Mapping, Sequence


FROZEN_CONFIG_KEYS = (
    "controller_mode",
    "controller_evidence_visibility",
    "measurement_control",
    "answer_policy",
    "evidence_control_mode",
    "evidence_state_mode",
    "max_rounds",
    "semantic_round_budget",
    "control_retry_budget",
    "max_investigations",
    "max_tasks_per_round",
    "caption_index_mode",
    "caption_query_strategy",
    "caption_query_policy",
    "effective_caption_query_strategy",
    "caption_config_digest",
    "occurrence_replay_prime",
    "anchor_execution_policy",
    "embedding",
    "input_digest",
    "models",
    "phase5r_mode",
    "web_enabled",
    "supporting_interval_source",
)


def collect_rows(
    run_roots: Sequence[Path], *, evaluation_record_root: Path
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root in run_roots:
        for prediction_path in sorted(Path(root).glob("cases/*/prediction.json")):
            run_dir = prediction_path.parent
            prediction = _read_json(prediction_path)
            config = _read_json(run_dir / "run_config.json")
            runtime = _read_json(run_dir / "runtime_summary.json")
            evaluation = _read_json(
                run_dir / "evaluation" / "mmlifelong_eval.json"
            )
            case_id = str(prediction.get("case_id", run_dir.name))
            arm = str(config.get("occurrence_method_arm", "none") or "none")
            if (arm, case_id) in seen:
                raise ValueError(f"duplicate occurrence result: {arm}:{case_id}")
            seen.add((arm, case_id))
            record = _read_json(
                Path(evaluation_record_root) / case_id / "evaluation_case.json"
            )
            clues = tuple(
                (float(value[0]), float(value[1]))
                for value in tuple(record.get("clue_intervals", ()) or ())
                if isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and len(value) == 2
            )
            observations = _read_jsonl(run_dir / "observation_log.jsonl")
            trace = tuple(
                row
                for row in tuple(runtime.get("trace", ()) or ())
                if isinstance(row, Mapping)
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
            state = _read_json(run_dir / "occurrence_resolution_state.json")
            matched_response = _read_json(
                run_dir / "matched_response_cache.json"
            )
            occurrence_metrics = _occurrence_resolution_metrics(
                arm=arm,
                state=state,
                trace=trace,
                observations=observations,
                clues=clues,
            )
            locator_scope_single_set_passed = _locator_scope_single_set_passed(
                state
            )
            selected_locator_pairs = _selected_locator_pairs(trace)
            executed_binding_pairs, bound_visual_ranges = _bound_visual_evidence(
                observations
            )
            locator_accounting = _locator_accounting(
                selected_locator_pairs,
                executed_binding_pairs,
                trace,
            )
            selected_locator_usage_rate = (
                len(selected_locator_pairs & executed_binding_pairs)
                / len(selected_locator_pairs)
                if selected_locator_pairs
                else None
            )
            released_unexecuted_count = sum(
                locator_accounting["release_counts"].values()
            )
            released_unexecuted_rate = (
                released_unexecuted_count / len(selected_locator_pairs)
                if selected_locator_pairs
                else None
            )
            answer_eval = evaluation.get("answer", {})
            grounding = evaluation.get("reference_grounding", {})
            metrics = runtime.get("runtime_metrics", {})
            score = answer_eval.get("score")
            ref_300 = grounding.get("ref_300")
            bound_visual_clue_recall = _interval_recall(
                bound_visual_ranges, clues
            )
            eligibility_round = _event_round(
                trace, "occurrence_treatment_eligible"
            )
            activation_round = _event_round(
                trace, "occurrence_arbitration_activated"
            )
            resolution_activation_round = _event_round(
                trace, "occurrence_resolution_activated"
            )
            resolution_activation_events = tuple(
                row
                for row in trace
                if row.get("type") == "occurrence_resolution_activated"
            )
            arbitration_activation_events = tuple(
                row
                for row in trace
                if row.get("type") == "occurrence_arbitration_activated"
            )
            treatment_cutoff_round = (
                resolution_activation_round
                if arm in {"a2-clean", "a3"}
                else eligibility_round
            )
            exposure = _event(trace, "occurrence_treatment_exposed")
            raw_no_oracle_audit = runtime.get("no_oracle_runtime_gate", {})
            no_oracle_audit = (
                raw_no_oracle_audit
                if isinstance(raw_no_oracle_audit, Mapping)
                else {}
            )
            raw_replay = no_oracle_audit.get("occurrence_replay", {})
            replay = raw_replay if isinstance(raw_replay, Mapping) else {}
            replay_mode = str(replay.get("mode", "live") or "live")
            replay_complete = (
                replay.get("consumption_complete")
                if replay_mode == "replay"
                else True
            )
            rows.append(
                {
                    "arm": arm,
                    "case_id": case_id,
                    "run_dir": str(run_dir),
                    "score": score,
                    "exact_correct": bool(score == 1.0),
                    "raw_exact": bool(score == 1.0),
                    "verified_correct": bool(
                        score == 1.0
                        and str(prediction.get("verification_status", ""))
                        == "verified"
                    ),
                    "ref_300": ref_300,
                    "correct_and_ref_300": bool(score == 1.0 and ref_300),
                    "grounded_correct_ref300": bool(score == 1.0 and ref_300),
                    "grounded_correct_bound_visual": bool(
                        score == 1.0
                        and isinstance(bound_visual_clue_recall, (int, float))
                        and bound_visual_clue_recall > 0
                    ),
                    "judge_model": answer_eval.get("judge_model"),
                    "parse_status": answer_eval.get("parse_status"),
                    "visual_frames": metrics.get("visual_frames_inspected"),
                    "vlm_calls": metrics.get("visual_interpretation_count"),
                    "semantic_rounds_used": metrics.get(
                        "semantic_rounds_used", metrics.get("rounds")
                    ),
                    "forced_finalize_round": metrics.get(
                        "forced_finalize_round"
                    ),
                    "extra_rounds_granted": metrics.get(
                        "extra_rounds_granted"
                    ),
                    "visual_windows": _visual_window_count(observations),
                    **occurrence_metrics,
                    "clue_count": len(clues),
                    "occurrence_handle_usage_rate": _handle_usage_rate(
                        trace, treatment_cutoff_round
                    ),
                    "selected_locator_usage_rate": selected_locator_usage_rate,
                    "selected_locator_count": len(selected_locator_pairs),
                    "selected_locator_accounting_applicable": bool(
                        selected_locator_pairs
                    ),
                    "selected_locator_inspected_count": len(
                        selected_locator_pairs & executed_binding_pairs
                    ),
                    "selected_locators_accounted": locator_accounting[
                        "accounted"
                    ],
                    "selected_locator_silent_drop_count": locator_accounting[
                        "silent_drop_count"
                    ],
                    "selected_locator_accounting_conflict_count": (
                        locator_accounting["conflict_count"]
                    ),
                    "released_unexecuted_rate": released_unexecuted_rate,
                    "released_at_budget_exhaustion_rate": (
                        locator_accounting["release_counts"][
                            "released_at_budget_exhaustion"
                        ]
                        / len(selected_locator_pairs)
                        if selected_locator_pairs
                        else None
                    ),
                    "released_on_set_retirement_rate": (
                        locator_accounting["release_counts"][
                            "released_on_set_retirement"
                        ]
                        / len(selected_locator_pairs)
                        if selected_locator_pairs
                        else None
                    ),
                    "released_by_revision_rate": (
                        locator_accounting["release_counts"][
                            "released_by_revision"
                        ]
                        / len(selected_locator_pairs)
                        if selected_locator_pairs
                        else None
                    ),
                    "contradictory_gate_state_count": sum(
                        row.get("type") == "contradictory_gate_state"
                        for row in trace
                    ),
                    "locator_scope_single_set_passed": (
                        locator_scope_single_set_passed
                    ),
                    "bound_visual_clue_recall": bound_visual_clue_recall,
                    "premature_occurrence_commit": any(
                        bool(row.get("premature_occurrence_commit"))
                        for row in trace
                        if row.get("type") == "reasoner_decision"
                    ),
                    "treatment_eligible_round": eligibility_round,
                    "arbitration_activation_round": activation_round,
                    "resolution_activation_round": resolution_activation_round,
                    "resolution_activation_threshold_valid": all(
                        int(row.get("candidate_count", 0) or 0) >= 1
                        for row in resolution_activation_events
                    ),
                    "arbitration_activation_threshold_valid": all(
                        int(row.get("candidate_count", 0) or 0) >= 2
                        for row in arbitration_activation_events
                    ),
                    "pre_treatment_signature": _pre_treatment_signature(
                        trace, treatment_cutoff_round
                    ),
                    "pre_treatment_prompt_digests": _pre_treatment_prompt_digests(
                        trace, treatment_cutoff_round
                    ),
                    "pre_activation_state_exposure": any(
                        row.get("type") == "reasoner_decision"
                        and row.get("occurrence_resolution_state_exposed")
                        and (
                            resolution_activation_round is None
                            or int(row.get("round", 0) or 0)
                            < resolution_activation_round
                        )
                        for row in trace
                    ),
                    "treatment_exposure": dict(exposure) if exposure else None,
                    "treatment_retrieval_identity_digest": (
                        _first_exposed_retrieval_identity(no_oracle_audit)
                    ),
                    "same_packet_text_budget_parity_passed": (
                        bool(no_oracle_audit.get("text_budget_parity_passed"))
                        if arm in {"a1-flat", "a1"}
                        and isinstance(no_oracle_audit, Mapping)
                        else None
                    ),
                    "no_oracle_gate_passed": bool(
                        no_oracle_audit.get(
                            "no_oracle_runtime_gate_passed", False
                        )
                    ),
                    "occurrence_replay_mode": replay_mode,
                    "occurrence_replay_fixture_digest": replay.get(
                        "fixture_digest"
                    ),
                    "occurrence_replay_complete": replay_complete,
                    "frozen_replay_full_consumption": replay_complete is True,
                    "occurrence_replay_prefix_valid": (
                        replay.get("consumed_prefix_valid")
                        if replay_mode == "replay"
                        else True
                    ),
                    "occurrence_replay_prime_configured": bool(
                        config.get("occurrence_replay_prime", False)
                    ),
                    "occurrence_replay_prime_requested": bool(
                        replay.get("prime_requested", False)
                    ),
                    "occurrence_replay_prime_consumed": bool(
                        replay.get("prime_consumed", False)
                    ),
                    "occurrence_replay_prime_event_count": len(
                        replay_prime_events
                    ),
                    "occurrence_replay_prime_event_completed": (
                        replay_prime_event_completed
                    ),
                    "occurrence_replay_prime_event_pre_reasoner": (
                        replay_prime_event_pre_reasoner
                    ),
                    "occurrence_replay_post_fixture_reuse_count": int(
                        replay.get("post_fixture_reuse_count", 0) or 0
                    ),
                    "occurrence_replay_identity_digests": list(
                        replay.get("consumed_identity_digests", ())
                        if replay_mode == "replay"
                        else no_oracle_audit.get(
                            "retrieval_identity_digests", ()
                        )
                    ),
                    "retired_locator_count": _retired_locator_count(state),
                    "matched_response_control": matched_response,
                    "frozen_config": _frozen_config(config),
                }
            )
    return tuple(rows)


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 20260813,
    trajectory_provenance: str = "unspecified",
) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm, case_id = str(row["arm"]), str(row["case_id"])
        if case_id in by_arm[arm]:
            raise ValueError(f"duplicate row: {arm}:{case_id}")
        by_arm[arm][case_id] = row
    arms = tuple(sorted(by_arm, key=_arm_sort_key))
    case_sets = {arm: set(by_arm[arm]) for arm in arms}
    aligned_cases = set.intersection(*case_sets.values()) if case_sets else set()
    expected = int(expected_cases) if expected_cases is not None else len(aligned_cases)
    frozen_complete_cases = {
        case_id
        for case_id in aligned_cases
        if all(
            _row_frozen_replay_complete(by_arm[arm][case_id]) for arm in arms
        )
    }
    all_analysis = _build_analysis_slice(
        by_arm,
        aligned_cases,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    frozen_analysis = _build_analysis_slice(
        by_arm,
        frozen_complete_cases,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    text_parity = _text_parity(by_arm)
    replay_parity = _frozen_replay_parity(by_arm)
    budget_symmetry = _budget_symmetry(by_arm)
    post_selection_balance = _post_selection_only_divergence(by_arm)
    matched_response_gate = _matched_pre_treatment_response_gate(by_arm)
    structural_checks = {
        "arms_present": bool(arms),
        "case_sets_aligned": bool(case_sets)
        and all(values == aligned_cases for values in case_sets.values()),
        "expected_case_count": len(aligned_cases) == expected,
        "all_no_oracle_gates_passed": all(
            bool(row.get("no_oracle_gate_passed")) for row in rows
        ),
        "frozen_configs_aligned": _configs_aligned(by_arm, aligned_cases),
        "all_judge_records_parsed": all(
            row.get("parse_status") == "parsed"
            for row in rows
            if row.get("score") is not None
        ),
        "a1_flat_text_budget_parity": text_parity.get("passed"),
        "a1_flat_same_packet_text_budget_parity": all(
            row.get("same_packet_text_budget_parity_passed") is True
            for arm in ("a1-flat", "a1")
            for row in by_arm.get(arm, {}).values()
        ),
        "no_pre_activation_occurrence_state": all(
            not row.get("pre_activation_state_exposure")
            for arm in ("a2-clean", "a3")
            for row in by_arm.get(arm, {}).values()
        ),
        "a3_selected_locators_accounted": (
            "a3" not in by_arm
            or all(
                row.get(
                    "selected_locators_accounted",
                    row.get("selected_locator_usage_rate") in {None, 1.0},
                )
                is True
                and int(
                    row.get("selected_locator_silent_drop_count", 0) or 0
                )
                == 0
                and int(
                    row.get(
                        "selected_locator_accounting_conflict_count", 0
                    )
                    or 0
                )
                == 0
                for row in by_arm["a3"].values()
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
            int(row.get("contradictory_gate_state_count", 0) or 0) == 0
            for row in rows
        ),
        "frozen_occurrence_replay_parity": replay_parity.get("passed"),
        "frozen_occurrence_replay_prime": replay_parity.get("prime_passed"),
        "budget_symmetry_passed": budget_symmetry.get("passed"),
        "locator_scope_single_set_passed": all(
            row.get("locator_scope_single_set_passed") is True
            for arm in ("a2-clean", "a3")
            for row in by_arm.get(arm, {}).values()
        ),
        "occurrence_activation_thresholds_valid": all(
            row.get("resolution_activation_threshold_valid") is True
            and row.get("arbitration_activation_threshold_valid") is True
            for arm in ("a2-clean", "a3")
            for row in by_arm.get(arm, {}).values()
        ),
    }
    return {
        "schema_version": "MMLifelongOccurrenceAgentReportV2",
        "primary_endpoint_note": (
            "QA accuracy is a secondary endpoint; mechanism metrics are the "
            "primary endpoint during development."
        ),
        "primary_analysis_set": "frozen_complete",
        "trajectory_provenance": str(trajectory_provenance),
        "definitions": {
            "verified_correct": "judge score == 1 and runtime verification_status == verified",
            "correct_and_ref_300": "judge score == 1 and reference_grounding.ref_300 is true",
            "candidate_recall_trajectory": "gold overlaps any candidate seen anywhere in the trajectory",
            "candidate_recall_active_set": "gold overlaps a candidate in the answer-critical active set",
            "candidate_recall_resolved_set": "gold overlaps a candidate in the set receiving the final accepted resolution",
            "pre_treatment_divergence": "A0 and treatment action/task-query signatures differ before each run first exposes an occurrence candidate set",
            "osa_any": "at least one selected occurrence overlaps a clue paired to the resolved set, conditioned on resolved-set recall",
            "osa_strict": "exactly one occurrence is selected and it overlaps a clue paired to the resolved set",
            "false_commit_rate": "final resolution is selected, conditioned on gold absent from the resolved set",
            "no_match_accuracy": "final resolution is no_match, conditioned on gold absent from the resolved set",
            "occurrence_handle_usage_rate": "post-eligibility visual-window tasks with occurrence_id divided by all post-eligibility visual-window tasks",
            "selected_locator_usage_rate": "accepted selected (locator_attempt_id, occurrence_id) pairs with executed candidate-bound visual observations",
            "selected_locator_accounting": "every selected locator has exactly one terminal outcome: inspected or one explicit release category",
            "released_unexecuted_rate": "selected locators released at finalization, retirement, or resolution revision divided by all selected locators",
            "matched_pre_treatment_responses": "A3 exactly replays A2-clean Reasoner and Investigator responses until scoped resolution is persisted; all later calls remain live",
            "bound_visual_clue_recall": "fraction of gold clue intervals overlapped by an executed occurrence-bound visual window",
            "budget_symmetry_passed": "configured semantic-round budgets are present, internally consistent, and equal across arms; realized semantic rounds remain a treatment cost endpoint",
        },
        "all_cases": all_analysis,
        "frozen_complete": frozen_analysis,
        "arms": frozen_analysis["arms"],
        "comparisons": frozen_analysis["comparisons"],
        "cases": _per_case_metrics(by_arm, aligned_cases),
        "text_budget_parity": text_parity,
        "frozen_occurrence_replay": replay_parity,
        "budget_symmetry": budget_symmetry,
        "post_selection_only_divergence": post_selection_balance,
        "matched_pre_treatment_responses": matched_response_gate,
        "structural_check_applicability": {
            "a3_selected_locators_accounted": any(
                bool(row.get("selected_locator_accounting_applicable"))
                for row in by_arm.get("a3", {}).values()
            )
        },
        "structural_checks": structural_checks,
        "structural_gate_passed": all(
            value is not False for value in structural_checks.values()
        ),
        "judge_models": sorted(
            {
                str(row["judge_model"])
                for row in rows
                if row.get("judge_model")
            }
        ),
        "case_count": len(frozen_complete_cases),
        "all_case_count": len(aligned_cases),
    }


def _build_analysis_slice(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    case_ids: Iterable[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    selected_ids = tuple(sorted(set(case_ids)))
    selected = {
        arm: {case_id: cases[case_id] for case_id in selected_ids if case_id in cases}
        for arm, cases in by_arm.items()
    }
    arms = tuple(sorted(selected, key=_arm_sort_key))
    arm_metrics = {
        arm: _aggregate_arm(tuple(selected[arm].values())) for arm in arms
    }
    comparisons: dict[str, Any] = {}
    comparison_pairs: list[tuple[str, str]] = []
    if "a0" in selected:
        comparison_pairs.extend(
            (arm, "a0") for arm in arms if arm != "a0"
        )
    for left, right in (("a2-clean", "a1"), ("a3", "a2-clean")):
        if left in selected and right in selected:
            comparison_pairs.append((left, right))
    for index, (left, right) in enumerate(dict.fromkeys(comparison_pairs)):
        comparisons[f"{left}-{right}"] = _paired_comparison(
            selected[left],
            selected[right],
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 7,
        )
    return {
        "n": len(selected_ids),
        "case_ids": list(selected_ids),
        "arms": arm_metrics,
        "comparisons": comparisons,
        "old_vs_new_metrics": {
            arm: {
                "candidate_recall_old_trajectory": metrics[
                    "legacy_candidate_recall"
                ],
                "candidate_recall_new_resolved_set": metrics[
                    "candidate_recall_resolved_set"
                ],
                "osa_old_trajectory_any": metrics["legacy_osa"],
                "osa_new_resolved_any": metrics["osa_any"],
                "osa_new_resolved_strict": metrics["osa_strict"],
                "abstention_old_any_historical_no_match": metrics[
                    "legacy_abstention_accuracy"
                ],
                "abstention_new_final_no_match": metrics[
                    "no_match_accuracy"
                ],
            }
            for arm, metrics in arm_metrics.items()
        },
    }


def _row_frozen_replay_complete(row: Mapping[str, Any]) -> bool:
    if "frozen_replay_full_consumption" in row:
        return row.get("frozen_replay_full_consumption") is True
    return row.get("occurrence_replay_complete") is True


def _per_case_metrics(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    case_ids: Iterable[str],
) -> dict[str, Any]:
    fields = (
        "raw_exact",
        "verified_correct",
        "grounded_correct_ref300",
        "grounded_correct_bound_visual",
        "candidate_recall_trajectory",
        "candidate_recall_active_set",
        "candidate_recall_resolved_set",
        "resolved_set_id",
        "final_resolution",
        "osa_any",
        "osa_strict",
        "osa_precision",
        "false_commit",
        "no_match_correct",
        "false_abstention",
        "semantic_rounds_used",
        "extra_rounds_granted",
        "retired_locator_count",
        "frozen_replay_full_consumption",
    )
    return {
        case_id: {
            arm: {field: row.get(field) for field in fields}
            for arm, cases in sorted(by_arm.items(), key=lambda item: _arm_sort_key(item[0]))
            if (row := cases.get(case_id)) is not None
        }
        for case_id in sorted(set(case_ids))
    }


def _aggregate_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if isinstance(row.get("score"), (int, float))]
    osa = [row for row in rows if row.get("osa_eligible")]
    candidate_absent = [
        row for row in rows if row.get("candidate_recall_resolved_set") is False
    ]
    candidate_present = [
        row for row in rows if row.get("candidate_recall_resolved_set") is True
    ]
    selected_rows = [
        row for row in rows if row.get("final_resolution") == "selected"
    ]
    final_resolution_counts = {
        resolution: sum(
            row.get("final_resolution") == resolution for row in rows
        )
        for resolution in ("selected", "no_match", "deferred", "unresolved")
    }
    return _aggregate_arm_result(
        rows=rows,
        scored=scored,
        osa=osa,
        candidate_absent=candidate_absent,
        candidate_present=candidate_present,
        selected_rows=selected_rows,
        final_resolution_counts=final_resolution_counts,
    )


def build_decomposition(
    rows: Sequence[Mapping[str, Any]],
    *,
    trajectory_provenance: str = "unspecified",
) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm, case_id = str(row["arm"]), str(row["case_id"])
        if case_id in by_arm[arm]:
            raise ValueError(f"duplicate row: {arm}:{case_id}")
        by_arm[arm][case_id] = row
    required = {"a0", "a2-clean", "a3"}
    missing = sorted(required - set(by_arm))
    if missing:
        raise ValueError(f"decomposition missing required arms: {', '.join(missing)}")
    case_sets = [set(cases) for cases in by_arm.values()]
    aligned = set.intersection(*case_sets) if case_sets else set()
    frozen_complete = {
        case_id
        for case_id in aligned
        if all(
            _row_frozen_replay_complete(cases[case_id])
            for cases in by_arm.values()
        )
    }
    slices = {
        "frozen_complete": _decompose_slice(by_arm, frozen_complete),
        "all_cases": _decompose_slice(by_arm, aligned),
    }
    return {
        "schema_version": "MMLifelongOccurrenceDecompositionV1",
        "trajectory_provenance": str(trajectory_provenance),
        "primary_analysis_set": "frozen_complete",
        "slices": slices,
    }


def _decompose_slice(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    case_ids: Iterable[str],
) -> dict[str, Any]:
    selected_ids = tuple(sorted(set(case_ids)))
    a0, a2_clean, a3 = by_arm["a0"], by_arm["a2-clean"], by_arm["a3"]
    losses = [
        case_id
        for case_id in selected_ids
        if _numeric_score(a0[case_id]) is not None
        and _numeric_score(a3[case_id]) is not None
        and _numeric_score(a3[case_id]) < _numeric_score(a0[case_id])
    ]
    table_a_rows = [
        {
            "case_id": case_id,
            "a0_score": a0[case_id].get("score"),
            "a0_ref_300": a0[case_id].get("ref_300"),
            "a0_bound_visual_clue_recall": a0[case_id].get(
                "bound_visual_clue_recall"
            ),
            "a3_score": a3[case_id].get("score"),
            "a3_final_resolution": a3[case_id].get("final_resolution"),
            "a3_osa_strict": a3[case_id].get("osa_strict"),
            "a3_bound_visual_clue_recall": a3[case_id].get(
                "bound_visual_clue_recall"
            ),
            "a3_semantic_rounds_used": a3[case_id].get(
                "semantic_rounds_used"
            ),
            "a3_frames": a3[case_id].get("visual_frames"),
        }
        for case_id in losses
    ]
    table_a_decision = _table_a_decision(table_a_rows)

    table_b_rows: list[dict[str, Any]] = []
    for arm, cases in (("a2-clean", a2_clean), ("a3", a3)):
        absent_rows = [
            cases[case_id]
            for case_id in selected_ids
            if cases[case_id].get("candidate_recall_resolved_set") is False
        ]
        for resolution in ("selected", "no_match", "deferred", "unresolved"):
            group = [
                row
                for row in absent_rows
                if row.get("final_resolution") == resolution
            ]
            table_b_rows.append(
                {
                    "arm": arm,
                    "final_resolution": resolution,
                    "n": len(group),
                    "mean_score": _optional_mean(row.get("score") for row in group),
                    "grounded_correct_ref300_rate": _mean_bool(
                        row.get("grounded_correct_ref300") for row in group
                    ),
                    "grounded_correct_bound_visual_rate": _mean_bool(
                        row.get("grounded_correct_bound_visual") for row in group
                    ),
                    "mean_frames": _optional_mean(
                        row.get("visual_frames") for row in group
                    ),
                    "mean_vlm_calls": _optional_mean(
                        row.get("vlm_calls") for row in group
                    ),
                    "mean_semantic_rounds_used": _optional_mean(
                        row.get("semantic_rounds_used") for row in group
                    ),
                }
            )

    table_c_rows: list[dict[str, Any]] = []
    for arm, cases in (("a2-clean", a2_clean), ("a3", a3)):
        present = [
            cases[case_id]
            for case_id in selected_ids
            if cases[case_id].get("candidate_recall_resolved_set") is True
        ]
        current = present
        stages = (
            ("GoldAccessible", lambda row: True),
            ("CorrectSelection", lambda row: row.get("osa_strict") is True),
            (
                "LocatorUsage",
                lambda row: isinstance(
                    row.get("selected_locator_usage_rate"), (int, float)
                )
                and float(row["selected_locator_usage_rate"]) > 0,
            ),
            (
                "GoldVisual",
                lambda row: isinstance(
                    row.get("bound_visual_clue_recall"), (int, float)
                )
                and float(row["bound_visual_clue_recall"]) > 0,
            ),
            ("CorrectAnswer", lambda row: row.get("raw_exact") is True),
        )
        previous_count = len(current)
        for index, (stage, predicate) in enumerate(stages):
            stage_rows = current if index == 0 else [row for row in current if predicate(row)]
            count = len(stage_rows)
            table_c_rows.append(
                {
                    "row_type": "funnel",
                    "arm": arm,
                    "stage": stage,
                    "count": count,
                    "previous_count": previous_count,
                    "conditional_rate": (
                        count / previous_count if previous_count else None
                    ),
                    "paired_n": None,
                    "wins": None,
                    "ties": None,
                    "losses": None,
                }
            )
            current = stage_rows
            previous_count = count

    a3_present_ids = [
        case_id
        for case_id in selected_ids
        if a3[case_id].get("candidate_recall_resolved_set") is True
    ]
    for baseline_arm, baseline in (("a0", a0), ("a2-clean", a2_clean)):
        outcome = _paired_outcomes(a3, baseline, a3_present_ids)
        table_c_rows.append(
            {
                "row_type": "paired_score_outcome",
                "arm": f"a3-{baseline_arm}",
                "stage": "CandidatePresentScore",
                "count": None,
                "previous_count": None,
                "conditional_rate": None,
                **outcome,
            }
        )

    return {
        "n": len(selected_ids),
        "case_ids": list(selected_ids),
        "table_a": {"rows": table_a_rows, **table_a_decision},
        "table_b": {"rows": table_b_rows},
        "table_c": {"rows": table_c_rows},
    }


def _numeric_score(row: Mapping[str, Any]) -> float | None:
    value = row.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def _table_a_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    loss_count = len(rows)
    a_evidence = sum(row.get("a0_bound_visual_clue_recall") == 0 for row in rows)
    b_evidence = sum(
        row.get("a3_osa_strict") is True
        and isinstance(row.get("a3_bound_visual_clue_recall"), (int, float))
        and float(row["a3_bound_visual_clue_recall"]) > 0
        for row in rows
    )
    a_majority = bool(loss_count) and a_evidence > loss_count / 2
    b_majority = bool(loss_count) and b_evidence > loss_count / 2
    if a_majority and not b_majority:
        classification = "A"
        conclusion = (
            "A3 primarily displaced ungrounded baseline lucky answers; keep "
            "grounded-correct as the endpoint and do not introduce evidence "
            "assimilation as a bottleneck."
        )
    elif b_majority and not a_majority:
        classification = "B"
        conclusion = (
            "A3 usually reached and selected gold evidence before losing; "
            "evidence assimilation becomes a candidate research question."
        )
    elif a_majority and b_majority:
        classification = "mixed_A_B"
        conclusion = (
            "Both mechanisms are a majority signal; do not make a single-cause "
            "claim without case-level review."
        )
    else:
        classification = "inconclusive"
        conclusion = (
            "Neither A nor B has majority support; evidence assimilation must "
            "not be claimed as the bottleneck."
        )
    return {
        "loss_count": loss_count,
        "a_ungrounded_baseline_count": a_evidence,
        "b_gold_seen_selection_count": b_evidence,
        "classification": classification,
        "conclusion": conclusion,
    }


def _paired_outcomes(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    case_ids: Iterable[str],
) -> dict[str, int]:
    differences = [
        float(left[case_id]["score"]) - float(right[case_id]["score"])
        for case_id in case_ids
        if case_id in left
        and case_id in right
        and isinstance(left[case_id].get("score"), (int, float))
        and isinstance(right[case_id].get("score"), (int, float))
    ]
    return {
        "paired_n": len(differences),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def _aggregate_arm_result(
    *,
    rows: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    osa: Sequence[Mapping[str, Any]],
    candidate_absent: Sequence[Mapping[str, Any]],
    candidate_present: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    final_resolution_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "n": len(rows),
        "scored_n": len(scored),
        "mean_score": _optional_mean(row.get("score") for row in scored),
        "exact_correct_rate": _mean_bool(row.get("exact_correct") for row in scored),
        "verified_correct_rate": _mean_bool(
            row.get("verified_correct") for row in scored
        ),
        "correct_and_ref_300_rate": _mean_bool(
            row.get("correct_and_ref_300") for row in scored
        ),
        "grounded_correct_ref300_rate": _mean_bool(
            row.get("grounded_correct_ref300") for row in scored
        ),
        "grounded_correct_bound_visual_rate": _mean_bool(
            row.get("grounded_correct_bound_visual") for row in scored
        ),
        "candidate_recall": _mean_bool(row.get("candidate_recall") for row in rows),
        "legacy_candidate_recall": _mean_bool(
            row.get("legacy_candidate_recall") for row in rows
        ),
        "candidate_recall_trajectory": _mean_bool(
            row.get("candidate_recall_trajectory") for row in rows
        ),
        "candidate_recall_active_set": _mean_bool(
            row.get("candidate_recall_active_set") for row in rows
        ),
        "candidate_recall_resolved_set": _mean_bool(
            row.get("candidate_recall_resolved_set") for row in rows
        ),
        "candidate_clue_recall": _optional_mean(
            row.get("candidate_clue_recall") for row in rows
        ),
        "occurrence_selection_accuracy": _mean_bool(
            row.get("osa_correct") for row in osa
        ),
        "osa_any": _mean_bool(row.get("osa_any") for row in osa),
        "osa_strict": _mean_bool(row.get("osa_strict") for row in osa),
        "osa_precision": _optional_mean(row.get("osa_precision") for row in osa),
        "legacy_osa": _mean_bool(
            row.get("legacy_osa_correct")
            for row in rows
            if row.get("legacy_osa_eligible")
        ),
        "osa_eligible_count": len(osa),
        "candidate_present_resolved_count": len(candidate_present),
        "candidate_absent_resolved_count": len(candidate_absent),
        "selected_case_count": len(selected_rows),
        "final_resolution_counts": final_resolution_counts,
        "selected_clue_recall": _optional_mean(
            row.get("selected_clue_recall") for row in osa
        ),
        "multi_selection_rate": _mean_bool(
            int(row.get("selected_occurrence_count", 0) or 0) > 1
            for row in rows
            if int(row.get("selected_occurrence_count", 0) or 0) > 0
        ),
        "abstention_accuracy": _mean_bool(
            row.get("no_match_correct") for row in candidate_absent
        ),
        "no_match_accuracy": _mean_bool(
            row.get("no_match_correct") for row in candidate_absent
        ),
        "legacy_abstention_accuracy": _mean_bool(
            row.get("legacy_abstention_correct")
            for row in rows
            if row.get("legacy_abstention_eligible")
        ),
        "false_commit_rate": _mean_bool(
            row.get("false_commit") for row in candidate_absent
        ),
        "false_commit_count": sum(
            row.get("false_commit") is True for row in candidate_absent
        ),
        "no_match_correct_count": sum(
            row.get("no_match_correct") is True for row in candidate_absent
        ),
        "abstention_eligible_count": len(candidate_absent),
        "false_abstention_rate": _mean_bool(
            row.get("false_abstention") for row in candidate_present
        ),
        "false_abstention_count": sum(
            row.get("false_abstention") is True for row in candidate_present
        ),
        "defer_rate": _mean_bool(
            row.get("deferred_occurrence_set") for row in rows
        ),
        "occurrence_handle_usage_rate": _optional_mean(
            row.get("occurrence_handle_usage_rate") for row in rows
        ),
        "selected_locator_usage_rate": _optional_mean(
            row.get("selected_locator_usage_rate") for row in rows
        ),
        "released_unexecuted_rate": _optional_mean(
            row.get("released_unexecuted_rate") for row in rows
        ),
        "released_at_budget_exhaustion_rate": _optional_mean(
            row.get("released_at_budget_exhaustion_rate") for row in rows
        ),
        "released_on_set_retirement_rate": _optional_mean(
            row.get("released_on_set_retirement_rate") for row in rows
        ),
        "released_by_revision_rate": _optional_mean(
            row.get("released_by_revision_rate") for row in rows
        ),
        "locator_scope_single_set_rate": _mean_bool(
            row.get("locator_scope_single_set_passed") for row in rows
        ),
        "bound_visual_clue_recall": _optional_mean(
            row.get("bound_visual_clue_recall") for row in rows
        ),
        "arbitration_activation_rate": _mean_bool(
            row.get("arbitration_activation_round") is not None
            for row in rows
        ),
        "resolution_activation_rate": _mean_bool(
            row.get("resolution_activation_round") is not None
            for row in rows
        ),
        "premature_commit_rate": _mean_bool(
            row.get("premature_occurrence_commit") for row in rows
        ),
        "mean_visual_frames": _optional_mean(
            row.get("visual_frames") for row in rows
        ),
        "mean_visual_windows": _optional_mean(
            row.get("visual_windows") for row in rows
        ),
        "mean_vlm_calls": _optional_mean(row.get("vlm_calls") for row in rows),
        "mean_semantic_rounds_used": _optional_mean(
            row.get("semantic_rounds_used") for row in rows
        ),
        "mean_forced_finalize_round": _optional_mean(
            row.get("forced_finalize_round") for row in rows
        ),
        "mean_extra_rounds_granted": _optional_mean(
            row.get("extra_rounds_granted") for row in rows
        ),
        "mean_retired_locator_count": _optional_mean(
            row.get("retired_locator_count") for row in rows
        ),
        "frozen_replay_full_consumption_rate": _mean_bool(
            row.get("frozen_replay_full_consumption") for row in rows
        ),
        "frozen_replay_prime_rate": _mean_bool(
            row.get("occurrence_replay_prime_configured") for row in rows
        ),
        "frozen_replay_prime_consumed_rate": _mean_bool(
            row.get("occurrence_replay_prime_consumed")
            for row in rows
            if row.get("occurrence_replay_prime_configured")
        ),
        "frozen_replay_prime_event_completed_rate": _mean_bool(
            row.get("occurrence_replay_prime_event_completed")
            for row in rows
            if row.get("occurrence_replay_prime_configured")
        ),
        "frozen_replay_prime_event_pre_reasoner_rate": _mean_bool(
            row.get("occurrence_replay_prime_event_pre_reasoner")
            for row in rows
            if row.get("occurrence_replay_prime_configured")
        ),
        "frozen_replay_post_fixture_reuse_mean": _optional_mean(
            row.get("occurrence_replay_post_fixture_reuse_count")
            for row in rows
        ),
    }


def _post_selection_only_divergence(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if "a2-clean" not in by_arm or "a3" not in by_arm:
        return {
            "applicable": False,
            "passed": None,
            "paired_case_count": 0,
            "mismatch_case_ids": [],
        }
    clean = by_arm["a2-clean"]
    actionable = by_arm["a3"]
    paired = sorted(set(clean) & set(actionable))

    def signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(row.get("resolved_set_id", "") or ""),
            str(row.get("final_resolution", "") or ""),
            tuple(row.get("selected_occurrence_ids", ()) or ()),
        )

    mismatches = [
        case_id
        for case_id in paired
        if signature(clean[case_id]) != signature(actionable[case_id])
    ]
    return {
        "applicable": True,
        "passed": set(clean) == set(actionable) and not mismatches,
        "paired_case_count": len(paired),
        "mismatch_case_ids": mismatches,
    }


def _matched_pre_treatment_response_gate(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    clean = by_arm.get("a2-clean", {})
    actionable = by_arm.get("a3", {})
    applicable = any(
        row.get("matched_response_control")
        for cases in (clean, actionable)
        for row in cases.values()
    )
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
        recorded = dict(
            clean[case_id].get("matched_response_control", {}) or {}
        )
        replayed = dict(
            actionable[case_id].get("matched_response_control", {}) or {}
        )
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


def _budget_symmetry(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    configured_budgets = {
        arm: sorted(
            {
                budget
                for row in cases.values()
                if (budget := _configured_semantic_round_budget(row))
                is not None
            }
        )
        for arm, cases in by_arm.items()
    }
    missing_arms = sorted(
        arm for arm, values in configured_budgets.items() if not values
    )
    inconsistent_arms = sorted(
        arm for arm, values in configured_budgets.items() if len(values) > 1
    )
    configured_values = sorted(
        {value for values in configured_budgets.values() for value in values}
    )
    configured_gap = (
        max(configured_values) - min(configured_values)
        if configured_values
        else None
    )
    arm_means = {
        arm: _optional_mean(
            row.get("semantic_rounds_used") for row in cases.values()
        )
        for arm, cases in by_arm.items()
    }
    values = [float(value) for value in arm_means.values() if value is not None]
    observed_gap = max(values) - min(values) if values else None
    return {
        "configured_semantic_round_budgets": configured_budgets,
        "configured_max_minus_min": configured_gap,
        "inconsistent_arms": inconsistent_arms,
        "missing_arms": missing_arms,
        "arm_mean_semantic_rounds": arm_means,
        "max_minus_min": observed_gap,
        "observed_realized_rounds_endpoint_only": True,
        "passed": bool(configured_budgets)
        and not missing_arms
        and not inconsistent_arms
        and configured_gap == 0,
    }


def _configured_semantic_round_budget(
    row: Mapping[str, Any],
) -> int | None:
    config = row.get("frozen_config")
    if not isinstance(config, Mapping):
        return None
    for key in ("semantic_round_budget", "max_rounds"):
        value = config.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _paired_score_delta(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    case_ids: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    differences = [
        float(left[case_id]["score"]) - float(right[case_id]["score"])
        for case_id in case_ids
        if isinstance(left[case_id].get("score"), (int, float))
        and isinstance(right[case_id].get("score"), (int, float))
    ]
    if not differences:
        return {
            "paired_n": 0,
            "mean_score_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "discordant_pairs": 0,
            "sign_test_p": None,
            "underpowered": True,
        }
    low, high = _bootstrap_mean_ci(
        differences, samples=bootstrap_samples, seed=seed
    )
    wins = sum(value > 0 for value in differences)
    ties = sum(value == 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    discordant = wins + losses
    return {
        "paired_n": len(differences),
        "mean_score_delta": mean(differences),
        "ci95_low": low,
        "ci95_high": high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "discordant_pairs": discordant,
        "sign_test_p": _exact_sign_test_p(wins, losses),
        "underpowered": discordant < 10,
    }


def _exact_sign_test_p(wins: int, losses: int) -> float | None:
    discordant = int(wins) + int(losses)
    if discordant <= 0:
        return None
    tail = sum(comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_comparison(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    paired_ids = sorted(set(left) & set(right))
    comparable_ids = [
        case_id
        for case_id in paired_ids
        if left[case_id].get("pre_treatment_signature") is not None
        and right[case_id].get("pre_treatment_signature") is not None
    ]
    matched_ids = [
        case_id
        for case_id in comparable_ids
        if left[case_id]["pre_treatment_signature"]
        == right[case_id]["pre_treatment_signature"]
    ]
    prompt_comparable_ids = [
        case_id
        for case_id in paired_ids
        if left[case_id].get("pre_treatment_prompt_digests") is not None
        and right[case_id].get("pre_treatment_prompt_digests") is not None
    ]
    prompt_matched_ids = [
        case_id
        for case_id in prompt_comparable_ids
        if left[case_id]["pre_treatment_prompt_digests"]
        == right[case_id]["pre_treatment_prompt_digests"]
    ]
    return {
        **_paired_score_delta(
            left,
            right,
            paired_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "pre_treatment_comparable_count": len(comparable_ids),
        "pre_treatment_divergence_count": len(comparable_ids) - len(matched_ids),
        "pre_treatment_divergence_rate": (
            (len(comparable_ids) - len(matched_ids)) / len(comparable_ids)
            if comparable_ids
            else None
        ),
        "pre_treatment_prompt_comparable_count": len(prompt_comparable_ids),
        "pre_treatment_prompt_divergence_count": (
            len(prompt_comparable_ids) - len(prompt_matched_ids)
        ),
        "pre_treatment_prompt_divergence_rate": (
            (len(prompt_comparable_ids) - len(prompt_matched_ids))
            / len(prompt_comparable_ids)
            if prompt_comparable_ids
            else None
        ),
        "matched_pre_treatment_subset": _paired_score_delta(
            left,
            right,
            matched_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
    }


def _pre_treatment_signature(
    trace: Sequence[Mapping[str, Any]], cutoff_round: int | None
) -> list[dict[str, Any]] | None:
    if cutoff_round is None:
        return None
    return [
        {
            "action": str(row.get("action", "")),
            "tasks": [
                {
                    "inspection_mode": str(task.get("inspection_mode", "")),
                    "caption_queries": list(task.get("caption_queries", ()) or ()),
                    "search_terms": list(task.get("search_terms", ()) or ()),
                    "time_range": task.get("time_range"),
                    "segment_id": str(task.get("segment_id", "")),
                }
                for task in tuple(row.get("tasks", ()) or ())
                if isinstance(task, Mapping)
            ],
        }
        for row in trace
        if row.get("type") == "reasoner_decision"
        and int(row.get("round", 0) or 0) < cutoff_round
    ]


def _pre_treatment_prompt_digests(
    trace: Sequence[Mapping[str, Any]], cutoff_round: int | None
) -> list[str] | None:
    if cutoff_round is None:
        return None
    return [
        str(row.get("prompt_digest", "") or "")
        for row in trace
        if row.get("type") == "reasoner_decision"
        and int(row.get("round", 0) or 0) < cutoff_round
    ]


def _handle_usage_rate(
    trace: Sequence[Mapping[str, Any]], eligibility_round: int | None
) -> float | None:
    if eligibility_round is None:
        return None
    tasks = [
        task
        for row in trace
        if row.get("type") == "reasoner_decision"
        and int(row.get("round", 0) or 0) >= eligibility_round
        for task in tuple(row.get("tasks", ()) or ())
        if isinstance(task, Mapping)
        and str(task.get("inspection_mode", "")) == "window"
    ]
    return (
        sum(bool(str(task.get("occurrence_id", "") or "")) for task in tasks)
        / len(tasks)
        if tasks
        else None
    )


def _occurrence_candidates(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float]]:
    candidates: dict[str, tuple[float, float]] = {}
    for row in observations:
        config = row.get("sampling_config", {})
        occurrence_set = (
            config.get("occurrence_set") if isinstance(config, Mapping) else None
        )
        if not isinstance(occurrence_set, Mapping):
            continue
        for candidate in tuple(occurrence_set.get("candidates", ()) or ()):
            if not isinstance(candidate, Mapping):
                continue
            occurrence_id = str(candidate.get("occurrence_id", "") or "")
            interval = candidate.get("time_range", ())
            if occurrence_id and isinstance(interval, Sequence) and len(interval) == 2:
                candidates[occurrence_id] = (
                    float(interval[0]),
                    float(interval[1]),
                )
    return candidates


def _occurrence_candidate_sets(
    observations: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> tuple[dict[str, dict[str, tuple[float, float]]], tuple[str, ...]]:
    sets: dict[str, dict[str, tuple[float, float]]] = {}
    order: list[str] = []
    for index, row in enumerate(observations):
        config = row.get("sampling_config", {})
        occurrence_set = (
            config.get("occurrence_set") if isinstance(config, Mapping) else None
        )
        if not isinstance(occurrence_set, Mapping):
            continue
        set_id = str(
            row.get(
                "attempt_id",
                occurrence_set.get(
                    "attempt_id", occurrence_set.get("set_id", "")
                ),
            )
            or ""
        ).strip()
        if not set_id:
            set_id = f"observation_{index:04d}"
        if set_id not in sets:
            sets[set_id] = {}
            order.append(set_id)
        _merge_candidate_ranges(sets[set_id], occurrence_set.get("candidates", ()))

    for raw_set in tuple(state.get("sets", ()) or ()):
        if not isinstance(raw_set, Mapping):
            continue
        set_id = str(
            raw_set.get("set_id", raw_set.get("locator_attempt_id", "")) or ""
        ).strip()
        if not set_id:
            continue
        if set_id not in sets:
            sets[set_id] = {}
            order.append(set_id)
        _merge_candidate_ranges(sets[set_id], raw_set.get("candidates", ()))
    return sets, tuple(order)


def _merge_candidate_ranges(
    destination: dict[str, tuple[float, float]], raw_candidates: Any
) -> None:
    for candidate in tuple(raw_candidates or ()):
        if not isinstance(candidate, Mapping):
            continue
        occurrence_id = str(candidate.get("occurrence_id", "") or "").strip()
        raw_range = candidate.get("time_range", ())
        if (
            occurrence_id
            and isinstance(raw_range, Sequence)
            and not isinstance(raw_range, (str, bytes))
            and len(raw_range) == 2
        ):
            destination[occurrence_id] = (
                float(raw_range[0]),
                float(raw_range[1]),
            )


def _accepted_resolution_ops(
    trace: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    accepted: list[dict[str, str]] = []
    for row in trace:
        if (
            row.get("type") != "reasoner_decision"
            or row.get("occurrence_ops_accepted") is False
        ):
            continue
        for operation in tuple(row.get("occurrence_ops", ()) or ()):
            if not isinstance(operation, Mapping):
                continue
            op = str(operation.get("op", operation.get("type", "")) or "").casefold()
            if op not in {"select", "no_match", "defer"}:
                continue
            accepted.append(
                {
                    "op": op,
                    "set_id": str(
                        operation.get(
                            "set_id", operation.get("locator_attempt_id", "")
                        )
                        or ""
                    ),
                    "occurrence_id": str(operation.get("occurrence_id", "") or ""),
                }
            )
    return tuple(accepted)


def _state_sets(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(raw_set.get("set_id", raw_set.get("locator_attempt_id", "")) or ""):
        raw_set
        for raw_set in tuple(state.get("sets", ()) or ())
        if isinstance(raw_set, Mapping)
        and str(
            raw_set.get("set_id", raw_set.get("locator_attempt_id", "")) or ""
        )
    }


def _occurrence_resolution_metrics(
    *,
    arm: str,
    state: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    clues: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    candidate_sets, set_order = _occurrence_candidate_sets(observations, state)
    trajectory_candidates = {
        occurrence_id: interval
        for candidates in candidate_sets.values()
        for occurrence_id, interval in candidates.items()
    }
    if not trajectory_candidates:
        trajectory_candidates = _occurrence_candidates(observations)

    raw_active_set_id = str(state.get("active_set_id", "") or "")
    active_set_id = raw_active_set_id or (set_order[-1] if set_order else "")
    resolution_ops = _accepted_resolution_ops(trace)
    last_resolution_op = resolution_ops[-1] if resolution_ops else None
    resolved_set_id = (
        str(last_resolution_op.get("set_id", "") or "")
        if last_resolution_op
        else ""
    )
    if not resolved_set_id:
        resolved_set_id = active_set_id

    active_candidates = candidate_sets.get(active_set_id, {})
    resolved_candidates = candidate_sets.get(resolved_set_id, {})
    if not active_candidates and not raw_active_set_id:
        active_candidates = trajectory_candidates
    if not resolved_candidates and not resolved_set_id:
        resolved_candidates = active_candidates or trajectory_candidates

    state_sets = _state_sets(state)
    resolved_state = state_sets.get(resolved_set_id, {})
    raw_selected = tuple(resolved_state.get("selected_occurrence_ids", ()) or ())
    selection_state_present = "selected_occurrence_ids" in resolved_state
    selected_ids = tuple(
        dict.fromkeys(str(value) for value in raw_selected if str(value))
    )
    if not selected_ids and resolved_set_id == active_set_id:
        selection_state_present = selection_state_present or (
            "selected_occurrence_ids" in state
        )
        selected_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in tuple(state.get("selected_occurrence_ids", ()) or ())
                if str(value)
            )
        )
    if not selected_ids and not selection_state_present:
        selected_ids = _selected_occurrence_ids(trace)
    if (
        not selected_ids
        and not selection_state_present
        and last_resolution_op
        and last_resolution_op["op"] == "select"
    ):
        selected_ids = tuple(
            dict.fromkeys(
                operation["occurrence_id"]
                for operation in resolution_ops
                if operation["op"] == "select"
                and (
                    not resolved_set_id
                    or not operation["set_id"]
                    or operation["set_id"] == resolved_set_id
                )
                and operation["occurrence_id"]
            )
        )

    resolution = str(resolved_state.get("resolution", "") or "").casefold()
    if resolution not in {"selected", "no_match", "deferred", "unresolved"}:
        resolution = ""
    if not resolution and resolved_set_id == active_set_id:
        resolution = str(state.get("active_resolution", "") or "").casefold()
    if resolution not in {"selected", "no_match", "deferred", "unresolved"}:
        if selected_ids:
            resolution = "selected"
        elif last_resolution_op and last_resolution_op["op"] == "no_match":
            resolution = "no_match"
        elif last_resolution_op and last_resolution_op["op"] == "defer":
            resolution = "deferred"
        else:
            resolution = "unresolved"

    selected_ranges = tuple(
        resolved_candidates[occurrence_id]
        for occurrence_id in selected_ids
        if occurrence_id in resolved_candidates
    )
    legacy_selected_ids = _selected_occurrence_ids(trace)
    legacy_selected_ranges = tuple(
        trajectory_candidates[occurrence_id]
        for occurrence_id in legacy_selected_ids
        if occurrence_id in trajectory_candidates
    )
    legacy_selected_correct = bool(
        legacy_selected_ranges
        and any(
            _overlap(selected, clue)
            for selected in legacy_selected_ranges
            for clue in clues
        )
    )
    paired_clues = tuple(
        clue
        for clue in clues
        if any(_overlap(candidate, clue) for candidate in resolved_candidates.values())
    )
    correct_selected_count = sum(
        any(_overlap(selected, clue) for clue in paired_clues)
        for selected in selected_ranges
    )
    trajectory_recall = _set_candidate_recall(trajectory_candidates, clues)
    active_recall = _set_candidate_recall(active_candidates, clues)
    resolved_recall = _set_candidate_recall(resolved_candidates, clues)
    scoped = arm in {"a2", "a2-clean", "a3"}
    osa_eligible = scoped and resolved_recall is True
    osa_any = (
        bool(correct_selected_count > 0) if osa_eligible else None
    )
    osa_strict = (
        bool(len(selected_ids) == 1 and correct_selected_count == 1)
        if osa_eligible
        else None
    )
    osa_precision = (
        correct_selected_count / len(selected_ids)
        if osa_eligible and selected_ids
        else None
    )
    absent = scoped and resolved_recall is False
    present = scoped and resolved_recall is True
    no_match_correct = resolution == "no_match" if absent else None
    false_commit = resolution == "selected" if absent else None
    false_abstention = resolution == "no_match" if present else None

    return {
        "candidate_count": len(trajectory_candidates),
        "candidate_recall_trajectory": trajectory_recall,
        "candidate_recall_active_set": active_recall,
        "candidate_recall_resolved_set": resolved_recall,
        "candidate_recall": trajectory_recall,
        "legacy_candidate_recall": trajectory_recall,
        "candidate_clue_recall": _interval_recall(
            tuple(trajectory_candidates.values()), clues
        ),
        "active_set_id": active_set_id or None,
        "resolved_set_id": resolved_set_id or None,
        "final_resolution": resolution,
        "selected_occurrence_ids": list(selected_ids),
        "selected_occurrence_count": len(selected_ids),
        "selected_occurrence_correct": osa_any,
        "selected_clue_recall": _interval_recall(selected_ranges, paired_clues),
        "osa_eligible": osa_eligible,
        "osa_any": osa_any,
        "osa_strict": osa_strict,
        "osa_precision": osa_precision,
        "osa_correct": osa_any,
        "legacy_osa_eligible": scoped and trajectory_recall is True,
        "legacy_osa_correct": (
            legacy_selected_correct
            if scoped and trajectory_recall is True
            else None
        ),
        "candidate_absent": resolved_recall is False,
        "abstained_no_match": resolution == "no_match",
        "deferred_occurrence_set": resolution == "deferred",
        "abstention_eligible": absent,
        "abstention_correct": no_match_correct,
        "legacy_abstention_eligible": (
            arm in {"a2-clean", "a3"} and trajectory_recall is False
        ),
        "legacy_abstention_correct": (
            bool(
                state.get("active_resolution") == "no_match"
                or _accepted_occurrence_op(trace, "no_match")
            )
            if arm in {"a2-clean", "a3"} and trajectory_recall is False
            else None
        ),
        "no_match_correct": no_match_correct,
        "false_commit": false_commit,
        "false_abstention": false_abstention,
    }


def _set_candidate_recall(
    candidates: Mapping[str, tuple[float, float]],
    clues: Sequence[tuple[float, float]],
) -> bool | None:
    if not clues:
        return None
    return any(
        _overlap(candidate, clue)
        for candidate in candidates.values()
        for clue in clues
    )


def _retired_locator_count(state: Mapping[str, Any]) -> int:
    retired_locators = tuple(state.get("retired_locators", ()) or ())
    if retired_locators:
        return sum(isinstance(locator, Mapping) for locator in retired_locators)
    return sum(
        len(tuple(raw_set.get("selected_occurrence_ids", ()) or ()))
        for raw_set in tuple(state.get("sets", ()) or ())
        if isinstance(raw_set, Mapping)
        and str(raw_set.get("lifecycle", "") or "") == "retired"
    )


def _selected_occurrence_ids(
    trace: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    selected: tuple[str, ...] = ()
    for row in trace:
        if row.get("type") != "reasoner_decision":
            continue
        raw_selected = row.get("selected_occurrence_ids")
        if isinstance(raw_selected, Sequence) and not isinstance(
            raw_selected, (str, bytes)
        ):
            selected = tuple(
                dict.fromkeys(str(value) for value in raw_selected if str(value))
            )
        elif "selected_occurrence_id" in row:
            value = str(row.get("selected_occurrence_id", "") or "")
            selected = (value,) if value else ()
    return selected


def _accepted_occurrence_op(
    trace: Sequence[Mapping[str, Any]], op_name: str
) -> bool:
    expected = str(op_name).casefold()
    return any(
        row.get("type") == "reasoner_decision"
        and row.get("occurrence_ops_accepted") is not False
        and any(
            str(operation.get("op", operation.get("type", "")) or "").casefold()
            == expected
            for operation in tuple(row.get("occurrence_ops", ()) or ())
            if isinstance(operation, Mapping)
        )
        for row in trace
    )


def _selected_locator_pairs(
    trace: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    return {
        (
            str(
                operation.get(
                    "set_id", operation.get("locator_attempt_id", "")
                )
                or ""
            ),
            str(operation.get("occurrence_id", "") or ""),
        )
        for row in trace
        if row.get("type") == "reasoner_decision"
        and row.get("occurrence_ops_accepted") is not False
        for operation in tuple(row.get("occurrence_ops", ()) or ())
        if isinstance(operation, Mapping)
        and str(
            operation.get("op", operation.get("type", "")) or ""
        ).casefold()
        == "select"
        and str(
            operation.get("set_id", operation.get("locator_attempt_id", ""))
            or ""
        )
        and str(operation.get("occurrence_id", "") or "")
    }


def _locator_accounting(
    selected_pairs: set[tuple[str, str]],
    executed_pairs: set[tuple[str, str]],
    trace: Sequence[Mapping[str, Any]],
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


def _locator_scope_single_set_passed(state: Mapping[str, Any]) -> bool:
    raw_sets = tuple(
        raw_set
        for raw_set in tuple(state.get("sets", ()) or ())
        if isinstance(raw_set, Mapping)
    )
    if not raw_sets:
        return not state.get("active_set_id")
    active_set_id = str(state.get("active_set_id", "") or "")
    active_set_ids = {
        str(raw_set.get("set_id", "") or "")
        for raw_set in raw_sets
        if str(raw_set.get("lifecycle", "") or "") == "active"
    }
    retired_set_ids = {
        str(raw_set.get("set_id", "") or "")
        for raw_set in raw_sets
        if str(raw_set.get("lifecycle", "") or "") == "retired"
    }
    serialized_retired = {
        str(value)
        for value in tuple(state.get("retired_set_ids", ()) or ())
        if str(value)
    }
    active_locators = tuple(state.get("active_locators", ()) or ())
    retired_locators = tuple(state.get("retired_locators", ()) or ())
    expected_active_pairs = {
        (active_set_id, str(occurrence_id))
        for raw_set in raw_sets
        if str(raw_set.get("set_id", "") or "") == active_set_id
        for occurrence_id in tuple(raw_set.get("selected_occurrence_ids", ()) or ())
        if str(occurrence_id)
    }
    expected_retired_pairs = {
        (str(raw_set.get("set_id", "") or ""), str(occurrence_id))
        for raw_set in raw_sets
        if str(raw_set.get("set_id", "") or "") in retired_set_ids
        for occurrence_id in tuple(raw_set.get("selected_occurrence_ids", ()) or ())
        if str(occurrence_id)
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


def _bound_visual_evidence(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[set[tuple[str, str]], tuple[tuple[float, float], ...]]:
    pairs: set[tuple[str, str]] = set()
    ranges: list[tuple[float, float]] = []
    for row in observations:
        config = row.get("sampling_config")
        binding = config.get("candidate_binding") if isinstance(config, Mapping) else None
        if not isinstance(binding, Mapping):
            continue
        locator_attempt_id = str(binding.get("locator_attempt_id", "") or "")
        occurrence_id = str(binding.get("occurrence_id", "") or "")
        if locator_attempt_id and occurrence_id:
            pairs.add((locator_attempt_id, occurrence_id))
        raw_range = binding.get("candidate_range", ())
        if (
            isinstance(raw_range, Sequence)
            and not isinstance(raw_range, (str, bytes))
            and len(raw_range) == 2
        ):
            ranges.append((float(raw_range[0]), float(raw_range[1])))
    return pairs, tuple(ranges)


def _first_exposed_retrieval_identity(
    audit: Mapping[str, Any],
) -> str | None:
    card_counts = tuple(audit.get("candidate_card_counts", ()) or ())
    identities = tuple(audit.get("retrieval_identity_digests", ()) or ())
    for index, count in enumerate(card_counts):
        if int(count or 0) > 0 and index < len(identities):
            value = str(identities[index] or "")
            return value or None
    return None


def _text_parity(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    if "a1" not in by_arm or "a1-flat" not in by_arm:
        return {"applicable": False, "passed": None, "paired_n": 0}
    case_ids = sorted(set(by_arm["a1"]) & set(by_arm["a1-flat"]))
    comparable = 0
    matched = 0
    for case_id in case_ids:
        left_signature = by_arm["a1"][case_id].get(
            "pre_treatment_signature"
        )
        right_signature = by_arm["a1-flat"][case_id].get(
            "pre_treatment_signature"
        )
        left_retrieval = by_arm["a1"][case_id].get(
            "treatment_retrieval_identity_digest"
        )
        right_retrieval = by_arm["a1-flat"][case_id].get(
            "treatment_retrieval_identity_digest"
        )
        if (
            left_signature is None
            or right_signature is None
            or left_signature != right_signature
            or not left_retrieval
            or not right_retrieval
            or left_retrieval != right_retrieval
        ):
            continue
        left = by_arm["a1"][case_id].get("treatment_exposure") or {}
        right = by_arm["a1-flat"][case_id].get("treatment_exposure") or {}
        if left.get("visible_text_digest") and right.get(
            "visible_text_digest"
        ):
            comparable += 1
            matched += left["visible_text_digest"] == right[
                "visible_text_digest"
            ]
    return {
        "applicable": True,
        "passed": matched == comparable if comparable else None,
        "status": (
            "passed"
            if comparable and matched == comparable
            else "failed"
            if comparable
            else "not_comparable"
        ),
        "paired_n": len(case_ids),
        "comparable_n": comparable,
        "matched_n": matched,
        "match_rate": matched / comparable if comparable else None,
    }


def _frozen_replay_parity(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    replay_arms = tuple(
        arm
        for arm, rows in by_arm.items()
        if any(
            row.get("occurrence_replay_mode") in {"record", "replay"}
            for row in rows.values()
        )
    )
    if not replay_arms:
        return {"applicable": False, "passed": None, "paired_n": 0}
    case_sets = [set(by_arm[arm]) for arm in replay_arms]
    case_ids = sorted(set.intersection(*case_sets)) if case_sets else []
    matched = 0
    complete = 0
    for case_id in case_ids:
        rows = [by_arm[arm][case_id] for arm in replay_arms]
        digests = {
            str(row.get("occurrence_replay_fixture_digest", "") or "")
            for row in rows
        }
        sequences = [
            tuple(row.get("occurrence_replay_identity_digests", ()) or ())
            for row in rows
        ]
        reference = max(sequences, key=len, default=())
        prefix_valid = bool(reference) and all(
            sequence
            and reference[: len(sequence)] == sequence
            and row.get("occurrence_replay_prefix_valid") is not False
            for row, sequence in zip(rows, sequences)
        )
        row_complete = all(
            row.get("occurrence_replay_complete") is True for row in rows
        )
        complete += row_complete
        matched += bool(
            prefix_valid
            and len(digests) == 1
            and "" not in digests
        )
    passed = bool(case_ids) and matched == len(case_ids)
    replay_rows = [
        row
        for arm in replay_arms
        for row in by_arm[arm].values()
        if row.get("occurrence_replay_mode") == "replay"
    ]
    prime_applicable = any(
        row.get("occurrence_replay_prime_configured") for row in replay_rows
    )
    prime_passed = (
        all(
            row.get("occurrence_replay_prime_configured") is True
            and row.get("occurrence_replay_prime_requested") is True
            and row.get("occurrence_replay_prime_consumed") is True
            and row.get("occurrence_replay_prime_event_count") == 1
            and row.get("occurrence_replay_prime_event_completed") is True
            and row.get("occurrence_replay_prime_event_pre_reasoner") is True
            for row in replay_rows
        )
        if prime_applicable and replay_rows
        else None
    )
    return {
        "applicable": True,
        "passed": passed,
        "paired_n": len(case_ids),
        "complete_n": complete,
        "full_consumption_rate": complete / len(case_ids) if case_ids else None,
        "matched_n": matched,
        "match_rate": matched / len(case_ids) if case_ids else None,
        "prime_applicable": prime_applicable,
        "prime_passed": prime_passed,
        "post_fixture_reuse_count": sum(
            int(row.get("occurrence_replay_post_fixture_reuse_count", 0) or 0)
            for row in replay_rows
        ),
        "arms": list(replay_arms),
    }


def _configs_aligned(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]], case_ids: set[str]
) -> bool:
    for case_id in case_ids:
        configs = [
            row[case_id].get("frozen_config") for row in by_arm.values()
        ]
        if any(config != configs[0] for config in configs[1:]):
            return False
    return True


def _frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in FROZEN_CONFIG_KEYS}


def _event(
    trace: Sequence[Mapping[str, Any]], event_type: str
) -> Mapping[str, Any] | None:
    return next((row for row in trace if row.get("type") == event_type), None)


def _event_round(
    trace: Sequence[Mapping[str, Any]], event_type: str
) -> int | None:
    row = _event(trace, event_type)
    return int(row.get("round", 0) or 0) if row else None


def _visual_window_count(observations: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        isinstance(row.get("sampling_config"), Mapping)
        and row["sampling_config"].get("mode") == "window"
        for row in observations
    )


def _overlap(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _interval_recall(
    predictions: Sequence[tuple[float, float]],
    targets: Sequence[tuple[float, float]],
) -> float | None:
    if not targets:
        return None
    return sum(
        any(_overlap(prediction, target) for prediction in predictions)
        for target in targets
    ) / len(targets)


def _optional_mean(values: Iterable[Any]) -> float | None:
    normalized = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(normalized) if normalized else None


def _mean_bool(values: Iterable[Any]) -> float | None:
    normalized = [float(bool(value)) for value in values if value is not None]
    return mean(normalized) if normalized else None


def _bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    boot = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(max(1, int(samples)))
    )
    return boot[int(0.025 * (len(boot) - 1))], boot[
        int(0.975 * (len(boot) - 1))
    ]


def _arm_sort_key(arm: str) -> tuple[int, str]:
    order = {
        "a0": 0,
        "a1-flat": 1,
        "a1": 2,
        "a2": 3,
        "a2-clean": 4,
        "a3": 5,
    }
    return order.get(arm, 99), arm


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong Occurrence Agent Report",
        "",
        f"Structural gate: **{'PASS' if report['structural_gate_passed'] else 'FAIL'}**",
        "",
        "**QA accuracy is a secondary endpoint; mechanism metrics are the primary endpoint during development.**",
        "",
        f"Primary analysis: frozen_complete (n={report['frozen_complete']['n']}); all_cases sensitivity n={report['all_cases']['n']}.",
        f"Trajectory provenance: `{report['trajectory_provenance']}`.",
        "",
        "| Arm | N | Mean | Raw exact | Verified | Grounded ref300 | Grounded visual | Recall trajectory | Recall resolved | OSA any | OSA strict | No-match | False commit | Locator use | Released | Bound visual recall | Rounds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, metrics in report["arms"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    str(metrics["n"]),
                    _fmt(metrics["mean_score"]),
                    _fmt(metrics["exact_correct_rate"]),
                    _fmt(metrics["verified_correct_rate"]),
                    _fmt(metrics["grounded_correct_ref300_rate"]),
                    _fmt(metrics["grounded_correct_bound_visual_rate"]),
                    _fmt(metrics["candidate_recall_trajectory"]),
                    _fmt(metrics["candidate_recall_resolved_set"]),
                    _fmt(metrics["osa_any"]),
                    _fmt(metrics["osa_strict"]),
                    _fmt(metrics["no_match_accuracy"]),
                    _fmt(metrics["false_commit_rate"]),
                    _fmt(metrics["selected_locator_usage_rate"]),
                    _fmt(metrics["released_unexecuted_rate"]),
                    _fmt(metrics["bound_visual_clue_recall"]),
                    _fmt(metrics["mean_semantic_rounds_used"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Paired Comparisons", ""])
    for name, comparison in report["comparisons"].items():
        lines.append(
            f"- {name}: delta {_fmt(comparison['mean_score_delta'])}, "
            f"95% CI [{_fmt(comparison['ci95_low'])}, {_fmt(comparison['ci95_high'])}], "
            f"W/T/L {comparison['wins']}/{comparison['ties']}/{comparison['losses']}; "
            f"exact sign p={_fmt(comparison['sign_test_p'])}, "
            f"underpowered={str(comparison['underpowered']).lower()}; "
            f"pre-treatment action divergence {_fmt(comparison['pre_treatment_divergence_rate'])}, "
            f"prompt divergence {_fmt(comparison['pre_treatment_prompt_divergence_rate'])}."
        )
    lines.extend(
        [
            "",
            "## Old vs New Metric Definitions",
            "",
            "| Arm | Recall old | Recall resolved | OSA old | OSA any | OSA strict | Abstain old | No-match final |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, metrics in report["frozen_complete"]["old_vs_new_metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _fmt(metrics["candidate_recall_old_trajectory"]),
                    _fmt(metrics["candidate_recall_new_resolved_set"]),
                    _fmt(metrics["osa_old_trajectory_any"]),
                    _fmt(metrics["osa_new_resolved_any"]),
                    _fmt(metrics["osa_new_resolved_strict"]),
                    _fmt(metrics["abstention_old_any_historical_no_match"]),
                    _fmt(metrics["abstention_new_final_no_match"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Gates", ""])
    for name, value in report["structural_checks"].items():
        lines.append(f"- {name}: {value}")
    return "\n".join(lines) + "\n"


def write_decomposition_outputs(
    report: Mapping[str, Any], output_dir: Path
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_specs = {
        "table_a_losses.csv": (
            "table_a",
            (
                "analysis_set",
                "case_id",
                "a0_score",
                "a0_ref_300",
                "a0_bound_visual_clue_recall",
                "a3_score",
                "a3_final_resolution",
                "a3_osa_strict",
                "a3_bound_visual_clue_recall",
                "a3_semantic_rounds_used",
                "a3_frames",
            ),
        ),
        "table_b_candidate_absent.csv": (
            "table_b",
            (
                "analysis_set",
                "arm",
                "final_resolution",
                "n",
                "mean_score",
                "grounded_correct_ref300_rate",
                "grounded_correct_bound_visual_rate",
                "mean_frames",
                "mean_vlm_calls",
                "mean_semantic_rounds_used",
            ),
        ),
        "table_c_candidate_present_funnel.csv": (
            "table_c",
            (
                "analysis_set",
                "row_type",
                "arm",
                "stage",
                "count",
                "previous_count",
                "conditional_rate",
                "paired_n",
                "wins",
                "ties",
                "losses",
            ),
        ),
    }
    paths: dict[str, str] = {}
    for filename, (table_key, fieldnames) in table_specs.items():
        path = destination / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for analysis_set, decomposition in report["slices"].items():
                for row in decomposition[table_key]["rows"]:
                    writer.writerow({"analysis_set": analysis_set, **row})
        paths[filename] = str(path)
    markdown_path = destination / "decomposition_summary.md"
    markdown_path.write_text(
        render_decomposition_markdown(report), encoding="utf-8"
    )
    paths[markdown_path.name] = str(markdown_path)
    return paths


def render_decomposition_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong Occurrence Decomposition",
        "",
        f"Trajectory provenance: `{report['trajectory_provenance']}`.",
        "",
        "The frozen-complete slice is primary; all-cases is sensitivity analysis.",
    ]
    for analysis_set in ("frozen_complete", "all_cases"):
        decomposition = report["slices"][analysis_set]
        table_a = decomposition["table_a"]
        lines.extend(
            [
                "",
                f"## {analysis_set} (n={decomposition['n']})",
                "",
                "### Table A: A3 losses vs A0",
                "",
                (
                    f"Classification: **{table_a['classification']}**. "
                    f"Losses={table_a['loss_count']}, "
                    f"A-signal={table_a['a_ungrounded_baseline_count']}, "
                    f"B-signal={table_a['b_gold_seen_selection_count']}."
                ),
                "",
                table_a["conclusion"],
                "",
                "### Table B: Candidate-absent final resolutions",
                "",
                "| Arm | Resolution | N | Mean score | Grounded ref300 | Grounded visual | Frames | VLM calls | Rounds |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in decomposition["table_b"]["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["arm"]),
                        str(row["final_resolution"]),
                        str(row["n"]),
                        _fmt(row["mean_score"]),
                        _fmt(row["grounded_correct_ref300_rate"]),
                        _fmt(row["grounded_correct_bound_visual_rate"]),
                        _fmt(row["mean_frames"]),
                        _fmt(row["mean_vlm_calls"]),
                        _fmt(row["mean_semantic_rounds_used"]),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Table C: Candidate-present funnel",
                "",
                "| Arm | Stage | Count | Previous | Conditional rate | W/T/L |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in decomposition["table_c"]["rows"]:
            outcome = (
                f"{row['wins']}/{row['ties']}/{row['losses']}"
                if row.get("row_type") == "paired_score_outcome"
                else ""
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["arm"]),
                        str(row["stage"]),
                        "" if row.get("count") is None else str(row["count"]),
                        (
                            ""
                            if row.get("previous_count") is None
                            else str(row["previous_count"])
                        ),
                        _fmt(row.get("conditional_rate")),
                        outcome,
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


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


def _decompose_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_mmlifelong_occurrence_agent.py decompose"
    )
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--trajectory-provenance", default="unspecified")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(tuple(argv))
    rows = collect_rows(
        tuple(Path(value) for value in args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
    )
    report = build_decomposition(
        rows, trajectory_provenance=args.trajectory_provenance
    )
    paths = write_decomposition_outputs(report, Path(args.output_dir))
    print(
        json.dumps(
            {
                "primary_n": report["slices"]["frozen_complete"]["n"],
                "all_n": report["slices"]["all_cases"]["n"],
                "primary_table_a_classification": report["slices"][
                    "frozen_complete"
                ]["table_a"]["classification"],
                "outputs": paths,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "decompose":
        _decompose_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--trajectory-provenance", default="unspecified")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    rows = collect_rows(
        tuple(Path(value) for value in args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
    )
    report = build_report(
        rows,
        expected_cases=args.expected_cases,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        trajectory_provenance=args.trajectory_provenance,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "arms": list(report["arms"]),
                "structural_gate_passed": report["structural_gate_passed"],
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
