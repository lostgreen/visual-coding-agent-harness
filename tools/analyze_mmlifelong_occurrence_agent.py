#!/usr/bin/env python3
"""Analyze no-oracle occurrence-method arms without feeding gold into runtime."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
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
            candidates = _occurrence_candidates(observations)
            selected_ids = _selected_occurrence_ids(trace)
            selected_ranges = tuple(
                candidates[occurrence_id]
                for occurrence_id in selected_ids
                if occurrence_id in candidates
            )
            candidate_recall = bool(clues) and any(
                _overlap(candidate_range, clue)
                for candidate_range in candidates.values()
                for clue in clues
            )
            selected_correct = bool(
                selected_ranges
                and any(
                    _overlap(selected_range, clue)
                    for selected_range in selected_ranges
                    for clue in clues
                )
            )
            candidate_clue_recall = _interval_recall(
                tuple(candidates.values()), clues
            )
            selected_clue_recall = _interval_recall(selected_ranges, clues)
            state = _read_json(run_dir / "occurrence_resolution_state.json")
            final_selected_pairs = _final_selected_pairs(state)
            executed_binding_pairs, bound_visual_ranges = _bound_visual_evidence(
                observations
            )
            selected_locator_usage_rate = (
                len(final_selected_pairs & executed_binding_pairs)
                / len(final_selected_pairs)
                if final_selected_pairs
                else None
            )
            abstained = bool(
                state.get("active_resolution") == "no_match"
                or _accepted_occurrence_op(trace, "no_match")
            )
            deferred = _accepted_occurrence_op(trace, "defer")
            answer_eval = evaluation.get("answer", {})
            grounding = evaluation.get("reference_grounding", {})
            metrics = runtime.get("runtime_metrics", {})
            score = answer_eval.get("score")
            ref_300 = grounding.get("ref_300")
            eligibility_round = _event_round(
                trace, "occurrence_treatment_eligible"
            )
            activation_round = _event_round(
                trace, "occurrence_arbitration_activated"
            )
            treatment_cutoff_round = (
                activation_round
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
            rows.append(
                {
                    "arm": arm,
                    "case_id": case_id,
                    "run_dir": str(run_dir),
                    "score": score,
                    "exact_correct": bool(score == 1.0),
                    "verified_correct": bool(
                        score == 1.0
                        and str(prediction.get("verification_status", ""))
                        == "verified"
                    ),
                    "ref_300": ref_300,
                    "correct_and_ref_300": bool(score == 1.0 and ref_300),
                    "judge_model": answer_eval.get("judge_model"),
                    "parse_status": answer_eval.get("parse_status"),
                    "visual_frames": metrics.get("visual_frames_inspected"),
                    "vlm_calls": metrics.get("visual_interpretation_count"),
                    "visual_windows": _visual_window_count(observations),
                    "candidate_count": len(candidates),
                    "candidate_recall": candidate_recall,
                    "candidate_clue_recall": candidate_clue_recall,
                    "selected_occurrence_ids": list(selected_ids),
                    "selected_occurrence_count": len(selected_ids),
                    "selected_occurrence_correct": selected_correct,
                    "selected_clue_recall": selected_clue_recall,
                    "osa_eligible": arm in {"a2", "a2-clean", "a3"}
                    and candidate_recall,
                    "osa_correct": (
                        selected_correct
                        if arm in {"a2", "a2-clean", "a3"}
                        and candidate_recall
                        else None
                    ),
                    "candidate_absent": bool(clues) and not candidate_recall,
                    "abstained_no_match": abstained,
                    "deferred_occurrence_set": deferred,
                    "abstention_eligible": arm in {"a2-clean", "a3"}
                    and bool(clues)
                    and not candidate_recall,
                    "abstention_correct": (
                        abstained
                        if arm in {"a2-clean", "a3"}
                        and bool(clues)
                        and not candidate_recall
                        else None
                    ),
                    "false_abstention": bool(candidate_recall and abstained),
                    "clue_count": len(clues),
                    "occurrence_handle_usage_rate": _handle_usage_rate(
                        trace, treatment_cutoff_round
                    ),
                    "selected_locator_usage_rate": selected_locator_usage_rate,
                    "bound_visual_clue_recall": _interval_recall(
                        bound_visual_ranges, clues
                    ),
                    "premature_occurrence_commit": any(
                        bool(row.get("premature_occurrence_commit"))
                        for row in trace
                        if row.get("type") == "reasoner_decision"
                    ),
                    "treatment_eligible_round": eligibility_round,
                    "arbitration_activation_round": activation_round,
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
                            activation_round is None
                            or int(row.get("round", 0) or 0)
                            < activation_round
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
                    "occurrence_replay_complete": (
                        replay.get("consumption_complete")
                        if replay_mode == "replay"
                        else True
                    ),
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
    arm_metrics = {
        arm: _aggregate_arm(tuple(by_arm[arm].values())) for arm in arms
    }
    comparisons = {}
    comparison_pairs: list[tuple[str, str]] = []
    if "a0" in by_arm:
        for arm in arms:
            if arm == "a0":
                continue
            comparison_pairs.append((arm, "a0"))
    for left, right in (("a2-clean", "a1"), ("a3", "a2-clean")):
        if left in by_arm and right in by_arm:
            comparison_pairs.append((left, right))
    for index, (left, right) in enumerate(dict.fromkeys(comparison_pairs)):
        comparisons[f"{left}-{right}"] = _paired_comparison(
            by_arm[left],
            by_arm[right],
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 7,
        )
    text_parity = _text_parity(by_arm)
    replay_parity = _frozen_replay_parity(by_arm)
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
        "a3_selected_locators_executed": all(
            row.get("selected_locator_usage_rate") in {None, 1.0}
            for row in by_arm.get("a3", {}).values()
        ),
        "frozen_occurrence_replay_parity": replay_parity.get("passed"),
        "frozen_occurrence_replay_prime": replay_parity.get("prime_passed"),
    }
    return {
        "schema_version": "MMLifelongOccurrenceAgentReportV2",
        "definitions": {
            "verified_correct": "judge score == 1 and runtime verification_status == verified",
            "correct_and_ref_300": "judge score == 1 and reference_grounding.ref_300 is true",
            "pre_treatment_divergence": "A0 and treatment action/task-query signatures differ before each run first exposes an occurrence candidate set",
            "occurrence_selection_accuracy": "selected occurrence overlaps at least one gold clue, conditioned on any natural candidate overlapping a clue",
            "occurrence_handle_usage_rate": "post-eligibility visual-window tasks with occurrence_id divided by all post-eligibility visual-window tasks",
            "abstention_accuracy": "no_match rate conditioned on no retrieved candidate overlapping any gold clue",
            "selected_locator_usage_rate": "final selected (locator_attempt_id, occurrence_id) pairs with executed candidate-bound visual observations",
            "bound_visual_clue_recall": "fraction of gold clue intervals overlapped by an executed occurrence-bound visual window",
        },
        "arms": arm_metrics,
        "comparisons": comparisons,
        "text_budget_parity": text_parity,
        "frozen_occurrence_replay": replay_parity,
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
        "case_count": len(aligned_cases),
    }


def _aggregate_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if isinstance(row.get("score"), (int, float))]
    osa = [row for row in rows if row.get("osa_eligible")]
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
        "candidate_recall": _mean_bool(row.get("candidate_recall") for row in rows),
        "candidate_clue_recall": _optional_mean(
            row.get("candidate_clue_recall") for row in rows
        ),
        "occurrence_selection_accuracy": _mean_bool(
            row.get("osa_correct") for row in osa
        ),
        "osa_eligible_count": len(osa),
        "selected_clue_recall": _optional_mean(
            row.get("selected_clue_recall") for row in osa
        ),
        "multi_selection_rate": _mean_bool(
            int(row.get("selected_occurrence_count", 0) or 0) > 1
            for row in rows
            if int(row.get("selected_occurrence_count", 0) or 0) > 0
        ),
        "abstention_accuracy": _mean_bool(
            row.get("abstention_correct")
            for row in rows
            if row.get("abstention_eligible")
        ),
        "abstention_eligible_count": sum(
            bool(row.get("abstention_eligible")) for row in rows
        ),
        "false_abstention_rate": _mean_bool(
            row.get("false_abstention")
            for row in rows
            if row.get("osa_eligible")
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
        "bound_visual_clue_recall": _optional_mean(
            row.get("bound_visual_clue_recall") for row in rows
        ),
        "arbitration_activation_rate": _mean_bool(
            row.get("arbitration_activation_round") is not None
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
        }
    low, high = _bootstrap_mean_ci(
        differences, samples=bootstrap_samples, seed=seed
    )
    return {
        "paired_n": len(differences),
        "mean_score_delta": mean(differences),
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


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


def _final_selected_pairs(state: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(raw_set.get("set_id", "") or ""), str(occurrence_id))
        for raw_set in tuple(state.get("sets", ()) or ())
        if isinstance(raw_set, Mapping) and raw_set.get("set_id")
        for occurrence_id in tuple(raw_set.get("selected_occurrence_ids", ()) or ())
        if str(occurrence_id)
    }


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
        "| Arm | N | Mean | Exact | Verified | Candidate recall | OSA | Abstain | Locator use | Bound visual recall | Premature | Frames | VLM calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    _fmt(metrics["candidate_recall"]),
                    _fmt(metrics["occurrence_selection_accuracy"]),
                    _fmt(metrics["abstention_accuracy"]),
                    _fmt(metrics["selected_locator_usage_rate"]),
                    _fmt(metrics["bound_visual_clue_recall"]),
                    _fmt(metrics["premature_commit_rate"]),
                    _fmt(metrics["mean_visual_frames"]),
                    _fmt(metrics["mean_vlm_calls"]),
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
            f"pre-treatment action divergence {_fmt(comparison['pre_treatment_divergence_rate'])}, "
            f"prompt divergence {_fmt(comparison['pre_treatment_prompt_divergence_rate'])}."
        )
    lines.extend(["", "## Gates", ""])
    for name, value in report["structural_checks"].items():
        lines.append(f"- {name}: {value}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
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
