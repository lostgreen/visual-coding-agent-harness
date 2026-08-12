#!/usr/bin/env python3
"""Build paired and stratified reports for the MM-Lifelong oracle ladder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence


ARMS = ("o0", "c0", "o1", "o1.5", "o1.75", "o2")
COMPARISONS = (
    ("c0-o0", "c0", "o0"),
    ("o1-c0", "o1", "c0"),
    ("o1.5-o1", "o1.5", "o1"),
    ("o1.75-o1.5", "o1.75", "o1.5"),
    ("o2-o1.75", "o2", "o1.75"),
    ("o2-o1", "o2", "o1"),
    ("o2-c0", "o2", "c0"),
    ("o1-o0_descriptive", "o1", "o0"),
)
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
    "anchor_execution_policy",
    "embedding",
    "input_digest",
    "models",
    "phase5r_mode",
    "web_enabled",
    "supporting_interval_source",
)


def collect_rows(
    run_roots: Sequence[Path],
    *,
    evaluation_record_root: Path,
    case_ids: frozenset[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root in run_roots:
        for prediction_path in sorted(Path(root).glob("cases/*/prediction.json")):
            run_dir = prediction_path.parent
            if case_ids is not None and run_dir.name not in case_ids:
                continue
            prediction = _read_json(prediction_path)
            config = _read_json(run_dir / "run_config.json")
            runtime = _read_json(run_dir / "runtime_summary.json")
            runtime_case = _read_json(run_dir / "case.json")
            case_id = str(prediction["case_id"])
            arm = str(config.get("oracle_arm", "o0")).casefold()
            key = arm, case_id
            if key in seen:
                raise ValueError(f"duplicate oracle result: {arm}:{case_id}")
            seen.add(key)
            evaluation_path = run_dir / "evaluation" / "mmlifelong_eval.json"
            evaluation = _read_json(evaluation_path) if evaluation_path.is_file() else {}
            record = _read_json(
                Path(evaluation_record_root) / case_id / "evaluation_case.json"
            )
            intervals = tuple(
                (float(value[0]), float(value[1]))
                for value in tuple(record.get("clue_intervals", ()) or ())
                if len(value) == 2
            )
            metrics = runtime.get("runtime_metrics", {})
            audit = runtime.get("oracle_intervention_audit")
            answer = evaluation.get("answer", {})
            grounding = evaluation.get("reference_grounding", {})
            trajectory = _trajectory_metrics(run_dir, intervals)
            rows.append(
                {
                    "arm": arm,
                    "case_id": case_id,
                    "question_type": str(runtime_case.get("question_type") or "Unknown"),
                    "clue_count": len(intervals),
                    "clue_duration_sec": sum(max(0.0, end - start) for start, end in intervals),
                    "score": answer.get("score"),
                    "parse_status": answer.get("parse_status"),
                    "judge_model": answer.get("judge_model"),
                    "official_judge_model_match": answer.get(
                        "official_judge_model_match"
                    ),
                    "ref_300": grounding.get("ref_300"),
                    "answer_rate": metrics.get("answer_rate"),
                    "reference_valid_rate": metrics.get("reference_valid_rate"),
                    "visual_frames": metrics.get("visual_frames_inspected"),
                    **trajectory,
                    "audit": dict(audit) if isinstance(audit, Mapping) else None,
                    "frozen_config": {
                        key: config.get(key) for key in FROZEN_CONFIG_KEYS
                    },
                }
            )
    if not rows:
        raise FileNotFoundError("no oracle-ladder predictions found")
    return tuple(rows)


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    bootstrap_samples: int = 10_000,
    seed: int = 20260811,
) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm, case_id = str(row["arm"]), str(row["case_id"])
        by_arm[arm][case_id] = row
    case_sets = {arm: set(by_arm.get(arm, {})) for arm in ARMS}
    common_cases = set.intersection(*(case_sets[arm] for arm in ARMS))
    aligned = all(case_sets[arm] == common_cases for arm in ARMS)

    arm_rows = []
    for arm in ARMS:
        group = tuple(by_arm.get(arm, {}).values())
        audits = tuple(
            row["audit"] for row in group if isinstance(row.get("audit"), Mapping)
        )
        arm_rows.append(
            {
                "arm": arm,
                "case_count": len(group),
                "mean_score": _optional_mean(row.get("score") for row in group),
                "exact_correct_rate": _optional_mean(
                    float(row["score"] == 1.0)
                    for row in group
                    if isinstance(row.get("score"), (int, float))
                ),
                "answer_rate": _optional_mean(row.get("answer_rate") for row in group),
                "reference_valid_rate": _optional_mean(
                    row.get("reference_valid_rate") for row in group
                ),
                "ref_300": _optional_mean(row.get("ref_300") for row in group),
                "mean_visual_frames": _optional_mean(
                    row.get("visual_frames") for row in group
                ),
                "natural_clue_recall": _optional_mean(
                    audit.get("natural_clue_recall") for audit in audits
                ),
                "final_clue_recall": _optional_mean(
                    audit.get("final_clue_recall") for audit in audits
                ),
                "mean_injected_candidates": _optional_mean(
                    audit.get("injected_candidate_count") for audit in audits
                ),
                "mean_selected_candidates": _optional_mean(
                    audit.get("selected_candidate_count") for audit in audits
                ),
                "selected_candidate_clue_recall": _optional_mean(
                    audit.get("selected_candidate_clue_recall") for audit in audits
                ),
                "mean_anchor_count": _optional_mean(
                    audit.get("anchor_count") for audit in audits
                ),
                "mean_visual_window_count": _optional_mean(
                    row.get("visual_window_count") for row in group
                ),
                "mean_visual_window_sec": _optional_mean(
                    row.get("mean_visual_window_sec") for row in group
                ),
                "clue_visual_recall": _optional_mean(
                    row.get("clue_visual_recall") for row in group
                ),
                "clue_center_visual_recall": _optional_mean(
                    row.get("clue_center_visual_recall") for row in group
                ),
                "selected_candidate_inspection_recall": _optional_mean(
                    row.get("selected_candidate_inspection_recall") for row in group
                ),
                "selected_candidate_request_recall": _optional_mean(
                    row.get("selected_candidate_request_recall") for row in group
                ),
                "anchor_inspection_recall": _optional_mean(
                    row.get("anchor_inspection_recall") for row in group
                ),
                "anchor_request_recall": _optional_mean(
                    row.get("anchor_request_recall") for row in group
                ),
                "anchor_frame_recall": _optional_mean(
                    row.get("anchor_frame_recall") for row in group
                ),
                "forced_anchor_recall": _optional_mean(
                    row.get("forced_anchor_recall") for row in group
                ),
                "execution_fidelity": _optional_mean(
                    row.get("execution_fidelity") for row in group
                ),
                "exact_frame_execution_fidelity": _optional_mean(
                    row.get("exact_frame_execution_fidelity") for row in group
                ),
                "mean_inspected_candidate_window_sec": _optional_mean(
                    row.get("mean_inspected_candidate_window_sec") for row in group
                ),
                "mean_inspected_anchor_window_sec": _optional_mean(
                    row.get("mean_inspected_anchor_window_sec") for row in group
                ),
            }
        )

    paired = []
    for name, left, right in COMPARISONS:
        differences = [
            float(by_arm[left][case_id]["score"])
            - float(by_arm[right][case_id]["score"])
            for case_id in sorted(common_cases)
            if isinstance(by_arm[left][case_id].get("score"), (int, float))
            and isinstance(by_arm[right][case_id].get("score"), (int, float))
        ]
        low, high = _bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=seed + len(paired),
        )
        paired.append(
            {
                "comparison": name,
                "case_count": len(differences),
                "mean_score_delta": mean(differences) if differences else None,
                "ci95_low": low,
                "ci95_high": high,
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }
        )

    matrix = []
    for case_id in sorted(common_cases):
        exemplar = by_arm["o0"][case_id]
        matrix.append(
            {
                "case_id": case_id,
                "question_type": exemplar["question_type"],
                "clue_count": exemplar["clue_count"],
                "clue_duration_sec": round(float(exemplar["clue_duration_sec"]), 3),
                "scores": {arm: by_arm[arm][case_id].get("score") for arm in ARMS},
            }
        )

    runtime_checks = {
        "all_arms_present": all(bool(by_arm.get(arm)) for arm in ARMS),
        "case_sets_aligned": aligned,
        "expected_case_count": aligned and len(common_cases) == int(expected_cases),
        "frozen_configs_aligned": _frozen_configs_aligned(by_arm, common_cases),
        "natural_caption_retrieval_aligned": _natural_caption_retrieval_aligned(
            by_arm,
            common_cases,
        ),
        "candidate_oracle_arms_full_clue_recall": all(
            float(row.get("audit", {}).get("final_clue_recall", -1.0)) == 1.0
            for row in rows
            if row.get("arm") in {"o1", "o1.5", "o1.75", "o2"}
        ),
        "o1_family_pool_size_preserved": all(
            int(row.get("audit", {}).get("natural_candidate_count", -1))
            == int(row.get("audit", {}).get("final_candidate_count", -2))
            for row in rows
            if row.get("arm") in {"o1", "o1.5", "o1.75"}
        ),
        "o1_family_candidate_pools_identical": _o1_family_candidate_pools_identical(
            by_arm,
            common_cases,
        ),
        "intermediate_guidance_valid": _intermediate_guidance_valid(
            by_arm,
            common_cases,
        ),
        "o2_exact_locators_complete": all(
            int(row.get("audit", {}).get("exact_locator_count", -1))
            == int(row.get("clue_count", -2))
            for row in rows
            if row.get("arm") == "o2"
        ),
    }
    evaluation_checks = {
        "all_evaluated": all(
            isinstance(row.get("score"), (int, float)) for row in rows
        ),
        "all_judge_responses_parsed": all(
            row.get("parse_status") == "parsed" for row in rows
        ),
    }
    checks = {**runtime_checks, **evaluation_checks}
    judge_models = sorted(
        {str(row["judge_model"]) for row in rows if row.get("judge_model")}
    )
    official = all(row.get("official_judge_model_match") is True for row in rows)
    return {
        "schema_version": "MMLifelongOracleLadderReportV2",
        "expected_cases": int(expected_cases),
        "common_case_count": len(common_cases),
        "runtime_gate_passed": all(runtime_checks.values()),
        "runtime_gate_checks": runtime_checks,
        "gate_passed": all(checks.values()),
        "gate_checks": checks,
        "judge_models": judge_models,
        "diagnostic_only": not official,
        "arms": arm_rows,
        "paired_comparisons": paired,
        "oracle_gap_recovery": _oracle_gap_recovery(
            by_arm,
            common_cases,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "strata": _stratified_rows(by_arm, common_cases),
        "case_matrix": matrix,
    }


def _frozen_configs_aligned(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    if not cases:
        return False
    return all(
        all(
            by_arm[arm][case_id].get("frozen_config")
            == by_arm["o0"][case_id].get("frozen_config")
            for arm in ARMS[1:]
        )
        for case_id in cases
    )


def _trajectory_metrics(
    run_dir: Path,
    clues: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    observations = _read_jsonl(Path(run_dir) / "observation_log.jsonl")
    visual_request_ranges: list[tuple[float, float]] = []
    visual_coverage_ranges: list[tuple[float, float]] = []
    attached_frame_times: list[float] = []
    forced_anchor_timestamps: list[float] = []
    visual_attachment_status: list[tuple[tuple[float, float], bool]] = []
    guidance: dict[str, Any] = {}
    for row in observations:
        config = row.get("sampling_config")
        config = dict(config) if isinstance(config, Mapping) else {}
        raw_guidance = config.get("oracle_guidance")
        if isinstance(raw_guidance, Mapping):
            guidance = dict(raw_guidance)
        modality = str(row.get("modality", config.get("modality", "")) or "").casefold()
        if modality not in {"visual", "ocr"}:
            continue
        persisted_frame_times = row.get("frame_times")
        if persisted_frame_times is None:
            # Retain compatibility with early analysis fixtures and traces.
            persisted_frame_times = row.get("attached_frame_times", ())
        attached_frame_times.extend(
            float(value)
            for value in tuple(persisted_frame_times or ())
            if isinstance(value, (int, float))
        )
        forced_in_attempt = tuple(
            float(value)
            for value in tuple(config.get("forced_anchor_timestamps_sec", ()) or ())
            if isinstance(value, (int, float))
        )
        forced_anchor_timestamps.extend(forced_in_attempt)
        requested_range = _normalized_range(row.get("requested_range"))
        if requested_range is not None:
            visual_request_ranges.append(requested_range)
            visual_attachment_status.append(
                (
                    requested_range,
                    str(row.get("execution_status", "completed")) == "completed"
                    and int(row.get("images_dropped", 0) or 0) == 0
                    and int(row.get("images_attached", 0) or 0)
                    == int(row.get("images_requested", 0) or 0),
                )
            )
        for raw_range in tuple(row.get("inspected_ranges", ()) or ()):
            interval = _normalized_range(raw_range)
            if interval is not None:
                visual_coverage_ranges.append(interval)

    selected_ranges = tuple(
        interval
        for candidate in tuple(guidance.get("selected_candidates", ()) or ())
        if isinstance(candidate, Mapping)
        if (interval := _normalized_range(candidate.get("inspection_range")))
        is not None
    )
    anchors = tuple(
        float(value)
        for value in tuple(guidance.get("anchor_timestamps_sec", ()) or ())
        if isinstance(value, (int, float))
    )
    clue_centers = tuple((start_sec + end_sec) / 2.0 for start_sec, end_sec in clues)
    requested_anchors, inspected_anchors = _execution_counts(
        anchors,
        visual_request_ranges,
        visual_coverage_ranges,
    )
    _, exact_frame_anchors = _exact_execution_counts(
        anchors,
        visual_request_ranges,
        attached_frame_times,
    )
    anchor_attachment_failure_count = sum(
        not complete
        and any(start_sec <= anchor <= end_sec for anchor in anchors)
        for (start_sec, end_sec), complete in visual_attachment_status
    )
    return {
        "visual_window_count": len(visual_request_ranges),
        "mean_visual_window_sec": _optional_mean(
            end_sec - start_sec for start_sec, end_sec in visual_request_ranges
        ),
        "clue_visual_recall": _range_recall(clues, visual_coverage_ranges),
        "clue_center_visual_recall": _point_recall(
            clue_centers,
            visual_coverage_ranges,
        ),
        "selected_candidate_inspection_recall": _range_recall(
            selected_ranges,
            visual_coverage_ranges,
        ),
        "selected_candidate_request_recall": _range_recall(
            selected_ranges,
            visual_request_ranges,
        ),
        "anchor_inspection_recall": _point_recall(
            anchors,
            visual_coverage_ranges,
        ),
        "anchor_request_recall": _point_recall(
            anchors,
            visual_request_ranges,
        ),
        "anchor_frame_recall": _timestamp_recall(
            anchors,
            attached_frame_times,
        ),
        "forced_anchor_recall": _timestamp_recall(
            anchors,
            forced_anchor_timestamps,
        ),
        "execution_fidelity": _execution_fidelity(
            anchors,
            visual_request_ranges,
            visual_coverage_ranges,
        ),
        "exact_frame_execution_fidelity": _exact_frame_execution_fidelity(
            anchors,
            visual_request_ranges,
            attached_frame_times,
        ),
        "anchor_target_count": len(anchors),
        "anchor_requested_count": requested_anchors,
        "anchor_inspected_count": inspected_anchors,
        "anchor_exact_frame_count": exact_frame_anchors,
        "forced_anchor_count": sum(
            any(abs(target - observed) <= 0.001 for observed in forced_anchor_timestamps)
            for target in anchors
        ),
        "anchor_attachment_failure_count": anchor_attachment_failure_count,
        "anchor_timestamps_sec": list(anchors),
        "mean_inspected_candidate_window_sec": _mean_minimum_window_width(
            selected_ranges,
            visual_request_ranges,
            point_targets=False,
        ),
        "mean_inspected_anchor_window_sec": _mean_minimum_window_width(
            anchors,
            visual_request_ranges,
            point_targets=True,
        ),
    }


def _range_recall(
    targets: Sequence[tuple[float, float]],
    observations: Sequence[tuple[float, float]],
) -> float | None:
    if not targets:
        return None
    return sum(
        any(_ranges_overlap(target, observed) for observed in observations)
        for target in targets
    ) / len(targets)


def _point_recall(
    targets: Sequence[float],
    observations: Sequence[tuple[float, float]],
) -> float | None:
    if not targets:
        return None
    return sum(
        any(start_sec <= target <= end_sec for start_sec, end_sec in observations)
        for target in targets
    ) / len(targets)


def _timestamp_recall(
    targets: Sequence[float],
    observations: Sequence[float],
) -> float | None:
    if not targets:
        return None
    return sum(
        any(abs(float(target) - float(observed)) <= 0.001 for observed in observations)
        for target in targets
    ) / len(targets)


def _execution_fidelity(
    anchors: Sequence[float],
    requests: Sequence[tuple[float, float]],
    observations: Sequence[tuple[float, float]],
) -> float | None:
    requested_count, executed_count = _execution_counts(
        anchors,
        requests,
        observations,
    )
    return executed_count / requested_count if requested_count else None


def _execution_counts(
    anchors: Sequence[float],
    requests: Sequence[tuple[float, float]],
    observations: Sequence[tuple[float, float]],
) -> tuple[int, int]:
    requested = tuple(
        target
        for target in anchors
        if any(start_sec <= target <= end_sec for start_sec, end_sec in requests)
    )
    executed = sum(
        any(start_sec <= target <= end_sec for start_sec, end_sec in observations)
        for target in requested
    )
    return len(requested), executed


def _exact_frame_execution_fidelity(
    anchors: Sequence[float],
    requests: Sequence[tuple[float, float]],
    attached_frame_times: Sequence[float],
) -> float | None:
    requested_count, executed_count = _exact_execution_counts(
        anchors,
        requests,
        attached_frame_times,
    )
    return executed_count / requested_count if requested_count else None


def _exact_execution_counts(
    anchors: Sequence[float],
    requests: Sequence[tuple[float, float]],
    attached_frame_times: Sequence[float],
) -> tuple[int, int]:
    requested = tuple(
        target
        for target in anchors
        if any(start_sec <= target <= end_sec for start_sec, end_sec in requests)
    )
    executed = sum(
        any(abs(target - observed) <= 0.001 for observed in attached_frame_times)
        for target in requested
    )
    return len(requested), executed


def _mean_minimum_window_width(
    targets: Sequence[Any],
    observations: Sequence[tuple[float, float]],
    *,
    point_targets: bool,
) -> float | None:
    widths: list[float] = []
    for target in targets:
        matching = [
            end_sec - start_sec
            for start_sec, end_sec in observations
            if (
                start_sec <= float(target) <= end_sec
                if point_targets
                else _ranges_overlap(target, (start_sec, end_sec))
            )
        ]
        if matching:
            widths.append(min(matching))
    return mean(widths) if widths else None


def _normalized_range(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 2:
        return None
    start_sec, end_sec = float(value[0]), float(value[1])
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec


def _ranges_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


def _natural_caption_retrieval_aligned(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    if not cases:
        return False
    for case_id in cases:
        signatures = []
        for arm in ("c0", "o1", "o1.5", "o1.75", "o2"):
            audit = by_arm[arm][case_id].get("audit")
            if not isinstance(audit, Mapping) or audit.get("applied") is not True:
                return False
            signatures.append(
                (
                    audit.get("caption_config_digest"),
                    audit.get("intervention_digest"),
                    audit.get("natural_candidate_count"),
                    audit.get("natural_clue_recall"),
                )
            )
        if len(set(signatures)) != 1:
            return False
    return True


def _o1_family_candidate_pools_identical(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    if not cases:
        return False
    for case_id in cases:
        signatures = []
        for arm in ("o1", "o1.5", "o1.75"):
            audit = by_arm[arm][case_id].get("audit")
            if not isinstance(audit, Mapping):
                return False
            signatures.append(
                json.dumps(
                    {
                        "passage_ids": audit.get("candidate_passage_ids"),
                        "intervals": audit.get("candidate_intervals"),
                        "shuffle_seed_digest": audit.get("shuffle_seed_digest"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        if len(set(signatures)) != 1:
            return False
    return True


def _intermediate_guidance_valid(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> bool:
    if not cases:
        return False
    for case_id in cases:
        row_15 = by_arm["o1.5"][case_id]
        row_175 = by_arm["o1.75"][case_id]
        audit_15 = row_15.get("audit")
        audit_175 = row_175.get("audit")
        if not isinstance(audit_15, Mapping) or not isinstance(audit_175, Mapping):
            return False
        if audit_15.get("guidance_type") != "selected_coarse_candidates":
            return False
        if (
            audit_175.get("guidance_type")
            != "selected_coarse_candidates_with_point_anchors"
        ):
            return False
        if any(
            audit.get("exact_boundaries_visible") is not False
            for audit in (audit_15, audit_175)
        ):
            return False
        if any(
            float(audit.get("selected_candidate_clue_recall", -1.0)) != 1.0
            for audit in (audit_15, audit_175)
        ):
            return False
        selected_signature_15 = (
            audit_15.get("selected_candidate_ranks"),
            audit_15.get("selected_candidate_passage_ids"),
            audit_15.get("selected_candidate_intervals"),
        )
        selected_signature_175 = (
            audit_175.get("selected_candidate_ranks"),
            audit_175.get("selected_candidate_passage_ids"),
            audit_175.get("selected_candidate_intervals"),
        )
        if selected_signature_15 != selected_signature_175:
            return False
        if int(audit_15.get("selected_candidate_count", 0) or 0) < 1:
            return False
        if int(audit_15.get("anchor_count", -1)) != 0:
            return False
        if int(audit_175.get("anchor_count", -1)) != int(row_175["clue_count"]):
            return False
        if len(tuple(audit_175.get("point_anchor_candidate_ranks", ()) or ())) != int(
            row_175["clue_count"]
        ):
            return False
        if len(
            tuple(audit_175.get("point_anchor_candidate_passage_ids", ()) or ())
        ) != int(row_175["clue_count"]):
            return False
    return True


def _oracle_gap_recovery(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(("o1.5", "o1.75")):
        eligible = [
            case_id
            for case_id in sorted(cases)
            if all(
                isinstance(by_arm[item][case_id].get("score"), (int, float))
                for item in ("o1", arm, "o2")
            )
        ]
        mean_numerator = [
            float(by_arm[arm][case_id]["score"])
            - float(by_arm["o1"][case_id]["score"])
            for case_id in eligible
        ]
        mean_denominator = [
            float(by_arm["o2"][case_id]["score"])
            - float(by_arm["o1"][case_id]["score"])
            for case_id in eligible
        ]
        exact_numerator = [
            float(by_arm[arm][case_id]["score"] == 1.0)
            - float(by_arm["o1"][case_id]["score"] == 1.0)
            for case_id in eligible
        ]
        exact_denominator = [
            float(by_arm["o2"][case_id]["score"] == 1.0)
            - float(by_arm["o1"][case_id]["score"] == 1.0)
            for case_id in eligible
        ]
        mean_point, mean_low, mean_high = _bootstrap_ratio_ci(
            mean_numerator,
            mean_denominator,
            samples=bootstrap_samples,
            seed=seed + 100 + arm_index,
        )
        exact_point, exact_low, exact_high = _bootstrap_ratio_ci(
            exact_numerator,
            exact_denominator,
            samples=bootstrap_samples,
            seed=seed + 200 + arm_index,
        )
        rows.append(
            {
                "arm": arm,
                "case_count": len(eligible),
                "mean_score_ogr": mean_point,
                "mean_score_ci95_low": mean_low,
                "mean_score_ci95_high": mean_high,
                "exact_correct_ogr": exact_point,
                "exact_correct_ci95_low": exact_low,
                "exact_correct_ci95_high": exact_high,
            }
        )
    return rows


def _stratified_rows(
    by_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cases: set[str],
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for arm in ARMS:
        for case_id in cases:
            row = by_arm[arm][case_id]
            score = row.get("score")
            if not isinstance(score, (int, float)):
                continue
            buckets = {
                "question_type": str(row["question_type"]),
                "clue_count": "single" if int(row["clue_count"]) == 1 else "multi",
                "clue_duration": _duration_bucket(float(row["clue_duration_sec"])),
            }
            for dimension, bucket in buckets.items():
                strata[(dimension, bucket, arm)].append(float(score))
    return [
        {
            "dimension": dimension,
            "bucket": bucket,
            "arm": arm,
            "case_count": len(values),
            "mean_score": mean(values),
        }
        for (dimension, bucket, arm), values in sorted(strata.items())
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "| Arm | N | Mean score | Exact | Answer rate | Ref@300 | Frames | Natural recall | Final recall | Selected | Anchors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["arms"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["arm"]),
                    str(row["case_count"]),
                    _fmt(row["mean_score"]),
                    _fmt(row["exact_correct_rate"]),
                    _fmt(row["answer_rate"]),
                    _fmt(row["ref_300"]),
                    _fmt(row["mean_visual_frames"]),
                    _fmt(row["natural_clue_recall"]),
                    _fmt(row["final_clue_recall"]),
                    _fmt(row["mean_selected_candidates"]),
                    _fmt(row["mean_anchor_count"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "| Arm | Visual windows | Mean window sec | Clue overlap | Clue center | Selected requested | Selected inspected | Anchor requested | Anchor inspected | Candidate min sec | Anchor min sec |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for row in report["arms"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["arm"]),
                    _fmt(row["mean_visual_window_count"]),
                    _fmt(row["mean_visual_window_sec"]),
                    _fmt(row["clue_visual_recall"]),
                    _fmt(row["clue_center_visual_recall"]),
                    _fmt(row["selected_candidate_request_recall"]),
                    _fmt(row["selected_candidate_inspection_recall"]),
                    _fmt(row["anchor_request_recall"]),
                    _fmt(row["anchor_inspection_recall"]),
                    _fmt(row["mean_inspected_candidate_window_sec"]),
                    _fmt(row["mean_inspected_anchor_window_sec"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "| Comparison | N | Delta | 95% CI | W/T/L |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in report["paired_comparisons"]:
        lines.append(
            f"| {row['comparison']} | {row['case_count']} | "
            f"{_fmt(row['mean_score_delta'])} | "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(
        (
            "",
            "| Arm | Mean-score OGR | 95% CI | Exact OGR | 95% CI |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in report["oracle_gap_recovery"]:
        lines.append(
            f"| {row['arm']} | {_fmt(row['mean_score_ogr'])} | "
            f"[{_fmt(row['mean_score_ci95_low'])}, {_fmt(row['mean_score_ci95_high'])}] | "
            f"{_fmt(row['exact_correct_ogr'])} | "
            f"[{_fmt(row['exact_correct_ci95_low'])}, {_fmt(row['exact_correct_ci95_high'])}] |"
        )
    return "\n".join(lines) + "\n"


def _bootstrap_ci(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not differences:
        return None, None
    rng = random.Random(seed)
    count = len(differences)
    values = sorted(
        mean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(max(1, int(samples)))
    )
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def _bootstrap_ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    if not numerators or len(numerators) != len(denominators):
        return None, None, None
    denominator = mean(denominators)
    point = None if abs(denominator) < 1e-12 else mean(numerators) / denominator
    rng = random.Random(seed)
    count = len(numerators)
    values: list[float] = []
    for _ in range(max(1, int(samples))):
        indices = [rng.randrange(count) for _ in range(count)]
        sampled_denominator = mean(denominators[index] for index in indices)
        if abs(sampled_denominator) < 1e-12:
            continue
        values.append(
            mean(numerators[index] for index in indices) / sampled_denominator
        )
    if not values:
        return point, None, None
    values.sort()
    return (
        point,
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    )


def _duration_bucket(value: float) -> str:
    if value <= 30.0:
        return "<=30s"
    if value <= 120.0:
        return "30-120s"
    if value <= 300.0:
        return "120-300s"
    return ">300s"


def _optional_mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not Path(path).is_file():
        return ()
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return tuple(rows)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired oracle-ladder runs.")
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = collect_rows(
        tuple(Path(value) for value in args.run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
        case_ids=(frozenset(args.case_id) if args.case_id else None),
    )
    report = build_report(
        rows,
        expected_cases=args.expected_cases,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_text(Path(args.out_json), json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_text(Path(args.out_md), render_markdown(report))
    print(
        json.dumps(
            {
                "gate_passed": report["gate_passed"],
                "runtime_gate_passed": report["runtime_gate_passed"],
                "common_case_count": report["common_case_count"],
                "out_json": args.out_json,
                "out_md": args.out_md,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
