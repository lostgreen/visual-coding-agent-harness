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
            candidates = _occurrence_candidates(observations)
            selected_id = _selected_occurrence_id(trace)
            selected_range = candidates.get(selected_id)
            candidate_recall = bool(clues) and any(
                _overlap(candidate_range, clue)
                for candidate_range in candidates.values()
                for clue in clues
            )
            selected_correct = bool(
                selected_range
                and any(_overlap(selected_range, clue) for clue in clues)
            )
            answer_eval = evaluation.get("answer", {})
            grounding = evaluation.get("reference_grounding", {})
            metrics = runtime.get("runtime_metrics", {})
            score = answer_eval.get("score")
            ref_300 = grounding.get("ref_300")
            eligibility_round = _event_round(
                trace, "occurrence_treatment_eligible"
            )
            exposure = _event(trace, "occurrence_treatment_exposed")
            raw_no_oracle_audit = runtime.get("no_oracle_runtime_gate", {})
            no_oracle_audit = (
                raw_no_oracle_audit
                if isinstance(raw_no_oracle_audit, Mapping)
                else {}
            )
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
                    "selected_occurrence_id": selected_id or None,
                    "selected_occurrence_correct": selected_correct,
                    "osa_eligible": arm == "a2" and candidate_recall,
                    "osa_correct": (
                        selected_correct
                        if arm == "a2" and candidate_recall
                        else None
                    ),
                    "clue_count": len(clues),
                    "occurrence_handle_usage_rate": _handle_usage_rate(
                        trace, eligibility_round
                    ),
                    "premature_occurrence_commit": any(
                        bool(row.get("premature_occurrence_commit"))
                        for row in trace
                        if row.get("type") == "reasoner_decision"
                    ),
                    "treatment_eligible_round": eligibility_round,
                    "pre_treatment_signature": _pre_treatment_signature(
                        trace, eligibility_round
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
    if "a0" in by_arm:
        for arm in arms:
            if arm == "a0":
                continue
            paired_ids = sorted(set(by_arm["a0"]) & set(by_arm[arm]))
            comparable_ids = [
                case_id
                for case_id in paired_ids
                if by_arm["a0"][case_id].get("pre_treatment_signature")
                is not None
                and by_arm[arm][case_id].get("pre_treatment_signature")
                is not None
            ]
            matched_ids = [
                case_id
                for case_id in comparable_ids
                if by_arm["a0"][case_id]["pre_treatment_signature"]
                == by_arm[arm][case_id]["pre_treatment_signature"]
            ]
            comparisons[f"{arm}-a0"] = {
                **_paired_score_delta(
                    by_arm[arm],
                    by_arm["a0"],
                    paired_ids,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
                "pre_treatment_comparable_count": len(comparable_ids),
                "pre_treatment_divergence_count": len(comparable_ids)
                - len(matched_ids),
                "pre_treatment_divergence_rate": (
                    (len(comparable_ids) - len(matched_ids)) / len(comparable_ids)
                    if comparable_ids
                    else None
                ),
                "matched_pre_treatment_subset": _paired_score_delta(
                    by_arm[arm],
                    by_arm["a0"],
                    matched_ids,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 1,
                ),
            }
    text_parity = _text_parity(by_arm)
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
    }
    return {
        "schema_version": "MMLifelongOccurrenceAgentReportV1",
        "definitions": {
            "verified_correct": "judge score == 1 and runtime verification_status == verified",
            "correct_and_ref_300": "judge score == 1 and reference_grounding.ref_300 is true",
            "pre_treatment_divergence": "A0 and treatment action/task-query signatures differ before each run first exposes an occurrence candidate set",
            "occurrence_selection_accuracy": "selected occurrence overlaps at least one gold clue, conditioned on any natural candidate overlapping a clue",
            "occurrence_handle_usage_rate": "post-eligibility visual-window tasks with occurrence_id divided by all post-eligibility visual-window tasks",
        },
        "arms": arm_metrics,
        "comparisons": comparisons,
        "text_budget_parity": text_parity,
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
        "occurrence_selection_accuracy": _mean_bool(
            row.get("osa_correct") for row in osa
        ),
        "osa_eligible_count": len(osa),
        "occurrence_handle_usage_rate": _optional_mean(
            row.get("occurrence_handle_usage_rate") for row in rows
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


def _selected_occurrence_id(trace: Sequence[Mapping[str, Any]]) -> str:
    selected = ""
    for row in trace:
        if row.get("type") == "reasoner_decision" and (
            "selected_occurrence_id" in row
        ):
            selected = str(row.get("selected_occurrence_id", "") or "")
    return selected


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
    order = {"a0": 0, "a1-flat": 1, "a1": 2, "a2": 3}
    return order.get(arm, 99), arm


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MM-Lifelong Occurrence Agent Report",
        "",
        f"Structural gate: **{'PASS' if report['structural_gate_passed'] else 'FAIL'}**",
        "",
        "| Arm | N | Mean | Exact | Verified | Correct & Ref@300 | Candidate recall | OSA | Handle use | Premature | Frames | VLM calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    _fmt(metrics["correct_and_ref_300_rate"]),
                    _fmt(metrics["candidate_recall"]),
                    _fmt(metrics["occurrence_selection_accuracy"]),
                    _fmt(metrics["occurrence_handle_usage_rate"]),
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
            f"pre-treatment divergence {_fmt(comparison['pre_treatment_divergence_rate'])}."
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
