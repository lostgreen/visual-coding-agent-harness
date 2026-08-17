#!/usr/bin/env python3
"""Analyze two OOB negative-only sidecar repeats against frozen winners."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import (
    load_negative_sidecar_snapshot,
    negative_sidecar_forbidden_paths,
    positive_source_manifest_digest,
    replay_source_manifest_digest,
    scan_persisted_json_surface,
)


REQUIRED_FALSE_BLOCKS = 5
ALLOWED_POSITIVE_BLOCKS = 0
EXPECTED_FALSE_COMMITS = 12
EXPECTED_POSITIVE_COMMITS = 8


def collect_repeat(root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).glob("cases/*/sidecar_result.json")):
        row = _read_json(path)
        case_id = str(row.get("case_id", path.parent.name) or path.parent.name)
        results[case_id] = row
    if not results:
        raise FileNotFoundError(f"no sidecar results under {root}")
    return results


def build_report(
    repeats: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    frozen_rows: Sequence[Mapping[str, Any]],
    expected_cases: int,
    expected_false_commits: int = EXPECTED_FALSE_COMMITS,
    expected_positive_commits: int = EXPECTED_POSITIVE_COMMITS,
    required_false_blocks: int = REQUIRED_FALSE_BLOCKS,
    allowed_positive_blocks: int = ALLOWED_POSITIVE_BLOCKS,
    structural_only: bool = False,
    independent_audit: Mapping[str, Any] | None = None,
    candidate_ids_by_case: Mapping[str, Sequence[str]] | None = None,
    include_partial_valid: bool = False,
    bootstrap_samples: int = 10000,
    permutation_samples: int = 10000,
    seed: int = 20260817,
) -> dict[str, Any]:
    if len(repeats) != 2:
        raise ValueError("exactly two sidecar repeats are required")
    labels = tuple(repeats)
    left_label, right_label = labels
    left = repeats[left_label]
    right = repeats[right_label]
    aligned_ids = tuple(sorted(set(left) & set(right)))
    if len(left) != expected_cases or len(right) != expected_cases:
        raise ValueError("each repeat must contain the expected case count")
    if len(aligned_ids) != expected_cases:
        raise ValueError("sidecar repeat case sets are not aligned")

    frozen = {
        str(row.get("case_id", "") or ""): row
        for row in frozen_rows
        if str(row.get("arm", "") or "") == "a4"
    }
    if not set(aligned_ids) <= set(frozen):
        raise ValueError("sidecar cases are missing from frozen selections")
    frozen = {case_id: frozen[case_id] for case_id in aligned_ids}
    non_singleton_winner_case_ids = tuple(
        case_id
        for case_id in aligned_ids
        if frozen[case_id].get("final_resolution") == "selected"
        and len(
            tuple(
                value
                for value in tuple(
                    frozen[case_id].get("selected_occurrence_ids", ()) or ()
                )
                if str(value)
            )
        )
        != 1
    )
    false_ids = tuple(
        case_id
        for case_id in aligned_ids
        if case_id not in non_singleton_winner_case_ids
        if frozen[case_id].get("candidate_recall_resolved_set") is False
        and frozen[case_id].get("final_resolution") == "selected"
    )
    positive_ids = tuple(
        case_id
        for case_id in aligned_ids
        if case_id not in non_singleton_winner_case_ids
        if frozen[case_id].get("candidate_recall_resolved_set") is True
        and frozen[case_id].get("final_resolution") == "selected"
    )
    strict_correct_ids = tuple(
        case_id for case_id in positive_ids if frozen[case_id].get("osa_strict") is True
    )

    gates = _structural_gates(
        repeats,
        aligned_ids=aligned_ids,
        independent_audit=independent_audit,
        include_partial_valid=include_partial_valid,
        non_singleton_winner_case_ids=non_singleton_winner_case_ids,
    )
    gates["checks"]["frozen_denominators_match"] = bool(
        len(false_ids) == int(expected_false_commits)
        and len(positive_ids) == int(expected_positive_commits)
    )
    gates["passed"] = all(gates["checks"].values())
    normalized = {
        label: _normalize_repeat(
            repeats[label],
            frozen=frozen,
            false_ids=false_ids,
            positive_ids=positive_ids,
            candidate_ids_by_case=candidate_ids_by_case or {},
        )
        for label in labels
    }
    per_repeat = {
        label: _repeat_metrics(
            normalized[label],
            false_ids=false_ids,
            positive_ids=positive_ids,
            strict_correct_ids=strict_correct_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 101,
        )
        for index, label in enumerate(labels)
    }
    stability = _stability_metrics(
        normalized[left_label],
        normalized[right_label],
        aligned_ids=aligned_ids,
        false_ids=false_ids,
        positive_ids=positive_ids,
    )
    strata_by_case = {
        case_id: _permutation_stratum(
            frozen[case_id],
            candidate_count=len(normalized[left_label][case_id]["candidate_ids"]),
        )
        for case_id in (*false_ids, *positive_ids)
    }
    permutation_nulls = {
        label: {
            "within_case_pseudo_winner": _within_case_pseudo_winner_null(
                normalized[label],
                case_ids=false_ids,
                observed_rate=per_repeat[label][
                    "false_winner_contradiction_coverage"
                ],
                samples=permutation_samples,
                seed=seed + index * 211,
            ),
            "correctness_label": _correctness_label_permutation_null(
                normalized[label],
                false_ids=false_ids,
                positive_ids=positive_ids,
                strata_by_case=strata_by_case,
                observed_gap=per_repeat[label]["coverage_collateral_gap"],
                samples=permutation_samples,
                seed=seed + index * 307,
            ),
        }
        for index, label in enumerate(labels)
    }
    coverage_passed = all(
        row["false_winner_contradicted_count"] >= int(required_false_blocks)
        for row in per_repeat.values()
    )
    collateral_passed = all(
        row["positive_winner_contradicted_count"] <= int(allowed_positive_blocks)
        for row in per_repeat.values()
    )
    stable = bool(stability["stability_threshold_passed"])
    discrimination_established = all(
        row["coverage_collateral_gap_bootstrap"]["ci95"][0] is not None
        and row["coverage_collateral_gap_bootstrap"]["ci95"][0] > 0
        for row in per_repeat.values()
    )
    if structural_only:
        decision = "STRUCTURAL_CANARY_ONLY"
    elif not gates["passed"]:
        decision = "HISTORICAL_RUN_PROVENANCE_INCOMPLETE"
    elif discrimination_established and stable:
        decision = "NEGATIVE_ROW_QUALITY_AUDIT_REQUIRED"
    else:
        decision = "WINNER_DISCRIMINATION_NOT_ESTABLISHED"

    case_rows = []
    for case_id in (*false_ids, *positive_ids):
        first = normalized[left_label][case_id]
        second = normalized[right_label][case_id]
        case_rows.append(
            {
                "case_id": case_id,
                "candidate_status": (
                    "candidate_absent" if case_id in false_ids else "candidate_present"
                ),
                "osa_strict": bool(frozen[case_id].get("osa_strict")),
                "frozen_winner": first["winner"],
                f"{left_label}_winner_contradicted": first["winner_contradicted"],
                f"{right_label}_winner_contradicted": second["winner_contradicted"],
                f"{left_label}_constraint_types": first["winner_constraint_types"],
                f"{right_label}_constraint_types": second["winner_constraint_types"],
                "winner_flag_agrees": (
                    first["winner_contradicted"] == second["winner_contradicted"]
                ),
            }
        )
    actual_models = sorted(
        {
            str(row.get("actual_model", "") or "")
            for repeat in repeats.values()
            for row in repeat.values()
            if str(row.get("actual_model", "") or "")
        }
    )
    return {
        "schema_version": "MMLifelongOccurrenceNegativeSidecarAnalysisV2",
        "repeat_labels": list(labels),
        "case_count": len(aligned_ids),
        "frozen_counts": {
            "false_commit_count": len(false_ids),
            "positive_commit_count": len(positive_ids),
            "strict_correct_commit_count": len(strict_correct_ids),
        },
        "qualification": {
            "required_false_blocks": int(required_false_blocks),
            "expected_false_commits": int(expected_false_commits),
            "allowed_positive_blocks": int(allowed_positive_blocks),
            "expected_positive_commits": int(expected_positive_commits),
            "coverage_passed": coverage_passed,
            "collateral_passed": collateral_passed,
            "exact_winner_flag_stability_passed": stable,
            "discrimination_established": discrimination_established,
            "structural_only": structural_only,
        },
        "structural_gates": gates,
        "independent_audit": dict(independent_audit or {}),
        "actual_models": actual_models,
        "per_repeat": per_repeat,
        "stability": stability,
        "permutation_nulls": permutation_nulls,
        "non_singleton_winner_case_ids": list(non_singleton_winner_case_ids),
        "partial_valid_policy": {
            "included_in_primary_analysis": bool(include_partial_valid),
        },
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "seed": seed,
        "case_rows": case_rows,
        "decision": decision,
        "winner_discrimination_conclusion": (
            "ESTABLISHED" if discrimination_established else "NOT_ESTABLISHED"
        ),
        "endpoint_values_were_gates": False,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    labels = tuple(report["repeat_labels"])
    lines = [
        "# WP11 OOB Negative-Only Sidecar",
        "",
        f"Decision: **{report['decision']}**",
        (
            "Winner-correctness discrimination: "
            f"**{report['winner_discrimination_conclusion']}**."
        ),
        "",
        (
            f"Structural gates: **{'PASS' if report['structural_gates']['passed'] else 'FAIL'}**. "
            "Endpoint values were not structural gates."
        ),
        f"Model: {', '.join(report['actual_models']) or 'unknown'}.",
        "",
        "| Repeat | False-winner coverage (Wilson 95%) | Positive collateral (Wilson 95%) | Gap (bootstrap 95%) | Strict-correct collision |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = report["per_repeat"][label]
        lines.append(
            f"| {label} | {row['false_winner_contradicted_count']}/"
            f"{row['false_commit_count']} {_fmt_ci(row['false_winner_contradiction_coverage_wilson95'])} | "
            f"{row['positive_winner_contradicted_count']}/"
            f"{row['positive_commit_count']} {_fmt_ci(row['positive_winner_contradiction_rate_wilson95'])} | "
            f"{_fmt(row['coverage_collateral_gap'])} "
            f"{_fmt_ci(row['coverage_collateral_gap_bootstrap']['ci95'])} | "
            f"{row['strict_correct_winner_contradicted_count']}/"
            f"{row['strict_correct_commit_count']} |"
        )
    stability = report["stability"]
    constraint_types = sorted(
        {
            constraint_type
            for label in labels
            for key in (
                "false_winner_contradiction_by_type",
                "positive_winner_contradiction_by_type",
            )
            for constraint_type in report["per_repeat"][label][key]
        }
    )
    lines.extend(
        [
            "",
            "## Winner contradiction by constraint type",
            "",
            f"| Type | {labels[0]} false | {labels[0]} positive | {labels[1]} false | {labels[1]} positive |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for constraint_type in constraint_types:
        lines.append(
            f"| {constraint_type} | "
            f"{report['per_repeat'][labels[0]]['false_winner_contradiction_by_type'].get(constraint_type, 0)} | "
            f"{report['per_repeat'][labels[0]]['positive_winner_contradiction_by_type'].get(constraint_type, 0)} | "
            f"{report['per_repeat'][labels[1]]['false_winner_contradiction_by_type'].get(constraint_type, 0)} | "
            f"{report['per_repeat'][labels[1]]['positive_winner_contradiction_by_type'].get(constraint_type, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Stability",
            "",
            (
                f"Row/candidate/constraint/strict-passage Jaccard: "
                f"{_fmt(stability['row_jaccard'])} / "
                f"{_fmt(stability['candidate_jaccard'])} / "
                f"{_fmt(stability['constraint_jaccard'])} / "
                f"{_fmt(stability['strict_passage_jaccard'])}."
            ),
            (
                f"Winner contradiction flag agreement: "
                f"{stability['winner_flag_agreement_count']}/"
                f"{stability['winner_flag_case_count']} "
                f"({_fmt(stability['winner_flag_agreement_rate'])}); exact="
                f"{stability['winner_flag_exact_agreement']}."
            ),
            (
                "Agreement Wilson 95% / Cohen's kappa / threshold pass: "
                f"{_fmt_ci(stability['winner_flag_agreement_wilson95'])} / "
                f"{_fmt(stability['winner_flag_cohen_kappa'])} / "
                f"{stability['stability_threshold_passed']}."
            ),
            (
                "False/positive winner-flag agreement: "
                f"{_fmt(stability['false_winner_flag_agreement_rate'])} / "
                f"{_fmt(stability['positive_winner_flag_agreement_rate'])}."
            ),
            (
                "Nonempty-case mean row Jaccard / both-empty cases: "
                f"{_fmt(stability['mean_case_row_jaccard_nonempty'])} / "
                f"{stability['both_empty_case_count']}."
            ),
            "",
            "## Zero-model permutation nulls",
            "",
            "| Repeat | Pseudo-winner null mean (95%) | p(actual false coverage >= null) | Label-null mean gap (95%) | p(actual gap >= null) |",
            "|---|---:|---:|---:|---:|",
            *(
                "| "
                f"{label} | "
                f"{_fmt(report['permutation_nulls'][label]['within_case_pseudo_winner'].get('mean_rate'))} "
                f"{_fmt_ci(report['permutation_nulls'][label]['within_case_pseudo_winner']['ci95'])} | "
                f"{_fmt(report['permutation_nulls'][label]['within_case_pseudo_winner']['p_ge_observed'])} | "
                f"{_fmt(report['permutation_nulls'][label]['correctness_label'].get('mean_gap'))} "
                f"{_fmt_ci(report['permutation_nulls'][label]['correctness_label']['ci95'])} | "
                f"{_fmt(report['permutation_nulls'][label]['correctness_label']['p_ge_observed'])} |"
                for label in labels
            ),
            "",
            "## Frozen winner cases",
            "",
            f"| Case | Status | OSA | {labels[0]} contradicted | {labels[1]} contradicted | Agreement |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in report["case_rows"]:
        lines.append(
            f"| {row['case_id']} | {row['candidate_status']} | "
            f"{row['osa_strict']} | "
            f"{row[f'{labels[0]}_winner_contradicted']} | "
            f"{row[f'{labels[1]}_winner_contradicted']} | "
            f"{row['winner_flag_agrees']} |"
        )
    lines.extend(
        [
            "",
            "Winner correctness discrimination is established only when the "
            "case-cluster bootstrap gap CI lower bound is above zero in both "
            "repeats. Historical endpoint thresholds are diagnostics, not gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def _normalize_repeat(
    repeat: Mapping[str, Mapping[str, Any]],
    *,
    frozen: Mapping[str, Mapping[str, Any]],
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
    candidate_ids_by_case: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    commit_ids = set((*false_ids, *positive_ids))
    normalized: dict[str, dict[str, Any]] = {}
    for case_id, result in repeat.items():
        frozen_row = frozen[case_id]
        selected = tuple(
            str(value)
            for value in tuple(frozen_row.get("selected_occurrence_ids", ()) or ())
            if str(value)
        )
        winner = selected[0] if len(selected) == 1 else ""
        contradiction_rows = tuple(
            row
            for row in tuple(result.get("contradiction_rows", ()) or ())
            if isinstance(row, Mapping)
        )
        winner_rows = tuple(
            row
            for row in contradiction_rows
            if winner and str(row.get("occurrence_id", "") or "") == winner
        )
        candidate_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in tuple(candidate_ids_by_case.get(case_id, ()) or ())
                if str(value)
            )
        )
        if not candidate_ids:
            candidate_ids = tuple(
                dict.fromkeys(
                    (
                        *(
                            str(row.get("occurrence_id", "") or "")
                            for row in contradiction_rows
                            if str(row.get("occurrence_id", "") or "")
                        ),
                        *((winner,) if winner else ()),
                    )
                )
            )
        contradicted_candidate_ids = tuple(
            sorted(
                {
                    str(row.get("occurrence_id", "") or "")
                    for row in contradiction_rows
                    if str(row.get("occurrence_id", "") or "")
                }
            )
        )
        normalized[case_id] = {
            "winner": winner if case_id in commit_ids else "",
            "winner_contradicted": bool(winner_rows),
            "winner_constraint_types": sorted(
                {
                    str(row.get("constraint_type", "unknown") or "unknown")
                    for row in winner_rows
                }
            ),
            "winner_passage_ids": sorted(
                {
                    str(passage_id)
                    for row in winner_rows
                    for passage_id in tuple(row.get("evidence_passage_ids", ()) or ())
                    if str(passage_id)
                }
            ),
            "rows": contradiction_rows,
            "candidate_ids": candidate_ids,
            "candidate_count": int(
                dict(result.get("input_counts", {}) or {}).get(
                    "candidate_count", len(candidate_ids)
                )
                or len(candidate_ids)
            ),
            "contradicted_candidate_ids": contradicted_candidate_ids,
        }
    return normalized


def _repeat_metrics(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
    strict_correct_ids: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    false_hits = tuple(
        case_id for case_id in false_ids if rows[case_id]["winner_contradicted"]
    )
    positive_hits = tuple(
        case_id for case_id in positive_ids if rows[case_id]["winner_contradicted"]
    )
    strict_hits = tuple(
        case_id
        for case_id in strict_correct_ids
        if rows[case_id]["winner_contradicted"]
    )
    false_rate = _ratio(len(false_hits), len(false_ids))
    positive_rate = _ratio(len(positive_hits), len(positive_ids))
    gap = (
        false_rate - positive_rate
        if false_rate is not None and positive_rate is not None
        else None
    )
    row_counts = [len(tuple(row.get("rows", ()) or ())) for row in rows.values()]
    contradicted_fractions = [
        _ratio(
            len(tuple(row.get("contradicted_candidate_ids", ()) or ())),
            int(row.get("candidate_count", 0) or 0),
        )
        for row in rows.values()
    ]
    contradicted_fractions = [
        value for value in contradicted_fractions if value is not None
    ]
    degenerate = tuple(
        case_id
        for case_id, row in rows.items()
        if int(row.get("candidate_count", 0) or 0) > 0
        and len(tuple(row.get("contradicted_candidate_ids", ()) or ()))
        >= int(row.get("candidate_count", 0) or 0)
    )
    return {
        "false_commit_count": len(false_ids),
        "false_winner_contradicted_count": len(false_hits),
        "false_winner_contradiction_coverage": false_rate,
        "false_winner_contradiction_coverage_wilson95": _wilson_interval(
            len(false_hits), len(false_ids)
        ),
        "positive_commit_count": len(positive_ids),
        "positive_winner_contradicted_count": len(positive_hits),
        "positive_winner_contradiction_rate": positive_rate,
        "positive_winner_contradiction_rate_wilson95": _wilson_interval(
            len(positive_hits), len(positive_ids)
        ),
        "coverage_collateral_gap": gap,
        "coverage_collateral_gap_bootstrap": _bootstrap_rate_gap(
            rows,
            false_ids=false_ids,
            positive_ids=positive_ids,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "strict_correct_commit_count": len(strict_correct_ids),
        "strict_correct_winner_contradicted_count": len(strict_hits),
        "strict_correct_winner_contradiction_rate": _ratio(
            len(strict_hits), len(strict_correct_ids)
        ),
        "false_winner_contradiction_by_type": _type_counts(rows, false_hits),
        "positive_winner_contradiction_by_type": _type_counts(rows, positive_hits),
        "false_winner_contradicted_case_ids": list(false_hits),
        "positive_winner_contradicted_case_ids": list(positive_hits),
        "contradiction_row_count": sum(row_counts),
        "rows_per_case_distribution": _distribution(row_counts),
        "contradicted_fraction_of_scope": _distribution(contradicted_fractions),
        "degenerate_contradict_all_case_ids": list(degenerate),
        "degenerate_contradict_all_rate": _ratio(len(degenerate), len(rows)),
        "by_scope_size": _scope_size_metrics(rows),
    }


def _stability_metrics(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    aligned_ids: Sequence[str],
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
) -> dict[str, Any]:
    left_sets = _evidence_sets(left)
    right_sets = _evidence_sets(right)
    commit_ids = (*false_ids, *positive_ids)
    agreement = sum(
        left[case_id]["winner_contradicted"] == right[case_id]["winner_contradicted"]
        for case_id in commit_ids
    )
    false_agreement = sum(
        left[case_id]["winner_contradicted"] == right[case_id]["winner_contradicted"]
        for case_id in false_ids
    )
    positive_agreement = sum(
        left[case_id]["winner_contradicted"] == right[case_id]["winner_contradicted"]
        for case_id in positive_ids
    )
    case_jaccards = [
        _case_row_jaccard(left[case_id], right[case_id])
        for case_id in aligned_ids
    ]
    nonempty_case_jaccards = [value for value in case_jaccards if value is not None]
    left_flags = [bool(left[case_id]["winner_contradicted"]) for case_id in commit_ids]
    right_flags = [
        bool(right[case_id]["winner_contradicted"]) for case_id in commit_ids
    ]
    kappa = _cohen_kappa(left_flags, right_flags)
    agreement_wilson95 = _wilson_interval(agreement, len(commit_ids))
    agreement_lower = agreement_wilson95[0]
    stability_passed = bool(
        kappa is not None
        and kappa >= 0.6
        and agreement_lower is not None
        and agreement_lower >= 0.75
    )
    return {
        "row_jaccard": _jaccard(left_sets["row"], right_sets["row"]),
        "candidate_jaccard": _jaccard(left_sets["candidate"], right_sets["candidate"]),
        "constraint_jaccard": _jaccard(
            left_sets["constraint"], right_sets["constraint"]
        ),
        "strict_passage_jaccard": _jaccard(left_sets["strict"], right_sets["strict"]),
        "mean_case_row_jaccard_nonempty": (
            mean(nonempty_case_jaccards) if nonempty_case_jaccards else None
        ),
        "both_empty_case_count": sum(value is None for value in case_jaccards),
        "winner_flag_case_count": len(commit_ids),
        "winner_flag_agreement_count": agreement,
        "winner_flag_agreement_rate": _ratio(agreement, len(commit_ids)),
        "winner_flag_agreement_wilson95": agreement_wilson95,
        "winner_flag_cohen_kappa": kappa,
        "stability_threshold_passed": stability_passed,
        "winner_flag_exact_agreement": agreement == len(commit_ids),
        "false_winner_flag_agreement_rate": _ratio(false_agreement, len(false_ids)),
        "positive_winner_flag_agreement_rate": _ratio(
            positive_agreement, len(positive_ids)
        ),
    }


def _structural_gates(
    repeats: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    aligned_ids: Sequence[str],
    independent_audit: Mapping[str, Any] | None,
    include_partial_valid: bool,
    non_singleton_winner_case_ids: Sequence[str],
) -> dict[str, Any]:
    labels = tuple(repeats)
    actual_models = {
        str(row.get("actual_model", "") or "")
        for repeat in repeats.values()
        for row in repeat.values()
    }
    accepted_statuses = {"success"}
    if include_partial_valid:
        accepted_statuses.add("partial_valid")
    independent_checks = dict(
        (independent_audit or {}).get("checks", {}) or {}
    )
    checks: dict[str, bool] = {
        "aligned_case_sets": all(
            set(repeats[label]) == set(aligned_ids) for label in labels
        ),
        "all_results_primary_eligible": all(
            row.get("status") in accepted_statuses
            for repeat in repeats.values()
            for row in repeat.values()
        ),
        "repeat_labels_valid": all(
            row.get("repeat_label") == label
            for label, repeat in repeats.items()
            for row in repeat.values()
        ),
        "actual_model_consistent": len(actual_models) == 1 and "" not in actual_models,
        "model_call_evidence_present": all(
            bool(row.get("model_response_digest"))
            for repeat in repeats.values()
            for row in repeat.values()
        ),
        "recorded_snapshot_inputs_matched": all(
            repeats[labels[0]][case_id].get("snapshot_digest")
            == repeats[labels[1]][case_id].get("snapshot_digest")
            for case_id in aligned_ids
        ),
        "non_singleton_winner_absent": not non_singleton_winner_case_ids,
        "independent_audit_available": bool(independent_audit),
    }
    checks.update(
        {f"independent_{key}": bool(value) for key, value in independent_checks.items()}
    )
    return {"passed": all(checks.values()), "checks": checks}


def _evidence_sets(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[tuple[str, ...]]]:
    result = {"row": set(), "candidate": set(), "constraint": set(), "strict": set()}
    for case_id, case in rows.items():
        for row in tuple(case.get("rows", ()) or ()):
            constraint_id = str(row.get("constraint_id", "") or "")
            occurrence_id = str(row.get("occurrence_id", "") or "")
            result["row"].add((case_id, constraint_id, occurrence_id))
            result["candidate"].add((case_id, occurrence_id))
            result["constraint"].add((case_id, constraint_id))
            for passage_id in tuple(row.get("evidence_passage_ids", ()) or ()):
                result["strict"].add(
                    (case_id, constraint_id, occurrence_id, str(passage_id))
                )
    return result


def _case_row_jaccard(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float | None:
    def keys(row: Mapping[str, Any]) -> set[tuple[str, str]]:
        return {
            (
                str(value.get("constraint_id", "") or ""),
                str(value.get("occurrence_id", "") or ""),
            )
            for value in tuple(row.get("rows", ()) or ())
        }

    return _jaccard(keys(left), keys(right))


def _type_counts(
    rows: Mapping[str, Mapping[str, Any]], case_ids: Sequence[str]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case_id in case_ids:
        counts.update(set(rows[case_id]["winner_constraint_types"]))
    return dict(sorted(counts.items()))


def build_independent_audit(
    repeat_roots: Mapping[str, Path],
    *,
    positive_run_root: Path,
    replay_fixture_root: Path,
    case_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    ids = tuple(sorted(set(case_ids)))
    positive_digest = positive_source_manifest_digest(positive_run_root, ids)
    replay_digest = replay_source_manifest_digest(replay_fixture_root, ids)
    snapshots: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for case_id in ids:
        try:
            snapshots[case_id] = load_negative_sidecar_snapshot(
                Path(positive_run_root) / "cases" / case_id,
                replay_fixture_path=(
                    Path(replay_fixture_root) / "cases" / f"{case_id}.json"
                ),
            )
        except Exception as exc:
            failures.append(
                {"case_id": case_id, "error_type": type(exc).__name__}
            )
    candidate_ids_by_case = {
        case_id: tuple(
            str(row.get("occurrence_id", "") or "")
            for row in snapshot.candidates
            if str(row.get("occurrence_id", "") or "")
        )
        for case_id, snapshot in snapshots.items()
    }
    repeat_checks: dict[str, dict[str, bool]] = {}
    persisted_scans: dict[str, dict[str, Any]] = {}
    for label, root in repeat_roots.items():
        run_root = Path(root)
        manifest_path = run_root / "run_manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        results = collect_repeat(run_root)
        persisted_scan = scan_persisted_json_surface(run_root)
        persisted_scans[label] = persisted_scan
        result_rows = [results.get(case_id, {}) for case_id in ids]
        repeat_checks[label] = {
            "snapshot_digest_recomputed_matches": all(
                case_id in snapshots
                and results.get(case_id, {}).get("snapshot_digest")
                == snapshots[case_id].digest
                for case_id in ids
            ),
            "source_digests_recomputed_match": all(
                case_id in snapshots
                and dict(results.get(case_id, {}).get("source_digests", {}) or {})
                == {
                    "case_sha256": snapshots[case_id].source_case_sha256,
                    "runtime_sha256": snapshots[case_id].source_runtime_sha256,
                    "replay_fixture_sha256": snapshots[
                        case_id
                    ].replay_fixture_sha256,
                }
                for case_id in ids
            ),
            "packet_match_exact": all(
                case_id in snapshots
                and snapshots[case_id].packet_match_mode == "exact"
                and results.get(case_id, {}).get("packet_match_mode") == "exact"
                for case_id in ids
            ),
            "persisted_surface_scan_passed": bool(persisted_scan["passed"]),
            "positive_root_unmodified_recorded": bool(
                manifest.get("positive_root_unmodified") is True
                and manifest.get("positive_source_manifest_before")
                == manifest.get("positive_source_manifest_after")
                == positive_digest
            ),
            "replay_manifest_recomputed_matches": (
                manifest.get("replay_source_manifest_digest") == replay_digest
            ),
            "reproducibility_settings_recorded": all(
                key in manifest
                for key in (
                    "temperature",
                    "top_p",
                    "requested_seed",
                    "provider_seed_supported",
                    "provider_reported_seed_support",
                    "response_format",
                )
            ),
            "attempt_provenance_complete": all(
                isinstance(row.get("attempt_count"), int)
                and row.get("attempt_count", 0) >= 1
                and isinstance(row.get("attempt_history"), Sequence)
                and not isinstance(row.get("attempt_history"), (str, bytes))
                and len(tuple(row.get("attempt_history", ()) or ()))
                == int(row.get("attempt_count", 0) or 0)
                and isinstance(row.get("status_history"), Sequence)
                and not isinstance(row.get("status_history"), (str, bytes))
                and "resumed_from_failure" in row
                for row in result_rows
            ),
            "out_of_band_root_separate": _roots_are_separate(
                run_root, positive_run_root
            ),
        }
    payload_checks = {
        case_id: not negative_sidecar_forbidden_paths(snapshot.model_payload())
        for case_id, snapshot in snapshots.items()
    }
    checks = {
        "snapshot_reconstruction_complete": len(snapshots) == len(ids) and not failures,
        "recomputed_payload_negative_only": len(payload_checks) == len(ids)
        and all(payload_checks.values()),
        **{
            f"{label}_{key}": value
            for label, values in repeat_checks.items()
            for key, value in values.items()
        },
    }
    return (
        {
            "schema_version": "MMLifelongOccurrenceNegativeSidecarIndependentAuditV1",
            "checks": checks,
            "passed": all(checks.values()),
            "failure_count": len(failures),
            "failures": failures,
            "positive_source_manifest_digest": positive_digest,
            "replay_source_manifest_digest": replay_digest,
            "persisted_surface_scans": persisted_scans,
        },
        candidate_ids_by_case,
    )


def _roots_are_separate(left: Path, right: Path) -> bool:
    left_resolved = Path(left).resolve()
    right_resolved = Path(right).resolve()
    return (
        left_resolved != right_resolved
        and left_resolved not in right_resolved.parents
        and right_resolved not in left_resolved.parents
    )


def _bootstrap_rate_gap(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not false_ids or not positive_ids or samples <= 0:
        return {"samples": samples, "ci95": [None, None], "positive_probability": None}
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(samples):
        sampled_false = [rng.choice(false_ids) for _ in false_ids]
        sampled_positive = [rng.choice(positive_ids) for _ in positive_ids]
        false_rate = mean(
            bool(rows[case_id]["winner_contradicted"])
            for case_id in sampled_false
        )
        positive_rate = mean(
            bool(rows[case_id]["winner_contradicted"])
            for case_id in sampled_positive
        )
        gaps.append(false_rate - positive_rate)
    gaps.sort()
    return {
        "samples": samples,
        "ci95": [_quantile(gaps, 0.025), _quantile(gaps, 0.975)],
        "positive_probability": sum(value > 0 for value in gaps) / len(gaps),
    }


def _within_case_pseudo_winner_null(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    observed_rate: float | None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible = tuple(
        case_id for case_id in case_ids if tuple(rows[case_id]["candidate_ids"])
    )
    if not eligible or samples <= 0:
        return {"samples": samples, "eligible_cases": len(eligible), "ci95": [None, None], "p_ge_observed": None}
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(samples):
        hits = 0
        for case_id in eligible:
            pseudo = rng.choice(tuple(rows[case_id]["candidate_ids"]))
            hits += pseudo in set(rows[case_id]["contradicted_candidate_ids"])
        rates.append(hits / len(eligible))
    rates.sort()
    return {
        "samples": samples,
        "eligible_cases": len(eligible),
        "mean_rate": mean(rates),
        "ci95": [_quantile(rates, 0.025), _quantile(rates, 0.975)],
        "p_ge_observed": (
            (1 + sum(value >= observed_rate for value in rates)) / (samples + 1)
            if observed_rate is not None
            else None
        ),
    }


def _correctness_label_permutation_null(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    false_ids: Sequence[str],
    positive_ids: Sequence[str],
    strata_by_case: Mapping[str, tuple[Any, ...]],
    observed_gap: float | None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = {case_id: case_id in false_ids for case_id in (*false_ids, *positive_ids)}
    strata: dict[tuple[Any, ...], list[str]] = {}
    for case_id in labels:
        strata.setdefault(strata_by_case[case_id], []).append(case_id)
    exchangeable = sum(
        len(case_ids)
        for case_ids in strata.values()
        if len({labels[case_id] for case_id in case_ids}) > 1
    )
    if not false_ids or not positive_ids or samples <= 0:
        return {"samples": samples, "exchangeable_case_count": exchangeable, "ci95": [None, None], "p_ge_observed": None}
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(samples):
        permuted = dict(labels)
        for case_ids in strata.values():
            shuffled = [labels[case_id] for case_id in case_ids]
            rng.shuffle(shuffled)
            permuted.update(zip(case_ids, shuffled))
        permuted_false = [case_id for case_id, value in permuted.items() if value]
        permuted_positive = [case_id for case_id, value in permuted.items() if not value]
        left = mean(bool(rows[case_id]["winner_contradicted"]) for case_id in permuted_false)
        right = mean(bool(rows[case_id]["winner_contradicted"]) for case_id in permuted_positive)
        gaps.append(left - right)
    gaps.sort()
    return {
        "samples": samples,
        "exchangeable_case_count": exchangeable,
        "mean_gap": mean(gaps),
        "ci95": [_quantile(gaps, 0.025), _quantile(gaps, 0.975)],
        "p_ge_observed": (
            (1 + sum(value >= observed_gap for value in gaps)) / (samples + 1)
            if observed_gap is not None
            else None
        ),
    }


def _permutation_stratum(
    row: Mapping[str, Any], *, candidate_count: int
) -> tuple[Any, ...]:
    return (
        int(row.get("sufficiency_scope_candidate_count", candidate_count) or candidate_count),
        row.get("sufficiency_minimum_support_margin"),
        row.get("sufficiency_sufficient_candidate_count"),
    )


def _scope_size_metrics(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows.values():
        grouped.setdefault(int(row.get("candidate_count", 0) or 0), []).append(row)
    return {
        str(size): {
            "case_count": len(group),
            "mean_contradiction_rows": mean(
                len(tuple(row.get("rows", ()) or ())) for row in group
            ),
            "mean_contradicted_fraction": mean(
                _ratio(
                    len(tuple(row.get("contradicted_candidate_ids", ()) or ())),
                    size,
                )
                or 0.0
                for row in group
            ),
        }
        for size, group in sorted(grouped.items())
    }


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None, "mean": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "q1": _quantile(ordered, 0.25),
        "median": median(ordered),
        "q3": _quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def _cohen_kappa(left: Sequence[bool], right: Sequence[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = mean(a == b for a, b in zip(left, right))
    left_positive = mean(left)
    right_positive = mean(right)
    expected = left_positive * right_positive + (1 - left_positive) * (1 - right_positive)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    if total <= 0:
        return [None, None]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _jaccard(left: set[Any], right: set[Any]) -> float | None:
    union = left | right
    return len(left & right) / len(union) if union else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _fmt_ci(value: Sequence[Any]) -> str:
    values = tuple(value or ())
    if len(values) != 2 or any(item is None for item in values):
        return "[NA, NA]"
    return f"[{_fmt(values[0])}, {_fmt(values[1])}]"


def _load_frozen_rows(
    run_root: Path, *, evaluation_record_root: Path
) -> tuple[dict[str, Any], ...]:
    module_path = Path(__file__).with_name("analyze_mmlifelong_occurrence_agent.py")
    spec = importlib.util.spec_from_file_location(
        "sidecar_frozen_analysis", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load occurrence analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(
        module.collect_rows(
            (Path(run_root),),
            evaluation_record_root=Path(evaluation_record_root),
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_repeat(value: str) -> tuple[str, Path]:
    label, separator, path = str(value).partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("repeat must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", action="append", type=_parse_repeat, required=True)
    parser.add_argument("--frozen-run-root", required=True)
    parser.add_argument("--replay-fixture-root", required=True)
    parser.add_argument("--evaluation-record-root", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument(
        "--expected-false-commits", type=int, default=EXPECTED_FALSE_COMMITS
    )
    parser.add_argument(
        "--expected-positive-commits", type=int, default=EXPECTED_POSITIVE_COMMITS
    )
    parser.add_argument(
        "--required-false-blocks", type=int, default=REQUIRED_FALSE_BLOCKS
    )
    parser.add_argument(
        "--allowed-positive-blocks", type=int, default=ALLOWED_POSITIVE_BLOCKS
    )
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--include-partial-valid", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    repeat_roots = {label: path for label, path in args.repeat}
    repeats = {label: collect_repeat(path) for label, path in args.repeat}
    aligned_ids = tuple(sorted(set.intersection(*(set(rows) for rows in repeats.values()))))
    independent_audit, candidate_ids_by_case = build_independent_audit(
        repeat_roots,
        positive_run_root=Path(args.frozen_run_root),
        replay_fixture_root=Path(args.replay_fixture_root),
        case_ids=aligned_ids,
    )
    frozen_rows = _load_frozen_rows(
        Path(args.frozen_run_root),
        evaluation_record_root=Path(args.evaluation_record_root),
    )
    report = build_report(
        repeats,
        frozen_rows=frozen_rows,
        expected_cases=args.expected_cases,
        expected_false_commits=args.expected_false_commits,
        expected_positive_commits=args.expected_positive_commits,
        required_false_blocks=args.required_false_blocks,
        allowed_positive_blocks=args.allowed_positive_blocks,
        structural_only=args.structural_only,
        independent_audit=independent_audit,
        candidate_ids_by_case=candidate_ids_by_case,
        include_partial_valid=args.include_partial_valid,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    _write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"SIDECAR_ANALYSIS_DONE decision={report['decision']} "
        f"structural_gate={report['structural_gates']['passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
