#!/usr/bin/env python3
"""Compare an operational conflict verifier with frozen blind reference labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from vcah.occurrence_evaluation_stats import (
    cohen_kappa,
    fisher_exact_two_sided,
    newcombe_difference_interval,
    wilson_interval,
)


VALID_VERDICTS = (
    "false_contradiction",
    "true_contradiction",
    "unclear",
)
POSITIVE_VERDICT = "true_contradiction"


def analyze_transfer(
    key: Mapping[str, Any],
    reference: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    reference_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    expected_reference_model: str,
    expected_runtime_model: str,
    bootstrap_samples: int,
    seed: int,
    kappa_threshold: float = 0.60,
    collateral_threshold: float = 0.20,
    fisher_alpha: float = 0.10,
) -> dict[str, Any]:
    key_rows = _key_rows(key)
    reference_rows, reference_invalid = _judgment_rows(reference)
    runtime_rows, runtime_invalid = _judgment_rows(runtime)
    expected_ids = set(key_rows)
    reference_ids = set(reference_rows)
    runtime_ids = set(runtime_rows)
    reference_model = str(reference.get("actual_model", "") or "")
    runtime_model = str(runtime.get("actual_model", "") or "")
    protocol_digest = str(key.get("judgment_protocol_digest", "") or "")

    manifest_checks = _manifest_checks(
        key,
        reference,
        runtime,
        reference_manifest=reference_manifest,
        runtime_manifest=runtime_manifest,
    )
    structural_checks = {
        "reference_item_set_exact": reference_ids == expected_ids,
        "runtime_item_set_exact": runtime_ids == expected_ids,
        "reference_verdicts_valid": not reference_invalid,
        "runtime_verdicts_valid": not runtime_invalid,
        "reference_protocol_matches": str(
            reference.get("judgment_protocol_digest", "") or ""
        )
        == protocol_digest,
        "runtime_protocol_matches": str(
            runtime.get("judgment_protocol_digest", "") or ""
        )
        == protocol_digest,
        "reference_model_matches": reference_model == expected_reference_model,
        "runtime_model_matches": runtime_model == expected_runtime_model,
        "models_are_distinct": bool(
            reference_model and runtime_model and reference_model != runtime_model
        ),
        **manifest_checks,
    }
    structural_gate_passed = all(structural_checks.values())
    paired_ids = tuple(sorted(expected_ids & reference_ids & runtime_ids))
    pairs = [
        (reference_rows[item_id], runtime_rows[item_id]) for item_id in paired_ids
    ]
    row_transfer = _row_transfer_metrics(pairs)
    reference_winners = _winner_report(
        key_rows,
        key.get("winner_cases", ()),
        reference_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 101,
    )
    runtime_winners = _winner_report(
        key_rows,
        key.get("winner_cases", ()),
        runtime_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 503,
    )
    winner_flag_transfer = _winner_flag_transfer(
        reference_winners.get("flags", {}), runtime_winners.get("flags", {})
    )

    row_kappa_passed = bool(
        row_transfer["cohen_kappa"] is not None
        and row_transfer["cohen_kappa"] >= kappa_threshold
    )
    per_repeat_criteria = {
        repeat: _winner_repeat_criteria(
            metrics,
            collateral_threshold=collateral_threshold,
            fisher_alpha=fisher_alpha,
        )
        for repeat, metrics in runtime_winners.get("per_repeat", {}).items()
    }
    winner_discrimination_passed = bool(
        per_repeat_criteria
        and all(
            row["false_candidate_separation_passed"]
            for row in per_repeat_criteria.values()
        )
    )
    correct_winner_collateral_passed = bool(
        per_repeat_criteria
        and all(
            row["correct_winner_collateral_passed"]
            for row in per_repeat_criteria.values()
        )
    )
    decision_checks = {
        "row_kappa_passed": row_kappa_passed,
        "winner_discrimination_passed": winner_discrimination_passed,
        "correct_winner_collateral_passed": correct_winner_collateral_passed,
    }
    transfer_feasible = bool(
        structural_gate_passed and all(decision_checks.values())
    )
    return {
        "schema_version": "MMLifelongOccurrenceConflictVerifierTransferV1",
        "study": "WP13-1 runtime conflict verifier transfer",
        "scope": "operationalization_feasibility_on_reused_frozen39",
        "efficacy_claim_allowed": False,
        "agent_behavior_changed": False,
        "qa_judge_run": False,
        "reference_model": reference_model,
        "runtime_model": runtime_model,
        "expected_item_count": len(expected_ids),
        "paired_item_count": len(paired_ids),
        "reference_invalid_item_ids": reference_invalid,
        "runtime_invalid_item_ids": runtime_invalid,
        "structural_checks": structural_checks,
        "structural_gate_passed": structural_gate_passed,
        "row_transfer": row_transfer,
        "binary_positive_definition": (
            "Reference true_contradiction is positive; reference "
            "false_contradiction and unclear are negative."
        ),
        "winner_level": {
            "reference": _without_flags(reference_winners),
            "runtime": _without_flags(runtime_winners),
            "runtime_vs_reference_flag_agreement": winner_flag_transfer,
        },
        "preregistered_thresholds": {
            "three_class_cohen_kappa_min": kappa_threshold,
            "candidate_present_collateral_rate_max": collateral_threshold,
            "strict_correct_collateral_rate_max": collateral_threshold,
            "false_candidate_newcombe_lower_must_exceed_zero": True,
            "false_candidate_fisher_two_sided_alpha": fisher_alpha,
        },
        "per_repeat_decision_criteria": per_repeat_criteria,
        "decision_checks": decision_checks,
        "transfer_feasible": transfer_feasible,
        "decision": (
            "RUNTIME_CONFLICT_VERIFIER_TRANSFER_FEASIBLE"
            if transfer_feasible
            else "RUNTIME_CONFLICT_VERIFIER_TRANSFER_NOT_FEASIBLE"
        ),
        "next_step": (
            "PREDECLARE_MECHANISM_TRANSPORT_COHORT"
            if transfer_feasible
            else "STOP_OR_REDESIGN_VERIFIER"
        ),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    row = report["row_transfer"]
    lines = [
        "# WP13 Runtime Conflict Verifier Transfer",
        "",
        f"Decision: **{report['decision']}**",
        "",
        (
            "This study tests operationalization feasibility on reused frozen39 "
            "items. It does not establish efficacy and does not modify Agent behavior."
        ),
        "",
        f"Structural gate passed: **{report['structural_gate_passed']}**",
        f"Reference/runtime models: `{report['reference_model']}` / `{report['runtime_model']}`",
        f"Paired blind items: {report['paired_item_count']}/{report['expected_item_count']}",
        "",
        "## Row Transfer",
        "",
        (
            f"Three-class agreement: {row['agreement_count']}/{row['paired_count']} "
            f"({_fmt(row['agreement_rate'])}, Wilson 95% "
            f"{_fmt_ci(row['agreement_wilson95'])}); Cohen's kappa "
            f"{_fmt(row['cohen_kappa'])}."
        ),
        (
            "`true_contradiction` transfer: precision "
            f"{_fmt(row['true_contradiction']['precision'])} "
            f"{_fmt_ci(row['true_contradiction']['precision_wilson95'])}; "
            f"recall {_fmt(row['true_contradiction']['recall'])} "
            f"{_fmt_ci(row['true_contradiction']['recall_wilson95'])}; "
            f"F1 {_fmt(row['true_contradiction']['f1'])}."
        ),
        "",
        "## Winner-Level Transfer",
        "",
        "| Model | Repeat | False winner | Candidate-present | Strict-correct | False-candidate delta (Newcombe/bootstrap; Fisher p) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_key in ("reference", "runtime"):
        model_rows = report["winner_level"][model_key]["per_repeat"]
        for repeat, metrics in model_rows.items():
            lines.append(
                f"| {model_key} | {repeat} | "
                f"{_rate_cell(metrics, 'false')} | "
                f"{_rate_cell(metrics, 'candidate_present')} | "
                f"{_rate_cell(metrics, 'strict_correct')} | "
                f"{_fmt(metrics['false_candidate_gap'])} "
                f"{_fmt_ci(metrics['false_candidate_gap_newcombe95'])} / "
                f"{_fmt_ci(metrics['false_candidate_gap_case_cluster_bootstrap']['ci95'])}; "
                f"{_fmt(metrics['false_candidate_fisher_exact_two_sided_p'])} |"
            )
    lines.extend(
        [
            "",
            "## Pre-Registered Decision",
            "",
        ]
    )
    for name, passed in report["decision_checks"].items():
        lines.append(f"- `{name}`: **{passed}**")
    lines.extend(
        [
            "",
            (
                "Repeated sidecar rows measure stability, not independent cohort "
                "replication. Newcombe and Fisher summaries are primary for the "
                "small boundary groups; case-cluster bootstrap is sensitivity only."
            ),
            "",
            f"Next step: **{report['next_step']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest_checks(
    key: Mapping[str, Any],
    reference: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    reference_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
) -> dict[str, bool]:
    expected = int(key.get("item_count", len(tuple(key.get("rows", ()) or ()))))
    return {
        "blind_bundle_digest_match": bool(
            reference_manifest.get("items_sha256")
            and reference_manifest.get("items_sha256")
            == runtime_manifest.get("items_sha256")
            and reference_manifest.get("key_sha256")
            == runtime_manifest.get("key_sha256")
        ),
        "reference_manifest_model_matches": reference_manifest.get("actual_model")
        == reference.get("actual_model"),
        "runtime_manifest_model_matches": runtime_manifest.get("actual_model")
        == runtime.get("actual_model"),
        "reference_one_item_per_call": reference_manifest.get("one_item_per_call")
        is True,
        "runtime_one_item_per_call": runtime_manifest.get("one_item_per_call") is True,
        "reference_no_shared_context": reference_manifest.get(
            "shared_conversation_context"
        )
        is False,
        "runtime_no_shared_context": runtime_manifest.get(
            "shared_conversation_context"
        )
        is False,
        "reference_no_prose_persisted": bool(
            reference_manifest.get("raw_response_persisted") is False
            and reference_manifest.get("prompt_persisted") is False
        ),
        "runtime_no_prose_persisted": bool(
            runtime_manifest.get("raw_response_persisted") is False
            and runtime_manifest.get("prompt_persisted") is False
        ),
        "reference_primary_count_matches": int(
            reference_manifest.get("primary_item_count", -1)
        )
        == expected,
        "runtime_primary_count_matches": int(
            runtime_manifest.get("primary_item_count", -1)
        )
        == expected,
        "runtime_primary_only": bool(
            runtime_manifest.get("primary_only") is True
            and int(runtime_manifest.get("reliability_item_count", -1)) == 0
            and int(runtime_manifest.get("task_count", -1)) == expected
        ),
    }


def _row_transfer_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    confusion = {
        reference: {
            runtime: sum(pair == (reference, runtime) for pair in pairs)
            for runtime in VALID_VERDICTS
        }
        for reference in VALID_VERDICTS
    }
    agreements = sum(left == right for left, right in pairs)
    true_positive = sum(
        reference == POSITIVE_VERDICT and runtime == POSITIVE_VERDICT
        for reference, runtime in pairs
    )
    false_positive = sum(
        reference != POSITIVE_VERDICT and runtime == POSITIVE_VERDICT
        for reference, runtime in pairs
    )
    false_negative = sum(
        reference == POSITIVE_VERDICT and runtime != POSITIVE_VERDICT
        for reference, runtime in pairs
    )
    true_negative = len(pairs) - true_positive - false_positive - false_negative
    precision_total = true_positive + false_positive
    recall_total = true_positive + false_negative
    precision = true_positive / precision_total if precision_total else None
    recall = true_positive / recall_total if recall_total else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "paired_count": len(pairs),
        "agreement_count": agreements,
        "agreement_rate": agreements / len(pairs) if pairs else None,
        "agreement_wilson95": wilson_interval(agreements, len(pairs)),
        "cohen_kappa": cohen_kappa(pairs, labels=VALID_VERDICTS),
        "confusion_matrix_reference_by_runtime": confusion,
        "true_contradiction": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": precision,
            "precision_wilson95": wilson_interval(true_positive, precision_total),
            "recall": recall,
            "recall_wilson95": wilson_interval(true_positive, recall_total),
            "f1": f1,
        },
    }


def _winner_report(
    key_rows: Mapping[str, Mapping[str, Any]],
    winner_values: Any,
    judgments: Mapping[str, str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    winners = {
        str(row.get("case_id", "") or ""): row
        for row in tuple(winner_values or ())
        if isinstance(row, Mapping) and str(row.get("case_id", "") or "")
    }
    false_ids = tuple(
        case_id
        for case_id, row in winners.items()
        if row.get("winner_class") == "false_winner"
    )
    candidate_ids = tuple(
        case_id
        for case_id, row in winners.items()
        if row.get("winner_class") == "candidate_present_winner"
    )
    strict_ids = tuple(
        case_id
        for case_id in candidate_ids
        if winners[case_id].get("strict_correct") is True
    )
    repeats = tuple(
        sorted(
            {
                str(row.get("repeat_label", "") or "")
                for row in key_rows.values()
                if str(row.get("repeat_label", "") or "")
            }
        )
    )
    per_repeat: dict[str, Any] = {}
    flags: dict[str, bool] = {}
    for repeat_index, repeat in enumerate(repeats):
        hit_ids = {
            str(row.get("case_id", "") or "")
            for item_id, row in key_rows.items()
            if row.get("repeat_label") == repeat
            and row.get("targets_selected_winner") is True
            and judgments.get(item_id) == POSITIVE_VERDICT
        }
        for case_id in (*false_ids, *candidate_ids):
            flags[f"{repeat}:{case_id}"] = case_id in hit_ids
        per_repeat[repeat] = _winner_metrics(
            hit_ids,
            false_ids=false_ids,
            candidate_ids=candidate_ids,
            strict_ids=strict_ids,
            samples=bootstrap_samples,
            seed=seed + repeat_index * 97,
        )
    return {
        "false_winner_count": len(false_ids),
        "candidate_present_winner_count": len(candidate_ids),
        "strict_correct_winner_count": len(strict_ids),
        "per_repeat": per_repeat,
        "flags": flags,
    }


def _winner_metrics(
    hit_ids: set[str],
    *,
    false_ids: Sequence[str],
    candidate_ids: Sequence[str],
    strict_ids: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    false_hits = sum(case_id in hit_ids for case_id in false_ids)
    candidate_hits = sum(case_id in hit_ids for case_id in candidate_ids)
    strict_hits = sum(case_id in hit_ids for case_id in strict_ids)
    false_rate = false_hits / len(false_ids) if false_ids else None
    candidate_rate = candidate_hits / len(candidate_ids) if candidate_ids else None
    strict_rate = strict_hits / len(strict_ids) if strict_ids else None
    return {
        **_group_rate("false", false_hits, len(false_ids)),
        **_group_rate("candidate_present", candidate_hits, len(candidate_ids)),
        **_group_rate("strict_correct", strict_hits, len(strict_ids)),
        "false_candidate_gap": _difference(false_rate, candidate_rate),
        "false_candidate_gap_newcombe95": newcombe_difference_interval(
            false_hits, len(false_ids), candidate_hits, len(candidate_ids)
        ),
        "false_candidate_fisher_exact_two_sided_p": fisher_exact_two_sided(
            false_hits, len(false_ids), candidate_hits, len(candidate_ids)
        ),
        "false_candidate_gap_case_cluster_bootstrap": _bootstrap_gap(
            hit_ids,
            false_ids=false_ids,
            comparison_ids=candidate_ids,
            samples=samples,
            seed=seed,
        ),
        "false_strict_gap": _difference(false_rate, strict_rate),
        "false_strict_gap_newcombe95": newcombe_difference_interval(
            false_hits, len(false_ids), strict_hits, len(strict_ids)
        ),
        "false_strict_fisher_exact_two_sided_p": fisher_exact_two_sided(
            false_hits, len(false_ids), strict_hits, len(strict_ids)
        ),
        "false_strict_gap_case_cluster_bootstrap": _bootstrap_gap(
            hit_ids,
            false_ids=false_ids,
            comparison_ids=strict_ids,
            samples=samples,
            seed=seed + 43,
        ),
    }


def _group_rate(prefix: str, hits: int, total: int) -> dict[str, Any]:
    return {
        f"{prefix}_hit_count": hits,
        f"{prefix}_count": total,
        f"{prefix}_rate": hits / total if total else None,
        f"{prefix}_rate_wilson95": wilson_interval(hits, total),
    }


def _bootstrap_gap(
    hit_ids: set[str],
    *,
    false_ids: Sequence[str],
    comparison_ids: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not false_ids or not comparison_ids or samples <= 0:
        return {"samples": samples, "valid_samples": 0, "ci95": [None, None]}
    rng = random.Random(seed)
    gaps = []
    for _ in range(samples):
        sampled_false = [rng.choice(false_ids) for _ in false_ids]
        sampled_comparison = [rng.choice(comparison_ids) for _ in comparison_ids]
        gaps.append(
            mean(case_id in hit_ids for case_id in sampled_false)
            - mean(case_id in hit_ids for case_id in sampled_comparison)
        )
    gaps.sort()
    return {
        "samples": samples,
        "valid_samples": len(gaps),
        "ci95": [_quantile(gaps, 0.025), _quantile(gaps, 0.975)],
    }


def _winner_repeat_criteria(
    metrics: Mapping[str, Any],
    *,
    collateral_threshold: float,
    fisher_alpha: float,
) -> dict[str, bool]:
    lower = tuple(metrics.get("false_candidate_gap_newcombe95", ()) or (None,))[0]
    fisher = metrics.get("false_candidate_fisher_exact_two_sided_p")
    candidate_rate = metrics.get("candidate_present_rate")
    strict_rate = metrics.get("strict_correct_rate")
    return {
        "false_candidate_separation_passed": bool(
            lower is not None
            and float(lower) > 0
            and fisher is not None
            and float(fisher) <= fisher_alpha
        ),
        "correct_winner_collateral_passed": bool(
            candidate_rate is not None
            and float(candidate_rate) <= collateral_threshold
            and strict_rate is not None
            and float(strict_rate) <= collateral_threshold
        ),
    }


def _winner_flag_transfer(
    reference: Mapping[str, bool], runtime: Mapping[str, bool]
) -> dict[str, Any]:
    ids = tuple(sorted(set(reference) & set(runtime)))
    pairs = [
        ("hit" if reference[item_id] else "clear", "hit" if runtime[item_id] else "clear")
        for item_id in ids
    ]
    agreements = sum(left == right for left, right in pairs)
    return {
        "paired_case_repeat_count": len(pairs),
        "agreement_count": agreements,
        "agreement_rate": agreements / len(pairs) if pairs else None,
        "agreement_wilson95": wilson_interval(agreements, len(pairs)),
        "cohen_kappa": cohen_kappa(pairs, labels=("clear", "hit")),
    }


def _key_rows(key: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {
        str(row.get("audit_item_id", "") or ""): row
        for row in tuple(key.get("rows", ()) or ())
        if isinstance(row, Mapping) and str(row.get("audit_item_id", "") or "")
    }
    if len(rows) != int(key.get("item_count", len(rows))):
        raise ValueError("audit key item count mismatch")
    return rows


def _judgment_rows(payload: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    rows: dict[str, str] = {}
    invalid: list[str] = []
    for row in tuple(payload.get("judgments", ()) or ()):
        if not isinstance(row, Mapping):
            continue
        item_id = str(row.get("audit_item_id", "") or "")
        verdict = str(row.get("verdict", "") or "").strip().casefold()
        if not item_id or verdict not in VALID_VERDICTS or item_id in rows:
            invalid.append(item_id or "<missing>")
            continue
        rows[item_id] = verdict
    return rows, sorted(invalid)


def _without_flags(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "flags"}


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


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


def _rate_cell(metrics: Mapping[str, Any], prefix: str) -> str:
    return (
        f"{metrics[f'{prefix}_hit_count']}/{metrics[f'{prefix}_count']} "
        f"{_fmt_ci(metrics[f'{prefix}_rate_wilson95'])}"
    )


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _fmt_ci(values: Sequence[Any]) -> str:
    items = tuple(values or ())
    if len(items) != 2 or any(value is None for value in items):
        return "[NA, NA]"
    return f"[{_fmt(items[0])}, {_fmt(items[1])}]"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-json", required=True)
    parser.add_argument("--reference-judgments-json", required=True)
    parser.add_argument("--runtime-judgments-json", required=True)
    parser.add_argument("--reference-run-manifest", required=True)
    parser.add_argument("--runtime-run-manifest", required=True)
    parser.add_argument("--expected-reference-model", required=True)
    parser.add_argument("--expected-runtime-model", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--kappa-threshold", type=float, default=0.60)
    parser.add_argument("--collateral-threshold", type=float, default=0.20)
    parser.add_argument("--fisher-alpha", type=float, default=0.10)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = analyze_transfer(
        _read_json(Path(args.key_json)),
        _read_json(Path(args.reference_judgments_json)),
        _read_json(Path(args.runtime_judgments_json)),
        reference_manifest=_read_json(Path(args.reference_run_manifest)),
        runtime_manifest=_read_json(Path(args.runtime_run_manifest)),
        expected_reference_model=args.expected_reference_model,
        expected_runtime_model=args.expected_runtime_model,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        kappa_threshold=args.kappa_threshold,
        collateral_threshold=args.collateral_threshold,
        fisher_alpha=args.fisher_alpha,
    )
    _write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"CONFLICT_VERIFIER_TRANSFER_DONE decision={report['decision']} "
        f"paired={report['paired_item_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
