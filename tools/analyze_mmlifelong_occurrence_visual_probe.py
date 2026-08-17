#!/usr/bin/env python3
"""Analyze the blind WP14 visual discriminability probe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from vcah.occurrence_evaluation_stats import (
    newcombe_difference_interval,
    wilson_interval,
)
from vcah.occurrence_negative_sidecar import file_sha256, stable_digest
from vcah.occurrence_visual_probe import (
    VISUAL_PROBE_CONTRACT,
    VISUAL_PROBE_PAIR_KINDS,
    VISUAL_PROBE_VERDICTS,
    audit_visual_probe_manifest,
)


FORBIDDEN_RESULT_KEYS = frozenset(
    {"api_key", "authorization", "bridge_url", "messages", "prompt", "raw_response", "token"}
)


def build_report(
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    probe_root: Path,
    run_manifest: Mapping[str, Any],
    expected_model: str,
    expected_items: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    provenance = audit_visual_probe_manifest(manifest, root=probe_root)
    items = {
        str(item.get("item_id", "") or ""): dict(item)
        for case in tuple(manifest.get("cases", ()) or ())
        if isinstance(case, Mapping) and bool(case.get("eligible"))
        for item in tuple(case.get("items", ()) or ())
        if isinstance(item, Mapping)
    }
    result_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_result_ids: list[str] = []
    for row in results:
        item_id = str(row.get("item_id", "") or "")
        if item_id in result_by_id:
            duplicate_result_ids.append(item_id)
        result_by_id[item_id] = row
    actual_model = str(run_manifest.get("actual_model", "") or "")
    forbidden_paths = sorted(
        {
            path
            for row in results
            for path in _forbidden_paths(row)
        }
    )
    structural_checks = {
        "provenance_structural_gate_passed": bool(
            provenance.get("structural_gate_passed")
        ),
        "item_count_matches_expected": len(items) == expected_items,
        "result_item_set_exact": set(result_by_id) == set(items),
        "result_ids_unique": not duplicate_result_ids,
        "all_results_successful": all(
            row.get("status") == "success" for row in result_by_id.values()
        ),
        "all_verdicts_valid": all(
            row.get("verdict") in VISUAL_PROBE_VERDICTS
            for row in result_by_id.values()
        ),
        "item_digests_match": all(
            result_by_id[item_id].get("item_digest") == stable_digest(item)
            for item_id, item in items.items()
            if item_id in result_by_id
        ),
        "observation_bindings_match": all(
            str(result_by_id[item_id].get("visual_observation_id", "") or "")
            == str(item.get("visual_observation_id", "") or "")
            for item_id, item in items.items()
            if item_id in result_by_id
        ),
        "actual_model_matches": actual_model == expected_model,
        "all_item_models_match": all(
            str(row.get("actual_model", "") or "") == expected_model
            for row in result_by_id.values()
        ),
        "probe_manifest_digest_matches": str(
            run_manifest.get("probe_manifest_sha256", "") or ""
        )
        == file_sha256(Path(probe_root) / "probe_manifest.json"),
        "no_forbidden_persisted_fields": not forbidden_paths,
        "prompt_and_raw_not_persisted": all(
            row.get("prompt_persisted") is False
            and row.get("raw_response_persisted") is False
            for row in result_by_id.values()
        ),
        "agent_behavior_unchanged": run_manifest.get("agent_behavior_changed")
        is False,
        "endpoint_values_not_structural_gates": True,
    }
    structural_gate_passed = all(structural_checks.values())
    labeled_rows = []
    for item_id, item in items.items():
        result = result_by_id.get(item_id)
        if result is None or result.get("verdict") not in VISUAL_PROBE_VERDICTS:
            continue
        labeled_rows.append(
            {
                "item_id": item_id,
                "case_id": str(item.get("case_id", "") or ""),
                "pair_kind": str(item.get("pair_kind", "") or ""),
                "constraint_type": str(item.get("constraint_type", "") or ""),
                "verdict": str(result.get("verdict", "") or ""),
            }
        )
    by_kind = {
        kind: _verdict_rates(
            tuple(row for row in labeled_rows if row["pair_kind"] == kind)
        )
        for kind in VISUAL_PROBE_PAIR_KINDS
    }
    matched = by_kind["matched"]
    mismatched = by_kind["mismatched"]
    null = by_kind["null"]
    delta = _difference(matched["supported_rate"], mismatched["supported_rate"])
    delta_newcombe = newcombe_difference_interval(
        matched["supported_count"],
        matched["count"],
        mismatched["supported_count"],
        mismatched["count"],
    )
    bootstrap = _case_cluster_bootstrap(
        labeled_rows,
        left_kind="matched",
        right_kind="mismatched",
        samples=bootstrap_samples,
        seed=seed,
    )
    matched_null_bootstrap = _case_cluster_bootstrap(
        labeled_rows,
        left_kind="matched",
        right_kind="null",
        samples=bootstrap_samples,
        seed=seed + 503,
    )
    constraint_types = sorted(
        {row["constraint_type"] for row in labeled_rows if row["constraint_type"]}
    )
    by_constraint_type = {
        constraint_type: {
            kind: _verdict_rates(
                tuple(
                    row
                    for row in labeled_rows
                    if row["constraint_type"] == constraint_type
                    and row["pair_kind"] == kind
                )
            )
            for kind in VISUAL_PROBE_PAIR_KINDS
        }
        for constraint_type in constraint_types
    }
    thresholds = dict(manifest.get("engineering_thresholds", {}) or {})
    gap_min = float(
        thresholds.get("matched_minus_mismatched_support_rate_min", 0.40)
    )
    null_max = float(thresholds.get("null_support_rate_max", 0.15))
    decision_checks = {
        "matched_minus_mismatched_support_rate_at_least_threshold": bool(
            delta is not None and delta >= gap_min
        ),
        "null_support_rate_at_most_threshold": bool(
            null["supported_rate"] is not None
            and null["supported_rate"] <= null_max
        ),
    }
    verifier_discriminability_passed = bool(
        structural_gate_passed and all(decision_checks.values())
    )
    return {
        "schema_version": "MMLifelongVisualDiscriminabilityProbeReportV1",
        "contract": VISUAL_PROBE_CONTRACT,
        "study": "WP14-1 provenance and WP14-2 blind visual discriminability",
        "scope": "frozen39 mechanism-development exploratory diagnostic",
        "efficacy_claim_allowed": False,
        "agent_behavior_changed": False,
        "qa_judge_run": False,
        "eligible_case_count": len({row["case_id"] for row in labeled_rows}),
        "item_count": len(labeled_rows),
        "actual_model": actual_model,
        "provenance": provenance,
        "structural_checks": structural_checks,
        "structural_gate_passed": structural_gate_passed,
        "forbidden_result_paths": forbidden_paths,
        "verdict_rates_by_pair_kind": by_kind,
        "primary": {
            "matched_supported_rate": matched["supported_rate"],
            "mismatched_supported_rate": mismatched["supported_rate"],
            "null_supported_rate": null["supported_rate"],
            "matched_minus_mismatched_supported_rate": delta,
            "matched_minus_mismatched_newcombe95": delta_newcombe,
            "matched_minus_mismatched_case_cluster_bootstrap": bootstrap,
            "matched_minus_null_case_cluster_bootstrap": matched_null_bootstrap,
        },
        "by_constraint_type": by_constraint_type,
        "engineering_thresholds": thresholds,
        "decision_checks": decision_checks,
        "verifier_discriminability_passed": verifier_discriminability_passed,
        "decision": (
            "PROCEED_TO_WP14_3_SHADOW"
            if verifier_discriminability_passed
            else "STOP_WP14_BEHAVIORAL_INTEGRATION"
        ),
        "next_step": (
            "WP14_3_SHADOW_PRECOMMIT_VERIFICATION"
            if verifier_discriminability_passed
            else "STOP_OR_REDESIGN_VISUAL_VERIFIER"
        ),
        "underpowered": True,
        "cumulative_design_overfitting_warning": True,
        "verified_correct_zero_is_warning_not_validity_gate": True,
        "day_test140_accessed": False,
        "week_accessed": False,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    primary = report["primary"]
    by_kind = report["verdict_rates_by_pair_kind"]
    lines = [
        "# MM-Lifelong WP14 Visual Discriminability Probe",
        "",
        f"Decision: **{report['decision']}**",
        "",
        (
            "This is a frozen39 mechanism-development diagnostic. It does not "
            "change Agent behavior, run QA judging, or establish efficacy."
        ),
        "",
        f"Structural gate passed: **{report['structural_gate_passed']}**",
        f"Runtime visual verifier: `{report['actual_model']}`",
        f"Eligible cases/items: {report['eligible_case_count']} / {report['item_count']}",
        "",
        "## Primary Discriminability",
        "",
        "| Pair kind | Supported | Contradicted | Unknown |",
        "|---|---:|---:|---:|",
    ]
    for kind in VISUAL_PROBE_PAIR_KINDS:
        row = by_kind[kind]
        lines.append(
            f"| {kind} | {_rate_count(row, 'supported')} | "
            f"{_rate_count(row, 'contradicted')} | {_rate_count(row, 'unknown')} |"
        )
    boot = primary["matched_minus_mismatched_case_cluster_bootstrap"]
    lines.extend(
        [
            "",
            (
                "Matched minus mismatched supported-rate gap: "
                f"**{_fmt(primary['matched_minus_mismatched_supported_rate'])}**; "
                f"Newcombe 95% {_fmt_ci(primary['matched_minus_mismatched_newcombe95'])}; "
                f"case-cluster bootstrap 95% {_fmt_ci(boot['ci95'])}."
            ),
            (
                "Null supported rate: "
                f"**{_fmt(primary['null_supported_rate'])}**."
            ),
            "",
            "Engineering thresholds were frozen before probe outcomes: gap >= 0.40 "
            "and null support <= 0.15. They are engineering Go/Stop criteria, not "
            "paper-level significance claims.",
            "",
            "## Structural Provenance",
            "",
        ]
    )
    counts = report["provenance"]["counts"]
    lines.extend(
        [
            f"- Observation provenance: {counts['item_count']}/{counts['expected_item_count']}",
            f"- Unbound observations: {counts['unbound_observation_count']}",
            f"- Wrong-occurrence bindings: {counts['wrong_occurrence_binding_count']}",
            f"- Invalid constraint bindings: {counts['invalid_constraint_binding_count']}",
            f"- Incomplete frame provenance: {counts['incomplete_frame_provenance_count']}",
            f"- Silent locator drops: {counts['silent_locator_drop_count']}",
            "",
            "## Interpretation Boundary",
            "",
            (
                "Only identity/event/action/temporal/order/location constraints were "
                "eligible. Gold labels, pair labels, R5 winner state and margin, "
                "questions, options and answers were hidden from the verifier."
            ),
            "",
            (
                "frozen39 is underpowered and has been used repeatedly for mechanism "
                "development. Day-test140 and Week remain sealed until the final method "
                "is frozen."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _verdict_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(str(row.get("verdict", "") or "") for row in rows)
    payload: dict[str, Any] = {"count": total}
    for verdict in VISUAL_PROBE_VERDICTS:
        count = counts[verdict]
        payload[f"{verdict}_count"] = count
        payload[f"{verdict}_rate"] = count / total if total else None
        payload[f"{verdict}_wilson95"] = wilson_interval(count, total)
    return payload


def _case_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_kind: str,
    right_kind: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id", "") or "")].append(row)
    case_ids = tuple(sorted(case_id for case_id in by_case if case_id))
    if not case_ids or samples <= 0:
        return {"samples": 0, "ci95": [None, None], "positive_probability": None}
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        expanded = [row for case_id in sampled for row in by_case[case_id]]
        left = [row for row in expanded if row.get("pair_kind") == left_kind]
        right = [row for row in expanded if row.get("pair_kind") == right_kind]
        left_rate = mean(row.get("verdict") == "supported" for row in left)
        right_rate = mean(row.get("verdict") == "supported" for row in right)
        values.append(left_rate - right_rate)
    values.sort()
    return {
        "samples": samples,
        "ci95": [_quantile(values, 0.025), _quantile(values, 0.975)],
        "positive_probability": sum(value > 0 for value in values) / len(values),
    }


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * max(0.0, min(1.0, probability))
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _difference(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left) - float(right)


def _forbidden_paths(value: Any, *, prefix: str = "$") -> list[str]:
    paths = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if str(key).casefold() in FORBIDDEN_RESULT_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, prefix=f"{prefix}[{index}]"))
    return paths


def _rate_count(row: Mapping[str, Any], verdict: str) -> str:
    return f"{row[f'{verdict}_count']}/{row['count']} ({_fmt(row[f'{verdict}_rate'])})"


def _fmt(value: Any) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{float(value):.4f}"


def _fmt_ci(value: Any) -> str:
    rows = tuple(value or ()) if isinstance(value, Sequence) else ()
    return "NA" if len(rows) != 2 else f"[{_fmt(rows[0])}, {_fmt(rows[1])}]"


def _read_results(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(_read_json(item) for item in sorted(Path(path).glob("items/*.json")))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    probe_root = Path(args.probe_root)
    result_root = Path(args.result_root)
    report = build_report(
        _read_json(probe_root / "probe_manifest.json"),
        _read_results(result_root),
        probe_root=probe_root,
        run_manifest=_read_json(result_root / "run_manifest.json"),
        expected_model=str(args.expected_model),
        expected_items=int(args.expected_items),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    _write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"VISUAL_PROBE_ANALYSIS_DONE decision={report['decision']} "
        f"structural={report['structural_gate_passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
