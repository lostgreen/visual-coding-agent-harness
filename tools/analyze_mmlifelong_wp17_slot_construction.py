#!/usr/bin/env python3
"""Analyze the frozen WP17 slot-construction development endpoints."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from vcah.occurrence_negative_sidecar import file_sha256

ARMS = ("e1c0", "e1c1", "e1c2")
ANALYSIS_PROTOCOL_CONTRACT = "WP17-slot-construction-endpoint-analysis-v1"
REPRESENTATION_METRICS = (
    "anchor_representation_coverage",
    "canonical_entity_coverage",
    "relation_state_coverage",
    "occurrence_coverage",
    "ordinal_accuracy",
)


def run(args: argparse.Namespace) -> Path:
    analysis_protocol_path = Path(args.analysis_protocol)
    if file_sha256(analysis_protocol_path) != str(args.expected_analysis_protocol_sha256):
        raise ValueError("WP17 endpoint analysis protocol SHA mismatch")
    analysis_protocol = _read_json(analysis_protocol_path)
    if analysis_protocol.get("contract") != ANALYSIS_PROTOCOL_CONTRACT:
        raise ValueError("WP17 endpoint analysis protocol contract mismatch")
    effect_policy = dict(analysis_protocol["effects"])
    if (
        int(args.bootstrap_samples) != int(effect_policy["bootstrap_samples"])
        or int(args.seed) != int(effect_policy["bootstrap_seed"])
    ):
        raise ValueError("WP17 endpoint bootstrap settings are frozen")

    construction_protocol_path = Path(args.construction_protocol)
    if file_sha256(construction_protocol_path) != str(
        args.expected_construction_protocol_sha256
    ):
        raise ValueError("WP17 construction protocol SHA mismatch")
    if analysis_protocol.get("construction_protocol_sha256") != file_sha256(
        construction_protocol_path
    ):
        raise ValueError("WP17 endpoint/construction protocol mismatch")
    construction_protocol = _read_json(construction_protocol_path)
    audit = _read_json(Path(args.structural_audit))
    if audit.get("structural_gate_passed") is not True:
        raise RuntimeError("WP17 endpoint analysis requires a passing structural audit")
    if audit.get("endpoint_values_evaluated") is not False:
        raise RuntimeError("WP17 structural audit already accessed endpoint values")

    run_root = Path(args.run_root)
    run_manifest = _read_json(run_root / "run_manifest.json")
    summary = _read_json(run_root / "run_summary.json")
    expected_results = int(run_manifest["expected_result_count"])
    if (
        run_manifest.get("protocol_manifest_sha256")
        != file_sha256(construction_protocol_path)
        or summary.get("complete") is not True
        or int(summary.get("successes", 0)) != expected_results
    ):
        raise RuntimeError("WP17 endpoint analysis run is incomplete or mismatched")

    segments = tuple(dict(row) for row in construction_protocol["segments"])
    segment_by_id = {str(row["segment_id"]): row for row in segments}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    result_paths: dict[tuple[str, str], Path] = {}
    for segment in segments:
        segment_id = str(segment["segment_id"])
        for arm in ARMS:
            path = run_root / "segments" / segment_id / f"{arm}.json"
            row = _read_json(path)
            if row.get("status") != "success":
                raise RuntimeError("WP17 endpoint analysis encountered a non-success row")
            rows[(segment_id, arm)] = row
            result_paths[(segment_id, arm)] = path

    case_scores = []
    annotations = dict(analysis_protocol["cases"])
    for case_id, raw_annotation in sorted(annotations.items()):
        annotation = dict(raw_annotation)
        anchor_segment_ids = tuple(
            str(segment["segment_id"])
            for segment in segments
            if _overlaps_any(segment, annotation["anchor_intervals"])
        )
        if not anchor_segment_ids:
            raise ValueError(f"WP17 annotation has no construction segment: {case_id}")
        for arm in ARMS:
            selected = [rows[(segment_id, arm)] for segment_id in anchor_segment_ids]
            trusted = [
                row
                for row in selected
                if row.get("ser_endpoint_eligible") is True
                and row.get("slot_transaction_abstained") is not True
            ]
            artifact = _normalize_text(
                " ".join(
                    _flatten_text(row["model_output"]["structured_event_record"])
                    for row in trusted
                )
            )
            entity_match = _first_match(artifact, annotation["entity_terms"])
            event_match = _first_match(artifact, annotation["event_terms"])
            state_match = _first_match(artifact, annotation["state_terms"])
            occurrence_match = _first_match(
                artifact, annotation.get("occurrence_terms", ())
            )
            ordinal_match = _first_match(artifact, annotation.get("ordinal_terms", ()))
            case_scores.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "anchor_segment_count": len(anchor_segment_ids),
                    "ser_endpoint_ineligible_anchor_rows": len(selected) - len(trusted),
                    "anchor_representation_coverage": int(
                        entity_match is not None and event_match is not None
                    ),
                    "canonical_entity_coverage": int(entity_match is not None),
                    "event_coverage": int(event_match is not None),
                    "relation_state_coverage": int(state_match is not None),
                    "occurrence_coverage": (
                        int(occurrence_match is not None)
                        if annotation.get("occurrence_terms")
                        else None
                    ),
                    "ordinal_accuracy": (
                        int(ordinal_match is not None)
                        if annotation.get("ordinal_terms")
                        else None
                    ),
                    "matched_annotation_terms": {
                        "entity": entity_match,
                        "event": event_match,
                        "state": state_match,
                        "occurrence": occurrence_match,
                        "ordinal": ordinal_match,
                    },
                    "annotation_timing": annotation["annotation_timing"],
                }
            )

    arm_metrics = {
        arm: {
            metric: _rate_summary(
                [
                    row[metric]
                    for row in case_scores
                    if row["arm"] == arm and row.get(metric) is not None
                ]
            )
            for metric in REPRESENTATION_METRICS
        }
        for arm in ARMS
    }
    paired = {
        "e1c2_minus_e1c1": _paired_metrics(
            case_scores,
            treatment="e1c2",
            control="e1c1",
            samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        ),
        "e1c1_minus_e1c0": _paired_metrics(
            case_scores,
            treatment="e1c1",
            control="e1c0",
            samples=int(args.bootstrap_samples),
            seed=int(args.seed) + 101,
        ),
    }

    context_cost = {
        arm: _distribution(
            [int(row.get("history_token_count", 0) or 0) for (sid, a), row in rows.items() if a == arm]
        )
        for arm in ARMS
    }
    per_arm_cost = {}
    for arm in ARMS:
        arm_rows = [row for (segment_id, value), row in rows.items() if value == arm]
        arm_paths = [path for (segment_id, value), path in result_paths.items() if value == arm]
        per_arm_cost[arm] = {
            "final_artifact_model_calls": sum(_row_model_calls(row) for row in arm_rows),
            "validation_retry_count": sum(
                int(row.get("validation_retry_count", 0) or 0) for row in arm_rows
            ),
            "transaction_abstentions": sum(
                row.get("slot_transaction_abstained") is True for row in arm_rows
            ),
            "result_storage_bytes": sum(path.stat().st_size for path in arm_paths),
        }

    e1c2_rows = [row for (segment_id, arm), row in rows.items() if arm == "e1c2"]
    state_write_ops = []
    for row in e1c2_rows:
        for operation in tuple(row.get("model_output", {}).get("slot_operations", ()) or ()):
            if operation.get("operation") in {"write", "update", "close"}:
                state_write_ops.append(dict(operation))
    unsupported_writes = sum(not tuple(row.get("observation_ids", ()) or ()) for row in state_write_ops)
    provenance = {
        "mechanical_reference_validity": {
            arm: 1.0 for arm in ARMS
        },
        "source": "passing independent replay and packet-local evidence audit",
        "unsupported_state_write_reference_rate": (
            unsupported_writes / len(state_write_ops) if state_write_ops else 0.0
        ),
        "unsupported_state_write_count": unsupported_writes,
        "state_write_count": len(state_write_ops),
        "visual_provenance_blind_sample": "pending",
    }
    lifecycle_events = [
        event for row in e1c2_rows for event in tuple(row.get("lifecycle_events", ()) or ())
    ]
    slot_churn = {
        "events_per_segment": len(lifecycle_events) / len(e1c2_rows),
        "mutation_events_per_segment": sum(
            event.get("operation") in {"write", "update", "close", "archive", "evict"}
            for event in lifecycle_events
        )
        / len(e1c2_rows),
        "implicit_retain_events": sum(
            event.get("operation") == "implicit_retain" for event in lifecycle_events
        ),
        "transaction_abstain_events": sum(
            event.get("operation") == "transaction_abstain" for event in lifecycle_events
        ),
    }

    primary = paired["e1c2_minus_e1c1"]
    decision_metrics = (
        "anchor_representation_coverage",
        "canonical_entity_coverage",
        "relation_state_coverage",
    )
    positive = [
        metric
        for metric in decision_metrics
        if primary[metric]["paired_delta"] > 0
    ]
    provenance_ok = unsupported_writes == 0
    token_gate = all(
        int(row.get("history_token_count", 0) or 0) <= 600 for row in rows.values()
    )
    raw_go = len(positive) >= 2 and provenance_ok and token_gate
    decision = (
        "PROVISIONAL_GO_PENDING_VISUAL_PROVENANCE"
        if raw_go
        else "NO_GO_UNDER_DEVELOPMENT_ENDPOINTS"
    )

    report = {
        "schema_version": "MMLifelongWP17SlotConstructionEndpointReportV1",
        "contract": "WP17-slot-construction-endpoint-report-v1",
        "decision": decision,
        "decision_is_development_only": True,
        "unbiased_publication_claim_allowed": False,
        "construction_protocol_sha256": file_sha256(construction_protocol_path),
        "analysis_protocol_sha256": file_sha256(analysis_protocol_path),
        "structural_audit_decision": audit.get("decision"),
        "counts": {
            "cases": len(annotations),
            "frozen_pre_outcome_cases": sum(
                row["annotation_timing"] == "frozen_before_construction_outcomes"
                for row in annotations.values()
            ),
            "post_hoc_cases": sum(
                row["annotation_timing"] != "frozen_before_construction_outcomes"
                for row in annotations.values()
            ),
            "results": len(rows),
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.seed),
        },
        "arm_metrics": arm_metrics,
        "paired_effects": paired,
        "primary_positive_endpoints": positive,
        "context_token_distribution": context_cost,
        "provenance": provenance,
        "slot_churn": slot_churn,
        "cost": {
            "per_arm_final_artifact": per_arm_cost,
            "continuation_accounting": summary.get("continuation", {}),
        },
        "case_scores": case_scores,
        "gates": {
            "structural_audit_passed": True,
            "all_results_success": len(rows) == expected_results,
            "context_within_600": token_gate,
            "mechanical_provenance_reference_valid": True,
            "unsupported_state_write_references_zero": provenance_ok,
            "visual_provenance_complete": False,
            "endpoint_values_were_not_structural_gates": True,
        },
        "warnings": [
            "The 11-case local timeline is underpowered and development-only.",
            "Case 0115 occurrence/ordinal endpoints are post-hoc and do not drive GO/NO-GO.",
            "The lexical annotation matcher is conservative and may miss valid paraphrases.",
            "A final GO claim remains blocked until the frozen blind visual-provenance sample is complete.",
        ],
        "endpoint_values_evaluated": True,
        "model_calls": 0,
        "day_test140_accessed": False,
        "week_accessed": False,
    }
    out_root = Path(args.out_root)
    report_path = out_root / "wp17_slot_construction_endpoint_report.json"
    markdown_path = out_root / "wp17_slot_construction_endpoint_report.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17 endpoint report output already exists")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(
        "WP17_SLOT_ENDPOINT_ANALYSIS_DONE "
        f"decision={decision} cases={len(annotations)} results={len(rows)} "
        f"positive={len(positive)} model_calls=0",
        flush=True,
    )
    return report_path


def _overlaps_any(segment: Mapping[str, Any], intervals: Sequence[Sequence[float]]) -> bool:
    start = float(segment["virtual_start_sec"])
    end = float(segment["virtual_end_sec"])
    return any(start < float(right) and end > float(left) for left, right in intervals)


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            parts.extend((str(key), _flatten_text(item)))
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value) if value is not None else ""


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _first_match(artifact: str, terms: Sequence[str]) -> str | None:
    for term in terms:
        normalized = _normalize_text(term)
        if normalized and normalized in artifact:
            return str(term)
    return None


def _rate_summary(values: Sequence[int]) -> dict[str, Any]:
    total = len(values)
    count = sum(int(value) for value in values)
    return {"count": count, "denominator": total, "rate": count / total if total else None}


def _paired_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_key = {(str(row["case_id"]), str(row["arm"])): row for row in rows}
    case_ids = sorted({str(row["case_id"]) for row in rows})
    report = {}
    for offset, metric in enumerate(REPRESENTATION_METRICS):
        differences = []
        for case_id in case_ids:
            left = by_key[(case_id, treatment)].get(metric)
            right = by_key[(case_id, control)].get(metric)
            if left is not None and right is not None:
                differences.append(int(left) - int(right))
        low, high = _bootstrap_mean_ci(
            differences, samples=samples, seed=seed + offset * 1009
        )
        report[metric] = {
            "treatment": treatment,
            "control": control,
            "paired_cases": len(differences),
            "paired_delta": mean(differences) if differences else 0.0,
            "bootstrap_ci95": [low, high],
            "wins_ties_losses": {
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            },
        }
    return report


def _bootstrap_mean_ci(
    values: Sequence[int], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = quantile * (len(values) - 1)
    left = int(position)
    right = min(len(values) - 1, left + 1)
    weight = position - left
    return float(values[left] * (1.0 - weight) + values[right] * weight)


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "mean": mean(ordered) if ordered else 0.0,
        "p50": _percentile(ordered, 0.5) if ordered else 0.0,
        "p95": _percentile(ordered, 0.95) if ordered else 0.0,
        "max": max(ordered) if ordered else 0,
    }


def _row_model_calls(row: Mapping[str, Any]) -> int:
    calls = 0
    for attempt in tuple(row.get("attempts", ()) or ()):
        calls += 1
        calls += int(
            dict(attempt.get("response_metadata", {}) or {}).get(
                "truncation_retry_count", 0
            )
            or 0
        )
    return calls


def _render_markdown(report: Mapping[str, Any]) -> str:
    arm_metrics = dict(report["arm_metrics"])
    primary = dict(report["paired_effects"]["e1c2_minus_e1c1"])
    lines = [
        "# MM-Lifelong WP17-3 Slot Construction Endpoint Report",
        "",
        f"- Decision: `{report['decision']}`",
        "- Primary effect: `E1C2 - E1C1`",
        "- Scope: 11-case local development timeline; not an unbiased publication claim.",
        "- Case 0115 terms are explicitly post-hoc.",
        "",
        "| Endpoint | E1C0 | E1C1 | E1C2 | E1C2-E1C1 (95% CI) | W/T/L |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "anchor_representation_coverage": "ARC",
        "canonical_entity_coverage": "CEC",
        "relation_state_coverage": "RSC",
        "occurrence_coverage": "Occurrence",
        "ordinal_accuracy": "Ordinal",
    }
    for metric in REPRESENTATION_METRICS:
        effect = primary[metric]
        wtl = effect["wins_ties_losses"]
        arm_values = []
        for arm in ARMS:
            row = arm_metrics[arm][metric]
            arm_values.append(
                "n/a" if row["rate"] is None else f"{100 * row['rate']:.2f}% ({row['count']}/{row['denominator']})"
            )
        lines.append(
            f"| {labels[metric]} | {arm_values[0]} | {arm_values[1]} | {arm_values[2]} | "
            f"{100 * effect['paired_delta']:+.2f}pp "
            f"[{100 * effect['bootstrap_ci95'][0]:+.2f}, {100 * effect['bootstrap_ci95'][1]:+.2f}] | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Transaction abstentions are scored as endpoint failures, not excluded.",
            "- Mechanical evidence-reference provenance passed through the structural audit.",
            "- Final GO remains pending until the blind visual-provenance sample is complete.",
            "- Lexical matching is conservative and may miss valid paraphrases.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-protocol", required=True)
    parser.add_argument("--expected-analysis-protocol-sha256", required=True)
    parser.add_argument("--construction-protocol", required=True)
    parser.add_argument("--expected-construction-protocol-sha256", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--structural-audit", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
