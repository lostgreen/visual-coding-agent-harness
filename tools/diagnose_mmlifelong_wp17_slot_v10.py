#!/usr/bin/env python3
"""Zero-model WP17 v10 forensics over completed slot-construction artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from vcah.wp17_slot_memory import budget_token_count


ARMS = ("e1c0", "e1c1", "e1c2")
SER_FIELDS = ("entities", "events", "state_changes", "relations", "occurrence_refs")
REPORT_CONTRACT = "WP17-slot-v10-zero-model-forensics-v1"


def build_report(
    *,
    run_root: Path,
    construction_protocol: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any],
    source_commit: str,
    dense_evidence_path: Path | None = None,
    canonical_terms: Sequence[str] = (),
    canonical_before_sec: float | None = None,
) -> dict[str, Any]:
    manifest = _read_json(run_root / "run_manifest.json")
    summary = _read_json(run_root / "run_summary.json")
    segments = tuple(dict(row) for row in construction_protocol["segments"])
    expected = len(segments) * len(ARMS)
    if summary.get("complete") is not True or int(summary.get("successes", 0)) != expected:
        raise RuntimeError("WP17 v10 forensics requires a complete matched run")

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for segment in segments:
        segment_id = str(segment["segment_id"])
        for arm in ARMS:
            row = _read_json(run_root / "segments" / segment_id / f"{arm}.json")
            if row.get("status") != "success":
                raise RuntimeError("WP17 v10 forensics encountered a non-success row")
            rows[(segment_id, arm)] = row

    ser_stats = _ser_statistics(rows)
    lifecycle = _lifecycle_statistics(rows)
    fingerprints = _failure_fingerprints(rows)
    coverage = {
        scope: _coverage_scope(
            rows=rows,
            segments=segments,
            annotations=dict(analysis_protocol["cases"]),
            committed_only=scope == "committed_memory",
        )
        for scope in ("raw_ser", "committed_memory")
    }
    canonical_evidence = _canonical_evidence_summary(
        dense_evidence_path,
        terms=canonical_terms,
        before_sec=canonical_before_sec,
    )
    e1c2_rows = [row for (segment_id, arm), row in rows.items() if arm == "e1c2"]
    abstentions = sum(row.get("slot_transaction_abstained") is True for row in e1c2_rows)

    return {
        "schema_version": "MMLifelongWP17SlotV10ForensicsV1",
        "contract": REPORT_CONTRACT,
        "source_commit": str(source_commit),
        "construction_source_commit": manifest.get("source_commit"),
        "counts": {
            "segments": len(segments),
            "results": len(rows),
            "e1c2_transaction_abstentions": abstentions,
            "e1c2_abstain_rate": abstentions / len(e1c2_rows) if e1c2_rows else None,
            "e1c2_illegal_operation_attempts": sum(
                int(row.get("illegal_operation_count", 0) or 0) for row in e1c2_rows
            ),
        },
        "ser_structure": ser_stats,
        "lifecycle": lifecycle,
        "failure_fingerprints": fingerprints,
        "strict_lexical_exploratory_coverage": coverage,
        "canonical_evidence_diagnostic": canonical_evidence,
        "interpretation_scope": {
            "frozen_no_go_preserved": True,
            "new_metrics_exploratory_only": True,
            "dev_case_outcomes_burned": True,
            "raw_ser_measures_perception_plus_context": True,
            "committed_memory_measures_end_to_end_reliability": True,
        },
        "model_calls": 0,
        "endpoint_values_evaluated": True,
        "day_test140_accessed": False,
        "week_accessed": False,
    }


def _ser_statistics(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    names = ("summary_tokens",) + tuple(f"{field}_count" for field in SER_FIELDS)
    by_arm: dict[str, dict[str, Any]] = {}
    values_by_arm: dict[str, dict[str, list[int]]] = {}
    for arm in ARMS:
        values: dict[str, list[int]] = {name: [] for name in names}
        for (segment_id, row_arm), row in rows.items():
            if row_arm != arm:
                continue
            ser = dict(row.get("model_output", {}).get("structured_event_record", {}) or {})
            values["summary_tokens"].append(budget_token_count(str(ser.get("summary", ""))))
            for field in SER_FIELDS:
                item = ser.get(field, ())
                values[f"{field}_count"].append(len(item) if isinstance(item, list) else 0)
        values_by_arm[arm] = values
        by_arm[arm] = {name: _distribution(items) for name, items in values.items()}

    paired: dict[str, dict[str, Any]] = {}
    segment_ids = sorted({segment_id for segment_id, arm in rows})
    for treatment, control, label in (
        ("e1c2", "e1c1", "e1c2_minus_e1c1"),
        ("e1c1", "e1c0", "e1c1_minus_e1c0"),
    ):
        paired[label] = {}
        for name in names:
            treatment_values = values_by_arm[treatment][name]
            control_values = values_by_arm[control][name]
            if len(treatment_values) != len(segment_ids) or len(control_values) != len(segment_ids):
                raise RuntimeError("WP17 SER paired structural rows are misaligned")
            paired[label][name] = _distribution(
                [left - right for left, right in zip(treatment_values, control_values)]
            )
    return {"per_arm": by_arm, "paired_differences": paired}


def _lifecycle_statistics(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    events = [
        dict(event)
        for (segment_id, arm), row in rows.items()
        if arm == "e1c2"
        for event in tuple(row.get("lifecycle_events", ()) or ())
    ]
    by_slot_operation = Counter(
        (str(event.get("slot", "none")), str(event.get("operation", "unknown")))
        for event in events
    )
    occurrence_events = [event for event in events if event.get("slot") == "occurrence_counter"]
    occurrence_capsules = sum(
        any(slot.get("slot") == "occurrence_counter" for slot in row.get("capsule", {}).get("slots", ()))
        for (segment_id, arm), row in rows.items()
        if arm == "e1c2"
    )
    return {
        "event_count": len(events),
        "by_slot_operation": [
            {"slot": slot, "operation": operation, "count": count}
            for (slot, operation), count in sorted(by_slot_operation.items())
        ],
        "occurrence_counter": {
            "event_count": len(occurrence_events),
            "write_count": sum(event.get("operation") == "write" for event in occurrence_events),
            "update_count": sum(event.get("operation") == "update" for event in occurrence_events),
            "capsule_segment_count": occurrence_capsules,
        },
        "runtime_lifecycle_sweep_count": sum(
            event.get("operation") == "runtime_lifecycle_sweep" for event in events
        ),
        "redundant_operation_count": sum(
            str(event.get("operation", "")).startswith("redundant_") for event in events
        ),
    }


def _failure_fingerprints(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    counter: Counter[tuple[str, str, str]] = Counter()
    missing_details = 0
    attempts = 0
    for (segment_id, arm), row in rows.items():
        if arm != "e1c2":
            continue
        for attempt in tuple(row.get("attempts", ()) or ()):
            if attempt.get("status") != "validation_failed":
                continue
            attempts += 1
            repair = dict(attempt.get("repair_contract", {}) or {})
            details = dict(repair.get("details", {}) or {})
            code = str(attempt.get("failure_code") or repair.get("error_code") or "unknown")
            slot = str(details.get("slot", "none"))
            status = str(details.get("current_status", details.get("from_status", "unknown")))
            counter[(code, slot, status)] += 1
            if not details:
                missing_details += 1
    return {
        "validation_failure_attempts": attempts,
        "missing_structured_details": missing_details,
        "histogram": [
            {"error_code": code, "slot": slot, "from_status": status, "count": count}
            for (code, slot, status), count in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def _coverage_scope(
    *,
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Any],
    committed_only: bool,
) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    for case_id, raw_annotation in sorted(annotations.items()):
        annotation = dict(raw_annotation)
        anchor_ids = [
            str(segment["segment_id"])
            for segment in segments
            if _overlaps_any(segment, annotation["anchor_intervals"])
        ]
        for arm in ARMS:
            selected = [rows[(segment_id, arm)] for segment_id in anchor_ids]
            included = [
                row
                for row in selected
                if not committed_only
                or (
                    row.get("ser_endpoint_eligible") is True
                    and row.get("slot_transaction_abstained") is not True
                )
            ]
            artifact = _normalize_text(
                " ".join(
                    _flatten_text(row.get("model_output", {}).get("structured_event_record", {}))
                    for row in included
                )
            )
            entity = _first_match(artifact, annotation.get("entity_terms", ()))
            event = _first_match(artifact, annotation.get("event_terms", ()))
            state = _first_match(artifact, annotation.get("state_terms", ()))
            occurrence = _first_match(artifact, annotation.get("occurrence_terms", ()))
            ordinal = _first_match(artifact, annotation.get("ordinal_terms", ()))
            case_rows.append(
                {
                    "case_id": str(case_id),
                    "arm": arm,
                    "anchor_rows": len(selected),
                    "included_rows": len(included),
                    "entity": int(entity is not None),
                    "event": int(event is not None),
                    "anchor": int(entity is not None and event is not None),
                    "state": int(state is not None),
                    "occurrence": int(occurrence is not None) if annotation.get("occurrence_terms") else None,
                    "ordinal": int(ordinal is not None) if annotation.get("ordinal_terms") else None,
                }
            )

    metrics: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in case_rows if row["arm"] == arm]
        metrics[arm] = {}
        for metric in ("entity", "event", "anchor", "state", "occurrence", "ordinal"):
            values = [int(row[metric]) for row in arm_rows if row[metric] is not None]
            metrics[arm][metric] = {
                "count": sum(values),
                "denominator": len(values),
                "rate": sum(values) / len(values) if values else None,
            }
        entity_rows = [row for row in arm_rows if row["entity"] == 1]
        metrics[arm]["event_match_given_entity_match"] = {
            "count": sum(row["event"] for row in entity_rows),
            "denominator": len(entity_rows),
            "rate": (
                sum(row["event"] for row in entity_rows) / len(entity_rows)
                if entity_rows
                else None
            ),
        }
        metrics[arm]["ineligible_anchor_rows"] = sum(
            row["anchor_rows"] - row["included_rows"] for row in arm_rows
        )
    return {"per_arm": metrics, "case_scores": case_rows}


def _canonical_evidence_summary(
    path: Path | None,
    *,
    terms: Sequence[str],
    before_sec: float | None,
) -> dict[str, Any]:
    if path is None:
        return {"evaluated": False}
    normalized_terms = {str(term): _normalize_text(str(term)) for term in terms}
    hits: dict[str, list[tuple[float, float]]] = {term: [] for term in normalized_terms}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            surface = _normalize_text(
                " ".join(
                    str(value)
                    for key in ("surface", "normalized_surface")
                    for value in (row.get(key),)
                    if value
                )
                + " "
                + " ".join(str(value) for value in tuple(row.get("surfaces", ()) or ()))
            )
            start = float(row.get("start_sec", 0.0) or 0.0)
            end = float(row.get("end_sec", start) or start)
            for term, normalized in normalized_terms.items():
                if normalized and normalized in surface:
                    hits[term].append((start, end))
    return {
        "evaluated": True,
        "before_sec": before_sec,
        "terms": [
            {
                "term": term,
                "total_hits": len(values),
                "hits_before_boundary": sum(
                    end <= float(before_sec) for start, end in values
                )
                if before_sec is not None
                else None,
                "earliest_start_sec": min((start for start, end in values), default=None),
                "latest_end_sec": max((end for start, end in values), default=None),
            }
            for term, values in hits.items()
        ],
    }


def _distribution(values: Sequence[int | float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": mean(ordered) if ordered else 0.0,
        "p50": _percentile(ordered, 0.5) if ordered else 0.0,
        "p95": _percentile(ordered, 0.95) if ordered else 0.0,
        "min": min(ordered) if ordered else 0.0,
        "max": max(ordered) if ordered else 0.0,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = quantile * (len(values) - 1)
    left = int(position)
    right = min(len(values) - 1, left + 1)
    weight = position - left
    return float(values[left] * (1.0 - weight) + values[right] * weight)


def _overlaps_any(segment: Mapping[str, Any], intervals: Sequence[Sequence[float]]) -> bool:
    start = float(segment["virtual_start_sec"])
    end = float(segment["virtual_end_sec"])
    return any(start < float(right) and end > float(left) for left, right in intervals)


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            part
            for key, item in value.items()
            for part in (str(key), _flatten_text(item))
        )
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value) if value is not None else ""


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _first_match(artifact: str, terms: Sequence[str]) -> str | None:
    for term in terms:
        normalized = _normalize_text(str(term))
        if normalized and normalized in artifact:
            return str(term)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = dict(report["counts"])
    raw = dict(report["strict_lexical_exploratory_coverage"]["raw_ser"]["per_arm"])
    committed = dict(
        report["strict_lexical_exploratory_coverage"]["committed_memory"]["per_arm"]
    )
    occurrence = dict(report["lifecycle"]["occurrence_counter"])
    lines = [
        "# WP17 v10 Zero-Model Forensics",
        "",
        f"- Source commit: `{report['source_commit']}`",
        f"- E1C2 transaction abstain: `{counts['e1c2_transaction_abstentions']}` / `{counts['segments']}`",
        f"- Occurrence-counter writes/updates: `{occurrence['write_count']}` / `{occurrence['update_count']}`",
        "- Model calls: `0`",
        "- Scope: exploratory development diagnosis; frozen NO-GO is preserved.",
        "",
        "| Scope | E1C0 ARC | E1C1 ARC | E1C2 ARC | E1C2 event/entity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, values in (("raw SER", raw), ("committed memory", committed)):
        arc = [values[arm]["anchor"] for arm in ARMS]
        conditional = values["e1c2"]["event_match_given_entity_match"]
        lines.append(
            f"| {label} | {arc[0]['count']}/{arc[0]['denominator']} | "
            f"{arc[1]['count']}/{arc[1]['denominator']} | "
            f"{arc[2]['count']}/{arc[2]['denominator']} | "
            f"{conditional['count']}/{conditional['denominator']} |"
        )
    lines.extend(("", "Full structured results are in `wp17_slot_failure_forensics.json`.", ""))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    report_path = out_root / "wp17_slot_failure_forensics.json"
    markdown_path = out_root / "wp17_slot_failure_forensics.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17 v10 forensics output already exists")
    report = build_report(
        run_root=Path(args.run_root),
        construction_protocol=_read_json(Path(args.construction_protocol)),
        analysis_protocol=_read_json(Path(args.analysis_protocol)),
        source_commit=str(args.source_commit),
        dense_evidence_path=(Path(args.dense_evidence) if args.dense_evidence else None),
        canonical_terms=tuple(args.canonical_term or ()),
        canonical_before_sec=args.canonical_before_sec,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(
        "WP17_SLOT_V10_FORENSICS_DONE "
        f"segments={report['counts']['segments']} "
        f"abstentions={report['counts']['e1c2_transaction_abstentions']} "
        "model_calls=0 holdout=false",
        flush=True,
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--construction-protocol", required=True)
    parser.add_argument("--analysis-protocol", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dense-evidence")
    parser.add_argument("--canonical-term", action="append", default=[])
    parser.add_argument("--canonical-before-sec", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
