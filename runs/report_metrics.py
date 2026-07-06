#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from runs.summary_schema import RunSummary
except ModuleNotFoundError:
    from summary_schema import RunSummary


def load_summary(path: Path) -> RunSummary:
    return RunSummary.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def trace_metrics(trace_path: Path) -> dict[str, Any]:
    evidence_scope = Counter()
    evidence_kind = Counter()
    evidence_polarity = Counter()
    failure_taxonomy = Counter()
    final_reasons = Counter()
    window_errors = Counter()
    if not trace_path.exists():
        return {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        for record in item.get("evidence_records") or ():
            evidence_scope[str(record.get("temporal_scope") or "unknown")] += 1
            evidence_kind[str(record.get("evidence_kind") or "unknown")] += 1
            evidence_polarity[str(record.get("observation_polarity") or "unknown")] += 1
        lineage = item.get("window_lineage") or {}
        if lineage.get("error"):
            window_errors[str(lineage["error"])] += 1
            failure_taxonomy["dropped_window"] += 1
        final = item.get("final_verification") or {}
        if final.get("reason"):
            reason = str(final["reason"])
            final_reasons[reason] += 1
            failure_taxonomy[_failure_bucket(reason)] += 1
    return {
        "evidence_scope_histogram": dict(evidence_scope),
        "evidence_kind_histogram": dict(evidence_kind),
        "evidence_polarity_histogram": dict(evidence_polarity),
        "window_errors": dict(window_errors),
        "final_reasons": dict(final_reasons),
        "failure_taxonomy": dict(failure_taxonomy),
    }


def report_metrics(summary_path: Path, *, trace_path: Path | None = None, baseline_path: Path | None = None) -> dict[str, Any]:
    summary = load_summary(summary_path)
    metrics = summary.to_metrics()
    if trace_path is not None:
        metrics.update(trace_metrics(trace_path))
    if baseline_path is not None and baseline_path.exists():
        baseline = load_summary(baseline_path)
        baseline_cases = {case.case_id: case for case in baseline.cases}
        new_wrong = []
        for case in summary.cases:
            old = baseline_cases.get(case.case_id)
            if old and old.correct and case.correct is False and not case.refused:
                new_wrong.append(case.case_id)
        metrics["new_wrong_count"] = len(new_wrong)
        metrics["new_wrong_cases"] = new_wrong
        metrics["refusal_rate_delta"] = metrics["refusal_rate"] - baseline.to_metrics()["refusal_rate"]
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report compact VCAH run metrics.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(report_metrics(args.summary, trace_path=args.trace, baseline_path=args.baseline), indent=2, sort_keys=True))
    return 0


def _failure_bucket(reason: str) -> str:
    if "claim_ledger" in reason:
        return "legacy_path_blocked"
    if "scope" in reason:
        return "scope_overreach"
    if "observability" in reason:
        return "observability_mismatch"
    if "predicate" in reason or "proxy" in reason:
        return "predicate_mismatch"
    if "coverage" in reason:
        return "window_coverage_failed"
    if "aggregation" in reason:
        return "aggregation_missing"
    return reason


if __name__ == "__main__":
    raise SystemExit(main())
