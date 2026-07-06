#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from runs.report_metrics import report_metrics
    from runs.summary_schema import load_summary  # type: ignore[attr-defined]
except Exception:
    from report_metrics import report_metrics
    from summary_schema import RunSummary

    def load_summary(path: Path) -> RunSummary:
        return RunSummary.from_mapping(json.loads(path.read_text(encoding="utf-8")))


DEFAULT_BLOCKED_WRONG = {"1810": "B", "1817": "C", "1823": "D"}


def evaluate_gate(
    summary_path: Path,
    *,
    baseline_path: Path | None = None,
    refusal_rate_max_delta: float = 0.05,
    blocked_wrong: dict[str, str] | None = None,
) -> dict[str, object]:
    summary = load_summary(summary_path)
    metrics = report_metrics(summary_path, baseline_path=baseline_path)
    failures: list[str] = []
    blocked_wrong = DEFAULT_BLOCKED_WRONG if blocked_wrong is None else blocked_wrong
    cases = {case.case_id: case for case in summary.cases}

    if baseline_path is not None and metrics.get("new_wrong_count", 0):
        failures.append("new_wrong_count")
    if "1822" in cases and cases["1822"].correct is not True:
        failures.append("case_1822_not_correct")
    for case_id, old_wrong_answer in blocked_wrong.items():
        case = cases.get(case_id)
        if case and case.final_answer == old_wrong_answer and case.correct is False:
            failures.append(f"target_case_still_wrong:{case_id}")
    if baseline_path is not None and float(metrics.get("refusal_rate_delta", 0.0)) > float(refusal_rate_max_delta):
        failures.append("refusal_rate_delta")

    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply compact VCAH regression gates.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--refusal-rate-max-delta", type=float, default=0.05)
    args = parser.parse_args(argv)
    result = evaluate_gate(
        args.summary,
        baseline_path=args.baseline,
        refusal_rate_max_delta=args.refusal_rate_max_delta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
