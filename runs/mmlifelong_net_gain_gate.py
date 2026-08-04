#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from evaluate.mmlifelong.metrics import recorded_case_diagnostics
from vcah.workspace import WorkingDocument


METRIC_KEYS = (
    "official_accuracy",
    "reference_valid",
    "visual_frames_inspected",
    "clue_frame_coverage",
    "retrieval_dedup_rate",
    "sampling_fidelity_mean",
    "sampling_fidelity_min",
    "anchor_consistency",
)


def collect_recorded_cases(root: Path) -> dict[str, dict[str, float | int]]:
    source = Path(root)
    candidates = (
        (source,)
        if source.name == "mmlifelong_eval.json"
        else tuple(source.rglob("mmlifelong_eval.json"))
    )
    cases: dict[str, dict[str, float | int]] = {}
    for evaluation_path in sorted(candidates):
        if not evaluation_path.is_file():
            continue
        case_root = (
            evaluation_path.parent.parent
            if evaluation_path.parent.name == "evaluation"
            else evaluation_path.parent
        )
        evaluation = _read_required_json(evaluation_path)
        case_id = str(evaluation.get("case_id", case_root.name) or case_root.name)
        if case_id in cases:
            raise ValueError(f"duplicate case_id in one paired run: {case_id}")
        summary = _read_required_json(case_root / "runtime_summary.json")
        document_payload = _read_required_json(case_root / "working_document.json")
        document = WorkingDocument.from_mapping(document_payload)
        observations = _read_jsonl(case_root / "observation_log.jsonl")
        grounding = evaluation.get("reference_grounding", {})
        grounding_metrics = grounding if isinstance(grounding, Mapping) else {}
        cases[case_id] = recorded_case_diagnostics(
            evaluation,
            document,
            observations,
            runtime_summary=summary,
            supporting_claim_ids=tuple(summary.get("supporting_claim_ids", ()) or ()),
            reference_intervals=tuple(
                grounding_metrics.get("reference_intervals", ()) or ()
            ),
        )
    if not cases:
        raise ValueError(f"no mmlifelong_eval.json files found under {source}")
    return cases


def evaluate_run_pair(
    baseline: Mapping[str, Mapping[str, float | int]],
    candidate: Mapping[str, Mapping[str, float | int]],
    *,
    label: str,
    run_identity: str = "",
    max_frame_ratio: float = 1.25,
) -> dict[str, Any]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    matched_ids = tuple(sorted(baseline_ids & candidate_ids))
    failures: list[str] = []
    if missing := sorted(baseline_ids - candidate_ids):
        failures.append("missing_candidate_cases:" + ",".join(missing))
    if extra := sorted(candidate_ids - baseline_ids):
        failures.append("unexpected_candidate_cases:" + ",".join(extra))
    if not matched_ids:
        failures.append("no_matched_cases")

    baseline_aggregate = _aggregate(tuple(baseline[case_id] for case_id in matched_ids))
    candidate_aggregate = _aggregate(tuple(candidate[case_id] for case_id in matched_ids))
    official_accuracy_delta = (
        candidate_aggregate["official_accuracy"]
        - baseline_aggregate["official_accuracy"]
    )
    frame_ratio = _cost_ratio(
        baseline_aggregate["visual_frames_inspected"],
        candidate_aggregate["visual_frames_inspected"],
    )
    checks = {
        "official_accuracy_improved": official_accuracy_delta > 0.0,
        "reference_non_regression": (
            candidate_aggregate["reference_valid"]
            >= baseline_aggregate["reference_valid"]
        ),
        "frame_cost_within_limit": (
            frame_ratio is not None and frame_ratio <= float(max_frame_ratio)
        ),
        "clue_frame_coverage_non_regression": (
            candidate_aggregate["clue_frame_coverage"]
            >= baseline_aggregate["clue_frame_coverage"]
        ),
        "anchor_consistency_non_regression": (
            candidate_aggregate["anchor_consistency"]
            >= baseline_aggregate["anchor_consistency"]
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "label": str(label),
        "run_identity": str(run_identity or label),
        "case_count": len(matched_ids),
        "case_ids": list(matched_ids),
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "official_accuracy_delta": official_accuracy_delta,
        "visual_frame_ratio": frame_ratio,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def evaluate_net_gain(
    pair_reports: Sequence[Mapping[str, Any]],
    *,
    min_independent_repeats: int = 2,
) -> dict[str, Any]:
    reports = tuple(dict(item) for item in pair_reports)
    failures: list[str] = []
    required = max(1, int(min_independent_repeats))
    identities = {
        str(report.get("run_identity") or report.get("label") or index)
        for index, report in enumerate(reports, start=1)
    }
    if len(identities) < required:
        failures.append(f"insufficient_independent_repeats:{len(identities)}<{required}")
    for index, report in enumerate(reports, start=1):
        if not bool(report.get("passed")):
            label = str(report.get("label", f"pair_{index}"))
            failures.append(f"paired_repeat_failed:{label}")
    return {
        "schema_version": "MMLifelongNetGainGateV2",
        "passed": not failures,
        "failures": failures,
        "submitted_pair_count": len(reports),
        "independent_repeat_count": len(identities),
        "required_independent_repeats": required,
        "pairs": list(reports),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply paired MM-Lifelong accuracy, reference, cost, clue and anchor gates."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("BASELINE_ROOT", "CANDIDATE_ROOT"),
        required=True,
    )
    parser.add_argument("--max-frame-ratio", type=float, default=1.25)
    parser.add_argument("--min-independent-repeats", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    reports = []
    for index, (baseline_root, candidate_root) in enumerate(args.pair, start=1):
        baseline_path = Path(baseline_root).resolve()
        candidate_path = Path(candidate_root).resolve()
        reports.append(
            evaluate_run_pair(
                collect_recorded_cases(baseline_path),
                collect_recorded_cases(candidate_path),
                label=f"pair_{index}",
                run_identity=f"{baseline_path}::{candidate_path}",
                max_frame_ratio=args.max_frame_ratio,
            )
        )
    result = evaluate_net_gain(
        reports,
        min_independent_repeats=args.min_independent_repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


def _aggregate(cases: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    if not cases:
        return {key: 0.0 for key in METRIC_KEYS}
    return {
        key: mean(float(case.get(key, 0.0) or 0.0) for case in cases)
        for key in METRIC_KEYS
    }


def _cost_ratio(baseline: float, candidate: float) -> float | None:
    if baseline > 0.0:
        return candidate / baseline
    return 1.0 if candidate <= 0.0 else None


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object in {path}")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return tuple(rows)


if __name__ == "__main__":
    raise SystemExit(main())
