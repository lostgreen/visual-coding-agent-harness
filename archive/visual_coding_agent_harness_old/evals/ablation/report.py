#!/usr/bin/env python3
"""Generate a compact markdown report for an ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import audit_trajectory


METRICS = [
    "accuracy",
    "final_rate",
    "need_more_evidence_rate",
    "unsupported_final_rate",
    "low_confidence_final_rate",
    "followup_success_rate",
    "avg_followups_per_case",
    "tool_nframes_compliance",
    "evidence_provenance_completeness",
    "context_budget_overflow_count",
    "route_violations",
]


def build_report(*, matrix_json: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_json.read_text(encoding="utf-8"))
    if not isinstance(matrix, Mapping):
        raise ValueError("matrix index must be an object")
    runs = []
    for entry in matrix.get("runs", []):
        if not isinstance(entry, Mapping):
            continue
        summary_path = Path(str(entry.get("summary_path", "")))
        summary = _read_json(summary_path)
        runs.append(_run_report(entry=entry, summary=summary, summary_path=summary_path))
    return {
        "schema_version": "AblationReportV1",
        "matrix_id": str(matrix.get("matrix_id", "")),
        "runs": runs,
        "best_accuracy_run": _best_run(runs, metric="accuracy"),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Ablation Report",
        "",
        f"Matrix: `{report.get('matrix_id', '')}`",
        "",
        "## Metrics",
        "",
        "| Run | Accuracy | Final | Need More | Unsupported | Followup | Evidence Chains | Traj Audit Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.get("runs", []):
        metrics = run.get("metrics", {})
        lines.append(
            "| {id} | {accuracy} | {final_rate} | {need_more_evidence_rate} | "
            "{unsupported_final_rate} | {followup_success_rate} | {evidence_chain_rows} | "
            "{trajectory_audit_failures} |".format(
                id=run.get("id", ""),
                accuracy=_fmt(metrics.get("accuracy")),
                final_rate=_fmt(metrics.get("final_rate")),
                need_more_evidence_rate=_fmt(metrics.get("need_more_evidence_rate")),
                unsupported_final_rate=_fmt(metrics.get("unsupported_final_rate")),
                followup_success_rate=_fmt(metrics.get("followup_success_rate")),
                evidence_chain_rows=run.get("evidence_chain_rows", 0),
                trajectory_audit_failures=run.get("trajectory_audit_failures", 0),
            )
        )
    lines.extend(["", "## Completeness", ""])
    for run in report.get("runs", []):
        lines.append(
            f"- `{run.get('id', '')}`: evidence_chains={run.get('evidence_chain_rows', 0)}, "
            f"trajectory_audits={len(run.get('trajectory_audits', []))}, "
            f"audit_failures={run.get('trajectory_audit_failures', 0)}"
        )
    best = report.get("best_accuracy_run") or ""
    if best:
        lines.extend(["", "## Key Findings", "", f"- Best accuracy run: `{best}`."])
    lines.append("")
    return "\n".join(lines)


def write_report(*, report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_report.json"
    md_path = output_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _run_report(*, entry: Mapping[str, Any], summary: Mapping[str, Any], summary_path: Path) -> dict[str, Any]:
    metrics = {name: summary.get(name, 0.0) for name in METRICS}
    evidence_chain_rows = _count_jsonl(Path(str(summary.get("evidence_chains_path", ""))), summary_path=summary_path)
    audits = _trajectory_audits(summary, summary_path=summary_path)
    return {
        "id": str(entry.get("id", "")),
        "status": str(entry.get("status", "")),
        "exit_code": entry.get("exit_code"),
        "summary_path": str(summary_path),
        "metrics": metrics,
        "evidence_chain_rows": evidence_chain_rows,
        "trajectory_audits": audits,
        "trajectory_audit_failures": sum(1 for audit in audits if int(audit.get("status", 0)) != 0),
    }


def _trajectory_audits(summary: Mapping[str, Any], *, summary_path: Path) -> list[dict[str, Any]]:
    audits = []
    for case in summary.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        artifacts = case.get("raw_artifacts", {})
        if not isinstance(artifacts, Mapping):
            continue
        trajectories = artifacts.get("training_trajectories", {})
        if not isinstance(trajectories, Mapping):
            continue
        for strategy, raw_path in trajectories.items():
            path = _resolve_path(Path(str(raw_path)), summary_path=summary_path)
            if not path.exists():
                audits.append({"strategy": str(strategy), "path": str(path), "status": 2, "output": "missing"})
                continue
            status, output = audit_trajectory(path)
            audits.append({"strategy": str(strategy), "path": str(path), "status": status, "output": output})
    return audits


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _count_jsonl(path: Path, *, summary_path: Path) -> int:
    resolved = _resolve_path(path, summary_path=summary_path)
    if not resolved.exists():
        return 0
    count = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _resolve_path(path: Path, *, summary_path: Path) -> Path:
    if path.is_absolute():
        return path
    return summary_path.parent / path


def _best_run(runs: Sequence[Mapping[str, Any]], *, metric: str) -> str:
    if not runs:
        return ""
    best = max(runs, key=lambda run: float(run.get("metrics", {}).get(metric, 0.0) or 0.0))
    return str(best.get("id", ""))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an ablation markdown report from a matrix index.")
    parser.add_argument("--matrix-json", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = build_report(matrix_json=args.matrix_json)
    output_dir = args.output_dir or args.matrix_json.parent
    _json_path, md_path = write_report(report=report, output_dir=output_dir)
    print(f"DONE report={md_path}", flush=True)


if __name__ == "__main__":
    main()
