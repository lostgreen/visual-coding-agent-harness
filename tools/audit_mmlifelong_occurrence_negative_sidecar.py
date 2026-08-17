#!/usr/bin/env python3
"""Audit frozen inputs for the OOB negative occurrence sidecar without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import (
    NEGATIVE_SIDECAR_CONTRACT,
    file_sha256,
    load_negative_sidecar_snapshot,
)


def build_audit(
    *,
    positive_run_root: Path,
    replay_fixture_root: Path,
    case_manifest: Path,
    expected_cases: int,
) -> dict[str, Any]:
    manifest = _read_json(case_manifest)
    case_ids = tuple(
        str(row.get("case_id", "") or "")
        for row in tuple(manifest.get("cases", ()) or ())
        if isinstance(row, Mapping) and str(row.get("case_id", "") or "")
    )
    if len(case_ids) != expected_cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("manifest case count or uniqueness check failed")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case_id in case_ids:
        try:
            snapshot = load_negative_sidecar_snapshot(
                Path(positive_run_root) / "cases" / case_id,
                replay_fixture_path=(
                    Path(replay_fixture_root) / "cases" / f"{case_id}.json"
                ),
            )
            model_payload = snapshot.model_payload()
            rows.append(
                {
                    "case_id": case_id,
                    "snapshot_digest": snapshot.digest,
                    "constraint_count": len(snapshot.constraints),
                    "candidate_count": len(snapshot.candidates),
                    "visible_passage_count": sum(
                        len(tuple(candidate.get("representative_passages", ()) or ()))
                        for candidate in snapshot.candidates
                    ),
                    "model_payload_key_count": len(model_payload),
                    "no_oracle_input_gate_passed": True,
                }
            )
        except Exception as exc:
            failures.append({"case_id": case_id, "error_type": type(exc).__name__})
    checks = {
        "expected_case_count": len(rows) == expected_cases,
        "zero_snapshot_failures": not failures,
        "all_no_oracle_inputs": all(row["no_oracle_input_gate_passed"] for row in rows),
        "all_constraints_present": all(row["constraint_count"] > 0 for row in rows),
        "all_candidates_present": all(row["candidate_count"] > 0 for row in rows),
        "all_candidates_have_passages": all(
            row["visible_passage_count"] >= row["candidate_count"] for row in rows
        ),
    }
    return {
        "schema_version": "MMLifelongOccurrenceNegativeSidecarInputAuditV1",
        "contract": NEGATIVE_SIDECAR_CONTRACT,
        "case_manifest_sha256": file_sha256(case_manifest),
        "expected_cases": expected_cases,
        "successful_snapshot_count": len(rows),
        "failure_count": len(failures),
        "structural_gate_passed": all(checks.values()),
        "checks": checks,
        "failures": failures,
        "case_rows": rows,
        "model_calls_used": False,
        "workspace_write_enabled": False,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# WP11 OOB Sidecar Input Audit",
                "",
                f"Structural gate: **{'PASS' if report['structural_gate_passed'] else 'FAIL'}**",
                "",
                (
                    f"Snapshots: {report['successful_snapshot_count']}/"
                    f"{report['expected_cases']}; failures: {report['failure_count']}."
                ),
                "",
                "No model calls were used and no positive workspace was modified.",
            ]
        )
        + "\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-run-root", required=True)
    parser.add_argument("--replay-fixture-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = build_audit(
        positive_run_root=Path(args.positive_run_root),
        replay_fixture_root=Path(args.replay_fixture_root),
        case_manifest=Path(args.case_manifest),
        expected_cases=args.expected_cases,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"SIDECAR_INPUT_AUDIT gate={report['structural_gate_passed']} "
        f"snapshots={report['successful_snapshot_count']}/"
        f"{report['expected_cases']}",
        flush=True,
    )
    if not report["structural_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
