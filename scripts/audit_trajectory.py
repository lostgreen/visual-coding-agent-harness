#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def audit_trajectory(path: Path) -> tuple[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return 2, "invalid trajectory: root payload is not an object"
    if payload.get("schema_version") != "TrainingTrajectoryV1":
        return 2, f"invalid schema: {payload.get('schema_version')}"
    tool_calls = payload.get("tool_calls", [])
    evidence_chain_ids = payload.get("evidence_chain_ids", [])
    final_decision = str(payload.get("final_decision", ""))
    errors = []
    if not isinstance(tool_calls, list) or not tool_calls:
        errors.append("missing tool_calls")
    if final_decision in {"final", "low_confidence_final"} and not evidence_chain_ids:
        errors.append("final decision has no evidence chains")
    lines = [
        f"case_id: {payload.get('case_id', '')}",
        f"final_decision: {final_decision}",
        f"selected_option: {payload.get('selected_option', '')}",
        f"is_correct: {payload.get('is_correct', '')}",
        f"tool_calls: {len(tool_calls) if isinstance(tool_calls, list) else 0}",
        f"evidence_chains: {len(evidence_chain_ids) if isinstance(evidence_chain_ids, list) else 0}",
        f"frame_sets: {len(payload.get('frame_set_ids', []) or [])}",
        f"context_budget_reports: {len(payload.get('context_budget_reports', []) or [])}",
    ]
    if errors:
        lines.append("errors: " + "; ".join(errors))
        return 1, "\n".join(lines)
    return 0, "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a compact TrainingTrajectory JSON artifact.")
    parser.add_argument("trajectory_json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    status, output = audit_trajectory(args.trajectory_json)
    print(output)
    raise SystemExit(status)


if __name__ == "__main__":
    main(sys.argv[1:])
