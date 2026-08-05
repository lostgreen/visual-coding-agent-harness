#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def collect_replay_root(root: Path) -> dict[str, Any]:
    cases = []
    for case_dir in sorted((Path(root) / "cases").glob("*")):
        audit_path = case_dir / "phase5r_replay.json"
        config_path = case_dir / "run_config.json"
        if not (audit_path.is_file() and config_path.is_file()):
            continue
        audit = _read_json(audit_path)
        config = _read_json(config_path)
        cases.append(
            {
                "case_id": str(audit.get("case_id", case_dir.name) or case_dir.name),
                "decision": str(audit.get("decision", "") or ""),
                "failed_checks": list(audit.get("failed_checks", ()) or ()),
                "expected_frames": int(
                    _mapping(audit.get("expected")).get("frame_count", 0) or 0
                ),
                "actual_frames": int(
                    _mapping(audit.get("actual")).get("frame_count", 0) or 0
                ),
                "checks": dict(_mapping(audit.get("checks"))),
                "cost_breakdown": dict(
                    _mapping(audit.get("cost_breakdown"))
                ),
                "phase5r_mode": str(config.get("phase5r_mode", "") or ""),
                "controller_mode": str(config.get("controller_mode", "") or ""),
                "recorded_fixture_digest": str(
                    config.get("recorded_fixture_digest", "") or ""
                ),
            }
        )
    return {
        "root": str(root),
        "case_count": len(cases),
        "case_ids": sorted(case["case_id"] for case in cases),
        "cases": cases,
    }


def gate_r1(
    root: Mapping[str, Any],
    *,
    expected_case_ids: Sequence[str],
) -> dict[str, Any]:
    cases = [dict(case) for case in tuple(root.get("cases", ()) or ())]
    expected_ids = sorted(str(case_id) for case_id in expected_case_ids)
    observed_ids = sorted(str(case.get("case_id", "") or "") for case in cases)
    checks = {
        "case_ids_exact": observed_ids == expected_ids,
        "recorded_replay_mode": bool(cases)
        and all(case.get("phase5r_mode") == "recorded_replay" for case in cases),
        "frozen_runtime_path": bool(cases)
        and all(case.get("controller_mode") == "frozen_baseline" for case in cases),
        "fixture_identity_present": bool(cases)
        and all(case.get("recorded_fixture_digest") for case in cases),
        "all_case_mechanical_parity": bool(cases)
        and all(case.get("decision") == "PASS" for case in cases),
    }
    return {
        "schema_version": "MGERPhase5RGateR1V1",
        "stage": "mechanical_determinism",
        "decision": "PASS" if all(checks.values()) else "STOP",
        "judge_required": False,
        "screening_only": True,
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "expected_case_ids": expected_ids,
        "observed_case_ids": observed_ids,
        "totals": {
            "expected_frames": sum(int(case.get("expected_frames", 0)) for case in cases),
            "actual_frames": sum(int(case.get("actual_frames", 0)) for case in cases),
            "passing_cases": sum(case.get("decision") == "PASS" for case in cases),
            "case_count": len(cases),
        },
        "cases": [
            {
                "case_id": case.get("case_id"),
                "decision": case.get("decision"),
                "failed_checks": case.get("failed_checks"),
                "expected_frames": case.get("expected_frames"),
                "actual_frames": case.get("actual_frames"),
                "cost_breakdown": case.get("cost_breakdown"),
            }
            for case in cases
        ],
        "root": str(root.get("root", "") or ""),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the Phase 5R R1 replay gate.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if bool(args.fixture_manifest) == bool(args.case_ids):
        parser.error("provide exactly one of --fixture-manifest or --case-ids")
    if args.fixture_manifest:
        manifest = _read_json(args.fixture_manifest)
        expected_case_ids = [str(item) for item in manifest.get("case_ids", ()) or ()]
    else:
        expected_case_ids = list(args.case_ids)
    result = gate_r1(
        collect_replay_root(args.root),
        expected_case_ids=expected_case_ids,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["decision"] == "PASS" else 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    raise SystemExit(main())
