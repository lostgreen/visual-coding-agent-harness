#!/usr/bin/env python3
"""Compact structural audit for no-oracle occurrence-method runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def audit_roots(
    bindings: Mapping[str, Path], *, expected_cases: int
) -> dict[str, Any]:
    per_arm: dict[str, dict[str, Any]] = {}
    case_sets: dict[str, set[str]] = {}
    for declared_arm, root in bindings.items():
        cases = []
        for prediction_path in sorted(Path(root).glob("cases/*/prediction.json")):
            run_dir = prediction_path.parent
            config = _read_json(run_dir / "run_config.json")
            runtime = _read_json(run_dir / "runtime_summary.json")
            trace = tuple(
                row
                for row in tuple(runtime.get("trace", ()) or ())
                if isinstance(row, Mapping)
            )
            decisions = tuple(
                row for row in trace if row.get("type") == "reasoner_decision"
            )
            occurrence_errors = tuple(
                error
                for row in trace
                if row.get("type") == "decision_schema_error"
                for error in tuple(row.get("errors", ()) or ())
                if isinstance(error, Mapping)
                and str(error.get("code", "")).startswith("occurrence_")
            )
            arm = str(config.get("occurrence_method_arm", "none") or "none")
            case_id = str(_read_json(prediction_path).get("case_id", run_dir.name))
            cases.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "models": config.get("models"),
                    "no_oracle": bool(
                        runtime.get("no_oracle_runtime_gate", {}).get(
                            "no_oracle_runtime_gate_passed", False
                        )
                    ),
                    "eligible_events": sum(
                        row.get("type") == "occurrence_treatment_eligible"
                        for row in trace
                    ),
                    "exposure_events": sum(
                        row.get("type") == "occurrence_treatment_exposed"
                        for row in trace
                    ),
                    "occurrence_ops": sum(
                        len(tuple(row.get("occurrence_ops", ()) or ()))
                        for row in decisions
                    ),
                    "ops_accepted": all(
                        row.get("occurrence_ops_accepted") is not False
                        for row in decisions
                    ),
                    "occurrence_schema_errors": len(occurrence_errors),
                    "state_file": (
                        run_dir / "occurrence_resolution_state.json"
                    ).is_file(),
                    "premature_commits": sum(
                        bool(row.get("premature_occurrence_commit"))
                        for row in decisions
                    ),
                }
            )
        case_sets[declared_arm] = {row["case_id"] for row in cases}
        per_arm[declared_arm] = {
            "n": len(cases),
            "declared_arm_match": all(
                row["arm"] == declared_arm for row in cases
            ),
            "models_ok": all(
                row["models"]
                == {
                    "reasoner": "pa/gmn-2.5-pr",
                    "investigator": "pa/gmn-2.5-pr",
                }
                for row in cases
            ),
            "no_oracle_gate_passed": all(row["no_oracle"] for row in cases),
            "eligible_event_count": sum(
                row["eligible_events"] for row in cases
            ),
            "exposure_event_count": sum(
                row["exposure_events"] for row in cases
            ),
            "occurrence_op_count": sum(row["occurrence_ops"] for row in cases),
            "all_occurrence_ops_accepted": all(
                row["ops_accepted"] for row in cases
            ),
            "occurrence_schema_error_count": sum(
                row["occurrence_schema_errors"] for row in cases
            ),
            "state_file_count": sum(row["state_file"] for row in cases),
            "premature_commit_count": sum(
                row["premature_commits"] for row in cases
            ),
        }
    common = set.intersection(*case_sets.values()) if case_sets else set()
    checks = {
        "expected_arms_present": set(bindings) == {"a0", "a1-flat", "a1", "a2"},
        "case_sets_aligned": bool(case_sets)
        and all(case_set == common for case_set in case_sets.values()),
        "expected_case_count": len(common) == int(expected_cases),
        "declared_arms_match": all(
            row["declared_arm_match"] for row in per_arm.values()
        ),
        "actual_models_match": all(row["models_ok"] for row in per_arm.values()),
        "no_oracle_gate_passed": all(
            row["no_oracle_gate_passed"] for row in per_arm.values()
        ),
        "eligible_events_complete": all(
            row["eligible_event_count"] == int(expected_cases)
            for row in per_arm.values()
        ),
        "treatment_exposure_complete": all(
            per_arm[arm]["exposure_event_count"] == int(expected_cases)
            for arm in ("a1-flat", "a1", "a2")
        ),
        "a0_has_no_treatment_exposure": per_arm.get("a0", {}).get(
            "exposure_event_count"
        )
        == 0,
        "a2_state_files_complete": per_arm.get("a2", {}).get(
            "state_file_count"
        )
        == int(expected_cases),
        "no_occurrence_schema_errors": sum(
            row["occurrence_schema_error_count"] for row in per_arm.values()
        )
        == 0,
        "all_submitted_occurrence_ops_accepted": all(
            row["all_occurrence_ops_accepted"] for row in per_arm.values()
        ),
    }
    return {
        "schema_version": "MMLifelongOccurrenceCanaryAuditV1",
        "per_arm": per_arm,
        "checks": checks,
        "structural_gate_passed": all(checks.values()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-root", action="append", nargs=2, metavar=("ARM", "ROOT"), required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    bindings = {arm: Path(root) for arm, root in args.arm_root}
    report = audit_roots(bindings, expected_cases=args.expected_cases)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "structural_gate_passed": report["structural_gate_passed"],
                "checks": report["checks"],
                "per_arm": report["per_arm"],
                "output_json": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
