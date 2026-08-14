#!/usr/bin/env python3
"""Compact structural audit for no-oracle occurrence-method runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


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
            raw_no_oracle_audit = runtime.get("no_oracle_runtime_gate", {})
            no_oracle_audit = (
                raw_no_oracle_audit
                if isinstance(raw_no_oracle_audit, Mapping)
                else {}
            )
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
            occurrence_validation_errors = tuple(
                error
                for error in occurrence_errors
                if str(error.get("code", ""))
                not in {
                    "occurrence_selection_required",
                    "occurrence_answer_required_after_selection",
                }
            )
            terminal_occurrence_failures = tuple(
                error
                for row in trace
                if row.get("type") == "decision_control_exhausted"
                for error in tuple(row.get("errors", ()) or ())
                if isinstance(error, Mapping)
                and str(error.get("code", "")).startswith("occurrence_")
            )
            selection_ops = sum(
                str(operation.get("op", operation.get("type", "")) or "")
                .casefold()
                == "select"
                for row in decisions
                for operation in tuple(row.get("occurrence_ops", ()) or ())
                if isinstance(operation, Mapping)
            )
            prior_selection_indices = tuple(
                index
                for index, row in enumerate(decisions)
                if row.get("action") != "answer"
                and row.get("occurrence_ops_accepted") is not False
                and any(
                    str(
                        operation.get("op", operation.get("type", "")) or ""
                    ).casefold()
                    == "select"
                    for operation in tuple(row.get("occurrence_ops", ()) or ())
                    if isinstance(operation, Mapping)
                )
            )
            selection_before_answer = bool(prior_selection_indices)
            answer_after_selection = any(
                row.get("action") == "answer"
                and index > prior_selection_indices[0]
                for index, row in enumerate(decisions)
            ) if prior_selection_indices else False
            selection_required = any(
                row.get("type") == "occurrence_treatment_eligible"
                and int(row.get("visible_occurrence_count", 0) or 0) > 1
                for row in trace
            )
            lifecycle_complete = bool(
                not selection_required
                or (selection_before_answer and answer_after_selection)
            )
            rejected_occurrence_op_attempts = sum(
                row.get("occurrence_ops_accepted") is False for row in decisions
            )
            arm = str(config.get("occurrence_method_arm", "none") or "none")
            case_id = str(_read_json(prediction_path).get("case_id", run_dir.name))
            cases.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "models": config.get("models"),
                    "no_oracle": bool(
                        no_oracle_audit.get(
                            "no_oracle_runtime_gate_passed", False
                        )
                    ),
                    "text_budget_parity": (
                        bool(no_oracle_audit.get("text_budget_parity_passed"))
                        if arm in {"a1-flat", "a1"}
                        else None
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
                    "selection_ops": selection_ops,
                    "selection_required": selection_required,
                    "selection_before_answer": selection_before_answer,
                    "answer_after_selection": answer_after_selection,
                    "lifecycle_complete": lifecycle_complete,
                    "ops_accepted": all(
                        row.get("occurrence_ops_accepted") is not False
                        for row in decisions
                    ),
                    "rejected_occurrence_op_attempts": (
                        rejected_occurrence_op_attempts
                    ),
                    "occurrence_schema_errors": len(occurrence_errors),
                    "selection_required_retries": sum(
                        str(error.get("code", ""))
                        == "occurrence_selection_required"
                        for error in occurrence_errors
                    ),
                    "answer_required_retries": sum(
                        str(error.get("code", ""))
                        == "occurrence_answer_required_after_selection"
                        for error in occurrence_errors
                    ),
                    "occurrence_validation_errors": len(
                        occurrence_validation_errors
                    ),
                    "terminal_occurrence_failures": len(
                        terminal_occurrence_failures
                    ),
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
            "text_budget_parity_passed": all(
                row["text_budget_parity"] is True
                for row in cases
                if row["arm"] in {"a1-flat", "a1"}
            ),
            "eligible_event_count": sum(
                row["eligible_events"] for row in cases
            ),
            "exposure_event_count": sum(
                row["exposure_events"] for row in cases
            ),
            "duplicate_eligibility_event_case_count": sum(
                row["eligible_events"] > 1 for row in cases
            ),
            "duplicate_exposure_event_case_count": sum(
                row["exposure_events"] > 1 for row in cases
            ),
            "eligible_without_exposure_case_count": sum(
                row["eligible_events"] > 0 and row["exposure_events"] == 0
                for row in cases
            ),
            "exposure_without_eligibility_case_count": sum(
                row["exposure_events"] > 0 and row["eligible_events"] == 0
                for row in cases
            ),
            "occurrence_op_count": sum(row["occurrence_ops"] for row in cases),
            "selection_case_count": sum(
                row["selection_ops"] > 0 for row in cases
            ),
            "selection_required_case_count": sum(
                row["selection_required"] for row in cases
            ),
            "selection_missing_case_count": sum(
                row["selection_required"] and row["selection_ops"] == 0
                for row in cases
            ),
            "selection_not_prior_case_count": sum(
                row["selection_required"]
                and not row["selection_before_answer"]
                for row in cases
            ),
            "answer_missing_after_selection_case_count": sum(
                row["selection_required"]
                and not row["answer_after_selection"]
                for row in cases
            ),
            "all_occurrence_ops_accepted": all(
                row["ops_accepted"] for row in cases
            ),
            "rejected_occurrence_op_attempt_count": sum(
                row["rejected_occurrence_op_attempts"] for row in cases
            ),
            "recovered_occurrence_op_rejection_case_count": sum(
                row["rejected_occurrence_op_attempts"] > 0
                and row["lifecycle_complete"]
                for row in cases
            ),
            "unrecovered_occurrence_op_rejection_case_count": sum(
                row["rejected_occurrence_op_attempts"] > 0
                and not row["lifecycle_complete"]
                for row in cases
            ),
            "occurrence_schema_error_count": sum(
                row["occurrence_schema_errors"] for row in cases
            ),
            "selection_required_retry_count": sum(
                row["selection_required_retries"] for row in cases
            ),
            "answer_required_retry_count": sum(
                row["answer_required_retries"] for row in cases
            ),
            "occurrence_validation_error_count": sum(
                row["occurrence_validation_errors"] for row in cases
            ),
            "unrecovered_occurrence_validation_error_case_count": sum(
                row["occurrence_validation_errors"] > 0
                and not row["lifecycle_complete"]
                for row in cases
            ),
            "terminal_occurrence_failure_count": sum(
                row["terminal_occurrence_failures"] for row in cases
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
        "eligibility_event_integrity": all(
            row["duplicate_eligibility_event_case_count"] == 0
            for row in per_arm.values()
        ),
        "treatment_exposure_integrity": all(
            per_arm[arm]["eligible_without_exposure_case_count"] == 0
            and per_arm[arm]["exposure_without_eligibility_case_count"] == 0
            and per_arm[arm]["duplicate_exposure_event_case_count"] == 0
            for arm in ("a1-flat", "a1", "a2")
        ),
        "a1_flat_same_packet_text_budget_parity": all(
            per_arm[arm]["text_budget_parity_passed"]
            for arm in ("a1-flat", "a1")
        ),
        "a0_has_no_treatment_exposure": per_arm.get("a0", {}).get(
            "exposure_event_count"
        )
        == 0,
        "a2_state_files_complete": per_arm.get("a2", {}).get(
            "state_file_count"
        )
        == int(expected_cases),
        "a2_selection_complete": (
            per_arm.get("a2", {}).get("selection_required_case_count", 0)
            > 0
            and per_arm.get("a2", {}).get("selection_missing_case_count") == 0
            and per_arm.get("a2", {}).get("selection_not_prior_case_count")
            == 0
            and per_arm.get("a2", {}).get(
                "answer_missing_after_selection_case_count"
            )
            == 0
        ),
        "no_unrecovered_occurrence_validation_errors": sum(
            row["unrecovered_occurrence_validation_error_case_count"]
            for row in per_arm.values()
        )
        == 0,
        "no_terminal_occurrence_failures": sum(
            row["terminal_occurrence_failure_count"]
            for row in per_arm.values()
        )
        == 0,
        "no_unrecovered_occurrence_op_rejections": sum(
            row["unrecovered_occurrence_op_rejection_case_count"]
            for row in per_arm.values()
        )
        == 0,
    }
    return {
        "schema_version": "MMLifelongOccurrenceCanaryAuditV3",
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
