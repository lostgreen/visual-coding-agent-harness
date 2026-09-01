#!/usr/bin/env python3
"""Zero-model reachability and repair audit for WP17 slot state machines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.wp17_slot_memory import WP17_SLOT_TRANSACTION_CONTRACT, SlotMemoryState


AUDIT_CONTRACT = "WP17-slot-state-reachability-audit-v1"
V10_AUDIT_CONTRACT = "WP17-slot-state-recoverability-audit-v2"
V10_LIFECYCLE_POLICY = "WP17-slot-lifecycle-reliability-v10"
STATUSES = ("absent", "active", "closed", "archived", "evicted")
OPERATIONS = ("write", "update", "retain", "close", "archive", "evict")
V10_LEGAL_MATRIX = {
    "absent": {"write"},
    "active": {"update", "retain", "close"},
    "closed": {"retain", "close", "archive", "evict"},
    "archived": {"write", "close", "archive", "evict"},
    "evicted": {"write", "close", "archive", "evict"},
}


def build_report(*, source_commit: str) -> dict[str, Any]:
    state = SlotMemoryState("e1c2", token_budget=600)
    operations_seen: list[str] = []

    def apply(segment: str, operations: list[dict[str, Any]], evidence: str) -> dict[str, Any]:
        payload = {
            "contract": WP17_SLOT_TRANSACTION_CONTRACT,
            "observations": [
                {
                    "observation_id": "obs-1",
                    "kind": "event",
                    "fact": "A visible state transition occurs.",
                    "evidence_ids": [evidence],
                    "participants": [],
                }
            ],
            "slot_operations": operations,
            "structured_event_record": {
                "entities": [],
                "events": [],
                "state_changes": [],
                "relations": [],
                "occurrence_refs": [],
                "summary": "A visible state transition occurs.",
            },
        }
        result = state.apply(
            payload,
            segment_id=segment,
            allowed_evidence_ids=(evidence,),
        )
        operations_seen.extend(row["operation"] for row in result["lifecycle_events"])
        return result

    apply(
        "s1",
        [
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "first"},
                "observation_ids": ["obs-1"],
            }
        ],
        "frame:s1",
    )
    apply(
        "s2",
        [
            {
                "operation": "update",
                "slot": "current_activity",
                "expected_version": 1,
                "value": {"activity": "second"},
                "observation_ids": ["obs-1"],
            }
        ],
        "frame:s2",
    )
    provenance_replaced = state.records["current_activity"]["provenance"] == ["frame:s2"]
    apply(
        "s3",
        [
            {
                "operation": "retain",
                "slot": "current_activity",
                "expected_version": 2,
                "observation_ids": ["obs-1"],
            }
        ],
        "frame:s3",
    )
    apply(
        "s4",
        [
            {
                "operation": "close",
                "slot": "current_activity",
                "expected_version": 3,
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "archive",
                "slot": "current_activity",
                "expected_version": 4,
                "observation_ids": [],
            },
            {
                "operation": "evict",
                "slot": "current_activity",
                "expected_version": 5,
                "observation_ids": [],
            },
        ],
        "frame:s4",
    )
    apply(
        "s5",
        [
            {
                "operation": "write",
                "slot": "active_encounter",
                "expected_version": 0,
                "value": {"event_id": "enc-1"},
                "observation_ids": ["obs-1"],
            }
        ],
        "frame:s5",
    )
    handoff = apply(
        "s6",
        [
            {
                "operation": "close",
                "slot": "active_encounter",
                "expected_version": 1,
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "archive",
                "slot": "active_encounter",
                "expected_version": 2,
                "observation_ids": [],
            },
            {
                "operation": "write",
                "slot": "active_encounter",
                "expected_version": 3,
                "value": {"event_id": "enc-2"},
                "observation_ids": ["obs-1"],
            },
        ],
        "frame:s6",
    )
    before_implicit = state.digest()
    encounter_version_before = state.records["active_encounter"]["version"]
    implicit = apply("s7", [], "frame:s7")
    restored = SlotMemoryState.from_snapshot(state.snapshot())
    checks = {
        "all_operations_reachable": set(
            ("write", "update", "retain", "close", "archive", "evict")
        ).issubset(operations_seen),
        "same_segment_handoff_reachable": [
            row["operation"] for row in handoff["lifecycle_events"]
        ]
        == ["close", "archive", "write"],
        "implicit_retain_recorded": any(
            row["operation"] == "implicit_retain"
            for row in implicit["lifecycle_events"]
        ),
        "implicit_retain_changes_state_only_by_transaction_ledger": before_implicit
        != state.digest()
        and state.records["active_encounter"]["version"]
        == encounter_version_before,
        "changed_update_replaces_provenance": provenance_replaced,
        "snapshot_round_trip_exact": restored.digest() == state.digest(),
        "capsule_within_budget": state.capsule()["within_budget"] is True,
        "model_calls_zero": True,
        "endpoint_values_not_evaluated": True,
    }
    checks["structural_gate_passed"] = all(checks.values())
    return {
        "schema_version": "MMLifelongWP17SlotReachabilityAuditV1",
        "contract": AUDIT_CONTRACT,
        "decision": (
            "WP17_SLOT_REACHABILITY_PASSED"
            if checks["structural_gate_passed"]
            else "WP17_SLOT_REACHABILITY_FAILED"
        ),
        "source_commit": str(source_commit),
        "operation_counts": {
            operation: operations_seen.count(operation)
            for operation in sorted(set(operations_seen))
        },
        "checks": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "model_calls": 0,
        "endpoint_values_evaluated": False,
    }


def build_v10_report(*, source_commit: str) -> dict[str, Any]:
    matrix_rows = []
    for status in STATUSES:
        for operation in OPERATIONS:
            state = _state_for_status(status)
            expected_legal = operation in V10_LEGAL_MATRIX[status]
            payload = _transaction(_operation(status, operation))
            try:
                result = state.apply(
                    payload,
                    segment_id=f"matrix-{status}-{operation}",
                    allowed_evidence_ids=("frame:matrix",),
                )
            except Exception as exc:
                repair = exc.repair_contract() if hasattr(exc, "repair_contract") else {}
                details = dict(repair.get("details", {}) or {})
                repair_operations = details.get("repair_operations")
                structured = (
                    bool(repair.get("error_code"))
                    and repair.get("error_code") != "slot_validation_error"
                    and isinstance(repair_operations, list)
                )
                versions_explicit = isinstance(repair_operations, list) and all(
                    isinstance(row, Mapping) and isinstance(row.get("expected_version"), int)
                    for row in repair_operations
                )
                recoverable = False
                if versions_explicit:
                    repaired = _state_for_status(status)
                    try:
                        repaired.apply(
                            _transaction(*repair_operations),
                            segment_id=f"repair-{status}-{operation}",
                            allowed_evidence_ids=("frame:matrix",),
                        )
                    except Exception:
                        recoverable = False
                    else:
                        recoverable = True
                matrix_rows.append(
                    {
                        "status": status,
                        "operation": operation,
                        "expected_legal": expected_legal,
                        "observed_legal": False,
                        "error_code": repair.get("error_code", type(exc).__name__),
                        "structured_repair": structured,
                        "repair_versions_explicit": versions_explicit,
                        "repair_recoverable": recoverable,
                    }
                )
            else:
                lifecycle_operations = [
                    str(row.get("operation", ""))
                    for row in tuple(result.get("lifecycle_events", ()) or ())
                ]
                matrix_rows.append(
                    {
                        "status": status,
                        "operation": operation,
                        "expected_legal": expected_legal,
                        "observed_legal": True,
                        "lifecycle_operations": lifecycle_operations,
                        "idempotent_event_recorded": (
                            status in {"closed", "archived", "evicted"}
                            and operation in {"close", "archive", "evict"}
                            and any(value.startswith("redundant_") for value in lifecycle_operations)
                        ),
                    }
                )

    illegal_rows = [row for row in matrix_rows if not row["expected_legal"]]
    monotone_idempotent_rows = [
        row
        for row in matrix_rows
        if row["expected_legal"]
        and row["status"] in {"closed", "archived", "evicted"}
        and row["operation"] in {"close", "archive", "evict"}
        and V10_LEGAL_MATRIX[row["status"]].__contains__(row["operation"])
        and (
            (row["status"] == "closed" and row["operation"] == "close")
            or (row["status"] == "archived" and row["operation"] in {"close", "archive"})
            or (row["status"] == "evicted" and row["operation"] in {"close", "archive", "evict"})
        )
    ]
    sweep_state = _new_v10_state()
    sweep_state.apply(
        _transaction(
            {
                "operation": "write",
                "slot": "current_activity",
                "expected_version": 0,
                "value": {"activity": "walking"},
                "observation_ids": ["obs-1"],
            },
            {
                "operation": "close",
                "slot": "current_activity",
                "expected_version": 1,
                "observation_ids": ["obs-1"],
            },
        ),
        segment_id="sweep-close",
        allowed_evidence_ids=("frame:matrix",),
    )
    sweep_result = sweep_state.apply(
        _transaction(),
        segment_id="sweep-next",
        allowed_evidence_ids=("frame:matrix",),
    )
    sweep_passed = (
        sweep_state.records["current_activity"]["status"] == "evicted"
        and any(
            row.get("operation") == "runtime_lifecycle_sweep"
            for row in sweep_result["lifecycle_events"]
        )
    )
    checks = {
        "matrix_expected_legality_exact": all(
            row["expected_legal"] == row["observed_legal"] for row in matrix_rows
        ),
        "error_repair_contract_complete": all(
            row.get("structured_repair") is True for row in illegal_rows
        ),
        "repair_sequence_version_explicit": all(
            row.get("repair_versions_explicit") is True for row in illegal_rows
        ),
        "error_recoverability_exhaustive": all(
            row.get("repair_recoverable") is True for row in illegal_rows
        ),
        "monotone_operations_idempotent": bool(monotone_idempotent_rows)
        and all(row.get("idempotent_event_recorded") is True for row in monotone_idempotent_rows),
        "closed_slot_sweep_bounded": sweep_passed,
        "snapshot_digest_reproducible": SlotMemoryState.from_snapshot(
            sweep_state.snapshot()
        ).digest()
        == sweep_state.digest(),
        "model_calls_zero": True,
        "endpoint_values_not_evaluated": True,
    }
    checks["structural_gate_passed"] = all(checks.values())
    return {
        "schema_version": "MMLifelongWP17SlotRecoverabilityAuditV2",
        "contract": V10_AUDIT_CONTRACT,
        "decision": (
            "WP17_SLOT_V10_RECOVERABILITY_PASSED"
            if checks["structural_gate_passed"]
            else "WP17_SLOT_V10_RECOVERABILITY_FAILED"
        ),
        "source_commit": str(source_commit),
        "lifecycle_policy": V10_LIFECYCLE_POLICY,
        "matrix_counts": {
            "combinations": len(matrix_rows),
            "expected_legal": sum(row["expected_legal"] for row in matrix_rows),
            "expected_illegal": len(illegal_rows),
            "legality_mismatches": sum(
                row["expected_legal"] != row["observed_legal"] for row in matrix_rows
            ),
            "unstructured_repairs": sum(
                row.get("structured_repair") is not True for row in illegal_rows
            ),
            "nonrecoverable_repairs": sum(
                row.get("repair_recoverable") is not True for row in illegal_rows
            ),
        },
        "matrix": matrix_rows,
        "checks": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "model_calls": 0,
        "endpoint_values_evaluated": False,
    }


def _new_v10_state() -> SlotMemoryState:
    try:
        return SlotMemoryState(
            "e1c2",
            token_budget=600,
            lifecycle_policy=V10_LIFECYCLE_POLICY,
        )
    except TypeError:
        return SlotMemoryState("e1c2", token_budget=600)


def _state_for_status(status: str) -> SlotMemoryState:
    state = _new_v10_state()
    if status != "absent":
        state.records["current_activity"] = {
            "slot": "current_activity",
            "version": 5,
            "status": status,
            "value": {"activity": "old"},
            "provenance": ["frame:old"],
            "last_verified_segment_id": "old",
        }
    return state


def _operation(status: str, operation: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "operation": operation,
        "slot": "current_activity",
        "expected_version": 0 if status == "absent" else 5,
        "observation_ids": ["obs-1"] if operation in {"write", "update", "close"} else [],
    }
    if operation in {"write", "update"}:
        row["value"] = {"activity": "new"}
    return row


def _transaction(*operations: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": WP17_SLOT_TRANSACTION_CONTRACT,
        "observations": [
            {
                "observation_id": "obs-1",
                "kind": "event",
                "fact": "A visible state transition occurs.",
                "evidence_ids": ["frame:matrix"],
                "participants": [],
            }
        ],
        "slot_operations": [dict(row) for row in operations],
        "structured_event_record": {
            "entities": [],
            "events": [],
            "state_changes": [],
            "relations": [],
            "occurrence_refs": [],
            "summary": "A visible state transition occurs.",
        },
    }


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    report_path = out_root / "wp17_slot_reachability_audit.json"
    markdown_path = out_root / "wp17_slot_reachability_audit.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17 reachability audit output already exists")
    report = (
        build_v10_report(source_commit=str(args.source_commit))
        if args.mode == "exhaustive-v10"
        else build_report(source_commit=str(args.source_commit))
    )
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    markdown_path.write_text(
        "\n".join(
            (
                "# WP17 Slot Reachability Audit",
                "",
                f"- Decision: `{report['decision']}`",
                f"- Structural gate: `{str(report['structural_gate_passed']).lower()}`",
                "- Model calls: `0`",
                "- Endpoint values evaluated: `false`",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(
        "WP17_SLOT_REACHABILITY_DONE "
        f"decision={report['decision']} gate={str(report['structural_gate_passed']).lower()} "
        "model_calls=0 endpoints=false",
        flush=True,
    )
    return report_path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--mode",
        choices=("happy-path-v9", "exhaustive-v10"),
        default="happy-path-v9",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
