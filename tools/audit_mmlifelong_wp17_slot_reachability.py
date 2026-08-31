#!/usr/bin/env python3
"""Zero-model reachability audit for the WP17 v9 slot state machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.wp17_slot_memory import WP17_SLOT_TRANSACTION_CONTRACT, SlotMemoryState


AUDIT_CONTRACT = "WP17-slot-state-reachability-audit-v1"


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


def run(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root)
    report_path = out_root / "wp17_slot_reachability_audit.json"
    markdown_path = out_root / "wp17_slot_reachability_audit.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError("WP17 reachability audit output already exists")
    report = build_report(source_commit=str(args.source_commit))
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
