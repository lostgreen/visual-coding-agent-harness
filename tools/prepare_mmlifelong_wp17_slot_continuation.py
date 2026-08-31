#!/usr/bin/env python3
"""Freeze an endpoint-blind continuation plan for an incomplete WP17 run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.wp17_slot_continuation import (
    WP17_SLOT_CONTINUATION_CONTRACT,
    WP17_SLOT_CONTINUATION_PLAN_CONTRACT,
    build_continuation_entries,
)
from vcah.wp17_slot_protocol import WP17_3_MANIFEST_CONTRACT


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    parent_protocol_path = Path(args.parent_protocol_manifest)
    if file_sha256(parent_protocol_path) != str(args.expected_parent_protocol_sha256):
        raise ValueError("WP17 continuation parent protocol SHA mismatch")
    parent_protocol = _read_json(parent_protocol_path)
    if parent_protocol.get("contract") != WP17_3_MANIFEST_CONTRACT:
        raise ValueError("WP17 continuation parent protocol contract mismatch")

    parent_root = Path(args.parent_run_root)
    parent_manifest_path = parent_root / "run_manifest.json"
    parent_summary_path = parent_root / "run_summary.json"
    parent_manifest = _read_json(parent_manifest_path)
    parent_summary = _read_json(parent_summary_path)
    parent_protocol_sha = file_sha256(parent_protocol_path)
    if parent_manifest.get("protocol_manifest_sha256") != parent_protocol_sha:
        raise ValueError("WP17 continuation parent run/protocol mismatch")

    segments = tuple(dict(row) for row in parent_protocol.get("segments", ()) or ())
    parent_rows: dict[tuple[str, str], dict[str, Any]] = {}
    parent_paths: dict[tuple[str, str], Path] = {}
    for segment in segments:
        segment_id = str(segment["segment_id"])
        for arm in ("e1c0", "e1c1", "e1c2"):
            path = parent_root / "segments" / segment_id / f"{arm}.json"
            parent_rows[(segment_id, arm)] = _read_json(path)
            parent_paths[(segment_id, arm)] = path

    raw_entries = build_continuation_entries(segments, parent_rows)
    entries = []
    for raw in raw_entries:
        row = dict(raw)
        key = (str(row["segment_id"]), str(row["arm"]))
        row["parent_result_sha256"] = file_sha256(parent_paths[key])
        entries.append(row)
    rerun_count = sum(row["action"] == "rerun" for row in entries)
    reuse_count = len(entries) - rerun_count
    hard_cap = rerun_count * 3
    if hard_cap <= 0:
        raise ValueError("WP17 continuation has no rows to rerun")

    plan = {
        "schema_version": "MMLifelongWP17SlotContinuationPlanV1",
        "contract": WP17_SLOT_CONTINUATION_PLAN_CONTRACT,
        "source_commit": str(args.source_commit),
        "parent_protocol_sha256": parent_protocol_sha,
        "parent_run_manifest_sha256": file_sha256(parent_manifest_path),
        "parent_run_summary_sha256": file_sha256(parent_summary_path),
        "parent_source_commit": str(parent_manifest.get("source_commit", "")),
        "parent_model_calls": int(parent_summary.get("model_calls", 0) or 0),
        "dependency_policy": {
            "e1c0": "rerun_non_success_only",
            "e1c1": "rerun_from_first_non_success_through_window_end",
            "e1c2": "rerun_from_first_non_success_through_window_end",
        },
        "counts": {
            "results": len(entries),
            "reuse": reuse_count,
            "rerun": rerun_count,
            "continuation_model_call_hard_cap": hard_cap,
        },
        "entries": entries,
        "endpoint_values_evaluated": False,
        "model_calls": 0,
    }
    plan_path = Path(args.out_plan)
    protocol_path = Path(args.out_protocol_manifest)
    if plan_path.exists() or protocol_path.exists():
        raise FileExistsError("WP17 continuation protocol/plan output already exists")
    _write_json_atomic(plan_path, plan)

    protocol = deepcopy(parent_protocol)
    protocol["counts"]["model_call_hard_cap"] = hard_cap
    protocol["counts"]["continuation_base_model_calls"] = rerun_count
    protocol["counts"]["continuation_reuse_results"] = reuse_count
    protocol["provenance"]["source_commit"] = str(args.source_commit)
    protocol["provenance"]["continuation_scope"] = (
        "endpoint-blind completion of missing rows with deterministic stateful suffix replay"
    )
    protocol["continuation"] = {
        "contract": WP17_SLOT_CONTINUATION_CONTRACT,
        "plan_sha256": file_sha256(plan_path),
        "parent_protocol_sha256": parent_protocol_sha,
        "parent_run_manifest_sha256": file_sha256(parent_manifest_path),
        "parent_run_summary_sha256": file_sha256(parent_summary_path),
        "planned_reuse_results": reuse_count,
        "planned_rerun_results": rerun_count,
        "continuation_model_call_hard_cap": hard_cap,
        "endpoint_values_evaluated": False,
    }
    protocol["gates"] = {
        "parent_protocol_structural_gate_passed": parent_protocol.get(
            "structural_gate_passed"
        )
        is True,
        "parent_run_result_count_exact": len(entries)
        == int(parent_protocol.get("counts", {}).get("segments", 0)) * 3,
        "dependency_plan_covers_every_result": len(entries) == 363,
        "reuse_and_rerun_partition_exact": reuse_count + rerun_count == len(entries),
        "continuation_cap_is_three_attempts_per_rerun": hard_cap == rerun_count * 3,
        "endpoint_values_not_structural_gates": True,
        "question_gold_official_intervals_hidden": all(
            parent_protocol.get("construction_input_visibility", {}).get(key) is False
            for key in (
                "question",
                "options",
                "gold_answer",
                "official_intervals",
                "evaluation_aliases",
                "case_ids",
            )
        ),
        "day_test140_and_week_not_accessed": True,
        "model_calls_during_freeze_zero": True,
    }
    protocol["gates"]["structural_gate_passed"] = all(protocol["gates"].values())
    protocol["structural_gate_passed"] = protocol["gates"]["structural_gate_passed"]
    protocol["model_calls_during_freeze"] = 0
    protocol["model_calls_launched"] = False
    _write_json_atomic(protocol_path, protocol)
    print(
        "WP17_SLOT_CONTINUATION_PREPARED "
        f"reuse={reuse_count} rerun={rerun_count} cap={hard_cap} endpoints=false",
        flush=True,
    )
    return protocol_path, plan_path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-protocol-manifest", required=True)
    parser.add_argument("--expected-parent-protocol-sha256", required=True)
    parser.add_argument("--parent-run-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-protocol-manifest", required=True)
    parser.add_argument("--out-plan", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
