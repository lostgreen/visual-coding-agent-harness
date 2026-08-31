#!/usr/bin/env python3
"""Prepare the endpoint-blind WP17-3 v9 repair protocol from v8 metadata."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.wp17_slot_memory import (
    WP17_CAPSULE_PROVENANCE_CONTRACT,
    WP17_SLOT_CAPSULE_CONTRACT,
    WP17_SLOT_REPAIR_CONTRACT,
)
from vcah.wp17_slot_protocol import WP17_3_PROTOCOL_CONTRACT


PREPARATION_CONTRACT = "WP17-3-v9-endpoint-blind-repair-preparation-v1"


def build_protocol(
    prior: Mapping[str, Any],
    *,
    trigger_segment_id: str,
    prior_manifest_sha256: str,
) -> dict[str, Any]:
    if prior.get("structural_gate_passed") is not True:
        raise ValueError("prior WP17-3 manifest did not pass its structural gate")
    segments = tuple(dict(row) for row in prior.get("segments", ()) or ())
    if len(segments) != 121:
        raise ValueError("v9 repair expects the frozen 121-segment scope")
    if sum(row.get("segment_id") == trigger_segment_id for row in segments) != 1:
        raise ValueError("v9 structural failure trigger is not unique")
    frame_fps = {float(row["frame_sampling_fps"]) for row in segments}
    max_frames = {int(row["max_frames"]) for row in segments}
    if len(frame_fps) != 1 or len(max_frames) != 1:
        raise ValueError("v9 repair requires one frozen frame policy")

    state_policy = deepcopy(dict(prior["state_policy"]))
    state_policy.update(
        {
            "capsule_contract": WP17_SLOT_CAPSULE_CONTRACT,
            "capsule_provenance_projection_contract": WP17_CAPSULE_PROVENANCE_CONTRACT,
            "maximum_operations_per_slot_per_transaction": 3,
            "omitted_working_slot_operation": "implicit_retain",
            "changed_update_provenance_policy": "replace",
            "transaction_abstain_preserves_state": True,
            "transaction_abstain_ser_endpoint_eligible": False,
            "repair_contract": WP17_SLOT_REPAIR_CONTRACT,
            "maximum_attempts_per_result": 3,
            "c1_c2_common_history_token_limit": 600,
            "c1_tail_preserves_original_text": True,
            "model_visible_capsule_excludes_provenance_summary": True,
            "capsule_overhead_share_is_diagnostic_not_gate": True,
        }
    )
    structural_gates = [
        "exact consecutive 5-segment structural-failure-covering canary chain",
        "question/gold/official-interval/case-id blind",
        "actual model/provider/config exact",
        "current packets are identical across arms",
        "all cited evidence resolves inside the current packet",
        "slot lifecycle reachability audit passes with zero model calls",
        "up to three ordered operations per slot replay atomically",
        "omitted working slots become version-stable implicit retain events",
        "changed-value UPDATE replaces provenance",
        "C1 suffix preserves original text under the shared 600-token cap",
        "structured semantic and serialization repair paths are distinct",
        "transaction abstain leaves slot state unchanged and marks SER endpoint-ineligible",
        "slot transactions and lifecycle independently replay",
        "capsule remains within 600 tokens",
        "zero terminal result failures",
        "endpoint values are not gates",
    ]
    endpoint_names = [
        str(value) if isinstance(value, str) else str(value.get("name", ""))
        for value in tuple(prior.get("construction_endpoints", ()) or ())
    ]
    additional_endpoints = (
        "implicit_retain_rate",
        "illegal_operation_rate",
        "slot_transaction_abstain_rate",
        "ser_endpoint_ineligible_rate",
        "realized_history_context_tokens",
        "capsule_overhead_share_diagnostic",
    )
    construction_endpoints = list(prior.get("construction_endpoints", ()))
    construction_endpoints.extend(
        name for name in additional_endpoints if name not in endpoint_names
    )
    return {
        "contract": WP17_3_PROTOCOL_CONTRACT,
        "protocol_frozen_before_wp17_3_outcomes": True,
        "endpoint_values_are_structural_gates": False,
        "scope": {
            "segment_duration_sec": 120.0,
            "frame_sampling_fps": next(iter(frame_fps)),
            "max_frames_per_segment": next(iter(max_frames)),
            "expected_segment_count": 121,
            "expected_base_calls": 363,
            "model_call_hard_cap": 440,
            "canary_trigger_segment_id": str(trigger_segment_id),
            "canary_selection_kind": "structural_failure_covering_chain",
            "expected_canary_segment_count": 5,
            "canary_model_call_hard_cap": 24,
            "expected_input_sha256": deepcopy(
                dict(prior.get("provenance", {}).get("input_sha256", {}))
            ),
            "day_test140_accessed": False,
            "week_outcomes_accessed": False,
        },
        "construction_input_visibility": deepcopy(
            dict(prior["construction_input_visibility"])
        ),
        "arms": deepcopy(list(prior["arms"])),
        "matched_control": deepcopy(dict(prior["matched_control"])),
        "model_policy": deepcopy(dict(prior["model_policy"])),
        "evidence_policy": deepcopy(dict(prior["evidence_policy"])),
        "state_policy": state_policy,
        "output_contract": deepcopy(dict(prior["output_contract"])),
        "structural_gates": structural_gates,
        "construction_endpoints": construction_endpoints,
        "development_decisions": deepcopy(dict(prior["development_decisions"])),
        "provenance": {
            "preparation_contract": PREPARATION_CONTRACT,
            "supersedes_protocol": str(prior.get("contract", "")),
            "superseded_manifest_sha256": str(prior_manifest_sha256),
            "repair_scope": (
                "endpoint-blind D1-D8 lifecycle expressibility, repair reliability, "
                "and C1 baseline fidelity; frozen evidence/model/scope preserved"
            ),
        },
    }


def run(args: argparse.Namespace) -> Path:
    prior_path = Path(args.prior_manifest)
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError("WP17 v9 protocol output already exists")
    prior = _read_json(prior_path)
    protocol = build_protocol(
        prior,
        trigger_segment_id=str(args.trigger_segment_id),
        prior_manifest_sha256=file_sha256(prior_path),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    temporary.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(out_path)
    print(
        "WP17_V9_PROTOCOL_PREPARED "
        "segments=121 base_calls=363 full_cap=440 canary_segments=5 canary_cap=24 "
        "model_calls=0 endpoints=false",
        flush=True,
    )
    return out_path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--trigger-segment-id", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
