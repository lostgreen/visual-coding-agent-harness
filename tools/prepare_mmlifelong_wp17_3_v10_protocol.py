#!/usr/bin/env python3
"""Prepare the endpoint-blind WP17-3 v10 reliability-policy protocol."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from vcah.occurrence_negative_sidecar import file_sha256
from vcah.wp17_slot_memory import (
    WP17_SLOT_LIFECYCLE_POLICY_V10,
    WP17_SLOT_REPAIR_CONTRACT,
)
from vcah.wp17_slot_protocol import WP17_3_PROTOCOL_CONTRACT


PREPARATION_CONTRACT = "WP17-3-v10-reliability-policy-preparation-v1"


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
        raise ValueError("v10 reliability protocol expects the frozen 121-segment scope")
    if sum(row.get("segment_id") == trigger_segment_id for row in segments) != 1:
        raise ValueError("v10 structural failure trigger is not unique")
    frame_fps = {float(row["frame_sampling_fps"]) for row in segments}
    max_frames = {int(row["max_frames"]) for row in segments}
    if len(frame_fps) != 1 or len(max_frames) != 1:
        raise ValueError("v10 reliability protocol requires one frozen frame policy")

    state_policy = deepcopy(dict(prior["state_policy"]))
    state_policy.update(
        {
            "lifecycle_policy": WP17_SLOT_LIFECYCLE_POLICY_V10,
            "closed_sweep_after_untouched_transactions": 1,
            "monotone_terminal_operations_idempotent": True,
            "repair_operations_include_explicit_versions": True,
            "all_illegal_transitions_have_structured_repair": True,
            "reliability_policy_variant": True,
            "repair_contract": WP17_SLOT_REPAIR_CONTRACT,
            "maximum_attempts_per_result": 3,
            "transaction_abstain_preserves_state": True,
            "transaction_abstain_ser_endpoint_eligible": False,
            "raw_ser_scope_includes_transaction_abstain": True,
            "committed_memory_scope_requires_successful_transaction": True,
        }
    )
    endpoint_analysis_policy = {
        "raw_ser_coverage": {
            "includes_transaction_abstain": True,
            "source": "validated_base_structured_event_record",
            "interpretation": "exploratory_perception_output",
        },
        "committed_memory_coverage": {
            "requires_successful_slot_transaction": True,
            "source": "committed_structured_event_record_and_slot_state",
            "interpretation": "end_to_end_memory_output",
        },
        "legacy_no_go_decision_scope": "committed_memory_coverage",
        "primary_semantic_evaluator": "arm_blind_pre_frozen_judge",
        "frozen_lexical_evaluator_role": "secondary_diagnostic",
        "development_case_count": 11,
        "development_cases_burned": True,
        "all_new_endpoint_results_exploratory": True,
        "day_test140_accessed": False,
        "week_outcomes_accessed": False,
    }
    endpoints = list(prior.get("construction_endpoints", ()))
    for name in (
        "raw_ser_coverage",
        "committed_memory_coverage",
        "runtime_lifecycle_sweep_rate",
        "redundant_operation_rate",
        "slot_transaction_abstain_rate",
        "ser_structural_item_counts",
        "occurrence_counter_activation_rate",
    ):
        if name not in endpoints:
            endpoints.append(name)
    structural_gates = list(prior.get("structural_gates", ()))
    structural_gates.extend(
        value
        for value in (
            "exhaustive 5-status x 6-operation reachability audit passes",
            "every illegal transition has a literal versioned repair sequence",
            "monotone terminal operations are version-stable idempotent no-ops",
            "untouched closed slots are swept after exactly one later transaction",
            "canary E1C2 transaction abstention count is zero",
            "full E1C2 transaction abstention rate is below five percent",
            "raw and committed endpoint scopes remain distinct",
            "development endpoints are exploratory and never structural gates",
        )
        if value not in structural_gates
    )
    decisions = deepcopy(dict(prior.get("development_decisions", {}) or {}))
    decisions.update(
        {
            "v10_scope": "runtime_reliability_policy_variant",
            "v10_is_pure_bug_fix": False,
            "dev_case_outcomes_already_used": True,
            "new_endpoint_claim_status": "exploratory_only",
        }
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
            "model_call_hard_cap": 500,
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
        "construction_endpoints": endpoints,
        "endpoint_analysis_policy": endpoint_analysis_policy,
        "development_decisions": decisions,
        "provenance": {
            "preparation_contract": PREPARATION_CONTRACT,
            "supersedes_protocol": str(prior.get("contract", "")),
            "superseded_manifest_sha256": str(prior_manifest_sha256),
            "repair_scope": (
                "endpoint-blind runtime transaction recoverability and lifecycle reliability; "
                "frozen evidence, model, segment scope, and three-arm treatments preserved"
            ),
        },
    }


def run(args: argparse.Namespace) -> Path:
    prior_path = Path(args.prior_manifest)
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError("WP17 v10 protocol output already exists")
    protocol = build_protocol(
        _read_json(prior_path),
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
        "WP17_V10_PROTOCOL_PREPARED segments=121 base_calls=363 full_cap=500 "
        "canary_segments=5 canary_cap=24 model_calls=0 endpoints=false",
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
