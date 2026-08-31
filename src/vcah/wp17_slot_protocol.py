"""Frozen 120-second three-arm protocol construction for WP17 slot memory."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from vcah.wp17_slot_memory import (
    WP17_BUDGET_TOKENIZER,
    WP17_CAPSULE_PROVENANCE_CONTRACT,
    WP17_MAX_OBSERVATIONS,
    WP17_MAX_OUTPUT_JSON_CHARS,
    WP17_MAX_STRUCTURED_EVENT_ITEMS,
    WP17_SLOT_CAPSULE_CONTRACT,
    WP17_SLOT_REPAIR_CONTRACT,
    WP17_SLOT_NAMES,
    WP17_SLOT_OPERATIONS,
    WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
)
from vcah.wp17_slot_runner import (
    WP17_EVIDENCE_ALIAS_CONTRACT,
    WP17_OCR_AGGREGATION_CONTRACT,
)


WP17_3_PROTOCOL_CONTRACT = "WP17-3-slot-memory-2min-protocol-v9"
WP17_3_MANIFEST_CONTRACT = "WP17-3-slot-memory-2min-manifest-v9"
WP17_3_ARMS = ("e1c0", "e1c1", "e1c2")


def build_wp17_3_protocol_manifest(
    protocol: Mapping[str, Any],
    *,
    timeline: Mapping[str, Any],
    dense_report: Mapping[str, Any],
    dense_audit: Mapping[str, Any],
    input_sha256: Mapping[str, str],
    source_commit: str,
) -> dict[str, Any]:
    if protocol.get("contract") != WP17_3_PROTOCOL_CONTRACT:
        raise ValueError("WP17-3 protocol contract mismatch")
    scope = dict(protocol.get("scope", {}) or {})
    segment_duration = float(scope["segment_duration_sec"])
    if segment_duration != 120.0:
        raise ValueError("WP17-3 main segment duration must be exactly 120 seconds")
    arms = tuple(str(row["arm"]) for row in protocol.get("arms", ()))
    if arms != WP17_3_ARMS:
        raise ValueError("WP17-3 arm order must be E1C0/E1C1/E1C2")

    segments: list[dict[str, Any]] = []
    by_window: dict[str, list[str]] = {}
    for raw_window in tuple(timeline.get("windows", ()) or ()):
        window = dict(raw_window)
        window_id = str(window["window_id"])
        start = float(window["virtual_start_sec"])
        end = float(window["virtual_end_sec"])
        count = int(math.ceil((end - start) / segment_duration))
        ids = []
        for local_index in range(count):
            segment_start = start + local_index * segment_duration
            segment_end = min(end, segment_start + segment_duration)
            segment_id = f"wp17slot_{window_id}_seg_{local_index:04d}"
            offset = len(segments) % len(WP17_3_ARMS)
            arm_order = list(WP17_3_ARMS[offset:] + WP17_3_ARMS[:offset])
            segments.append(
                {
                    "segment_id": segment_id,
                    "window_id": window_id,
                    "segment_ordinal": len(segments),
                    "window_segment_ordinal": local_index,
                    "virtual_start_sec": round(segment_start, 3),
                    "virtual_end_sec": round(segment_end, 3),
                    "frame_sampling_fps": float(scope["frame_sampling_fps"]),
                    "max_frames": int(scope["max_frames_per_segment"]),
                    "arm_execution_order": arm_order,
                }
            )
            ids.append(segment_id)
        by_window[window_id] = ids
    trigger_id = str(scope["canary_trigger_segment_id"])
    matching = [row for row in segments if row["segment_id"] == trigger_id]
    if len(matching) != 1:
        raise ValueError("WP17-3 canary trigger must resolve to exactly one segment")
    trigger = matching[0]
    canary_ids = by_window[str(trigger["window_id"])]
    canary_count = int(scope["expected_canary_segment_count"])
    if len(canary_ids) < canary_count:
        raise ValueError("WP17-3 canary window is too short")
    trigger_index = canary_ids.index(trigger_id)
    first = max(
        0,
        min(len(canary_ids) - canary_count, trigger_index - canary_count + 1),
    )
    canary_chain = canary_ids[first : first + canary_count]

    base_calls = len(segments) * len(WP17_3_ARMS)
    expected_sha = dict(scope.get("expected_input_sha256", {}) or {})
    matched = dict(protocol.get("matched_control", {}) or {})
    state_policy = dict(protocol.get("state_policy", {}) or {})
    evidence_policy = dict(protocol.get("evidence_policy", {}) or {})
    output_contract = dict(protocol.get("output_contract", {}) or {})
    visibility = dict(protocol.get("construction_input_visibility", {}) or {})
    serialized_segments = json.dumps(segments, ensure_ascii=False)
    checks = {
        "protocol_frozen_before_wp17_3_outcomes": protocol.get(
            "protocol_frozen_before_wp17_3_outcomes"
        )
        is True,
        "timeline_structural_gate_passed": bool(timeline.get("structural_gate_passed")),
        "dense_ocr_structural_gate_passed": bool(dense_report.get("structural_gate_passed")),
        "dense_ocr_audit_gate_passed": bool(
            dense_audit.get("structural_and_promotion_gate_passed")
        ),
        "input_sha256_exact": expected_sha == dict(input_sha256),
        "three_primary_arms_exact": arms == WP17_3_ARMS,
        "segment_count_exact": len(segments) == int(scope["expected_segment_count"]),
        "base_call_count_exact": base_calls == int(scope["expected_base_calls"]),
        "full_hard_cap_covers_base": int(scope["model_call_hard_cap"]) >= base_calls,
        "full_hard_cap_exact": int(scope["model_call_hard_cap"]) == 440,
        "retry_capacity_at_least_20pct": (
            int(scope["model_call_hard_cap"]) - base_calls
        )
        / base_calls
        >= 0.20,
        "canary_chain_exact": len(canary_chain) == canary_count
        and canary_count == 5
        and all(
            int(segments[next(i for i, row in enumerate(segments) if row["segment_id"] == right)]["window_segment_ordinal"])
            == int(segments[next(i for i, row in enumerate(segments) if row["segment_id"] == left)]["window_segment_ordinal"]) + 1
            for left, right in zip(canary_chain, canary_chain[1:])
        ),
        "canary_covers_structural_failure_trigger": trigger_id in canary_chain
        and scope.get("canary_selection_kind")
        == "structural_failure_covering_chain",
        "canary_call_cap_exact": int(scope["canary_model_call_hard_cap"]) == 24,
        "question_gold_official_intervals_hidden": all(
            visibility.get(key) is False
            for key in (
                "question",
                "options",
                "gold_answer",
                "official_intervals",
                "evaluation_aliases",
                "case_ids",
            )
        ),
        "shared_current_packets": all(
            matched.get(key) is True
            for key in (
                "same_frame_packet_all_arms",
                "same_asr_packet_all_arms",
                "same_ocr_packet_all_arms",
                "state_stores_are_arm_isolated",
                "stateful_arms_run_in_timeline_order",
            )
        ),
        "cross_treatment_response_replay_disabled": matched.get(
            "cross_treatment_response_replay"
        )
        is False,
        "logical_roles_not_call_count": protocol.get("model_policy", {}).get(
            "logical_roles_require_separate_calls"
        )
        is False,
        "single_call_fusion_frozen": protocol.get("model_policy", {}).get(
            "default_call_strategy"
        )
        == "single_call_fusion",
        "token_budget_exact": int(state_policy.get("history_token_budget", 0)) == 600,
        "capsule_soft_target_exact": int(
            state_policy.get("capsule_soft_target_tokens", 0)
        )
        == 400,
        "tokenizer_exact": state_policy.get("budget_tokenizer") == WP17_BUDGET_TOKENIZER,
        "slot_schema_exact": tuple(state_policy.get("slots", ())) == WP17_SLOT_NAMES,
        "slot_operations_exact": tuple(state_policy.get("operations", ()))
        == WP17_SLOT_OPERATIONS,
        "ocr_surface_aggregation_exact": evidence_policy.get(
            "ocr_aggregation_contract"
        )
        == WP17_OCR_AGGREGATION_CONTRACT
        and evidence_policy.get("source_lineage_preserved") is True,
        "packet_local_evidence_alias_exact": evidence_policy.get(
            "evidence_alias_contract"
        )
        == WP17_EVIDENCE_ALIAS_CONTRACT
        and evidence_policy.get("aliases_canonicalized_before_persistence") is True,
        "bounded_output_contract_exact": int(
            output_contract.get("max_observations", 0)
        )
        == WP17_MAX_OBSERVATIONS
        and int(output_contract.get("max_structured_event_items_per_field", 0))
        == WP17_MAX_STRUCTURED_EVENT_ITEMS
        and int(output_contract.get("max_json_chars", 0))
        == WP17_MAX_OUTPUT_JSON_CHARS
        and output_contract.get("max_evidence_ids_per_observation") is None
        and int(output_contract.get("target_evidence_ids_per_observation", 0))
        == WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
        "deterministic_repair_contract_exact": state_policy.get(
            "slot_operation_observation_ids_reference_observation_ids"
        )
        is True
        and output_contract.get("structured_event_singletons_normalized_to_lists")
        is True
        and output_contract.get("structured_event_missing_lists_normalized_to_empty")
        is True,
        "capsule_provenance_projection_exact": state_policy.get(
            "capsule_contract"
        )
        == WP17_SLOT_CAPSULE_CONTRACT
        and state_policy.get("capsule_provenance_projection_contract")
        == WP17_CAPSULE_PROVENANCE_CONTRACT
        and state_policy.get("working_capsule_contains_raw_provenance_ids") is False
        and state_policy.get("full_provenance_preserved_in_state_and_ledger") is True,
        "retain_refresh_contract_exact": state_policy.get(
            "retain_may_refresh_provenance_without_value_change"
        )
        is True,
        "v9_lifecycle_contract_exact": state_policy.get(
            "maximum_operations_per_slot_per_transaction"
        )
        == 3
        and state_policy.get("omitted_working_slot_operation") == "implicit_retain"
        and state_policy.get("changed_update_provenance_policy") == "replace"
        and state_policy.get("transaction_abstain_preserves_state") is True
        and state_policy.get("transaction_abstain_ser_endpoint_eligible") is False
        and state_policy.get("repair_contract") == WP17_SLOT_REPAIR_CONTRACT,
        "common_history_cap_and_byte_preservation": state_policy.get(
            "c1_c2_common_history_token_limit"
        )
        == 600
        and state_policy.get("c1_tail_preserves_original_text") is True,
        "no_slot_count_cap": state_policy.get("slot_count_cap") is None,
        "archive_evict_preserve_long_term": state_policy.get(
            "archive_evict_delete_long_term_memory"
        )
        is False,
        "endpoint_values_not_structural_gates": protocol.get(
            "endpoint_values_are_structural_gates"
        )
        is False,
        "no_day_test140_or_week_outcomes": scope.get("day_test140_accessed") is False
        and scope.get("week_outcomes_accessed") is False,
        "source_paths_not_persisted": "source_path" not in serialized_segments,
        "zero_model_calls_during_freeze": True,
    }
    checks["structural_gate_passed"] = all(checks.values())
    return {
        "schema_version": "MMLifelongWP17SlotMemory2minManifestV9",
        "contract": WP17_3_MANIFEST_CONTRACT,
        "decision": (
            "WP17_3_SLOT_PROTOCOL_FROZEN"
            if checks["structural_gate_passed"]
            else "WP17_3_SLOT_PROTOCOL_FREEZE_FAILED"
        ),
        "counts": {
            "windows": len(tuple(timeline.get("windows", ()) or ())),
            "segments": len(segments),
            "arms": len(WP17_3_ARMS),
            "base_model_calls": base_calls,
            "model_call_hard_cap": int(scope["model_call_hard_cap"]),
            "canary_segments": len(canary_chain),
            "canary_base_model_calls": len(canary_chain) * len(WP17_3_ARMS),
            "canary_model_call_hard_cap": int(scope["canary_model_call_hard_cap"]),
        },
        "segments": segments,
        "canary_segment_chain": canary_chain,
        "arms": [dict(row) for row in protocol["arms"]],
        "matched_control": matched,
        "model_policy": dict(protocol["model_policy"]),
        "evidence_policy": evidence_policy,
        "state_policy": state_policy,
        "output_contract": output_contract,
        "structural_gates": list(protocol["structural_gates"]),
        "construction_endpoints": list(protocol["construction_endpoints"]),
        "development_decisions": dict(protocol["development_decisions"]),
        "construction_input_visibility": visibility,
        "gates": checks,
        "structural_gate_passed": checks["structural_gate_passed"],
        "model_calls_during_freeze": 0,
        "model_calls_launched": False,
        "provenance": {
            "source_commit": str(source_commit),
            "input_sha256": dict(input_sha256),
            "supersedes_protocol": protocol.get("provenance", {}).get(
                "supersedes_protocol"
            ),
            "repair_scope": protocol.get("provenance", {}).get("repair_scope"),
            "preserves_frozen_30s_protocol": True,
        },
    }
