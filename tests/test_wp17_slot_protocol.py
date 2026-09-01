from __future__ import annotations

from vcah.wp17_slot_memory import (
    WP17_BUDGET_TOKENIZER,
    WP17_CAPSULE_PROVENANCE_CONTRACT,
    WP17_MAX_OBSERVATIONS,
    WP17_MAX_OUTPUT_JSON_CHARS,
    WP17_MAX_STRUCTURED_EVENT_ITEMS,
    WP17_SLOT_NAMES,
    WP17_SLOT_OPERATIONS,
    WP17_SLOT_LIFECYCLE_POLICY_V10,
    WP17_SLOT_REPAIR_CONTRACT,
    WP17_SLOT_CAPSULE_CONTRACT,
    WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
)
from vcah.wp17_slot_protocol import (
    WP17_3_ARMS,
    WP17_3_PROTOCOL_CONTRACT,
    build_wp17_3_protocol_manifest,
)
from vcah.wp17_slot_runner import (
    WP17_EVIDENCE_ALIAS_CONTRACT,
    WP17_OCR_AGGREGATION_CONTRACT,
)


def test_wp17_3_manifest_freezes_three_arms_and_consecutive_canary() -> None:
    protocol = {
        "contract": WP17_3_PROTOCOL_CONTRACT,
        "protocol_frozen_before_wp17_3_outcomes": True,
        "endpoint_values_are_structural_gates": False,
        "scope": {
            "segment_duration_sec": 120.0,
            "frame_sampling_fps": 1.0,
            "max_frames_per_segment": 120,
            "expected_segment_count": 6,
            "expected_base_calls": 18,
            "model_call_hard_cap": 440,
            "canary_trigger_segment_id": "wp17slot_window-0_seg_0003",
            "canary_selection_kind": "structural_failure_covering_chain",
            "expected_canary_segment_count": 5,
            "canary_model_call_hard_cap": 24,
            "expected_input_sha256": {
                "timeline": "timeline-sha",
                "dense_report": "dense-sha",
                "dense_audit": "audit-sha",
            },
            "day_test140_accessed": False,
            "week_outcomes_accessed": False,
        },
        "construction_input_visibility": {
            "question": False,
            "options": False,
            "gold_answer": False,
            "official_intervals": False,
            "evaluation_aliases": False,
            "case_ids": False,
        },
        "arms": [{"arm": arm} for arm in WP17_3_ARMS],
        "matched_control": {
            "same_frame_packet_all_arms": True,
            "same_asr_packet_all_arms": True,
            "same_ocr_packet_all_arms": True,
            "state_stores_are_arm_isolated": True,
            "stateful_arms_run_in_timeline_order": True,
            "cross_treatment_response_replay": False,
        },
        "model_policy": {
            "logical_roles_require_separate_calls": False,
            "default_call_strategy": "single_call_fusion",
        },
        "evidence_policy": {
            "ocr_aggregation_contract": WP17_OCR_AGGREGATION_CONTRACT,
            "source_lineage_preserved": True,
            "evidence_alias_contract": WP17_EVIDENCE_ALIAS_CONTRACT,
            "aliases_canonicalized_before_persistence": True,
        },
        "state_policy": {
            "history_token_budget": 600,
            "capsule_soft_target_tokens": 400,
            "slot_operation_observation_ids_reference_observation_ids": True,
            "budget_tokenizer": WP17_BUDGET_TOKENIZER,
            "slot_count_cap": None,
            "slots": list(WP17_SLOT_NAMES),
            "operations": list(WP17_SLOT_OPERATIONS),
            "archive_evict_delete_long_term_memory": False,
            "capsule_contract": WP17_SLOT_CAPSULE_CONTRACT,
            "capsule_provenance_projection_contract": WP17_CAPSULE_PROVENANCE_CONTRACT,
            "working_capsule_contains_raw_provenance_ids": False,
            "full_provenance_preserved_in_state_and_ledger": True,
            "retain_may_refresh_provenance_without_value_change": True,
            "maximum_operations_per_slot_per_transaction": 3,
            "omitted_working_slot_operation": "implicit_retain",
            "changed_update_provenance_policy": "replace",
            "transaction_abstain_preserves_state": True,
            "transaction_abstain_ser_endpoint_eligible": False,
            "repair_contract": WP17_SLOT_REPAIR_CONTRACT,
            "c1_c2_common_history_token_limit": 600,
            "c1_tail_preserves_original_text": True,
        },
        "output_contract": {
            "structured_event_singletons_normalized_to_lists": True,
            "structured_event_missing_lists_normalized_to_empty": True,
            "max_observations": WP17_MAX_OBSERVATIONS,
            "max_structured_event_items_per_field": WP17_MAX_STRUCTURED_EVENT_ITEMS,
            "max_json_chars": WP17_MAX_OUTPUT_JSON_CHARS,
            "max_evidence_ids_per_observation": None,
            "target_evidence_ids_per_observation": WP17_TARGET_OBSERVATION_EVIDENCE_IDS,
        },
        "structural_gates": [],
        "construction_endpoints": [],
        "development_decisions": {},
    }
    timeline = {
        "structural_gate_passed": True,
        "windows": [
            {
                "window_id": "window-0",
                "virtual_start_sec": 10.0,
                "virtual_end_sec": 730.0,
                "case_ids": ["case-a"],
            }
        ],
    }
    manifest = build_wp17_3_protocol_manifest(
        protocol,
        timeline=timeline,
        dense_report={"structural_gate_passed": True},
        dense_audit={"structural_and_promotion_gate_passed": True},
        input_sha256={
            "timeline": "timeline-sha",
            "dense_report": "dense-sha",
            "dense_audit": "audit-sha",
        },
        source_commit="source",
    )

    assert manifest["structural_gate_passed"] is True
    assert manifest["counts"]["segments"] == 6
    assert manifest["counts"]["canary_base_model_calls"] == 15
    chain = manifest["canary_segment_chain"]
    ordinals = {
        row["segment_id"]: row["window_segment_ordinal"]
        for row in manifest["segments"]
    }
    assert [ordinals[value] for value in chain] == [0, 1, 2, 3, 4]
    assert manifest["segments"][1]["arm_execution_order"] == ["e1c1", "e1c2", "e1c0"]
    assert manifest["model_calls_during_freeze"] == 0

    protocol["state_policy"].update(
        {
            "lifecycle_policy": WP17_SLOT_LIFECYCLE_POLICY_V10,
            "closed_sweep_after_untouched_transactions": 1,
            "monotone_terminal_operations_idempotent": True,
            "repair_operations_include_explicit_versions": True,
            "all_illegal_transitions_have_structured_repair": True,
            "reliability_policy_variant": True,
            "raw_ser_scope_includes_transaction_abstain": True,
            "committed_memory_scope_requires_successful_transaction": True,
        }
    )
    protocol["scope"]["model_call_hard_cap"] = 500
    protocol["endpoint_analysis_policy"] = {"development_cases_burned": True}
    v10_manifest = build_wp17_3_protocol_manifest(
        protocol,
        timeline=timeline,
        dense_report={"structural_gate_passed": True},
        dense_audit={"structural_and_promotion_gate_passed": True},
        input_sha256={
            "timeline": "timeline-sha",
            "dense_report": "dense-sha",
            "dense_audit": "audit-sha",
        },
        source_commit="source-v10",
    )
    assert v10_manifest["structural_gate_passed"] is True
    assert v10_manifest["schema_version"].endswith("V10")
    assert v10_manifest["gates"]["v10_reliability_policy_exact"] is True
    assert v10_manifest["endpoint_analysis_policy"]["development_cases_burned"] is True
