from __future__ import annotations

from vcah.wp17_slot_memory import (
    WP17_BUDGET_TOKENIZER,
    WP17_CAPSULE_PROVENANCE_CONTRACT,
    WP17_MAX_OBSERVATIONS,
    WP17_MAX_OUTPUT_JSON_CHARS,
    WP17_MAX_STRUCTURED_EVENT_ITEMS,
    WP17_SLOT_NAMES,
    WP17_SLOT_OPERATIONS,
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
            "expected_segment_count": 4,
            "expected_base_calls": 12,
            "model_call_hard_cap": 14,
            "canary_chain_case_id": "case-a",
            "expected_canary_segment_count": 3,
            "canary_model_call_hard_cap": 12,
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
        },
        "output_contract": {
            "structured_event_singletons_normalized_to_lists": True,
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
                "virtual_end_sec": 490.0,
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
    assert manifest["counts"]["segments"] == 4
    assert manifest["counts"]["canary_base_model_calls"] == 9
    chain = manifest["canary_segment_chain"]
    ordinals = {
        row["segment_id"]: row["window_segment_ordinal"]
        for row in manifest["segments"]
    }
    assert [ordinals[value] for value in chain] == [1, 2, 3]
    assert manifest["segments"][1]["arm_execution_order"] == ["e1c1", "e1c2", "e1c0"]
    assert manifest["model_calls_during_freeze"] == 0
