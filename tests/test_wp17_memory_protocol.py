from __future__ import annotations

from vcah.wp17_memory_protocol import (
    WP17_2_ARMS,
    WP17_2_PROTOCOL_CONTRACT,
    build_wp17_2_protocol_manifest,
)


def test_wp17_2_manifest_freezes_segments_rotation_and_canary() -> None:
    protocol = {
        "contract": WP17_2_PROTOCOL_CONTRACT,
        "protocol_frozen_before_wp17_2_outcomes": True,
        "endpoint_values_are_structural_gates": False,
        "provenance": {
            "parent_dense_data_commit": "dense",
            "parent_dense_audit_commit": "audit",
        },
        "scope": {
            "segment_duration_sec": 30.0,
            "frame_sampling_fps": 1.0,
            "expected_segment_count": 3,
            "expected_base_calls": 12,
            "model_call_hard_cap": 14,
            "canary_case_ids": ["case-a"],
            "expected_canary_segment_count": 1,
            "expected_input_sha256": {
                "timeline": "timeline-sha",
                "dense_report": "dense-sha",
                "dense_audit": "audit-sha",
            },
            "day_test140_accessed": False,
            "week_accessed": False,
        },
        "construction_input_visibility": {
            "question": False,
            "options": False,
            "gold_answer": False,
            "official_intervals": False,
            "evaluation_aliases": False,
            "case_ids": False,
        },
        "arms": [
            {"arm": arm, "ocr_evidence": arm in {"c10", "c11"}}
            for arm in WP17_2_ARMS
        ],
        "matched_control": {
            "cross_treatment_response_replay": False,
            "c10_c11_share_frozen_ocr_packet": True,
            "state_stores_are_arm_isolated": True,
        },
        "output_contract": {},
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
                "virtual_end_sec": 75.0,
                "case_ids": ["case-a"],
            }
        ],
    }
    manifest = build_wp17_2_protocol_manifest(
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
    assert manifest["counts"] == {
        "windows": 1,
        "segments": 3,
        "arms": 4,
        "base_model_calls": 12,
        "model_call_hard_cap": 14,
        "canary_segments": 1,
    }
    assert manifest["segments"][0]["arm_execution_order"] == list(WP17_2_ARMS)
    assert manifest["segments"][1]["arm_execution_order"] == [
        "c01",
        "c10",
        "c11",
        "c00",
    ]
    assert manifest["segments"][-1]["virtual_end_sec"] == 75.0
    assert manifest["canary_segment_ids_by_evaluation_case"]["case-a"] == (
        manifest["segments"][1]["segment_id"]
    )
    assert manifest["model_calls_during_freeze"] == 0

