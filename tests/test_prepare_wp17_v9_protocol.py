from __future__ import annotations

import importlib.util
from pathlib import Path

from vcah.wp17_slot_memory import (
    WP17_SLOT_CAPSULE_CONTRACT,
    WP17_SLOT_REPAIR_CONTRACT,
)
from vcah.wp17_slot_protocol import WP17_3_PROTOCOL_CONTRACT


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "prepare_mmlifelong_wp17_3_v9_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_wp17_v9", MODULE_PATH)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_v9_preparation_preserves_scope_but_replaces_lifecycle_contract() -> None:
    prior = {
        "contract": "WP17-3-slot-memory-2min-manifest-v8",
        "structural_gate_passed": True,
        "segments": [
            {
                "segment_id": f"wp17slot_wp17_window_0000_seg_{index:04d}",
                "frame_sampling_fps": 1.0,
                "max_frames": 120,
            }
            for index in range(121)
        ],
        "state_policy": {"history_token_budget": 600},
        "construction_input_visibility": {
            "question": False,
            "options": False,
            "gold_answer": False,
            "official_intervals": False,
            "evaluation_aliases": False,
            "case_ids": False,
        },
        "arms": [{"arm": arm} for arm in ("e1c0", "e1c1", "e1c2")],
        "matched_control": {"same_frame_packet_all_arms": True},
        "model_policy": {"actual_model": "model"},
        "evidence_policy": {"ocr_source": "frozen"},
        "output_contract": {"max_observations": 16},
        "construction_endpoints": ["anchor_representation_coverage"],
        "development_decisions": {},
        "provenance": {"input_sha256": {"timeline": "sha"}},
    }

    protocol = PREPARE.build_protocol(
        prior,
        trigger_segment_id="wp17slot_wp17_window_0000_seg_0003",
        prior_manifest_sha256="prior-sha",
    )

    assert protocol["contract"] == WP17_3_PROTOCOL_CONTRACT
    assert protocol["scope"]["expected_segment_count"] == 121
    assert protocol["scope"]["model_call_hard_cap"] == 440
    assert protocol["scope"]["expected_canary_segment_count"] == 5
    assert protocol["state_policy"]["capsule_contract"] == WP17_SLOT_CAPSULE_CONTRACT
    assert protocol["state_policy"]["repair_contract"] == WP17_SLOT_REPAIR_CONTRACT
    assert protocol["state_policy"]["omitted_working_slot_operation"] == "implicit_retain"
    assert "slot_transaction_abstain_rate" in protocol["construction_endpoints"]
