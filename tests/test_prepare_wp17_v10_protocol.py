from __future__ import annotations

import importlib.util
from pathlib import Path

from vcah.wp17_slot_memory import WP17_SLOT_LIFECYCLE_POLICY_V10
from vcah.wp17_slot_protocol import WP17_3_PROTOCOL_CONTRACT


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "prepare_mmlifelong_wp17_3_v10_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_wp17_v10", MODULE_PATH)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_v10_preparation_freezes_reliability_variant_and_endpoint_scopes() -> None:
    segments = [
        {
            "segment_id": f"wp17slot_window_seg_{index:04d}",
            "frame_sampling_fps": 1.0,
            "max_frames": 120,
        }
        for index in range(121)
    ]
    prior = {
        "contract": "WP17-3-slot-memory-2min-manifest-v9",
        "structural_gate_passed": True,
        "segments": segments,
        "state_policy": {
            "history_token_budget": 600,
            "transaction_abstain_ser_endpoint_eligible": False,
        },
        "construction_input_visibility": {"question": False},
        "arms": [{"arm": arm} for arm in ("e1c0", "e1c1", "e1c2")],
        "matched_control": {"same_frame_packet_all_arms": True},
        "model_policy": {"actual_model": "model"},
        "evidence_policy": {"ocr_source": "frozen"},
        "output_contract": {"max_observations": 16},
        "structural_gates": [],
        "construction_endpoints": ["anchor_representation_coverage"],
        "development_decisions": {},
        "provenance": {"input_sha256": {"timeline": "sha"}},
    }

    protocol = PREPARE.build_protocol(
        prior,
        trigger_segment_id="wp17slot_window_seg_0003",
        prior_manifest_sha256="prior-sha",
    )

    assert protocol["contract"] == WP17_3_PROTOCOL_CONTRACT
    assert protocol["scope"]["model_call_hard_cap"] == 500
    state_policy = protocol["state_policy"]
    assert state_policy["lifecycle_policy"] == WP17_SLOT_LIFECYCLE_POLICY_V10
    assert state_policy["closed_sweep_after_untouched_transactions"] == 1
    assert state_policy["monotone_terminal_operations_idempotent"] is True
    assert state_policy["repair_operations_include_explicit_versions"] is True
    assert protocol["endpoint_analysis_policy"]["development_cases_burned"] is True
    assert (
        protocol["endpoint_analysis_policy"]["raw_ser_coverage"][
            "includes_transaction_abstain"
        ]
        is True
    )
    assert protocol["development_decisions"]["v10_is_pure_bug_fix"] is False
    assert "runtime_lifecycle_sweep_rate" in protocol["construction_endpoints"]
