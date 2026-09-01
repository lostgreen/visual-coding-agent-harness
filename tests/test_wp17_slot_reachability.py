from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_mmlifelong_wp17_slot_reachability.py"
)
SPEC = importlib.util.spec_from_file_location("wp17_slot_reachability", MODULE_PATH)
assert SPEC and SPEC.loader
REACHABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REACHABILITY)


def test_v9_state_machine_reachability_gate_passes_without_model_calls() -> None:
    report = REACHABILITY.build_report(source_commit="test")

    assert report["structural_gate_passed"] is True
    assert report["model_calls"] == 0
    assert report["endpoint_values_evaluated"] is False
    assert report["operation_counts"]["implicit_retain"] == 1
    assert report["checks"]["same_segment_handoff_reachable"] is True


def test_v10_exhaustive_recoverability_gate_passes_without_model_calls() -> None:
    report = REACHABILITY.build_v10_report(source_commit="test")

    assert report["structural_gate_passed"] is True
    assert report["matrix_counts"] == {
        "combinations": 30,
        "expected_legal": 16,
        "expected_illegal": 14,
        "legality_mismatches": 0,
        "unstructured_repairs": 0,
        "nonrecoverable_repairs": 0,
    }
    assert report["checks"]["monotone_operations_idempotent"] is True
    assert report["checks"]["closed_slot_sweep_bounded"] is True
    assert report["model_calls"] == 0
