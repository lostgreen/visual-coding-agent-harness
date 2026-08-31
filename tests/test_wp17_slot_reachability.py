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
