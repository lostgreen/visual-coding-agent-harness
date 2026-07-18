from __future__ import annotations

from vcah.semantic_evidence import canonical_fact_snapshot
from vcah.types import EvidenceRecord


def _record(transition: dict[str, object]) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="ev_octopus",
        beat_id="",
        start_sec=10.0,
        end_sec=20.0,
        modality="visual",
        pointer="virtual://octopus",
        verbatim="The food changes color.",
        frame_refs=("before.jpg", "after.jpg"),
        operation_metadata={
            "structured_parse_status": "parsed",
            "state_transitions": [transition],
        },
    )


def test_same_object_transition_requires_before_after_and_identity_witnesses() -> None:
    snapshot = canonical_fact_snapshot((_record({
        "object_hypothesis_id": "octopus_batch_01",
        "attribute_type": "surface_color",
        "raw_value_before": "grayish-white",
        "raw_value_after": "tan",
        "before_witness": [10.0, 11.0],
        "after_witness": [18.0, 19.0],
        "same_object_relation": "unknown",
        "coverage_occlusion_status": "clear",
    }),)).to_dict()

    assert snapshot["state_transitions"] == []
    assert snapshot["unresolved_state_transitions"][0]["status"] == "unknown"


def test_same_object_transition_preserves_raw_values_and_witness_ranges() -> None:
    snapshot = canonical_fact_snapshot((_record({
        "object_hypothesis_id": "octopus_batch_01",
        "attribute_type": "surface_color",
        "raw_value_before": "grayish-white",
        "raw_value_after": "tan",
        "before_witness": [10.0, 11.0],
        "after_witness": [18.0, 19.0],
        "same_object_relation": "supported",
        "coverage_occlusion_status": "clear",
    }),)).to_dict()

    transition = snapshot["state_transitions"][0]
    assert transition["status"] == "supported"
    assert transition["raw_value_before"] == "grayish-white"
    assert transition["raw_value_after"] == "tan"
    assert transition["same_object_relation"] is True
