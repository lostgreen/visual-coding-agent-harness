from __future__ import annotations

import pytest

from vcah.types import InvestigatorOutputInvalid, validate_investigator_output


def test_investigator_output_accepts_evidence_records_only() -> None:
    validate_investigator_output(
        {
            "evidence": [
                {
                    "evidence_id": "ev_0001",
                    "beat_id": "bt00001",
                    "start_sec": 1.0,
                    "end_sec": 1.0,
                    "modality": "visual",
                    "pointer": "bt00001@1.000-1.000",
                    "verbatim": "A boat is visible on the water.",
                    "temporal_scope": "local_frame",
                    "evidence_kind": "visual_observation",
                    "observation_polarity": "positive",
                    "sampling_coverage": "sparse",
                    "frame_refs": ["frame_001.jpg"],
                }
            ]
        }
    )


def test_investigator_output_rejects_option_status_table() -> None:
    with pytest.raises(InvestigatorOutputInvalid):
        validate_investigator_output({"A": {"status": "supported", "support": [], "contradict": []}})


def test_investigator_output_rejects_mixed_evidence_and_option_judgment() -> None:
    with pytest.raises(InvestigatorOutputInvalid):
        validate_investigator_output(
            {
                "evidence": [
                    {
                        "evidence_id": "ev_0001",
                        "beat_id": "bt00001",
                        "start_sec": 1.0,
                        "end_sec": 1.0,
                        "modality": "visual",
                        "pointer": "bt00001@1.000-1.000",
                        "verbatim": "A boat is visible.",
                    }
                ],
                "status": "supported",
            }
        )


def test_investigator_output_rejects_unknown_nested_evidence_keys() -> None:
    with pytest.raises(InvestigatorOutputInvalid):
        validate_investigator_output(
            {
                "evidence": [
                    {
                        "evidence_id": "ev_0001",
                        "beat_id": "bt00001",
                        "start_sec": 1.0,
                        "end_sec": 1.0,
                        "modality": "visual",
                        "pointer": "bt00001@1.000-1.000",
                        "verbatim": "A boat is visible.",
                        "status": "supported",
                    }
                ]
            }
        )
