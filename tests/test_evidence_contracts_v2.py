from __future__ import annotations

import pytest

from visual_coding_agent_harness.contracts.evidence import EvidenceRecord


def test_evidence_record_requires_verbatim_and_frame_pointer() -> None:
    with pytest.raises(ValueError, match="verbatim"):
        EvidenceRecord(
            evidence_id="ev1",
            claim="red car",
            stance="supports",
            modality="asr",
            time_sec=1.0,
            pointer="cue1",
            verbatim=" ",
            query_id="q1",
            beat_id="bt00001",
        )

    with pytest.raises(ValueError, match="image pointer"):
        EvidenceRecord(
            evidence_id="ev2",
            claim="red car",
            stance="supports",
            modality="frame",
            time_sec=1.0,
            pointer="frame.json",
            verbatim="A red car is visible.",
            query_id="q1",
            beat_id="bt00001",
        )

    record = EvidenceRecord(
        evidence_id="ev3",
        claim="red car",
        stance="supports",
        modality="frame",
        time_sec=1.0,
        pointer="frame.jpg",
        verbatim="A red car is visible.",
        query_id="q1",
        beat_id="bt00001",
    )

    assert EvidenceRecord.from_dict(record.to_dict()) == record
