from __future__ import annotations

from dataclasses import asdict

import pytest

from vcah.types import CoverageSegment, EvidenceRecord, InvestigatorOutputInvalid, validate_evidence_record


def test_evidence_record_capability_round_trip() -> None:
    record = EvidenceRecord(
        evidence_id="ev_0001",
        beat_id="bt00001",
        start_sec=12.0,
        end_sec=12.0,
        modality="visual",
        pointer="bt00001@12.000-12.000",
        verbatim="No crocodile is visible in this sampled frame.",
        temporal_scope="local_frame",
        evidence_kind="visual_observation",
        observation_polarity="negative",
        sampling_coverage="sparse",
        frame_refs=("frame_001.jpg",),
        request_ids=("win_0001",),
        coverage_manifest=(CoverageSegment("win_0001", 10.0, 20.0, "visual", 1.0),),
    )

    restored = EvidenceRecord(**asdict(record))

    assert restored == record
    assert restored.temporal_scope == "local_frame"
    assert restored.observation_polarity == "negative"
    assert restored.sampling_coverage == "sparse"


def test_asr_quote_is_manifest_complete_not_global_exhaustive() -> None:
    record = EvidenceRecord(
        evidence_id="ev_0001",
        beat_id="bt00001",
        start_sec=0.0,
        end_sec=4.0,
        modality="asr",
        pointer="bt00001@0.000-4.000",
        verbatim="The narrator mentions the bridge.",
    )

    assert record.temporal_scope == "window"
    assert record.evidence_kind == "quote"
    assert record.sampling_coverage == "complete_for_manifest"


def test_derived_evidence_requires_parent_and_coverage() -> None:
    with pytest.raises(ValueError):
        EvidenceRecord(
            evidence_id="ev_derived",
            beat_id="",
            start_sec=None,
            end_sec=None,
            modality="derived",
            pointer="aggregate",
            verbatim="Three distinct entities are observed.",
            evidence_kind="aggregate",
        )


def test_validate_evidence_record_rejects_option_judgment() -> None:
    record = EvidenceRecord(
        evidence_id="ev_0001",
        beat_id="bt00001",
        start_sec=0.0,
        end_sec=1.0,
        modality="visual",
        pointer="bt00001@0.000-1.000",
        verbatim="This supports option C.",
        attestation_model="vision",
    )

    with pytest.raises(InvestigatorOutputInvalid):
        validate_evidence_record(record)
