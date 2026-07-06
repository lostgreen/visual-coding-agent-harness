from __future__ import annotations

from dataclasses import asdict

import pytest

from vcah.types import Claim, ClaimContract, CoverageSegment, EvidenceRecord, InvestigatorOutputInvalid, validate_evidence_record
from vcah.verifier import verify_claim


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
    assert record.sampling_coverage == "unknown"


def test_asr_quote_with_matching_manifest_is_complete_for_manifest() -> None:
    record = EvidenceRecord(
        evidence_id="ev_0001",
        beat_id="bt00001",
        start_sec=0.0,
        end_sec=4.0,
        modality="asr",
        pointer="bt00001@0.000-4.000",
        verbatim="The narrator mentions the bridge.",
        coverage_manifest=(CoverageSegment("win_0001", 0.0, 4.0, "asr", 1.0),),
    )

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


def test_required_observability_defaults_to_all_modalities() -> None:
    evidence = (
        EvidenceRecord(
            evidence_id="ev_0001",
            beat_id="bt00001",
            start_sec=0.0,
            end_sec=4.0,
            modality="asr",
            pointer="bt00001@0.000-4.000",
            verbatim="The narrator mentions the bridge.",
            coverage_manifest=(CoverageSegment("win_0001", 0.0, 4.0, "asr", 1.0),),
        ),
    )
    claim = Claim(
        "cl_01",
        "A",
        "The narrator mentions the bridge.",
        contract=ClaimContract(required_observability=("asr", "visual")),
    )

    verdict = verify_claim(claim, evidence)

    assert verdict.status == "unknown"
    assert "observability_mismatch" in verdict.capability_checks


def test_observability_any_accepts_one_required_modality() -> None:
    evidence = (
        EvidenceRecord(
            evidence_id="ev_0001",
            beat_id="bt00001",
            start_sec=0.0,
            end_sec=4.0,
            modality="asr",
            pointer="bt00001@0.000-4.000",
            verbatim="The narrator mentions the bridge.",
            coverage_manifest=(CoverageSegment("win_0001", 0.0, 4.0, "asr", 1.0),),
        ),
    )
    claim = Claim(
        "cl_01",
        "A",
        "The narrator mentions the bridge.",
        contract=ClaimContract(required_observability=("asr", "visual"), observability_mode="any"),
    )

    verdict = verify_claim(claim, evidence)

    assert verdict.status == "supported"


def test_full_video_relation_claim_rejects_window_evidence() -> None:
    evidence = (
        EvidenceRecord(
            evidence_id="ev_0001",
            beat_id="bt00001",
            start_sec=0.0,
            end_sec=4.0,
            modality="asr",
            pointer="bt00001@0.000-4.000",
            verbatim="The most important obstacle is mentioned.",
            coverage_manifest=(CoverageSegment("win_0001", 0.0, 4.0, "asr", 1.0),),
        ),
    )
    claim = Claim(
        "cl_01",
        "A",
        "The most important obstacle is mentioned.",
        contract=ClaimContract(required_scope="full_video", observation_target="relation", required_observability=("asr",)),
    )

    verdict = verify_claim(claim, evidence)

    assert verdict.status == "unknown"
    assert verdict.reason == "aggregation_or_coverage_missing"
