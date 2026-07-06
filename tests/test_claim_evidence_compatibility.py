from __future__ import annotations

from vcah.types import Claim, ClaimContract, CoverageSegment, EvidenceRecord
from vcah.verifier import verify_claim


def test_local_visual_absence_cannot_refute_broader_narrative_claim() -> None:
    claim = Claim(
        "cl_croc",
        "C",
        "The material identifies crocodiles as a major obstacle to diving.",
        contract=ClaimContract(
            required_scope="multi_window",
            observation_target="relation",
            required_observability=("asr",),
        ),
    )
    evidence = EvidenceRecord(
        evidence_id="ev_no_croc",
        beat_id="bt00001",
        start_sec=1240.2,
        end_sec=1240.2,
        modality="visual",
        pointer="bt00001@1240.200-1240.200",
        verbatim="No crocodile is visible in this sampled frame.",
        temporal_scope="local_frame",
        evidence_kind="visual_observation",
        observation_polarity="negative",
        sampling_coverage="sparse",
        frame_refs=("frame_1240.jpg",),
    )

    verdict = verify_claim(claim, (evidence,))

    assert verdict.status == "unknown"
    assert verdict.reason == "observability_mismatch"


def test_proxy_background_cannot_support_strong_relation_claim() -> None:
    claim = Claim(
        "cl_location",
        "D",
        "Unknown accurate location was the most important diving difficulty.",
        contract=ClaimContract(required_scope="multi_window", observation_target="relation", required_observability=("visual",)),
    )
    evidence = EvidenceRecord(
        evidence_id="ev_map",
        beat_id="bt00002",
        start_sec=2100.0,
        end_sec=2100.0,
        modality="visual",
        pointer="bt00002@2100.000-2100.000",
        verbatim="A map graphic shows a broad blue area without a clearly pinpointed fort location.",
        temporal_scope="multi_window",
        evidence_kind="visual_observation",
        observation_polarity="positive",
        sampling_coverage="sparse",
        request_ids=("win_a", "win_b"),
    )

    verdict = verify_claim(claim, (evidence,))

    assert verdict.status == "unknown"
    assert verdict.entailment_kind == "proxy"
    assert verdict.reason == "proxy_evidence_cannot_support_claim"


def test_sampled_entity_observation_cannot_support_exact_full_video_count() -> None:
    claim = Claim(
        "cl_count",
        "B",
        "There are exactly three distinct commentators across the full video.",
        contract=ClaimContract(
            required_scope="full_video",
            quantifier="distinct_count",
            observation_target="entity",
            aggregation="deduplicate",
            required_observability=("visual",),
        ),
    )
    evidence = EvidenceRecord(
        evidence_id="ev_person",
        beat_id="bt00003",
        start_sec=342.0,
        end_sec=342.0,
        modality="visual",
        pointer="bt00003@342.000-342.000",
        verbatim="A single talking-head commentator is visible.",
        temporal_scope="local_frame",
        evidence_kind="entity_observation",
        observation_polarity="positive",
        sampling_coverage="sparse",
        frame_refs=("frame_342.jpg",),
    )

    verdict = verify_claim(claim, (evidence,))

    assert verdict.status == "unknown"
    assert verdict.reason == "aggregation_or_coverage_missing"


def test_derived_aggregate_can_support_count_when_contract_is_satisfied() -> None:
    claim = Claim(
        "cl_count",
        "B",
        "There are exactly three distinct commentators across the full video.",
        contract=ClaimContract(
            required_scope="full_video",
            quantifier="distinct_count",
            observation_target="entity",
            aggregation="deduplicate",
            required_observability=("visual",),
        ),
    )
    evidence = EvidenceRecord(
        evidence_id="ev_aggregate",
        beat_id="",
        start_sec=None,
        end_sec=None,
        modality="derived",
        pointer="entity_count",
        verbatim="Three distinct commentators are observed across the full video.",
        temporal_scope="full_video",
        evidence_kind="aggregate",
        observation_polarity="positive",
        sampling_coverage="complete_for_manifest",
        parent_evidence_ids=("ev_p1", "ev_p2", "ev_p3"),
        coverage_manifest=(CoverageSegment("win_full", 0.0, 3000.0, "visual", 1.0),),
    )

    verdict = verify_claim(claim, (evidence,))

    assert verdict.status == "supported"
    assert verdict.support_evidence_ids == ("ev_aggregate",)
