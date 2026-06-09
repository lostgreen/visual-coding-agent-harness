from dataclasses import asdict

from visual_coding_agent_harness.contracts import ClaimModality, EvidenceBinding


def test_evidence_binding_roundtrips_with_effective_claim_modality():
    binding = EvidenceBinding(
        evidence_id="E1",
        obs_id="obs-7",
        target_id="T1",
        subject="red car",
        relation="appears before",
        status="supported",
        mention_timestamp_sec=12.5,
        source="timeline_verifier",
        snippet="The red car appears first.",
        claim_modality=ClaimModality.NARRATED_FACT,
    )

    payload = asdict(binding)
    restored = EvidenceBinding(**payload)

    assert payload["claim_modality"] is ClaimModality.NARRATED_FACT
    assert restored == binding
