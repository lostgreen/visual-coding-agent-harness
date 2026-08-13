from __future__ import annotations

from copy import deepcopy

import pytest

from vcah.caption_occurrence import build_caption_occurrence_set
from vcah.caption_schema import stable_digest
from vcah.occurrence_agent import (
    OccurrencePacketTransform,
    assert_no_oracle_packet,
    validate_occurrence_method_configuration,
)


def _packet() -> dict:
    hits = [
        {
            "passage_id": "p1",
            "caption_id": "c1",
            "rank": 1,
            "virtual_start_sec": 10.0,
            "virtual_end_sec": 20.0,
            "fused_score": 0.9,
            "text": "The player enters a shop and talks to the merchant.",
            "metadata": {
                "source_video_ids": ["v1"],
                "source_segments": ["s1"],
                "query_matches": [{"query": "enters the shop", "rank": 1}],
            },
        },
        {
            "passage_id": "p2",
            "caption_id": "c2",
            "rank": 2,
            "virtual_start_sec": 500.0,
            "virtual_end_sec": 510.0,
            "fused_score": 0.8,
            "text": "Later, the player enters another shop after a battle.",
            "metadata": {
                "source_video_ids": ["v2"],
                "source_segments": ["s2"],
                "query_matches": [{"query": "after the battle", "rank": 2}],
            },
        },
    ]
    return {
        "queries": ["enters shop", "after battle"],
        "index_digest": "index-digest",
        "query_fingerprint": "query-digest",
        "hits": hits,
        "occurrence_set": build_caption_occurrence_set(hits),
        "rendered": "rendered locator hits",
    }


def test_a0_is_an_identity_transform_with_no_oracle_audit(tmp_path) -> None:
    packet = _packet()
    before = stable_digest(packet)
    transform = OccurrencePacketTransform(
        arm="a0", audit_path=tmp_path / "audit.json"
    )

    transformed = transform(packet)

    assert transformed is packet
    assert stable_digest(transformed) == before
    assert transform.audit["no_oracle_runtime_gate_passed"] is True
    assert transform.audit["retrieval_parity_passed"] is True
    assert transform.audit["candidate_card_counts"] == [0]


def test_a1_adds_bounded_cards_without_changing_retrieval(tmp_path) -> None:
    packet = _packet()
    original = deepcopy(packet)
    transform = OccurrencePacketTransform(
        arm="a1", audit_path=tmp_path / "audit.json"
    )

    transformed = transform(packet)

    assert packet == original
    assert transformed["hits"] == original["hits"]
    assert (
        transformed["occurrence_set"]["candidates"]
        == original["occurrence_set"]["candidates"]
    )
    cards = transformed["occurrence_set"]["candidate_cards"]
    assert [card["occurrence_id"] for card in cards] == [
        candidate["occurrence_id"]
        for candidate in original["occurrence_set"]["candidates"]
    ]
    assert cards[0]["representative_passages"][0]["caption_excerpt"].startswith(
        "The player enters"
    )
    assert cards[0]["evidence_role"] == "locator_only"
    assert cards[0]["answer_support"] is False
    assert transform.audit["candidate_card_counts"] == [2]
    assert transform.audit["retrieval_parity_passed"] is True


def test_no_oracle_gate_rejects_nested_benchmark_annotations() -> None:
    with pytest.raises(ValueError, match="gold_answer"):
        assert_no_oracle_packet(
            {"runtime_metadata": {"gold_answer": "hidden"}},
            surface="runtime_case",
        )


def test_method_arms_cannot_be_combined_with_oracle_interventions() -> None:
    assert (
        validate_occurrence_method_configuration(
            method_arm="a1", oracle_arm="o0", oracle_intervention=None
        )
        == "a1"
    )
    with pytest.raises(ValueError, match="requires O0"):
        validate_occurrence_method_configuration(
            method_arm="a1", oracle_arm="o1", oracle_intervention="oracle.json"
        )
