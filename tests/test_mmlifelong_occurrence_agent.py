from __future__ import annotations

from copy import deepcopy

import pytest

from vcah.caption_occurrence import build_caption_occurrence_set
from vcah.caption_schema import stable_digest
from vcah.occurrence_agent import (
    OccurrencePacketTransform,
    OccurrenceResolutionStateV1,
    assert_no_oracle_packet,
    candidate_card_excerpt_digest,
    validate_occurrence_method_configuration,
)
from vcah.interactive_agents import _frozen_reasoner_prompt
from vcah.multiround import (
    ReasonerDecision,
    _occurrence_answer_errors,
    _occurrence_treatment_surface,
    _visible_occurrence_ids,
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
    assert transform.audit["text_budget_parity_passed"] is True
    parity = transform.audit["text_budget_parity_checks"][0]
    assert parity["grouped_text_digest"] == parity["flat_text_digest"]
    assert parity["grouped_text_chars"] == parity["flat_text_chars"]
    assert len(transform.audit["retrieval_identity_digests"]) == 1


def test_no_oracle_gate_rejects_nested_benchmark_annotations() -> None:
    with pytest.raises(ValueError, match="gold_answer"):
        assert_no_oracle_packet(
            {"runtime_metadata": {"gold_answer": "hidden"}},
            surface="runtime_case",
        )


def test_a1_flat_matches_a1_text_budget_without_occurrence_binding(tmp_path) -> None:
    packet = _packet()
    grouped = OccurrencePacketTransform(
        arm="a1", audit_path=tmp_path / "grouped.json"
    )(packet)
    flattened = OccurrencePacketTransform(
        arm="a1-flat", audit_path=tmp_path / "flat.json"
    )(packet)

    cards = grouped["occurrence_set"]["candidate_cards"]
    passages = flattened["occurrence_set"]["flat_candidate_passages"]
    assert [
        passage["caption_excerpt"]
        for card in cards
        for passage in card["representative_passages"]
    ] == [passage["caption_excerpt"] for passage in passages]
    assert all("occurrence_id" not in passage for passage in passages)
    assert all("source_video_ids" not in passage for passage in passages)
    assert all("candidate_rank" not in passage for passage in passages)
    assert all(passage["time_range"] for passage in passages)
    assert all(passage["passage_id"] for passage in passages)
    assert candidate_card_excerpt_digest(cards)


def test_a2_occurrence_state_rejects_hidden_ids_transactionally() -> None:
    state = OccurrenceResolutionStateV1()
    state.sync_visible(("occ_1", "occ_2"))

    rejected = state.apply_ops(
        (
            {"op": "eliminate", "occurrence_id": "occ_1"},
            {"op": "select", "occurrence_id": "invented"},
        )
    )
    assert rejected["accepted"] is False
    assert state.states["occ_1"] == "active"
    assert state.selected_occurrence_id == ""

    assert state.apply_ops(
        ({"op": "eliminate", "occurrence_id": "occ_1"},)
    )["accepted"] is True
    assert state.states["occ_1"] == "eliminated"
    assert state.apply_ops(
        ({"op": "select", "occurrence_id": "occ_2"},)
    )["accepted"] is True
    assert state.selected_occurrence_id == "occ_2"
    assert state.apply_ops(
        ({"op": "eliminate", "occurrence_id": "occ_2"},)
    )["accepted"] is True
    assert state.apply_ops(
        ({"op": "keep", "occurrence_id": "occ_2"},)
    )["accepted"] is False
    assert state.apply_ops(
        ({"op": "reopen", "occurrence_id": "occ_2"},)
    )["accepted"] is True


def test_grouped_and_flat_treatment_surfaces_have_text_parity(tmp_path) -> None:
    packet = _packet()
    grouped_packet = OccurrencePacketTransform(
        arm="a1", audit_path=tmp_path / "grouped.json"
    )(packet)
    flat_packet = OccurrencePacketTransform(
        arm="a1-flat", audit_path=tmp_path / "flat.json"
    )(packet)
    grouped_status = {
        "caption_occurrence_sets": [
            {
                "candidates": [
                    {
                        "occurrence_id": card["occurrence_id"],
                        "candidate_card": card,
                    }
                    for card in grouped_packet["occurrence_set"][
                        "candidate_cards"
                    ]
                ]
            }
        ]
    }
    flat_status = {
        "caption_occurrence_sets": [
            {
                "candidates": [
                    {"occurrence_id": candidate["occurrence_id"]}
                    for candidate in flat_packet["occurrence_set"][
                        "candidates"
                    ]
                ]
            }
        ],
        "flat_occurrence_passages": flat_packet["occurrence_set"][
            "flat_candidate_passages"
        ],
        "flat_occurrence_queries": flat_packet["occurrence_set"][
            "flat_candidate_queries"
        ],
    }

    grouped = _occurrence_treatment_surface(grouped_status)
    flattened = _occurrence_treatment_surface(flat_status)
    assert grouped and flattened
    assert grouped["visible_excerpt_digest"] == flattened[
        "visible_excerpt_digest"
    ]
    assert grouped["visible_excerpt_chars"] == flattened[
        "visible_excerpt_chars"
    ]
    assert grouped["visible_text_digest"] == flattened["visible_text_digest"]
    assert _visible_occurrence_ids(flat_status) == tuple(
        candidate["occurrence_id"]
        for candidate in flat_packet["occurrence_set"]["candidates"]
    )


def test_a2_prompt_is_only_enabled_by_resolution_state() -> None:
    base = {"question": "q", "options": {}, "mechanical_status": {}}
    assert "Explicit occurrence arbitration" not in _frozen_reasoner_prompt(base)
    enabled = {
        **base,
        "mechanical_status": {
            "occurrence_resolution_state": {
                "schema_version": "OccurrenceResolutionStateV1"
            }
        },
    }
    assert "Explicit occurrence arbitration" in _frozen_reasoner_prompt(enabled)


def test_a2_requires_reasoner_selection_before_answer() -> None:
    state = OccurrenceResolutionStateV1()
    state.sync_visible(("occ_1", "occ_2"))

    missing = _occurrence_answer_errors(
        ReasonerDecision(action="answer", answer="answer"), state
    )
    assert [error["code"] for error in missing] == [
        "occurrence_selection_required"
    ]

    same_decision = ReasonerDecision(
        action="answer",
        answer="answer",
        occurrence_ops=({"op": "select", "occurrence_id": "occ_2"},),
    )
    assert [
        error["code"] for error in _occurrence_answer_errors(same_decision, state)
    ] == ["occurrence_selection_required"]
    assert state.apply_ops(same_decision.occurrence_ops)["accepted"] is True
    assert _occurrence_answer_errors(
        ReasonerDecision(action="answer", answer="answer"), state
    ) == []


def test_a2_prompt_requires_selection_in_a_prior_decision() -> None:
    prompt = _frozen_reasoner_prompt(
        {
            "question": "q",
            "options": {"A": "a", "B": "b"},
            "mechanical_status": {
                "occurrence_resolution_state": {
                    "schema_version": "OccurrenceResolutionStateV1",
                    "viable_occurrence_ids": ["occ_1", "occ_2"],
                }
            },
        }
    )

    assert "selection is a separate committed step" in prompt
    assert 'Selection commit schema: {"action":"update_workspace"' in prompt
    assert "Do not answer while multiple viable occurrences remain unselected" in prompt

    final_prompt = _frozen_reasoner_prompt(
        {
            "question": "q",
            "options": {"A": "a", "B": "b"},
            "force_finalize": True,
            "final_attempt": 2,
            "mechanical_status": {
                "occurrence_resolution_state": {
                    "schema_version": "OccurrenceResolutionStateV1",
                    "viable_occurrence_ids": ["occ_1", "occ_2"],
                }
            },
        }
    )
    assert "Investigation is closed; return action=update_workspace" in final_prompt
    assert "Return action=answer only" not in final_prompt


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
