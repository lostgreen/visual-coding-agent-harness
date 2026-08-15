from __future__ import annotations

from copy import deepcopy

import pytest

from vcah.caption_occurrence import build_caption_occurrence_set
from vcah.caption_schema import stable_digest
from vcah.occurrence_agent import (
    OccurrencePacketTransform,
    OccurrenceResolutionStateV1,
    OccurrenceResolutionStateV2,
    assert_no_oracle_packet,
    candidate_card_excerpt_digest,
    validate_occurrence_method_configuration,
)
from vcah.interactive_agents import _frozen_reasoner_prompt
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    _actionable_locator_errors,
    _occurrence_answer_errors,
    _occurrence_locator_statuses,
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


def test_frozen_occurrence_replay_records_and_reuses_identical_packet(
    tmp_path,
) -> None:
    packet = {**_packet(), "config_digest": "caption-digest"}
    fixture_path = tmp_path / "fixture.json"
    recorder = OccurrencePacketTransform(
        arm="a0",
        audit_path=tmp_path / "record-audit.json",
        case_id="case-1",
        caption_config_digest="caption-digest",
        replay_record_path=fixture_path,
    )

    recorder(packet)

    natural = deepcopy(packet)
    natural["queries"] = ["different stochastic query"]
    natural["query_fingerprint"] = "different-query-digest"
    replayer = OccurrencePacketTransform(
        arm="a3",
        audit_path=tmp_path / "replay-audit.json",
        case_id="case-1",
        caption_config_digest="caption-digest",
        replay_fixture_path=fixture_path,
    )
    transformed = replayer(natural)

    assert transformed["queries"] == packet["queries"]
    assert transformed["hits"] == packet["hits"]
    assert transformed["occurrence_set"]["candidate_cards"]
    assert replayer.audit["occurrence_replay"]["mode"] == "replay"
    assert replayer.audit["occurrence_replay"]["consumption_complete"] is True
    assert (
        replayer.audit["retrieval_identity_digests"]
        == recorder.audit["retrieval_identity_digests"]
    )
    with pytest.raises(ValueError, match="exhausted"):
        replayer(natural)


def test_frozen_occurrence_replay_prime_reuses_fixed_candidate_pool(
    tmp_path,
) -> None:
    packet = {
        **_packet(),
        "config_digest": "caption-digest",
        "top_k": 7,
        "index_mode": "hybrid",
    }
    fixture_path = tmp_path / "fixture.json"
    recorder = OccurrencePacketTransform(
        arm="a0",
        audit_path=tmp_path / "record-audit.json",
        case_id="case-1",
        caption_config_digest="caption-digest",
        replay_record_path=fixture_path,
    )
    recorder(packet)

    with pytest.raises(ValueError, match="requires a replay fixture"):
        OccurrencePacketTransform(
            arm="a3",
            audit_path=tmp_path / "invalid-prime.json",
            replay_prime=True,
        )

    replayer = OccurrencePacketTransform(
        arm="a3",
        audit_path=tmp_path / "prime-audit.json",
        case_id="case-1",
        caption_config_digest="caption-digest",
        replay_fixture_path=fixture_path,
        replay_prime=True,
    )
    spec = replayer.replay_prime_task_spec
    first = replayer(packet)
    repeated = replayer(packet)

    assert spec["queries"] == tuple(packet["queries"])
    assert spec["top_k"] == 7
    assert first["hits"] == repeated["hits"] == packet["hits"]
    replay_audit = replayer.audit["occurrence_replay"]
    assert replay_audit["prime_requested"] is True
    assert replay_audit["prime_consumed"] is True
    assert replay_audit["post_fixture_reuse_count"] == 1
    assert replay_audit["consumed_packet_count"] == 1


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


def test_a2_occurrence_state_preserves_previously_exposed_ids() -> None:
    state = OccurrenceResolutionStateV1()
    state.sync_visible(("occ_1", "occ_2"))
    state.sync_visible(("occ_2", "occ_3"))

    assert state.current_visible_ids == ("occ_1", "occ_2", "occ_3")
    assert state.apply_ops(
        ({"op": "select", "occurrence_id": "occ_1"},)
    )["accepted"] is True
    assert state.selected_occurrence_id == "occ_1"


def test_a2_clean_resolves_singletons_and_arbitrates_multiple_candidates() -> None:
    state = OccurrenceResolutionStateV2()
    assert state.sync_sets(
        (
            {
                "attempt_id": "attempt_single",
                "semantic_target": ["single target"],
                "candidates": [{"occurrence_id": "occ_1"}],
            },
        )
    ) is True
    assert state.activated is True
    assert state.candidate_count == 1
    assert state.resolution_required is True
    assert state.arbitration_required is False
    assert state.selection_required is True

    assert state.sync_sets(
        (
            {
                "attempt_id": "attempt_ambiguous",
                "semantic_target": ["ambiguous target"],
                "candidates": [
                    {"occurrence_id": "occ_2", "time_range": [10, 20]},
                    {"occurrence_id": "occ_3", "time_range": [30, 40]},
                ],
            },
        )
    ) is True
    assert state.activated is True
    assert state.active_set_id == "attempt_ambiguous"
    assert state.candidate_count == 2
    assert state.resolution_required is True
    assert state.arbitration_required is True
    assert state.selection_required is True


def test_a2_clean_keeps_locator_attempt_sets_separate() -> None:
    state = OccurrenceResolutionStateV2()
    state.sync_sets(
        (
            {
                "attempt_id": "attempt_tiger",
                "semantic_target": ["tiger encounter"],
                "candidates": [
                    {"occurrence_id": "tiger_1"},
                    {"occurrence_id": "tiger_2"},
                ],
            },
        )
    )
    state.sync_sets(
        (
            {
                "attempt_id": "attempt_item",
                "semantic_target": ["item acquisition"],
                "candidates": [
                    {"occurrence_id": "item_1"},
                    {"occurrence_id": "item_2"},
                ],
            },
        )
    )

    assert set(state.sets) == {"attempt_tiger", "attempt_item"}
    assert state.sets["attempt_tiger"].resolution == "deferred"
    assert state.sets["attempt_tiger"].viable_occurrence_ids == (
        "tiger_1",
        "tiger_2",
    )
    assert state.sets["attempt_item"].viable_occurrence_ids == (
        "item_1",
        "item_2",
    )
    rejected = state.apply_ops(
        (
            {
                "op": "select",
                "set_id": "attempt_item",
                "occurrence_id": "tiger_1",
            },
        )
    )
    assert rejected["accepted"] is False
    assert rejected["errors"][0]["code"] == "occurrence_id_not_in_set"


def test_scoped_locators_only_expose_the_active_set() -> None:
    state = OccurrenceResolutionStateV2()
    state.sync_sets(
        (
            {
                "attempt_id": "attempt_old",
                "candidates": [
                    {"occurrence_id": "old_1", "time_range": [1, 2]},
                    {"occurrence_id": "old_2", "time_range": [3, 4]},
                ],
            },
        )
    )
    assert state.apply_ops(
        (
            {
                "op": "select",
                "set_id": "attempt_old",
                "occurrence_id": "old_1",
            },
        )
    )["accepted"] is True
    state.sync_sets(
        (
            {
                "attempt_id": "attempt_new",
                "candidates": [
                    {"occurrence_id": "new_1", "time_range": [5, 6]},
                    {"occurrence_id": "new_2", "time_range": [7, 8]},
                ],
            },
        )
    )
    assert state.apply_ops(
        (
            {
                "op": "select",
                "set_id": "attempt_new",
                "occurrence_id": "new_2",
            },
        )
    )["accepted"] is True

    assert state.active_set_id == "attempt_new"
    assert state.retired_set_ids == ("attempt_old",)
    assert state.active_locators() == (
        {
            "set_id": "attempt_new",
            "locator_attempt_id": "attempt_new",
            "occurrence_id": "new_2",
            "time_range": [7, 8],
            "status": "selected_for_active_set",
        },
    )
    assert state.retired_locators()[0]["set_id"] == "attempt_old"
    assert state.retired_locators()[0]["status"] == "retired_history"
    retired_op = state.apply_ops(
        (
            {
                "op": "select",
                "set_id": "attempt_old",
                "occurrence_id": "old_2",
            },
        )
    )
    assert retired_op["accepted"] is False
    assert retired_op["errors"][0]["code"] == "occurrence_set_not_active"
    serialized = state.to_dict()
    assert serialized["retired_set_ids"] == ["attempt_old"]
    assert [value["lifecycle"] for value in serialized["sets"]] == [
        "retired",
        "active",
    ]


def test_a2_clean_supports_abstention_and_multiple_selections() -> None:
    state = OccurrenceResolutionStateV2()
    state.sync_sets(
        (
            {
                "attempt_id": "attempt_order",
                "semantic_target": ["boss order"],
                "candidates": [
                    {"occurrence_id": "boss_1", "time_range": [1, 2]},
                    {"occurrence_id": "boss_2", "time_range": [3, 4]},
                    {"occurrence_id": "boss_3", "time_range": [5, 6]},
                ],
            },
        )
    )
    assert state.apply_ops(
        (
            {
                "op": "select",
                "set_id": "attempt_order",
                "occurrence_id": "boss_1",
            },
            {
                "op": "select",
                "set_id": "attempt_order",
                "occurrence_id": "boss_3",
            },
        )
    )["accepted"] is True
    assert state.selected_occurrence_ids == ("boss_1", "boss_3")
    assert len(state.active_locators()) == 2

    state.sync_sets(
        (
            {
                "attempt_id": "attempt_missing",
                "semantic_target": ["missing event"],
                "candidates": [
                    {"occurrence_id": "wrong_1"},
                    {"occurrence_id": "wrong_2"},
                ],
            },
        )
    )
    assert state.apply_ops(
        ({"op": "defer", "set_id": "attempt_missing"},)
    )["accepted"] is True
    assert state.search_required is True
    errors = _occurrence_answer_errors(
        ReasonerDecision(action="answer", answer="answer"), state
    )
    assert [error["code"] for error in errors] == ["occurrence_search_required"]
    assert state.apply_ops(
        ({"op": "no_match", "set_id": "attempt_missing"},)
    )["accepted"] is True
    assert _occurrence_answer_errors(
        ReasonerDecision(action="answer", answer="answer"), state
    ) == []


def test_a3_requires_selected_locator_binding_before_answer() -> None:
    state = OccurrenceResolutionStateV2()
    state.sync_sets(
        (
            {
                "attempt_id": "attempt_locator",
                "semantic_target": ["target event"],
                "candidates": [
                    {"occurrence_id": "occ_1", "time_range": [10, 20]},
                    {"occurrence_id": "occ_2", "time_range": [30, 40]},
                ],
            },
        )
    )
    state.apply_ops(
        (
            {
                "op": "select",
                "set_id": "attempt_locator",
                "occurrence_id": "occ_2",
            },
        )
    )
    pending = _occurrence_locator_statuses(state, ())
    assert pending[0]["status"] == "selected_for_active_set"
    assert pending[0]["inspection_status"] == "pending_inspection"
    assert [
        error["code"]
        for error in _actionable_locator_errors(
            ReasonerDecision(action="answer", answer="answer"),
            pending,
            investigation_budget_remaining=1,
        )
    ] == ["occurrence_locator_inspection_required"]
    unbound = ReasonerDecision(
        action="investigate",
        tasks=(
            InvestigationTask(
                query_id="inspect",
                goal="inspect target",
                time_range=(30, 40),
            ),
        ),
    )
    assert [
        error["code"]
        for error in _actionable_locator_errors(
            unbound,
            pending,
            investigation_budget_remaining=1,
        )
    ] == ["occurrence_locator_binding_required"]
    bound = ReasonerDecision(
        action="investigate",
        tasks=(
            InvestigationTask(
                query_id="inspect",
                goal="inspect target",
                occurrence_id="occ_2",
                locator_attempt_id="attempt_locator",
            ),
        ),
    )
    assert _actionable_locator_errors(
        bound,
        pending,
        investigation_budget_remaining=1,
    ) == []

    inspected = _occurrence_locator_statuses(
        state,
        (
            {
                "sampling_config": {
                    "candidate_binding": {
                        "locator_attempt_id": "attempt_locator",
                        "occurrence_id": "occ_2",
                    }
                }
            },
        ),
    )
    assert inspected[0]["status"] == "selected_for_active_set"
    assert inspected[0]["inspection_status"] == "inspected"
    assert inspected[0]["inspected"] is True


def test_a2_clean_and_a3_prompt_activate_only_after_state_exposure() -> None:
    base = {"question": "q", "options": {}, "mechanical_status": {}}
    baseline_prompt = _frozen_reasoner_prompt(base)
    assert "Scoped occurrence arbitration" not in baseline_prompt

    scoped_state = {
        "schema_version": "OccurrenceResolutionStateV2",
        "active_set_id": "attempt_locator",
        "candidate_count": 1,
        "resolution_required": True,
        "arbitration_required": False,
        "selection_required": True,
        "search_required": False,
        "active_resolution": "unresolved",
        "selected_occurrence_ids": [],
        "sets": [],
    }
    scoped_prompt = _frozen_reasoner_prompt(
        {
            **base,
            "mechanical_status": {"occurrence_resolution_state": scoped_state},
        }
    )
    assert "Scoped occurrence resolution is enabled" in scoped_prompt
    assert "Scoped occurrence arbitration is enabled" not in scoped_prompt
    assert '"set_id":"attempt_visible_id"' in scoped_prompt
    assert "no_match" in scoped_prompt

    arbitration_prompt = _frozen_reasoner_prompt(
        {
            **base,
            "mechanical_status": {
                "occurrence_resolution_state": {
                    **scoped_state,
                    "candidate_count": 2,
                    "arbitration_required": True,
                }
            },
        }
    )
    assert "Scoped occurrence arbitration is enabled" in arbitration_prompt

    actionable_prompt = _frozen_reasoner_prompt(
        {
            **base,
            "mechanical_status": {
                "occurrence_resolution_state": {
                    **scoped_state,
                    "selection_required": False,
                    "active_resolution": "selected",
                    "selected_occurrence_ids": ["occ_2"],
                },
                "active_occurrence_locators": [
                    {
                        "locator_attempt_id": "attempt_locator",
                        "occurrence_id": "occ_2",
                    }
                ],
            },
        }
    )
    assert "copy both locator_attempt_id and occurrence_id" in actionable_prompt


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


def test_a2_registers_ids_from_every_visible_occurrence_set() -> None:
    status = {
        "caption_occurrence_sets": [
            {
                "candidates": [
                    {"occurrence_id": "occ_first"},
                    {"occurrence_id": "occ_shared"},
                ]
            },
            {
                "candidates": [
                    {"occurrence_id": "occ_shared"},
                    {"occurrence_id": "occ_last"},
                ]
            },
        ]
    }

    assert _visible_occurrence_ids(status) == (
        "occ_first",
        "occ_shared",
        "occ_last",
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


def test_a2_selection_requirement_survives_elimination_to_one_candidate() -> None:
    state = OccurrenceResolutionStateV1()
    state.sync_visible(("occ_1", "occ_2"))
    assert state.selection_required is True
    assert state.apply_ops(
        ({"op": "eliminate", "occurrence_id": "occ_1"},)
    )["accepted"] is True
    assert state.viable_occurrence_ids == ("occ_2",)

    errors = _occurrence_answer_errors(
        ReasonerDecision(action="answer", answer="answer"), state
    )
    assert [error["code"] for error in errors] == [
        "occurrence_selection_required"
    ]


def test_a2_final_attempt_requires_answer_after_prior_selection() -> None:
    state = OccurrenceResolutionStateV1()
    state.sync_visible(("occ_1", "occ_2"))
    assert state.apply_ops(
        ({"op": "select", "occurrence_id": "occ_2"},)
    )["accepted"] is True

    errors = _occurrence_answer_errors(
        ReasonerDecision(action="investigate"),
        state,
        require_answer=True,
    )
    assert [error["code"] for error in errors] == [
        "occurrence_answer_required_after_selection"
    ]
    assert _occurrence_answer_errors(
        ReasonerDecision(action="answer", answer="answer"),
        state,
        require_answer=True,
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
                    "selection_required": True,
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
                    "selection_required": True,
                }
            },
        }
    )
    assert "Investigation is closed; return action=update_workspace" in final_prompt
    assert "Return action=answer only" not in final_prompt

    selected_final_prompt = _frozen_reasoner_prompt(
        {
            "question": "q",
            "options": {"A": "a", "B": "b"},
            "force_finalize": True,
            "final_attempt": 1,
            "mechanical_status": {
                "occurrence_resolution_state": {
                    "schema_version": "OccurrenceResolutionStateV1",
                    "viable_occurrence_ids": ["occ_1", "occ_2"],
                    "selection_required": True,
                    "selected_occurrence_id": "occ_2",
                }
            },
        }
    )
    assert "occurrence selection is already persisted" in selected_final_prompt
    assert "Return action=answer only" in selected_final_prompt
    assert "no tasks and no occurrence_ops" in selected_final_prompt


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
