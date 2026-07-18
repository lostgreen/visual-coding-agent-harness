from vcah.semantic_evidence import canonical_fact_snapshot, event_candidate_ledger, qualify_absence
from vcah.types import EvidenceRecord


def _position_qualification() -> dict[str, str]:
    return {
        "required_prior_state": "supported",
        "transition": "supported",
        "same_subject": "supported",
        "episode_boundary": "supported",
    }


def _event_evidence(evidence_id: str, events: list[dict]) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        beat_id="",
        start_sec=0.0,
        end_sec=30.0,
        modality="visual",
        pointer=f"virtual://{evidence_id}",
        verbatim="Structured event observations.",
        frame_refs=("frame.jpg",),
        operation_metadata={"events": events},
    )


def test_canonical_snapshot_keeps_raw_count_separate_from_confirmed_count() -> None:
    events = [
        {
            "event_key": f"overtake_{occurrence}",
            "event_class": "overtake",
            "counting_unit": "overtake",
            "participant_ids": [f"racer_{occurrence}", "camera_holder"],
            "start_sec": occurrence * 10.0 + offset,
            "end_sec": occurrence * 10.0 + 2.0 + offset,
            "supports_question_event": True,
            "qualification": _position_qualification(),
        }
        for occurrence, repeats in ((1, 3), (2, 2), (3, 2))
        for offset in [index * 0.1 for index in range(repeats)]
    ]

    snapshot = canonical_fact_snapshot((_event_evidence("ev_events", events),)).to_dict()

    assert snapshot["raw_candidate_counts"]["events"] == 7
    assert snapshot["canonical_fact_counts"]["events"] == 3
    assert len(snapshot["ordered_events"]) == 3


def test_supports_question_event_creates_candidate_not_ledger_qualified_event() -> None:
    evidence = _event_evidence(
        "ev_candidate_only",
        [{
            "event_key": "title card appears",
            "description": "The opening title card appears.",
            "start_sec": 1.0,
            "end_sec": 2.0,
            "supports_question_event": True,
        }],
    )

    ledger = event_candidate_ledger((evidence,))

    assert ledger["observed_event_candidate_count"] == 1
    assert ledger["confirmed_event_candidate_count"] == 0
    assert ledger["confirmed_event_candidates"] == []


def test_absence_requires_coverage_visibility_and_known_dwell_time() -> None:
    unknown_dwell = qualify_absence((0.0, 10.0), ((0.0, 10.0),), 0.5, None, "clear")
    sparse = qualify_absence((0.0, 10.0), ((0.0, 10.0),), 1.1, 2.0, "clear")
    occluded = qualify_absence((0.0, 10.0), ((0.0, 10.0),), 0.5, 2.0, "occluded")
    qualified = qualify_absence((0.0, 10.0), ((0.0, 10.0),), 0.5, 2.0, "clear")

    assert unknown_dwell.status == "unknown_due_to_coverage"
    assert sparse.status == "unknown_due_to_coverage"
    assert occluded.status == "unknown_due_to_visibility"
    assert qualified.status == "qualified_absence"


def test_countable_observation_without_entity_hypothesis_remains_unresolved() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_entity",
        beat_id="",
        start_sec=1.0,
        end_sec=2.0,
        modality="visual",
        pointer="virtual://ev_entity",
        verbatim="A racer is directly visible.",
        frame_refs=("frame.jpg",),
        operation_metadata={
            "entities": [{"entity_observation_id": "obs:racer", "countable": True}],
        },
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["resolved_entities"] == []
    assert snapshot["unresolved_entity_bindings"][0]["entity_observation_id"] == "obs:racer"


def test_same_focal_transition_with_conflicting_actor_attributes_is_one_candidate() -> None:
    evidence = _event_evidence(
        "ev_overtakes",
        [
            {
                "local_id": "blue_pass",
                "event_key": "blue helmet passes recorder",
                "event_class": "other",
                "counting_unit": "question-defined unit",
                "participant_ids": ["video recorder", "racer in blue helmet"],
                "transition": "recorder loses first place",
                "start_sec": 464.0,
                "end_sec": 465.5,
                "supports_question_event": True,
                "qualification": _position_qualification(),
            },
            {
                "local_id": "black_pass",
                "event_key": "black suit passes recorder",
                "event_class": "other",
                "counting_unit": "overtake_of_recorder",
                "participant_ids": ["recorder", "racer in black suit"],
                "transition": "recorder is overtaken and drops rank",
                "start_sec": 464.0,
                "end_sec": 465.0,
                "supports_question_event": True,
                "qualification": _position_qualification(),
            },
        ],
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["canonical_fact_counts"]["events"] == 1
    assert snapshot["conflicted_events"] == []
    row = snapshot["qualified_events"][0]
    assert row["participant_binding_status"] == "ambiguous"
    assert row["participant_binding_conflicts"][0]["reason"] == (
        "same_occurrence_has_ambiguous_actor_attributes"
    )
    assert len(row["merge_history"]) == 2


def test_focal_position_transition_without_qualification_is_only_observed() -> None:
    evidence = _event_evidence(
        "ev_unqualified_overtake",
        [{
            "event_key": "black racer passes recorder",
            "participant_ids": ["recorder", "black racer"],
            "transition": "The black racer overtakes the recorder, who loses first place.",
            "start_sec": 100.0,
            "end_sec": 101.0,
            "supports_question_event": True,
        }],
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["canonical_fact_counts"]["events"] == 0
    assert len(snapshot["observed_event_candidates"]) == 1
    assert snapshot["observed_event_candidates"][0]["candidate_status"] == "observed_candidate"


def test_model_qualified_claim_cannot_bypass_state_before_requirement() -> None:
    evidence = _event_evidence(
        "ev_model_claim",
        [{
            "event_key": "black racer passes recorder",
            "participant_ids": ["recorder", "black racer"],
            "transition": "The recorder loses first place.",
            "start_sec": 100.0,
            "end_sec": 101.0,
            "supports_question_event": True,
            "qualification_status": "qualified",
        }],
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["qualified_events"] == []
    assert snapshot["incomplete_events"][0]["qualification_status"] == "incomplete"
    statuses = {
        row["requirement_id"]: row["status"]
        for row in snapshot["incomplete_events"][0]["requirement_evaluations"]
    }
    assert statuses["req_event_candidate_001_prior_state"] == "unknown"


def test_generic_state_tracking_event_requires_explicit_precondition() -> None:
    evidence = _event_evidence(
        "ev_state_tracking",
        [{
            "event_key": "ingredient changes color",
            "participant_ids": ["octopus_batch_01"],
            "transition": "The surface turns tan.",
            "start_sec": 10.0,
            "end_sec": 12.0,
            "supports_question_event": True,
        }],
    )

    snapshot = canonical_fact_snapshot(
        (evidence,),
        require_event_precondition=True,
    ).to_dict()

    assert snapshot["qualified_events"] == []
    assert snapshot["incomplete_events"][0]["qualification_status"] == "incomplete"


def test_position_event_requirements_are_derived_from_canonical_observables() -> None:
    evidence = _event_evidence(
        "ev_observable_qualification",
        [{
            "event_key": "black racer passes recorder",
            "participant_ids": ["recorder", "black racer"],
            "state_before": "The recorder is in first place.",
            "transition": "The black racer overtakes the recorder.",
            "state_after": "The recorder drops to second place.",
            "preconditions_met": True,
            "start_sec": 100.0,
            "end_sec": 101.0,
            "supports_question_event": True,
        }],
    )

    snapshot = canonical_fact_snapshot(
        (evidence,),
        require_event_precondition=True,
    ).to_dict()

    assert len(snapshot["qualified_events"]) == 1
    assert all(
        row["status"] == "supported"
        for row in snapshot["qualified_events"][0]["requirement_evaluations"]
    )


def test_kml_ambiguous_passers_merge_while_independent_qualified_events_count() -> None:
    evidence = _event_evidence(
        "ev_kml_overtakes",
        [
            {
                "event_key": "green racer passes recorder",
                "event_class": "other",
                "counting_unit": "question-defined unit",
                "participant_ids": ["video recorder", "racer green jacket"],
                "transition": "A racer in a green jacket approaches and passes the recorder on the right.",
                "start_sec": 462.5,
                "end_sec": 465.0,
                "supports_question_event": True,
                "qualification": _position_qualification(),
            },
            {
                "event_key": "blue helmet passes recorder",
                "event_class": "other",
                "counting_unit": "question-defined unit",
                "participant_ids": ["recorder", "racer in blue helmet"],
                "transition": "Racer in blue helmet passes the video recorder on the left.",
                "start_sec": 464.0,
                "end_sec": 465.5,
                "supports_question_event": True,
                "qualification": _position_qualification(),
            },
            {
                "event_key": "person passes camera operator",
                "participant_ids": ["person 1", "camera operator"],
                "transition": "Person 1 passes the camera operator on the right side.",
                "start_sec": 464.5,
                "end_sec": 465.5,
                "supports_question_event": True,
            },
            {
                "event_key": "yellow racer pass",
                "event_class": "other",
                "counting_unit": "question-defined unit",
                "participant_ids": ["recorder", "yellow racer"],
                "transition": "The yellow racer passes the recorder.",
                "start_sec": 163.0,
                "end_sec": 164.0,
                "supports_question_event": True,
                "qualification": _position_qualification(),
            },
            {
                "event_key": "lime racer pass",
                "event_class": "other",
                "counting_unit": "question-defined unit",
                "participant_ids": ["recorder", "lime racer"],
                "transition": "The lime racer overtakes the recorder.",
                "start_sec": 725.0,
                "end_sec": 726.0,
                "supports_question_event": True,
                "qualification": _position_qualification(),
            },
        ],
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["canonical_fact_counts"]["events"] == 3
    assert snapshot["conflicted_events"] == []
    ambiguous = [
        row for row in snapshot["qualified_events"]
        if row["participant_binding_status"] == "ambiguous"
    ]
    assert len(ambiguous) == 1
    assert len(snapshot["observed_event_candidates"]) == 3


def test_simultaneous_events_without_shared_focal_subject_stay_distinct() -> None:
    evidence = _event_evidence(
        "ev_parallel",
        [
            {
                "event_key": "red racer passes blue racer",
                "event_class": "other",
                "counting_unit": "overtake",
                "participant_ids": ["red racer", "blue racer"],
                "start_sec": 10.0,
                "end_sec": 11.0,
                "supports_question_event": True,
            },
            {
                "event_key": "green racer passes yellow racer",
                "event_class": "other",
                "counting_unit": "overtake",
                "participant_ids": ["green racer", "yellow racer"],
                "start_sec": 10.0,
                "end_sec": 11.0,
                "supports_question_event": True,
            },
        ],
    )

    assert canonical_fact_snapshot((evidence,)).to_dict()["canonical_fact_counts"]["events"] == 2


def test_event_participant_hypothesis_enters_canonical_entity_view() -> None:
    evidence = _event_evidence(
        "ev_participant",
        [{
            "event_key": "green racer overtakes recorder",
            "event_class": "other",
            "counting_unit": "overtake_episode",
            "participant_ids": ["recorder", "green racer"],
            "participants": [{
                "participant_id": "green racer",
                "entity_hypothesis_id": "racer_green_01",
                "role": "overtaker",
                "visual_signature": "green jacket; black helmet",
                "attributes": {"jacket_color": "green", "helmet_color": "black"},
                "association_confidence": 0.84,
            }],
            "start_sec": 10.0,
            "end_sec": 11.0,
            "supports_question_event": True,
        }],
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["resolved_entities"][0]["entity_id"] == "racer_green_01"
    assert snapshot["resolved_entities"][0]["role"] == "overtaker"


def test_supported_cross_window_association_resolves_event_participant_attributes() -> None:
    event = _event_evidence(
        "ev_anchor",
        [{
            "event_key": "second racer overtakes recorder",
            "event_class": "overtake",
            "counting_unit": "overtake_episode",
            "participant_ids": ["recorder", "second racer"],
            "participants": [{"participant_id": "second racer", "role": "overtaker"}],
            "start_sec": 10.0,
            "end_sec": 11.0,
            "supports_question_event": True,
        }],
    )
    linked = EvidenceRecord(
        evidence_id="ev_link",
        beat_id="",
        start_sec=90.0,
        end_sec=95.0,
        modality="visual",
        pointer="virtual://ev_link",
        verbatim="The same racer is visible later in green clothing.",
        frame_refs=("later.jpg",),
        operation_metadata={
            "entities": [{
                "local_id": "target_1",
                "entity_observation_id": "obs_later:target_1",
                "visual_signature": "green jacket; black helmet",
                "attributes": {"clothing_color": "green", "helmet_color": "black"},
            }],
            "entity_associations": [{
                "association_id": "assoc_second",
                "source_participant_id": "second racer",
                "source_event_key": "second racer overtakes recorder",
                "target_entity_observation_id": "obs_later:target_1",
                "entity_hypothesis_id": "second_racer_anchor",
                "status": "supported",
                "confidence": 0.86,
                "shared_attributes": {"jacket": "green", "helmet": "black"},
                "distinguishing_attributes": {"clothing_color": "green"},
            }],
        },
    )

    snapshot = canonical_fact_snapshot((event, linked)).to_dict()

    assert snapshot["canonical_fact_counts"]["entity_associations"] == 1
    assert snapshot["resolved_entities"][0]["entity_id"] == "second_racer_anchor"
    assert snapshot["resolved_entities"][0]["attributes"]["clothing_color"] == "green"
    assert all(row["entity_id"] != "second racer" for row in snapshot["unresolved_entity_bindings"])


def test_cross_window_association_merges_later_attributes_into_existing_hypothesis() -> None:
    event = _event_evidence(
        "ev_anchor_known",
        [{
            "event_key": "racer overtakes recorder",
            "event_class": "overtake",
            "counting_unit": "overtake_episode",
            "participant_ids": ["recorder", "green racer"],
            "participants": [{
                "participant_id": "green racer",
                "entity_hypothesis_id": "racer_green_01",
                "association_confidence": 0.8,
                "role": "overtaker",
                "attributes": {"clothing_color": "green"},
            }],
            "start_sec": 10.0,
            "end_sec": 11.0,
            "supports_question_event": True,
        }],
    )
    linked = EvidenceRecord(
        evidence_id="ev_finish",
        beat_id="",
        start_sec=90.0,
        end_sec=95.0,
        modality="visual",
        pointer="virtual://finish",
        verbatim="The linked racer finishes third.",
        frame_refs=("finish.jpg",),
        operation_metadata={
            "entities": [{
                "local_id": "target_1",
                "entity_observation_id": "obs_finish:target_1",
                "attributes": {"finish_position": "third"},
            }],
            "entity_associations": [{
                "source_participant_id": "green racer",
                "source_event_key": "racer overtakes recorder",
                "target_entity_observation_id": "obs_finish:target_1",
                "entity_hypothesis_id": "racer_green_01",
                "status": "supported",
                "confidence": 0.9,
                "shared_attributes": {"jacket": "green", "helmet": "black"},
                "distinguishing_attributes": {"finish_position": "third"},
            }],
        },
    )

    entity = canonical_fact_snapshot((event, linked)).to_dict()["resolved_entities"][0]

    assert entity["attributes"] == {"clothing_color": "green", "finish_position": "third"}
    assert entity["association_confidence"] == 0.9


def test_narrative_fact_requires_both_setup_and_outcome() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_narrative",
        beat_id="",
        start_sec=1.0,
        end_sec=8.0,
        modality="visual",
        pointer="virtual://ev_narrative",
        verbatim="Joe changes his decision after the confrontation.",
        frame_refs=("story.jpg",),
        operation_metadata={
            "narrative_facts": [
                {
                    "fact_id": "complete_bridge",
                    "episode_id": "door_confrontation_01",
                    "timeline_phase": "final",
                    "anchor_match": True,
                    "subject_id": "joe",
                    "relation_type": "final_decision",
                    "predicate": "stay_with_family",
                    "setup_state": "Joe intends to leave.",
                    "outcome_state": "Joe stays with his family.",
                    "inference": "The confrontation changed his decision.",
                    "confidence": 0.82,
                    "hypothesis_assessments": [{"option_id": "B", "status": "supported"}],
                },
                {
                    "fact_id": "missing_outcome",
                    "episode_id": "door_confrontation_01",
                    "relation_type": "inferred_intention",
                    "subject_id": "joe",
                    "setup_state": "Joe intends to leave.",
                    "outcome_state": "",
                    "inference": "He may reconsider.",
                    "confidence": 0.9,
                },
            ],
        },
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert [row["fact_id"] for row in snapshot["inferred_facts"]] == ["complete_bridge"]
    assert snapshot["unresolved_inferences"] == []


def test_incomplete_narrative_fact_remains_unresolved_without_complete_bridge() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_partial_narrative",
        beat_id="",
        start_sec=1.0,
        end_sec=3.0,
        modality="visual",
        pointer="virtual://partial_narrative",
        verbatim="Only the setup is visible.",
        frame_refs=("setup.jpg",),
        operation_metadata={
            "narrative_facts": [{
                "fact_id": "partial_bridge",
                "setup_state": "Joe intends to leave.",
                "outcome_state": "",
                "inference": "He may reconsider.",
                "confidence": 0.9,
            }],
        },
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert [row["fact_id"] for row in snapshot["unresolved_inferences"]] == ["partial_bridge"]


def test_siren_cooccurrence_does_not_support_agent_causation() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_siren",
        beat_id="",
        start_sec=1.0,
        end_sec=8.0,
        modality="visual",
        pointer="virtual://ev_siren",
        verbatim="A siren is heard while Joe is visible.",
        frame_refs=("siren.jpg",),
        operation_metadata={
            "narrative_facts": [{
                "fact_id": "siren_cooccurrence",
                "episode_id": "street_01",
                "timeline_phase": "outcome",
                "anchor_match": True,
                "subject_id": "joe",
                "relation_type": "temporal_cooccurrence",
                "predicate": "joe_called_police",
                "setup_state": "Joe is outside.",
                "outcome_state": "A siren is audible.",
                "inference": "The events occur close in time.",
                "confidence": 0.8,
                "hypothesis_assessments": [{"option_id": "C", "status": "supported"}],
            }],
        },
    )

    fact = canonical_fact_snapshot((evidence,)).to_dict()["inferred_facts"][0]

    assert fact["relation_type"] == "temporal_cooccurrence"
    assert fact["hypothesis_assessments"][0]["status"] == "unknown"


def test_narrative_facts_remain_namespaced_by_episode() -> None:
    def narrative(evidence_id: str, episode_id: str, predicate: str) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            beat_id="",
            start_sec=1.0,
            end_sec=8.0,
            modality="visual",
            pointer=f"virtual://{evidence_id}",
            verbatim=predicate,
            frame_refs=(f"{evidence_id}.jpg",),
            operation_metadata={
                "narrative_facts": [{
                    "fact_id": evidence_id,
                    "episode_id": episode_id,
                    "timeline_phase": "final",
                    "anchor_match": episode_id == "door_01",
                    "subject_id": "joe",
                    "relation_type": "observed_action",
                    "predicate": predicate,
                    "setup_state": "An episode begins.",
                    "outcome_state": "An action is visible.",
                    "inference": predicate,
                    "confidence": 0.8,
                }],
            },
        )

    facts = canonical_fact_snapshot((
        narrative("door_fact", "door_01", "conceal_event"),
        narrative("cake_fact", "cake_01", "eat_cake"),
    )).to_dict()["inferred_facts"]

    assert {(row["episode_id"], row["predicate"]) for row in facts} == {
        ("door_01", "conceal_event"),
        ("cake_01", "eat_cake"),
    }
