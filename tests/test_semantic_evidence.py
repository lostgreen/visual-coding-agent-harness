from vcah.semantic_evidence import canonical_fact_snapshot, qualify_absence
from vcah.types import EvidenceRecord


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
        }
        for occurrence, repeats in ((1, 3), (2, 2), (3, 2))
        for offset in [index * 0.1 for index in range(repeats)]
    ]

    snapshot = canonical_fact_snapshot((_event_evidence("ev_events", events),)).to_dict()

    assert snapshot["raw_candidate_counts"]["events"] == 7
    assert snapshot["canonical_fact_counts"]["events"] == 3
    assert len(snapshot["ordered_events"]) == 3


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


def test_same_focal_transition_window_merges_multiple_overtaker_descriptions() -> None:
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
            },
        ],
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["canonical_fact_counts"]["events"] == 1
    event = snapshot["confirmed_events"][0]
    assert event["participant_ids"] == ["camera_holder", "racer in blue helmet", "racer in black suit"]
    assert len(event["merge_history"]) == 2


def test_kml_passes_and_camera_operator_aliases_merge_one_overtake_episode() -> None:
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
            },
        ],
    )

    snapshot = canonical_fact_snapshot((evidence,)).to_dict()

    assert snapshot["canonical_fact_counts"]["events"] == 3
    assert len(snapshot["confirmed_events"][1]["merge_history"]) == 3


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
                    "subject_id": "joe",
                    "setup_state": "Joe intends to leave.",
                    "outcome_state": "Joe stays with his family.",
                    "inference": "The confrontation changed his decision.",
                    "confidence": 0.82,
                    "hypothesis_assessments": [{"option_id": "B", "status": "supported"}],
                },
                {
                    "fact_id": "missing_outcome",
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
